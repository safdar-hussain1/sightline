/**
 * Reading frames out of the visitor's video, in this browser.
 *
 * The file is opened from an object URL (`blob:`), which names bytes already
 * in this browser: nothing is sent anywhere. Frames are taken by seeking --
 * `currentTime` set, `seeked` awaited, with a time limit on every seek -- so
 * each frame is the one at its planned time, whatever the video's own frame
 * rate. Two video elements seek side by side, taking alternate frames: a
 * browser decodes each from the keyframe before it, and two decoders at work
 * read a video several times faster than one, while more than two gained
 * nothing where it was measured. One of the two is the element on screen, so
 * the visitor watches the video being read.
 *
 * Each frame is cropped to the part the model reads and scaled to its input
 * with `drawImage` at `imageSmoothingQuality = "high"`. Training shrank its
 * crops with OpenCV's INTER_AREA instead; the two are close, not identical,
 * so the page never claims that its input from a visitor's video matches
 * what training would have made of it.
 */

import type { Bytes } from "../cockpit/sprites.ts";
import { ClipError, type Evidence, type Rect } from "./ownclip-plan.ts";

/** How many video elements seek at once. */
const SEEKERS = 2;

/** The longest a seek, or the opening of the file, may take while the page is
 * on screen, ms. A seek that takes longer is a video this browser cannot step
 * through. Time with the tab in the background does not count: browsers hold
 * back decoding there. */
const SEEK_LIMIT_MS = 8000;
const OPEN_LIMIT_MS = 15000;

/** How often the time limits are checked, ms. */
const LIMIT_TICK_MS = 250;

/** A video opened for reading. */
export interface VideoSource {
  readonly width: number;
  readonly height: number;
  readonly duration: number;
  /** Read the frames at `times` (seconds), each cropped to `crop` and scaled
   * to `size`, as RGB bytes; `onFrame` hears each as it arrives -- in about,
   * not exactly, their order -- with the picture it came from. Rejects with a
   * `ClipError` when the browser cannot read a frame, or when `signal` aborts. */
  read(
    times: readonly number[],
    crop: Rect,
    size: readonly [number, number],
    onFrame: (j: number, rgb: Bytes, picture: HTMLVideoElement) => void,
    signal: AbortSignal,
  ): Promise<void>;
  /** Let go of every element but the one on screen. */
  close(): void;
}

export type OpenVideo = (url: string, evidence: Evidence, signal: AbortSignal, display: HTMLVideoElement) => Promise<VideoSource>;

function aborted(): DOMException {
  return new DOMException("The reading was stopped", "AbortError");
}

/** Wait for one of `video`'s events, failing on its `error` event, on
 * `signal`, or once `limitMs` of on-screen time has passed (`onLimit`). */
function waitFor(
  video: HTMLVideoElement,
  event: string,
  signal: AbortSignal,
  limitMs: number,
  onError: () => Error,
  onLimit: () => Error,
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) {
      reject(aborted());
      return;
    }
    let waited = 0;
    let last = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now();
      if (document.visibilityState !== "hidden") waited += now - last;
      last = now;
      if (waited > limitMs) finish(onLimit());
    }, LIMIT_TICK_MS);
    const finish = (error?: Error) => {
      window.clearInterval(timer);
      video.removeEventListener(event, done);
      video.removeEventListener("error", failed);
      signal.removeEventListener("abort", stop);
      if (error) reject(error);
      else resolve();
    };
    const done = () => finish();
    const failed = () => finish(onError());
    const stop = () => finish(aborted());
    video.addEventListener(event, done);
    video.addEventListener("error", failed);
    signal.addEventListener("abort", stop);
  });
}

/** Open `url` in `video` and wait until its first frame can be drawn. */
async function open(video: HTMLVideoElement, url: string, evidence: Evidence, signal: AbortSignal): Promise<Evidence> {
  const refused = () => new ClipError(`the browser refused the file (${video.error?.message ?? "no detail"})`, { ...evidence, mediaError: video.error?.code ?? 0 });
  const slow = () => new ClipError("the file did not open in time", { ...evidence, seekTimedOut: true });
  video.muted = true;
  video.preload = "auto";
  const metadata = waitFor(video, "loadedmetadata", signal, OPEN_LIMIT_MS, refused, slow);
  video.src = url;
  video.load();
  await metadata;
  const seen: Evidence = { ...evidence, width: video.videoWidth, height: video.videoHeight, duration: video.duration };
  if (seen.width === 0 || seen.height === 0) throw new ClipError("the file has no picture", seen);
  if (!Number.isFinite(video.duration)) throw new ClipError("the file reports no length", seen);
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    await waitFor(video, "loadeddata", signal, OPEN_LIMIT_MS, refused, slow);
  }
  return seen;
}

/** Seek `video` to `t` and wait for the frame there. */
function seek(video: HTMLVideoElement, t: number, evidence: Evidence, signal: AbortSignal): Promise<void> {
  const failed = () => new ClipError(`a frame could not be decoded (${video.error?.message ?? "no detail"})`, { ...evidence, mediaError: video.error?.code ?? 0 });
  const slow = () => new ClipError(`the seek to ${t.toFixed(3)} s never finished`, { ...evidence, seekTimedOut: true });
  const seeked = waitFor(video, "seeked", signal, SEEK_LIMIT_MS, failed, slow);
  video.currentTime = t;
  return seeked;
}

/** Let `video` go of its file, so its decoder is freed. */
export function emptyVideo(video: HTMLVideoElement): void {
  video.pause();
  video.removeAttribute("src");
  video.load();
}

/** Open the file at `url` for reading, with `display` (the video on screen)
 * as one of the elements that seek through it. */
export const openVideo: OpenVideo = async (url, evidence, signal, display) => {
  const others = Array.from({ length: SEEKERS - 1 }, () => document.createElement("video"));
  const seekers = [display, ...others];
  try {
    const [seen] = await Promise.all(seekers.map((video) => open(video, url, evidence, signal)));
    const known = seen as Evidence;
    return {
      width: known.width as number,
      height: known.height as number,
      duration: known.duration as number,
      read: (times, crop, [w, h], onFrame, outer) => readFrames(seekers, known, times, crop, w, h, onFrame, outer),
      close: () => others.forEach(emptyVideo),
    };
  } catch (error) {
    others.forEach(emptyVideo);
    throw error;
  }
};

async function readFrames(
  seekers: readonly HTMLVideoElement[],
  evidence: Evidence,
  times: readonly number[],
  crop: Rect,
  w: number,
  h: number,
  onFrame: (j: number, rgb: Bytes, picture: HTMLVideoElement) => void,
  outer: AbortSignal,
): Promise<void> {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("no 2D canvas to read the frames into");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  // A failure in one seeker stops the other, and the failure is what the
  // caller hears, not the other's being stopped.
  const local = new AbortController();
  const stop = () => local.abort();
  outer.addEventListener("abort", stop);
  let failure: unknown = null;
  try {
    await Promise.allSettled(
      seekers.map(async (video, i) => {
        try {
          for (let j = i; j < times.length; j += seekers.length) {
            await seek(video, times[j] as number, evidence, local.signal);
            context.drawImage(video, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, w, h);
            const rgba = context.getImageData(0, 0, w, h).data;
            const rgb = new Uint8Array(w * h * 3);
            for (let p = 0, from = 0, to = 0; p < w * h; p += 1, from += 4, to += 3) {
              rgb[to] = rgba[from] as number;
              rgb[to + 1] = rgba[from + 1] as number;
              rgb[to + 2] = rgba[from + 2] as number;
            }
            onFrame(j, rgb, video);
          }
        } catch (error) {
          if (failure === null && !(error instanceof DOMException && error.name === "AbortError")) failure = error;
          local.abort();
        }
      }),
    );
  } finally {
    outer.removeEventListener("abort", stop);
  }
  if (outer.aborted) throw aborted();
  if (failure !== null) throw failure;
  if (local.signal.aborted) throw aborted();
}
