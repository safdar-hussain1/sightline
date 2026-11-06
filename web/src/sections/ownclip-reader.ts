/**
 * Reading frames out of the visitor's video, in this browser.
 *
 * The file is opened from an object URL (`blob:`), which names bytes already
 * in this browser: nothing is sent anywhere. Frames are taken by seeking --
 * `currentTime` set, `seeked` awaited, with a time limit on every seek -- so
 * each frame is the one at its planned time, whatever the video's own frame
 * rate.
 *
 * Two video elements can seek side by side: a browser decodes each frame
 * from the keyframe before it, and two decoders at work read a video several
 * times faster than one (more than two gained nothing where it was
 * measured). The element on screen opens the file first, alone, and the
 * second opens only once the first has its first frame: opened at once, two
 * elements can make a browser refuse a file it plays well one at a time (a
 * VP9 WebM in Chrome, as phones and browser recorders make them). The second
 * is never needed: if it cannot open the file, or fails part way, the
 * reading carries on with the first, and a file is called unreadable only
 * when the element on screen, alone, could not read it.
 *
 * A file cut short -- still copying, a download that broke off -- can keep
 * answering every seek with its last decodable frame, as if it were whole.
 * So each frame is checked against the one the same element drew before:
 * pixels that repeat while the frame on screen falls well behind the time
 * asked for (`requestVideoFrameCallback` says which frame is on screen, and
 * stops when nothing new comes) mean the file gives no new frames from
 * there. Reading stops at the first frame after the last real one, and none
 * of the repeated frames is passed on. A scene that does not change is not
 * mistaken for this: its frames are new, only their pixels repeat.
 *
 * Each frame is cropped to the part the model reads and scaled to its input
 * with `drawImage` at `imageSmoothingQuality = "high"`. Training shrank its
 * crops with OpenCV's INTER_AREA instead; the two are close, not identical,
 * so the page never claims that its input from a visitor's video matches
 * what training would have made of it.
 */

import type { Bytes } from "../cockpit/sprites.ts";
import { ClipError, type Evidence, type Rect } from "./ownclip-plan.ts";

/** The longest a seek, or the opening of the file, may take while the page is
 * on screen, ms. A seek that takes longer is a video this browser cannot step
 * through. Time with the tab in the background does not count: browsers hold
 * back decoding there. */
const SEEK_LIMIT_MS = 8000;
const OPEN_LIMIT_MS = 15000;
/** How long the browser may take to work out the length of a video that does
 * not state one, ms. */
const LENGTH_LIMIT_MS = 5000;
/** How long after that seek lands its length may still arrive, ms. */
const LENGTH_GRACE_MS = 300;

/** How often the time limits are checked, ms. */
const LIMIT_TICK_MS = 250;

/** How far the frame on screen may fall behind the time asked for, seconds,
 * while its pixels repeat, before the file counts as giving no new frames.
 * Longer than one frame of any video worth reading (a frame shown for a
 * second and a half would be a slide show), so a slow frame rate is never
 * mistaken for a file cut short. */
const STALE_S = 1.5;

/** How a reading ended: `end` is the number of frames read when the file
 * stopped giving new frames before the last one asked for; null when every
 * frame was read. */
export interface ReadOutcome {
  readonly end: number | null;
}

/** A video opened for reading. */
export interface VideoSource {
  readonly width: number;
  readonly height: number;
  /** Seconds; Infinity when neither the file nor a seek past its end said. */
  readonly duration: number;
  /** Read the frames at `times` (seconds), each cropped to `crop` and scaled
   * to `size`, as RGB bytes; `onFrame` hears each as it arrives -- in about,
   * not exactly, their order -- with the picture it came from. A frame at or
   * past the outcome's `end` may have been heard before the end was known,
   * and is to be dropped. Rejects with a `ClipError` when the browser cannot
   * read a frame, or when `signal` aborts. */
  read(
    times: readonly number[],
    crop: Rect,
    size: readonly [number, number],
    onFrame: (j: number, rgb: Bytes, picture: HTMLVideoElement) => void,
    signal: AbortSignal,
  ): Promise<ReadOutcome>;
  /** Let go of every element but the one on screen. */
  close(): void;
}

export type OpenVideo = (url: string, evidence: Evidence, signal: AbortSignal, display: HTMLVideoElement) => Promise<VideoSource>;

/** What the reader needs from the browser, so a test can stand in for it. */
export interface ReaderDeps {
  /** A new, unattached video element. */
  create(): HTMLVideoElement;
  /** The frame on screen in `video`, `crop` of it scaled to `w` x `h`, as RGB. */
  grab(video: HTMLVideoElement, crop: Rect, w: number, h: number): Bytes;
  /** Call `onFrame` with the media time of every frame `video` presents,
   * until the returned function is called; null where the browser cannot say. */
  watch(video: HTMLVideoElement, onFrame: (mediaTime: number) => void): (() => void) | null;
  /** Whether the page is on screen (not a tab in the background). */
  visible(): boolean;
}

function aborted(): DOMException {
  return new DOMException("The reading was stopped", "AbortError");
}

const isAbort = (error: unknown) => error instanceof DOMException && error.name === "AbortError";

/** Wait for one of `video`'s events, failing on its `error` event, on
 * `signal`, or once `limitMs` of on-screen time has passed (`onLimit`). */
function waitFor(
  video: HTMLVideoElement,
  event: string,
  signal: AbortSignal,
  limitMs: number,
  onError: () => Error,
  onLimit: () => Error,
  visible: () => boolean,
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
      if (visible()) waited += now - last;
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

/** The browser's word on why `video` failed, for the evidence. */
function mediaEvidence(video: HTMLVideoElement, evidence: Evidence): Evidence {
  const error = video.error;
  return { ...evidence, mediaError: error?.code ?? 0, ...(error?.message ? { errorMessage: error.message } : {}) };
}

/** Seek `video` to `t` and wait for the frame there. */
function seek(video: HTMLVideoElement, t: number, evidence: Evidence, signal: AbortSignal, deps: ReaderDeps): Promise<void> {
  const failed = () => new ClipError(`a frame could not be decoded (${video.error?.message ?? "no detail"})`, mediaEvidence(video, evidence));
  const slow = () => new ClipError(`the seek to ${t.toFixed(3)} s never finished`, { ...evidence, seekTimedOut: true });
  const seeked = waitFor(video, "seeked", signal, SEEK_LIMIT_MS, failed, slow, deps.visible);
  video.currentTime = t;
  return seeked;
}

/** A video that does not state its length (a browser recorder's WebM) is
 * asked for it by a seek far past its end, which makes the browser find the
 * end; then it goes back to the start. Its length stays Infinity if the
 * browser could not tell. */
async function recoverLength(video: HTMLVideoElement, evidence: Evidence, signal: AbortSignal, deps: ReaderDeps): Promise<void> {
  await new Promise<void>((resolve) => {
    let grace = 0;
    const finish = () => {
      window.clearTimeout(timer);
      window.clearTimeout(grace);
      video.removeEventListener("durationchange", changed);
      video.removeEventListener("seeked", seeked);
      resolve();
    };
    const changed = () => {
      if (Number.isFinite(video.duration)) finish();
    };
    // The seek past the end has landed: the length is known by now, or comes
    // with the next change, or not at all.
    const seeked = () => {
      if (Number.isFinite(video.duration)) finish();
      else grace = window.setTimeout(finish, LENGTH_GRACE_MS);
    };
    const timer = window.setTimeout(finish, LENGTH_LIMIT_MS);
    video.addEventListener("durationchange", changed);
    video.addEventListener("seeked", seeked);
    video.currentTime = 1e101;
  });
  if (signal.aborted) throw aborted();
  await seek(video, 0, evidence, signal, deps);
}

/** Open `url` in `video` and wait until its first frame can be drawn. */
async function open(video: HTMLVideoElement, url: string, evidence: Evidence, signal: AbortSignal, deps: ReaderDeps): Promise<Evidence> {
  const refused = () => new ClipError(`the browser refused the file (${video.error?.message ?? "no detail"})`, mediaEvidence(video, evidence));
  const slow = () => new ClipError("the file did not open in time", { ...evidence, seekTimedOut: true });
  video.muted = true;
  video.preload = "auto";
  const metadata = waitFor(video, "loadedmetadata", signal, OPEN_LIMIT_MS, refused, slow, deps.visible);
  video.src = url;
  video.load();
  await metadata;
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    throw new ClipError("the file has no picture", { ...evidence, width: video.videoWidth, height: video.videoHeight, duration: video.duration });
  }
  if (!Number.isFinite(video.duration)) await recoverLength(video, evidence, signal, deps);
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    await waitFor(video, "loadeddata", signal, OPEN_LIMIT_MS, refused, slow, deps.visible);
  }
  return { ...evidence, width: video.videoWidth, height: video.videoHeight, duration: video.duration };
}

/** Let `video` go of its file, so its decoder is freed. */
export function emptyVideo(video: HTMLVideoElement): void {
  video.pause();
  video.removeAttribute("src");
  video.load();
}

// -- The browser's own steps ------------------------------------------------------------

function browserGrab(): ReaderDeps["grab"] {
  let canvas: HTMLCanvasElement | null = null;
  let context: CanvasRenderingContext2D | null = null;
  return (video, crop, w, h) => {
    if (!canvas || canvas.width !== w || canvas.height !== h) {
      canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      context = canvas.getContext("2d", { willReadFrequently: true });
      if (context) {
        context.imageSmoothingEnabled = true;
        context.imageSmoothingQuality = "high";
      }
    }
    if (!context) throw new Error("no 2D canvas to read the frames into");
    context.drawImage(video, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, w, h);
    const rgba = context.getImageData(0, 0, w, h).data;
    const rgb = new Uint8Array(w * h * 3);
    for (let p = 0, from = 0, to = 0; p < w * h; p += 1, from += 4, to += 3) {
      rgb[to] = rgba[from] as number;
      rgb[to + 1] = rgba[from + 1] as number;
      rgb[to + 2] = rgba[from + 2] as number;
    }
    return rgb;
  };
}

function browserWatch(video: HTMLVideoElement, onFrame: (mediaTime: number) => void): (() => void) | null {
  if (typeof video.requestVideoFrameCallback !== "function") return null;
  let stopped = false;
  let handle = 0;
  const presented = (_now: number, metadata: { mediaTime: number }) => {
    if (stopped) return;
    onFrame(metadata.mediaTime);
    handle = video.requestVideoFrameCallback(presented);
  };
  handle = video.requestVideoFrameCallback(presented);
  return () => {
    stopped = true;
    video.cancelVideoFrameCallback(handle);
  };
}

const browserDeps = (): ReaderDeps => ({
  create: () => document.createElement("video"),
  grab: browserGrab(),
  watch: browserWatch,
  visible: () => document.visibilityState !== "hidden",
});

// -- Opening and reading ------------------------------------------------------------------

/** Open the file at `url` for reading, with `display` (the video on screen)
 * as the first of the elements that seek through it. */
export async function openVideo(
  url: string,
  evidence: Evidence,
  signal: AbortSignal,
  display: HTMLVideoElement,
  deps: ReaderDeps = browserDeps(),
): Promise<VideoSource> {
  const seen = await open(display, url, evidence, signal, deps);
  let second: HTMLVideoElement | null = null;
  try {
    second = deps.create();
    await open(second, url, evidence, signal, deps);
  } catch (error) {
    if (second) emptyVideo(second);
    second = null;
    if (isAbort(error) || signal.aborted) throw aborted();
    // The second element is only for speed: the first reads on alone.
  }
  const others = second ? [second] : [];
  return {
    width: seen.width as number,
    height: seen.height as number,
    duration: seen.duration as number,
    read: (times, crop, [w, h], onFrame, outer) => readFrames([display, ...others], seen, times, crop, w, h, onFrame, outer, deps),
    close: () => others.forEach(emptyVideo),
  };
}

/** What one element has seen as it reads. */
interface Seeker {
  readonly video: HTMLVideoElement;
  /** The pixels it drew last. */
  prev: Bytes | null;
  /** Frames whose pixels repeat the one before, held back until it is
   * clear whether the scene is still or the file has stopped. */
  held: Array<{ readonly j: number; readonly rgb: Bytes }>;
  /** The media time of the last frame it presented; null where unknown. */
  lastMedia: number | null;
  /** Seeks finished while the page was on screen, since it last was not. */
  visibleSeeks: number;
  stopWatch: (() => void) | null;
}

const same = (a: Bytes, b: Bytes) => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return false;
  return true;
};

async function readFrames(
  videos: readonly HTMLVideoElement[],
  evidence: Evidence,
  times: readonly number[],
  crop: Rect,
  w: number,
  h: number,
  onFrame: (j: number, rgb: Bytes, picture: HTMLVideoElement) => void,
  outer: AbortSignal,
  deps: ReaderDeps,
): Promise<ReadOutcome> {
  const step = times.length > 1 ? (times[1] as number) - (times[0] as number) : 0;
  const seekers: Seeker[] = videos.map((video) => {
    const seeker: Seeker = { video, prev: null, held: [], lastMedia: null, visibleSeeks: 0, stopWatch: null };
    seeker.stopWatch = deps.watch(video, (mediaTime) => {
      seeker.lastMedia = mediaTime;
    });
    return seeker;
  });
  // A failure in the last element still reading stops the reading, and the
  // failure is what the caller hears, not the others' being stopped.
  const local = new AbortController();
  const stop = () => local.abort();
  outer.addEventListener("abort", stop);
  let next = 0;
  const retry: number[] = [];
  /** The first frame past the file's real end, once known. */
  let cut = times.length;
  const alive = new Set(seekers);
  let failure: unknown = null;

  const take = (): number | undefined => {
    while (retry.length > 0) {
      const j = retry.shift() as number;
      if (j < cut) return j;
    }
    return next < cut ? next++ : undefined;
  };

  const pass = (seeker: Seeker, j: number, rgb: Bytes) => {
    if (j < cut) onFrame(j, rgb, seeker.video);
  };

  /** The frame `rgb` at `j` from `seeker`: passed on, held back, or the sign
   * that the file stops before it. False once reading should stop. */
  const examine = (seeker: Seeker, j: number, rgb: Bytes): boolean => {
    const t = times[j] as number;
    if (seeker.prev === null || !same(rgb, seeker.prev)) {
      for (const held of seeker.held) pass(seeker, held.j, held.rgb);
      seeker.held = [];
      seeker.prev = rgb;
      pass(seeker, j, rgb);
      return true;
    }
    seeker.held.push({ j, rgb });
    const behind =
      seeker.stopWatch !== null && seeker.lastMedia !== null
        ? t - seeker.lastMedia
        : t - (times[(seeker.held[0] as { j: number }).j] as number);
    if (!(deps.visible() && seeker.visibleSeeks >= 2 && behind > STALE_S)) return true;
    // The file gives no new frames from here. The first frame past its end
    // is the first asked for a step after the last one it showed.
    const last = seeker.stopWatch !== null && seeker.lastMedia !== null ? seeker.lastMedia : null;
    const first = (seeker.held[0] as { j: number }).j;
    const past = last === null ? first : times.findIndex((time) => time >= last + step - 1e-9);
    cut = Math.min(cut, past < 0 ? first : Math.min(first, past));
    seeker.held = [];
    return false;
  };

  const work = async (seeker: Seeker) => {
    for (;;) {
      const j = take();
      if (j === undefined) break;
      try {
        await seek(seeker.video, times[j] as number, evidence, local.signal, deps);
      } catch (error) {
        if (isAbort(error) || local.signal.aborted) return;
        if (alive.size > 1) {
          // Another element reads on, and takes this frame too.
          alive.delete(seeker);
          retry.push(j);
          if (seeker.video !== videos[0]) emptyVideo(seeker.video);
          return;
        }
        if (failure === null) failure = error;
        local.abort();
        return;
      }
      seeker.visibleSeeks = deps.visible() ? seeker.visibleSeeks + 1 : 0;
      let rgb: Bytes;
      try {
        rgb = deps.grab(seeker.video, crop, w, h);
      } catch (error) {
        if (failure === null) failure = error;
        local.abort();
        return;
      }
      if (!examine(seeker, j, rgb)) break;
    }
    // A still scene that lasted to the end of what was read is real.
    for (const held of seeker.held) pass(seeker, held.j, held.rgb);
    seeker.held = [];
  };

  try {
    await Promise.allSettled(seekers.map(work));
    // A frame given back by an element that dropped out after the others had
    // finished is read by one still reading.
    while (failure === null && !local.signal.aborted && retry.some((j) => j < cut)) {
      await Promise.allSettled([...alive].map(work));
    }
  } finally {
    outer.removeEventListener("abort", stop);
    for (const seeker of seekers) seeker.stopWatch?.();
  }
  if (outer.aborted) throw aborted();
  if (failure !== null) throw failure;
  if (local.signal.aborted) throw aborted();
  return { end: cut < times.length ? cut : null };
}
