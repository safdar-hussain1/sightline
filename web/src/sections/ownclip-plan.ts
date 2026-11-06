/**
 * Reading a visitor's own video, planned: which part of each frame the model
 * reads, when the frames are taken, what went wrong when a file cannot be
 * read, and the states a reading passes through. Pure functions, no DOM, so
 * every rule here is tested on its own (ownclip.ts does the reading).
 */

import { CROP_ROWS } from "../cockpit/geometry.ts";
import type { Gaps } from "../data/timing.ts";

/** The most frames the page reads from one video. A longer video is read from
 * its start up to here, and the page says so. */
export const MAX_FRAMES = 600;

// -- Where the model looks ----------------------------------------------------------

/** A source rectangle on a video frame, in its pixels: the part `drawImage`
 * reads. */
export interface Rect {
  readonly sx: number;
  readonly sy: number;
  readonly sw: number;
  readonly sh: number;
}

/**
 * The part of a `videoW` x `videoH` frame the model reads, for a model trained
 * on camera frames of `camera` ([width, height], from the clips' own record).
 *
 * The rule: training kept the bottom `CROP_ROWS` rows of each camera frame at
 * full width. A video frame of the camera's shape, or wider, gives the same
 * share of its height, at full width. A frame taller than the camera's shape
 * (a phone held upright) first gives up its top and bottom: the band across
 * its middle with the camera's shape, as wide as the frame; then that band's
 * bottom share. So the part read always has the training crop's height share,
 * and from a portrait video also the training crop's shape.
 */
export function cropRect(videoW: number, videoH: number, camera: readonly [number, number]): Rect {
  const [cameraW, cameraH] = camera;
  return readBand(videoW, videoH, cameraW / cameraH, Math.min(CROP_ROWS, cameraH) / cameraH);
}

/** `cropRect`'s rule for a camera of `aspect` (width over height) whose
 * bottom `share` of the height is read. */
export function readBand(videoW: number, videoH: number, aspect: number, share: number): Rect {
  const band = Math.min(videoH, videoW / aspect);
  const bandTop = (videoH - band) / 2;
  const sh = band * share;
  return { sx: 0, sy: bandTop + band - sh, sw: videoW, sh };
}

// -- When the frames are taken --------------------------------------------------------

/** How the page steps through a video: `rate` frames a second, `stepS` apart,
 * so that a frame and its history frame `lag` steps back are `gapS` apart. */
export interface Pace {
  /** The model's oldest history frame, in frames back. */
  readonly lag: number;
  /** The measured median time between a frame and the one `lag` before it on
   * the drive the model learned from; null where it was not measured. */
  readonly gapS: number | null;
  readonly stepS: number;
  readonly rate: number;
}

/**
 * The pace for a model with history `lags` (frames back), given the measured
 * gaps of the drive it learned from (`gaps`, from results.json's timing) and
 * the clips' nominal frame rate.
 *
 * The model only ever compares a frame with the one `lag` frames before it, so
 * the page steps through a video at the pace that puts those two the measured
 * median gap apart: `gap / lag` seconds a step. With more than one lag, the
 * oldest is exact and the others fall in proportion. Where no gap was
 * measured the page steps at the clips' frame rate and counts frames only.
 */
export function readingPace(lags: readonly number[], gaps: Gaps | null, clipFps: number): Pace {
  const lag = lags.length === 0 ? 0 : Math.max(...lags);
  const gapS = lag > 0 ? (gaps?.get(lag) ?? null) : null;
  if (gapS === null || !(gapS > 0)) {
    return { lag, gapS: null, stepS: 1 / clipFps, rate: clipFps };
  }
  return { lag, gapS, stepS: gapS / lag, rate: lag / gapS };
}

/** The whole steps of `1 / fps` seconds inside `duration` seconds; none for a
 * length that is not a positive finite number. The epsilon keeps a step that
 * ends exactly at the end from being lost to rounding. */
function wholeSteps(duration: number, fps: number): number {
  if (!Number.isFinite(duration) || !(duration > 0) || !Number.isFinite(fps) || !(fps > 0)) return 0;
  return Math.floor(duration * fps + 1e-9);
}

/**
 * Where to take the frames of a video `duration` seconds long, at `fps`
 * frames a second: one per whole step inside the video, each in the middle of
 * its step (so a decoder that rounds a seek either way still lands in it), at
 * most `maxFrames` of them from the start. A video shorter than one step has
 * none.
 */
export function sampleTimes(duration: number, fps: number, maxFrames = MAX_FRAMES): number[] {
  const n = Math.min(maxFrames, wholeSteps(duration, fps));
  return Array.from({ length: n }, (_, j) => (j + 0.5) / fps);
}

/** Whether a video is longer than the page reads, so only its start is read. */
export function exceedsCap(duration: number, fps: number, maxFrames = MAX_FRAMES): boolean {
  return wholeSteps(duration, fps) > maxFrames;
}

// -- What a file is -------------------------------------------------------------------

/** The video containers the page knows by their first bytes. */
export type Container = "mp4" | "matroska" | "avi" | "ogg" | "mpeg";

const ascii = (bytes: Uint8Array, at: number, text: string) =>
  bytes.length >= at + text.length && [...text].every((c, i) => bytes[at + i] === c.charCodeAt(0));

/** The ISO base media boxes a QuickTime or MP4 file can open with. */
const ISO_BOXES = ["ftyp", "moov", "mdat", "free", "skip", "wide", "pnot"];

/**
 * The container `bytes` (the first bytes of a file) open, or null when they
 * open none the page knows. Only used once the browser has refused a file, to
 * tell a video it cannot decode from a file that is no video at all.
 */
export function sniffContainer(bytes: Uint8Array): Container | null {
  if (ISO_BOXES.some((box) => ascii(bytes, 4, box))) return "mp4";
  if (bytes[0] === 0x1a && bytes[1] === 0x45 && bytes[2] === 0xdf && bytes[3] === 0xa3) return "matroska";
  if (ascii(bytes, 0, "RIFF") && ascii(bytes, 8, "AVI ")) return "avi";
  if (ascii(bytes, 0, "OggS")) return "ogg";
  if (bytes[0] === 0 && bytes[1] === 0 && bytes[2] === 1 && bytes[3] === 0xba) return "mpeg";
  if (bytes.length > 188 && bytes[0] === 0x47 && bytes[188] === 0x47) return "mpeg";
  return null;
}

// -- What went wrong ------------------------------------------------------------------

/** The kinds of failure the page tells apart, each with its own message. */
export type Failure = "not-a-video" | "codec" | "empty" | "too-long" | "unknown";

/** What the page saw of a file before it gave up on it. */
export interface Evidence {
  /** The file's size in bytes. */
  readonly bytes?: number;
  /** Its container, sniffed from its first bytes; null when it opens none. */
  readonly container?: Container | null;
  /** The browser's MediaError code, when it refused the file. */
  readonly mediaError?: number;
  /** The picture's size and the video's length, once the browser read them. */
  readonly width?: number;
  readonly height?: number;
  readonly duration?: number;
  /** The frames there are to read, and the fewest one answer needs. */
  readonly frames?: number;
  readonly needed?: number;
  /** A seek that never finished. */
  readonly seekTimedOut?: boolean;
}

/** A video the page could not read, with what it saw. */
export class ClipError extends Error {
  constructor(
    message: string,
    readonly evidence: Evidence,
  ) {
    super(message);
    this.name = "ClipError";
  }
}

/** Which kind of failure `err` is. Anything that is not a `ClipError` -- the
 * model not downloading, a canvas that cannot be read -- is "unknown". */
export function classifyFailure(err: unknown): Failure {
  if (!(err instanceof ClipError)) return "unknown";
  const e = err.evidence;
  if (e.bytes === 0) return "empty";
  if (e.mediaError !== undefined) return e.container ? "codec" : "not-a-video";
  if (e.seekTimedOut) return "codec";
  if (e.width === 0 || e.height === 0) return "not-a-video";
  if (e.duration !== undefined && !Number.isFinite(e.duration)) return "too-long";
  if (e.frames !== undefined && e.needed !== undefined && e.frames < e.needed) return "empty";
  return "unknown";
}

// -- The states of a reading ------------------------------------------------------------

/** Where a reading stands. `total` frames are read; `answerable` of them have
 * the history the model needs, and get an answer each. */
export type OwnClipState =
  | { readonly phase: "idle" }
  | { readonly phase: "loading"; readonly name: string }
  | {
      readonly phase: "sampling";
      readonly name: string;
      readonly read: number;
      readonly total: number;
      readonly answered: number;
      readonly answerable: number;
      readonly capped: boolean;
    }
  | {
      readonly phase: "running";
      readonly name: string;
      readonly total: number;
      readonly answered: number;
      readonly answerable: number;
      readonly capped: boolean;
    }
  | { readonly phase: "done"; readonly name: string; readonly total: number; readonly answerable: number; readonly capped: boolean }
  | { readonly phase: "error"; readonly name: string; readonly failure: Failure };

export type OwnClipEvent =
  /** A file was chosen: it replaces whatever was on screen. */
  | { readonly type: "choose"; readonly name: string }
  /** The browser opened it and the frames to read are known. */
  | { readonly type: "opened"; readonly total: number; readonly answerable: number; readonly capped: boolean }
  /** The first `count` frames have been read. */
  | { readonly type: "read"; readonly count: number }
  /** The model has answered `count` frames. */
  | { readonly type: "answered"; readonly count: number }
  | { readonly type: "fail"; readonly failure: Failure }
  | { readonly type: "cancel" };

/**
 * The state after `event`: idle, loading, sampling (frames read so far), then
 * running (the model finishing the frames read), then done -- or error from
 * any of the working states, or idle again on cancel. A new file replaces any
 * state. An event that does not belong to the state, or a count that goes
 * backwards or past the end, leaves the state as it was (the same object).
 */
export function advance(state: OwnClipState, event: OwnClipEvent): OwnClipState {
  const working = state.phase === "loading" || state.phase === "sampling" || state.phase === "running";
  switch (event.type) {
    case "choose":
      return { phase: "loading", name: event.name };
    case "cancel":
      return working ? { phase: "idle" } : state;
    case "fail":
      return working ? { phase: "error", name: state.name, failure: event.failure } : state;
    case "opened":
      if (state.phase !== "loading") return state;
      return { phase: "sampling", name: state.name, read: 0, total: event.total, answered: 0, answerable: event.answerable, capped: event.capped };
    case "read": {
      if (state.phase !== "sampling" || event.count < state.read || event.count > state.total) return state;
      if (event.count < state.total) return { ...state, read: event.count };
      return finish({ phase: "running", name: state.name, total: state.total, answered: state.answered, answerable: state.answerable, capped: state.capped });
    }
    case "answered": {
      if (state.phase !== "sampling" && state.phase !== "running") return state;
      if (event.count < state.answered || event.count > state.answerable) return state;
      const next = { ...state, answered: event.count };
      return next.phase === "running" ? finish(next) : next;
    }
  }
}

/** Running with every frame answered is done. */
function finish(state: Extract<OwnClipState, { phase: "running" }>): OwnClipState {
  return state.answered < state.answerable
    ? state
    : { phase: "done", name: state.name, total: state.total, answerable: state.answerable, capped: state.capped };
}
