/**
 * What the page around the cockpit may read of it.
 *
 * How it works draws the model's input and answer from the clip the cockpit
 * has loaded, and the Results view checks this browser's answers against the
 * ones computed in Python. Both read the cockpit through this interface, which
 * carries no code of the cockpit's: the cockpit, the model runtime and the clip
 * arrive in a chunk of their own, and the page's sections must not pull them
 * in.
 */

import type { Clip, ModelKey } from "../data/types.ts";
import type { Bytes, DecodeCheck } from "./sprites.ts";

export type CockpitState = "loading" | "ready" | "unavailable";

export interface LiveStatus {
  readonly state: CockpitState;
  /** The footage is playing. */
  readonly playing: boolean;
  /** The frame on screen. */
  readonly frame: number;
  /** The clip whose frames are decoded: the one on screen, or the one loading
   * once its frames are in. Null while no frame of any clip is to be had. */
  readonly clip: Clip | null;
  /** The model on screen. */
  readonly model: ModelKey;
}

/** One frame as the model saw it, and what it answered. */
export interface LiveFrame {
  readonly clip: Clip;
  readonly k: number;
  readonly model: ModelKey;
  /** The model's input frames, RGB, frame_w x frame_h each: frame k first,
   * then the history frames stacked with it, in the clip's lag order. */
  readonly inputs: readonly Bytes[];
  /** The model's angle on frame k, degrees; null until it has run on it. */
  readonly angle: number | null;
  /** Where it looked on frame k, one byte per pixel (see `quantizeSaliency`);
   * null until it has run on it. */
  readonly heat: Uint8Array | null;
}

/** How this browser's angles on the loaded clip compare with the clip's
 * published ones, computed in Python by the reference runtime on the
 * reference decoder's frames. The comparison measures the runtimes only when
 * `check` is "match": the browser then ran the model on the very same pixels. */
export interface Agreement {
  readonly clip: Clip;
  readonly model: ModelKey;
  readonly check: DecodeCheck["status"];
  /** Frames the browser has run so far, of `total`. */
  readonly compared: number;
  readonly total: number;
  /** The largest difference over the frames compared, degrees; NaN before
   * any. */
  readonly maxDiffDeg: number;
}

export interface LiveCockpit {
  status(): LiveStatus;
  /** Frame k of the decoded clip, for the model on screen; null without one. */
  frameAt(k: number): LiveFrame | null;
  /** The comparison for the model on screen; null without a decoded clip. */
  agreement(): Agreement | null;
  /** This browser's time to run the model, milliseconds per frame, over every
   * batch after each model's first (which carries the runtime's warm-up); null
   * until there is one. */
  msPerFrame(): number | null;
}

/** The frames run so far (`done`) and the largest difference there between
 * this browser's angles and `expected`. */
export function compareAngles(
  angles: ArrayLike<number>,
  done: ArrayLike<number>,
  expected: readonly number[],
): { compared: number; maxDiff: number } {
  let compared = 0;
  let maxDiff = Number.NaN;
  for (let k = 0; k < expected.length; k += 1) {
    if (!done[k]) continue;
    const diff = Math.abs((angles[k] as number) - (expected[k] as number));
    // An answer that is not a number differs by more than any number: it is
    // counted as an infinite difference, never skipped.
    const counted = Number.isNaN(diff) ? Number.POSITIVE_INFINITY : diff;
    maxDiff = compared === 0 ? counted : Math.max(maxDiff, counted);
    compared += 1;
  }
  return { compared, maxDiff };
}
