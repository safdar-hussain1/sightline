/**
 * The error strip: how far off the model is on every frame of the clip, and
 * where the clip is now.
 *
 * One column per frame. A column's height is the gap between the model's angle
 * and the driver's on that frame, on one scale shared by both models so that
 * switching models compares like with like; its colour is how hard the driver
 * was turning (straight, gentle, sharp -- the engine's own bins, from the
 * driver's angle), on one violet ramp from dim to bright. A hairline marks the
 * clip's average error. The strip is also the clip's scrubber: a slider that
 * takes clicks, drags and the keyboard.
 *
 * It draws from the expected outputs in the clip file the moment the clip
 * loads, and the cockpit overwrites each column with the model's live output
 * as the browser computes it.
 */

import type { ModelKey } from "../data/types.ts";

export type Regime = "straight" | "gentle" | "sharp";

/** The |driver angle| boundaries between straight and gentle, and gentle and
 * sharp, degrees. The engine's bins; a test reads them from its source. */
export const REGIME_BINS_DEG: readonly [number, number] = [5, 30];

export function regimeOf(humanDeg: number): Regime {
  const turn = Math.abs(humanDeg);
  if (turn < REGIME_BINS_DEG[0]) return "straight";
  if (turn < REGIME_BINS_DEG[1]) return "gentle";
  return "sharp";
}

/** Headroom above the clip's largest error, so the tallest column does not
 * touch the playhead's knob. */
const HEADROOM = 1.1;

/** The error, in degrees, that fills the strip's height: the largest error
 * either model makes anywhere on the clip, with a little headroom. */
export function errorScale(
  labels: readonly number[],
  expected: Readonly<Record<ModelKey, readonly number[]>>,
): number {
  let largest = 0;
  for (const values of Object.values(expected)) {
    values.forEach((value, k) => {
      largest = Math.max(largest, Math.abs(value - (labels[k] ?? value)));
    });
  }
  return largest > 0 ? largest * HEADROOM : 1;
}

/** Each frame's absolute error as a share of `scale`, at most 1; NaN where the
 * frame has no prediction yet. */
export function columnHeights(labels: readonly number[], values: ArrayLike<number>, scale: number): Float32Array {
  const heights = new Float32Array(labels.length);
  for (let k = 0; k < labels.length; k += 1) {
    const value = values[k] ?? Number.NaN;
    heights[k] = Number.isFinite(value) ? Math.min(1, Math.abs(value - (labels[k] as number)) / scale) : Number.NaN;
  }
  return heights;
}

/** The frame drawn at `x` on a strip `width` wide holding `n` frames. */
export function frameAtX(x: number, width: number, n: number): number {
  return Math.min(n - 1, Math.max(0, Math.floor((x / width) * n)));
}

/** Where a key press on the strip moves the clip: a frame with the arrows, a
 * second's worth with Page Up and Page Down, the ends with Home and End. Null
 * for any other key. */
export function stepForKey(key: string, k: number, n: number, fps: number): number | null {
  const second = Math.round(fps);
  const moves: Record<string, number> = {
    ArrowRight: k + 1,
    ArrowUp: k + 1,
    ArrowLeft: k - 1,
    ArrowDown: k - 1,
    PageUp: k + second,
    PageDown: k - second,
    Home: 0,
    End: n - 1,
  };
  const target = moves[key];
  return target === undefined ? null : Math.min(n - 1, Math.max(0, target));
}

/** The colours the strip draws with, read from its element's CSS. */
interface Palette {
  readonly straight: string;
  readonly gentle: string;
  readonly sharp: string;
  readonly average: string;
  readonly playhead: string;
}

function readPalette(element: HTMLElement): Palette {
  const style = getComputedStyle(element);
  const token = (name: string) => style.getPropertyValue(name).trim();
  return {
    straight: token("--regime-straight"),
    gentle: token("--regime-gentle"),
    sharp: token("--regime-sharp"),
    average: token("--strip-average"),
    playhead: token("--strip-playhead"),
  };
}

/** Space above the columns for the playhead's knob, CSS pixels. */
const KNOB = 6;

export interface StripOptions {
  /** The visitor moved the clip to frame `k`. */
  onSeek(k: number): void;
  /** The slider's spoken value at frame `k`. */
  describe(k: number): string;
}

/** The strip on screen: a canvas inside an element with the slider role. */
export class ErrorStrip {
  private labels: readonly number[] = [];
  private regimes: Regime[] = [];
  private heights: Float32Array = new Float32Array(0);
  private scale = 1;
  private average = Number.NaN;
  private frame = 0;
  private palette: Palette;
  private readonly context: CanvasRenderingContext2D | null;

  constructor(
    private readonly element: HTMLElement,
    private readonly canvas: HTMLCanvasElement,
    private readonly options: StripOptions,
  ) {
    this.context = canvas.getContext("2d");
    this.palette = readPalette(element);
    element.addEventListener("keydown", (event) => {
      const target = stepForKey(event.key, this.frame, this.labels.length, this.fps);
      if (target !== null && this.labels.length > 0) {
        event.preventDefault();
        this.options.onSeek(target);
      }
    });
    element.addEventListener("pointerdown", (event) => {
      if (this.labels.length === 0) return;
      try {
        element.setPointerCapture(event.pointerId);
      } catch {
        // A pointer the browser no longer tracks: the press still seeks, without the drag.
      }
      this.seekTo(event.clientX);
    });
    element.addEventListener("pointermove", (event) => {
      if (element.hasPointerCapture(event.pointerId)) this.seekTo(event.clientX);
    });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  private fps = 1;

  private seekTo(clientX: number): void {
    const box = this.canvas.getBoundingClientRect();
    this.options.onSeek(frameAtX(clientX - box.left, box.width, this.labels.length));
  }

  /** A new clip: its driver angles, frame rate, error scale and average error. */
  setClip(labels: readonly number[], fps: number, scale: number): void {
    this.labels = labels;
    this.fps = fps;
    this.scale = scale;
    this.regimes = labels.map(regimeOf);
    this.heights = new Float32Array(labels.length).fill(Number.NaN);
    this.element.setAttribute("aria-valuemin", "0");
    this.element.setAttribute("aria-valuemax", String(labels.length - 1));
    this.palette = readPalette(this.element);
  }

  /** The model's angle on every frame (NaN where there is none yet) and its
   * average error on the clip. */
  setValues(values: ArrayLike<number>, averageError: number): void {
    this.heights = columnHeights(this.labels, values, this.scale);
    this.average = averageError;
    this.draw();
  }

  /** Move the playhead. `announce` also updates the slider's spoken value --
   * on a seek, not on every frame of playback, which would chatter. */
  setFrame(k: number, announce: boolean): void {
    this.frame = k;
    if (announce) {
      this.element.setAttribute("aria-valuenow", String(k));
      this.element.setAttribute("aria-valuetext", this.options.describe(k));
    }
    this.draw();
  }

  draw(): void {
    const context = this.context;
    if (!context) return;
    const ratio = window.devicePixelRatio || 1;
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (width === 0 || height === 0) return;
    if (this.canvas.width !== Math.round(width * ratio) || this.canvas.height !== Math.round(height * ratio)) {
      this.canvas.width = Math.round(width * ratio);
      this.canvas.height = Math.round(height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const n = this.labels.length;
    if (n === 0) return;

    const plot = height - KNOB;
    const column = width / n;
    const gap = column >= 3 ? 1 : 0;
    for (let k = 0; k < n; k += 1) {
      const share = this.heights[k] as number;
      if (!(share > 0)) continue;
      const h = Math.max(1, share * plot);
      context.fillStyle = this.palette[this.regimes[k] as Regime];
      context.fillRect(k * column, height - h, column - gap, h);
    }

    if (this.average > 0 && this.average < this.scale) {
      const y = Math.round(height - (this.average / this.scale) * plot) + 0.5;
      context.strokeStyle = this.palette.average;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }

    const x = (this.frame + 0.5) * column;
    context.fillStyle = this.palette.playhead;
    context.fillRect(x - 1, KNOB, 2, height - KNOB);
    context.beginPath();
    context.arc(x, KNOB, KNOB - 1, 0, 2 * Math.PI);
    context.fill();
  }
}
