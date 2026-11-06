/**
 * The trace under the visitor's video: the model's estimate frame by frame,
 * one violet line over the frames read, right turns above the middle line
 * and left below it. While the video is being read the trace is also the
 * progress: the frames read so far are a lighter ground, and the line draws
 * itself as the model answers. Once the video is read, a range input laid
 * over it (ownclip.ts) makes it the scrubber, and a playhead marks the frame
 * on screen.
 *
 * The trace has no ground truth to draw: nobody recorded how the wheel was
 * turned in a visitor's video.
 */

/** The smallest half-height of the trace, degrees of steering-wheel angle: a
 * video the model reads as nearly straight is drawn nearly flat, not blown
 * up to fill the strip. */
export const MIN_EXTENT_DEG = 20;
/** Room above and below the largest estimate. */
const HEADROOM = 1.15;

/** The angle at the trace's top edge (and, negated, its bottom): the largest
 * estimate either way with some headroom, and never less than
 * `MIN_EXTENT_DEG`. NaN (no answer yet) is skipped. */
export function traceExtent(angles: ArrayLike<number>): number {
  let largest = 0;
  for (let k = 0; k < angles.length; k += 1) {
    const v = Math.abs(angles[k] as number);
    if (v > largest) largest = v;
  }
  return Math.max(MIN_EXTENT_DEG, largest * HEADROOM);
}

/** The trace's y for `angle`, in a strip `height` tall whose half-height is
 * `extent` degrees: right turns (positive) up. */
export function traceY(angle: number, extent: number, height: number): number {
  const share = Math.max(-1, Math.min(1, angle / extent));
  return height / 2 - share * (height / 2);
}

interface Palette {
  readonly line: string;
  readonly ground: string;
  readonly read: string;
  readonly middle: string;
  readonly playhead: string;
}

function readPalette(element: HTMLElement): Palette {
  const style = getComputedStyle(element);
  const token = (name: string) => style.getPropertyValue(name).trim();
  return {
    line: token("--model"),
    ground: token("--console-2"),
    read: token("--trace-read"),
    middle: token("--console-line"),
    playhead: token("--strip-playhead"),
  };
}

export class Trace {
  private angles: Float32Array = new Float32Array(0);
  private read = 0;
  private frame = -1;
  private palette: Palette;
  private readonly context: CanvasRenderingContext2D | null;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.context = canvas.getContext("2d");
    this.palette = readPalette(canvas);
    if (typeof ResizeObserver === "function") new ResizeObserver(() => this.draw()).observe(canvas);
  }

  /** A new video of `total` frames: nothing read, nothing answered. */
  reset(total: number): void {
    this.angles = new Float32Array(total).fill(Number.NaN);
    this.read = 0;
    this.frame = -1;
    this.palette = readPalette(this.canvas);
    this.draw();
  }

  /** The frames read so far, and the model's estimates (NaN where none). */
  update(read: number, angles: Float32Array): void {
    this.read = read;
    this.angles = angles;
    this.draw();
  }

  /** The frame on screen, or -1 for none (no playhead). */
  setFrame(k: number): void {
    this.frame = k;
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
    const n = this.angles.length;
    if (n === 0) return;
    const step = width / n;
    const { palette } = this;

    // The frames read so far.
    context.fillStyle = palette.read;
    context.fillRect(0, 0, Math.min(width, this.read * step), height);

    // Straight ahead.
    context.strokeStyle = palette.middle;
    context.lineWidth = 1;
    context.setLineDash([3, 4]);
    context.beginPath();
    context.moveTo(0, Math.round(height / 2) + 0.5);
    context.lineTo(width, Math.round(height / 2) + 0.5);
    context.stroke();
    context.setLineDash([]);

    // The model's estimate, broken wherever a frame has none.
    const extent = traceExtent(this.angles);
    context.strokeStyle = palette.line;
    context.lineWidth = 2;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.beginPath();
    let drawing = false;
    for (let k = 0; k < n; k += 1) {
      const angle = this.angles[k] as number;
      if (!Number.isFinite(angle)) {
        drawing = false;
        continue;
      }
      const x = (k + 0.5) * step;
      const y = traceY(angle, extent, height);
      if (drawing) context.lineTo(x, y);
      else context.moveTo(x, y);
      drawing = true;
    }
    context.stroke();

    if (this.frame >= 0 && this.frame < n) {
      const x = (this.frame + 0.5) * step;
      context.fillStyle = palette.playhead;
      context.fillRect(Math.round(x) - 1, 0, 2, height);
      const angle = this.angles[this.frame] as number;
      if (Number.isFinite(angle)) {
        context.beginPath();
        context.arc(x, traceY(angle, extent, height), 4, 0, 2 * Math.PI);
        context.fillStyle = palette.line;
        context.fill();
        context.lineWidth = 2;
        context.strokeStyle = palette.playhead;
        context.stroke();
      }
    }
  }
}
