/**
 * Headline figures that count up to their value the first time they are seen.
 *
 * A figure counts in its own final text's shape -- the same decimals, the same
 * unit, a true minus, its thousands grouped -- so the count never shows a
 * digit the figure does not have, and it ends on exactly the text the
 * formatters wrote. A figure stated as a bound ("<0.001°") or with no value
 * does not count at all: there is nothing honest to count through. Under
 * reduced motion every figure is shown at once.
 */

const MINUS = "−";

/** Fast at first, settling gently onto the value. */
export function easeOut(progress: number): number {
  const p = Math.min(1, Math.max(0, progress));
  return 1 - (1 - p) ** 4;
}

/** The parts of a formatted figure: its sign, its digits and what follows. */
const FIGURE = /^(−?)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(.*)$/;

function grouped(digits: string): string {
  return digits.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * The text of a figure at `progress` (0 to 1) of its count from `from` to
 * `value`, where `final` is the figure's own text. At 1 it is `final` itself.
 */
export function countText(final: string, value: number, from: number, progress: number): string {
  return progress >= 1 ? final : textAt(final, value, from, easeOut(progress));
}

/** The text of a figure at `eased` (0 to 1) of the way from `from` to
 * `value`, already eased: for a drawing that moves in step with a count. At 1
 * it is `final` itself. */
export function textAt(final: string, value: number, from: number, eased: number): string {
  const parts = FIGURE.exec(final);
  if (!parts || !Number.isFinite(value) || !Number.isFinite(from) || eased >= 1) {
    return final;
  }
  const [, , whole = "", fraction = "", unit = ""] = parts;
  const current = from + (value - from) * Math.max(0, eased);
  const text = Math.abs(current).toFixed(fraction.length);
  const [intPart = "", fracPart] = text.split(".");
  const digits = (whole.includes(",") || final.includes(",") ? grouped(intPart) : intPart) + (fracPart ? `.${fracPart}` : "");
  // A count passing through zero shows zero, never a negative zero.
  const negative = current < 0 && Number(text) !== 0;
  return `${negative ? MINUS : ""}${digits}${unit}`;
}

/** One figure to count: its element, its final text and value, where it
 * starts, and how long after the others it starts (ms). `onProgress` hears
 * how far the count has come, eased (0 to 1, ending at exactly 1), so a
 * drawing of the figure can move in step with its number. */
export interface Count {
  readonly el: HTMLElement;
  readonly final: string;
  readonly value: number;
  readonly from: number;
  readonly delay?: number;
  readonly onProgress?: (eased: number) => void;
}

/** The page's clock and animation frames; tests pass their own. */
export interface Frames {
  now(): number;
  request(callback: (time: number) => void): number;
  cancel(handle: number): void;
}

const BROWSER_FRAMES: Frames = {
  now: () => performance.now(),
  request: (callback) => requestAnimationFrame(callback),
  cancel: (handle) => cancelAnimationFrame(handle),
};

/**
 * Count each figure from its start to its final text over `duration` ms, after
 * its own delay. Under `reduced` motion each shows its final text at once.
 * Returns a function that stops the count, leaving every final text shown.
 */
export function runCounts(
  counts: readonly Count[],
  { duration, reduced, frames = BROWSER_FRAMES }: { duration: number; reduced: boolean; frames?: Frames },
): () => void {
  const finish = () => {
    for (const { el, final, onProgress } of counts) {
      el.textContent = final;
      onProgress?.(1);
    }
  };
  if (reduced || counts.length === 0) {
    finish();
    return () => undefined;
  }
  const start = frames.now();
  let handle = 0;
  let stopped = false;
  const draw = (time: number) => {
    if (stopped) return;
    let running = false;
    for (const { el, final, value, from, delay = 0, onProgress } of counts) {
      const progress = Math.max(0, time - start - delay) / duration;
      el.textContent = countText(final, value, from, progress);
      onProgress?.(progress >= 1 ? 1 : easeOut(progress));
      if (progress < 1) running = true;
    }
    if (running) handle = frames.request(draw);
  };
  draw(start);
  return () => {
    stopped = true;
    frames.cancel(handle);
    finish();
  };
}
