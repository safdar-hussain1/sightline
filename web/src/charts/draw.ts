/**
 * What every chart on the Results view draws with: SVG elements, a tween on
 * the page's frame clock, two easings, and the width a chart has to fill.
 *
 * A chart is laid out at the pixel width of its box, so its type stays the
 * page's own size at any width; when the box changes width the chart is laid
 * out again, at once. Motion runs on the same clock the headline's counts use,
 * so tests can step it by hand; under reduced motion a tween jumps to its end.
 */

import type { Frames } from "../sections/countup.ts";

const NS = "http://www.w3.org/2000/svg";

type Attrs = Readonly<Record<string, string | number | null>>;

/** Positions to a hundredth of a pixel: the drawing is no less exact, and the
 * markup stays readable. */
const tidy = (value: string | number) => (typeof value === "number" ? String(Math.round(value * 100) / 100) : value);

/** Set (or, for null, remove) each attribute. */
export function attrs(element: Element, values: Attrs): void {
  for (const [name, value] of Object.entries(values)) {
    if (value === null) element.removeAttribute(name);
    else element.setAttribute(name, tidy(value));
  }
}

/** A new SVG element with `values`, appended to `parent` if given. */
export function svg<K extends keyof SVGElementTagNameMap>(tag: K, values: Attrs = {}, parent?: Element): SVGElementTagNameMap[K] {
  const element = document.createElementNS(NS, tag);
  attrs(element, values);
  parent?.append(element);
  return element;
}

/** A text element of one or more lines, `gap` pixels apart. */
export function lines(text: SVGTextElement, rows: readonly string[], x: number, gap: number): void {
  text.replaceChildren(
    ...rows.map((row, i) => {
      const span = svg("tspan", { x, dy: i === 0 ? 0 : gap });
      span.textContent = row;
      return span;
    }),
  );
}

/** Fast in, gentle settle (the page's counts use the same). */
export const easeOut = (t: number) => 1 - (1 - Math.min(1, Math.max(0, t))) ** 4;

/** Gentle at both ends: for a morph between two states the visitor chose. */
export const easeInOut = (t: number) => {
  const p = Math.min(1, Math.max(0, t));
  return p < 0.5 ? 4 * p * p * p : 1 - (-2 * p + 2) ** 3 / 2;
};

const BROWSER_FRAMES: Frames = {
  now: () => performance.now(),
  request: (callback) => requestAnimationFrame(callback),
  cancel: (handle) => cancelAnimationFrame(handle),
};

/** Call `step` with the time elapsed, as a share of `duration` (0 to 1),
 * every frame until it reaches 1; under `reduced` motion once, at 1. Returns
 * a function that stops it at its end. */
export function tween(
  duration: number,
  step: (progress: number) => void,
  { reduced, frames = BROWSER_FRAMES }: { readonly reduced: boolean; readonly frames?: Frames | undefined },
): () => void {
  if (reduced || !(duration > 0)) {
    step(1);
    return () => undefined;
  }
  const start = frames.now();
  let handle = 0;
  let done = false;
  const draw = (time: number) => {
    if (done) return;
    const progress = Math.min(1, Math.max(0, (time - start) / duration));
    step(progress);
    if (progress < 1) handle = frames.request(draw);
    else done = true;
  };
  draw(start);
  return () => {
    if (done) return;
    done = true;
    frames.cancel(handle);
    step(1);
  };
}

/** The share of an element's own animation done at `progress` of the whole,
 * when it starts `delay` and lasts `length` (both shares of the whole). */
export const local = (progress: number, delay: number, length: number) => Math.min(1, Math.max(0, (progress - delay) / length));

/** Calls `layout` with the box's width now and whenever it changes. */
export function watchWidth(box: HTMLElement, layout: (width: number) => void): void {
  let width = Math.round(box.getBoundingClientRect().width);
  layout(width);
  if (typeof ResizeObserver !== "function") return;
  new ResizeObserver(() => {
    const next = Math.round(box.getBoundingClientRect().width);
    if (next !== width && next > 0) {
      width = next;
      layout(width);
    }
  }).observe(box);
}
