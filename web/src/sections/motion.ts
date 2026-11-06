/**
 * The sections' shared motion: one easing, the visitor's motion preference,
 * and whether an element is on screen.
 *
 * The page's motion is calm and has one rhythm: things arrive fast and settle
 * slowly (an ease-out), nothing loops for show, and under reduced motion every
 * change is instant. The stylesheet's reduced-motion rules cannot reach
 * animations started from script, so each is checked here before it starts.
 */

/** Fast in, gentle settle: the easing of every motion the sections start,
 * and the stylesheet's --ease-out. */
export const EASE_OUT = "cubic-bezier(0.16, 1, 0.3, 1)";

/** How long each kind of motion takes, ms: the stylesheet's --motion-move
 * and --motion-settle, and the one step only scripts take, a chart or a
 * figure drawing itself in the first time it comes into view. */
export const MOTION_MS = {
  /** Something arriving: a view, a picture, a number's text. */
  move: 320,
  /** A chart changing into the choice the visitor made. */
  settle: 560,
  /** A chart or a figure drawing itself in, once. */
  draw: 1400,
} as const;

export function prefersReducedMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Whether any of `el` is inside the viewport, grown by `margin` pixels at
 * the bottom (so a motion can start just before it comes into sight). */
export function onScreen(el: Element, margin = 0): boolean {
  const box = el.getBoundingClientRect();
  if (box.width === 0 && box.height === 0) return false;
  return box.bottom > 0 && box.top < innerHeight + margin;
}
