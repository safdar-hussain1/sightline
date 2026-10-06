/**
 * The steering wheel: the model's wheel, the driver's ghost, and the gap
 * between them.
 *
 * Angles are steering-wheel degrees, clockwise positive, as the labels and the
 * model use them. The wheel turns by the angle itself, never wrapped: a label
 * of 501.8° is a wheel turned past one and a quarter turns, and the drawing
 * says so. The model's wheel is drawn solid in its violet; the driver's is a
 * ghost -- faint spokes behind the model's and an outlined pointer on the track
 * outside the rim -- and the arc along that track between the two pointers is
 * how far apart they are.
 */

/** The drawing's centre; its viewBox is twice this on each side. */
export const WHEEL_CENTRE = 120;

/** Radius of the track outside the rim that the pointers and the gap ride. */
const TRACK_R = 111;
/** Radius of the rim's centre line. */
const RIM_R = 88;
/** Half the angular width of the stripe at the top of the rim, in degrees. */
const STRIPE_HALF_DEG = 7;
/** The largest gap the arc draws: an arc of a whole turn has no length. */
const MAX_ARC_DEG = 359.5;

/** How far the wheel is drawn turned for a steering angle: the angle itself. */
export function wheelRotationDeg(angleDeg: number): number {
  return angleDeg;
}

/** The point at `angleDeg` clockwise from twelve o'clock on a circle of radius
 * `r` around (`cx`, `cy`), in SVG coordinates (y grows downwards). */
export function rimPoint(angleDeg: number, r: number, cx: number, cy: number): [number, number] {
  const t = (angleDeg * Math.PI) / 180;
  return [cx + r * Math.sin(t), cy - r * Math.cos(t)];
}

const round = (v: number) => Math.round(v * 100) / 100;

/** An SVG arc from `fromDeg` clockwise to `toDeg` (fromDeg < toDeg). */
function arc(fromDeg: number, toDeg: number, r: number, cx: number, cy: number): string {
  const [x0, y0] = rimPoint(fromDeg, r, cx, cy);
  const [x1, y1] = rimPoint(toDeg, r, cx, cy);
  const large = toDeg - fromDeg > 180 ? 1 : 0;
  return `M${round(x0)} ${round(y0)}A${r} ${r} 0 ${large} 1 ${round(x1)} ${round(y1)}`;
}

/**
 * The path of the gap between the model's angle and the driver's along a
 * circle of radius `r`: clockwise from the smaller angle to the larger,
 * whichever of the two it is. Empty when they agree or either is missing; a
 * gap of a whole turn or more is drawn just short of a full circle (the
 * readout carries the true number).
 */
export function errorArcPath(modelDeg: number, humanDeg: number, r: number, cx: number, cy: number): string {
  if (!Number.isFinite(modelDeg) || !Number.isFinite(humanDeg) || modelDeg === humanDeg) {
    return "";
  }
  const from = Math.min(modelDeg, humanDeg);
  const span = Math.min(Math.abs(modelDeg - humanDeg), MAX_ARC_DEG);
  return arc(from, from + span, r, cx, cy);
}

/** Turn the drawing made by `wheelSvg` to the model's and the driver's angles.
 * With no model angle (its output for this frame is not ready yet) the model's
 * wheel stays where it was and no gap is drawn. */
export function renderWheel(svg: SVGSVGElement, modelDeg: number | null, humanDeg: number): void {
  const model = svg.querySelector<SVGGElement>('[data-wheel="model"]');
  const human = svg.querySelector<SVGGElement>('[data-wheel="human"]');
  const gap = svg.querySelector<SVGPathElement>('[data-wheel="error"]');
  if (model && modelDeg !== null) model.style.transform = `rotate(${wheelRotationDeg(modelDeg)}deg)`;
  if (human) human.style.transform = `rotate(${wheelRotationDeg(humanDeg)}deg)`;
  gap?.setAttribute(
    "d",
    modelDeg === null ? "" : errorArcPath(modelDeg, humanDeg, TRACK_R, WHEEL_CENTRE, WHEEL_CENTRE),
  );
}

/** The wheel, unturned. Decorative to assistive technology: the readouts
 * beside it carry the same numbers as text. */
export function wheelSvg(): string {
  const c = WHEEL_CENTRE;
  const spokes = `M${c - 82} ${c + 4}H${c - 27}M${c + 27} ${c + 4}H${c + 82}M${c} ${c + 27}V${c + 82}`;
  const pointer = (tip: number, base: number, half: number) =>
    `M${c} ${c - tip}L${c - half} ${c - base}H${c + half}Z`;
  return (
    `<svg class="wheel" viewBox="0 0 ${2 * c} ${2 * c}" aria-hidden="true" focusable="false">` +
    `<circle class="wheel__track" cx="${c}" cy="${c}" r="${TRACK_R}"/>` +
    `<path class="wheel__gap" data-wheel="error" d=""/>` +
    `<g class="wheel__turn wheel__ghost" data-wheel="human">` +
    `<path class="wheel__ghost-spokes" d="${spokes}"/>` +
    `<path class="wheel__ghost-pointer" d="${pointer(TRACK_R - 9, TRACK_R + 8, 8.5)}"/>` +
    `</g>` +
    `<g class="wheel__turn wheel__model" data-wheel="model">` +
    `<circle class="wheel__rim" cx="${c}" cy="${c}" r="${RIM_R}"/>` +
    `<path class="wheel__stripe" d="${arc(-STRIPE_HALF_DEG, STRIPE_HALF_DEG, RIM_R, c, c)}"/>` +
    `<path class="wheel__spokes" d="${spokes}"/>` +
    `<circle class="wheel__hub" cx="${c}" cy="${c}" r="26"/>` +
    `<path class="wheel__pointer" d="${pointer(TRACK_R - 7, TRACK_R + 6, 6.5)}"/>` +
    `</g>` +
    `</svg>`
  );
}
