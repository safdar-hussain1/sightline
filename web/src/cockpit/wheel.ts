/**
 * The steering wheel: the model's wheel, the driver's, and the gap between
 * them.
 *
 * Angles are steering-wheel degrees, clockwise positive, as the labels and the
 * model use them. Each wheel turns by its angle itself, never wrapped: a label
 * of 501.8° is a wheel turned past one and a quarter turns, and the drawing
 * says so.
 *
 * The model's wheel is solid, in its violet. The driver's is drawn over it as
 * a pale wireframe -- the centre lines of its spokes, reaching just past the
 * rim, and a tick across the rim at its top -- so where the two agree the pale
 * lines run down the middle of the violet spokes, and where they differ a
 * second wheel shows. Outside the rim a track carries the comparison at a
 * glance: the model's pointer reaches up to it from the rim, the driver's pale
 * pointer down to it from outside, and the arc along it between the two
 * pointers is the gap. The pointers stop short of the track on either side,
 * so the arc is never hidden under one of them, even when the two are only a
 * few degrees apart -- the size of most gaps.
 */

/** The drawing's centre. */
export const WHEEL_CENTRE = 120;
/** How far from the centre the drawing reaches: its viewBox is the square
 * this far round the centre. */
export const WHEEL_EXTENT = 126;

/** The rim: the radius of its centre line, and its width. */
export const RIM_R = 80;
export const RIM_WIDTH = 15;
const HUB_R = 24;

/** The circle the gap is drawn along, and the gap's width: wide enough to
 * read when the two wheels are a few degrees apart. */
export const TRACK_R = 106;
export const GAP_WIDTH = 9;

/** A pointer at twelve o'clock: a triangle from its `base` to its `tip`
 * (radii), `half` wide either side of the centre line at its base. */
export interface Pointer {
  readonly base: number;
  readonly tip: number;
  readonly half: number;
}

/** The model's pointer stands on its rim and reaches up towards the track. */
export const MODEL_POINTER: Pointer = { base: RIM_R + RIM_WIDTH / 2 + 1, tip: TRACK_R - GAP_WIDTH / 2 - 1, half: 7 };
/** The driver's pointer hangs outside the track and reaches down towards it. */
export const DRIVER_POINTER: Pointer = { base: WHEEL_EXTENT - 2, tip: TRACK_R + GAP_WIDTH / 2 + 1, half: 7.5 };

/** Half the angular width of the stripe at the top of the model's rim, in degrees. */
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

/** A pointer's triangle at twelve o'clock. */
function pointerPath({ base, tip, half }: Pointer): string {
  const c = WHEEL_CENTRE;
  return `M${c} ${c - tip}L${c - half} ${c - base}H${c + half}Z`;
}

/** Three spokes, left, right and down, between radii `inner` and `outer`; the
 * side spokes sit a little below the centre line, as a steering wheel's do. */
function spokesPath(inner: number, outer: number): string {
  const c = WHEEL_CENTRE;
  return `M${c - outer} ${c + 4}H${c - inner}M${c + inner} ${c + 4}H${c + outer}M${c} ${c + inner}V${c + outer}`;
}

/** The wheel, unturned. Decorative to assistive technology: the readouts
 * beside it carry the same numbers as text. Every length that decides whether
 * two marks overlap is written here, not in the stylesheet. */
export function wheelSvg(): string {
  const c = WHEEL_CENTRE;
  const rimOuter = RIM_R + RIM_WIDTH / 2;
  const box = `${c - WHEEL_EXTENT} ${c - WHEEL_EXTENT} ${2 * WHEEL_EXTENT} ${2 * WHEEL_EXTENT}`;
  const ghost = `${spokesPath(HUB_R + 4, rimOuter + 4)}M${c} ${c - (RIM_R - RIM_WIDTH / 2)}V${c - rimOuter}`;
  return (
    `<svg class="wheel" viewBox="${box}" aria-hidden="true" focusable="false">` +
    `<circle class="wheel__track" cx="${c}" cy="${c}" r="${TRACK_R}"/>` +
    `<path class="wheel__gap" data-wheel="error" d="" stroke-width="${GAP_WIDTH}"/>` +
    `<g class="wheel__turn wheel__model" data-wheel="model">` +
    `<circle class="wheel__rim" cx="${c}" cy="${c}" r="${RIM_R}" stroke-width="${RIM_WIDTH}"/>` +
    `<path class="wheel__stripe" d="${arc(-STRIPE_HALF_DEG, STRIPE_HALF_DEG, RIM_R, c, c)}" stroke-width="${RIM_WIDTH}"/>` +
    `<path class="wheel__spokes" d="${spokesPath(HUB_R + 2, RIM_R - 2)}"/>` +
    `<circle class="wheel__hub" cx="${c}" cy="${c}" r="${HUB_R}"/>` +
    `<path class="wheel__pointer" d="${pointerPath(MODEL_POINTER)}"/>` +
    `</g>` +
    `<g class="wheel__turn wheel__ghost" data-wheel="human">` +
    `<path class="wheel__ghost-lines" d="${ghost}"/>` +
    `<path class="wheel__ghost-pointer" d="${pointerPath(DRIVER_POINTER)}"/>` +
    `</g>` +
    `</svg>`
  );
}
