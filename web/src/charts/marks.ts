/**
 * Small pieces of geometry the cards' drawings share: how wide a label will
 * be, a horizontal bar with a rounded data end, a point and an arc on a
 * steering wheel, and where two labels go so they never overlap. Pure, so the
 * drawings' rules can be tested without a browser laying anything out.
 */

/** About how wide `text` sets at `size` px in the page's sans: a generous
 * estimate, so a label is given room rather than clipped. */
export function textWidth(text: string, size = 12.5): number {
  return Math.ceil(text.length * size * 0.56);
}

const r2 = (n: number) => Math.round(n * 100) / 100;

/** A horizontal bar from `x0` (its base, square) to `x1` (its data end,
 * rounded by up to 4 px), `y` to `y + h`. A bar of no length is a hairline at
 * its base. */
export function hbarPath(x0: number, x1: number, y: number, h: number): string {
  const length = x1 - x0;
  if (Math.abs(length) < 0.01) return `M${r2(x0)} ${r2(y)}v${r2(h)}`;
  const r = Math.min(4, Math.abs(length), h / 2);
  const s = Math.sign(length);
  const tip = x1 - s * r;
  return (
    `M${r2(x0)} ${r2(y)}H${r2(tip)}Q${r2(x1)} ${r2(y)} ${r2(x1)} ${r2(y + r)}` +
    `V${r2(y + h - r)}Q${r2(x1)} ${r2(y + h)} ${r2(tip)} ${r2(y + h)}H${r2(x0)}Z`
  );
}

/** The point at `angle` degrees round a wheel centred on (`cx`, `cy`), `r`
 * from its centre: 0 at the top, turning clockwise as the wheel turns right. */
export function wheelPoint(cx: number, cy: number, r: number, angle: number): [number, number] {
  const a = (angle * Math.PI) / 180;
  return [r2(cx + r * Math.sin(a)), r2(cy - r * Math.cos(a))];
}

/** The arc from `from` to `to` degrees clockwise round the wheel, as a path.
 * A whole turn or more is drawn as two half arcs, so it never vanishes. */
export function arcPath(cx: number, cy: number, r: number, from: number, to: number): string {
  const sweep = to - from;
  if (sweep <= 0) {
    const [x, y] = wheelPoint(cx, cy, r, from);
    return `M${x} ${y}`;
  }
  if (sweep >= 360) {
    const [x0, y0] = wheelPoint(cx, cy, r, from);
    const [x1, y1] = wheelPoint(cx, cy, r, from + 180);
    return `M${x0} ${y0}A${r} ${r} 0 0 1 ${x1} ${y1}A${r} ${r} 0 0 1 ${x0} ${y0}`;
  }
  const [x0, y0] = wheelPoint(cx, cy, r, from);
  const [x1, y1] = wheelPoint(cx, cy, r, to);
  return `M${x0} ${y0}A${r} ${r} 0 ${sweep > 180 ? 1 : 0} 1 ${x1} ${y1}`;
}

/** A label's place: its anchor point, which end of the text sits there, and
 * which line it is on: 0 the dots' own line, -1 the line above, 1 below. */
export interface LabelPlace {
  readonly x: number;
  readonly anchor: "start" | "end";
  readonly line: -1 | 0 | 1;
}

/**
 * Where the two labels of a dumbbell go: the lower value's label to the left
 * of its dot and the higher's to the right, on the dots' line, wherever that
 * fits between `left` and `right`; a label with no room on its side sits on
 * the line above, reaching inwards from its dot, and if two would meet there,
 * the higher one goes below. The two never overlap. `low` and `high` are the
 * dots' positions, `lowWidth` and `highWidth` their labels' widths, `gap` the
 * space between a dot's centre and its label.
 */
export function dumbbellLabels(
  low: number,
  high: number,
  lowWidth: number,
  highWidth: number,
  left: number,
  right: number,
  gap = 11,
): { readonly low: LabelPlace; readonly high: LabelPlace } {
  const lowSide = low - gap - lowWidth >= left;
  const highSide = high + gap + highWidth <= right;
  const lowPlace: LabelPlace = lowSide ? { x: low - gap, anchor: "end", line: 0 } : { x: Math.max(left, low - 6), anchor: "start", line: -1 };
  let highPlace: LabelPlace = highSide ? { x: high + gap, anchor: "start", line: 0 } : { x: Math.min(right, high + 6), anchor: "end", line: -1 };
  if (!lowSide && !highSide && lowPlace.x + lowWidth + 8 > highPlace.x - highWidth) highPlace = { ...highPlace, line: 1 };
  return { low: lowPlace, high: highPlace };
}

/** The span a label covers, from its place and width. */
export function labelSpan(place: LabelPlace, width: number): [number, number] {
  return place.anchor === "start" ? [place.x, place.x + width] : [place.x - width, place.x];
}
