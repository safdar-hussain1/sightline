/**
 * The cards' drawings, one idea each, every mark labelled where it stands:
 *
 * - what the extra frames buy: a dumbbell per set of frames, from the network
 *   with fewer frames to the one with more, on how much better than always
 *   steering straight each is;
 * - where it struggles: each regime beside a small drawing of its road, the
 *   shipped model's error and steering straight's;
 * - the standard recipe: a steering wheel turned to its largest answer, on a
 *   gauge that ends where its arctangent output must, with the turns the
 *   driver made past that end; and its errors beside steering straight's;
 * - the baselines: rulers in degrees, with the rules that never look at the
 *   road, the rules that read the answers, and Sightline between them.
 *
 * Each builds its marks into a group at the card's width and returns its
 * height and `paint(t)`, which shows it at `t` of its entrance (1: final).
 * Text is in the page's inks; a mark beside it carries the colour.
 */

import { attrs, local, svg } from "../charts/draw.ts";
import { arcPath, dumbbellLabels, hbarPath, labelSpan, textWidth, wheelPoint } from "../charts/marks.ts";
import { linear, niceAxis } from "../charts/scale.ts";
import { fmtCount, fmtDeg, fmtPct } from "../format.ts";
import { exactDeg, regimeNames, YARDSTICK_LABELS, yardstickSetName } from "./copy.ts";
import { textAt } from "./countup.ts";
import { REGIMES, type Legacy, type LegacyScore, type Regime, type Regimes, type YardstickSet } from "./figures.ts";

export interface CardDrawing {
  readonly height: number;
  readonly paint: (t: number) => void;
}

const PAD = 4;
const LABEL = 12.5;
const NOTE = 12;

/** A label beside a mark: its name, quiet, then its figure in the page's ink. */
function label(parent: SVGElement, x: number, y: number, anchor: "start" | "middle" | "end", name: string, figure: string) {
  const text = svg("text", { class: "card-chart__label", x, y, "text-anchor": anchor }, parent);
  if (name) svg("tspan", { class: "card-chart__name" }, text).textContent = `${name} `;
  const value = svg("tspan", { class: "card-chart__figure" }, text);
  value.textContent = figure;
  return { text, value };
}

/** A row's title at baseline `y`, with a quieter `extra` after it, or on a
 * line of its own when the two do not fit in `room`. Returns the height used. */
function title(parent: SVGElement, x: number, y: number, words: string, extra = "", room = Infinity): number {
  const text = svg("text", { class: "card-chart__title", x, y }, parent);
  text.textContent = words;
  if (!extra) return 14;
  if (textWidth(`${words} · ${extra}`) <= room) {
    svg("tspan", { class: "card-chart__extra" }, text).textContent = ` · ${extra}`;
    return 14;
  }
  svg("tspan", { class: "card-chart__extra", x, dy: 15 }, text).textContent = extra;
  return 29;
}

// -- What the extra frames buy -------------------------------------------------

export interface FramesRow {
  readonly title: string;
  /** The skills, 1 - error / steering straight's, with more frames and with fewer. */
  readonly more: number;
  readonly fewer: number;
}

/** One dumbbell per set of frames, from fewer frames to more, on one scale of
 * how much better than always steering straight. */
export function framesDrawing(group: SVGGElement, width: number, rows: readonly FramesRow[], words: { readonly more: string; readonly fewer: string }): CardDrawing {
  const values = rows.flatMap((row) => [row.more, row.fewer]);
  const axis = niceAxis(Math.min(...values), Math.max(...values), 5);
  const texts = rows.map((row) => ({ more: `${words.more} ${fmtPct(row.more)}`, fewer: `${words.fewer} ${fmtPct(row.fewer)}` }));
  const room = Math.max(...texts.map((t) => textWidth(t.more))) + 14;
  const x = linear(axis.extent, [PAD + 8, Math.max(PAD + 60, width - room)]);
  const zero = x(0);

  // The line every skill is measured from.
  const top = 14;
  const note = svg("text", { class: "card-chart__note", x: zero, y: top - 2, "text-anchor": zero > width / 2 ? "end" : "start" }, group);
  note.textContent = "always steering straight";
  const zeroLine = svg("line", { class: "card-chart__zero", x1: zero, x2: zero, y1: top + 6 }, group);

  const moving: Array<(t: number) => void> = [];
  let y = top + 14;
  // Titles stand clear of the zero line when it runs down the left.
  const titleX = zero < width / 3 ? zero + 8 : PAD;
  rows.forEach((row, i) => {
    title(group, titleX, y + 12, row.title);
    const lineY = y + 46;
    const low = Math.min(row.more, row.fewer);
    const lowIsFewer = row.fewer <= row.more;
    const lowText = (lowIsFewer ? texts[i]?.fewer : texts[i]?.more) ?? "";
    const highText = (lowIsFewer ? texts[i]?.more : texts[i]?.fewer) ?? "";
    const places = dumbbellLabels(x(low), x(Math.max(row.more, row.fewer)), textWidth(lowText), textWidth(highText), titleX, width - PAD);
    const fewerPlace = lowIsFewer ? places.low : places.high;
    const morePlace = lowIsFewer ? places.high : places.low;
    const lineOf = (line: number) => lineY + 4 + line * 19;

    const link = svg("line", { class: "card-chart__link", x1: x(row.fewer), y1: lineY, y2: lineY }, group);
    svg("circle", { class: "card-chart__dot series--oneframe", cx: x(row.fewer), cy: lineY, r: 6 }, group);
    const dot = svg("circle", { class: "card-chart__dot series--sightlinenet", cy: lineY, r: 6.5 }, group);
    label(group, fewerPlace.x, lineOf(fewerPlace.line), fewerPlace.anchor, words.fewer, fmtPct(row.fewer));
    const moreLabel = label(group, morePlace.x, lineOf(morePlace.line), morePlace.anchor, words.more, fmtPct(row.more));
    moving.push((t) => {
      const at = x(row.fewer) + (x(row.more) - x(row.fewer)) * t;
      attrs(link, { x2: at });
      attrs(dot, { cx: at });
      attrs(moreLabel.text, { opacity: local(t, 0.5, 0.5) });
      moreLabel.value.textContent = textAt(fmtPct(row.more), row.more * 100, row.fewer * 100, t);
    });
    const below = fewerPlace.line === 1 || morePlace.line === 1;
    y = lineY + (below ? 40 : 22);
  });

  // The scale, under the rows.
  const axisY = y + 4;
  attrs(zeroLine, { y2: axisY - 10 });
  for (const tick of axis.ticks) {
    const text = svg("text", { class: "card-chart__tick", x: x(tick), y: axisY, "text-anchor": "middle" }, group);
    text.textContent = fmtPct(tick);
  }
  return { height: axisY + 6, paint: (t) => moving.forEach((move) => move(t)) };
}

// -- Where it struggles -------------------------------------------------------------

const ICON = 36;

/** A small drawing of each regime's road, as a road sign draws a bend: a band
 * going straight on, bending gently, or turning hard, with its centre line.
 * It names the regime; it measures nothing. */
export const ROADS: Readonly<Record<Regime, string>> = {
  straight: `M${ICON / 2} ${ICON - 4}V4`,
  gentle: `M${ICON / 2 - 4} ${ICON - 4}C${ICON / 2 - 4} ${ICON / 2 + 3} ${ICON / 2 - 1} ${ICON / 2 - 5} ${ICON - 9} 6`,
  sharp: `M${ICON / 2 - 7} ${ICON - 4}V${ICON / 2 + 2}Q${ICON / 2 - 7} ${ICON / 2 - 6} ${ICON / 2 + 1} ${ICON / 2 - 6}H${ICON - 4}`,
};

/** Each regime's road beside its two bars: the shipped model's error and
 * steering straight's, on one scale. */
export function regimesDrawing(group: SVGGElement, width: number, regimes: Regimes): CardDrawing {
  const names = regimeNames(regimes.edges);
  const model = regimes.models.sightlinenet.errors;
  const zero = regimes.straight;
  const rows = REGIMES.map((regime) => ({
    regime,
    bars: [
      { series: "sightlinenet", name: "Sightline", value: model[regime] },
      { series: "straight", name: "steering straight", value: zero[regime] },
    ],
  }));
  const x0 = ICON + 12;
  const longest = Math.max(...rows.flatMap((row) => row.bars.map((bar) => textWidth(`${bar.name} ${fmtDeg(bar.value)}`))));
  const axis = niceAxis(0, Math.max(...rows.flatMap((row) => row.bars.map((bar) => bar.value))), 4);
  const x = linear(axis.extent, [x0, Math.max(x0 + 40, width - longest - 10)]);
  const grow: Array<(t: number) => void> = [];
  let y = 0;
  for (const row of rows) {
    const road = svg("g", { class: "card-chart__road", transform: `translate(0 ${y + 4})` }, group);
    svg("rect", { class: "card-chart__road-tile", width: ICON, height: ICON, rx: 9 }, road);
    svg("path", { class: "card-chart__road-band", d: ROADS[row.regime] }, road);
    svg("path", { class: "card-chart__road-line", d: ROADS[row.regime] }, road);
    const used = title(group, x0, y + 12, names[row.regime], `${fmtCount(regimes.frames[row.regime])} frames`, width - x0);
    row.bars.forEach((bar, i) => {
      const top = y + used + 6 + i * 16;
      const path = svg("path", { class: `card-chart__bar series--${bar.series}` }, group);
      const tag = label(group, x(bar.value) + 6, top + 9.5, "start", bar.name, fmtDeg(bar.value));
      grow.push((t) => {
        const end = x(0) + (x(bar.value) - x(0)) * t;
        attrs(path, { d: hbarPath(x(0), end, top, 11) });
        attrs(tag.text, { x: end + 6, opacity: t > 0 ? 1 : 0 });
        tag.value.textContent = textAt(fmtDeg(bar.value), bar.value, 0, t);
      });
    });
    y += used + 52;
  }
  return { height: y - 14, paint: (t) => grow.forEach((step) => step(t)) };
}

// -- The standard recipe ---------------------------------------------------------

/** The wheel turned to the standard recipe's largest answer, on a gauge of
 * how far the wheel turns either way: the reach of its arctangent output up
 * to the limit, the part it used, and the rest of the turn, where the driver
 * went and it cannot. Drawn from `top`, `width` wide; returns its height. */
function reachDrawing(group: SVGGElement, width: number, legacy: Legacy): CardDrawing {
  const side = Math.max(...["largest answer", "turned past it", `${fmtCount(legacy.beyond)} frames`].map((t) => textWidth(t)));
  const r = Math.max(40, Math.min(80, (width - 2 * side - 40) / 2));
  const cx = width / 2;
  const cy = 20 + r;
  const turn = (angle: number) => Math.min(angle, 359.99);
  svg("path", { class: "card-chart__reach-track", d: arcPath(cx, cy, r, 0, legacy.cap) }, group);
  svg("path", { class: "card-chart__beyond", d: arcPath(cx, cy, r, legacy.cap, turn(360)) }, group);
  const used = svg("path", { class: "card-chart__reach series-stroke--pilotnet" }, group);
  const [w1x, w1y] = wheelPoint(cx, cy, r - 9, legacy.cap);
  const [w2x, w2y] = wheelPoint(cx, cy, r + 9, legacy.cap);
  svg("line", { class: "card-chart__wall", x1: w1x, y1: w1y, x2: w2x, y2: w2y }, group);
  const [t0x, t0y] = wheelPoint(cx, cy, r + 8, 0);
  const zeroMark = svg("text", { class: "card-chart__tick", x: t0x, y: t0y - 2, "text-anchor": "middle" }, group);
  // Straight ahead, where every turn of the wheel is measured from.
  zeroMark.textContent = exactDeg(0);

  // The wheel itself, its marker at the top when it points straight ahead.
  const inner = r - 21;
  const wheel = svg("g", { class: "card-chart__wheel" }, group);
  svg("circle", { class: "card-chart__wheel-rim", cx, cy, r: inner }, wheel);
  for (const angle of [90, 180, 270]) {
    const [x1, y1] = wheelPoint(cx, cy, 6, angle);
    const [x2, y2] = wheelPoint(cx, cy, inner, angle);
    svg("line", { class: "card-chart__wheel-spoke", x1, y1, x2, y2 }, wheel);
  }
  svg("circle", { class: "card-chart__wheel-hub", cx, cy, r: 6 }, wheel);
  const [m1x, m1y] = wheelPoint(cx, cy, inner - 5, 0);
  const [m2x, m2y] = wheelPoint(cx, cy, inner + 5, 0);
  svg("line", { class: "card-chart__wheel-marker series-stroke--pilotnet", x1: m1x, y1: m1y, x2: m2x, y2: m2y }, wheel);

  // Its words, outside the gauge: the largest answer on the right, the limit
  // at the foot, the turns past it on the left.
  const [ax, ay] = wheelPoint(cx, cy, r + 14, 128);
  const answer = svg("text", { class: "card-chart__label", x: ax, y: ay, "text-anchor": "start" }, group);
  svg("tspan", { class: "card-chart__name", x: ax }, answer).textContent = "largest answer";
  const value = svg("tspan", { class: "card-chart__figure", x: ax, dy: 16 }, answer);
  const limit = label(group, cx, cy + r + 24, "middle", "limit", exactDeg(legacy.cap));
  void limit;
  const [bx, by] = wheelPoint(cx, cy, r + 14, 232);
  const beyond = svg("text", { class: "card-chart__label", x: bx, y: by, "text-anchor": "end" }, group);
  svg("tspan", { class: "card-chart__figure", x: bx }, beyond).textContent = `${fmtCount(legacy.beyond)} frames`;
  svg("tspan", { class: "card-chart__name", x: bx, dy: 16 }, beyond).textContent = "turned past it";

  return {
    height: cy + r + 32,
    paint: (t) => {
      const angle = legacy.largest * t;
      attrs(used, { d: arcPath(cx, cy, r, 0, angle) });
      attrs(wheel, { transform: `rotate(${Math.round(angle * 100) / 100} ${cx} ${cy})` });
      value.textContent = textAt(fmtDeg(legacy.largest), legacy.largest, 0, t);
    },
  };
}

/** Bars of an error beside steering straight's, a row per set of frames. */
function errorRows(group: SVGGElement, width: number, top: number, rows: ReadonlyArray<{ readonly title: string; readonly extra: string; readonly model: number; readonly straight: number }>, series: string, name: string): CardDrawing {
  const longest = Math.max(...rows.flatMap((row) => [textWidth(`${name} ${fmtDeg(row.model)}`), textWidth(`steering straight ${fmtDeg(row.straight)}`)]));
  const axis = niceAxis(0, Math.max(...rows.flatMap((row) => [row.model, row.straight])), 4);
  const x = linear(axis.extent, [PAD, Math.max(PAD + 40, width - longest - 10)]);
  const grow: Array<(t: number) => void> = [];
  let y = top;
  for (const row of rows) {
    const used = title(group, PAD, y + 12, row.title, row.extra, width - PAD);
    [
      { series, name, value: row.model },
      { series: "straight", name: "steering straight", value: row.straight },
    ].forEach((bar, i) => {
      const barTop = y + used + 6 + i * 16;
      const path = svg("path", { class: `card-chart__bar series--${bar.series}` }, group);
      const tag = label(group, x(bar.value) + 6, barTop + 9.5, "start", bar.name, fmtDeg(bar.value));
      grow.push((t) => {
        const end = x(0) + (x(bar.value) - x(0)) * t;
        attrs(path, { d: hbarPath(x(0), end, barTop, 11) });
        attrs(tag.text, { x: end + 6, opacity: t > 0 ? 1 : 0 });
        tag.value.textContent = textAt(fmtDeg(bar.value), bar.value, 0, t);
      });
    });
    y += used + 48;
  }
  return { height: y - 10 - top, paint: (t) => grow.forEach((step) => step(t)) };
}

/** The standard recipe's reach, then its errors beside steering straight's. */
export function recipeDrawing(group: SVGGElement, width: number, legacy: Legacy, sets: ReadonlyArray<readonly [title: string, score: LegacyScore]>): CardDrawing {
  const reach = reachDrawing(svg("g", {}, group), width, legacy);
  const rows = errorRows(
    svg("g", {}, group),
    width,
    reach.height + 18,
    sets.map(([words, score]) => ({ title: words, extra: `${fmtCount(score.frames)} frames`, model: score.model, straight: score.straight })),
    "pilotnet",
    "PilotNet",
  );
  return {
    height: reach.height + 18 + rows.height,
    paint: (t) => {
      reach.paint(t);
      rows.paint(t);
    },
  };
}

// -- The baselines -----------------------------------------------------------------

/** A ruler in degrees per set of frames: the rules that never look at the
 * road and the rules that read the answers at their errors, and Sightline's. */
export function yardsticksDrawing(group: SVGGElement, width: number, sets: readonly YardstickSet[]): CardDrawing {
  const values = sets.flatMap((set) => [set.model, ...Object.values(set.errors)]);
  const axis = niceAxis(0, Math.max(...values), 4);
  const x = linear(axis.extent, [PAD + 6, width - PAD - 6]);
  const enter: Array<(t: number) => void> = [];
  let y = 0;
  for (const set of sets) {
    const e = set.errors;
    const used = title(group, PAD, y + 12, yardstickSetName(set), `${fmtCount(set.frames)} frames`, width - PAD);
    const lineY = y + used + 38;
    svg("line", { class: "card-chart__ruler", x1: x(axis.extent[0]), x2: x(axis.extent[1]), y1: lineY, y2: lineY }, group);
    for (const tick of axis.ticks) {
      svg("line", { class: "card-chart__ruler-tick", x1: x(tick), x2: x(tick), y1: lineY, y2: lineY + 5 }, group);
      const text = svg("text", { class: "card-chart__tick", x: x(tick), y: lineY + 18, "text-anchor": tick === axis.extent[0] ? "start" : tick === axis.extent[1] ? "end" : "middle" }, group);
      text.textContent = `${fmtCount(tick)}°`;
    }
    // The marks, the quiet ones first so Sightline's sits on top.
    const marks = svg("g", { class: "card-chart__marks" }, group);
    const diamond = (cx: number) => `M${cx} ${lineY - 6.5}L${cx + 6.5} ${lineY}L${cx} ${lineY + 6.5}L${cx - 6.5} ${lineY}Z`;
    for (const name of ["zoh", "extrap"] as const) svg("path", { class: "card-chart__mark card-chart__mark--answers", d: diamond(x(e[name])) }, marks);
    svg("rect", { class: "card-chart__mark series--straight", x: x(e.mean) - 5.5, y: lineY - 5.5, width: 11, height: 11, rx: 2 }, marks);
    svg("circle", { class: "card-chart__mark series--straight", cx: x(e.zero), cy: lineY, r: 6 }, marks);
    const model = svg("circle", { class: "card-chart__mark series--sightlinenet", cx: x(set.model), cy: lineY, r: 7.5 }, marks);
    const modelLabel = label(group, Math.min(Math.max(x(set.model), PAD + 50), width - PAD - 50), lineY - 13, "middle", "Sightline", fmtDeg(set.model));

    // The two groups of rules, each named, under their marks; one under the
    // other where the ruler is too narrow to hold them side by side.
    const groups = [
      { heading: "reading the answers", x: x(0) - 4, anchor: "start" as const, lines: [[YARDSTICK_LABELS.zoh, e.zoh], [YARDSTICK_LABELS.extrap, e.extrap]] as const },
      { heading: "never looking at the road", x: Math.max(x(e.zero), x(e.mean)) + 6, anchor: "end" as const, lines: [[YARDSTICK_LABELS.zero, e.zero], [YARDSTICK_LABELS.mean, e.mean]] as const },
    ];
    const widths = groups.map((g) => Math.max(textWidth(g.heading, NOTE), ...g.lines.map(([n, v]) => textWidth(`${n} ${fmtDeg(v)}`, LABEL))));
    const [leftGroup, rightGroup] = groups;
    const spans = groups.map((g, i) => labelSpan({ x: Math.max(PAD, Math.min(width - PAD, g.x)), anchor: g.anchor, line: 0 }, widths[i] ?? 0));
    const stacked = (spans[0]?.[1] ?? 0) + 12 > (spans[1]?.[0] ?? 0);
    let groupY = lineY + 40;
    for (const [i, g] of [leftGroup, rightGroup].entries()) {
      if (!g) continue;
      const gx = Math.max(PAD, Math.min(width - PAD, g.x));
      const rowY = stacked ? groupY + i * 3 * 16 : groupY;
      const heading = svg("text", { class: "card-chart__note", x: gx, y: rowY, "text-anchor": g.anchor }, group);
      heading.textContent = g.heading;
      g.lines.forEach(([n, v], k) => label(group, gx, rowY + 16 * (k + 1), g.anchor, n, fmtDeg(v)));
    }
    groupY += stacked ? 6 * 16 : 3 * 16;
    enter.push((t) => {
      attrs(marks, { opacity: local(t, 0, 0.5) });
      attrs(model, { cx: x(set.model * t) });
      modelLabel.value.textContent = textAt(fmtDeg(set.model), set.model, 0, t);
      attrs(modelLabel.text, { opacity: local(t, 0.4, 0.6) });
    });
    y = groupY + 10;
  }
  return { height: y - 6, paint: (t) => enter.forEach((step) => step(t)) };
}
