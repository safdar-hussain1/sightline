/**
 * The paired experiment's slope chart, as geometry: for each training seed a
 * line from the network's error trained on shuffled frames (left) to its error
 * trained with stretches held back (right), both on that seed's paired frames;
 * each seed's always-steering-straight error on the same frames as a grey
 * line behind them; the means of both ends and the mean ratio as labels.
 *
 * Pure: a width and the data in, positions out, so the drawing, its draw-in
 * and its morph between the two networks share one source and are tested
 * without a browser. Both networks are laid out on one scale, so switching
 * between them moves only what differs.
 */

import { fmtDeg, fmtRatio } from "../format.ts";
import type { PairedNetwork } from "../sections/figures.ts";
import { linear, niceAxis } from "./scale.ts";

export interface SlopeLine {
  readonly seed: number;
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
}

/** A label whose number can count: its value, its final text, where it sits. */
export interface ValueLabel {
  readonly x: number;
  readonly y: number;
  readonly value: number;
  readonly text: string;
}

export interface SlopeLayout {
  readonly width: number;
  readonly height: number;
  readonly narrow: boolean;
  readonly plot: { readonly left: number; readonly right: number; readonly top: number; readonly bottom: number };
  readonly x: { readonly shuffled: number; readonly blocked: number };
  readonly ticks: ReadonlyArray<{ readonly y: number; readonly text: string }>;
  /** Steering straight on each seed's frames. */
  readonly straight: ReadonlyArray<{ readonly seed: number; readonly y: number }>;
  readonly straightLabel: { readonly x: number; readonly y: number; readonly text: string };
  readonly lines: readonly SlopeLine[];
  readonly means: { readonly shuffled: ValueLabel; readonly blocked: ValueLabel };
  readonly ratio: ValueLabel;
}

/** The decimals an axis's ticks need to be printed as they are: none for
 * steps of 5, one for steps of 2.5. */
export function tickDecimals(ticks: readonly number[]): number {
  return Math.max(0, ...ticks.map((value) => (String(value).split(".")[1] ?? "").length));
}

/** Below this width the chart tightens its margins and type. */
export const SLOPE_NARROW = 480;

/** The chart for `network`, `width` pixels wide, on the scale every network
 * in `all` shares. */
export function slopeLayout(network: PairedNetwork, all: readonly PairedNetwork[], width: number): SlopeLayout {
  const narrow = width < SLOPE_NARROW;
  const height = narrow ? 300 : 340;
  const gutter = narrow ? 30 : 38; // the tick labels
  const room = narrow ? 50 : 76; // each side's mean label
  const plot = { left: gutter, right: width - 4, top: 30, bottom: height - 46 };
  const x = { shuffled: plot.left + room, blocked: plot.right - room };
  const highest = Math.max(...all.flatMap((each) => each.seeds.flatMap((seed) => [seed.blocked, seed.shuffled, seed.straight])));
  const axis = niceAxis(0, highest, 4);
  const y = linear(axis.extent, [plot.bottom, plot.top]);
  const straights = network.seeds.map((seed) => seed.straight);
  const low = Math.min(...straights);
  const high = Math.max(...straights);
  const range = fmtDeg(low) === fmtDeg(high) ? fmtDeg(low) : `${fmtDeg(low)} to ${fmtDeg(high)}`;
  return {
    width,
    height,
    narrow,
    plot,
    x,
    ticks: axis.ticks.map((value) => ({ y: y(value), text: fmtDeg(value, tickDecimals(axis.ticks)) })),
    straight: network.seeds.map((seed) => ({ seed: seed.seed, y: y(seed.straight) })),
    straightLabel: { x: plot.right, y: y(high) - 8, text: `Always steering straight: ${range}` },
    lines: network.seeds.map((seed) => ({ seed: seed.seed, x1: x.shuffled, y1: y(seed.shuffled), x2: x.blocked, y2: y(seed.blocked) })),
    means: {
      shuffled: { x: x.shuffled - 16, y: y(network.shuffled), value: network.shuffled, text: fmtDeg(network.shuffled) },
      blocked: { x: x.blocked + 16, y: y(network.blocked), value: network.blocked, text: fmtDeg(network.blocked) },
    },
    ratio: { x: (x.shuffled + x.blocked) / 2, y: plot.bottom - 14, value: network.ratio.mean, text: fmtRatio(network.ratio.mean) },
  };
}

const mix = (a: number, b: number, t: number) => a + (b - a) * t;

/** `from` turned `t` (0 to 1) of the way into `to`, for the morph between the
 * networks: both laid out at the same width, with as many seeds. Values move
 * with their positions; texts are `to`'s, for a count to end on. */
export function lerpSlope(from: SlopeLayout, to: SlopeLayout, t: number): SlopeLayout {
  const label = (a: ValueLabel, b: ValueLabel): ValueLabel => ({ x: mix(a.x, b.x, t), y: mix(a.y, b.y, t), value: mix(a.value, b.value, t), text: b.text });
  return {
    ...to,
    straight: to.straight.map((line, i) => ({ ...line, y: mix(from.straight[i]?.y ?? line.y, line.y, t) })),
    straightLabel: { ...to.straightLabel, y: mix(from.straightLabel.y, to.straightLabel.y, t) },
    lines: to.lines.map((line, i) => {
      const was = from.lines[i] ?? line;
      return { ...line, y1: mix(was.y1, line.y1, t), y2: mix(was.y2, line.y2, t) };
    }),
    means: { shuffled: label(from.means.shuffled, to.means.shuffled), blocked: label(from.means.blocked, to.means.blocked) },
    ratio: label(from.ratio, to.ratio),
  };
}

/** The seed line nearest the point (`px`, `py`), and how far it is. */
export function nearestLine(layout: SlopeLayout, px: number, py: number): { readonly index: number; readonly distance: number } | null {
  let best: { index: number; distance: number } | null = null;
  layout.lines.forEach((line, index) => {
    const dx = line.x2 - line.x1;
    const dy = line.y2 - line.y1;
    const along = dx === 0 && dy === 0 ? 0 : Math.max(0, Math.min(1, ((px - line.x1) * dx + (py - line.y1) * dy) / (dx * dx + dy * dy)));
    const distance = Math.hypot(px - (line.x1 + along * dx), py - (line.y1 + along * dy));
    if (best === null || distance < best.distance) best = { index, distance };
  });
  return best;
}
