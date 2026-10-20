/**
 * The cards' small charts, as geometry: rows of horizontal bars on one scale,
 * each row titled, each bar labelled at its tip. Bars stand on a zero line --
 * zero degrees of error, or always steering straight for a skill -- and a
 * negative value runs left of it, labelled at its left. Pure, like the other
 * charts' layouts.
 */

import { linear, niceAxis } from "./scale.ts";

/** One bar: its series (a colour), its value, its label. */
export interface BarSpec {
  readonly series: string;
  readonly value: number;
  readonly text: string;
}

export interface RowSpec {
  readonly title: string;
  readonly bars: readonly BarSpec[];
}

export interface HBar extends BarSpec {
  readonly row: number;
  readonly x0: number;
  readonly x1: number;
  readonly y: number;
  readonly h: number;
  readonly labelX: number;
  readonly anchor: "start" | "end";
}

export interface BarsLayout {
  readonly width: number;
  readonly height: number;
  /** The zero line, behind each row's bars (never through a title). */
  readonly zero: ReadonlyArray<{ readonly x: number; readonly y1: number; readonly y2: number }>;
  readonly titles: ReadonlyArray<{ readonly x: number; readonly y: number; readonly text: string }>;
  readonly bars: readonly HBar[];
}

const BAR_H = 12;
const BAR_GAP = 5;
const TITLE_H = 20;
const ROW_GAP = 14;

/** `rows` laid out `width` pixels wide, leaving `room` pixels beyond the
 * longest bar for its label. */
export function barsLayout(rows: readonly RowSpec[], width: number, room = 96): BarsLayout {
  const values = rows.flatMap((row) => row.bars.map((bar) => bar.value));
  const axis = niceAxis(Math.min(...values), Math.max(...values), 4);
  const negative = axis.extent[0] < 0;
  const left = negative ? room * 0.75 : 2;
  const x = linear(axis.extent, [left, width - room]);
  const bars: HBar[] = [];
  const titles: Array<BarsLayout["titles"][number]> = [];
  const zero: Array<BarsLayout["zero"][number]> = [];
  let y = 0;
  rows.forEach((row, r) => {
    titles.push({ x: 0, y: y + 13, text: row.title });
    y += TITLE_H;
    zero.push({ x: x(0), y1: y - 3, y2: y + row.bars.length * (BAR_H + BAR_GAP) - BAR_GAP + 3 });
    row.bars.forEach((bar) => {
      const end = x(bar.value);
      const start = x(0);
      const down = bar.value < 0;
      bars.push({ ...bar, row: r, x0: start, x1: end, y, h: BAR_H, labelX: down ? end - 6 : end + 6, anchor: down ? "end" : "start" });
      y += BAR_H + BAR_GAP;
    });
    y += ROW_GAP - BAR_GAP;
  });
  const height = y - ROW_GAP + 6;
  return { width, height, zero, titles, bars };
}
