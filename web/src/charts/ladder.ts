/**
 * The generalisation ladder, as geometry: one panel per rung, each with a bar
 * per network, in two views.
 *
 * "skill" -- how much better than always steering straight each network is
 * (1 - its error / steering straight's, on the same frames): bars stand on a
 * zero line that is steering straight itself, above it when the network beats
 * it and below it when it does worse, with one standard deviation over the
 * trainings as a whisker. Every rung shares the zero line, which is why this
 * is the view that compares rungs.
 *
 * "error" -- the mean error in degrees, each training's error as a dot, and
 * each rung's own steering-straight error as a grey tick. The ticks stand at
 * different heights: raw error is not comparable from rung to rung.
 *
 * A network never trained with a rung's split has no bar there, and the panel
 * says so in words. Wide, the panels stand in a row under the stepper; narrow,
 * two by two, each with its rung's name. Pure: positions out, nothing drawn.
 */

import { fmtDeg, fmtPct } from "../format.ts";
import { LADDER_MODELS, type Ladder, type LadderModel } from "../sections/figures.ts";
import { linear, niceAxis } from "./scale.ts";
import { tickDecimals } from "./slope.ts";

export type LadderView = "skill" | "error";

export interface Panel {
  readonly rung: number;
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
  /** The rung's name, drawn in the panel when the stepper is not above it. */
  readonly title: string | null;
}

export interface Bar {
  readonly model: LadderModel;
  readonly rung: number;
  readonly x: number;
  readonly w: number;
  /** Where the bar stands (the zero line, or zero degrees) and where it ends. */
  readonly base: number;
  readonly end: number;
  readonly value: number;
  readonly text: string;
  /** Its value label's baseline: above an upward bar, below a downward one. */
  readonly labelY: number;
}

export interface LadderLayout {
  readonly view: LadderView;
  readonly width: number;
  readonly height: number;
  readonly grid: boolean;
  readonly panels: readonly Panel[];
  /** The value grid, per row of panels. */
  readonly ticks: ReadonlyArray<{ readonly y: number; readonly x1: number; readonly x2: number; readonly text: string; readonly labelX: number }>;
  /** Always steering straight, in each panel. */
  readonly baselines: ReadonlyArray<{ readonly rung: number; readonly x1: number; readonly x2: number; readonly y: number; readonly text: string; readonly labelX: number }>;
  readonly bars: readonly Bar[];
  readonly missing: ReadonlyArray<{ readonly model: LadderModel; readonly rung: number; readonly x: number; readonly y: number }>;
  /** One standard deviation of the skill, either side of the mean. */
  readonly whiskers: ReadonlyArray<{ readonly model: LadderModel; readonly rung: number; readonly x: number; readonly y1: number; readonly y2: number }>;
  /** Each training's error. */
  readonly dots: ReadonlyArray<{ readonly model: LadderModel; readonly rung: number; readonly cx: number; readonly cy: number }>;
  readonly spotlight: { readonly x: number; readonly y: number; readonly w: number; readonly h: number };
}

/** Below this width the panels go two by two. */
export const LADDER_GRID = 560;

/** The ladder in `view`, `width` pixels wide, with the rung `selected` lit. */
export function ladderLayout(ladder: Ladder, view: LadderView, width: number, selected: number): LadderLayout {
  const grid = width < LADDER_GRID;
  const gutter = grid ? 34 : 42;
  const columns = grid ? 2 : ladder.rungs.length;
  const rows = Math.ceil(ladder.rungs.length / columns);
  const gap = grid ? 12 : 0;
  const titleRoom = grid ? 26 : 0;
  const panelH = grid ? 220 : 300;
  const panelW = (width - gutter - gap * (columns - 1)) / columns;
  const height = rows * panelH + (rows - 1) * 16;

  // One scale for every panel, so a bar's length means the same everywhere.
  const values: number[] = [];
  for (const model of LADDER_MODELS) {
    for (const bar of ladder.bars[model]) {
      if (!bar) continue;
      if (view === "skill") values.push(bar.skill.mean - bar.skill.sd, bar.skill.mean + bar.skill.sd);
      else values.push(bar.error.mean, ...bar.error.values);
    }
  }
  if (view === "error") values.push(...ladder.rungs.map((rung) => rung.straight));
  const axis = niceAxis(Math.min(...values), Math.max(...values), 4);
  const top = 18 + titleRoom;
  const bottom = panelH - 20;
  const decimals = tickDecimals(axis.ticks);
  const format = (value: number) => (view === "skill" ? fmtPct(value) : fmtDeg(value, decimals));

  const panels: Panel[] = ladder.rungs.map((rung, i) => {
    const column = i % columns;
    const row = Math.floor(i / columns);
    return { rung: i, x: gutter + column * (panelW + gap), y: row * (panelH + 16), w: panelW, h: panelH, title: grid ? rung.name : null };
  });
  const yAt = (panel: Panel) => linear(axis.extent, [panel.y + bottom, panel.y + top]);

  const slot = Math.min(26, (panelW - 24) / 3.6);
  const space = grid ? 5 : 7;
  const group = 3 * slot + 2 * space;

  const ticks = Array.from({ length: rows }, (_, row) => {
    const first = panels[row * columns] as Panel;
    const last = panels[Math.min(panels.length - 1, row * columns + columns - 1)] as Panel;
    const y = yAt(first);
    return axis.ticks.map((value) => ({ y: y(value), x1: first.x, x2: last.x + last.w, text: format(value), labelX: first.x - 6 }));
  }).flat();

  const bars: Bar[] = [];
  const missing: Array<LadderLayout["missing"][number]> = [];
  const whiskers: Array<LadderLayout["whiskers"][number]> = [];
  const dots: Array<LadderLayout["dots"][number]> = [];
  const baselines = panels.map((panel) => {
    const y = yAt(panel);
    const rung = ladder.rungs[panel.rung];
    const start = panel.x + (panel.w - group) / 2;
    return view === "skill"
      ? { rung: panel.rung, x1: panel.x, x2: panel.x + panel.w, y: y(0), text: "", labelX: panel.x + panel.w }
      : { rung: panel.rung, x1: start - 8, x2: start + group + 8, y: y(rung?.straight ?? 0), text: fmtDeg(rung?.straight ?? 0), labelX: start + group + 12 };
  });

  for (const panel of panels) {
    const y = yAt(panel);
    const start = panel.x + (panel.w - group) / 2;
    LADDER_MODELS.forEach((model, m) => {
      const x = start + m * (slot + space);
      const bar = ladder.bars[model][panel.rung] ?? null;
      if (bar === null) {
        missing.push({ model, rung: panel.rung, x, y: y(0) - 8 });
        return;
      }
      if (view === "skill") {
        const { mean, sd } = bar.skill;
        const up = mean >= 0;
        bars.push({
          model,
          rung: panel.rung,
          x,
          w: slot,
          base: y(0),
          end: y(mean),
          value: mean,
          text: fmtPct(mean),
          labelY: up ? y(mean + sd) - 7 : y(mean - sd) + 15,
        });
        whiskers.push({ model, rung: panel.rung, x: x + slot / 2, y1: y(mean + sd), y2: y(mean - sd) });
      } else {
        const highest = Math.max(bar.error.mean, ...bar.error.values);
        bars.push({ model, rung: panel.rung, x, w: slot, base: y(0), end: y(bar.error.mean), value: bar.error.mean, text: fmtDeg(bar.error.mean), labelY: y(highest) - 9 });
        bar.error.values.forEach((value, k) => {
          dots.push({ model, rung: panel.rung, cx: x + slot / 2 + (k - (bar.error.values.length - 1) / 2) * Math.min(6, slot / 4), cy: y(value) });
        });
      }
    });
  }

  const lit = panels[Math.max(0, Math.min(panels.length - 1, selected))] as Panel;
  return {
    view,
    width,
    height,
    grid,
    panels,
    ticks,
    baselines,
    bars,
    missing,
    whiskers,
    dots,
    spotlight: { x: lit.x + 2, y: lit.y, w: lit.w - 4, h: lit.h },
  };
}

const mix = (a: number, b: number, t: number) => a + (b - a) * t;

/** The bars and the always-straight lines of `from` turned `t` of the way
 * into `to`'s: the morph between the two views, at one width. Whiskers, dots
 * and labels are `to`'s (they cross-fade, they do not move). */
export function lerpLadder(from: LadderLayout, to: LadderLayout, t: number): LadderLayout {
  const key = (bar: { model: string; rung: number }) => `${bar.model}/${bar.rung}`;
  const was = new Map(from.bars.map((bar) => [key(bar), bar]));
  return {
    ...to,
    baselines: to.baselines.map((line, i) => {
      const a = from.baselines[i] ?? line;
      return { ...line, x1: mix(a.x1, line.x1, t), x2: mix(a.x2, line.x2, t), y: mix(a.y, line.y, t) };
    }),
    bars: to.bars.map((bar) => {
      const a = was.get(key(bar)) ?? bar;
      return { ...bar, base: mix(a.base, bar.base, t), end: mix(a.end, bar.end, t) };
    }),
    ticks: to.ticks,
  };
}
