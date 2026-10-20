/**
 * The generalisation ladder: four rungs, each testing further from the frames
 * a network trained on, and how SightlineNet, PilotNet and the one-frame
 * SightlineNet fare on each.
 *
 * Every rung's bars are drawn as soon as the data is in; the stepper lights
 * one rung (the furthest, at first) and says in a sentence what moved further
 * from the training frames there, and a click on a rung's panel lights it too.
 * The first view is "better than always steering straight", the view that
 * compares rungs; the toggle morphs the bars into raw error in degrees, where
 * each rung's own steering-straight line stands at its own height. The first
 * time the chart nears the screen in the Results view its bars grow from the
 * zero line, rung after rung, their numbers counting in step. Under reduced
 * motion every change is instant.
 */

import { attrs, easeInOut, easeOut, lines, local, svg, tween, watchWidth } from "../charts/draw.ts";
import { ladderLayout, lerpLadder, type LadderLayout, type LadderView } from "../charts/ladder.ts";
import { fmtDeg, fmtPct } from "../format.ts";
import type { ViewName } from "../router.ts";
import { NETWORK_NAMES, STRAIGHT, barWords, capitalised, ladderAlt, ladderCaption, ladderLead, measurementsFailure, rungCaption, rungNote } from "./copy.ts";
import { textAt, type Frames } from "./countup.ts";
import { LADDER_MODELS, type Ladder, type LadderModel, type Splits } from "./figures.ts";
import { onScreen, prefersReducedMotion } from "./motion.ts";

const GROW_MS = 1500;
const MORPH_MS = 640;
const LEAD_PX = 80;
/** A bar's rounded end, px; its foot on the line stays square. */
const CAP = 4;

/** A bar from `base` to `end`, rounded at `end` only. */
export function barPath(x: number, w: number, base: number, end: number): string {
  const h = end - base;
  if (Math.abs(h) < 0.01) return `M${x} ${base}h${w}`;
  const r = Math.min(CAP, Math.abs(h), w / 2);
  const s = Math.sign(h); // +1 when the bar runs down the page
  const tip = end - s * r;
  const round = (n: number) => Math.round(n * 100) / 100;
  return (
    `M${round(x)} ${round(base)}V${round(tip)}Q${round(x)} ${round(end)} ${round(x + r)} ${round(end)}` +
    `H${round(x + w - r)}Q${round(x + w)} ${round(end)} ${round(x + w)} ${round(tip)}V${round(base)}Z`
  );
}

interface Marks {
  readonly bar: SVGPathElement;
  readonly label: SVGTextElement;
  readonly whisker: SVGPathElement;
  /** Each training's error, a tick across the bar, on a halo that keeps it
   * clear of the bar's fill and of the page behind it. */
  readonly seeds: ReadonlyArray<readonly [halo: SVGPathElement, tick: SVGPathElement]>;
}

interface Drawing {
  readonly root: SVGSVGElement;
  readonly title: SVGTitleElement;
  readonly desc: SVGDescElement;
  readonly spotlight: SVGRectElement;
  readonly grid: SVGGElement;
  readonly titles: readonly SVGTextElement[];
  readonly baselines: readonly SVGLineElement[];
  readonly baselineLabels: readonly SVGTextElement[];
  readonly marks: ReadonlyMap<string, Marks>;
  readonly missing: ReadonlyMap<string, SVGTextElement>;
}

/** How far each part of a frame of the drawing has come. */
interface Frame {
  readonly layout: LadderLayout;
  /** Each bar's growth from the line, by its key. */
  readonly grow: (key: string) => number;
  /** The layout whose labels and ticks are shown, and how strongly. */
  readonly words: LadderLayout;
  readonly wordsOpacity: number;
  /** The labels' counts, eased: 1 is their final text. */
  readonly count: number;
  readonly whiskers: number;
  readonly seeds: number;
}

const key = (model: string, rung: number) => `${model}/${rung}`;

export class LadderSection {
  /** The section in the page, or null on a page without it. */
  static mount(): LadderSection | null {
    const root = document.querySelector<HTMLElement>('[data-ladder="root"]');
    return root ? new LadderSection(root) : null;
  }

  private ladder: Ladder | null = null;
  private splits: Splits | null = null;
  private view: LadderView = "skill";
  private selected = 0;
  private width = 0;
  private layout: LadderLayout | null = null;
  private drawing: Drawing | null = null;
  private drawn = false;
  private onView: boolean;
  private stop: () => void = () => undefined;

  constructor(
    private readonly root: HTMLElement,
    private readonly reduced: () => boolean = prefersReducedMotion,
    private readonly frames?: Frames,
  ) {
    this.onView = document.documentElement.dataset["view"] === "results";
    this.part("stepper").addEventListener("change", (event) => {
      const value = Number((event.target as HTMLInputElement).value);
      if (Number.isInteger(value)) this.select(value);
    });
    this.part("views").addEventListener("change", (event) => {
      const value = (event.target as HTMLInputElement).value;
      if (value === "skill" || value === "error") this.setView(value);
    });
    if (typeof IntersectionObserver === "function") {
      new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) this.maybeGrow();
      }, { rootMargin: `0px 0px ${LEAD_PX}px 0px` }).observe(this.part("plot"));
    }
    const plot = this.part("plot");
    plot.addEventListener("pointermove", (event) => this.point(event));
    plot.addEventListener("pointerleave", () => this.point(null));
    plot.addEventListener("click", (event) => this.pick(event));
  }

  private part<T extends HTMLElement = HTMLElement>(name: string): T {
    const element = this.root.querySelector<T>(`[data-ladder="${name}"]`);
    if (!element) throw new Error(`the ladder has no ${name}`);
    return element;
  }

  /** Write the ladder in: its stepper from the data's rungs, the furthest
   * rung lit, every bar drawn at its value at once. */
  show(ladder: Ladder, splits: Splits): void {
    this.ladder = ladder;
    this.splits = splits;
    // The view picked, before the data or by the browser restoring the page.
    const view = this.root.querySelector<HTMLInputElement>('input[name="ladder-view"]:checked')?.value;
    if (view === "skill" || view === "error") this.view = view;
    this.selected = ladder.rungs.length - 1;
    this.buildStepper(ladder);
    this.buildCaptions(ladder, splits);
    this.fillTable(ladder);
    this.drawn = this.reduced();
    this.root.dataset["state"] = "ready";
    this.caption();
    watchWidth(this.part("plot"), (width) => this.resize(width));
  }

  /** Say why there is no ladder, in its lead, where it is seen: the figure
   * is put away. */
  fail(error: unknown): void {
    this.part("lead").textContent = measurementsFailure(error);
    this.root.dataset["state"] = "unavailable";
  }

  viewShown(view: ViewName): void {
    this.onView = view === "results";
    this.maybeGrow();
  }

  private buildStepper(ladder: Ladder): void {
    const stepper = this.part("stepper");
    const legend = stepper.querySelector("legend");
    const steps = ladder.rungs.map((rung, i) => {
      const label = document.createElement("label");
      label.className = "stepper__step";
      const input = document.createElement("input");
      input.className = "stepper__input";
      input.type = "radio";
      input.name = "ladder-rung";
      input.value = String(i);
      input.checked = i === this.selected;
      // The rung's sentence describes it, read when the button is reached.
      input.setAttribute("aria-describedby", this.part("caption").id);
      const mark = document.createElement("span");
      mark.className = "stepper__mark";
      mark.setAttribute("aria-hidden", "true");
      const name = document.createElement("span");
      name.className = "stepper__name";
      name.textContent = rung.name;
      label.append(input, mark, name);
      return label;
    });
    stepper.replaceChildren(...(legend ? [legend] : []), ...steps);
  }

  /** Every rung's caption, unseen, in the caption's own cell: the cell is as
   * tall as the longest, so stepping moves nothing below it. */
  private buildCaptions(ladder: Ladder, splits: Splits): void {
    const box = this.part("captions");
    for (const old of box.querySelectorAll(".ladder__sizer")) old.remove();
    ladder.rungs.forEach((rung, i) => {
      const sizer = document.createElement("p");
      sizer.className = "ladder__said ladder__sizer";
      sizer.setAttribute("aria-hidden", "true");
      const name = document.createElement("span");
      name.className = "ladder__rung";
      name.textContent = capitalised(rung.name);
      sizer.append(name, ` ${rungCaption(rung, splits)} ${rungNote(ladder, i)}`);
      box.append(sizer);
    });
  }

  private caption(): void {
    if (!this.ladder || !this.splits) return;
    const rung = this.ladder.rungs[this.selected];
    if (!rung) return;
    this.part("caption-rung").textContent = capitalised(rung.name);
    this.part("caption-text").textContent = `${rungCaption(rung, this.splits)} ${rungNote(this.ladder, this.selected)}`;
  }

  private resize(width: number): void {
    if (!this.ladder || width <= 0) return;
    this.stop();
    this.width = width;
    this.layout = ladderLayout(this.ladder, this.view, width, this.selected);
    // In a row, each step stands over its rung's panel.
    this.root.style.setProperty("--panels-start", this.layout.grid ? "0px" : `${this.layout.panels[0]?.x ?? 0}px`);
    this.build(this.ladder, this.layout);
    this.paint(this.still(this.layout));
    this.maybeGrow();
  }

  /** A frame with everything in place. */
  private still(layout: LadderLayout): Frame {
    return { layout, grow: () => 1, words: layout, wordsOpacity: 1, count: 1, whiskers: layout.view === "skill" ? 1 : 0, seeds: layout.view === "error" ? 1 : 0 };
  }

  private build(ladder: Ladder, layout: LadderLayout): void {
    const plot = this.part("plot");
    const root = svg("svg", { class: "chart__svg ladder-chart", role: "img", "aria-labelledby": "ladder-svg-title ladder-svg-desc" });
    const title = svg("title", { id: "ladder-svg-title" }, root);
    const desc = svg("desc", { id: "ladder-svg-desc" }, root);
    const spotlight = svg("rect", { class: "ladder-chart__spotlight", rx: 14 }, root);
    const grid = svg("g", { class: "chart__grid" }, root);
    const titles = layout.panels.map(() => svg("text", { class: "ladder-chart__title" }, root));
    const lines = svg("g", { class: "ladder-chart__baselines" }, root);
    const baselines = layout.panels.map(() => svg("line", { class: "ladder-chart__baseline" }, lines));
    const baselineLabels = layout.panels.map(() => svg("text", { class: "chart__note", "text-anchor": "start" }, lines));
    const bars = svg("g", { class: "ladder-chart__bars" }, root);
    const marks = new Map<string, Marks>();
    const missing = new Map<string, SVGTextElement>();
    ladder.rungs.forEach((_, rung) => {
      for (const model of LADDER_MODELS) {
        const bar = ladder.bars[model][rung];
        if (!bar) {
          missing.set(key(model, rung), svg("text", { class: "ladder-chart__missing" }, bars));
          continue;
        }
        const series = `series--${model === "sightlinenet-1f" ? "oneframe" : model}`;
        marks.set(key(model, rung), {
          bar: svg("path", { class: `ladder-chart__bar ${series}` }, bars),
          whisker: svg("path", { class: "ladder-chart__whisker" }, bars),
          seeds: bar.error.values.map(() => [svg("path", { class: "ladder-chart__seed-halo" }, bars), svg("path", { class: "ladder-chart__seed" }, bars)] as const),
          label: svg("text", { class: "chart__value chart__value--small", "text-anchor": "middle" }, bars),
        });
      }
    });
    const old = plot.querySelector("svg");
    if (old) old.replaceWith(root);
    else plot.prepend(root);
    this.drawing = { root, title, desc, spotlight, grid, titles, baselines, baselineLabels, marks, missing };
    this.describe();
  }

  /** The words that follow the view: the lead says what a bar's height
   * means in it, and the chart's title, caption and text alternative. */
  private describe(): void {
    if (this.root.dataset["state"] !== "unavailable") this.part("lead").textContent = ladderLead(this.view);
    this.part("chart-title").textContent =
      this.view === "skill" ? "How much better than always steering straight" : "Mean error, in degrees of steering-wheel angle";
    this.part("chart-caption").textContent = ladderCaption(this.ladder, this.view);
    if (!this.ladder || !this.drawing) return;
    const alt = ladderAlt(this.ladder, this.view);
    this.drawing.title.textContent = alt.title;
    this.drawing.desc.textContent = alt.desc;
  }

  private paint(frame: Frame): void {
    const d = this.drawing;
    if (!d) return;
    const { layout, words } = frame;
    attrs(d.root, { viewBox: `0 0 ${layout.width} ${layout.height}`, width: layout.width, height: layout.height });
    d.root.dataset["view"] = layout.view;
    this.light(layout);

    // The value grid, cross-faded between the views' scales.
    const rules = words.ticks.map((tick, i) => {
      const line = d.grid.children[i * 2] instanceof SVGLineElement ? (d.grid.children[i * 2] as SVGLineElement) : svg("line", { class: "chart__rule" });
      const text = d.grid.children[i * 2 + 1] instanceof SVGTextElement ? (d.grid.children[i * 2 + 1] as SVGTextElement) : svg("text", { class: "chart__tick", "text-anchor": "end" });
      attrs(line, { x1: tick.x1, x2: tick.x2, y1: tick.y, y2: tick.y });
      attrs(text, { x: tick.labelX, y: tick.y + 4 });
      text.textContent = tick.text;
      return [line, text];
    });
    d.grid.replaceChildren(...rules.flat());
    attrs(d.grid, { opacity: frame.wordsOpacity });

    layout.panels.forEach((panel, i) => {
      const title = d.titles[i];
      if (!title) return;
      attrs(title, { x: panel.x + 10, y: panel.y + 18, opacity: panel.title ? 1 : 0 });
      title.textContent = panel.title ?? "";
      title.classList.toggle("is-lit", i === this.selected);
    });

    layout.baselines.forEach((line, i) => {
      const el = d.baselines[i];
      if (el) attrs(el, { x1: line.x1, x2: line.x2, y1: line.y, y2: line.y });
    });
    words.baselines.forEach((line, i) => {
      const el = d.baselineLabels[i];
      if (!el) return;
      attrs(el, { x: line.labelX, y: line.y + 4, opacity: line.text ? frame.wordsOpacity : 0 });
      el.textContent = line.text;
    });

    const labelled = new Map(words.bars.map((bar) => [key(bar.model, bar.rung), bar]));
    for (const bar of layout.bars) {
      const marks = d.marks.get(key(bar.model, bar.rung));
      if (!marks) continue;
      const grown = frame.grow(key(bar.model, bar.rung));
      const end = bar.base + (bar.end - bar.base) * grown;
      attrs(marks.bar, { d: barPath(bar.x, bar.w, bar.base, end) });
      marks.bar.classList.toggle("is-lit", bar.rung === this.selected);
      const label = labelled.get(key(bar.model, bar.rung)) ?? bar;
      const count = frame.count * (words === layout ? grown : 1);
      // A growing bar carries its label at its tip, as far from it as the
      // label will stand from the finished bar.
      const ride = words === layout ? end - bar.end : 0;
      attrs(marks.label, { x: label.x + label.w / 2, y: label.labelY + ride, opacity: frame.wordsOpacity * (grown > 0 ? 1 : 0) });
      // A skill counts in the percent its text shows, an error in degrees.
      const shown = words.view === "skill" ? label.value * 100 : label.value;
      marks.label.textContent = textAt(label.text, shown, 0, count);
      marks.label.classList.toggle("is-lit", bar.rung === this.selected);
    }
    // Whiskers belong to the skill view, each training's tick to the error
    // view; each is drawn where its own view puts it and faded in or out.
    const skill = this.layoutFor("skill");
    const error = this.layoutFor("error");
    for (const whisker of skill?.whiskers ?? []) {
      const marks = d.marks.get(key(whisker.model, whisker.rung));
      if (!marks) continue;
      const [x, y1, y2] = [whisker.x, whisker.y1, whisker.y2].map((n) => Math.round(n * 100) / 100) as [number, number, number];
      attrs(marks.whisker, { d: `M${x - 4} ${y1}h8M${x} ${y1}V${y2}M${x - 4} ${y2}h8`, opacity: frame.whiskers });
    }
    const seen = new Map<string, number>();
    const columns = new Map((error?.bars ?? []).map((bar) => [key(bar.model, bar.rung), bar]));
    for (const dot of error?.dots ?? []) {
      const k = key(dot.model, dot.rung);
      const marks = d.marks.get(k);
      const column = columns.get(k);
      if (!marks || !column) continue;
      const n = seen.get(k) ?? 0;
      seen.set(k, n + 1);
      const pair = marks.seeds[n];
      if (!pair) continue;
      const y = Math.round(dot.cy * 100) / 100;
      const tick = `M${Math.round((column.x - 2) * 100) / 100} ${y}h${Math.round((column.w + 4) * 100) / 100}`;
      for (const el of pair) attrs(el, { d: tick, opacity: frame.seeds });
    }
    for (const note of words.missing) {
      const el = d.missing.get(key(note.model, note.rung));
      if (!el) continue;
      attrs(el, { x: note.x, y: note.y - 15, opacity: frame.wordsOpacity });
      lines(el, ["not trained", "this way"], note.x, 15);
    }
  }

  /** The light behind the selected rung's panel: it slides there (the
   * stylesheet times it). */
  private light(layout: LadderLayout): void {
    const d = this.drawing;
    const panel = layout.panels[this.selected];
    if (!d || !panel) return;
    attrs(d.spotlight, { width: panel.w - 4, height: panel.h });
    d.spotlight.style.transform = `translate(${panel.x + 2}px, ${panel.y}px)`;
  }

  /** The layout of `view` at the width on screen. */
  private layoutFor(view: LadderView): LadderLayout | null {
    if (!this.ladder || this.width <= 0) return null;
    return view === this.layout?.view ? this.layout : ladderLayout(this.ladder, view, this.width, this.selected);
  }

  /** Grow the bars once, rung after rung, when the chart first nears the screen. */
  private maybeGrow(): void {
    if (this.drawn || !this.layout || !this.onView) return;
    if (!onScreen(this.part("plot"), LEAD_PX)) return;
    this.drawn = true;
    const layout = this.layout;
    const rungs = layout.panels.length;
    this.stop = tween(
      GROW_MS,
      (p) => {
        const grow = (k: string) => {
          const [model = "", rung = "0"] = k.split("/");
          const order = LADDER_MODELS.indexOf(model as LadderModel);
          return easeOut(local(p, (Number(rung) / rungs) * 0.42 + order * 0.05, 0.42));
        };
        this.paint({ layout, grow, words: layout, wordsOpacity: 1, count: 1, whiskers: local(p, 0.78, 0.22), seeds: layout.view === "error" ? local(p, 0.78, 0.22) : 0 });
      },
      { reduced: this.reduced(), frames: this.frames },
    );
  }

  /** Light rung `index`: the stepper, the caption, the panel. */
  private select(index: number): void {
    if (!this.ladder || index < 0 || index >= this.ladder.rungs.length) return;
    this.selected = index;
    const input = this.part("stepper").querySelector<HTMLInputElement>(`input[value="${index}"]`);
    if (input && !input.checked) input.checked = true;
    this.caption();
    if (!this.layout || !this.ladder) return;
    this.layout = ladderLayout(this.ladder, this.view, this.width, index);
    // The rest of the drawing stays as it is; only the light moves.
    const d = this.drawing;
    if (!d) return;
    this.light(this.layout);
    d.titles.forEach((title, i) => title.classList.toggle("is-lit", i === index));
    for (const [k, marks] of d.marks) {
      const lit = Number(k.split("/")[1]) === index;
      marks.bar.classList.toggle("is-lit", lit);
      marks.label.classList.toggle("is-lit", lit);
    }
  }

  /** Morph the bars into the other view; its words change at once, data or not. */
  private setView(view: LadderView): void {
    if (view === this.view) return;
    const before = this.layout;
    this.view = view;
    this.describe();
    if (!this.ladder) return;
    this.point(null);
    if (!before) return;
    this.stop();
    const after = ladderLayout(this.ladder, view, this.width, this.selected);
    this.layout = after;
    this.drawn = true;
    this.describe();
    this.stop = tween(
      MORPH_MS,
      (p) => {
        const t = easeInOut(p);
        const words = p < 0.5 ? before : after;
        const wordsOpacity = p < 0.5 ? 1 - p * 2 : p * 2 - 1;
        this.paint({
          layout: lerpLadder(before, after, t),
          grow: () => 1,
          words,
          wordsOpacity,
          count: 1,
          // What belongs to the view being left goes early; what belongs to
          // the view arriving comes late, once the bars are nearly there.
          whiskers: view === "skill" ? local(p, 0.6, 0.4) : 1 - local(p, 0, 0.35),
          seeds: view === "error" ? local(p, 0.6, 0.4) : 1 - local(p, 0, 0.35),
        });
      },
      { reduced: this.reduced(), frames: this.frames },
    );
  }

  /** The panel under a click, lit. */
  private pick(event: MouseEvent): void {
    const layout = this.layout;
    const d = this.drawing;
    if (!layout || !d) return;
    const box = d.root.getBoundingClientRect();
    const px = event.clientX - box.left;
    const py = event.clientY - box.top;
    const panel = layout.panels.find((p) => px >= p.x && px <= p.x + p.w && py >= p.y && py <= p.y + p.h);
    if (panel) this.select(panel.rung);
  }

  /** The bar under the pointer, in words; null puts it away. */
  private point(event: PointerEvent | null): void {
    const tip = this.part("tip");
    const layout = this.layout;
    const d = this.drawing;
    const clear = () => {
      tip.hidden = true;
      d?.root.classList.remove("is-pointing");
      for (const marks of d?.marks.values() ?? []) marks.bar.classList.remove("is-hot");
    };
    if (!event || !layout || !d || !this.ladder) return clear();
    const box = d.root.getBoundingClientRect();
    const px = event.clientX - box.left;
    const py = event.clientY - box.top;
    const panel = layout.panels.find((p) => px >= p.x && px <= p.x + p.w && py >= p.y && py <= p.y + p.h);
    if (!panel) return clear();
    const bar = layout.bars
      .filter((b) => b.rung === panel.rung)
      .find((b) => px >= b.x - 4 && px <= b.x + b.w + 4);
    if (!bar) return clear();
    const data = this.ladder.bars[bar.model][bar.rung] ?? null;
    const rung = this.ladder.rungs[bar.rung];
    d.root.classList.add("is-pointing");
    for (const [k, marks] of d.marks) marks.bar.classList.toggle("is-hot", k === key(bar.model, bar.rung));
    tip.textContent = `${capitalised(rung?.name ?? "")}: ${barWords(bar.model, data, this.view)}`;
    tip.hidden = false;
    const left = Math.min(Math.max(8, px + 14), Math.max(8, box.width - tip.offsetWidth - 8));
    const top = Math.max(4, py - tip.offsetHeight - 14);
    tip.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  }

  /** Every number of the chart, as a table. */
  private fillTable(ladder: Ladder): void {
    const table = this.part<HTMLTableElement>("table");
    const caption = table.querySelector("caption");
    table.replaceChildren(...(caption ? [caption] : []));
    const head = table.createTHead().insertRow();
    for (const heading of ["Rung", "Network", "Better than always steering straight", "Mean error", "Each training", STRAIGHT]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = heading;
      head.append(cell);
    }
    const body = table.createTBody();
    ladder.rungs.forEach((rung, i) => {
      for (const model of LADDER_MODELS) {
        const bar = ladder.bars[model][i] ?? null;
        const row = body.insertRow();
        const name = document.createElement("th");
        name.scope = "row";
        name.textContent = rung.name;
        row.append(name);
        row.insertCell().textContent = NETWORK_NAMES[model].name;
        if (bar === null) {
          const cell = row.insertCell();
          cell.colSpan = 3;
          cell.textContent = "not trained this way";
        } else {
          row.insertCell().textContent = `${fmtPct(bar.skill.mean)} ±${fmtPct(bar.skill.sd)}`;
          row.insertCell().textContent = fmtDeg(bar.error.mean);
          row.insertCell().textContent = bar.error.values.map((value) => fmtDeg(value)).join(", ");
        }
        row.insertCell().textContent = fmtDeg(rung.straight);
      }
    });
  }
}
