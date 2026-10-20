/**
 * The paired experiment: the same frames, scored by a network trained on
 * shuffled frames and by the same network trained with whole stretches held
 * back -- seed by seed, as a slope chart -- with a sketch of why those frames
 * are the fair ones to compare on.
 *
 * The page is served with the chart's frame in quiet grey and plain words;
 * once results.json has loaded every number is written in and the chart is
 * drawn at its final values at once, so it is never empty and never shows a
 * zero it did not measure. The first time it nears the screen, in the Results
 * view, it draws itself in: the shuffled scores first, then each seed's line
 * rising to its held-back score, the labels counting in step. Picking the
 * other network morphs the chart into it on the same scale. Under reduced
 * motion both are instant.
 */

import { attrs, easeInOut, easeOut, lines, local, svg, tween, watchWidth } from "../charts/draw.ts";
import { lerpSlope, nearestLine, slopeLayout, type SlopeLayout } from "../charts/slope.ts";
import { fmtCount, fmtDeg, fmtRatio } from "../format.ts";
import type { ViewName } from "../router.ts";
import {
  NETWORK_NAMES,
  measurementsFailure,
  pairedAlt,
  pairedFrames,
  pairedLead,
  pairedMeaning,
  pairedScale,
  splitsText,
  STRAIGHT,
} from "./copy.ts";
import { textAt, type Frames } from "./countup.ts";
import type { Figures, Network, PairedNetwork } from "./figures.ts";
import { onScreen, prefersReducedMotion } from "./motion.ts";

const DRAW_MS = 1500;
const MORPH_MS = 560;
/** How far below the screen the chart starts drawing itself in, px. */
const LEAD_PX = 80;
/** How near a seed's line the pointer must be to pick it out, px. */
const REACH_PX = 22;
const DOT_R = 6;

type Paired = Readonly<Record<Network, PairedNetwork>>;

/** How far each part of the drawing has come: 1 everywhere is the finished
 * chart. */
interface Progress {
  /** The shuffled scores: the left dots and their label. */
  readonly left: number;
  /** Each seed's line, from its left dot towards its right one. */
  readonly lines: readonly number[];
  /** Each seed's right dot. */
  readonly right: readonly number[];
  /** The counts of the held-back mean and the ratio, already eased: they
   * move with what they label. */
  readonly count: number;
  /** While the chart draws in, the held-back mean's label rises from the
   * shuffled mean's height with its count. */
  readonly rise?: boolean;
}

const DONE: Progress = { left: 1, lines: [1, 1, 1, 1, 1, 1], right: [1, 1, 1, 1, 1, 1], count: 1 };

interface Drawing {
  readonly root: SVGSVGElement;
  readonly title: SVGTitleElement;
  readonly desc: SVGDescElement;
  readonly ticks: ReadonlyArray<{ readonly line: SVGLineElement; readonly text: SVGTextElement }>;
  readonly straight: readonly SVGLineElement[];
  readonly straightLabel: SVGTextElement;
  readonly lines: readonly SVGLineElement[];
  readonly left: readonly SVGCircleElement[];
  readonly right: readonly SVGCircleElement[];
  readonly meanLeft: SVGTextElement;
  readonly meanRight: SVGTextElement;
  readonly ratio: SVGTextElement;
  readonly axisLeft: SVGTextElement;
  readonly axisRight: SVGTextElement;
}

export class PairedSection {
  /** The section in the page, or null on a page without it. */
  static mount(): PairedSection | null {
    const root = document.querySelector<HTMLElement>('[data-paired="root"]');
    return root ? new PairedSection(root) : null;
  }

  private data: Paired | null = null;
  private network: Network = "sightlinenet";
  private layout: SlopeLayout | null = null;
  private drawing: Drawing | null = null;
  private drawn = false;
  private onView: boolean;
  private stop: () => void = () => undefined;
  /** What the labels were counting from when the last motion started. */
  private from: { blocked: number; ratio: number; shuffled: number } | null = null;

  constructor(
    private readonly root: HTMLElement,
    private readonly reduced: () => boolean = prefersReducedMotion,
    private readonly frames?: Frames,
  ) {
    this.onView = document.documentElement.dataset["view"] === "results";
    this.part("networks").addEventListener("change", (event) => {
      const value = (event.target as HTMLInputElement).value;
      if (value === "sightlinenet" || value === "pilotnet") this.choose(value);
    });
    if (typeof IntersectionObserver === "function") {
      new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) this.maybeDraw();
      }, { rootMargin: `0px 0px ${LEAD_PX}px 0px` }).observe(this.part("plot"));
    }
    const plot = this.part("plot");
    plot.addEventListener("pointermove", (event) => this.point(event));
    plot.addEventListener("pointerleave", () => this.point(null));
  }

  private part<T extends HTMLElement = HTMLElement>(name: string): T {
    const element = this.root.querySelector<T>(`[data-paired="${name}"]`);
    if (!element) throw new Error(`the paired experiment has no ${name}`);
    return element;
  }

  /** Write the experiment in and draw its chart, final at once, for the
   * network picked (the browser may have restored a pick from before). */
  show(paired: Paired, gaps: Figures["gaps"]): void {
    this.data = paired;
    const picked = this.root.querySelector<HTMLInputElement>('input[name="paired-network"]:checked')?.value;
    if (picked === "sightlinenet" || picked === "pilotnet") this.network = picked;
    const splits = splitsText(gaps);
    this.part("shuffled").textContent = splits.shuffled;
    this.part("blocked").textContent = splits.blocked;
    this.fillTable(paired);
    this.drawn = this.reduced();
    this.root.dataset["state"] = "ready";
    this.describe();
    watchWidth(this.part("plot"), (width) => this.resize(width));
  }

  /** Say why there is no chart, in place of it. */
  fail(error: unknown): void {
    this.part("lead").textContent = measurementsFailure(error);
    this.root.dataset["state"] = "unavailable";
  }

  /** The router switched views. */
  viewShown(view: ViewName): void {
    this.onView = view === "results";
    this.maybeDraw();
  }

  /** The words around the chart, for the network on screen. */
  private describe(): void {
    if (!this.data) return;
    const network = this.data[this.network];
    const name = NETWORK_NAMES[this.network].name;
    this.part("lead").textContent = pairedLead(name, network);
    this.part("meaning").textContent = pairedMeaning(name, network.ratio.mean);
    this.part("scale").textContent = pairedScale(network);
    this.part("caption").textContent = pairedFrames(network);
    if (this.drawing) {
      const alt = pairedAlt(name, network);
      this.drawing.title.textContent = alt.title;
      this.drawing.desc.textContent = alt.desc;
      this.drawing.root.dataset["network"] = this.network;
    }
  }

  private resize(width: number): void {
    if (!this.data || width <= 0) return;
    this.stop();
    const all = [this.data.sightlinenet, this.data.pilotnet];
    this.layout = slopeLayout(this.data[this.network], all, width);
    if (!this.drawing || this.drawing.ticks.length !== this.layout.ticks.length) this.build(this.layout);
    this.paint(this.layout, DONE);
    this.maybeDraw();
  }

  /** The chart's elements, made once for the data's seeds and ticks. */
  private build(layout: SlopeLayout): void {
    const plot = this.part("plot");
    const root = svg("svg", { class: "chart__svg slope", role: "img", "aria-labelledby": "paired-svg-title paired-svg-desc" });
    const title = svg("title", { id: "paired-svg-title" }, root);
    const desc = svg("desc", { id: "paired-svg-desc" }, root);
    const grid = svg("g", { class: "chart__grid" }, root);
    const ticks = layout.ticks.map(() => ({ line: svg("line", { class: "chart__rule" }, grid), text: svg("text", { class: "chart__tick" }, grid) }));
    const back = svg("g", { class: "slope__straight" }, root);
    const straight = layout.straight.map(() => svg("line", { class: "slope__straight-line" }, back));
    const straightLabel = svg("text", { class: "chart__note", "text-anchor": "end" }, back);
    const marks = svg("g", { class: "slope__marks" }, root);
    const seedLines = layout.lines.map(() => svg("line", { class: "slope__line" }, marks));
    const left = layout.lines.map(() => svg("circle", { class: "slope__dot slope__dot--shuffled" }, marks));
    const right = layout.lines.map(() => svg("circle", { class: "slope__dot slope__dot--blocked" }, marks));
    const labels = svg("g", { class: "slope__labels" }, root);
    const meanLeft = svg("text", { class: "chart__value", "text-anchor": "end" }, labels);
    const meanRight = svg("text", { class: "chart__value", "text-anchor": "start" }, labels);
    const ratio = svg("text", { class: "slope__ratio", "text-anchor": "middle" }, labels);
    const axisLeft = svg("text", { class: "chart__axis", "text-anchor": "middle" }, root);
    const axisRight = svg("text", { class: "chart__axis", "text-anchor": "middle" }, root);
    const old = plot.querySelector("svg");
    if (old) old.replaceWith(root);
    else plot.prepend(root);
    this.drawing = { root, title, desc, ticks, straight, straightLabel, lines: seedLines, left, right, meanLeft, meanRight, ratio, axisLeft, axisRight };
    this.describe();
  }

  /** Draw `layout`, each part `progress` of the way in. */
  private paint(layout: SlopeLayout, progress: Progress): void {
    const d = this.drawing;
    if (!d) return;
    const { width, height, plot, x } = layout;
    attrs(d.root, { viewBox: `0 0 ${width} ${height}`, width, height });
    d.root.classList.toggle("slope--narrow", layout.narrow);
    layout.ticks.forEach((tick, i) => {
      const el = d.ticks[i];
      if (!el) return;
      attrs(el.line, { x1: plot.left, x2: plot.right, y1: tick.y, y2: tick.y });
      el.line.classList.toggle("chart__rule--zero", i === 0);
      attrs(el.text, { x: plot.left - 8, y: tick.y + 4, "text-anchor": "end" });
      el.text.textContent = tick.text;
    });
    layout.straight.forEach((line, i) => {
      const el = d.straight[i];
      if (el) attrs(el, { x1: plot.left, x2: plot.right, y1: line.y, y2: line.y });
    });
    attrs(d.straightLabel, { x: layout.straightLabel.x, y: layout.straightLabel.y });
    d.straightLabel.textContent = layout.straightLabel.text;
    layout.lines.forEach((line, i) => {
      const along = easeOut(progress.lines[i] ?? 1);
      attrs(d.lines[i] as SVGLineElement, {
        x1: line.x1,
        y1: line.y1,
        x2: line.x1 + (line.x2 - line.x1) * along,
        y2: line.y1 + (line.y2 - line.y1) * along,
        opacity: along > 0 ? 1 : 0,
      });
      attrs(d.left[i] as SVGCircleElement, { cx: line.x1, cy: line.y1, r: DOT_R * easeOut(progress.left) });
      attrs(d.right[i] as SVGCircleElement, { cx: line.x2, cy: line.y2, r: DOT_R * easeOut(progress.right[i] ?? 1) });
    });
    const { shuffled, blocked } = layout.means;
    const from = this.from;
    const counted = progress.count;
    attrs(d.meanLeft, { x: shuffled.x, y: shuffled.y + 6, opacity: easeOut(progress.left) });
    lines(d.meanLeft, [textAt(shuffled.text, shuffled.value, from?.shuffled ?? shuffled.value, counted), "mean"], shuffled.x, 17);
    const rightY = progress.rise ? shuffled.y + (blocked.y - shuffled.y) * counted : blocked.y;
    attrs(d.meanRight, { x: blocked.x, y: rightY + 6, opacity: progress.count > 0 ? 1 : 0 });
    lines(d.meanRight, [textAt(blocked.text, blocked.value, from?.blocked ?? blocked.value, counted), "mean"], blocked.x, 17);
    attrs(d.ratio, { x: layout.ratio.x, y: layout.ratio.y, opacity: progress.count > 0 ? 1 : 0 });
    d.ratio.textContent = `${textAt(layout.ratio.text, layout.ratio.value, from?.ratio ?? layout.ratio.value, counted)} as much, on average`;
    attrs(d.axisLeft, { x: x.shuffled, y: height - 26 });
    lines(d.axisLeft, ["trained on", "shuffled frames"], x.shuffled, 15);
    attrs(d.axisRight, { x: x.blocked, y: height - 26 });
    lines(d.axisRight, layout.narrow ? ["trained with", "stretches held back"] : ["trained with stretches", "held back"], x.blocked, 15);
  }

  /** Draw the chart in, once, as soon as it is in the Results view and near the screen. */
  private maybeDraw(): void {
    if (this.drawn || !this.layout || !this.onView || !this.data) return;
    if (!onScreen(this.part("plot"), LEAD_PX)) return;
    this.drawn = true;
    const layout = this.layout;
    const network = this.data[this.network];
    // The held-back mean counts up from the shuffled one, the ratio from one.
    this.from = { shuffled: network.shuffled, blocked: network.shuffled, ratio: 1 };
    const seeds = layout.lines.length;
    this.stop = tween(
      DRAW_MS,
      (p) => {
        const progress: Progress = {
          left: local(p, 0, 0.14),
          lines: Array.from({ length: seeds }, (_, i) => local(p, 0.1 + i * 0.08, 0.52)),
          right: Array.from({ length: seeds }, (_, i) => local(p, 0.5 + i * 0.08, 0.14)),
          count: easeOut(local(p, 0.1, 0.62)),
          rise: true,
        };
        this.paint(layout, progress);
        if (p >= 1) this.from = null;
      },
      { reduced: this.reduced(), frames: this.frames },
    );
    this.lightSketch();
  }

  /** The frames both splits hold out light up, one after another. */
  private lightSketch(): void {
    if (this.reduced()) return;
    this.root.querySelectorAll<HTMLElement>(".sketch__frame--both").forEach((frame, i) => {
      frame.animate?.([{ opacity: 0.2, transform: "scaleY(0.6)" }, { opacity: 1, transform: "none" }], {
        duration: 520,
        delay: 260 + i * 140,
        easing: "cubic-bezier(0.16, 1, 0.3, 1)",
        fill: "backwards",
      });
    });
  }

  /** The visitor picked the other network: morph into it on the same scale. */
  private choose(network: Network): void {
    if (network === this.network) return;
    const before = this.layout;
    this.network = network;
    this.describe();
    this.point(null);
    if (!this.data || !before) return;
    this.stop();
    const after = slopeLayout(this.data[network], [this.data.sightlinenet, this.data.pilotnet], before.width);
    this.from = { shuffled: before.means.shuffled.value, blocked: before.means.blocked.value, ratio: before.ratio.value };
    this.layout = after;
    this.drawn = true;
    this.stop = tween(
      MORPH_MS,
      (p) => {
        const t = easeInOut(p);
        this.paint(lerpSlope(before, after, t), { ...DONE, count: t });
        if (p >= 1) this.from = null;
      },
      { reduced: this.reduced(), frames: this.frames },
    );
  }

  /** Pick out the seed under the pointer, with its numbers; null puts it back. */
  private point(event: PointerEvent | null): void {
    const d = this.drawing;
    const layout = this.layout;
    const tip = this.part("tip");
    const clear = () => {
      tip.hidden = true;
      d?.root.classList.remove("is-pointing");
      for (const element of [...(d?.lines ?? []), ...(d?.left ?? []), ...(d?.right ?? [])]) element.classList.remove("is-lit");
    };
    if (!event || !d || !layout || !this.data) return clear();
    const box = d.root.getBoundingClientRect();
    const px = event.clientX - box.left;
    const py = event.clientY - box.top;
    const near = nearestLine(layout, px, py);
    if (!near || near.distance > REACH_PX) return clear();
    const seed = this.data[this.network].seeds[near.index];
    if (!seed) return clear();
    d.root.classList.add("is-pointing");
    [d.lines, d.left, d.right].forEach((group) => group.forEach((element, i) => element.classList.toggle("is-lit", i === near.index)));
    tip.textContent =
      `Seed ${seed.seed}, ${fmtCount(seed.frames)} frames: ${fmtDeg(seed.shuffled)} trained on shuffled frames, ` +
      `${fmtDeg(seed.blocked)} with stretches held back (${fmtRatio(seed.ratio)}). ${STRAIGHT}: ${fmtDeg(seed.straight)}.`;
    tip.hidden = false;
    const left = Math.min(Math.max(8, px + 14), Math.max(8, box.width - tip.offsetWidth - 8));
    const top = Math.max(4, py - tip.offsetHeight - 14);
    tip.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  }

  /** Every number of the chart, as a table, for both networks. */
  private fillTable(paired: Paired): void {
    const table = this.part<HTMLTableElement>("table");
    const caption = table.querySelector("caption");
    table.replaceChildren(...(caption ? [caption] : []));
    const head = table.createTHead().insertRow();
    for (const heading of ["Network", "Seed", "Frames both splits held out", "Trained on shuffled frames", "Trained with stretches held back", "Times as much", STRAIGHT]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = heading;
      head.append(cell);
    }
    const body = table.createTBody();
    for (const network of ["sightlinenet", "pilotnet"] as const) {
      for (const seed of paired[network].seeds) {
        const row = body.insertRow();
        const name = document.createElement("th");
        name.scope = "row";
        name.textContent = NETWORK_NAMES[network].name;
        row.append(name);
        for (const text of [String(seed.seed), fmtCount(seed.frames), fmtDeg(seed.shuffled), fmtDeg(seed.blocked), fmtRatio(seed.ratio), fmtDeg(seed.straight)]) {
          row.insertCell().textContent = text;
        }
      }
    }
  }
}
