/**
 * The Results view's cards: what the second frame buys, where the model
 * struggles, the standard-recipe PilotNet, and where every number comes from.
 *
 * Each card reads its own part of results.json, so a malformed part costs
 * only its card. Each small chart is drawn at its values the moment the data
 * is in, and its bars grow from their zero line once, the first time the card
 * nears the screen in the Results view (instantly under reduced motion). The
 * trust card repeats the live check the headline band runs, without a second
 * announcement: only the band's verdict speaks to a screen reader.
 */

import { barsLayout, type BarsLayout, type RowSpec } from "../charts/bars.ts";
import { attrs, easeOut, lines, local, svg, tween, watchWidth } from "../charts/draw.ts";
import { fmtCount, fmtDeg, fmtPct } from "../format.ts";
import type { ViewName } from "../router.ts";
import {
  exactDeg,
  exportText,
  legacyAlt,
  legacyText,
  measurementsFailure,
  protocolTexts,
  recipeNote,
  regimeNames,
  regimesAlt,
  regimesText,
  skillWords,
  twoFramesText,
  driveTitle,
  type ParityLook,
} from "./copy.ts";
import { textAt, type Frames } from "./countup.ts";
import { REGIMES, type ExportCheck, type Ladder, type Legacy, type Regimes, type Splits } from "./figures.ts";
import { onScreen, prefersReducedMotion } from "./motion.ts";

const GROW_MS = 1100;
const LEAD_PX = 60;

export type CardName = "frames" | "regimes" | "recipe" | "trust";

/** A card's chart: rows of bars, an optional drawing below them, and its
 * text alternative: a title and every number it draws. */
interface ChartSpec {
  /** Names its title and description elements: unique on the page. */
  readonly id: string;
  readonly rows: readonly RowSpec[];
  readonly title: string;
  readonly alt: string;
  /** How much room the labels need beyond the longest bar, px. */
  readonly room: number;
  /** Draws below the bars, from `top`, `width` wide; returns its height. */
  readonly extra?: (group: SVGGElement, top: number, width: number) => number;
}

/** One card's chart, drawn at its box's width. */
class CardChart {
  private layout: BarsLayout | null = null;
  private root: SVGSVGElement | null = null;
  private bars: SVGRectElement[] = [];
  private labels: SVGTextElement[] = [];
  private drawn: boolean;

  constructor(
    private readonly plot: HTMLElement,
    private readonly spec: ChartSpec,
    private readonly reduced: () => boolean,
    private readonly frames: Frames | undefined,
    private readonly visible: () => boolean,
  ) {
    this.drawn = reduced();
    watchWidth(plot, (width) => this.draw(width));
    if (typeof IntersectionObserver === "function") {
      new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) this.maybeGrow();
      }, { rootMargin: `0px 0px ${LEAD_PX}px 0px` }).observe(plot);
    }
  }

  private draw(width: number): void {
    if (width <= 0) return;
    const layout = barsLayout(this.spec.rows, width, this.spec.room);
    this.layout = layout;
    const { id } = this.spec;
    const root = svg("svg", { class: "chart__svg card-chart", role: "img", "aria-labelledby": `${id}-chart-title ${id}-chart-desc` });
    svg("title", { id: `${id}-chart-title` }, root).textContent = this.spec.title;
    svg("desc", { id: `${id}-chart-desc` }, root).textContent = this.spec.alt;
    for (const segment of layout.zero) svg("line", { class: "card-chart__zero", x1: segment.x, x2: segment.x, y1: segment.y1, y2: segment.y2 }, root);
    for (const title of layout.titles) {
      const text = svg("text", { class: "card-chart__title", x: title.x, y: title.y }, root);
      text.textContent = title.text;
    }
    this.bars = layout.bars.map((bar) => svg("rect", { class: `card-chart__bar series--${bar.series}`, y: bar.y, height: bar.h, rx: 3 }, root));
    this.labels = layout.bars.map((bar) => svg("text", { class: "chart__value chart__value--small", x: bar.labelX, y: bar.y + bar.h - 1.5, "text-anchor": bar.anchor }, root));
    let height = layout.height;
    if (this.spec.extra) height += this.spec.extra(svg("g", {}, root), height + 10, width) + 10;
    attrs(root, { viewBox: `0 0 ${width} ${height}`, width, height });
    const old = this.plot.querySelector("svg");
    if (old) old.replaceWith(root);
    else this.plot.prepend(root);
    this.root = root;
    this.paint(() => 1);
    this.maybeGrow();
  }

  private paint(grow: (index: number) => number): void {
    const layout = this.layout;
    if (!layout) return;
    layout.bars.forEach((bar, i) => {
      const g = grow(i);
      const end = bar.x0 + (bar.x1 - bar.x0) * g;
      attrs(this.bars[i] as SVGRectElement, { x: Math.min(bar.x0, end), width: Math.abs(end - bar.x0) });
      const label = this.labels[i] as SVGTextElement;
      attrs(label, { x: bar.labelX + (end - bar.x1), opacity: g > 0 ? 1 : 0 });
      const units = bar.text.endsWith("%") ? bar.value * 100 : bar.value;
      label.textContent = textAt(bar.text, units, 0, g);
    });
  }

  maybeGrow(): void {
    if (this.drawn || !this.layout || !this.root || !this.visible()) return;
    if (!onScreen(this.plot, LEAD_PX)) return;
    this.drawn = true;
    const count = this.layout.bars.length;
    tween(GROW_MS, (p) => this.paint((i) => easeOut(local(p, (i / Math.max(1, count)) * 0.35, 0.6))), { reduced: this.reduced(), frames: this.frames });
  }
}

export class Cards {
  /** The cards in the page, or null on a page without them. */
  static mount(): Cards | null {
    const root = document.getElementById("results-view");
    return root?.querySelector('[data-card="frames"]') ? new Cards(root) : null;
  }

  private charts: CardChart[] = [];
  private onView: boolean;

  constructor(
    private readonly root: HTMLElement,
    private readonly reduced: () => boolean = prefersReducedMotion,
    private readonly frames?: Frames,
  ) {
    this.onView = document.documentElement.dataset["view"] === "results";
  }

  private part<T extends HTMLElement = HTMLElement>(name: string): T {
    const element = this.root.querySelector<T>(`[data-card="${name}"]`);
    if (!element) throw new Error(`the cards have no ${name}`);
    return element;
  }

  viewShown(view: ViewName): void {
    this.onView = view === "results";
    for (const chart of this.charts) chart.maybeGrow();
  }

  private chart(name: string, spec: ChartSpec): void {
    this.charts.push(new CardChart(this.part(`${name}-plot`), spec, this.reduced, this.frames, () => this.onView));
  }

  /** What the second frame buys: SightlineNet with two frames and with one,
   * on another day and on held-back stretches of its own drive. */
  showFrames(ladder: Ladder): void {
    const text = twoFramesText(ladder);
    if (!text) throw new Error("the ladder has no rung that holds both SightlineNets");
    this.part("frames-lead").textContent = text.lead;
    this.part("frames-also").textContent = text.also;
    const rows: RowSpec[] = [];
    for (const kind of ["another-day", "blocked"] as const) {
      const i = ladder.rungs.findIndex((rung) => rung.kind === kind);
      const two = ladder.bars.sightlinenet[i];
      const one = ladder.bars["sightlinenet-1f"][i];
      if (i < 0 || !two || !one) continue;
      rows.push({
        title: kind === "another-day" ? "Another day’s drive" : "Held-back stretches of its own drive",
        bars: [
          { series: "sightlinenet", value: two.skill.mean, text: fmtPct(two.skill.mean) },
          { series: "oneframe", value: one.skill.mean, text: fmtPct(one.skill.mean) },
        ],
      });
    }
    const alt =
      "How much better than always steering straight SightlineNet is with two frames and with one: " +
      rows.map((row) => `${row.title}: two frames ${skillWords(row.bars[0]?.value ?? 0)}, one frame ${skillWords(row.bars[1]?.value ?? 0)}`).join("; ") +
      ".";
    this.chart("frames", { id: "frames", rows, title: "SightlineNet with two frames and with one, against always steering straight", alt, room: 48 });
    this.part("frames").dataset["state"] = "ready";
  }

  /** Where it struggles: the shipped model and steering straight on each regime. */
  showRegimes(regimes: Regimes): void {
    const text = regimesText(regimes);
    this.part("regimes-lead").textContent = text.lead;
    this.part("regimes-straight").textContent = text.straight;
    const names = regimeNames(regimes.edges);
    const rows: RowSpec[] = REGIMES.map((regime) => ({
      title: `${names[regime]} (${fmtCount(regimes.frames[regime])} frames)`,
      bars: [
        { series: "sightlinenet", value: regimes.models.sightlinenet.errors[regime], text: fmtDeg(regimes.models.sightlinenet.errors[regime]) },
        { series: "straight", value: regimes.straight[regime], text: fmtDeg(regimes.straight[regime]) },
      ],
    }));
    this.chart("regimes", { id: "regimes", rows, title: "Mean error by how hard the driver was turning", alt: regimesAlt(regimes), room: 48 });
    this.part("regimes").dataset["state"] = "ready";
  }

  /** The standard-recipe PilotNet, and how far its output can turn the wheel. */
  showRecipe(legacy: Legacy): void {
    const text = legacyText(legacy);
    this.part("recipe-lead").textContent = text.lead;
    this.part("recipe-other").textContent = text.other;
    this.part("recipe-cap").textContent = text.cap;
    this.part("recipe-recipe").textContent = recipeNote(legacy.recipe);
    const other = driveTitle(legacy.blocked.drive);
    const rows: RowSpec[] = [
      [`Its own training drive (${fmtCount(legacy.own.frames)} frames)`, legacy.own],
      [`Held-back stretches of ${other} (${fmtCount(legacy.blocked.frames)} frames)`, legacy.blocked],
      [`The last part of ${other} (${fmtCount(legacy.later.frames)} frames)`, legacy.later],
    ].map(([title, score]) => {
      const s = score as Legacy["own"];
      return {
        title: title as string,
        bars: [
          { series: "pilotnet", value: s.model, text: fmtDeg(s.model) },
          { series: "straight", value: s.straight, text: fmtDeg(s.straight) },
        ],
      };
    });
    const alt = `${legacyAlt(legacy)} Its largest answer is ${fmtDeg(legacy.largest)}; its output can never pass ${exactDeg(legacy.cap)}, and the driver turned past that on ${fmtCount(legacy.beyond)} frames of ${other}.`;
    this.chart("recipe", {
      id: "recipe",
      rows,
      title: "The standard-recipe PilotNet against always steering straight, and its largest answer against its limit",
      alt,
      room: 48,
      extra: (group, top, width) => reach(group, top, width, legacy),
    });
    this.part("recipe").dataset["state"] = "ready";
  }

  /** Where the numbers come from: the export check and the split rules. */
  showTrust(parity: { sightline: ExportCheck; shuffled: ExportCheck }, splits: Splits): void {
    this.part("export").textContent = exportText(parity);
    const texts = protocolTexts(splits);
    this.part("protocol-shuffled").textContent = texts.shuffled;
    this.part("protocol-blocked").textContent = texts.blocked;
    this.part("protocol-later").textContent = texts.later;
    this.part("protocol-another-day").textContent = texts.anotherDay;
    this.part("trust").dataset["state"] = "ready";
  }

  /** The live check, as the band's chip reads it. */
  live(look: ParityLook, text: string): void {
    const line = this.part("live");
    if (line.textContent !== text) line.textContent = text;
    line.dataset["state"] = look;
  }

  /** Say why a card has no numbers, in place of them. */
  fail(card: CardName, error: unknown): void {
    const lead = card === "trust" ? "export" : `${card}-lead`;
    this.part(lead).textContent = measurementsFailure(error);
    this.part(card).dataset["state"] = "unavailable";
  }
}

/** The standard recipe's reach: its largest answer against the limit of its
 * arctangent output, and the frames where the driver turned past that limit.
 * Drawn into `group` from `top`; returns its height. */
function reach(group: SVGGElement, top: number, width: number, legacy: Legacy): number {
  const wall = width - 92;
  const x = (angle: number) => 2 + (angle / legacy.cap) * (wall - 2);
  const title = svg("text", { class: "card-chart__title", x: 0, y: top + 13 }, group);
  title.textContent = "Its largest answer, against its limit";
  const y = top + 38;
  const limit = svg("text", { class: "chart__value chart__value--small", x: wall + 1, y: y - 9, "text-anchor": "middle" }, group);
  limit.textContent = `limit ${exactDeg(legacy.cap)}`;
  svg("rect", { class: "card-chart__track", x: 2, y, width: wall - 2, height: 12, rx: 3 }, group);
  svg("rect", { class: "card-chart__bar series--pilotnet", x: 2, y, width: Math.max(0, x(legacy.largest) - 2), height: 12, rx: 3 }, group);
  svg("rect", { class: "card-chart__beyond", x: wall + 5, y, width: width - wall - 7, height: 12, rx: 3 }, group);
  svg("line", { class: "card-chart__wall", x1: wall + 1, x2: wall + 1, y1: y - 4, y2: y + 16 }, group);
  const largest = svg("text", { class: "chart__value chart__value--small", x: 2, y: y + 29 }, group);
  largest.textContent = `largest answer ${fmtDeg(legacy.largest)}`;
  const beyond = svg("text", { class: "chart__note", x: wall + 8, y: y + 29 }, group);
  lines(beyond, [`${fmtCount(legacy.beyond)} frames`, "turned past it"], wall + 8, 14);
  return 86;
}
