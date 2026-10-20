/**
 * The Results view's cards: the baselines every finding is measured against,
 * what the extra frames buy, where the model struggles, the standard-recipe
 * PilotNet, and where every number comes from.
 *
 * Each card reads its own part of results.json, so a malformed part costs
 * only its card. Each card's drawing (see drawings.ts) is drawn at its values
 * the moment the data is in, and draws itself in once, the first time the
 * card nears the screen in the Results view (instantly under reduced motion).
 * The trust card repeats the live check the headline band runs, without a
 * second announcement: only the band's verdict speaks to a screen reader.
 */

import { attrs, easeOut, svg, tween, watchWidth } from "../charts/draw.ts";
import type { ViewName } from "../router.ts";
import {
  driveTitle,
  exactDeg,
  exportText,
  frameWords,
  framesAlt,
  framesTitleOf,
  legacyAlt,
  legacyText,
  measurementsFailure,
  protocolTexts,
  recipeNote,
  regimesAlt,
  regimesText,
  twoFramesText,
  yardsticksAlt,
  yardsticksVerdict,
  type ParityLook,
} from "./copy.ts";
import type { Frames } from "./countup.ts";
import { framesDrawing, recipeDrawing, regimesDrawing, yardsticksDrawing, type CardDrawing, type FramesRow } from "./drawings.ts";
import type { ExportCheck, FrameCounts, Ladder, Legacy, Regimes, Splits, YardstickSet } from "./figures.ts";
import { fmtCount, fmtDeg } from "../format.ts";
import { onScreen, prefersReducedMotion } from "./motion.ts";

const GROW_MS = 1100;
const LEAD_PX = 60;

export type CardName = "baselines" | "frames" | "regimes" | "recipe" | "trust";

/** A card's drawing and its text alternative: a title and every number it draws. */
interface ChartSpec {
  /** Names its title and description elements: unique on the page. */
  readonly id: string;
  readonly title: string;
  readonly alt: string;
  readonly draw: (group: SVGGElement, width: number) => CardDrawing;
}

/** One card's drawing, at its box's width. */
class CardChart {
  private drawing: CardDrawing | null = null;
  private drawn: boolean;
  private stop: () => void = () => undefined;

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
    this.stop();
    const { id } = this.spec;
    const root = svg("svg", { class: "chart__svg card-chart", role: "img", "aria-labelledby": `${id}-chart-title ${id}-chart-desc` });
    svg("title", { id: `${id}-chart-title` }, root).textContent = this.spec.title;
    svg("desc", { id: `${id}-chart-desc` }, root).textContent = this.spec.alt;
    const drawing = this.spec.draw(svg("g", {}, root), width);
    attrs(root, { viewBox: `0 0 ${width} ${drawing.height}`, width, height: drawing.height });
    const old = this.plot.querySelector("svg");
    if (old) old.replaceWith(root);
    else this.plot.prepend(root);
    this.drawing = drawing;
    drawing.paint(1);
    this.maybeGrow();
  }

  maybeGrow(): void {
    const drawing = this.drawing;
    if (this.drawn || !drawing || !this.visible()) return;
    if (!onScreen(this.plot, LEAD_PX)) return;
    this.drawn = true;
    this.stop = tween(GROW_MS, (p) => drawing.paint(easeOut(p)), { reduced: this.reduced(), frames: this.frames });
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

  /** The baselines: every rule beside Sightline, on the blocked split's test
   * frames and on another day's drive. */
  showBaselines(sets: readonly YardstickSet[]): void {
    this.part("baselines-verdict").textContent = yardsticksVerdict(sets);
    this.chart("baselines", {
      id: "baselines",
      title: "Mean error of each baseline and of Sightline, on the same frames",
      alt: yardsticksAlt(sets),
      draw: (group, width) => yardsticksDrawing(group, width, sets),
    });
    this.part("baselines").dataset["state"] = "ready";
  }

  /** What the extra frames buy: SightlineNet with more frames and with fewer,
   * on another day and on held-back stretches of its own drive. */
  showFrames(ladder: Ladder, frames: FrameCounts | null): void {
    const text = twoFramesText(ladder, frames);
    if (!text) throw new Error("the ladder has no rung that holds both SightlineNets");
    this.part("frames-title").textContent = framesTitleOf(frames);
    this.part("frames-lead").textContent = text.lead;
    this.part("frames-also").textContent = text.also;
    const rows: FramesRow[] = [];
    for (const kind of ["another-day", "blocked"] as const) {
      const i = ladder.rungs.findIndex((rung) => rung.kind === kind);
      const more = ladder.bars.sightlinenet[i];
      const fewer = ladder.bars["sightlinenet-1f"][i];
      if (i < 0 || !more || !fewer) continue;
      rows.push({ title: kind === "another-day" ? "Another day’s drive" : "Held-back stretches of its own drive", more: more.skill.mean, fewer: fewer.skill.mean });
    }
    const alt = framesAlt(rows, frames);
    const words = frameWords(frames);
    this.chart("frames", { id: "frames", title: alt.title, alt: alt.desc, draw: (group, width) => framesDrawing(group, width, rows, words) });
    this.part("frames").dataset["state"] = "ready";
  }

  /** Where it struggles: the shipped model and steering straight on each regime. */
  showRegimes(regimes: Regimes): void {
    const text = regimesText(regimes);
    this.part("regimes-lead").textContent = text.lead;
    this.part("regimes-straight").textContent = text.straight;
    this.chart("regimes", {
      id: "regimes",
      title: "Mean error by how hard the driver was turning",
      alt: regimesAlt(regimes),
      draw: (group, width) => regimesDrawing(group, width, regimes),
    });
    this.part("regimes").dataset["state"] = "ready";
  }

  /** The standard-recipe PilotNet: how far its output can turn the wheel, and
   * its errors beside steering straight's. */
  showRecipe(legacy: Legacy): void {
    const text = legacyText(legacy);
    this.part("recipe-lead").textContent = text.lead;
    this.part("recipe-other").textContent = text.other;
    this.part("recipe-cap").textContent = text.cap;
    this.part("recipe-recipe").textContent = recipeNote(legacy.recipe);
    const other = driveTitle(legacy.blocked.drive);
    const sets = [
      ["Its own training drive", legacy.own],
      [`Held-back stretches of ${other}`, legacy.blocked],
      [`The last part of ${other}`, legacy.later],
    ] as const;
    const alt =
      `${legacyAlt(legacy)} Its largest answer is ${fmtDeg(legacy.largest)}; its output can never pass ${exactDeg(legacy.cap)}, ` +
      `and the driver turned past that on ${fmtCount(legacy.beyond)} frames of ${other}.`;
    this.chart("recipe", {
      id: "recipe",
      title: "The standard-recipe PilotNet: its largest answer against its limit, and its error against always steering straight",
      alt,
      draw: (group, width) => recipeDrawing(group, width, legacy, sets),
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
