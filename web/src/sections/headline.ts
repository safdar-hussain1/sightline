/**
 * The Results view's headline band: the paired leakage ratio for both
 * networks, Sightline's skill on another day, the lookup table under both
 * splits, and a live check of this browser against Python.
 *
 * The figures count up by themselves, once, the first time the band is on
 * screen -- on a deep link to the results as on a switch to them -- and under
 * reduced motion they appear at their values at once. Each counts in its own
 * final text's shape, so it ends on exactly what the formatters wrote, and
 * assistive technology reads only that final text. Until results.json has
 * loaded the figures hold their place as quiet placeholders, never as zeros.
 *
 * The check reads the cockpit's own answers on the clip it has loaded, against
 * the clip's published ones, while the Results view is on screen.
 */

import type { LiveCockpit } from "../cockpit/live.ts";
import { fmtDeg, fmtRatio } from "../format.ts";
import type { ViewName } from "../router.ts";
import { countText, runCounts, textAt, type Count, type Frames } from "./countup.ts";
import {
  anotherDayFigure,
  anotherDayNote,
  anotherDayRuler,
  measurementsFailure,
  memoriserNote,
  pairedNote,
  pairedRuler,
  parityText,
  resultsLede,
  splitsText,
} from "./copy.ts";
import type { Figures } from "./figures.ts";
import { onScreen, prefersReducedMotion } from "./motion.ts";

const COUNT_MS = 1500;
/** How often the live check is read while the view is on screen, ms. */
const CHECK_MS = 600;

/** A figure's final text, its value in the units the text shows, and where
 * its count starts: a ratio from one (no flattery), the rest from zero. A
 * figure with a ruler moves one of its bars in step: `bar` goes from the
 * length `from` to the length `to` (degrees) as the count runs. */
interface Target {
  readonly name: string;
  readonly final: string;
  readonly value: number;
  readonly from: number;
  readonly delay: number;
  readonly bar?: { readonly part: string; readonly from: number; readonly to: number };
}

export class Headline {
  /** The band in the page, or null on a page without it. */
  static mount(): Headline | null {
    const section = document.getElementById("results-view");
    return section ? new Headline(section) : null;
  }

  private readonly root: HTMLElement;
  private targets: Target[] = [];
  /** The rulers' shared scale: the largest error any of them draws. */
  private scale = 1;
  private counted = false;
  private onView = false;
  private cockpit: LiveCockpit | null | undefined = undefined;
  private checking = 0;

  /** `frames` is the clock the counts run on: the page's own unless a test
   * passes one. */
  constructor(
    section: HTMLElement,
    private readonly reduced: () => boolean = prefersReducedMotion,
    private readonly frames?: Frames,
  ) {
    const root = section.querySelector<HTMLElement>('[data-results="root"]');
    if (!root) throw new Error("the Results view's markup is missing");
    this.root = root;
    this.onView = document.documentElement.dataset["view"] === "results";
    if (typeof IntersectionObserver === "function") {
      new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) this.maybeCount();
      }).observe(this.part("band"));
    }
    document.addEventListener("visibilitychange", () => this.check());
  }

  private part<T extends HTMLElement = HTMLElement>(name: string): T {
    const element = this.root.querySelector<T>(`[data-results="${name}"]`);
    if (!element) throw new Error(`the Results view has no ${name}`);
    return element;
  }

  private count(name: string): HTMLElement {
    const element = this.root.querySelector<HTMLElement>(`[data-count="${name}"]`);
    if (!element) throw new Error(`the Results view has no figure ${name}`);
    return element;
  }

  /** Write the figures in, and count them up if the band is on screen. */
  show(figures: Figures): void {
    const { paired, anotherDay, memoriser, gaps, runs } = figures;
    const day = anotherDayFigure(anotherDay);
    this.part("lede").textContent = resultsLede(runs);
    this.part("paired-note").textContent = pairedNote(paired);
    this.part("another-day-label").textContent = day.label;
    this.part("another-day-note").textContent = anotherDayNote(anotherDay);
    this.part("memoriser-note").textContent = memoriserNote(memoriser);
    const splits = splitsText(gaps);
    this.part("shuffled").textContent = splits.shuffled;
    this.part("blocked").textContent = splits.blocked;

    // The rulers: the errors behind the paired ratio and behind another
    // day's skill, every bar on one scale.
    const sightline = paired.sightlinenet;
    this.part("ruler-paired").setAttribute("aria-label", pairedRuler(sightline));
    this.part("ruler-another-day").setAttribute("aria-label", anotherDayRuler(anotherDay));
    const errors: Array<[string, number]> = [
      ["paired-shuffled", sightline.shuffled],
      ["paired-blocked", sightline.blocked],
      ["day-model", anotherDay.model],
      ["day-straight", anotherDay.straight],
    ];
    this.scale = Math.max(...errors.map(([, value]) => value)) || 1;
    for (const [part, value] of errors) {
      this.part(`value-${part}`).textContent = fmtDeg(value);
      this.bar(part, value);
    }

    this.targets = [
      {
        name: "ratio",
        final: fmtRatio(sightline.ratio.mean),
        value: sightline.ratio.mean,
        from: 1,
        delay: 0,
        // At a ratio of one the held-back bar is the shuffled one's length.
        bar: { part: "paired-blocked", from: sightline.shuffled, to: sightline.blocked },
      },
      { name: "ratio-pilotnet", final: fmtRatio(paired.pilotnet.ratio.mean), value: paired.pilotnet.ratio.mean, from: 1, delay: 90 },
      {
        name: "another-day",
        final: day.text,
        value: day.value * 100,
        from: 0,
        delay: 150,
        // At no skill at all, Sightline's bar is steering straight's length.
        bar: { part: "day-model", from: anotherDay.straight, to: anotherDay.model },
      },
      { name: "memoriser-shuffled", final: fmtDeg(memoriser.shuffled), value: memoriser.shuffled, from: 0, delay: 210 },
      { name: "memoriser-blocked", final: fmtDeg(memoriser.blocked), value: memoriser.blocked, from: 0, delay: 260 },
    ];
    const still = this.reduced();
    for (const { name, final, value, from } of this.targets) {
      const el = this.count(name);
      // The final text sets the figure's width, and is what is read aloud.
      (el.querySelector(".count__final") as HTMLElement).textContent = final;
      (el.querySelector(".count__text") as HTMLElement).textContent = final;
      // Until it counts, the figure waits at the start of its count; it is
      // only ever seen there for the moment before the count begins.
      (el.querySelector(".count__shown") as HTMLElement).textContent = still ? final : countText(final, value, from, 0);
    }
    if (!still) {
      for (const { bar } of this.targets) if (bar) this.moveBar(bar, 0);
    }
    this.counted = still;
    this.root.dataset["state"] = "ready";
    this.maybeCount();
  }

  /** Say why there are no figures, in place of them. */
  fail(error: unknown): void {
    this.part("lede").textContent = measurementsFailure(error);
    this.root.dataset["state"] = "unavailable";
  }

  /** The cockpit, for the live check; null when the demo could not start. */
  attach(cockpit: LiveCockpit | null): void {
    this.cockpit = cockpit;
    this.check();
  }

  /** The router switched views. */
  viewShown(view: ViewName): void {
    this.onView = view === "results";
    this.maybeCount();
    this.check();
  }

  private maybeCount(): void {
    if (this.counted || this.targets.length === 0 || !this.onView) return;
    if (!onScreen(this.part("band"))) return;
    this.counted = true;
    const counts: Count[] = this.targets.map(({ name, final, value, from, delay, bar }) => ({
      el: this.count(name).querySelector(".count__shown") as HTMLElement,
      final,
      value,
      from,
      delay,
      ...(bar ? { onProgress: (eased: number) => this.moveBar(bar, eased) } : {}),
    }));
    runCounts(counts, { duration: COUNT_MS, reduced: this.reduced(), ...(this.frames ? { frames: this.frames } : {}) });
  }

  /** A ruler's moving bar at `eased` of its way, its value at its tip. */
  private moveBar(bar: NonNullable<Target["bar"]>, eased: number): void {
    this.bar(bar.part, bar.from + (bar.to - bar.from) * eased);
    this.part(`value-${bar.part}`).textContent = textAt(fmtDeg(bar.to), bar.to, bar.from, eased);
  }

  /** Draw a ruler's bar `degrees` long, on the band's scale. */
  private bar(part: string, degrees: number): void {
    this.part(`bar-${part}`).style.setProperty("--share", String(Math.max(0, degrees) / this.scale));
  }

  /** Read the live check now, and again shortly while the view is on screen. */
  private check(): void {
    clearTimeout(this.checking);
    if (this.cockpit === undefined) return; // the cockpit's code is still on its way
    const cockpit = this.cockpit;
    const agreement = cockpit?.agreement() ?? null;
    const state = cockpit ? cockpit.status().state : "unavailable";
    const trust = this.part("trust");
    this.part("trust-text").textContent = parityText(agreement, state);
    const done =
      agreement !== null && agreement.check === "match" && agreement.compared === agreement.total && Number.isFinite(agreement.maxDiffDeg);
    trust.dataset["state"] =
      done ? "checked"
      : state === "unavailable" || (agreement !== null && agreement.compared > 0 && agreement.check !== "match") ? "note"
      : "checking";
    if (cockpit && this.onView && document.visibilityState !== "hidden") {
      this.checking = window.setTimeout(() => this.check(), CHECK_MS);
    }
  }
}
