/**
 * The demo view's accuracy figure: Sightline's error on minutes of driving it
 * never trained on, against always steering straight on the same frames, and
 * what testing it the usual way would have said instead.
 *
 * The page is served with plain sentences and the bars waiting; once
 * results.json has loaded the sentences carry their numbers and the bars
 * their lengths, at once and with no interaction. The first time the figure
 * nears the screen the bars grow into place from zero, unless the visitor
 * prefers reduced motion; a visitor who never scrolls to it, or a screenshot
 * of the whole page, finds them already drawn.
 */

import { fmtDeg } from "../format.ts";
import { accuracyCaption, accuracyLabel, accuracySentence, leakageSentence, measurementsFailure } from "./copy.ts";
import type { Figures } from "./figures.ts";
import { EASE_OUT, onScreen, prefersReducedMotion } from "./motion.ts";

/** How far below the screen the figure starts growing, px. */
const LEAD_PX = 120;

export class Accuracy {
  /** The figure in the page, or null on a page without it. */
  static mount(): Accuracy | null {
    const section = document.getElementById("accuracy");
    return section ? new Accuracy(section) : null;
  }

  private readonly root: HTMLElement;
  private grown = false;

  constructor(
    section: HTMLElement,
    private readonly reduced: () => boolean = prefersReducedMotion,
  ) {
    const root = section.querySelector<HTMLElement>('[data-accuracy="root"]');
    if (!root) throw new Error("the accuracy figure's markup is missing");
    this.root = root;
  }

  private part<T extends HTMLElement = HTMLElement>(name: string): T {
    const element = this.root.querySelector<T>(`[data-accuracy="${name}"]`);
    if (!element) throw new Error(`the accuracy figure has no ${name}`);
    return element;
  }

  /** Write the figures in. */
  show(figures: Figures): void {
    const { heldOut } = figures;
    this.part("sentence").textContent = accuracySentence(heldOut);
    this.part("caption").textContent = accuracyCaption(heldOut);
    this.part("plot").setAttribute("aria-label", accuracyLabel(heldOut));
    this.part("leak").textContent = leakageSentence(figures.paired.sightlinenet.ratio.mean);
    // One scale for both bars, from zero to the larger error.
    const top = Math.max(heldOut.model, heldOut.straight);
    const bars: Array<[string, number]> = [
      ["straight", heldOut.straight],
      ["model", heldOut.model],
    ];
    for (const [name, value] of bars) {
      this.part(`bar-${name}`).style.setProperty("--share", String(top > 0 ? value / top : 0));
      this.part(`value-${name}`).textContent = fmtDeg(value);
    }
    this.root.dataset["state"] = "ready";
    this.watch();
  }

  /** Say why there are no figures, in place of them. */
  fail(error: unknown): void {
    this.part("sentence").textContent = measurementsFailure(error);
    this.root.dataset["state"] = "unavailable";
  }

  /** Grow the bars once, when the figure first nears the screen. */
  private watch(): void {
    if (this.reduced() || typeof IntersectionObserver !== "function") return;
    const figure = this.part("figure");
    if (onScreen(figure)) {
      // Already in sight: grow now, before the drawn bars are ever painted.
      this.grow();
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          observer.disconnect();
          this.grow();
        }
      },
      { rootMargin: `0px 0px ${LEAD_PX}px 0px` },
    );
    observer.observe(figure);
  }

  private grow(): void {
    if (this.grown) return;
    this.grown = true;
    ["straight", "model"].forEach((name, i) => {
      this.part(`bar-${name}`).animate?.([{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }], {
        duration: 1100,
        delay: 120 + i * 160,
        easing: EASE_OUT,
        fill: "backwards",
      });
      this.part(`value-${name}`).animate?.([{ opacity: 0 }, { opacity: 1 }], {
        duration: 400,
        delay: 700 + i * 160,
        easing: "ease-out",
        fill: "backwards",
      });
    });
  }
}
