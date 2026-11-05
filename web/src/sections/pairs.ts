/**
 * The frame next door: the leakage report's example test frames, each beside
 * the training frame that looks most like it under the shuffled split and
 * under the blocked split, with how far apart they were taken, how alike they
 * look and what the driver steered on each.
 *
 * The first pair is on screen the moment the data is in; the thumbnails pick
 * another, and its pictures settle in with a short fade (instant under reduced
 * motion, and when the pick came from the keyboard, where an animation only
 * slows the next key). Under the panels the test frame lies over its shuffled
 * neighbour, with a slider between them; the first time it nears the screen
 * the slider sweeps across once by itself, to show how little changes, then
 * rests in the middle for the visitor to drag. Every number is the data's.
 */

import { easeInOut, local, tween } from "../charts/draw.ts";
import type { ViewName } from "../router.ts";
import { gapWords, measurementsFailure, neighbourWords, pairAlt, pairsLead, pickerLegend, testFrameWords } from "./copy.ts";
import type { Frames } from "./countup.ts";
import type { FramePair, FramePairs } from "./figures.ts";
import { fmtCount, fmtFixed } from "../format.ts";
import { onScreen, prefersReducedMotion } from "./motion.ts";

const SPLITS = ["test", "shuffled", "blocked"] as const;
type Panel = (typeof SPLITS)[number];

/** The picture fade, ms, and the strong ease-out it settles with. */
const FADE_MS = 220;
const FADE_EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
/** The slider's one sweep, ms. */
const SWEEP_MS = 1800;
/** A pick this soon after a press on the thumbnails came from the pointer. */
const POINTER_MS = 800;
const LEAD_PX = 60;

export class PairsSection {
  /** The section in the page, or null on a page without it. */
  static mount(): PairsSection | null {
    const root = document.querySelector<HTMLElement>('[data-pairs="root"]');
    return root ? new PairsSection(root) : null;
  }

  private data: FramePairs | null = null;
  private folder: URL | null = null;
  private picked = 0;
  private pressed = -Infinity;
  private swept: boolean;
  private onView: boolean;
  private readonly photos = new Map<Panel, HTMLImageElement>();
  private readonly under: HTMLImageElement;
  private readonly over: HTMLImageElement;

  constructor(
    private readonly root: HTMLElement,
    private readonly reduced: () => boolean = prefersReducedMotion,
    private readonly frames?: Frames,
  ) {
    this.onView = document.documentElement.dataset["view"] === "results";
    this.swept = reduced();
    const picker = this.part("picker");
    picker.addEventListener("pointerdown", () => {
      this.pressed = performance.now();
    });
    picker.addEventListener("change", (event) => {
      const value = Number((event.target as HTMLInputElement).value);
      if (Number.isInteger(value)) this.select(value, performance.now() - this.pressed < POINTER_MS);
    });
    const range = this.part<HTMLInputElement>("compare-range");
    range.addEventListener("input", () => {
      this.swept = true;
      this.setCompare(Number(range.value));
    });
    this.under = document.createElement("img");
    this.over = document.createElement("img");
    for (const img of [this.under, this.over]) {
      img.decoding = "async";
      img.alt = "";
    }
    if (typeof IntersectionObserver === "function") {
      new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) this.maybeSweep();
      }, { rootMargin: `0px 0px -${LEAD_PX}px 0px` }).observe(this.part("compare"));
    }
  }

  private part<T extends HTMLElement = HTMLElement>(name: string): T {
    const element = this.root.querySelector<T>(`[data-pairs="${name}"]`);
    if (!element) throw new Error(`the frame pairs have no ${name}`);
    return element;
  }

  /** Write the pairs in: the thumbnails, the first pair's panels and the
   * table. `url` is the pairs file's own address, its images beside it;
   * `aspect` the camera frame's width over its height. */
  show(pairs: FramePairs, url: string, aspect: number | null): void {
    this.data = pairs;
    this.folder = new URL("./", url);
    if (aspect !== null) this.root.style.setProperty("--photo-aspect", String(aspect));
    this.part("lead").textContent = pairsLead(pairs);
    this.buildPicker(pairs);
    for (const split of SPLITS) {
      const img = document.createElement("img");
      img.decoding = "async";
      this.photos.set(split, img);
      this.part(`photo-${split}`).replaceChildren(img);
    }
    this.part("compare-under").replaceChildren(this.under);
    this.part("compare-over").replaceChildren(this.over);
    // The scale every pixel difference is drawn on: the most unlike pair.
    const largest = Math.max(...pairs.pairs.flatMap((pair) => [pair.shuffled.pixelRmse, pair.blocked.pixelRmse]));
    this.root.style.setProperty("--likeness-scale", String(largest));
    const restored = Number(this.root.querySelector<HTMLInputElement>('input[name="pairs-frame"]:checked')?.value ?? 0);
    this.select(Number.isInteger(restored) && restored >= 0 && restored < pairs.pairs.length ? restored : 0, false);
    this.setCompare(Number(this.part<HTMLInputElement>("compare-range").value));
    this.fillTable(pairs);
    this.root.dataset["state"] = "ready";
    this.maybeSweep();
  }

  fail(error: unknown): void {
    this.part("lead").textContent = measurementsFailure(error);
    this.root.dataset["state"] = "unavailable";
  }

  viewShown(view: ViewName): void {
    this.onView = view === "results";
    this.maybeSweep();
  }

  private image(name: string): string {
    return new URL(name, this.folder ?? document.baseURI).href;
  }

  private buildPicker(pairs: FramePairs): void {
    const picker = this.part("picker");
    const legend = picker.querySelector("legend");
    if (legend) legend.textContent = pickerLegend(pairs);
    const thumbs = pairs.pairs.map((pair, i) => {
      const label = document.createElement("label");
      label.className = "pairs__thumb";
      const input = document.createElement("input");
      input.className = "pairs__input";
      input.type = "radio";
      input.name = "pairs-frame";
      input.value = String(i);
      input.checked = i === this.picked;
      const img = document.createElement("img");
      img.className = "pairs__thumb-image";
      img.src = this.image(pair.image);
      img.alt = "";
      img.decoding = "async";
      img.loading = "lazy";
      const words = testFrameWords(pair);
      const name = document.createElement("span");
      name.className = "visually-hidden";
      name.textContent = `${words.frame}, steering `;
      const steer = document.createElement("span");
      steer.className = "pairs__thumb-label";
      steer.textContent = words.steering.replace(/^Steering /, "");
      label.append(input, img, name, steer);
      return label;
    });
    picker.replaceChildren(...(legend ? [legend] : []), ...thumbs);
  }

  /** Show pair `index`; its pictures fade in when the pick came from the pointer. */
  private select(index: number, animate: boolean): void {
    const pairs = this.data;
    const pair = pairs?.pairs[index];
    if (!pairs || !pair) return;
    this.picked = index;
    const input = this.root.querySelector<HTMLInputElement>(`input[name="pairs-frame"][value="${index}"]`);
    if (input && !input.checked) input.checked = true;
    const images: Record<Panel, string> = { test: pair.image, shuffled: pair.shuffled.image, blocked: pair.blocked.image };
    const alts: Record<Panel, string> = {
      test: `Test frame ${fmtCount(pair.frame)}: the road ahead`,
      shuffled: `The shuffled split’s closest training frame, ${neighbourWords(pair, "shuffled").where}`,
      blocked: `The blocked split’s closest training frame, ${neighbourWords(pair, "blocked").where}`,
    };
    for (const split of SPLITS) {
      const img = this.photos.get(split);
      if (!img) continue;
      img.src = this.image(images[split]);
      img.alt = alts[split];
      this.facts(split, pair);
      if (animate && !this.reduced()) {
        this.part(`photo-${split}`).animate?.(
          [
            { opacity: 0.25, filter: "blur(3px)", transform: "scale(0.985)" },
            { opacity: 1, filter: "blur(0)", transform: "none" },
          ],
          { duration: FADE_MS, easing: FADE_EASE },
        );
      }
    }
    this.part("alt").textContent = pairAlt(pair);
    this.under.src = this.image(pair.image);
    this.over.src = this.image(pair.shuffled.image);
    this.part("compare-text").textContent =
      `Test frame ${fmtCount(pair.frame)} and the shuffled split’s closest training frame, ${neighbourWords(pair, "shuffled").where}: ` +
      `pixel difference ${fmtFixed(pair.shuffled.pixelRmse)}.`;
  }

  /** A panel's facts: where its frame is, how alike it looks, what the
   * driver steered, and the likeness as a bar on the pairs' shared scale. */
  private facts(split: Panel, pair: FramePair): void {
    const list = this.part(`facts-${split}`);
    const item = (text: string, className = "") => {
      const li = document.createElement("li");
      if (className) li.className = className;
      li.textContent = text;
      return li;
    };
    if (split === "test") {
      const words = testFrameWords(pair);
      list.replaceChildren(item(words.frame), item(words.steering));
      return;
    }
    const words = neighbourWords(pair, split);
    const meter = document.createElement("li");
    meter.className = "pair-meter";
    meter.setAttribute("aria-hidden", "true");
    const bar = document.createElement("span");
    bar.className = `pair-meter__bar pair-meter__bar--${split}`;
    bar.style.setProperty("--likeness", String(pair[split].pixelRmse));
    meter.append(bar);
    list.replaceChildren(item(words.where, "pair-panel__where"), item(words.likeness), meter, item(words.steering));
  }

  /** The slider at `value` (0 to 100): the neighbour shows to its right. */
  private setCompare(value: number): void {
    const v = Math.min(100, Math.max(0, value));
    this.part("compare-over").style.clipPath = `inset(0 0 0 ${v}%)`;
    this.part("compare-handle").style.left = `${v}%`;
    const range = this.part<HTMLInputElement>("compare-range");
    if (Number(range.value) !== Math.round(v)) range.value = String(Math.round(v));
    range.setAttribute("aria-valuetext", `${fmtCount(Math.round(100 - v))}% of the picture shows the shuffled neighbour`);
  }

  /** The slider's one sweep, the first time it nears the screen in the view:
   * out to one side, across to the other, back to the middle. */
  private maybeSweep(): void {
    if (this.swept || !this.data || !this.onView) return;
    if (!onScreen(this.part("compare"), -LEAD_PX)) return;
    this.swept = true;
    const start = Number(this.part<HTMLInputElement>("compare-range").value);
    const reach = 28;
    tween(
      SWEEP_MS,
      (p) => {
        // Three legs, each eased: to start - reach, to start + reach, home.
        const leg = (from: number, to: number, at: number, length: number) => from + (to - from) * easeInOut(local(p, at, length));
        const v = p < 0.3 ? leg(start, start - reach, 0, 0.3) : p < 0.75 ? leg(start - reach, start + reach, 0.3, 0.45) : leg(start + reach, start, 0.75, 0.25);
        this.setCompare(v);
      },
      { reduced: this.reduced(), frames: this.frames },
    );
  }

  private fillTable(pairs: FramePairs): void {
    const table = this.part<HTMLTableElement>("table");
    const caption = table.querySelector("caption");
    table.replaceChildren(...(caption ? [caption] : []));
    const head = table.createTHead().insertRow();
    for (const heading of ["Test frame", "Steering", "Split", "Closest training frame", "Frames apart", "Time apart", "Pixel difference", "Its steering"]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = heading;
      head.append(cell);
    }
    const body = table.createTBody();
    for (const pair of pairs.pairs) {
      for (const [split, name] of [["shuffled", "Shuffled"], ["blocked", "Blocked"]] as const) {
        const side = pair[split];
        const row = body.insertRow();
        const th = document.createElement("th");
        th.scope = "row";
        th.textContent = fmtCount(pair.frame);
        row.append(th);
        row.insertCell().textContent = testFrameWords(pair).steering.replace("Steering ", "");
        row.insertCell().textContent = name;
        row.insertCell().textContent = fmtCount(side.frame);
        row.insertCell().textContent = fmtCount(side.gapFrames);
        row.insertCell().textContent = gapWords(side.gapS);
        row.insertCell().textContent = fmtFixed(side.pixelRmse);
        row.insertCell().textContent = neighbourWords(pair, split).steering.replace("Steering ", "");
      }
    }
  }
}
