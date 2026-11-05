// @vitest-environment happy-dom
/**
 * The Results view's studies as a visitor meets them: the paired experiment,
 * the ladder and the cards, once results.json has loaded (or has not). happy-dom
 * lays nothing out, so each test says how wide the charts are and whether they
 * are on screen, and the motion runs on a clock the test moves by hand.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ladderLayout } from "../charts/ladder.ts";
import { slopeLayout } from "../charts/slope.ts";
import { AssetError } from "../data/load.ts";
import { textWidth } from "../charts/marks.ts";
import { Cards } from "./cards.ts";
import {
  exportText,
  ladderAlt,
  ladderCaption,
  ladderLead,
  legacyText,
  measurementsFailure,
  pairedAlt,
  pairedFrames,
  pairedLead,
  pairedMeaning,
  pairedScale,
  protocolTexts,
  regimesAlt,
  regimesText,
  rungCaption,
  rungNote,
  splitsText,
  framesAlt,
  twoFramesText,
  yardsticksAlt,
  yardsticksVerdict,
} from "./copy.ts";
import type { Frames } from "./countup.ts";
import { readFigures, readFrames, readLadder, readLegacy, readPaired, readParity, readRegimes, readSplits, readYardsticks } from "./figures.ts";
import { patch, resultsFixture } from "./fixture.ts";
import { LadderSection } from "./ladder.ts";
import { resultsHtml } from "./results-markup.ts";
import { PairedSection } from "./paired.ts";

const raw = () => resultsFixture();
const q = <T extends Element = HTMLElement>(selector: string) => document.querySelector<T>(selector) as T;
const all = (selector: string) => [...document.querySelectorAll<HTMLElement>(selector)];
const text = (selector: string) => q(selector).textContent ?? "";

/** Every element is `width` wide, on screen or not. */
function layoutAs(width: number, visible = true): void {
  const box = visible
    ? { x: 0, y: 100, top: 100, bottom: 500, left: 0, right: width, width, height: 400 }
    : { x: 0, y: 3000, top: 3000, bottom: 3400, left: 0, right: width, width, height: 400 };
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({ ...box, toJSON: () => box } as DOMRect);
}

/** A clock the test moves by hand, one animation frame at a time. */
function clock(): Frames & { tick(ms: number): void } {
  let now = 0;
  let queued: Array<(t: number) => void> = [];
  return {
    now: () => now,
    request: (callback) => queued.push(callback),
    cancel: () => void (queued = []),
    tick(ms: number) {
      now += ms;
      const run = queued;
      queued = [];
      for (const callback of run) callback(now);
    },
  };
}

let observers: Array<(entries: Array<{ isIntersecting: boolean }>) => void> = [];
const intersect = () => {
  for (const observe of observers) observe([{ isIntersecting: true }]);
};

function mountResults(): HTMLElement {
  const section = document.createElement("section");
  section.id = "results-view";
  section.innerHTML = resultsHtml();
  document.body.append(section);
  return section;
}

beforeEach(() => {
  document.body.innerHTML = "";
  document.documentElement.dataset["view"] = "results";
  observers = [];
  vi.stubGlobal(
    "IntersectionObserver",
    class {
      constructor(callback: (entries: Array<{ isIntersecting: boolean }>) => void) {
        observers.push(callback);
      }
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  vi.stubGlobal("ResizeObserver", undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  delete document.documentElement.dataset["view"];
});

// -- The paired experiment -----------------------------------------------------------------

describe("the paired experiment", () => {
  const mount = (reduced: boolean, frames = clock()) => {
    mountResults();
    return { section: new PairedSection(q('[data-paired="root"]'), () => reduced, frames), frames };
  };
  const lines = () => all(".slope__line");
  const num = (el: Element, name: string) => Number(el.getAttribute(name));

  it("writes its words, its table and its chart from the data, with no interaction", () => {
    layoutAs(600, false);
    const { section } = mount(false);
    const paired = readPaired(raw());
    section.show(paired, readFigures(raw()).gaps);
    const network = paired.sightlinenet;
    expect(text('[data-paired="lead"]')).toBe(pairedLead("SightlineNet", network));
    expect(text('[data-paired="meaning"]')).toBe(pairedMeaning("SightlineNet", network.ratio.mean));
    expect(text('[data-paired="scale"]')).toBe(pairedScale(network));
    expect(text('[data-paired="caption"]')).toBe(pairedFrames(network));
    const splits = splitsText(readFigures(raw()).gaps);
    expect([text('[data-paired="shuffled"]'), text('[data-paired="blocked"]')]).toEqual([splits.shuffled, splits.blocked]);
    expect(all('[data-paired="table"] tbody tr')).toHaveLength(6);
    expect(text('[data-paired="table"] tbody tr')).toBe("SightlineNet0893" + "3.0°11.2°3.7×17.3°");
    // The chart, final at once: its frame replaced, every seed drawn to its end.
    const svg = q<SVGSVGElement>('[data-paired="plot"] svg');
    expect(svg.getAttribute("role")).toBe("img");
    expect(svg.classList.contains("slope--waiting")).toBe(false);
    const alt = pairedAlt("SightlineNet", network);
    expect([text("#paired-svg-title"), text("#paired-svg-desc")]).toEqual([alt.title, alt.desc]);
    const layout = slopeLayout(network, [paired.sightlinenet, paired.pilotnet], 600);
    expect(lines()).toHaveLength(3);
    lines().forEach((line, i) => expect(num(line, "y2")).toBeCloseTo(layout.lines[i]?.y2 ?? NaN, 1));
    expect(all(".slope .chart__value").map((el) => el.textContent)).toEqual(["3.3°mean", "10.1°mean"]);
    expect(text(".slope__ratio")).toBe("3.0× as much, on average");
    expect(q('[data-paired="root"]').dataset["state"]).toBe("ready");
  });

  it("draws itself in once it nears the screen in the Results view: shuffled scores first, then each line rising", () => {
    layoutAs(600, false);
    const { section, frames } = mount(false);
    section.show(readPaired(raw()), readFigures(raw()).gaps);
    const final = lines().map((line) => num(line, "x2"));
    layoutAs(600, true);
    intersect();
    // The draw-in has started: every line still at its left end.
    for (const line of lines()) expect(num(line, "x2")).toBe(num(line, "x1"));
    expect(text(".slope__ratio")).toMatch(/^1\.0× /);
    frames.tick(700);
    const mid = lines().map((line) => num(line, "x2"));
    mid.forEach((x, i) => expect(x).toBeLessThan(final[i] ?? 0));
    expect(mid[0]).toBeGreaterThan(num(lines()[0] as Element, "x1"));
    frames.tick(2000);
    expect(lines().map((line) => num(line, "x2"))).toEqual(final);
    expect(all(".slope .chart__value").map((el) => el.textContent)).toEqual(["3.3°mean", "10.1°mean"]);
    // Once only.
    intersect();
    expect(lines().map((line) => num(line, "x2"))).toEqual(final);
  });

  it("does not draw itself in while its view is hidden behind the demo", () => {
    document.documentElement.dataset["view"] = "demo";
    layoutAs(600, true);
    const { section } = mount(false);
    section.show(readPaired(raw()), readFigures(raw()).gaps);
    intersect();
    for (const line of lines()) expect(num(line, "x2")).toBeGreaterThan(num(line, "x1"));
  });

  it("is final at once under reduced motion, and stays so", () => {
    layoutAs(600, true);
    const { section } = mount(true);
    section.show(readPaired(raw()), readFigures(raw()).gaps);
    intersect();
    section.viewShown("results");
    for (const line of lines()) expect(num(line, "x2")).toBeGreaterThan(num(line, "x1"));
  });

  it("morphs into PilotNet on the same scale when the visitor picks it, and says what PilotNet scored", () => {
    layoutAs(600, true);
    const { section, frames } = mount(true);
    const paired = readPaired(raw());
    section.show(paired, readFigures(raw()).gaps);
    const before = lines().map((line) => num(line, "y2"));
    const scale = all(".slope .chart__tick").map((el) => el.textContent);
    const radio = q<HTMLInputElement>('input[name="paired-network"][value="pilotnet"]');
    radio.checked = true;
    radio.dispatchEvent(new Event("change", { bubbles: true }));
    // Under reduced motion the morph is instant.
    const target = slopeLayout(paired.pilotnet, [paired.sightlinenet, paired.pilotnet], 600);
    lines().forEach((line, i) => expect(num(line, "y2")).toBeCloseTo(target.lines[i]?.y2 ?? NaN, 1));
    expect(lines().map((line) => num(line, "y2"))).not.toEqual(before);
    expect(all(".slope .chart__tick").map((el) => el.textContent)).toEqual(scale);
    expect(q("svg.slope").dataset["network"]).toBe("pilotnet");
    expect(text('[data-paired="lead"]')).toBe(pairedLead("PilotNet", paired.pilotnet));
    expect(text("#paired-svg-title")).toBe(pairedAlt("PilotNet", paired.pilotnet).title);
    expect(all(".slope .chart__value").map((el) => el.textContent)).toEqual(["4.1°mean", "13.1°mean"]);
    void frames;
  });

  it("morphs in motion: halfway is between the two networks, the end is PilotNet's", () => {
    layoutAs(600, true);
    const { section, frames } = mount(false);
    const paired = readPaired(raw());
    section.show(paired, readFigures(raw()).gaps);
    frames.tick(5000); // the draw-in ends
    const from = lines().map((line) => num(line, "y2"));
    const radio = q<HTMLInputElement>('input[name="paired-network"][value="pilotnet"]');
    radio.checked = true;
    radio.dispatchEvent(new Event("change", { bubbles: true }));
    frames.tick(280);
    const to = slopeLayout(paired.pilotnet, [paired.sightlinenet, paired.pilotnet], 600).lines.map((line) => line.y2);
    lines().forEach((line, i) => {
      const y = num(line, "y2");
      expect(y).toBeLessThan(Math.max(from[i] ?? 0, to[i] ?? 0));
      expect(y).toBeGreaterThan(Math.min(from[i] ?? 0, to[i] ?? 0));
    });
    frames.tick(1000);
    lines().forEach((line, i) => expect(num(line, "y2")).toBeCloseTo(to[i] ?? NaN, 1));
  });

  it("names a seed and its numbers under the pointer, and puts them away after", () => {
    layoutAs(600, true);
    const { section } = mount(true);
    const paired = readPaired(raw());
    section.show(paired, readFigures(raw()).gaps);
    const layout = slopeLayout(paired.sightlinenet, [paired.sightlinenet, paired.pilotnet], 600);
    const line = layout.lines[1];
    const plot = q('[data-paired="plot"]');
    const x = ((line?.x1 ?? 0) + (line?.x2 ?? 0)) / 2;
    const y = ((line?.y1 ?? 0) + (line?.y2 ?? 0)) / 2 + 100; // the box's top is at 100
    plot.dispatchEvent(new PointerEvent("pointermove", { clientX: x, clientY: y, bubbles: true }));
    expect(q('[data-paired="tip"]').hidden).toBe(false);
    expect(text('[data-paired="tip"]')).toBe(
      "Seed 1, 889 frames: 3.3° trained on shuffled frames, 8.6° with stretches held back (2.6×). Always steering straight: 16.2°.",
    );
    expect(lines()[1]?.classList.contains("is-lit")).toBe(true);
    plot.dispatchEvent(new PointerEvent("pointerleave", { bubbles: true }));
    expect(q('[data-paired="tip"]').hidden).toBe(true);
  });

  it("draws the network picked before the data arrived", () => {
    layoutAs(600, true);
    const { section } = mount(true);
    q<HTMLInputElement>('input[name="paired-network"][value="pilotnet"]').checked = true;
    section.show(readPaired(raw()), readFigures(raw()).gaps);
    expect(q("svg.slope").dataset["network"]).toBe("pilotnet");
    expect(all(".slope .chart__value").map((el) => el.textContent)).toEqual(["4.1°mean", "13.1°mean"]);
  });

  it("says why there is no chart when the measurements cannot be had", () => {
    const { section } = mount(false);
    const error = new AssetError("gone", "missing", "data/results.json");
    section.fail(error);
    expect(text('[data-paired="lead"]')).toBe(measurementsFailure(error));
    expect(q('[data-paired="root"]').dataset["state"]).toBe("unavailable");
  });
});

// -- The ladder -------------------------------------------------------------------------------

describe("the ladder", () => {
  const mount = (reduced: boolean, frames = clock()) => {
    mountResults();
    return { section: new LadderSection(q('[data-ladder="root"]'), () => reduced, frames), frames };
  };
  const bars = () => all(".ladder-chart__bar");
  const pick = (name: string, value: string) => {
    const radio = q<HTMLInputElement>(`input[name="${name}"][value="${value}"]`);
    radio.checked = true;
    radio.dispatchEvent(new Event("change", { bubbles: true }));
  };

  it("builds its stepper from the data's rungs, lights the furthest, and draws every bar at once", () => {
    layoutAs(1000, false);
    const { section } = mount(false);
    const ladder = readLadder(raw());
    const splits = readSplits(raw());
    section.show(ladder, splits);
    const steps = all(".stepper__step");
    expect(steps.map((step) => step.textContent)).toEqual(["shuffled", "blocked", "later in the drive", "another day"]);
    expect(q<HTMLInputElement>('input[name="ladder-rung"]:checked').value).toBe("3");
    expect(text('[data-ladder="caption-rung"]')).toBe("Another day");
    expect(text('[data-ladder="caption-text"]')).toBe(`${rungCaption(ladder.rungs[3] as never, splits)} ${rungNote(ladder, 3)}`);
    // Every rung's sentence sits unseen beside it, so stepping moves nothing.
    expect(all(".ladder__sizer")).toHaveLength(4);
    expect(all(".ladder__sizer").every((sizer) => sizer.getAttribute("aria-hidden") === "true")).toBe(true);
    expect(bars()).toHaveLength(10);
    expect(all(".ladder-chart__missing").map((el) => el.textContent)).toEqual(["not trainedthis way", "not trainedthis way"]);
    const alt = ladderAlt(ladder, "skill");
    expect([text("#ladder-svg-title"), text("#ladder-svg-desc")]).toEqual([alt.title, alt.desc]);
    expect(all('[data-ladder="table"] tbody tr')).toHaveLength(12);
    expect(all('[data-ladder="table"] td').filter((cell) => cell.textContent === "not trained this way")).toHaveLength(2);
    const labels = all(".ladder-chart__bars .chart__value").map((el) => el.textContent);
    expect(labels).toContain("−45%");
    expect(labels).toContain("78%");
  });

  it("draws PilotNet below the line where it does worse than steering straight", () => {
    layoutAs(1000, true);
    const { section } = mount(true);
    const ladder = readLadder(raw());
    section.show(ladder, readSplits(raw()));
    const layout = ladderLayout(ladder, "skill", 1000, 3);
    const later = layout.bars.find((bar) => bar.model === "pilotnet" && bar.rung === 2);
    expect(later?.end).toBeGreaterThan(later?.base ?? Infinity);
    const path = bars().find((el) => el.getAttribute("class")?.includes("series--pilotnet") && (el.getAttribute("d") ?? "").includes(`V${Math.round((later?.base ?? 0) * 100) / 100}Z`));
    expect(path).toBeDefined();
  });

  it("steps to a rung: the caption says what moved further away, and the light moves to its panel", () => {
    layoutAs(1000, true);
    const { section } = mount(true);
    const ladder = readLadder(raw());
    const splits = readSplits(raw());
    section.show(ladder, splits);
    const before = q<SVGRectElement>(".ladder-chart__spotlight").style.transform;
    pick("ladder-rung", "2");
    expect(text('[data-ladder="caption-rung"]')).toBe("Later in the drive");
    expect(text('[data-ladder="caption-text"]')).toBe(`${rungCaption(ladder.rungs[2] as never, splits)} ${rungNote(ladder, 2)}`);
    expect(q<SVGRectElement>(".ladder-chart__spotlight").style.transform).not.toBe(before);
  });

  it("grows its bars from the line the first time it nears the screen, numbers counting in step", () => {
    layoutAs(1000, false);
    const { section, frames } = mount(false);
    section.show(readLadder(raw()), readSplits(raw()));
    const final = bars().map((bar) => bar.getAttribute("d"));
    const finalLabels = all(".ladder-chart__bars .chart__value").map((el) => el.getAttribute("y"));
    layoutAs(1000, true);
    intersect();
    // Started: every bar flat on its line, every label at zero.
    expect(bars().every((bar) => /h[\d.]+$/.test(bar.getAttribute("d") ?? ""))).toBe(true);
    frames.tick(600);
    const growing = bars().map((bar) => bar.getAttribute("d"));
    expect(growing).not.toEqual(final);
    // Each label rides its growing bar: nearer the line than it will finally stand.
    const riding = all(".ladder-chart__bars .chart__value").map((el) => el.getAttribute("y"));
    expect(riding).not.toEqual(finalLabels);
    frames.tick(3000);
    expect(bars().map((bar) => bar.getAttribute("d"))).toEqual(final);
    expect(all(".ladder-chart__bars .chart__value").map((el) => el.getAttribute("y"))).toEqual(finalLabels);
  });

  it("morphs into raw error: degrees, each training as a tick, each rung's own steering-straight line", () => {
    layoutAs(1000, true);
    const { section, frames } = mount(false);
    const ladder = readLadder(raw());
    section.show(ladder, readSplits(raw()));
    frames.tick(5000);
    pick("ladder-view", "error");
    frames.tick(320);
    // Halfway: the labels are changing over, the bars between the views.
    frames.tick(2000);
    const labels = all(".ladder-chart__bars .chart__value").map((el) => el.textContent);
    expect(labels).toContain("13.6°");
    expect(labels).toContain("2.9°");
    expect(all(".ladder-chart__dot")).toEqual([]);
    expect(all(".ladder-chart__seed")).toHaveLength(30);
    expect(all(".ladder-chart__seed").every((tick) => tick.tagName.toLowerCase() === "path" && tick.getAttribute("opacity") === "1")).toBe(true);
    expect(all(".ladder-chart__whisker").every((w) => w.getAttribute("opacity") === "0")).toBe(true);
    const ticks = all(".ladder-chart__baselines .chart__note").map((el) => el.textContent);
    expect(ticks).toEqual(["13.1°", "17.5°", "9.4°", "15.5°"]);
    const alt = ladderAlt(ladder, "error");
    expect(text("#ladder-svg-title")).toBe(alt.title);
    expect(text('[data-ladder="chart-title"]')).toBe("Mean error, in degrees of steering-wheel angle");
    expect(text('[data-ladder="chart-caption"]')).toBe(ladderCaption(ladder, "error"));
  });

  it("keeps every training's tick visible where trainings scored nearly the same", () => {
    layoutAs(1000, true);
    const { section, frames } = mount(false);
    section.show(readLadder(raw()), readSplits(raw()));
    pick("ladder-view", "error");
    frames.tick(5000);
    // SightlineNet on the shuffled rung: 2.8°, 2.9° and 3.0°, under a pixel apart.
    const spans = all(".ladder-chart__seed")
      .slice(0, 3)
      .map((tick) => {
        const [, x, , w] = /^M([\d.]+) ([\d.]+)h([\d.]+)$/.exec(tick.getAttribute("d") ?? "") ?? [];
        return [Number(x), Number(x) + Number(w)] as const;
      })
      .sort((a, b) => a[0] - b[0]);
    expect(spans).toHaveLength(3);
    for (let i = 1; i < spans.length; i += 1) expect((spans[i]?.[0] ?? 0) >= (spans[i - 1]?.[1] ?? 0)).toBe(true);
  });

  it("says what a bar's height means in the view on screen: in raw error a bar above the line never beats steering straight", () => {
    layoutAs(1000, true);
    const { section, frames } = mount(false);
    const ladder = readLadder(raw());
    section.show(ladder, readSplits(raw()));
    expect(text('[data-ladder="lead"]')).toBe(ladderLead("skill"));
    expect(text('[data-ladder="chart-caption"]')).toBe(ladderCaption(ladder, "skill"));
    pick("ladder-view", "error");
    frames.tick(5000);
    expect(text('[data-ladder="lead"]')).toBe(ladderLead("error"));
    expect(text('[data-ladder="lead"]')).not.toMatch(/above the line beats/);
    pick("ladder-view", "skill");
    expect(text('[data-ladder="lead"]')).toBe(ladderLead("skill"));
  });

  it("says what the bars mean for the view picked even before the data arrives", () => {
    const { section } = mount(true);
    pick("ladder-view", "error");
    expect(text('[data-ladder="lead"]')).toBe(ladderLead("error"));
    expect(text('[data-ladder="chart-title"]')).toBe("Mean error, in degrees of steering-wheel angle");
    expect(section).toBeDefined();
  });

  it("describes each rung's radio button by the rung's sentence, and never announces the sentence by itself", () => {
    layoutAs(1000, true);
    const { section } = mount(true);
    section.show(readLadder(raw()), readSplits(raw()));
    const caption = q('[data-ladder="caption"]');
    expect(caption.hasAttribute("aria-live")).toBe(false);
    expect(caption.closest("[aria-live], [role=status], [role=alert]")).toBeNull();
    const radios = all('input[name="ladder-rung"]');
    expect(radios).toHaveLength(4);
    expect(radios.every((radio) => radio.getAttribute("aria-describedby") === caption.id)).toBe(true);
  });

  it("names the rung under the pointer with a capital, as a sentence starts", () => {
    layoutAs(1000, true);
    const { section } = mount(true);
    const ladder = readLadder(raw());
    section.show(ladder, readSplits(raw()));
    const layout = ladderLayout(ladder, "skill", 1000, 3);
    const bar = layout.bars.find((b) => b.model === "pilotnet" && b.rung === 3);
    const panel = layout.panels[3];
    const plot = q('[data-ladder="plot"]');
    plot.dispatchEvent(new PointerEvent("pointermove", { clientX: (bar?.x ?? 0) + 2, clientY: 100 + (panel?.y ?? 0) + (panel?.h ?? 0) / 2, bubbles: true }));
    expect(q('[data-ladder="tip"]').hidden).toBe(false);
    expect(text('[data-ladder="tip"]')).toBe("Another day: PilotNet: 24% worse than always steering straight (±8%)");
  });

  it("shows the view picked before the data arrived, whichever way it was picked", () => {
    layoutAs(1000, true);
    const { section } = mount(true);
    pick("ladder-view", "error");
    section.show(readLadder(raw()), readSplits(raw()));
    expect(text('[data-ladder="chart-title"]')).toBe("Mean error, in degrees of steering-wheel angle");
    expect(all(".ladder-chart__bars .chart__value").map((el) => el.textContent)).toContain("13.6°");
  });

  it("shows the view a browser restored with the page, which comes with no change event", () => {
    layoutAs(1000, true);
    const { section } = mount(true);
    q<HTMLInputElement>('input[name="ladder-view"][value="error"]').checked = true;
    section.show(readLadder(raw()), readSplits(raw()));
    expect(text('[data-ladder="chart-title"]')).toBe("Mean error, in degrees of steering-wheel angle");
    expect(all(".ladder-chart__bars .chart__value").map((el) => el.textContent)).toContain("13.6°");
  });

  it("says why there is no ladder where it is seen, in its lead, when the measurements cannot be had", () => {
    const { section } = mount(false);
    const error = new AssetError("gone", "unreachable", "data/results.json");
    section.fail(error);
    expect(text('[data-ladder="lead"]')).toBe(measurementsFailure(error));
    expect(q('[data-ladder="root"]').dataset["state"]).toBe("unavailable");
  });
});

// -- The cards ----------------------------------------------------------------------------------

describe("the cards", () => {
  const mount = (reduced = true) => new Cards(mountResults(), () => reduced, clock());
  /** A drawing's labels as read: each label's name and figure. */
  const labels = (card: string) => all(`[data-card="${card}-plot"] .card-chart__label`).map((el) => el.textContent ?? "");

  it("write each card's words and drawing from its own part of the data", () => {
    layoutAs(340, true);
    const cards = mount();
    const ladder = readLadder(raw());
    const frames = readFrames(raw());
    const regimes = readRegimes(raw());
    const legacy = readLegacy(raw());
    const sets = readYardsticks(raw());
    cards.showBaselines(sets);
    cards.showFrames(ladder, frames);
    cards.showRegimes(regimes);
    cards.showRecipe(legacy);
    cards.showTrust(readParity(raw()), readSplits(raw()));
    const text2 = twoFramesText(ladder, frames);
    expect(text('[data-card="frames-title"]')).toBe("What a second frame buys");
    expect([text('[data-card="frames-lead"]'), text('[data-card="frames-also"]')]).toEqual([text2?.lead, text2?.also]);
    expect(text('[data-card="regimes-straight"]')).toBe(regimesText(regimes).straight);
    expect(text('[data-card="recipe-cap"]')).toBe(legacyText(legacy).cap);
    expect(text('[data-card="baselines-verdict"]')).toBe(yardsticksVerdict(sets));
    expect(text('[data-card="export"]')).toBe(exportText(readParity(raw())));
    expect(text('[data-card="protocol-blocked"]')).toBe(protocolTexts(readSplits(raw())).blocked);
    const regimesChart = q('[data-card="regimes-plot"] svg');
    expect(regimesChart.getAttribute("role")).toBe("img");
    expect(regimesChart.getAttribute("aria-labelledby")).toBe("regimes-chart-title regimes-chart-desc");
    expect(text("#regimes-chart-desc")).toBe(regimesAlt(regimes));
    expect(text("#regimes-chart-title")).toBe("Mean error by how hard the driver was turning");
    expect(text("#frames-chart-desc")).toBe(framesAlt([
      { title: "Another day’s drive", more: ladder.bars.sightlinenet[3]?.skill.mean ?? 0, fewer: ladder.bars["sightlinenet-1f"][3]?.skill.mean ?? 0 },
      { title: "Held-back stretches of its own drive", more: ladder.bars.sightlinenet[1]?.skill.mean ?? 0, fewer: ladder.bars["sightlinenet-1f"][1]?.skill.mean ?? 0 },
    ], frames).desc);
    expect(text("#baselines-chart-desc")).toBe(yardsticksAlt(sets));
    expect(text("#recipe-chart-desc")).toMatch(/Its largest answer, either way, is 169\.8°; its output can never pass 180° either way, and the driver turned past that on 235 frames of the 2018 drive\.$/);
    for (const name of ["baselines", "frames", "regimes", "recipe", "trust"]) expect(q(`[data-card="${name}"]`).dataset["state"]).toBe("ready");
  });

  it("label every mark where it stands, by name and figure, never by a key to match colours against", () => {
    layoutAs(340, true);
    const cards = mount();
    cards.showFrames(readLadder(raw()), readFrames(raw()));
    cards.showRegimes(readRegimes(raw()));
    cards.showRecipe(readLegacy(raw()));
    cards.showBaselines(readYardsticks(raw()));
    expect(labels("frames")).toEqual(["one frame 2%", "two frames 31%", "one frame 35%", "two frames 41%"]);
    expect(labels("regimes")).toEqual([
      "Sightline 6.1°", "steering straight 2.8°",
      "Sightline 5.5°", "steering straight 12.6°",
      "Sightline 36.0°", "steering straight 78.4°",
    ]);
    expect(labels("recipe")).toEqual([
      "169.8°largest answereither way", "limit 180° either way", "235 frames of the 2018 driveturned past it",
      "PilotNet 4.5°", "steering straight 15.5°",
      "PilotNet 22.0°", "steering straight 17.5°",
      "PilotNet 19.4°", "steering straight 9.4°",
    ]);
    expect(labels("baselines")).toEqual([
      "Sightline 10.1°", "hold the last label 0.6°", "linear extrapolation 0.7°", "steering straight 17.5°", "mean angle 17.8°",
      "Sightline 9.8°", "hold the last label 0.5°", "linear extrapolation 0.5°", "steering straight 15.5°", "mean angle 15.5°",
    ]);
    expect(all('[data-card="regimes-plot"] .card-chart__bar')).toHaveLength(6);
    expect(all('[data-card="regimes-plot"] .card-chart__road')).toHaveLength(3);
    // Text wears the page's inks; the marks beside it carry the colours.
    expect(all(".card-chart__label").every((el) => !/series--/.test(el.getAttribute("class") ?? ""))).toBe(true);
  });

  it("draw the standard recipe's largest answer as a size either way, never as a turn to one side", () => {
    layoutAs(540, true);
    const cards = mount();
    cards.showRecipe(readLegacy(raw()));
    const plot = q('[data-card="recipe-plot"]');
    // The answer's size on both sides of straight ahead, mirror images of each other.
    const arcs = all('[data-card="recipe-plot"] .card-chart__reach');
    expect(arcs).toHaveLength(2);
    const ends = arcs.map((arc) => (arc.getAttribute("d") ?? "").match(/-?[\d.]+/g)?.map(Number) ?? []);
    const cx = 270;
    const [a, b] = ends as [number[], number[]];
    expect(a[0]).toBeCloseTo(cx, 6);
    expect(b.at(-2)).toBeCloseTo(cx, 6);
    expect((a.at(-2) ?? 0) - cx).toBeCloseTo(cx - (b[0] ?? 0), 1);
    // Nothing turns to one side.
    expect(plot.querySelector("[transform*=rotate]")).toBeNull();
    const words = labels("recipe").join(" | ");
    expect(words).toMatch(/largest answer.*either way/);
    expect(words).toMatch(/limit 180° either way/);
  });

  it("keep the standard recipe's words inside its drawing at the narrowest phone", () => {
    for (const width of [240, 280, 540]) {
      document.body.innerHTML = "";
      layoutAs(width, true);
      const cards = mount();
      cards.showRecipe(readLegacy(raw()));
      for (const text of all('[data-card="recipe-plot"] svg text')) {
        const x = Number(text.getAttribute("x"));
        const anchor = text.getAttribute("text-anchor") ?? "start";
        const longest = Math.max(...[...text.querySelectorAll("tspan")].map((span) => (span.getAttribute("x") === null ? 0 : textWidth(span.textContent ?? ""))), textWidth(text.textContent ?? "") * (text.querySelector("tspan[x]") ? 0 : 1));
        const [left, right] = anchor === "middle" ? [x - longest / 2, x + longest / 2] : anchor === "end" ? [x - longest, x] : [x, x + longest];
        expect(left, `${width}: ${text.textContent}`).toBeGreaterThanOrEqual(0);
        expect(right, `${width}: ${text.textContent}`).toBeLessThanOrEqual(width);
      }
    }
  });

  it("draw themselves in once they near the screen: the dumbbell's dot travels from fewer frames to more", () => {
    layoutAs(340, false);
    const frames = clock();
    const cards = new Cards(mountResults(), () => false, frames);
    cards.showFrames(readLadder(raw()), readFrames(raw()));
    const dots = () => all('[data-card="frames-plot"] .card-chart__dot.series--sightlinenet').map((dot) => Number(dot.getAttribute("cx")));
    const final = dots();
    layoutAs(340, true);
    intersect();
    const fewer = all('[data-card="frames-plot"] .card-chart__dot.series--oneframe').map((dot) => Number(dot.getAttribute("cx")));
    expect(dots()).toEqual(fewer);
    frames.tick(400);
    expect(dots().every((x, i) => x > (fewer[i] ?? 0) && x < (final[i] ?? 0))).toBe(true);
    frames.tick(2000);
    expect(dots()).toEqual(final);
  });

  it("repeat the live check as the band reads it", () => {
    const cards = mount();
    cards.live("checked", "Checked live: on all 420 frames.");
    expect(text('[data-card="live"]')).toBe("Checked live: on all 420 frames.");
    expect(q('[data-card="live"]').dataset["state"]).toBe("checked");
    expect(q('[data-card="live"]').closest("[aria-live], [role=status]")).toBeNull();
  });

  it("say why one card has no numbers without taking the others with it", () => {
    layoutAs(340, true);
    const cards = mount();
    const json = patch(raw(), "regimes.zero.straight.n", 1);
    let thrown: unknown;
    try {
      cards.showRegimes(readRegimes(json));
    } catch (error) {
      thrown = error;
      cards.fail("regimes", error);
    }
    cards.showFrames(readLadder(raw()), readFrames(raw()));
    expect(thrown).toBeInstanceOf(AssetError);
    expect(text('[data-card="regimes-lead"]')).toBe(measurementsFailure(thrown));
    expect(q('[data-card="regimes"]').dataset["state"]).toBe("unavailable");
    expect(q('[data-card="frames"]').dataset["state"]).toBe("ready");
  });
});

// -- The frame next door, and how the drive was split ---------------------------------------

import pairsFile from "../../public/data/leakage/pairs.json?raw";
import shippedFile from "../../public/data/results.json?raw";
import clipFile from "../../public/data/clips/c2018/clip.json?raw";
import { clockTime, neighbourWords as words2, pairsLead as lead2, timelineAlt as alt2, timelineNote as note2 } from "./copy.ts";
import { timeOfFrame as timeOf } from "../charts/timeline.ts";
import { readClipWindow as clipWindow, readPairs as pairsOf, readTimeline as timelineOf } from "./figures.ts";
import { PairsSection } from "./pairs.ts";
import { TimelineSection } from "./timeline.ts";

const realRaw = () => JSON.parse(shippedFile) as Record<string, unknown>;
const URL_PAIRS = "http://localhost/data/leakage/pairs.json";

describe("the frame next door", () => {
  const mount = (reduced: boolean, frames = clock()) => {
    mountResults();
    return { section: new PairsSection(q('[data-pairs="root"]'), () => reduced, frames), frames };
  };
  const pairs = () => pairsOf(JSON.parse(pairsFile) as Record<string, unknown>, realRaw());
  const photo = (split: string) => q<HTMLImageElement>(`[data-pairs="photo-${split}"] img`);

  it("shows the first pair the moment the data is in, every number and picture, with no click", () => {
    layoutAs(1000, false);
    const { section } = mount(true);
    const data = pairs();
    section.show(data, URL_PAIRS, 455 / 256);
    expect(text('[data-pairs="lead"]')).toBe(lead2(data));
    expect(all(".pairs__thumb")).toHaveLength(6);
    // The data's pairs run from the gentlest steering to the hardest, and the picker says so.
    expect(text(".pairs__legend")).toBe("Pick a test frame, from the gentlest steering to the hardest");
    expect(q<HTMLInputElement>('input[name="pairs-frame"]:checked').value).toBe("0");
    expect(photo("test").src).toBe("http://localhost/data/leakage/frame-7917.jpg");
    expect(photo("shuffled").src).toBe("http://localhost/data/leakage/frame-7916.jpg");
    expect(photo("blocked").src).toBe("http://localhost/data/leakage/frame-28219.jpg");
    expect(all('[data-pairs="facts-shuffled"] li').map((li) => li.textContent)).toEqual(["1 frame before it, 41 ms", "Pixel difference 8.8 grey levels", "", "Steering 0.5° right"]);
    expect(all('[data-pairs="table"] tbody tr')).toHaveLength(12);
    expect(q('[data-pairs="root"]').dataset["state"]).toBe("ready");
    expect(q('[data-pairs="root"]').style.getPropertyValue("--photo-aspect")).toBe(String(455 / 256));
  });

  it("shows another pair when its thumbnail is picked, its pictures settling in from the pointer, at once from the keyboard", () => {
    layoutAs(1000, true);
    const { section } = mount(false);
    const data = pairs();
    section.show(data, URL_PAIRS, null);
    const animate = vi.fn();
    for (const split of ["test", "shuffled", "blocked"]) q(`[data-pairs="photo-${split}"]`).animate = animate as unknown as HTMLElement["animate"];
    const pick = (i: number, pointer: boolean) => {
      const input = q<HTMLInputElement>(`input[name="pairs-frame"][value="${i}"]`);
      if (pointer) q('[data-pairs="picker"]').dispatchEvent(new Event("pointerdown", { bubbles: true }));
      input.checked = true;
      input.dispatchEvent(new Event("change", { bubbles: true }));
    };
    pick(3, false);
    expect(animate).not.toHaveBeenCalled();
    expect(photo("test").src).toMatch(/frame-29273\.jpg$/);
    pick(5, true);
    expect(animate).toHaveBeenCalledTimes(3);
    expect(photo("blocked").src).toMatch(/frame-21414\.jpg$|frame-\d+\.jpg$/);
    const pair = data.pairs[5];
    expect(text('[data-pairs="facts-blocked"] .pair-panel__where')).toBe(pair ? words2(pair, "blocked").where : "");
  });

  it("lays the test frame over its shuffled neighbour, the slider choosing how much of each shows", () => {
    layoutAs(1000, false);
    const { section } = mount(true);
    section.show(pairs(), URL_PAIRS, null);
    const range = q<HTMLInputElement>('[data-pairs="compare-range"]');
    expect(q('[data-pairs="compare-over"]').style.clipPath).toBe("inset(0 0 0 50%)");
    range.value = "20";
    range.dispatchEvent(new Event("input", { bubbles: true }));
    expect(q('[data-pairs="compare-over"]').style.clipPath).toBe("inset(0 0 0 20%)");
    expect(q('[data-pairs="compare-handle"]').style.left).toBe("20%");
    expect(range.getAttribute("aria-valuetext")).toBe("80% of the picture shows the shuffled neighbour");
    expect(q<HTMLImageElement>('[data-pairs="compare-over"] img').src).toMatch(/frame-7916\.jpg$/);
  });

  it("sweeps the slider across once by itself the first time it nears the screen, and never under reduced motion", () => {
    layoutAs(1000, true);
    const { section, frames } = mount(false);
    section.show(pairs(), URL_PAIRS, null);
    const over = () => q('[data-pairs="compare-over"]').style.clipPath;
    frames.tick(400);
    expect(over()).not.toBe("inset(0 0 0 50%)");
    frames.tick(4000);
    expect(over()).toBe("inset(0 0 0 50%)");
    document.body.innerHTML = "";
    const still = mount(true);
    still.section.show(pairs(), URL_PAIRS, null);
    still.frames.tick(400);
    expect(q('[data-pairs="compare-over"]').style.clipPath).toBe("inset(0 0 0 50%)");
  });

  it("gives the slider to the visitor the moment they reach for it: the sweep stops where it is, whatever they reach with", () => {
    const touches: Array<[string, string]> = [
      ["compare-range", "pointerdown"],
      ["compare-range", "keydown"],
      ["compare-frame", "pointerdown"],
    ];
    for (const [target, event] of touches) {
      document.body.innerHTML = "";
      layoutAs(1000, true);
      const { section, frames } = mount(false);
      section.show(pairs(), URL_PAIRS, null);
      const over = () => q('[data-pairs="compare-over"]').style.clipPath;
      frames.tick(400);
      const frozen = over();
      expect(frozen).not.toBe("inset(0 0 0 50%)");
      q(`[data-pairs="${target}"]`).dispatchEvent(new Event(event, { bubbles: true }));
      // Nothing more is drawn by the sweep: not at the next frame, not when it would have ended.
      expect(over()).toBe(frozen);
      frames.tick(300);
      frames.tick(4000);
      expect(over()).toBe(frozen);
    }
    // And a drag after the press holds: the sweep never pulls the handle back.
    document.body.innerHTML = "";
    layoutAs(1000, true);
    const { section, frames } = mount(false);
    section.show(pairs(), URL_PAIRS, null);
    frames.tick(400);
    const range = q<HTMLInputElement>('[data-pairs="compare-range"]');
    range.dispatchEvent(new Event("pointerdown", { bubbles: true }));
    range.value = "90";
    range.dispatchEvent(new Event("input", { bubbles: true }));
    frames.tick(4000);
    expect(q('[data-pairs="compare-over"]').style.clipPath).toBe("inset(0 0 0 90%)");
    expect(range.value).toBe("90");
  });

  it("says why there are no pairs, in its lead, when they cannot be had", () => {
    const { section } = mount(true);
    const error = new AssetError("gone", "missing", URL_PAIRS);
    section.fail(error);
    expect(text('[data-pairs="lead"]')).toBe(measurementsFailure(error));
    expect(q('[data-pairs="root"]').dataset["state"]).toBe("unavailable");
  });
});

describe("how the drive was split", () => {
  const mount = (reduced: boolean, frames = clock()) => {
    mountResults();
    return { section: new TimelineSection(q('[data-timeline="root"]'), () => reduced, frames), frames };
  };
  const layer = (i: number) => q(`.timeline-chart__layer[data-split="${["shuffled", "blocked", "later"][i] ?? ""}"]`);
  const clipWidth = (name: string) => Number(document.querySelector(`clipPath[id$="-${name}"] rect`)?.getAttribute("width"));

  it("draws the drive at once on the blocked split, every stretch where the splits module puts it, with the clip marked", () => {
    layoutAs(1000, false);
    const { section } = mount(true);
    const timeline = timelineOf(realRaw());
    section.show(timeline, clipWindow(JSON.parse(clipFile)));
    expect(text('[data-timeline="lead"]')).toMatch(/^The 2018 drive runs 55\.9 minutes/);
    expect(all(".timeline-chart__layer")).toHaveLength(3);
    // Shuffled, blocked, later: only the blocked layer is shown.
    expect([layer(0).style.display, layer(1).style.display, layer(2).style.display]).toEqual(["none", "", "none"]);
    expect(layer(1).querySelectorAll(".timeline-chart__segment")).toHaveLength(timeline.protocols.blocked.segments.length);
    expect(layer(2).querySelectorAll(".timeline-chart__segment")).toHaveLength(5);
    expect(text('[data-timeline="note"]')).toBe(note2(timeline, "blocked"));
    expect(text('[data-timeline="count-dropped"]')).toBe("12,000 frames");
    expect(q(".timeline-chart desc").textContent).toBe(alt2(timeline, "blocked"));
    expect(q(".timeline-chart").getAttribute("role")).toBe("img");
    expect(document.querySelector(".timeline-chart__clip-window")).not.toBeNull();
    expect(text('[data-timeline="clip"]')).toMatch(/sits inside one of the blocked split’s test stretches\.$/);
    expect(all('[data-timeline="table"] tbody tr')).toHaveLength(3);
  });

  it("places every stretch by the drive's own clock: the later split's test part starts where its first frame was recorded", () => {
    layoutAs(1000, false);
    const { section } = mount(true);
    const timeline = timelineOf(realRaw());
    section.show(timeline, null);
    const test = layer(2).querySelectorAll(".timeline-chart__segment")[4] as SVGRectElement;
    // Frame 54,501 is 37 frames into bucket 851.
    const k = Math.floor(54501 / 64);
    const t0 = timeline.clock.startS[k] ?? 0;
    const t1 = timeline.clock.startS[k + 1] ?? 0;
    const seconds = t0 + ((54501 - k * 64) / 64) * (t1 - t0);
    expect(Number(test.getAttribute("x"))).toBeCloseTo((seconds / timeline.clock.durationS) * 1000, 1);
  });

  it("sweeps a new split across the old one from the drive's start, the counts moving in step", () => {
    layoutAs(1000, true);
    const { section, frames } = mount(false);
    const timeline = timelineOf(realRaw());
    section.show(timeline, null);
    frames.tick(5000);
    const radio = q<HTMLInputElement>('input[name="timeline-split"][value="later"]');
    radio.checked = true;
    radio.dispatchEvent(new Event("change", { bubbles: true }));
    frames.tick(360);
    expect(clipWidth("later")).toBeGreaterThan(0);
    expect(clipWidth("later")).toBeLessThan(1000);
    expect(layer(1).style.display).toBe("");
    expect(text('[data-timeline="count-dropped"]')).not.toBe("1,000 frames");
    frames.tick(2000);
    expect(clipWidth("later")).toBe(1000);
    expect(layer(1).style.display).toBe("none");
    expect(text('[data-timeline="count-dropped"]')).toBe("1,000 frames");
    expect(text('[data-timeline="note"]')).toBe(note2(timeline, "later"));
  });

  it("switches at once under reduced motion, and draws the shuffled split as an illustration", () => {
    layoutAs(1000, true);
    const { section } = mount(true);
    const timeline = timelineOf(realRaw());
    section.show(timeline, null);
    const radio = q<HTMLInputElement>('input[name="timeline-split"][value="shuffled"]');
    radio.checked = true;
    radio.dispatchEvent(new Event("change", { bubbles: true }));
    expect([layer(0).style.display, layer(1).style.display]).toEqual(["", "none"]);
    expect(layer(0).querySelectorAll(".timeline-chart__cell").length).toBeGreaterThan(100);
    expect(text('[data-timeline="note"]')).toMatch(/an illustration: it does not say which frame went where\.$/);
  });

  it("names the stretch under the pointer: its part, its times on the drive's clock and its frames", () => {
    layoutAs(1000, true);
    const { section } = mount(true);
    const timeline = timelineOf(realRaw());
    section.show(timeline, null);
    const band = q(".timeline-chart__segment");
    const top = Number(band.getAttribute("y")) + 100;
    q('[data-timeline="plot"]').dispatchEvent(new PointerEvent("pointermove", { clientX: 5, clientY: top + 4, bubbles: true }));
    expect(q('[data-timeline="tip"]').hidden).toBe(false);
    const first = timeline.protocols.blocked.segments[0];
    expect(text('[data-timeline="tip"]')).toBe(`Train: 0:00 to ${clockTime(timeOf(timeline.clock, first?.to ?? 0))}, frames 0 to ${((first?.to ?? 1) - 1).toLocaleString("en-GB")}`);
    q('[data-timeline="plot"]').dispatchEvent(new PointerEvent("pointerleave", { bubbles: true }));
    expect(q('[data-timeline="tip"]').hidden).toBe(true);
  });

  it("says why there is no timeline, in its lead, when the measurements cannot be had", () => {
    const { section } = mount(true);
    const error = new AssetError("gone", "unreachable", "data/results.json");
    section.fail(error);
    expect(text('[data-timeline="lead"]')).toBe(measurementsFailure(error));
    expect(q('[data-timeline="root"]').dataset["state"]).toBe("unavailable");
  });
});

describe("the Results view whose code did not load", () => {
  it("leaves no part waiting: each says so in plain words, and none reads as checking", async () => {
    const { markResultsUnavailable, resultsCodeFailure } = await import("./unavailable.ts");
    mountResults();
    expect(all('[data-state="loading"]').length).toBeGreaterThan(5);
    markResultsUnavailable();
    expect(all('#results-view [data-state="loading"]')).toHaveLength(0);
    for (const root of ["results", "paired", "pairs", "timeline", "ladder"]) {
      expect(q(`[data-${root}="root"]`).dataset["state"]).toBe("unavailable");
    }
    expect(text(".results__lede")).toBe(resultsCodeFailure());
    expect(text('[data-timeline="lead"]')).toBe(resultsCodeFailure());
    expect(text('[data-card="export"]')).toBe(resultsCodeFailure());
  });
});
