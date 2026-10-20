// @vitest-environment happy-dom
/**
 * The sections as a visitor meets them: the accuracy figure and the Results
 * band once results.json has loaded (or has not), and How it works reading a
 * cockpit. The cockpit is a fake (`cockpit`), the screen is decided by the test
 * (`onScreen`), and the counts run on a clock the test moves by hand.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Agreement, LiveCockpit, LiveFrame, LiveStatus } from "../cockpit/live.ts";
import { AssetError } from "../data/load.ts";
import { driveGaps } from "../data/timing.ts";
import type { Clip } from "../data/types.ts";
import { Accuracy } from "./accuracy.ts";
import { accuracyCaption, accuracySentence, leakageSentence, measurementsFailure, pairedLabel, parityState } from "./copy.ts";
import type { Frames } from "./countup.ts";
import { readFigures } from "./figures.ts";
import { patch, resultsFixture } from "./fixture.ts";
import { Headline } from "./headline.ts";
import { HowItWorks } from "./how.ts";
import { accuracyHtml, howHtml, resultsHtml } from "./markup.ts";

const figures = () => readFigures(resultsFixture());

function section(id: string, html: string): HTMLElement {
  const element = document.createElement("section");
  element.id = id;
  element.innerHTML = html;
  document.body.append(element);
  return element;
}

const q = <T extends Element = HTMLElement>(selector: string) => document.querySelector<T>(selector) as T;
const text = (selector: string) => q(selector).textContent ?? "";

/** happy-dom lays nothing out: the test says what is on screen. */
function onScreen(visible: boolean): void {
  const box = visible
    ? { x: 0, y: 100, top: 100, bottom: 400, left: 0, right: 800, width: 800, height: 300 }
    : { x: 0, y: 0, top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 };
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({ ...box, toJSON: () => box } as DOMRect);
}

/** The intersection observers the sections make; a test fires them. */
let observers: Array<(entries: Array<{ isIntersecting: boolean }>) => void> = [];
const intersect = (isIntersecting: boolean) => {
  for (const observe of observers) observe([{ isIntersecting }]);
};

let animated: Element[] = [];

beforeEach(() => {
  document.body.innerHTML = "";
  delete document.documentElement.dataset["view"];
  observers = [];
  animated = [];
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
  vi.spyOn(Element.prototype, "animate").mockImplementation(function (this: Element) {
    animated.push(this);
    return { cancel: () => undefined } as unknown as Animation;
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

// -- The accuracy figure ------------------------------------------------------------

describe("the accuracy figure", () => {
  const mount = (reduced: boolean) => new Accuracy(section("accuracy", accuracyHtml()), () => reduced);
  const share = (name: string) => Number(q(`[data-accuracy="bar-${name}"]`).style.getPropertyValue("--share"));

  it("writes its sentences and its bars from the figures, with no interaction", () => {
    onScreen(false);
    mount(false).show(figures());
    const { heldOut, paired } = figures();
    expect(text('[data-accuracy="sentence"]')).toBe(accuracySentence(heldOut));
    expect(text('[data-accuracy="caption"]')).toBe(accuracyCaption(heldOut));
    expect(text('[data-accuracy="leak"]')).toBe(leakageSentence(paired.sightlinenet.ratio.mean));
    expect(q('[data-accuracy="plot"]').getAttribute("aria-label")).toMatch(/Sightline 10\.1°, always steering straight 17\.5°/);
    expect(share("straight")).toBe(1);
    expect(share("model")).toBeCloseTo(heldOut.model / heldOut.straight, 12);
    expect([text('[data-accuracy="value-model"]'), text('[data-accuracy="value-straight"]')]).toEqual(["10.1°", "17.5°"]);
    expect(q('[data-accuracy="root"]').dataset["state"]).toBe("ready");
  });

  it("grows its bars once, as it nears the screen, and never under reduced motion", () => {
    onScreen(false);
    mount(false).show(figures());
    expect(animated).toEqual([]);
    intersect(true);
    const bars = animated.filter((el) => el.classList.contains("bars__bar"));
    expect(bars).toHaveLength(2);
    intersect(true);
    expect(animated.filter((el) => el.classList.contains("bars__bar"))).toHaveLength(2);

    document.body.innerHTML = "";
    animated = [];
    onScreen(true);
    mount(true).show(figures());
    intersect(true);
    expect(animated).toEqual([]);
  });

  it("grows its bars at once when it is already on screen", () => {
    onScreen(true);
    mount(false).show(figures());
    expect(animated.filter((el) => el.classList.contains("bars__bar"))).toHaveLength(2);
  });

  it("says why there are no figures when the measurements cannot be had", () => {
    const error = new AssetError("gone", "missing", "data/results.json");
    mount(false).fail(error);
    expect(text('[data-accuracy="sentence"]')).toBe(measurementsFailure(error));
    expect(q('[data-accuracy="root"]').dataset["state"]).toBe("unavailable");
  });
});

// -- The Results band --------------------------------------------------------------------

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

const shown = (name: string) => text(`[data-count="${name}"] .count__shown`);
const spoken = (name: string) => text(`[data-count="${name}"] .count__text`);
const barShare = (part: string) => Number(q(`[data-results="bar-${part}"]`).style.getPropertyValue("--share"));

describe("the Results band", () => {
  const mount = (reduced: boolean, frames = clock()) => ({
    band: new Headline(section("results-view", resultsHtml()), () => reduced, frames),
    frames,
  });

  it("shows its final figures at once under reduced motion", () => {
    onScreen(true);
    const { band } = mount(true);
    band.show(figures());
    expect(["ratio", "ratio-pilotnet", "another-day", "memoriser-shuffled", "memoriser-blocked"].map(shown)).toEqual([
      "3.0×",
      "3.2×",
      "31%",
      "0.6°",
      "21.9°",
    ]);
    expect(barShare("paired-blocked")).toBeCloseTo(10.086980146092957 / 15.52269723393061, 12);
    expect(barShare("day-model")).toBeCloseTo(10.640418845640982 / 15.52269723393061, 12);
    expect(text('[data-results="value-paired-blocked"]')).toBe("10.1°");
  });

  it("waits at the start of each count, out of sight, until its view is on screen; then counts once, by itself", () => {
    onScreen(false);
    const { band, frames } = mount(false);
    band.show(figures());
    expect(["ratio", "another-day", "memoriser-blocked"].map(shown)).toEqual(["1.0×", "0%", "0.0°"]);
    // Read aloud, it is always the final figure.
    expect(spoken("ratio")).toBe("3.0×");
    // The held-back bar starts at the shuffled one's length; Sightline's at steering straight's.
    expect(barShare("paired-blocked")).toBeCloseTo(barShare("paired-shuffled"), 12);
    expect(barShare("day-model")).toBe(1);
    expect(text('[data-results="value-day-model"]')).toBe("15.5°");

    onScreen(true);
    band.viewShown("results");
    frames.tick(700);
    expect(Number(shown("ratio").replace("×", ""))).toBeGreaterThan(1);
    expect(barShare("paired-blocked")).toBeGreaterThan(barShare("paired-shuffled"));
    frames.tick(2000);
    expect(["ratio", "ratio-pilotnet", "another-day", "memoriser-shuffled", "memoriser-blocked"].map(shown)).toEqual([
      "3.0×",
      "3.2×",
      "31%",
      "0.6°",
      "21.9°",
    ]);
    expect(text('[data-results="value-paired-blocked"]')).toBe("10.1°");
    expect(text('[data-results="value-day-model"]')).toBe("10.6°");

    // Once only: coming back to the view does not count again.
    band.viewShown("demo");
    band.viewShown("results");
    intersect(true);
    expect(shown("ratio")).toBe("3.0×");
  });

  it("counts on a deep link to the results as soon as the figures arrive", () => {
    document.documentElement.dataset["view"] = "results";
    onScreen(true);
    const { band, frames } = mount(false);
    band.show(figures());
    frames.tick(3000);
    expect(shown("another-day")).toBe("31%");
  });

  it("waits while its view is on but the band is scrolled away, and counts when it comes into sight", () => {
    document.documentElement.dataset["view"] = "results";
    onScreen(false);
    const { band, frames } = mount(false);
    band.show(figures());
    frames.tick(3000);
    expect(shown("ratio")).toBe("1.0×");
    onScreen(true);
    intersect(true);
    frames.tick(3000);
    expect(shown("ratio")).toBe("3.0×");
  });

  it("does not count while its view is hidden behind the demo", () => {
    onScreen(true);
    const { band, frames } = mount(false);
    band.viewShown("demo");
    band.show(figures());
    frames.tick(3000);
    expect(shown("ratio")).toBe("1.0×");
  });

  it("reads the live check from the cockpit: passing within the clip's tolerance, failing beyond it", () => {
    vi.useFakeTimers();
    document.documentElement.dataset["view"] = "results";
    const { band } = mount(true);
    const clipFile = { title: "2018: minutes it never trained on", tolerance: { angle_deg: 0.001 } } as Clip;
    let agreement: Agreement = { clip: clipFile, model: "sightline", check: "match", compared: 420, total: 420, maxDiffDeg: 0.0000305 };
    const cockpit = {
      agreement: () => agreement,
      status: () => ({ state: "ready" }) as LiveStatus,
    } as unknown as LiveCockpit;
    band.attach(cockpit);
    expect(text('[data-results="trust-verdict"]')).toBe(parityState(agreement, "ready").text);
    expect(q('[data-results="trust"]').dataset["state"]).toBe("checked");

    // The wrong model served under Sightline's name: its frames decode the
    // same, its angles do not. A final verdict is read again when the view
    // comes back.
    agreement = { ...agreement, maxDiffDeg: 124.8157 };
    band.viewShown("demo");
    band.viewShown("results");
    expect(q('[data-results="trust"]').dataset["state"]).toBe("mismatch");
    expect(text('[data-results="trust-verdict"]')).toMatch(/do not match the same model run in Python/);

    agreement = { ...agreement, check: "differ", maxDiffDeg: 0 };
    band.viewShown("results");
    expect(q('[data-results="trust"]').dataset["state"]).toBe("note");
  });

  it("announces the live check's verdict politely, once, and never its progress", async () => {
    vi.useFakeTimers();
    document.documentElement.dataset["view"] = "results";
    const { band } = mount(true);
    const verdict = q('[data-results="trust-verdict"]');
    const progress = q('[data-results="trust-text"]');
    expect(verdict.getAttribute("role")).toBe("status");
    expect(progress.closest('[role="status"], [role="alert"], [aria-live]')).toBeNull();

    const clipFile = { title: "2018: minutes it never trained on", tolerance: { angle_deg: 0.001 } } as Clip;
    let agreement: Agreement = { clip: clipFile, model: "sightline", check: "match", compared: 96, total: 420, maxDiffDeg: 0.0000305 };
    band.attach({ agreement: () => agreement, status: () => ({ state: "ready" }) as LiveStatus } as unknown as LiveCockpit);
    expect(progress.textContent).toMatch(/^Checking Sightline live: 96 of 420 frames/);
    expect(progress.hidden).toBe(false);
    expect(verdict.textContent).toBe("");

    const writes: MutationRecord[] = [];
    new MutationObserver((records) => void writes.push(...records)).observe(verdict, { childList: true, characterData: true, subtree: true });
    agreement = { ...agreement, compared: 300 };
    vi.advanceTimersByTime(700);
    await Promise.resolve();
    expect(writes).toEqual([]);
    expect(progress.textContent).toMatch(/300 of 420 frames/);

    agreement = { ...agreement, compared: 420 };
    vi.advanceTimersByTime(700);
    expect(verdict.textContent).toBe(parityState(agreement, "ready").text);
    expect(progress.hidden).toBe(true);
    // The verdict is final: the check stops reading, and says it once.
    expect(vi.getTimerCount()).toBe(0);
    vi.advanceTimersByTime(5000);
    await Promise.resolve();
    expect(writes).toHaveLength(1);
  });

  it("announces a failing check when it fails and when it completes, never at each frame, and stops reading after", async () => {
    vi.useFakeTimers();
    document.documentElement.dataset["view"] = "results";
    const { band } = mount(true);
    const verdict = q('[data-results="trust-verdict"]');
    const progress = q('[data-results="trust-text"]');
    // Each write puts new text in: count those, not the records a write makes.
    const writes: string[] = [];
    new MutationObserver((records) => {
      for (const record of records) if (record.addedNodes.length > 0 || record.type === "characterData") writes.push(verdict.textContent ?? "");
    }).observe(verdict, { childList: true, characterData: true, subtree: true });

    // The twin served under Sightline's name: its frames decode the same, its angles do not.
    const clipFile = { title: "2018: minutes it never trained on", tolerance: { angle_deg: 0.001 } } as Clip;
    let agreement: Agreement = { clip: clipFile, model: "sightline", check: "match", compared: 24, total: 420, maxDiffDeg: 61.2 };
    band.attach({ agreement: () => agreement, status: () => ({ state: "ready" }) as LiveStatus } as unknown as LiveCockpit);
    await Promise.resolve();
    expect(verdict.textContent).toBe("Sightline’s angles in this browser do not match the same model run in Python.");
    expect(progress.textContent).toMatch(/^On the 24 frames of the demo’s 2018 clip run so far, they differ by up to 61\.2°, more than the 0\.001° allowed\.$/);
    expect(writes).toHaveLength(1);

    for (const compared of [96, 180, 300, 419]) {
      agreement = { ...agreement, compared, maxDiffDeg: 124.8157 };
      vi.advanceTimersByTime(700);
    }
    await Promise.resolve();
    expect(writes).toHaveLength(1);
    expect(progress.textContent).toMatch(/^On the 419 frames of the demo’s 2018 clip run so far, they differ by up to 124\.9°/);

    agreement = { ...agreement, compared: 420 };
    vi.advanceTimersByTime(700);
    await Promise.resolve();
    expect(writes).toHaveLength(2);
    expect(verdict.textContent).toBe(
      "Sightline’s angles in this browser do not match the same model run in Python: on the demo’s 2018 clip they differ by up to 124.9°, more than the 0.001° allowed.",
    );
    expect(progress.hidden).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
    vi.advanceTimersByTime(5000);
    await Promise.resolve();
    expect(writes).toHaveLength(2);
  });

  it("reads the check again when its view comes back, and keeps reading only while it can change", () => {
    vi.useFakeTimers();
    document.documentElement.dataset["view"] = "results";
    const { band } = mount(true);
    const clipFile = { title: "2018: minutes it never trained on", tolerance: { angle_deg: 0.001 } } as Clip;
    let agreement: Agreement = { clip: clipFile, model: "sightline", check: "match", compared: 420, total: 420, maxDiffDeg: 0 };
    band.attach({ agreement: () => agreement, status: () => ({ state: "ready" }) as LiveStatus } as unknown as LiveCockpit);
    expect(vi.getTimerCount()).toBe(0);
    // The visitor picked the other model on the demo, and came back.
    band.viewShown("demo");
    agreement = { ...agreement, model: "shuffled", compared: 30 };
    band.viewShown("results");
    expect(text('[data-results="trust-text"]')).toMatch(/^Checking the twin live: 30 of 420 frames/);
    expect(vi.getTimerCount()).toBe(1);
    agreement = { ...agreement, compared: 420 };
    vi.advanceTimersByTime(700);
    expect(text('[data-results="trust-verdict"]')).toMatch(/^Checked live: on all 420 frames/);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("says why there is no live check when the demo's code never started", () => {
    const { band } = mount(true);
    band.attach(null);
    expect(text('[data-results="trust-verdict"]')).toBe("The live check needs the demo, which could not start in this browser.");
    expect(q('[data-results="trust"]').dataset["state"]).toBe("note");
  });

  it("calls the paired figure flattery only when both networks' ratios are above one", () => {
    const json = resultsFixture();
    patch(json, "paired_leakage.pilotnet.ratio", { mean: 0.9, sd: 0.1, values: [0.8, 0.9, 1] });
    const { band } = mount(true);
    band.show(readFigures(json));
    expect(text('[data-results="paired-label"]')).toBe(pairedLabel(0.9));
  });

  it("names both networks in the paired figure", () => {
    const { band } = mount(true);
    band.show(figures());
    const names = [...document.querySelectorAll('.finding--hero .finding__split')].map((el) => el.textContent);
    expect(names).toEqual(["SightlineNet", "PilotNet"]);
  });

  it("stops calling the paired ratio flattery if it falls below one, and still counts it from one", () => {
    onScreen(false);
    const json = resultsFixture();
    patch(json, "paired_leakage.sightlinenet.ratio", { mean: 0.8, sd: 0.1, values: [0.7, 0.8, 0.9] });
    const { band, frames } = mount(false);
    band.show(readFigures(json));
    expect(text('[data-results="paired-label"]')).toBe(pairedLabel(0.8));
    expect(shown("ratio")).toBe("1.0×");
    onScreen(true);
    band.viewShown("results");
    frames.tick(400);
    expect(Number(shown("ratio").replace("×", ""))).toBeLessThan(1);
    frames.tick(3000);
    expect(shown("ratio")).toBe("0.8×");
  });

  it("says why there are no figures when the measurements cannot be had", () => {
    const { band } = mount(false);
    const error = new AssetError("gone", "unreachable", "data/results.json");
    band.fail(error);
    expect(text('[data-results="lede"]')).toBe(measurementsFailure(error));
    expect(q('[data-results="root"]').dataset["state"]).toBe("unavailable");
  });
});

// -- How it works -------------------------------------------------------------------

describe("How it works", () => {
  const CLIP = {
    id: "c2018",
    drive: "drive2018",
    title: "2018: minutes it never trained on",
    fps: 17.5,
    n: 40,
    lead_in: 2,
    frame_lags: [2],
    sprite: { frame_w: 200, frame_h: 66 },
  } as unknown as Clip;
  const frame = (fill: number) => new Uint8Array(200 * 66 * 3).fill(fill);

  /** A cockpit whose clip, frame and answers the test sets. */
  function fakeCockpit() {
    const world = {
      status: { state: "loading", playing: false, frame: 0, clip: CLIP, model: "sightline" } as LiveStatus,
      answers: new Map<number, number>([[0, 12.6]]),
    };
    const cockpit: LiveCockpit = {
      status: () => world.status,
      frameAt: (k: number): LiveFrame | null =>
        world.status.clip === null
          ? null
          : {
              clip: world.status.clip,
              k,
              model: "sightline",
              inputs: [frame(k + 2), frame(k)],
              angle: world.answers.get(k) ?? null,
              heat: world.answers.has(k) ? new Uint8Array(200 * 66).fill(128) : null,
            },
      agreement: () => null,
      msPerFrame: () => 2.53,
    };
    return { world, cockpit };
  }

  let puts: HTMLCanvasElement[] = [];
  let clears: HTMLCanvasElement[] = [];
  const drawn = () => puts.flatMap((canvas) => canvas.dataset["how"] ?? []);
  const GAPS = driveGaps(resultsFixture());

  beforeEach(() => {
    puts = [];
    clears = [];
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function (this: HTMLCanvasElement) {
      const canvas = this;
      return {
        createImageData: (w: number, h: number) => new ImageData(w, h),
        putImageData: () => void puts.push(canvas),
        drawImage: () => undefined,
        clearRect: () => void clears.push(canvas),
      } as unknown as CanvasRenderingContext2D;
    });
  });

  const mount = (reduced: boolean) => new HowItWorks(section("how-it-works", howHtml()), () => reduced);
  const steps = () => q('[data-how="steps"]');

  it("shows the drawings until the clip's frames are decoded, then the first frame at once", () => {
    vi.useFakeTimers();
    const { world, cockpit } = fakeCockpit();
    world.status = { ...world.status, clip: null };
    const how = mount(false);
    how.setGaps(GAPS);
    how.attach(cockpit);
    expect(steps().dataset["live"]).toBe("false");
    expect(q('[data-how="toggle"]').hidden).toBe(true);

    world.status = { ...world.status, clip: CLIP };
    vi.advanceTimersByTime(300);
    expect(steps().dataset["live"]).toBe("true");
    expect(text('[data-how="frames-title"]')).toBe("Two frames, a median 100\u00A0ms apart on the 2018 drive");
    expect([text('[data-how="age-0"]'), text('[data-how="age-1"]')]).toEqual(["now", "two frames before"]);
    // The page's three canvases (the heat is composited through an unseen one).
    expect(puts.flatMap((canvas) => canvas.dataset["how"] ?? []).sort()).toEqual(["frame-0", "frame-1", "heat"]);
    expect(text('[data-how="angle"]') + " " + text('[data-how="side"]')).toBe("12.6° right");
    expect(text('[data-how="speed"]')).toBe("Here it takes about 2.5\u00A0ms a frame.");
    expect(q('[data-how="toggle"]').hidden).toBe(false);
  });

  it("draws a frame the model has not answered yet without a number, and fills it in when the answer comes", () => {
    vi.useFakeTimers();
    const { world, cockpit } = fakeCockpit();
    world.answers.clear();
    const how = mount(true);
    how.attach(cockpit);
    expect(steps().dataset["live"]).toBe("true");
    expect(q('[data-how="readout"]').hidden).toBe(true);
    world.answers.set(0, -3.2);
    vi.advanceTimersByTime(300);
    expect(q('[data-how="readout"]').hidden).toBe(false);
    expect(text('[data-how="angle"]') + " " + text('[data-how="side"]')).toBe("3.2° left");
  });

  it("follows the cockpit's frame while its steps are paused and on screen", () => {
    vi.useFakeTimers();
    const { world, cockpit } = fakeCockpit();
    world.answers.set(7, 40);
    const how = mount(true);
    how.attach(cockpit);
    intersect(true);
    world.status = { ...world.status, frame: 7 };
    vi.advanceTimersByTime(300);
    expect(text('[data-how="angle"]')).toBe("40.0°");
  });

  it("goes back to its drawings between clips: no angle, the wheel at rest, the frames cleared", () => {
    vi.useFakeTimers();
    const { world, cockpit } = fakeCockpit();
    const how = mount(true);
    how.attach(cockpit);
    expect(text('[data-how="angle"]')).toBe("12.6°");
    world.status = { ...world.status, clip: null };
    vi.advanceTimersByTime(300);
    expect(steps().dataset["live"]).toBe("false");
    expect(q('[data-how="readout"]').hidden).toBe(true);
    expect(q<SVGGElement>('[data-how="wheel"] [data-wheel="model"]').style.transform).toBe("rotate(0deg)");
    expect(clears.flatMap((canvas) => canvas.dataset["how"] ?? []).sort()).toEqual(["frame-0", "frame-1", "heat"]);
    // Once at rest it stays at rest, without clearing again on every look.
    const cleared = clears.length;
    vi.advanceTimersByTime(1000);
    expect(clears.length).toBe(cleared);
  });

  it("out of sight, draws a clip's first frame and fills in its answer, but does not follow every frame", () => {
    vi.useFakeTimers();
    const { world, cockpit } = fakeCockpit();
    world.answers.clear();
    const how = mount(true);
    how.attach(cockpit);
    intersect(false);
    expect(drawn().sort()).toEqual(["frame-0", "frame-1", "heat"]);
    world.answers.set(0, 5).set(5, 9).set(9, -2);
    vi.advanceTimersByTime(300); // the first frame's answer arrives: drawn once
    const afterAnswer = drawn().length;
    expect(text('[data-how="angle"]')).toBe("5.0°");
    for (const k of [5, 9]) {
      world.status = { ...world.status, frame: k };
      vi.advanceTimersByTime(300);
    }
    expect(drawn().length).toBe(afterAnswer);
    intersect(true); // back in sight: it catches up at once
    expect(text('[data-how="angle"]')).toBe("2.0°");
  });

  it("counts frames, not milliseconds, on a drive without timestamps", () => {
    const { world, cockpit } = fakeCockpit();
    world.status = { ...world.status, clip: { ...CLIP, id: "c2017", drive: "drive2017", title: "2017: another day" } as Clip };
    const how = mount(true);
    how.attach(cockpit);
    // Before results.json says which drive has timestamps, nothing is timed.
    expect(text('[data-how="frames-title"]')).toBe("This frame and the one two frames before it");
    how.setGaps(GAPS);
    expect(text('[data-how="frames-title"]')).toBe("This frame and the one two frames before it");
    expect(text('[data-how="age-1"]')).toBe("two frames before");
    expect(text('[data-how="frames-text"]')).toMatch(/on the 2018 drive, whose frames do, a frame and the one two before it were taken a median 100\u00A0ms apart/);
  });

  it("starts paused under reduced motion, playing otherwise, and the button switches it", () => {
    const { cockpit } = fakeCockpit();
    const still = mount(true);
    still.attach(cockpit);
    expect(text('[data-how="toggle-label"]')).toBe("Play");
    document.body.innerHTML = "";
    const moving = mount(false);
    moving.attach(cockpit);
    expect(text('[data-how="toggle-label"]')).toBe("Pause");
    q<HTMLButtonElement>('[data-how="toggle"]').click();
    expect(text('[data-how="toggle-label"]')).toBe("Play");
    expect(q('[data-how="toggle"]').dataset["playing"]).toBe("false");
  });

  it("keeps its drawings when the demo cannot start", () => {
    const how = mount(false);
    how.attach(null);
    expect(steps().dataset["live"]).toBe("false");
    expect(q('[data-how="toggle"]').hidden).toBe(true);
  });
});
