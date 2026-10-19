// @vitest-environment happy-dom
import { describe, expect, it } from "vitest";

import { fmtCount, fmtDeg, fmtPct, fmtRatio } from "../format.ts";
import { countText, easeOut, runCounts, textAt, type Frames } from "./countup.ts";

describe("countText", () => {
  it("ends on exactly the figure's own text", () => {
    expect(countText(fmtRatio(3.0466), 3.0466, 1, 1)).toBe("3.0×");
    expect(countText(fmtPct(0.3145), 31.45, 0, 1)).toBe("31%");
    expect(countText(fmtDeg(21.8676), 21.8676, 0, 1)).toBe("21.9°");
  });

  it("starts where the count starts, with the final text's decimals and unit", () => {
    expect(countText("3.0×", 3.0466, 1, 0)).toBe("1.0×");
    expect(countText("31%", 31.45, 0, 0)).toBe("0%");
    expect(countText("0.64°", 0.638, 0, 0)).toBe("0.00°");
  });

  it("passes through the values in between, eased, never past the figure", () => {
    const texts = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99].map((p) => countText("21.9°", 21.8676, 0, p));
    const numbers = texts.map((text) => Number(text.replace("°", "")));
    for (let i = 1; i < numbers.length; i += 1) {
      expect(numbers[i], texts.join(" ")).toBeGreaterThanOrEqual(numbers[i - 1] as number);
    }
    expect(Math.max(...numbers)).toBeLessThanOrEqual(21.9);
    expect(texts[3]).toBe(`${(21.8676 * easeOut(0.5)).toFixed(1)}°`);
  });

  it("keeps a true minus on a count below zero, and never shows a negative zero", () => {
    expect(countText("−24%", -24.23, 0, 1)).toBe("−24%");
    expect(countText("−24%", -24.23, 0, 0.5)).toBe(`−${(24.23 * easeOut(0.5)).toFixed(0)}%`);
    expect(countText("−24%", -24.23, 0, 0.001)).toBe("0%");
  });

  it("keeps the thousands grouped while it counts", () => {
    expect(countText(fmtCount(45404), 45404, 0, 0.5)).toMatch(/^\d{2},\d{3}$/);
    expect(countText(fmtCount(45404), 45404, 0, 1)).toBe("45,404");
  });

  it("does not count a figure stated as a bound, or no figure at all", () => {
    expect(countText("<0.001°", 0.0002, 0, 0.3)).toBe("<0.001°");
    expect(countText(">99.99%", 99.999, 0, 0.3)).toBe(">99.99%");
    expect(countText("—", Number.NaN, 0, 0.3)).toBe("—");
  });
});

describe("textAt", () => {
  it("is the figure's text at an eased point of its count, for drawings that move in step", () => {
    expect(textAt("10.1°", 10.087, 3.348, 0)).toBe("3.3°");
    expect(textAt("10.1°", 10.087, 3.348, 0.5)).toBe(`${(3.348 + (10.087 - 3.348) * 0.5).toFixed(1)}°`);
    expect(textAt("10.1°", 10.087, 3.348, 1)).toBe("10.1°");
    // A count downwards, from steering straight's error to Sightline's.
    expect(textAt("10.6°", 10.64, 15.52, 0)).toBe("15.5°");
    expect(textAt("10.6°", 10.64, 15.52, 1)).toBe("10.6°");
    expect(countText("10.6°", 10.64, 15.52, 0.5)).toBe(textAt("10.6°", 10.64, 15.52, easeOut(0.5)));
  });
});

describe("runCounts", () => {
  /** A clock the test moves by hand, one animation frame at a time. */
  function clock(): Frames & { tick(ms: number): void } {
    let now = 0;
    let queued: Array<(t: number) => void> = [];
    return {
      now: () => now,
      request: (callback) => {
        queued.push(callback);
        return queued.length;
      },
      cancel: () => {
        queued = [];
      },
      tick(ms: number) {
        now += ms;
        const run = queued;
        queued = [];
        for (const callback of run) callback(now);
      },
    };
  }

  const figure = (final: string) => {
    const el = document.createElement("span");
    el.textContent = final;
    return el;
  };

  it("counts every figure up to its own text, and stops there", () => {
    const frames = clock();
    const a = figure("3.0×");
    const b = figure("31%");
    runCounts(
      [
        { el: a, final: "3.0×", value: 3.0466, from: 1 },
        { el: b, final: "31%", value: 31.45, from: 0, delay: 100 },
      ],
      { duration: 1000, reduced: false, frames },
    );
    expect(a.textContent).toBe("1.0×");
    expect(b.textContent).toBe("0%");
    frames.tick(500);
    expect(a.textContent).not.toBe("3.0×");
    expect(Number(b.textContent?.replace("%", ""))).toBeGreaterThan(0);
    frames.tick(700);
    expect(a.textContent).toBe("3.0×");
    expect(b.textContent).toBe("31%");
    frames.tick(100);
    expect(b.textContent).toBe("31%");
  });

  it("shows the final figures at once under reduced motion", () => {
    const frames = clock();
    const a = figure("");
    runCounts([{ el: a, final: "3.0×", value: 3.0466, from: 1 }], { duration: 1000, reduced: true, frames });
    expect(a.textContent).toBe("3.0×");
  });

  it("tells each figure how far its count has come, eased, ending at one", () => {
    const frames = clock();
    const seen: number[] = [];
    runCounts([{ el: figure(""), final: "3.0×", value: 3.0466, from: 1, onProgress: (p) => seen.push(p) }], {
      duration: 1000,
      reduced: false,
      frames,
    });
    frames.tick(500);
    frames.tick(600);
    expect(seen[0]).toBe(0);
    expect(seen[1]).toBeCloseTo(easeOut(0.5), 10);
    expect(seen.at(-1)).toBe(1);
  });

  it("tells it the count is done at once under reduced motion", () => {
    const seen: number[] = [];
    runCounts([{ el: figure(""), final: "31%", value: 31.45, from: 0, onProgress: (p) => seen.push(p) }], {
      duration: 1000,
      reduced: true,
      frames: clock(),
    });
    expect(seen).toEqual([1]);
  });

  it("can be stopped, leaving the final figures", () => {
    const frames = clock();
    const a = figure("");
    const stop = runCounts([{ el: a, final: "21.9°", value: 21.8676, from: 0 }], { duration: 1000, reduced: false, frames });
    frames.tick(200);
    stop();
    expect(a.textContent).toBe("21.9°");
    frames.tick(2000);
    expect(a.textContent).toBe("21.9°");
  });
});
