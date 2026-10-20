import { describe, expect, it } from "vitest";

import { linear, niceAxis, niceStep } from "./scale.ts";

describe("linear", () => {
  it("maps the domain onto the range, either way round", () => {
    const y = linear([0, 20], [300, 0]);
    expect(y(0)).toBe(300);
    expect(y(20)).toBe(0);
    expect(y(5)).toBe(225);
    expect(linear([-1, 1], [0, 100])(0)).toBe(50);
  });

  it("maps a domain of no width to the middle of the range", () => {
    expect(linear([3, 3], [0, 10])(3)).toBe(5);
  });
});

describe("niceStep", () => {
  it("steps by 1, 2, 2.5 or 5 times a power of ten", () => {
    expect(niceStep(18.8, 4)).toBe(5);
    expect(niceStep(1.7, 4)).toBe(0.5);
    expect(niceStep(0.9, 4)).toBe(0.25);
    expect(niceStep(21, 4)).toBe(10);
    expect(niceStep(0, 4)).toBe(1);
  });
});

describe("niceAxis", () => {
  it("starts an error axis at zero and ends it on a round figure at or above the largest value", () => {
    expect(niceAxis(0, 18.825535512678442)).toEqual({ extent: [0, 20], ticks: [0, 5, 10, 15, 20] });
    expect(niceAxis(2.8, 20.51000439511419)).toEqual({ extent: [0, 30], ticks: [0, 10, 20, 30] });
  });

  it("keeps zero inside a skill axis that runs below it, with round ticks either side", () => {
    expect(niceAxis(-0.889, 0.779)).toEqual({ extent: [-1, 1], ticks: [-1, -0.5, 0, 0.5, 1] });
    expect(niceAxis(0.02, 0.41)).toEqual({ extent: [0, 0.6], ticks: [0, 0.2, 0.4, 0.6] });
    expect(niceAxis(0.02, 0.41, 5)).toEqual({ extent: [0, 0.5], ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5] });
  });

  it("does not widen past a value that is itself a round figure", () => {
    expect(niceAxis(0, 20).extent).toEqual([0, 20]);
    expect(niceAxis(-0.5, 0.5).extent).toEqual([-0.5, 0.5]);
  });

  it("never prints a tick as a negative zero", () => {
    expect(Object.is(niceAxis(-0.3, 0.6).ticks.find((t) => t === 0), -0)).toBe(false);
  });
});
