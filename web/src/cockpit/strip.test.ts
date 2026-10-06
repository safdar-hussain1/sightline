import { describe, expect, it } from "vitest";

import { REGIME_BINS_DEG, columnHeights, errorScale, frameAtX, regimeOf, stepForKey } from "./strip.ts";

describe("regimeOf", () => {
  it("bins a frame by how hard the driver is turning, either way", () => {
    const [straight, sharp] = REGIME_BINS_DEG;
    expect(regimeOf(0)).toBe("straight");
    expect(regimeOf(straight - 0.01)).toBe("straight");
    expect(regimeOf(-(straight - 0.01))).toBe("straight");
    expect(regimeOf(straight)).toBe("gentle");
    expect(regimeOf(-(sharp - 0.01))).toBe("gentle");
    expect(regimeOf(sharp)).toBe("sharp");
    expect(regimeOf(-501.8)).toBe("sharp");
  });
});

describe("errorScale", () => {
  it("fits the largest error either model makes on the clip, so both share one scale", () => {
    const labels = [0, 10, -20];
    const expected = { sightline: [1, 14, -21], shuffled: [0.5, 10, -9] };
    // The largest error is the twin's 11° on the last frame.
    const scale = errorScale(labels, expected);
    expect(scale).toBeGreaterThanOrEqual(11);
    expect(scale).toBeLessThan(11 * 1.25);
  });

  it("never collapses to zero, even on a clip where both models are exact", () => {
    expect(errorScale([1, 2], { sightline: [1, 2], shuffled: [1, 2] })).toBeGreaterThan(0);
  });
});

describe("columnHeights", () => {
  it("is each frame's absolute error as a share of the scale, capped at the top", () => {
    const heights = columnHeights([0, 10, -20, 5], new Float32Array([2, 5, -10, Number.NaN]), 10);
    [0.2, 0.5, 1].forEach((share, k) => expect(heights[k]).toBeCloseTo(share, 6));
    expect(heights[3]).toBeNaN(); // a frame with no prediction yet draws no column
  });
});

describe("frameAtX", () => {
  it("maps a point on the strip to the frame drawn there, clamped to the clip", () => {
    expect(frameAtX(0, 420, 420)).toBe(0);
    expect(frameAtX(419.9, 420, 420)).toBe(419);
    expect(frameAtX(210, 420, 420)).toBe(210);
    expect(frameAtX(100, 200, 420)).toBe(210);
    expect(frameAtX(-5, 200, 420)).toBe(0);
    expect(frameAtX(205, 200, 420)).toBe(419);
  });
});

describe("stepForKey", () => {
  it("steps a frame with the arrow keys and a second with Page Up and Page Down", () => {
    expect(stepForKey("ArrowRight", 10, 420, 17.5)).toBe(11);
    expect(stepForKey("ArrowUp", 10, 420, 17.5)).toBe(11);
    expect(stepForKey("ArrowLeft", 10, 420, 17.5)).toBe(9);
    expect(stepForKey("ArrowDown", 10, 420, 17.5)).toBe(9);
    expect(stepForKey("PageUp", 10, 420, 17.5)).toBe(28);
    expect(stepForKey("PageDown", 30, 420, 17.5)).toBe(12);
  });

  it("jumps to the ends with Home and End, and stays inside the clip", () => {
    expect(stepForKey("Home", 200, 420, 17.5)).toBe(0);
    expect(stepForKey("End", 200, 420, 17.5)).toBe(419);
    expect(stepForKey("ArrowLeft", 0, 420, 17.5)).toBe(0);
    expect(stepForKey("ArrowRight", 419, 420, 17.5)).toBe(419);
    expect(stepForKey("PageDown", 3, 420, 17.5)).toBe(0);
  });

  it("leaves every other key alone", () => {
    expect(stepForKey("Enter", 10, 420, 17.5)).toBeNull();
    expect(stepForKey("a", 10, 420, 17.5)).toBeNull();
  });
});
