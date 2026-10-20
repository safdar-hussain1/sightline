import { describe, expect, it } from "vitest";

import { arcPath, dumbbellLabels, hbarPath, labelSpan, textWidth, wheelPoint } from "./marks.ts";

describe("a wheel's geometry", () => {
  it("puts 0° at the top and turns clockwise, as the wheel turns right", () => {
    expect(wheelPoint(100, 100, 50, 0)).toEqual([100, 50]);
    expect(wheelPoint(100, 100, 50, 90)).toEqual([150, 100]);
    expect(wheelPoint(100, 100, 50, 180)).toEqual([100, 150]);
    expect(wheelPoint(100, 100, 50, 270)).toEqual([50, 100]);
  });

  it("draws an arc clockwise, taking the long way round past half a turn", () => {
    expect(arcPath(100, 100, 50, 0, 90)).toBe("M100 50A50 50 0 0 1 150 100");
    expect(arcPath(100, 100, 50, 0, 270)).toBe("M100 50A50 50 0 1 1 50 100");
    expect(arcPath(100, 100, 50, 30, 30)).toBe("M125 56.7");
    expect(arcPath(100, 100, 50, 0, 360)).toBe("M100 50A50 50 0 0 1 100 150A50 50 0 0 1 100 50");
  });
});

describe("a horizontal bar", () => {
  it("is square at its base and rounded at its data end, either way", () => {
    expect(hbarPath(10, 110, 0, 10)).toBe("M10 0H106Q110 0 110 4V6Q110 10 106 10H10Z");
    expect(hbarPath(110, 10, 0, 10)).toBe("M110 0H14Q10 0 10 4V6Q10 10 14 10H110Z");
    expect(hbarPath(10, 10, 0, 10)).toBe("M10 0v10");
  });
});

describe("a dumbbell's labels", () => {
  const overlap = (a: [number, number], b: [number, number]) => a[0] < b[1] && b[0] < a[1];

  it("sit outside the dots on their own line when there is room", () => {
    const places = dumbbellLabels(150, 200, 80, 90, 8, 400);
    expect(places).toEqual({ low: { x: 139, anchor: "end", line: 0 }, high: { x: 211, anchor: "start", line: 0 } });
  });

  it("go above a dot with no room beside it, and never meet", () => {
    for (const [low, high] of [
      [10, 30],
      [10, 290],
      [200, 300],
      [12, 296],
      [60, 70],
    ] as const) {
      const lowWidth = textWidth("one frame 2%");
      const highWidth = textWidth("two frames 31%");
      const places = dumbbellLabels(low, high, lowWidth, highWidth, 8, 310);
      const a = labelSpan(places.low, lowWidth);
      const b = labelSpan(places.high, highWidth);
      if (places.low.line === places.high.line) expect(overlap(a, b), `${low}, ${high}`).toBe(false);
      expect(a[0]).toBeGreaterThanOrEqual(8);
    }
  });

  it("send the higher label below the line when both would sit above it and meet", () => {
    const places = dumbbellLabels(12, 100, 84, 98, 8, 120);
    expect(places.low).toEqual({ x: 8, anchor: "start", line: -1 });
    expect(places.high.line).toBe(1);
  });
});
