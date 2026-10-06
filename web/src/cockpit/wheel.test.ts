import { describe, expect, it } from "vitest";

import { WHEEL_CENTRE, errorArcPath, renderWheel, rimPoint, wheelRotationDeg, wheelSvg } from "./wheel.ts";

/** The numbers in an SVG path, in order. */
const numbers = (d: string) => (d.match(/-?\d+(?:\.\d+)?/g) ?? []).map(Number);

describe("wheelRotationDeg", () => {
  it("turns the wheel by the angle itself, past half a turn and more", () => {
    expect(wheelRotationDeg(501.8)).toBe(501.8);
    expect(wheelRotationDeg(-338.8)).toBe(-338.8);
    expect(wheelRotationDeg(12.5)).toBe(12.5);
    expect(wheelRotationDeg(0)).toBe(0);
  });
});

describe("rimPoint", () => {
  it("measures clockwise from twelve o'clock, as a steering wheel turns", () => {
    const [x0, y0] = rimPoint(0, 100, 0, 0);
    expect([x0, y0].map((v) => Math.round(v) + 0)).toEqual([0, -100]);
    const [x90, y90] = rimPoint(90, 100, 0, 0);
    expect([x90, y90].map((v) => Math.round(v) + 0)).toEqual([100, 0]); // a right turn moves the top to the right
    const [x270, y270] = rimPoint(-90, 100, 0, 0);
    expect([x270, y270].map((v) => Math.round(v) + 0)).toEqual([-100, 0]);
  });
});

describe("errorArcPath", () => {
  it("draws nothing when the model and the driver agree, or when either is missing", () => {
    expect(errorArcPath(10, 10, 100, 0, 0)).toBe("");
    expect(errorArcPath(Number.NaN, 10, 100, 0, 0)).toBe("");
    expect(errorArcPath(10, Number.POSITIVE_INFINITY, 100, 0, 0)).toBe("");
  });

  it("runs clockwise along the rim from the smaller angle to the larger", () => {
    const d = errorArcPath(30, 0, 100, 0, 0);
    expect(d.startsWith("M")).toBe(true);
    const [mx, my, rx, ry, rotation, large, sweep, ex, ey] = numbers(d);
    expect([mx, my]).toEqual([0, -100]);
    expect([rx, ry, rotation, large, sweep]).toEqual([100, 100, 0, 0, 1]);
    expect(ex).toBeCloseTo(50, 2);
    expect(ey).toBeCloseTo(-86.6, 1);
    expect(errorArcPath(0, 30, 100, 0, 0)).toBe(d); // the same gap, whichever side the model is on
  });

  it("takes the long way round for a gap over half a turn", () => {
    const [, , , , , large] = numbers(errorArcPath(-100, 100, 100, 0, 0));
    expect(large).toBe(1);
    const [, , , , , small] = numbers(errorArcPath(-80, 80, 100, 0, 0));
    expect(small).toBe(0);
  });

  it("stops short of a full circle for a gap of a whole turn or more", () => {
    const d = errorArcPath(0, 400, 100, 0, 0);
    const [mx, my, , , , large, , ex, ey] = numbers(d);
    expect(large).toBe(1);
    expect(Math.hypot((ex ?? 0) - (mx ?? 0), (ey ?? 0) - (my ?? 0))).toBeGreaterThan(0.01);
  });
});

/** Just enough of an SVG element for `renderWheel`. */
function fakeWheel() {
  const parts = new Map<string, { style: { transform: string }; attributes: Map<string, string> }>();
  for (const name of ["model", "human", "error"]) {
    parts.set(name, { style: { transform: "" }, attributes: new Map() });
  }
  const svg = {
    querySelector: (selector: string) => {
      const name = /data-wheel="?(\w+)"?/.exec(selector)?.[1] ?? "";
      const part = parts.get(name);
      return part
        ? { style: part.style, setAttribute: (k: string, v: string) => void part.attributes.set(k, v) }
        : null;
    },
  };
  return { svg: svg as unknown as SVGSVGElement, parts };
}

describe("renderWheel", () => {
  it("turns the model's wheel and the driver's ghost by their own angles", () => {
    const { svg, parts } = fakeWheel();
    renderWheel(svg, 501.8, -12.25);
    expect(parts.get("model")?.style.transform).toBe("rotate(501.8deg)");
    expect(parts.get("human")?.style.transform).toBe("rotate(-12.25deg)");
  });

  it("leaves the model's wheel alone and draws no gap when its output is not ready", () => {
    const { svg, parts } = fakeWheel();
    renderWheel(svg, 30, 10);
    renderWheel(svg, null, -4);
    expect(parts.get("model")?.style.transform).toBe("rotate(30deg)");
    expect(parts.get("human")?.style.transform).toBe("rotate(-4deg)");
    expect(parts.get("error")?.attributes.get("d")).toBe("");
  });

  it("draws the gap between them along the rim, and clears it when they agree", () => {
    const { svg, parts } = fakeWheel();
    renderWheel(svg, 20, 5);
    const d = parts.get("error")?.attributes.get("d") ?? "";
    expect(d).toMatch(/^M.*A/);
    renderWheel(svg, 7, 7);
    expect(parts.get("error")?.attributes.get("d")).toBe("");
  });
});

describe("wheelSvg", () => {
  it("holds the three parts renderWheel turns, hidden from assistive technology", () => {
    const markup = wheelSvg();
    for (const part of ["model", "human", "error"]) expect(markup).toContain(`data-wheel="${part}"`);
    expect(markup).toMatch(/^<svg\b[^>]*aria-hidden="true"/);
    expect(markup).toContain(`viewBox="0 0 ${2 * WHEEL_CENTRE} ${2 * WHEEL_CENTRE}"`);
  });
});
