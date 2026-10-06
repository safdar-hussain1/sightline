import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  HEAT_FLOOR,
  HEAT_GAMMA,
  HEAT_STOPS,
  heatImageFromBytes,
  heatLUT,
  quantizeSaliency,
  relativeLuminance,
  saliencyToImageData,
  stopIndex,
} from "./saliency.ts";

/** Vitest blanks stylesheets, even imported as text, so the tie to the page's
 * colour tokens reads the file through Node, which runs these tests. */
declare const process: {
  getBuiltinModule(id: "node:fs"): { readFileSync(path: URL, encoding: "utf8"): string };
};
const styles = process.getBuiltinModule("node:fs").readFileSync(new URL("../styles.css", import.meta.url), "utf8");

/** The browser's ImageData, as far as these functions use it. */
class FakeImageData {
  constructor(
    readonly data: Uint8ClampedArray,
    readonly width: number,
    readonly height: number,
  ) {}
}

beforeEach(() => {
  vi.stubGlobal("ImageData", FakeImageData);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const rgb = (lut: Uint8ClampedArray, i: number) => [lut[4 * i], lut[4 * i + 1], lut[4 * i + 2]];
const hex = (h: string) => [1, 3, 5].map((at) => Number.parseInt(h.slice(at, at + 2), 16));

describe("heatLUT", () => {
  it("has 256 opaque RGBA entries", () => {
    const lut = heatLUT();
    expect(lut).toHaveLength(256 * 4);
    for (let i = 0; i < 256; i += 1) expect(lut[4 * i + 3]).toBe(255);
  });

  it("is the page's heat ramp: the --heat-0..4 stops in styles.css, in order", () => {
    const tokens = [...styles.matchAll(/--heat-(\d):\s*(#[0-9a-f]{6})\s*;/gi)];
    // Each stop is declared once, for both themes.
    expect(tokens.map((m) => Number(m[1]))).toEqual([0, 1, 2, 3, 4]);
    expect(tokens.map((m) => (m[2] ?? "").toLowerCase())).toEqual([...HEAT_STOPS]);
    const lut = heatLUT();
    HEAT_STOPS.forEach((stop, k) => {
      expect(rgb(lut, stopIndex(k)), `stop ${k} at ${stopIndex(k)}`).toEqual(hex(stop));
    });
    expect(stopIndex(0)).toBe(0);
    expect(stopIndex(HEAT_STOPS.length - 1)).toBe(255);
  });

  it("rises monotonically in luminance, so it reads as brightness without colour", () => {
    const lut = heatLUT();
    let previous = -1;
    for (let i = 0; i < 256; i += 1) {
      const [r = 0, g = 0, b = 0] = rgb(lut, i);
      const y = relativeLuminance(r, g, b);
      expect(y, `entry ${i}`).toBeGreaterThanOrEqual(previous);
      previous = y;
    }
    expect(relativeLuminance(255, 255, 255)).toBeCloseTo(1, 10);
  });

  it("is built once and shared", () => {
    expect(heatLUT()).toBe(heatLUT());
  });
});

describe("saliencyToImageData", () => {
  it("maps no attention to the ramp's first entry and full attention to its last", () => {
    const lut = heatLUT();
    const sal = new Float32Array(200 * 66);
    sal[1] = 1;
    const image = saliencyToImageData(sal);
    expect([image.width, image.height]).toEqual([200, 66]);
    expect([...image.data.slice(0, 3)]).toEqual(rgb(lut, 0));
    expect([...image.data.slice(4, 7)]).toEqual(rgb(lut, 255));
  });

  it("covers full attention at the given opacity, and no attention at a fraction of it", () => {
    const image = saliencyToImageData(new Float32Array([0, 0.04, 0.25, 1]), 4, 1, 0.6);
    const alphas = [...image.data].filter((_, i) => i % 4 === 3);
    expect(alphas[3]).toBe(153);
    expect(alphas[0]).toBe(Math.round(153 * HEAT_FLOOR));
    expect(alphas[0]).toBeGreaterThan(0); // the ignored road is dimmed, not left untouched
    expect(alphas).toEqual([...alphas].sort((a, b) => a - b)); // more attention, more cover
    expect(alphas[1]).toBeLessThan(alphas[2] as number);
  });

  it("clamps values outside [0, 1] and draws a missing value as no attention", () => {
    const lut = heatLUT();
    const image = saliencyToImageData(new Float32Array([-0.5, 7, Number.NaN]), 3, 1);
    expect([...image.data.slice(0, 3)]).toEqual(rgb(lut, 0));
    expect([...image.data.slice(4, 7)]).toEqual(rgb(lut, 255));
    expect([...image.data.slice(8, 11)]).toEqual(rgb(lut, 0));
  });

  it("places attention on the ramp through the display transfer, rising with attention", () => {
    const lut = heatLUT();
    const values = [0.01, 0.04, 0.15, 0.25, 0.45, 0.8];
    const image = saliencyToImageData(new Float32Array(values), values.length, 1);
    values.forEach((v, i) => {
      expect([...image.data.slice(4 * i, 4 * i + 3)], `attention ${v}`).toEqual(
        // The map is float32: the expectation starts from the stored value.
        rgb(lut, Math.round(Math.fround(v) ** HEAT_GAMMA * 255)),
      );
    });
    expect(HEAT_GAMMA).toBeGreaterThan(0);
    expect(HEAT_GAMMA).toBeLessThanOrEqual(1);
    // The maps are sparse: a typical attention of 0.04 must land well up the
    // ramp, not in its first few entries as a linear mapping would put it.
    expect(quantizeSaliency(new Float32Array([0.04]))[0]).toBeGreaterThanOrEqual(256 / 8);
  });

  it("refuses a map of the wrong size", () => {
    expect(() => saliencyToImageData(new Float32Array(10), 200, 66)).toThrow(/13200/);
  });

  it("draws the same picture from the stored bytes as from the floats", () => {
    const sal = new Float32Array(Array.from({ length: 300 }, (_, i) => (i * 37) % 101 / 100));
    const fromFloats = saliencyToImageData(sal, 30, 10);
    const fromBytes = heatImageFromBytes(quantizeSaliency(sal), 30, 10);
    expect([...fromBytes.data]).toEqual([...fromFloats.data]);
  });
});
