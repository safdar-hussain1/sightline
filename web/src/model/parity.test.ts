import { describe, expect, it } from "vitest";

import { AssetError } from "../data/load.ts";
import { compareOutputs, compareWithin, describeComparison, float32FromBase64, parseSelftestInputs } from "./parity.ts";

/** Little-endian float32 bytes of `values`, in base64: how the site data writes them. */
function b64(values: readonly number[]): string {
  const view = new DataView(new ArrayBuffer(4 * values.length));
  values.forEach((v, i) => view.setFloat32(4 * i, v, true));
  let binary = "";
  for (const byte of new Uint8Array(view.buffer)) binary += String.fromCharCode(byte);
  return btoa(binary);
}

describe("compareWithin", () => {
  it("passes values within the tolerance, and reports the largest difference and where it is", () => {
    const result = compareWithin([1, 2.0004, 3], [1, 2, 3.0001], 0.001);
    expect(result.ok).toBe(true);
    expect(result.count).toBe(3);
    expect(result.worst).toBe(1);
    expect(result.maxDiff).toBeCloseTo(0.0004, 12);
  });

  it("fails once one value is past the tolerance, naming it", () => {
    const result = compareWithin([1, 2, 3.5], [1, 2, 3], 0.001);
    expect(result.ok).toBe(false);
    expect(result.worst).toBe(2);
    expect(result.maxDiff).toBe(0.5);
    expect(result.problem).toMatch(/0\.5.*at 2.*0\.001/);
  });

  it("counts a difference equal to the tolerance as within it", () => {
    expect(compareWithin([0.5], [0], 0.5).ok).toBe(true);
    expect(compareWithin([0.5000001], [0], 0.5).ok).toBe(false);
  });

  it("never passes a NaN or an infinity, on either side, whatever the tolerance", () => {
    for (const [actual, expected] of [
      [[Number.NaN], [0]],
      [[0], [Number.NaN]],
      [[Number.POSITIVE_INFINITY], [Number.POSITIVE_INFINITY]],
      [[Number.NEGATIVE_INFINITY], [0]],
      [[1, Number.NaN, 1], [1, 1, 1]],
    ] as const) {
      const result = compareWithin(actual, expected, 1e9);
      expect(result.ok, `${actual} against ${expected}`).toBe(false);
      expect(result.maxDiff).toBe(Number.POSITIVE_INFINITY);
      expect(result.problem).toMatch(/not a finite number/);
    }
    expect(compareWithin([1, Number.NaN, 1], [1, 1, 1], 1).worst).toBe(1);
  });

  it("fails when the lengths differ, even if every shared value agrees", () => {
    const result = compareWithin([1, 2], [1, 2, 3], 1);
    expect(result.ok).toBe(false);
    expect(result.problem).toMatch(/2 values where 3/);
    expect(compareWithin([1, 2, 3], [1, 2], 1).ok).toBe(false);
  });

  it("fails when there is nothing to compare", () => {
    expect(compareWithin([], [], 1).ok).toBe(false);
    expect(compareWithin([], [], 1).problem).toMatch(/nothing/);
  });

  it("refuses a tolerance that is not a finite number of at least zero", () => {
    for (const tolerance of [Number.NaN, -1, Number.POSITIVE_INFINITY]) {
      expect(compareWithin([1], [1], tolerance).ok, String(tolerance)).toBe(false);
    }
    expect(compareWithin([1], [1], 0).ok).toBe(true);
  });

  it("reads typed arrays as well as plain lists", () => {
    expect(compareWithin(Float32Array.of(0.25, 0.5), [0.25, 0.5], 0).ok).toBe(true);
  });

  it("describes what it found in a few words", () => {
    expect(describeComparison(compareWithin([1, 2.00002], [1, 2], 0.001), "°")).toMatch(/2 values.*0\.00002°.*at 1/);
    expect(describeComparison(compareWithin([1], [1], 0.001), "")).toMatch(/max \|Δ\| 0\b/);
  });
});

describe("float32FromBase64", () => {
  it("decodes little-endian float32 values exactly, signed zero and the extremes included", () => {
    const values = [1.5, -2.25, 0, 0.1, 1.401298464324817e-45, 3.4028234663852886e38, -1e-30];
    const decoded = float32FromBase64(b64([...values, -0]));
    expect(decoded).toHaveLength(values.length + 1);
    values.forEach((v, i) => expect(decoded[i]).toBe(Math.fround(v)));
    expect(Object.is(decoded[values.length], -0)).toBe(true);
  });

  it("refuses text that is not base64, or bytes that are not whole float32 values", () => {
    expect(() => float32FromBase64("not base64!")).toThrow();
    expect(() => float32FromBase64(btoa("abcdef"))).toThrow(/6 bytes/);
  });
});

/** A self-test file for `n` stacks of `channels` channels. */
function selftestJson(n: number, channels: number, overrides: Record<string, unknown> = {}): Record<string, unknown> {
  const maps = new Array<number>(n * 66 * 200).fill(0.25);
  return {
    shape: [n, 66, 200, channels],
    dtype: "uint8",
    angles: { sightline: new Array<number>(n).fill(1.5), shuffled: new Array<number>(n).fill(-2) },
    saliency_shape: [n, 66, 200],
    saliency_f32_b64: { sightline: b64(maps), shuffled: b64(maps.map((v) => v / 2)) },
    tolerance: { angle_deg: 0.001, saliency: 0.001 },
    ...overrides,
  };
}

describe("parseSelftestInputs", () => {
  const frames = (n: number, channels: number) => new Uint8Array(n * 66 * 200 * channels);

  it("reads the stacks, each model's angles and its decoded saliency maps", () => {
    const inputs = parseSelftestInputs(selftestJson(2, 6), frames(2, 6), "expected.json");
    expect(inputs.n).toBe(2);
    expect(inputs.channels).toBe(6);
    expect(inputs.angles.shuffled).toEqual([-2, -2]);
    expect(inputs.saliency.sightline).toHaveLength(2 * 66 * 200);
    expect(inputs.saliency.shuffled[0]).toBe(0.125);
    expect(inputs.tolerance).toEqual({ angle_deg: 0.001, saliency: 0.001 });
  });

  it("refuses frames that are not the shape the file states", () => {
    expect(() => parseSelftestInputs(selftestJson(2, 6), frames(2, 3), "expected.json")).toThrow(AssetError);
    expect(() => parseSelftestInputs(selftestJson(2, 6, { shape: [2, 66, 199, 6] }), frames(2, 6), "e")).toThrow(/shape/);
    expect(() => parseSelftestInputs(selftestJson(2, 6, { dtype: "float32" }), frames(2, 6), "e")).toThrow(/dtype/);
  });

  it("refuses angles or maps that do not cover every stack", () => {
    const short = { sightline: [1], shuffled: [1, 2] };
    expect(() => parseSelftestInputs(selftestJson(2, 6, { angles: short }), frames(2, 6), "e")).toThrow(/angles\.sightline/);
    const maps = { sightline: b64([0.5]), shuffled: b64([0.5]) };
    expect(() => parseSelftestInputs(selftestJson(2, 6, { saliency_f32_b64: maps }), frames(2, 6), "e")).toThrow(/saliency/);
    expect(() => parseSelftestInputs(selftestJson(2, 6, { saliency_shape: [2, 66] }), frames(2, 6), "e")).toThrow(/saliency_shape/);
  });

  it("refuses a missing or unusable tolerance", () => {
    for (const tolerance of [undefined, { angle_deg: 0.001 }, { angle_deg: -1, saliency: 0.001 }, { angle_deg: "0.1", saliency: 0.1 }]) {
      expect(() => parseSelftestInputs(selftestJson(1, 6, { tolerance }), frames(1, 6), "e")).toThrow(/tolerance/);
    }
  });
});

describe("compareOutputs", () => {
  it("compares a model's angles and maps with its own references, each against its own tolerance", () => {
    const inputs = parseSelftestInputs(selftestJson(1, 6), new Uint8Array(66 * 200 * 6), "e");
    const close = { angles: Float32Array.of(1.5004), saliency: new Float32Array(66 * 200).fill(0.2504) };
    const both = compareOutputs(close, inputs, "sightline");
    expect(both.angles.ok && both.saliency.ok).toBe(true);
    const other = compareOutputs(close, inputs, "shuffled");
    expect(other.angles.ok).toBe(false);
    expect(other.saliency.ok).toBe(false);
  });
});
