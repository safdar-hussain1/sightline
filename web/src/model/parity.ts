/**
 * How closely this runtime's model outputs match the reference ones.
 *
 * The site data carries the outputs onnxruntime computed in Python on the very
 * model files the page loads: an angle for every scored frame of each clip, in
 * its clip.json, and for eight raw frame stacks (data/selftest/frames.bin)
 * each model's angles and saliency maps, in data/selftest/expected.json, the
 * maps as base64 little-endian float32. A comparison passes only when there are
 * as many values as references and every one is a finite number within the
 * tolerance of its reference: a NaN, an infinity or a missing value is never
 * within any tolerance.
 */

import { AssetError } from "../data/load.ts";
import { MODEL_KEYS, type ModelKey } from "../data/types.ts";
import type { Bytes } from "../cockpit/sprites.ts";
import { INPUT_H, INPUT_W, type Outputs } from "./runner.ts";

export interface Comparison {
  readonly ok: boolean;
  /** How many values were compared. */
  readonly count: number;
  /** The largest |actual - expected|, infinite where either side is not a
   * finite number; NaN when nothing was compared. */
  readonly maxDiff: number;
  /** The index of the first value with the largest difference; -1 when
   * nothing was compared. */
  readonly worst: number;
  readonly tolerance: number;
  /** Why the comparison failed, in words; empty when it passed. */
  readonly problem: string;
}

/** A difference or a tolerance in two significant digits. */
function figure(x: number): string {
  return Number.isFinite(x) ? String(Number(x.toPrecision(2))) : String(x);
}

/** Every value of `actual` against the value of `expected` at the same index. */
export function compareWithin(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance: number): Comparison {
  const failed = (problem: string, maxDiff = Number.NaN, worst = -1, count = 0): Comparison => ({
    ok: false,
    count,
    maxDiff,
    worst,
    tolerance,
    problem,
  });
  if (!(Number.isFinite(tolerance) && tolerance >= 0)) {
    return failed(`the tolerance ${tolerance} is not a finite number of at least 0`);
  }
  if (actual.length !== expected.length) {
    return failed(`${actual.length} values where ${expected.length} were expected`);
  }
  if (expected.length === 0) {
    return failed("there was nothing to compare");
  }
  let maxDiff = -1;
  let worst = -1;
  for (let i = 0; i < expected.length; i += 1) {
    const a = actual[i] as number;
    const e = expected[i] as number;
    const diff = Number.isFinite(a) && Number.isFinite(e) ? Math.abs(a - e) : Number.POSITIVE_INFINITY;
    if (diff > maxDiff) {
      maxDiff = diff;
      worst = i;
    }
  }
  const count = expected.length;
  if (maxDiff === Number.POSITIVE_INFINITY) {
    return failed(`value ${worst} or its reference is not a finite number`, maxDiff, worst, count);
  }
  if (maxDiff > tolerance) {
    return failed(`off by up to ${figure(maxDiff)} at ${worst}, more than the ${figure(tolerance)} allowed`, maxDiff, worst, count);
  }
  return { ok: true, count, maxDiff, worst, tolerance, problem: "" };
}

/** "420 values, max |Δ| 0.00002° at 17", or what went wrong. */
export function describeComparison(result: Comparison, unit: string): string {
  if (!result.ok && result.count === 0) return result.problem;
  const found = `${result.count} value${result.count === 1 ? "" : "s"}, max |Δ| ${figure(result.maxDiff)}${unit} at ${result.worst}`;
  return result.ok ? found : `${found}: ${result.problem}`;
}

/** Little-endian float32 values from base64. */
export function float32FromBase64(text: string): Float32Array {
  const binary = atob(text);
  if (binary.length % 4 !== 0) {
    throw new Error(`${binary.length} bytes are not a whole number of float32 values`);
  }
  const view = new DataView(new ArrayBuffer(binary.length));
  for (let i = 0; i < binary.length; i += 1) view.setUint8(i, binary.charCodeAt(i));
  const values = new Float32Array(binary.length / 4);
  for (let i = 0; i < values.length; i += 1) values[i] = view.getFloat32(4 * i, true);
  return values;
}

/** data/selftest/: raw frame stacks, and each model's reference outputs on them. */
export interface SelftestInputs {
  readonly n: number;
  readonly channels: number;
  /** `n` stacks of 66 x 200 x `channels` bytes, NHWC. */
  readonly frames: Bytes;
  readonly angles: Readonly<Record<ModelKey, readonly number[]>>;
  /** `n` maps of 66 x 200 per model, row-major. */
  readonly saliency: Readonly<Record<ModelKey, Float32Array>>;
  readonly tolerance: { readonly angle_deg: number; readonly saliency: number };
}

type Json = Readonly<Record<string, unknown>>;

function object(value: unknown): Json | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Json) : null;
}

function sameList(value: unknown, wanted: readonly number[]): boolean {
  return Array.isArray(value) && value.length === wanted.length && value.every((v, i) => v === wanted[i]);
}

/** expected.json (as parsed) and frames.bin, checked against each other.
 * Throws an `AssetError` of kind "invalid" naming the first key that is wrong. */
export function parseSelftestInputs(json: unknown, frames: Bytes, url: string): SelftestInputs {
  const fail = (key: string, wanted: string): never => {
    throw new AssetError(`${url}: ${key} must be ${wanted}`, "invalid", url);
  };
  const o = object(json) ?? fail("the file", "an object");
  const shape = o["shape"];
  const [n, , , channels] = Array.isArray(shape) ? (shape as unknown[]) : [];
  if (
    !Number.isInteger(n) || (n as number) < 1 ||
    !Number.isInteger(channels) || (channels as number) < 3 || (channels as number) % 3 !== 0 ||
    !sameList(shape, [n as number, INPUT_H, INPUT_W, channels as number])
  ) {
    fail("shape", `[n, ${INPUT_H}, ${INPUT_W}, channels], with channels a multiple of 3`);
  }
  const count = n as number;
  const depth = channels as number;
  if (o["dtype"] !== "uint8") fail("dtype", '"uint8"');
  if (frames.length !== count * INPUT_H * INPUT_W * depth) {
    fail("shape", `the shape of the ${frames.length} bytes of the frames file`);
  }
  if (!sameList(o["saliency_shape"], [count, INPUT_H, INPUT_W])) fail("saliency_shape", `[${count}, ${INPUT_H}, ${INPUT_W}]`);
  const angles = object(o["angles"]) ?? fail("angles", "an object");
  const maps = object(o["saliency_f32_b64"]) ?? fail("saliency_f32_b64", "an object");
  const tolerance = object(o["tolerance"]) ?? fail("tolerance", "an object");
  const positive = (key: "angle_deg" | "saliency"): number => {
    const value = tolerance[key];
    return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fail(`tolerance.${key}`, "a positive number");
  };
  const tolerances = { angle_deg: positive("angle_deg"), saliency: positive("saliency") };
  const perModel = MODEL_KEYS.map((key) => {
    const list = angles[key];
    if (!Array.isArray(list) || list.length !== count || !list.every((v) => typeof v === "number" && Number.isFinite(v))) {
      fail(`angles.${key}`, `a list of ${count} numbers`);
    }
    const text = maps[key];
    let decoded: Float32Array | null = null;
    try {
      decoded = typeof text === "string" ? float32FromBase64(text) : null;
    } catch {
      decoded = null;
    }
    if (decoded === null || decoded.length !== count * INPUT_H * INPUT_W) {
      fail(`saliency_f32_b64.${key}`, `base64 float32 maps of ${count} x ${INPUT_H} x ${INPUT_W}`);
    }
    return [key, list as number[], decoded as Float32Array] as const;
  });
  return {
    n: count,
    channels: depth,
    frames,
    angles: Object.fromEntries(perModel.map(([key, list]) => [key, list])) as Record<ModelKey, number[]>,
    saliency: Object.fromEntries(perModel.map(([key, , map]) => [key, map])) as Record<ModelKey, Float32Array>,
    tolerance: tolerances,
  };
}

/** A model's outputs on the self-test stacks against its references. */
export function compareOutputs(outputs: Outputs, inputs: SelftestInputs, model: ModelKey): { angles: Comparison; saliency: Comparison } {
  return {
    angles: compareWithin(outputs.angles, inputs.angles[model], inputs.tolerance.angle_deg),
    saliency: compareWithin(outputs.saliency, inputs.saliency[model], inputs.tolerance.saliency),
  };
}
