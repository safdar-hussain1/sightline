/**
 * Constants the page shares with the Python engine, checked against the
 * engine's own source so the two cannot drift apart: the crop the model sees,
 * the size of its input, and the bins that sort frames into straight, gentle
 * and sharp.
 */

import { describe, expect, it } from "vitest";

import { CROP_ROWS } from "./cockpit/geometry.ts";
import { REGIME_BINS_DEG } from "./cockpit/strip.ts";
import { INPUT_H, INPUT_W } from "./model/runner.ts";

/** Node runs these tests; the page's own TypeScript knows nothing of it. */
declare const process: {
  getBuiltinModule(id: "node:fs"): { readFileSync(path: URL, encoding: "utf8"): string };
};

const engine = (path: string) =>
  process.getBuiltinModule("node:fs").readFileSync(new URL(`../../src/sightline/${path}`, import.meta.url), "utf8");

/** The number assigned to `name` at the top level of a Python source. */
function constant(source: string, name: string): number {
  const match = new RegExp(`^${name}\\s*=\\s*([\\d.]+)\\s*$`, "m").exec(source);
  expect(match, `${name} in the engine source`).not.toBeNull();
  return Number(match?.[1]);
}

describe("the engine's constants", () => {
  it("crop and input size: the bottom rows of the camera frame, scaled to the model's input", () => {
    const preprocess = engine("data/preprocess.py");
    expect(CROP_ROWS).toBe(constant(preprocess, "CROP_BOTTOM"));
    expect(INPUT_H).toBe(constant(preprocess, "INPUT_HEIGHT"));
    expect(INPUT_W).toBe(constant(preprocess, "INPUT_WIDTH"));
  });

  it("regime bins: the |driver angle| boundaries of straight, gentle and sharp", () => {
    const match = /^REGIME_BINS\s*=\s*\(([\d.]+),\s*([\d.]+)\)/m.exec(engine("evaluate.py"));
    expect(match).not.toBeNull();
    expect([...REGIME_BINS_DEG]).toEqual([Number(match?.[1]), Number(match?.[2])]);
  });
});
