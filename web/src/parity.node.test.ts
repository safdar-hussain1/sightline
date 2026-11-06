// @vitest-environment node
/**
 * The published models, run by the page's own runtime outside a browser.
 *
 * onnxruntime-web's WebAssembly build -- the very bundle and wasm binary the
 * page loads -- runs here in Node, on one thread, on the self-test's raw frame
 * stacks (data/selftest/frames.bin), through the page's own `infer`. Both
 * models' angles and saliency maps must match the outputs onnxruntime computed
 * in Python (data/selftest/expected.json) within the tolerances that file
 * records. Everything is read from the published files beside the page, so it
 * needs no browser and no network.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import * as ort from "onnxruntime-web/wasm";

import { MODEL_KEYS, type ModelKey } from "./data/types.ts";
import { modelMetadata } from "./model/metadata.ts";
import { compareOutputs, describeComparison, parseSelftestInputs, type SelftestInputs } from "./model/parity.ts";
import { infer } from "./model/runner.ts";

declare const process: {
  getBuiltinModule(id: "node:fs"): {
    readFileSync(path: URL): Uint8Array;
    readFileSync(path: URL, encoding: "utf8"): string;
  };
};
const fs = process.getBuiltinModule("node:fs");
const published = (path: string) => new URL(`../public/${path}`, import.meta.url);
const bytesOf = (path: string) => new Uint8Array(fs.readFileSync(published(path)));
const jsonOf = (path: string): Record<string, any> => JSON.parse(fs.readFileSync(published(path), "utf8"));

const results = jsonOf("data/results.json");
const RUNS: Readonly<Record<ModelKey, string>> = { sightline: results["shipped"]["run"], shuffled: results["twin"]["run"] };

let inputs: SelftestInputs;
const sessions = new Map<ModelKey, ort.InferenceSession>();

beforeAll(async () => {
  const selftest = results["assets"]["selftest"];
  inputs = parseSelftestInputs(jsonOf(selftest["expected"]), bytesOf(selftest["frames"]), selftest["expected"]);
  // The page's settings: the WebAssembly provider on one thread, the graph
  // optimised as the page has it (see createSession in model/runner.ts).
  ort.env.wasm.numThreads = 1;
  for (const key of MODEL_KEYS) {
    const session = await ort.InferenceSession.create(bytesOf(results["assets"]["models"][key]), {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    sessions.set(key, session);
  }
}, 60_000);

afterAll(async () => {
  for (const session of sessions.values()) await session.release();
});

describe("the published models in onnxruntime-web, outside a browser", () => {
  it("are the runs results.json names", () => {
    for (const key of MODEL_KEYS) {
      expect(modelMetadata(bytesOf(results["assets"]["models"][key])).get("run")).toBe(RUNS[key]);
    }
  });

  it("run on one thread", () => {
    expect(ort.env.wasm.numThreads).toBe(1);
  });

  for (const key of MODEL_KEYS) {
    it(`${key}: angles and saliency on the self-test stacks match onnxruntime in Python`, async () => {
      const session = sessions.get(key) as ort.InferenceSession;
      const outputs = await infer(session, inputs.frames, inputs.n, inputs.channels);
      const { angles, saliency } = compareOutputs(outputs, inputs, key);
      console.info(`${key} (${RUNS[key]}): angles ${describeComparison(angles, "°")}; saliency ${describeComparison(saliency, "")}`);
      expect(angles.problem).toBe("");
      expect(saliency.problem).toBe("");
      expect(angles.count).toBe(inputs.n);
      expect(saliency.count).toBe(inputs.n * 66 * 200);
    });
  }

  it("give each model its own answers: the twin's references fail the shipped model", async () => {
    const outputs = await infer(sessions.get("sightline") as ort.InferenceSession, inputs.frames, inputs.n, inputs.channels);
    expect(compareOutputs(outputs, inputs, "shuffled").angles.ok).toBe(false);
  });
});
