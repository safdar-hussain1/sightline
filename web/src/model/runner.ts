/**
 * Running the steering model in the browser.
 *
 * The model is one ONNX graph from uint8 frame stacks to two outputs: the
 * steering angle in degrees (`steering_deg`, N x 1) and the saliency map
 * (`saliency`, N x 66 x 200). onnxruntime-web runs it with its WebAssembly
 * provider, pinned to one thread: GitHub Pages sends no cross-origin
 * isolation headers, so the page has no SharedArrayBuffer and threads cannot
 * be had -- asking for them only makes the runtime probe and fall back. The
 * wasm binary is the one Vite bundles beside the app, so the page never
 * fetches a runtime from anywhere else.
 */

import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";

import { fetchOk } from "../data/load.ts";

/** The model's input frame, as the sprites carry it. */
export const INPUT_H = 66;
export const INPUT_W = 200;

const INPUT_NAME = "frames";
const ANGLE_OUTPUT = "steering_deg";
const SALIENCY_OUTPUT = "saliency";

function configureRuntime(): void {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = { wasm: wasmUrl };
}

/** The number of threads the runtime is configured to use (always one here). */
export function threadsInUse(): number {
  return ort.env.wasm.numThreads ?? 1;
}

/** Download the model at `url` and open a session on it. A model that is not
 * on the server fails with an `AssetError` of kind "missing". */
export async function createSession(url: string): Promise<ort.InferenceSession> {
  const bytes = new Uint8Array(await (await fetchOk(url)).arrayBuffer());
  configureRuntime();
  return ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

export interface Outputs {
  /** One angle per stack, degrees. */
  readonly angles: Float32Array;
  /** One 66 x 200 map per stack, row-major, values in [0, 1]. */
  readonly saliency: Float32Array;
}

/** Run `n` frame stacks (`n` x 66 x 200 x `channels` bytes, NHWC) through the
 * model. */
export async function infer(
  session: ort.InferenceSession,
  frames: Uint8Array,
  n: number,
  channels: number,
): Promise<Outputs> {
  const size = n * INPUT_H * INPUT_W * channels;
  if (frames.length !== size) {
    throw new Error(`${n} stacks of ${INPUT_W} x ${INPUT_H} x ${channels} are ${size} bytes, not ${frames.length}`);
  }
  const input = new ort.Tensor("uint8", frames, [n, INPUT_H, INPUT_W, channels]);
  const outputs = await session.run({ [INPUT_NAME]: input });
  const angles = outputs[ANGLE_OUTPUT]?.data;
  const saliency = outputs[SALIENCY_OUTPUT]?.data;
  if (!(angles instanceof Float32Array) || !(saliency instanceof Float32Array)) {
    throw new Error(`the model did not return float ${ANGLE_OUTPUT} and ${SALIENCY_OUTPUT} outputs`);
  }
  return { angles, saliency };
}
