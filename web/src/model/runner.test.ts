import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";

import { AssetError } from "../data/load.ts";
import { INPUT_H, INPUT_W, createSession, infer, threadsInUse } from "./runner.ts";

/** What the mocked runtime saw: the options of each session it created, and
 * the feeds of each run. */
const seen = vi.hoisted(() => ({
  created: [] as Array<{ model: unknown; options: Record<string, unknown>; threads: unknown; paths: unknown }>,
  feeds: [] as Array<Record<string, { type: string; data: Uint8Array; dims: number[] }>>,
}));

vi.mock("onnxruntime-web/wasm", () => {
  const env = { wasm: { numThreads: 0 as number, wasmPaths: undefined as unknown } };
  class Tensor {
    constructor(
      readonly type: string,
      readonly data: Uint8Array,
      readonly dims: number[],
    ) {}
  }
  const session = {
    inputNames: ["frames"],
    outputNames: ["steering_deg", "saliency"],
    run: async (feeds: Record<string, Tensor>) => {
      seen.feeds.push(feeds);
      const n = feeds["frames"]?.dims[0] ?? 0;
      return {
        steering_deg: { data: Float32Array.from({ length: n }, (_, i) => 10 * i + 0.5) },
        saliency: { data: new Float32Array(n * 66 * 200).fill(0.25) },
      };
    },
  };
  return {
    env,
    Tensor,
    InferenceSession: {
      create: async (model: unknown, options: Record<string, unknown>) => {
        seen.created.push({ model, options, threads: env.wasm.numThreads, paths: env.wasm.wasmPaths });
        return session;
      },
    },
  };
});

beforeEach(() => {
  seen.created.length = 0;
  seen.feeds.length = 0;
  vi.stubGlobal("fetch", async (url: string) =>
    url.endsWith("missing.onnx") ? new Response("", { status: 404 }) : new Response(new Uint8Array([7, 8, 9])),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("createSession", () => {
  it("runs single-threaded on the wasm provider, from the bundled wasm binary", async () => {
    await createSession("models/sightline.onnx");
    expect(seen.created).toHaveLength(1);
    const [{ model, options, threads, paths }] = seen.created as [(typeof seen.created)[number]];
    // GitHub Pages sends no cross-origin isolation headers, so no shared memory, so one thread.
    expect(threads).toBe(1);
    expect(paths).toEqual({ wasm: wasmUrl });
    expect(String(wasmUrl)).toMatch(/ort-wasm-simd-threaded.*\.wasm/);
    expect(options).toEqual({ executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    expect([...(model as Uint8Array)]).toEqual([7, 8, 9]);
    expect(threadsInUse()).toBe(1);
    expect(ort.env.wasm.numThreads).toBe(1);
  });

  it("reports a model that is not on the server as missing", async () => {
    await expect(createSession("models/missing.onnx")).rejects.toBeInstanceOf(AssetError);
    await expect(createSession("models/missing.onnx")).rejects.toMatchObject({ kind: "missing" });
    expect(seen.created).toHaveLength(0);
  });
});

describe("infer", () => {
  it("feeds uint8 frame stacks as `frames` and reads the angle and saliency outputs", async () => {
    const session = await createSession("models/sightline.onnx");
    const n = 2;
    const channels = 6;
    const frames = new Uint8Array(n * INPUT_H * INPUT_W * channels).fill(3);
    const { angles, saliency } = await infer(session, frames, n, channels);
    const feed = seen.feeds[0]?.["frames"];
    expect(feed?.type).toBe("uint8");
    expect(feed?.dims).toEqual([n, 66, 200, channels]);
    expect(feed?.data).toBe(frames);
    expect([...angles]).toEqual([0.5, 10.5]);
    expect(saliency).toHaveLength(n * 66 * 200);
  });

  it("refuses frames that are not n stacks of the model's input size", async () => {
    const session = await createSession("models/sightline.onnx");
    await expect(infer(session, new Uint8Array(10), 1, 6)).rejects.toThrow(/79200/);
  });

  it("runs one inference at a time, across sessions, in the order asked, and survives a failed run", async () => {
    const events: string[] = [];
    let active = 0;
    let widest = 0;
    const gates: Array<() => void> = [];
    const fake = (name: string, fail = false) =>
      ({
        run: async () => {
          events.push(`start ${name}`);
          active += 1;
          widest = Math.max(widest, active);
          await new Promise<void>((resolve) => gates.push(resolve));
          active -= 1;
          events.push(`end ${name}`);
          if (fail) throw new Error(`${name} failed`);
          return { steering_deg: { data: new Float32Array(1) }, saliency: { data: new Float32Array(66 * 200) } };
        },
      }) as unknown as Parameters<typeof infer>[0];
    const frames = new Uint8Array(66 * 200 * 6);
    const a = infer(fake("a"), frames, 1, 6);
    const b = infer(fake("b", true), frames, 1, 6);
    const c = infer(fake("c"), frames, 1, 6);
    const bFailed = expect(b).rejects.toThrow("b failed");
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(events).toEqual(["start a"]);
    for (const name of ["a", "b", "c"]) {
      gates.shift()?.();
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(events.filter((event) => event.startsWith("end"))).toContain(`end ${name}`);
    }
    await Promise.all([a, bFailed, c]);
    expect(widest).toBe(1);
    expect(events).toEqual(["start a", "end a", "start b", "end b", "start c", "end c"]);
  });
});
