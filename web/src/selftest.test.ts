// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import cockpitSource from "./cockpit/cockpit.ts?raw";
import { cockpitHtml } from "./cockpit/markup.ts";
import { stackAt, type Bytes, type DecodeCheck } from "./cockpit/sprites.ts";
import { parseClip, parseResults } from "./data/load.ts";
import { MODEL_KEYS, type ModelKey } from "./data/types.ts";
import { fmtDeg } from "./format.ts";
import mainSource from "./main.ts?raw";
import { modelMetadata } from "./model/metadata.ts";
import type { Outputs } from "./model/runner.ts";
import { CHECKS, formatVerdict, readCockpit, runSelftest, selftestRequested, type CheckResult, type Page, type Shown } from "./selftest.ts";

const TITLE = "Sightline — a steering model that shows where it looks";
const MAP = 66 * 200;

const passed = (name: string): CheckResult => ({ name, ok: true, detail: "fine" });
const failed = (name: string, detail = "wrong"): CheckResult => ({ name, ok: false, detail });

describe("formatVerdict", () => {
  it("passes as PASS k/k threads=1 when every check passed on one thread", () => {
    expect(formatVerdict(CHECKS.map(passed), 1)).toBe("PASS 6/6 threads=1");
    expect(formatVerdict([passed("a")], 1)).toBe("PASS 1/1 threads=1");
  });

  it("fails as FAIL k/m, naming the first failing check and why", () => {
    const results = [passed("sessions"), failed("metadata", "shuffled is run x"), passed("selftest-inputs"), failed("decode-hash")];
    expect(formatVerdict(results, 1)).toBe("FAIL 2/4 metadata: shuffled is run x");
  });

  it("fails on any thread count but one, even when every check passed", () => {
    expect(formatVerdict(CHECKS.map(passed), 4)).toBe("FAIL 6/6 threads=4");
    expect(formatVerdict(CHECKS.map(passed), Number.NaN)).toBe("FAIL 6/6 threads=NaN");
  });

  it("fails when no check ran", () => {
    expect(formatVerdict([], 1)).toBe("FAIL 0/0 no checks ran");
  });

  it("never starts a failing verdict with PASS, whatever the checks are called or say", () => {
    const names = ["PASS", "PASS 6/6 threads=1", "", " ", "\nPASS", "sessions"];
    for (const name of names) {
      for (const detail of ["", "PASS", "PASS 6/6 threads=1"]) {
        for (const threads of [1, 2]) {
          const verdict = formatVerdict([passed("a"), failed(name, detail), passed("b")], threads);
          expect(verdict.startsWith("FAIL 2/3 "), JSON.stringify([name, detail, threads, verdict])).toBe(true);
        }
      }
    }
  });

  it("only ever passes in one exact shape", () => {
    for (const results of [CHECKS.map(passed), [passed("x"), passed("y")]]) {
      expect(formatVerdict(results, 1)).toMatch(/^PASS (\d+)\/\1 threads=1$/);
    }
  });

  it("keeps the reason on one line, and short", () => {
    const verdict = formatVerdict([failed("clip-angles", `c2018\n\n  off by ${"9".repeat(400)}`)], 1);
    expect(verdict).not.toMatch(/\n/);
    expect(verdict).toMatch(/^FAIL 0\/1 clip-angles: c2018 off by 9+…$/);
    expect(verdict.length).toBeLessThan(200);
  });
});

describe("selftestRequested", () => {
  it("is true for selftest=1 alone", () => {
    expect(selftestRequested("?selftest=1")).toBe(true);
    expect(selftestRequested("?view=results&selftest=1")).toBe(true);
    for (const search of ["", "?selftest", "?selftest=0", "?selftest=true", "?selftest=1x", "?self-test=1", "?x=selftest=1"]) {
      expect(selftestRequested(search), search).toBe(false);
    }
  });

  it("is what main.ts asks before it imports the self-test, which it only ever does dynamically", () => {
    const lines = mainSource.split("\n").filter((line) => line.includes("selftest"));
    expect(lines.filter((line) => /^\s*import\s/.test(line))).toEqual([]);
    expect(mainSource).toMatch(/if \(new URLSearchParams\(location\.search\)\.get\("selftest"\) === "1"\) void import\("\.\/selftest\.ts"\);/);
  });
});

describe("the tab title", () => {
  let writes: string[] = [];

  beforeEach(() => {
    writes = [];
    let proto: object | null = Object.getPrototypeOf(document);
    while (proto && !Object.getOwnPropertyDescriptor(proto, "title")) proto = Object.getPrototypeOf(proto);
    const descriptor = proto ? Object.getOwnPropertyDescriptor(proto, "title") : undefined;
    descriptor?.set?.call(document, TITLE);
    Object.defineProperty(document, "title", {
      configurable: true,
      get: () => descriptor?.get?.call(document),
      set: (value: string) => {
        writes.push(value);
        descriptor?.set?.call(document, value);
      },
    });
    vi.spyOn(console, "info").mockImplementation(() => undefined);
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    delete (document as { title?: string }).title;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    history.replaceState(null, "", "/");
  });

  for (const address of ["/", "/?selftest=0", "/?selftest=true#results"]) {
    it(`is never written, and nothing is fetched, when the module loads at ${address}`, async () => {
      history.replaceState(null, "", address);
      const fetched = vi.fn(async () => new Response("{}"));
      vi.stubGlobal("fetch", fetched);
      vi.resetModules();
      await import("./selftest.ts");
      await new Promise((resolve) => setTimeout(resolve, 30));
      expect(writes).toEqual([]);
      expect(document.title).toBe(TITLE);
      expect(fetched).not.toHaveBeenCalled();
    });
  }

  it("carries the verdict when the module loads at ?selftest=1", async () => {
    history.replaceState(null, "", "/?selftest=1#results");
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new Error("offline"))));
    vi.resetModules();
    await import("./selftest.ts");
    await vi.waitFor(() => expect(document.title).toMatch(/^FAIL /));
    expect(writes[0]).toMatch(/^RUNNING /);
    expect(document.title).toMatch(/^FAIL 0\/6 sessions: results\.json did not load/);
  });
});

describe("readCockpit", () => {
  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("reads the cockpit's own markup: its state, and once it runs its clip, model and average", () => {
    document.body.innerHTML = `<section class="cockpit" id="cockpit">${cockpitHtml()}</section>`;
    expect(readCockpit()).toEqual({ state: "loading", status: "Loading the model and the footage" });
    const screen = document.querySelector<HTMLElement>(".cockpit__console") as HTMLElement;
    screen.dataset["state"] = "ready";
    // The cockpit writes the clip picker's buttons itself, under this name.
    expect(cockpitSource).toContain('input.name = "cockpit-clip"');
    document.querySelector('[data-cockpit="clips"]')?.insertAdjacentHTML(
      "beforeend",
      '<input type="radio" name="cockpit-clip" value="c2017" /><input type="radio" name="cockpit-clip" value="c2018" checked />',
    );
    (document.querySelector('[data-cockpit="average"]') as HTMLElement).textContent = "14.3°";
    expect(readCockpit()).toEqual({ state: "ready", clip: "c2018", model: "sightline", average: "14.3°" });
    screen.dataset["state"] = "unavailable";
    (document.querySelector('[data-cockpit="status"]') as HTMLElement).textContent = " The clip could not load. ";
    expect(readCockpit()).toEqual({ state: "unavailable", status: "The clip could not load." });
  });

  it("says when the page has no cockpit", () => {
    expect(readCockpit()).toEqual({ state: "absent", status: "" });
  });
});

// -- The checks, on a page made up here ---------------------------------------------------

const BASE = "https://example.test/site/";
const RUN: Readonly<Record<ModelKey, string>> = { sightline: "run-blocked", shuffled: "run-leaky" };

function varint(value: number): number[] {
  const out: number[] = [];
  let rest = value;
  while (rest >= 0x80) {
    out.push((rest % 0x80) | 0x80);
    rest = Math.floor(rest / 0x80);
  }
  return [...out, rest];
}
const text = (value: string) => [...new TextEncoder().encode(value)];
const delimited = (field: number, payload: number[]) => [...varint(field * 8 + 2), ...varint(payload.length), ...payload];
const onnx = (meta: Record<string, string>): Bytes =>
  new Uint8Array([
    ...delimited(7, [1, 2, 3]),
    ...Object.entries(meta).flatMap(([key, value]) => delimited(14, [...delimited(1, text(key)), ...delimited(2, text(value))])),
  ]);

function b64(values: ArrayLike<number>): string {
  const view = new DataView(new ArrayBuffer(4 * values.length));
  for (let i = 0; i < values.length; i += 1) view.setFloat32(4 * i, values[i] as number, true);
  let binary = "";
  for (const byte of new Uint8Array(view.buffer)) binary += String.fromCharCode(byte);
  return btoa(binary);
}

interface FakeSession {
  readonly run: string;
}

/** The made-up models: an angle from the first pixel of each frame in the
 * stack, offset by model, and a flat saliency map. */
function answer(run: string, stack: Bytes): number {
  return (stack[0] as number) / 10 - (stack[3] as number) / 20 + (run === RUN.sightline ? 0 : 40);
}
const flatMap = (run: string) => (run === RUN.sightline ? 0.5 : 0.25);

function fakeInfer(session: FakeSession, frames: Bytes, n: number, channels: number): Outputs {
  const stride = MAP * channels;
  const angles = new Float32Array(n);
  for (let i = 0; i < n; i += 1) angles[i] = answer(session.run, frames.subarray(i * stride, (i + 1) * stride));
  return { angles, saliency: new Float32Array(n * MAP).fill(flatMap(session.run)) };
}

/** One way to break the made-up page. */
interface Breaks {
  twinFile?: "the shipped file";
  wrongLags?: boolean;
  selftestAngle?: number;
  badDigestAt?: number;
  maeOff?: number;
  shown?: Shown;
  resultsMissing?: boolean;
  threads?: number;
  inferFails?: ModelKey;
  noTolerance?: boolean;
  missingModel?: ModelKey;
}

const fingerprint = (frame: Bytes) => `frame ${frame[0]} of ${frame.length}`;

function world(breaks: Breaks = {}): Page<FakeSession> {
  const leadIn = 2;
  const n = 12;
  const lags = [2];
  const frames: Bytes[] = Array.from({ length: leadIn + n }, (_, i) => new Uint8Array(MAP * 3).fill((i * 37) % 256));
  const expectedOn = (run: string) =>
    Array.from({ length: n }, (_, k) => Math.fround(answer(run, stackAt(frames, leadIn + k, lags))));
  const labels = Array.from({ length: n }, (_, k) => k - 4.5);
  const expected = { sightline: expectedOn(RUN.sightline), shuffled: expectedOn(RUN.shuffled) };
  const mae = (values: readonly number[]) => values.reduce((sum, v, k) => sum + Math.abs(v - (labels[k] as number)), 0) / n;
  const clip = {
    id: "c1",
    drive: "drive2018",
    title: "2018: minutes it never trained on",
    fps: 17.5,
    n,
    lead_in: leadIn,
    first_frame: 100,
    frame_lags: lags,
    rule: "a made-up clip",
    labels_deg: labels,
    expected,
    sprite: {
      files: ["sprite-00.jpg"],
      frames_per_sheet: 60,
      frame_w: 200,
      frame_h: 66,
      decoded_sha256: frames.map((frame, i) => (i === breaks.badDigestAt ? "another" : fingerprint(frame))),
    },
    video: "clip.mp4",
    video_size: [456, 256],
    camera_size: [455, 256],
    clip_mae_deg: {
      sightline: mae(expected.sightline) + (breaks.maeOff ?? 0),
      shuffled: mae(expected.shuffled),
      zero: labels.reduce((sum, v) => sum + Math.abs(v), 0) / n,
    },
    ...(breaks.noTolerance ? {} : { tolerance: { angle_deg: 0.001 } }),
  };
  const stacks = new Uint8Array([...stackAt(frames, 5, lags), ...stackAt(frames, 9, lags)]);
  const selftestAngles = (run: string) => [0, 1].map((i) => Math.fround(answer(run, stacks.subarray(i * MAP * 6, (i + 1) * MAP * 6))));
  const shippedAngles = selftestAngles(RUN.sightline);
  if (breaks.selftestAngle !== undefined) shippedAngles[0] = (shippedAngles[0] as number) + breaks.selftestAngle;
  const files: Record<string, unknown> = {
    "data/results.json": {
      assets: {
        models: { sightline: "models/a.onnx", shuffled: "models/b.onnx" },
        clips: { c1: "data/clips/c1/clip.json" },
        selftest: { expected: "data/selftest/expected.json", frames: "data/selftest/frames.bin" },
      },
      shipped: { run: RUN.sightline, model: "sightlinenet", protocol: "blocked" },
      twin: { run: RUN.shuffled, model: "sightlinenet" },
    },
    "data/clips/c1/clip.json": clip,
    "models/a.onnx": onnx({ run: RUN.sightline, model: "sightlinenet", frame_lags: "[2]" }),
    "models/b.onnx":
      breaks.twinFile === undefined
        ? onnx({ run: RUN.shuffled, model: "sightlinenet", frame_lags: breaks.wrongLags ? "[1]" : "[2]" })
        : onnx({ run: RUN.sightline, model: "sightlinenet", frame_lags: "[2]" }),
    "data/selftest/frames.bin": stacks,
    "data/selftest/expected.json": {
      shape: [2, 66, 200, 6],
      dtype: "uint8",
      angles: { sightline: shippedAngles, shuffled: selftestAngles(RUN.shuffled) },
      saliency_shape: [2, 66, 200],
      saliency_f32_b64: {
        sightline: b64(new Float32Array(2 * MAP).fill(0.5)),
        shuffled: b64(new Float32Array(2 * MAP).fill(0.25)),
      },
      tolerance: { angle_deg: 0.001, saliency: 0.001 },
    },
  };
  if (breaks.missingModel) delete files[breaks.missingModel === "sightline" ? "models/a.onnx" : "models/b.onnx"];
  const file = (url: string): unknown => {
    const path = url.startsWith(BASE) ? url.slice(BASE.length) : url;
    if (!(path in files)) throw new Error(`${url} is not on this server`);
    return files[path];
  };
  return {
    base: BASE,
    results: async (base) => {
      if (breaks.resultsMissing) throw new Error(`${base}data/results.json is not on this server`);
      return parseResults(file(`${base}data/results.json`), "results.json");
    },
    clip: async (url) => parseClip(file(url), url),
    bytes: async (url) => file(url) as Bytes,
    json: async (url) => file(url),
    open: async (bytes) => ({ run: modelMetadata(bytes).get("run") ?? "" }),
    infer: async (session, stack, count, channels) => {
      if (breaks.inferFails !== undefined && session.run === RUN[breaks.inferFails]) throw new Error("the runtime gave up");
      return fakeInfer(session, stack, count, channels);
    },
    threads: () => breaks.threads ?? 1,
    decode: async (decoded) => {
      expect(decoded.id).toBe("c1");
      return frames;
    },
    hashes: async (decoded, digests): Promise<DecodeCheck> => {
      const differ = decoded.flatMap((frame, i) => (fingerprint(frame) === digests[i] ? [] : [i]));
      return differ.length === 0 ? { status: "match" } : { status: "differ", frames: differ };
    },
    cockpit: async () => breaks.shown ?? { state: "ready", clip: "c1", model: "sightline", average: fmtDeg(clip.clip_mae_deg.sightline) },
    idle: async () => undefined,
  };
}

async function verdictOf(breaks: Breaks = {}): Promise<{ verdict: string; titles: string[]; results: readonly CheckResult[] }> {
  const titles: string[] = [];
  const { verdict, results } = await runSelftest(world(breaks), (title) => titles.push(title));
  return { verdict, titles, results };
}

describe("runSelftest", () => {
  beforeEach(() => {
    vi.spyOn(console, "info").mockImplementation(() => undefined);
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("passes a page whose every file and answer is the published one", async () => {
    const { verdict, titles, results } = await verdictOf();
    expect(verdict).toBe("PASS 6/6 threads=1");
    expect(results.map((r) => r.name)).toEqual([...CHECKS]);
    expect(titles[0]).toMatch(/^RUNNING 0\/6/);
    expect(titles.slice(1, -1)).toEqual(CHECKS.map((name, i) => `RUNNING ${i + 1}/6 ${name}`));
    expect(titles.at(-1)).toBe(verdict);
    expect(results.find((r) => r.name === "clip-mae")?.detail).toMatch(/consistency/);
  });

  it("fails the metadata check first when the twin's file holds the shipped model", async () => {
    const { verdict, results } = await verdictOf({ twinFile: "the shipped file" });
    expect(verdict).toMatch(/^FAIL 3\/6 metadata: shuffled is run "run-blocked", not run-leaky/);
    expect(results.filter((r) => !r.ok).map((r) => r.name)).toEqual(["metadata", "selftest-inputs", "clip-angles"]);
  });

  it("fails the metadata check on lags that are not the clips'", async () => {
    expect((await verdictOf({ wrongLags: true })).verdict).toMatch(/^FAIL \d\/6 metadata: .*lags/);
  });

  it("fails the self-test inputs on one reference angle off by more than the tolerance", async () => {
    expect((await verdictOf({ selftestAngle: 0.01 })).verdict).toMatch(/^FAIL 5\/6 selftest-inputs: sightline angles .* at 0/);
    expect((await verdictOf({ selftestAngle: 0.0005 })).verdict).toBe("PASS 6/6 threads=1");
  });

  it("fails the decode check on one frame that is not the reference decoder's", async () => {
    const { verdict } = await verdictOf({ badDigestAt: 7 });
    expect(verdict).toMatch(/^FAIL 5\/6 decode-hash: c1: frame 7 of 14/);
  });

  it("fails the clip average when it does not follow from the clip's own angles and labels", async () => {
    expect((await verdictOf({ maeOff: 1e-3 })).verdict).toMatch(/^FAIL 5\/6 clip-mae: c1 sightline/);
    expect((await verdictOf({ maeOff: 1e-12 })).verdict).toBe("PASS 6/6 threads=1");
  });

  it("fails the clip average when the cockpit shows another figure, or is not running", async () => {
    const other: Shown = { state: "ready", clip: "c1", model: "sightline", average: "99.9°" };
    expect((await verdictOf({ shown: other })).verdict).toMatch(/^FAIL 5\/6 clip-mae: the cockpit shows 99\.9°/);
    const down: Shown = { state: "unavailable", status: "This browser could not run the model." };
    expect((await verdictOf({ shown: down })).verdict).toMatch(/^FAIL 5\/6 clip-mae: the cockpit is not running \(unavailable: This browser/);
  });

  it("fails the clip angles of a model whose runs fail, and runs every other check", async () => {
    const { verdict, results } = await verdictOf({ inferFails: "shuffled" });
    expect(verdict).toMatch(/^FAIL 4\/6 selftest-inputs: shuffled: the runtime gave up/);
    expect(results.filter((r) => !r.ok).map((r) => r.name)).toEqual(["selftest-inputs", "clip-angles"]);
  });

  it("fails the clip angles of a clip that records no tolerance", async () => {
    expect((await verdictOf({ noTolerance: true })).verdict).toMatch(/^FAIL 5\/6 clip-angles: c1 records no angle tolerance/);
  });

  it("fails every check that needs a model file that is not there", async () => {
    const { verdict, results } = await verdictOf({ missingModel: "shuffled" });
    expect(verdict).toMatch(/^FAIL 2\/6 sessions: shuffled models\/b\.onnx: /);
    expect(results.filter((r) => !r.ok).map((r) => r.name)).toEqual(["sessions", "metadata", "selftest-inputs", "clip-angles"]);
  });

  it("fails every check when results.json does not load", async () => {
    const { verdict, results } = await verdictOf({ resultsMissing: true });
    expect(verdict).toMatch(/^FAIL 0\/6 sessions: results\.json did not load/);
    expect(results.every((r) => !r.ok)).toBe(true);
  });

  it("fails on more than one thread", async () => {
    expect((await verdictOf({ threads: 2 })).verdict).toBe("FAIL 6/6 threads=2");
  });

  it("compares every scored frame of every clip for both models", async () => {
    const page = world();
    const seen: Record<string, number> = {};
    const counting: Page<FakeSession> = {
      ...page,
      infer: async (session, stack, count, channels) => {
        seen[session.run] = (seen[session.run] ?? 0) + count;
        return page.infer(session, stack, count, channels);
      },
    };
    expect((await runSelftest(counting, () => undefined)).verdict).toBe("PASS 6/6 threads=1");
    for (const key of MODEL_KEYS) expect(seen[RUN[key]]).toBe(2 + 12);
  });
});
