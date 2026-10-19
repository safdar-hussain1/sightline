// @vitest-environment happy-dom
/**
 * The cockpit as a visitor meets it: loading, playing, switching clips and
 * models, and failing. The model runtime, the data files and the sprite
 * decoder are fakes (`world`), so each test decides what downloads, when, and
 * whether it fails; the video element is driven by hand (`clipVideo`), one
 * presented frame at a time.
 *
 * The fake model's angle on scored frame k is 100 + lead_in + k for Sightline
 * and 200 + lead_in + k for the twin (each frame's bytes hold its sprite
 * index), and the driver's is k - 5, so every angle on screen says which
 * model and which frame it came from.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AssetError } from "../data/load.ts";
import type { Clip, Results } from "../data/types.ts";
import { Cockpit } from "./cockpit.ts";
import { mountCockpitMarkup } from "./markup.ts";
import { ErrorStrip } from "./strip.ts";

const N = 24;
const LEAD_IN = 2;

/** What the fakes do; each test sets the parts it needs. */
const world = vi.hoisted(() => ({
  sessionFor: (url: string): Promise<{ url: string }> => Promise.resolve({ url }),
  sessionUrls: [] as string[],
  runs: [] as string[],
  inferGate: null as Promise<void> | null,
  /** How many runs go through before the gate holds the rest. */
  gateAfter: 0,
  inferFails: (_url: string): boolean => false,
  decode: (_clip: unknown, _base: string, _onPhase?: (phase: "downloading" | "decoding") => void): Promise<Uint8Array[]> =>
    Promise.reject(new Error("no frames")),
  check: (): Promise<{ status: string; frames?: number[] }> => Promise.resolve({ status: "match" }),
  results: (): Promise<unknown> => Promise.reject(new Error("no results")),
  clip: (_url: string): Promise<unknown> => Promise.reject(new Error("no clip")),
}));

vi.mock("../model/runner.ts", () => ({
  INPUT_H: 66,
  INPUT_W: 200,
  threadsInUse: () => 1,
  createSession: (url: string) => {
    world.sessionUrls.push(url);
    return world.sessionFor(url);
  },
  infer: async (session: { url: string }, frames: Uint8Array, n: number) => {
    world.runs.push(session.url);
    if (world.inferGate && world.runs.length > world.gateAfter) await world.inferGate;
    if (world.inferFails(session.url)) throw new Error(`the model at ${session.url} failed`);
    const stride = frames.length / n;
    const bias = session.url.includes("shuffled") ? 200 : 100;
    return {
      angles: Float32Array.from({ length: n }, (_, i) => bias + (frames[i * stride] as number)),
      saliency: new Float32Array(n * 66 * 200).fill(0.5),
    };
  },
}));

vi.mock("../data/load.ts", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../data/load.ts")>();
  return { ...actual, loadResults: () => world.results(), loadClip: (url: string) => world.clip(url) };
});

vi.mock("./sprites.ts", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./sprites.ts")>();
  return {
    ...actual,
    decodeSprites: (clip: unknown, base: string, onPhase?: (phase: "downloading" | "decoding") => void) =>
      world.decode(clip, base, onPhase),
    checkDecoded: () => world.check(),
  };
});

// -- The data ----------------------------------------------------------------------

const RESULTS: Results = {
  assets: {
    models: { sightline: "models/sightline.onnx", shuffled: "models/sightline-shuffled.onnx" },
    clips: { c2017: "data/clips/c2017/clip.json", c2018: "data/clips/c2018/clip.json" },
  },
  shipped: { run: "s", model: "sightlinenet", protocol: "blocked" },
  twin: { run: "t" },
  raw: {},
};

function makeClip(id: "c2018" | "c2017"): Clip {
  const twinFrac = id === "c2018" ? 0.7 : 0;
  return {
    id,
    drive: id === "c2018" ? "drive2018" : "drive2017",
    title: id === "c2018" ? "2018: minutes it never trained on" : "2017: another day",
    fps: 17.5,
    n: N,
    lead_in: LEAD_IN,
    first_frame: 100,
    frame_lags: [2],
    rule: "a test clip",
    labels_deg: Array.from({ length: N }, (_, k) => k - 5),
    expected: {
      sightline: Array.from({ length: N }, (_, k) => 100 + LEAD_IN + k),
      shuffled: Array.from({ length: N }, (_, k) => 200 + LEAD_IN + k),
    },
    sprite: {
      files: ["sprite-00.jpg"],
      frames_per_sheet: 60,
      frame_w: 200,
      frame_h: 66,
      decoded_sha256: Array.from({ length: N + LEAD_IN }, () => "0".repeat(64)),
    },
    video: "clip.mp4",
    video_size: [456, 256],
    camera_size: [455, 256],
    clip_mae_deg: id === "c2018" ? { sightline: 14.26, shuffled: 5.63, zero: 46.7 } : { sightline: 10.78, shuffled: 14.17, zero: 42.11 },
    twin_trained_frac: twinFrac,
    twin_neighbour_frac: twinFrac === 0 ? 0 : 0.97,
  };
}

/** Sprite frame t: every byte is t. */
const frames = () => Array.from({ length: N + LEAD_IN }, (_, t) => new Uint8Array(200 * 66 * 3).fill(t));

// -- The page around it ----------------------------------------------------------------

interface Recorder {
  ops: string[];
}

/** A 2D context that records what was drawn, since the test DOM draws nothing. */
function recordingContext(): CanvasRenderingContext2D & Recorder {
  const ops: string[] = [];
  const record = (name: string) => () => void ops.push(name);
  return {
    ops,
    putImageData: record("put"),
    clearRect: record("clear"),
    drawImage: record("draw"),
    createImageData: (w: number, h: number) => new ImageData(w, h),
    setTransform: () => undefined,
    fillRect: () => undefined,
    beginPath: () => undefined,
    arc: () => undefined,
    fill: () => undefined,
    moveTo: () => undefined,
    lineTo: () => undefined,
    stroke: () => undefined,
  } as unknown as CanvasRenderingContext2D & Recorder;
}

const contexts = new WeakMap<HTMLCanvasElement, CanvasRenderingContext2D & Recorder>();

/** The video element, driven by hand: it loads (or fails) when asked, seeks at
 * once, and presents the frames the test names. Like a real media element, a
 * new load drops the events the last one had not fired yet. */
function clipVideo(video: HTMLVideoElement) {
  let time = 0;
  let next = 1;
  let pending: ReturnType<typeof setTimeout> | undefined;
  const callbacks = new Map<number, (now: number, meta: { mediaTime: number }) => void>();
  const control = {
    fails: false,
    failures: 0,
    plays: 0,
    /** The next play() waits on this, once. */
    holdPlay: null as Promise<void> | null,
  };
  Object.defineProperty(video, "currentTime", {
    configurable: true,
    get: () => time,
    set: (t: number) => {
      time = t;
      queueMicrotask(() => video.dispatchEvent(new Event("seeked")));
    },
  });
  video.load = () => {
    clearTimeout(pending);
    const failing = control.fails;
    pending = setTimeout(() => {
      if (failing) control.failures += 1;
      video.dispatchEvent(new Event(failing ? "error" : "loadeddata"));
    }, 0);
  };
  video.play = () => {
    control.plays += 1;
    const held = control.holdPlay;
    control.holdPlay = null;
    return held ?? Promise.resolve();
  };
  video.pause = () => undefined;
  Object.assign(video, {
    requestVideoFrameCallback: (cb: (now: number, meta: { mediaTime: number }) => void) => {
      callbacks.set(next, cb);
      return next++;
    },
    cancelVideoFrameCallback: (handle: number) => void callbacks.delete(handle),
  });
  return {
    control,
    /** The browser presents clip frame k. */
    present(k: number) {
      const pending = [...callbacks.values()];
      callbacks.clear();
      for (const cb of pending) cb(0, { mediaTime: (k + 0.5) / 17.5 });
    },
  };
}

let idle: Array<() => void> = [];
let warnings: unknown[][] = [];
let unhandled: unknown[] = [];
const onUnhandled = (reason: unknown) => void unhandled.push(reason);

declare const process: {
  on(event: "unhandledRejection", listener: (reason: unknown) => void): void;
  off(event: "unhandledRejection", listener: (reason: unknown) => void): void;
};

beforeEach(() => {
  world.sessionFor = (url) => Promise.resolve({ url });
  world.sessionUrls = [];
  world.runs = [];
  world.inferGate = null;
  world.gateAfter = 0;
  world.inferFails = () => false;
  world.decode = async () => frames();
  world.check = async () => ({ status: "match" });
  world.results = async () => RESULTS;
  world.clip = async (url) => makeClip(url.includes("c2017") ? "c2017" : "c2018");
  idle = [];
  warnings = [];
  unhandled = [];
  process.on("unhandledRejection", onUnhandled);
  vi.spyOn(console, "warn").mockImplementation((...args: unknown[]) => void warnings.push(args));
  vi.stubGlobal("requestIdleCallback", (cb: () => void) => void idle.push(cb));
  vi.stubGlobal(
    "IntersectionObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function (this: HTMLCanvasElement) {
    let context = contexts.get(this);
    if (!context) {
      context = recordingContext();
      contexts.set(this, context);
    }
    return context as never;
  });
  document.body.innerHTML = `<section class="cockpit" id="cockpit" aria-label="Live demo"></section>`;
});

afterEach(async () => {
  // Let anything still settling report itself before the next test starts.
  await ticks(5);
  process.off("unhandledRejection", onUnhandled);
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  expect(unhandled, "unhandled promise rejections").toEqual([]);
});

// -- Helpers ---------------------------------------------------------------------------

const ticks = async (count = 1) => {
  for (let i = 0; i < count; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
};

/** Wait until `ready()` holds, giving the page's promises and timers their turns. */
async function until(ready: () => boolean, what: string): Promise<void> {
  for (let i = 0; i < 400; i += 1) {
    if (ready()) return;
    await ticks();
  }
  throw new Error(`gave up waiting until ${what}`);
}

const el = (name: string) => document.querySelector<HTMLElement>(`[data-cockpit="${name}"]`) as HTMLElement;
const say = (name: string) => el(name).textContent ?? "";
const consoleState = () => document.querySelector<HTMLElement>(".cockpit__console")?.dataset["state"];
const shown = (name: string) => (el(name).hidden ? "" : say(name));
const checked = (group: string) => document.querySelector<HTMLInputElement>(`input[name="${group}"]:checked`)?.value;
const radio = (group: string, value: string) =>
  document.querySelector<HTMLInputElement>(`input[name="${group}"][value="${value}"]`) as HTMLInputElement;
const context = (name: string) => contexts.get(el(name) as HTMLCanvasElement) as CanvasRenderingContext2D & Recorder;

/** The visitor picks an option of a radio group. */
function pick(group: string, value: string): void {
  const input = radio(group, value);
  input.checked = true;
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

function key(name: string): void {
  el("strip").dispatchEvent(new KeyboardEvent("keydown", { key: name, bubbles: true, cancelable: true }));
}

/** Mount the cockpit and start it. */
function mount() {
  const root = mountCockpitMarkup() as HTMLElement;
  const video = clipVideo(root.querySelector("video") as HTMLVideoElement);
  const cockpit = new Cockpit(root);
  const started = cockpit.start();
  return { cockpit, video, started };
}

/** Mount the cockpit and wait until the opening clip plays. */
async function playing() {
  const page = mount();
  await until(() => consoleState() === "ready" && el("play").dataset["playing"] === "true", "the clip plays");
  return page;
}

const sightlineAt = (k: number) => `${100 + LEAD_IN + k}.0°`;
const twinAt = (k: number) => `${200 + LEAD_IN + k}.0°`;
const TWIN = "models/sightline-shuffled.onnx";

// -- The twin ----------------------------------------------------------------------------

describe("the twin, when it cannot be had", () => {
  it("keeps the clip playing on Sightline when the twin's download fails mid-play, and says so once", async () => {
    let failTwin: (error: unknown) => void = () => undefined;
    world.sessionFor = (url) =>
      url.includes("shuffled") ? new Promise((_, reject) => (failTwin = reject)) : Promise.resolve({ url });
    const { video } = await playing();
    video.present(3);
    expect(say("model-angle")).toBe(sightlineAt(3));

    pick("cockpit-model", "shuffled");
    // While the twin downloads the clip plays on, on Sightline: the frames on
    // screen are never handed to a model that has not arrived.
    video.present(4);
    video.present(5);
    await ticks(3);
    expect(say("model-name")).toBe("Sightline");
    expect(say("model-angle")).toBe(sightlineAt(5));
    expect(world.runs.filter((url) => url.includes("shuffled"))).toEqual([]);

    failTwin(new AssetError(`could not reach ${TWIN}: Failed to fetch`, "unreachable", TWIN));
    await until(() => shown("notice") !== "", "the cockpit says the twin is missing");
    video.present(6);
    video.present(7);
    await ticks(3);

    expect(consoleState()).toBe("ready");
    expect(shown("status")).toBe("");
    expect(shown("notice")).toBe("The twin could not be loaded, so Sightline is shown instead; pick the twin again to retry.");
    expect(checked("cockpit-model")).toBe("sightline");
    expect(say("model-name")).toBe("Sightline");
    expect(say("model-angle")).toBe(sightlineAt(7));
    expect((el("play") as HTMLButtonElement).disabled).toBe(false);
    expect(el("play").dataset["playing"]).toBe("true");
  });

  it("falls back to Sightline, still live, when the twin arrives but cannot run", async () => {
    world.inferFails = (url) => url.includes("shuffled");
    const { video } = await playing();
    pick("cockpit-model", "shuffled");
    await until(() => checked("cockpit-model") === "sightline" && shown("notice") !== "", "the cockpit falls back");
    video.present(9);
    await until(() => say("model-angle") === sightlineAt(9), "Sightline drives frame 9");
    expect(consoleState()).toBe("ready");
    expect(shown("status")).toBe("");
    expect(shown("notice")).toMatch(/^The twin could not be loaded/);
    expect(el("play").dataset["playing"]).toBe("true");
  });

  it("downloads the twin quietly once the clip runs, and a failure there says nothing until the visitor asks for it", async () => {
    let attempts = 0;
    world.sessionFor = (url) => {
      if (!url.includes("shuffled")) return Promise.resolve({ url });
      attempts += 1;
      return attempts === 1 ? Promise.reject(new AssetError("gone", "unreachable", TWIN)) : Promise.resolve({ url });
    };
    const { video } = await playing();
    expect(idle).toHaveLength(1);
    for (const run of idle.splice(0)) run();
    await ticks(3);
    expect(shown("notice")).toBe("");
    expect(consoleState()).toBe("ready");

    pick("cockpit-model", "shuffled"); // tried again, and this time it arrives
    await until(() => say("model-name") === "Its twin", "the twin is on screen");
    video.present(4);
    await until(() => say("model-angle") === twinAt(4), "the twin drives frame 4");
    expect(attempts).toBe(2);
    expect(shown("notice")).toBe("");
  });

  it("stays on Sightline when the visitor changes their mind before the twin arrives", async () => {
    let arrive: (session: { url: string }) => void = () => undefined;
    world.sessionFor = (url) => (url.includes("shuffled") ? new Promise((resolve) => (arrive = resolve)) : Promise.resolve({ url }));
    const { video } = await playing();
    pick("cockpit-model", "shuffled");
    pick("cockpit-model", "sightline");
    arrive({ url: TWIN });
    await ticks(5);
    video.present(2);
    await ticks(2);
    expect(say("model-name")).toBe("Sightline");
    expect(say("model-angle")).toBe(sightlineAt(2));
    expect(checked("cockpit-model")).toBe("sightline");
  });

  it("loads the next clip on Sightline when the twin on screen cannot run on it", async () => {
    let twinBroken = false;
    world.inferFails = (url) => twinBroken && url.includes("shuffled");
    const { video } = await playing();
    pick("cockpit-model", "shuffled");
    await until(() => say("model-name") === "Its twin", "the twin is on screen");
    twinBroken = true;
    pick("cockpit-clip", "c2017");
    await until(() => consoleState() === "ready", "the next clip is ready");
    video.present(2);
    await until(() => say("model-angle") === sightlineAt(2), "Sightline drives the next clip");
    expect(say("model-name")).toBe("Sightline");
    expect(checked("cockpit-model")).toBe("sightline");
    expect(shown("notice")).toMatch(/^The twin could not be loaded/);
    expect(shown("status")).toBe("");
  });

  it("switches to the twin with the clip's own account of what it trained on", async () => {
    const { video } = await playing();
    pick("cockpit-model", "shuffled");
    await until(() => say("model-name") === "Its twin", "the twin is on screen");
    video.present(1);
    await until(() => say("model-angle") === twinAt(1), "the twin drives frame 1");
    expect(say("note")).toMatch(/70% of these very frames/);
    expect(say("average")).toBe("5.6°");
  });
});

// -- The strip -----------------------------------------------------------------------------

describe("the strip", () => {
  it("does nothing while the clip is still loading", async () => {
    // The frames are decoded and the strip has the clip; the model is still starting.
    let start: () => void = () => undefined;
    world.sessionFor = (url) => new Promise((resolve) => (start = () => resolve({ url })));
    mount();
    await until(() => shown("status") === "Starting the model", "the model is starting");
    el("strip").focus();
    key("ArrowRight");
    key("End");
    el("strip").dispatchEvent(new PointerEvent("pointerdown", { clientX: 40, pointerId: 1, bubbles: true }));
    await ticks(3);
    expect(say("driver-angle")).toBe("—");
    expect(world.runs).toEqual([]);
    expect(el("strip").getAttribute("aria-disabled")).toBe("true");
    start();
    await until(() => consoleState() === "ready", "the clip is ready");
    expect(el("strip").getAttribute("aria-valuenow")).toBe("0");
    expect(say("driver-angle")).toBe("5.0°");
  });

  it("is inert once the demo cannot run: no new download, and no driver's angle on the failure screen", async () => {
    world.sessionFor = (url) => Promise.reject(new AssetError(`could not reach ${url}`, "unreachable", url));
    mount();
    await until(() => consoleState() === "unavailable", "the demo gives up");
    const requested = world.sessionUrls.length;
    el("strip").focus();
    key("ArrowRight");
    key("End");
    el("strip").dispatchEvent(new PointerEvent("pointerdown", { clientX: 40, pointerId: 1, bubbles: true }));
    await ticks(5);
    expect(world.sessionUrls).toHaveLength(requested);
    expect(say("driver-angle")).toBe("—");
    expect(say("model-angle")).toBe("—");
    expect(el("strip").getAttribute("aria-disabled")).toBe("true");
    expect(el("strip").tabIndex).toBe(-1);
    expect(shown("status")).toMatch(/could not download/);
  });

  it("moves the clip a frame at a time with the arrow keys once it runs", async () => {
    const { video } = await playing();
    video.present(0);
    (el("play") as HTMLButtonElement).click(); // pause
    key("ArrowRight");
    key("ArrowRight");
    await until(() => say("model-angle") === sightlineAt(2), "frame 2 is on screen");
    expect(el("strip").getAttribute("aria-valuenow")).toBe("2");
  });
});

// -- What is on screen while something is not ready ------------------------------------------

describe("stale values", () => {
  it("clears the last clip's numbers, heat and notes while the next clip loads", async () => {
    const { video } = await playing();
    pick("cockpit-model", "shuffled");
    await until(() => say("model-name") === "Its twin", "the twin is on screen");
    video.present(5);
    await until(() => say("model-angle") === twinAt(5), "the twin drives frame 5");
    expect(say("note")).toMatch(/70%/);

    let release: (value: Uint8Array[]) => void = () => undefined;
    world.decode = () => new Promise((resolve) => (release = resolve));
    const heat = context("heat");
    const eye = context("eye");
    heat.ops.length = 0;
    eye.ops.length = 0;
    pick("cockpit-clip", "c2017");
    await until(() => consoleState() === "loading", "the next clip loads");
    await ticks(3);

    for (const name of ["model-angle", "driver-angle", "gap", "average"]) expect(say(name), name).toBe("—");
    expect(say("model-side") + say("driver-side")).toBe("");
    expect(say("note")).toBe("The same network, trained on shuffled frames.");
    expect(el("strip").getAttribute("aria-valuetext")).toBe("No clip loaded yet");
    expect(heat.ops.at(-1)).toBe("clear");
    expect(eye.ops.at(-1)).toBe("clear");
    expect(shown("notice")).toBe("");

    release(frames());
    await until(() => consoleState() === "ready", "the next clip is ready");
    expect(say("note")).toBe("The same network, trained on shuffled frames. It never trained on any frame of this clip.");
    expect(say("average")).toBe("14.2°");
  });

  it("shows nothing of the model's on a frame whose output is still being computed", async () => {
    // Only the opening run (frames 0 to 3) goes through; the rest wait.
    let release: () => void = () => undefined;
    world.inferGate = new Promise<void>((resolve) => (release = resolve));
    world.gateAfter = 1;
    const { video } = await playing();
    video.present(1);
    await until(() => say("model-angle") === sightlineAt(1), "frame 1 is on screen");
    const heat = context("heat");
    const eye = context("eye");
    (el("play") as HTMLButtonElement).click(); // pause
    key("End");
    await ticks(2);
    expect(say("driver-angle")).toBe(`${N - 1 - 5}.0°`);
    expect(say("model-angle")).toBe("—");
    expect(say("gap")).toBe("—");
    expect(heat.ops.at(-1)).toBe("clear");
    expect(eye.ops.at(-1)).toBe("put"); // the frame itself, without heat
    expect(el("wheel").dataset["waiting"]).toBe("true");

    release();
    world.inferGate = null;
    await until(() => say("model-angle") === sightlineAt(N - 1), "the last frame's output arrives");
    expect(heat.ops.at(-1)).toBe("put");
    expect(eye.ops.at(-1)).toBe("draw");
    expect(el("wheel").dataset["waiting"]).toBe("false");
  });
});

// -- Failing, and recovering ------------------------------------------------------------------

describe("when a clip cannot run", () => {
  it("lets the visitor try the other clip", async () => {
    world.decode = (clip) =>
      (clip as Clip).id === "c2018" ? Promise.reject(new AssetError("bad sheet", "undecodable", "x")) : Promise.resolve(frames());
    mount();
    await until(() => consoleState() === "unavailable", "the first clip fails");
    expect(shown("status")).toMatch(/could not decode/);
    expect((el("clips") as HTMLFieldSetElement).disabled).toBe(false);
    expect((el("models") as HTMLFieldSetElement).disabled).toBe(true);
    pick("cockpit-clip", "c2017");
    await until(() => consoleState() === "ready", "the other clip runs");
    expect(shown("status")).toBe("");
  });

  it("tries the same clip again when the visitor picks it again", async () => {
    let attempts = 0;
    world.decode = async () => {
      attempts += 1;
      if (attempts === 1) throw new AssetError("could not reach sprite-00.jpg", "unreachable", "x");
      return frames();
    };
    mount();
    await until(() => consoleState() === "unavailable", "the clip fails");
    radio("cockpit-clip", "c2018").click(); // already checked: a click, and no change
    await until(() => consoleState() === "ready", "the clip runs on the second try");
    expect(attempts).toBe(2);
    expect(shown("status")).toBe("");
  });

  it("does not reload a clip that is running when its option is clicked again", async () => {
    await playing();
    const decodes = world.decode;
    let again = 0;
    world.decode = (clip, base, onPhase) => {
      again += 1;
      return decodes(clip, base, onPhase);
    };
    radio("cockpit-clip", "c2018").click();
    await ticks(5);
    expect(again).toBe(0);
    expect(consoleState()).toBe("ready");
  });

  it("reports a failed clip once, with no unhandled rejection from its video", async () => {
    world.decode = () => Promise.reject(new AssetError("bad sheet", "undecodable", "x"));
    const root = mountCockpitMarkup() as HTMLElement;
    const video = clipVideo(root.querySelector("video") as HTMLVideoElement);
    video.control.fails = true; // the video fails too, before anyone waits for it
    await new Cockpit(root).start();
    await until(() => consoleState() === "unavailable", "the clip fails");
    await ticks(5);
    expect(shown("status")).toMatch(/could not decode/);
  });

  it("leaves no unhandled rejection behind when the visitor switches clips mid-load", async () => {
    let release: (value: Uint8Array[]) => void = () => undefined;
    world.decode = (clip) =>
      (clip as Clip).id === "c2018" ? new Promise((resolve) => (release = resolve)) : Promise.resolve(frames());
    const root = mountCockpitMarkup() as HTMLElement;
    const video = clipVideo(root.querySelector("video") as HTMLVideoElement);
    video.control.fails = true; // the first clip's video fails while its frames are still decoding
    void new Cockpit(root).start();
    await until(() => video.control.failures === 1, "the first clip's video fails");
    video.control.fails = false;
    pick("cockpit-clip", "c2017");
    release(frames());
    await until(() => consoleState() === "ready", "the next clip runs");
    expect(checked("cockpit-clip")).toBe("c2017");
  });
});

describe("playing", () => {
  it("keeps the next clip playing, and saying so, when the last clip's play is cut short", async () => {
    const root = mountCockpitMarkup() as HTMLElement;
    const video = clipVideo(root.querySelector("video") as HTMLVideoElement);
    let cutShort: (error: unknown) => void = () => undefined;
    video.control.holdPlay = new Promise<void>((_, reject) => (cutShort = reject));
    void new Cockpit(root).start();
    await until(() => consoleState() === "ready" && video.control.plays === 1, "the first clip asks to play");
    pick("cockpit-clip", "c2017");
    await until(() => consoleState() === "ready" && el("play").dataset["playing"] === "true", "the next clip plays");
    cutShort(new DOMException("The play() request was interrupted by a new load request.", "AbortError"));
    await ticks(3);
    expect(el("play").dataset["playing"]).toBe("true");
    expect(say("play-label")).toBe("Pause");
  });
});

describe("telling the visitor what it is doing", () => {
  it("says it is downloading the frames, then decoding them", async () => {
    const said: string[] = [];
    world.decode = async (_clip, _base, onPhase) => {
      onPhase?.("downloading");
      said.push(shown("status"));
      await ticks(1);
      onPhase?.("decoding");
      said.push(shown("status"));
      return frames();
    };
    await playing();
    expect(said).toEqual(["Downloading the frames the model reads", "Decoding the frames the model reads"]);
  });

  it("never claims the footage matches when this browser cannot check it", async () => {
    world.check = async () => ({ status: "unverified" });
    const values = vi.spyOn(ErrorStrip.prototype, "setValues");
    const { cockpit } = await playing();
    expect(shown("notice")).toMatch(/could not check/);
    expect(cockpit.frameCheck).toBe("unverified");
    // The strip is not drawn from the published outputs; only from what this
    // browser computes.
    const first = values.mock.calls[0]?.[0] as ArrayLike<number>;
    expect(Number.isNaN(first[N - 1])).toBe(true);
  });

  it("says how many frames this browser decodes differently, and still runs", async () => {
    world.check = async () => ({ status: "differ", frames: [3, 4] });
    const { cockpit } = await playing();
    expect(shown("notice")).toMatch(/decodes 2 of the clip’s 26 frames slightly differently/);
    expect(cockpit.frameCheck).toBe("differ");
  });

  it("draws the strip from the published outputs at once when the frames match", async () => {
    const values = vi.spyOn(ErrorStrip.prototype, "setValues");
    const { cockpit } = await playing();
    expect(shown("notice")).toBe("");
    expect(cockpit.frameCheck).toBe("match");
    const first = values.mock.calls[0]?.[0] as ArrayLike<number>;
    expect(first[N - 1]).toBe(100 + LEAD_IN + N - 1);
  });
});
