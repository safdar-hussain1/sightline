// @vitest-environment happy-dom
/**
 * "Try it on your own video" as a visitor meets it: a file chosen, read and
 * played, or refused with a message of its kind; a reading cancelled, or
 * replaced by another file; and the cockpit and the network left alone
 * throughout. The video reader and the model session are fakes the test
 * drives (the browser's own reading is checked in a real browser); one test
 * runs the real runner on a mocked runtime.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { parseResults } from "../data/load.ts";
import type { Clip, Results } from "../data/types.ts";
import { steerParts } from "../format.ts";
import type { LiveCockpit, LiveStatus } from "../cockpit/live.ts";
import { OwnClip, cockpitClip, loadContext, shippedSession, type ClipFacts, type Context, type ObjectUrls, type OpenSession, type Session } from "./ownclip.ts";
import { answeredNote, doneNote, failureWords } from "./ownclip-copy.ts";
import { ownclipHtml } from "./ownclip-markup.ts";
import { ClipError, MAX_FRAMES, readingPace, type Evidence } from "./ownclip-plan.ts";
import type { OpenVideo, VideoSource } from "./ownclip-reader.ts";
import resultsText from "../../public/data/results.json?raw";
import clip2017Text from "../../public/data/clips/c2017/clip.json?raw";
import clip2018Text from "../../public/data/clips/c2018/clip.json?raw";
import controllerSource from "./ownclip.ts?raw";
import readerSource from "./ownclip-reader.ts?raw";
import planSource from "./ownclip-plan.ts?raw";

/** The runtime, mocked: one session per create, each run answering with the
 * first byte of each stack, unless the session was made to fail. */
const runtime = vi.hoisted(() => ({ created: [] as Array<{ failing: boolean; released: boolean }>, failNext: false }));

vi.mock("onnxruntime-web/wasm", () => {
  class Tensor {
    constructor(
      readonly type: string,
      readonly data: Uint8Array,
      readonly dims: number[],
    ) {}
  }
  return {
    env: { wasm: { numThreads: 0, wasmPaths: undefined } },
    Tensor,
    InferenceSession: {
      create: async () => {
        const record = { failing: runtime.failNext, released: false };
        runtime.failNext = false;
        runtime.created.push(record);
        return {
          run: async (feeds: Record<string, Tensor>) => {
            if (record.failing) throw new Error("this session cannot run");
            const input = feeds["frames"] as Tensor;
            const n = input.dims[0] as number;
            const stride = input.data.length / n;
            return {
              steering_deg: { data: Float32Array.from({ length: n }, (_, i) => input.data[i * stride] as number) },
              saliency: { data: new Float32Array(n * 66 * 200) },
            };
          },
          release: async () => {
            record.released = true;
          },
        };
      },
    },
  };
});

const W = 200;
const H = 66;

const context = (): Context => ({
  modelUrl: "https://example.test/models/sightline.onnx",
  reading: {
    pace: readingPace([2], new Map([[2, 0.1]]), 17.5),
    lags: [2],
    camera: [455, 256],
    inputW: W,
    inputH: H,
    drive: "drive2018",
    trainedOnDrive: true,
  },
});

/** A frame whose every byte is `j`, so a stack says which frames are in it. */
const frameOf = (j: number) => new Uint8Array(W * H * 3).fill(j % 256);

/** The first bytes of an MP4 file, and of a PNG picture. */
const MP4_HEAD = Uint8Array.from([0, 0, 0, 24, ...Array.from("ftypisom", (c) => c.charCodeAt(0)), 0, 0, 2, 0]);
const PNG_HEAD = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13]);

const videoFile = (name = "drive.mp4", head: Uint8Array<ArrayBuffer> = MP4_HEAD) => new File([head, new Uint8Array(4096)], name, { type: "video/mp4" });

interface Script {
  readonly width?: number;
  readonly height?: number;
  readonly duration?: number;
  /** The reader fails at this frame, with this error. */
  readonly failAt?: { readonly frame: number; readonly evidence: Evidence };
  /** The reader waits for the test before this frame. */
  readonly holdAt?: number;
  /** Opening fails with this evidence (added to what the page passed in). */
  readonly refuse?: Evidence;
  /** The file gives no new frames from this frame on: the reader passes on
   * the frames before it (and the one at it, before it knew), then says so. */
  readonly endAt?: number;
}

/** A fake video reader that reads frames on a timer, and what it saw. */
function fakeVideo(script: Script = {}) {
  const seen = { opened: [] as Array<{ url: string; evidence: Evidence }>, signals: [] as AbortSignal[], closed: 0, planned: 0, release: () => undefined as void };
  let release: () => void = () => undefined;
  const held = new Promise<void>((resolve) => (release = resolve));
  seen.release = () => release();
  const open: OpenVideo = async (url, evidence, signal) => {
    seen.opened.push({ url, evidence });
    seen.signals.push(signal);
    if (script.refuse) throw new ClipError("refused", { ...evidence, ...script.refuse });
    const source: VideoSource = {
      width: script.width ?? 1280,
      height: script.height ?? 720,
      duration: script.duration ?? 1,
      read: async (times, _crop, _size, onFrame, readSignal) => {
        seen.planned = times.length;
        const last = script.endAt === undefined ? times.length : Math.min(times.length, script.endAt + 1);
        for (let j = 0; j < last; j += 1) {
          if (j === script.holdAt) await held;
          await new Promise((resolve) => setTimeout(resolve, 0));
          if (readSignal.aborted) throw new DOMException("stopped", "AbortError");
          if (script.failAt && j === script.failAt.frame) throw new ClipError("a seek never finished", { ...evidence, ...script.failAt.evidence });
          onFrame(j, frameOf(j), document.createElement("video"));
        }
        return { end: script.endAt === undefined || script.endAt >= times.length ? null : script.endAt };
      },
      close: () => {
        seen.closed += 1;
      },
    };
    return source;
  };
  return { open, seen };
}

/** A fake model session: each answer is the first byte of its stack (the
 * frame's own number), and every run and release is recorded. */
function fakeSessions(options: { failOpen?: boolean; failRun?: boolean } = {}) {
  const log = { opened: 0, runs: [] as Array<{ n: number; channels: number; firsts: number[][] }>, released: 0 };
  const open: OpenSession = async () => {
    log.opened += 1;
    if (options.failOpen) throw new Error("the model could not be downloaded");
    const session: Session = {
      run: async (frames, n, channels) => {
        if (options.failRun) throw new Error("the model failed");
        const stride = frames.length / n;
        const firsts = Array.from({ length: n }, (_, i) => Array.from(frames.subarray(i * stride, i * stride + channels)));
        log.runs.push({ n, channels, firsts });
        return {
          angles: Float32Array.from({ length: n }, (_, i) => frames[i * stride] as number),
          saliency: new Float32Array(n * W * H),
        };
      },
      release: async () => {
        log.released += 1;
      },
    };
    return session;
  };
  return { open, log };
}

function fakeUrls() {
  const made: string[] = [];
  const revoked: string[] = [];
  const urls: ObjectUrls = {
    create: () => {
      const url = `blob:https://example.test/${made.length + 1}`;
      made.push(url);
      return url;
    },
    revoke: (url) => revoked.push(url),
  };
  return { urls, made, revoked };
}

const q = <T extends Element = HTMLElement>(name: string) => document.querySelector<T>(`[data-ownclip="${name}"]`) as T;
const text = (name: string) => q(name).textContent ?? "";
const shown = (name: string) => !q(name).hidden;

/** happy-dom does not play video: seeks land at once, play and pause do
 * nothing, so the player can be driven by the test. */
function quietVideo(): void {
  const video = q<HTMLVideoElement>("video");
  let time = 0;
  Object.defineProperty(video, "currentTime", {
    get: () => time,
    set: (t: number) => {
      time = t;
      queueMicrotask(() => video.dispatchEvent(new Event("seeked")));
    },
  });
  video.play = vi.fn(async () => undefined);
  video.pause = vi.fn();
  video.load = vi.fn();
}

function mount(options: { video?: ReturnType<typeof fakeVideo>; sessions?: ReturnType<typeof fakeSessions>; openSession?: OpenSession; reduced?: boolean } = {}) {
  const section = document.createElement("section");
  section.id = "own-video";
  section.innerHTML = ownclipHtml();
  document.body.append(section);
  quietVideo();
  const video = options.video ?? fakeVideo();
  const sessions = options.sessions ?? fakeSessions();
  const urls = fakeUrls();
  const own = new OwnClip(section, {
    context: Promise.resolve(context()),
    openVideo: video.open,
    openSession: options.openSession ?? sessions.open,
    urls: urls.urls,
    reduced: () => options.reduced ?? true,
  });
  return { section, own, video, sessions, urls };
}

const settle = async (ticks = 5) => {
  for (let i = 0; i < ticks; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
};

beforeEach(() => {
  document.body.innerHTML = "";
  runtime.created.length = 0;
  runtime.failNext = false;
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/** The files the site publishes that the context is read from. */
const PUBLISHED: Readonly<Record<string, string>> = {
  "data/results.json": resultsText,
  "data/clips/c2017/clip.json": clip2017Text,
  "data/clips/c2018/clip.json": clip2018Text,
};
const published = (path: string) => {
  const text = PUBLISHED[path];
  if (text === undefined) throw new Error(`the test serves no ${path}`);
  return text;
};

// -- What the page's data says ------------------------------------------------------------

describe("the reading's context, from the data the site publishes", () => {
  it("takes the history, the input and the camera from the clips, and the pace from the measured gap of the drive the model learned from", async () => {
    const raw = JSON.parse(published("data/results.json")) as Record<string, unknown>;
    const results = parseResults(raw, "data/results.json");
    const clip = JSON.parse(published("data/clips/c2018/clip.json")) as ClipFacts;
    const context = await loadContext(Promise.resolve(results), Promise.resolve(clip));
    const timing = (raw["timing"] as Record<string, { lag_gap_s: Array<{ lag: number; median: number }> } | null>)["drive2018"];
    const gap = timing?.lag_gap_s.find((entry) => entry.lag === 2)?.median;
    expect(context.reading.lags).toEqual([2]);
    expect(context.reading.camera).toEqual([455, 256]);
    expect([context.reading.inputW, context.reading.inputH]).toEqual([200, 66]);
    expect(context.reading.drive).toBe("drive2018");
    expect(context.reading.trainedOnDrive).toBe(true);
    expect(context.reading.pace.gapS).toBe(gap);
    expect(context.reading.pace.stepS).toBeCloseTo((gap as number) / 2, 12);
    expect(context.modelUrl).toMatch(/\/models\/sightline\.onnx$/);
  });
});

describe("the clip's facts, from the cockpit", () => {
  const results = () => parseResults(JSON.parse(published("data/results.json")), "data/results.json");
  const clip = () => JSON.parse(published("data/clips/c2018/clip.json")) as Clip;
  const live = (statuses: Array<Partial<LiveStatus>>): LiveCockpit => {
    let call = 0;
    return {
      status: () => ({ state: "loading", playing: false, frame: 0, clip: null, model: "sightline", ...(statuses[Math.min(call++, statuses.length - 1)] ?? {}) }),
      frameAt: () => null,
      agreement: () => null,
      msPerFrame: () => null,
    };
  };
  const fetched: string[] = [];
  beforeEach(() => {
    fetched.length = 0;
    vi.stubGlobal("fetch", async (url: string) => {
      fetched.push(String(url));
      const path = String(url).replace(/^https?:\/\/[^/]+\//, "");
      return new Response(published(path), { headers: { "content-type": "application/json" } });
    });
  });

  it("reads them from the cockpit's clip once it has one, and fetches nothing", async () => {
    const cockpit = live([{}, {}, { state: "ready", clip: clip() }]);
    const facts = await cockpitClip(Promise.resolve(cockpit), Promise.resolve(results() as Results), 1).facts;
    expect(facts.frame_lags).toEqual([2]);
    expect(facts.camera_size).toEqual([455, 256]);
    expect(fetched).toEqual([]);
  });

  it("fetches the first clip file itself when the cockpit cannot start, or fails", async () => {
    await cockpitClip(Promise.resolve(null), Promise.resolve(results() as Results), 1).facts;
    expect(fetched).toHaveLength(1);
    await cockpitClip(Promise.resolve(live([{ state: "unavailable" }])), Promise.resolve(results() as Results), 1).facts;
    expect(fetched).toHaveLength(2);
    expect(fetched[0]).toMatch(/data\/clips\/c2017\/clip\.json$/);
  });

  it("does not wait on the cockpit's decoded frames once a video is chosen: it fetches the clip file at once", async () => {
    // The cockpit is still decoding, for as long as it likes.
    const source = cockpitClip(Promise.resolve(live([{}])), Promise.resolve(results() as Results), 1, 60_000);
    const started = Date.now();
    const facts = await source.now();
    expect(Date.now() - started).toBeLessThan(1000);
    expect(facts.camera_size).toEqual([455, 256]);
    expect(fetched).toHaveLength(1);
    // Asked again, it fetches nothing more.
    await source.now();
    expect(fetched).toHaveLength(1);
  });

  it("fetches nothing when a video is chosen after the cockpit has its clip", async () => {
    const source = cockpitClip(Promise.resolve(live([{ state: "ready", clip: clip() }])), Promise.resolve(results() as Results), 1);
    await source.facts;
    await source.now();
    expect(fetched).toEqual([]);
  });
});

describe("a page whose measurements cannot be had", () => {
  it("leaves no failure unheard when results.json does not load", async () => {
    const heard: unknown[] = [];
    const listen = (reason: unknown) => heard.push(reason);
    const node = (globalThis as { process?: { on(e: string, f: (r: unknown) => void): void; off(e: string, f: (r: unknown) => void): void } }).process;
    node?.on("unhandledRejection", listen);
    try {
      const section = document.createElement("section");
      section.id = "own-video";
      section.innerHTML = ownclipHtml();
      document.body.append(section);
      const failed = Promise.reject(new Error("results.json is not on this server"));
      OwnClip.mount(failed, Promise.resolve(null));
      for (let i = 0; i < 10; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
    } finally {
      node?.off("unhandledRejection", listen);
    }
    expect(heard).toEqual([]);
  });
});

// -- From file to player ----------------------------------------------------------------

describe("a video read from start to finish", () => {
  it("goes loading, sampling, running, done, with its progress on screen", async () => {
    const { own, sessions } = mount();
    const phases: string[] = [];
    const watch = new MutationObserver(() => phases.push(q("console").dataset["phase"] ?? ""));
    watch.observe(q("console"), { attributes: true, attributeFilter: ["data-phase"] });
    const reading = own.choose(videoFile());
    expect(own.status()).toEqual({ phase: "loading", name: "drive.mp4" });
    expect(text("status")).toBe("Opening your video");
    await reading;
    await settle(1);
    watch.disconnect();
    expect(phases[0]).toBe("loading");
    expect(phases).toContain("sampling");
    expect(phases.at(-1)).toBe("done");
    // A second of video at the measured pace: 20 frames, 18 with history.
    expect(own.status()).toEqual({ phase: "done", name: "drive.mp4", total: 20, answerable: 18, capped: false, early: false });
    expect(sessions.log.runs.reduce((sum, run) => sum + run.n, 0)).toBe(18);
  });

  it("stacks each frame with the one two frames before it, as the model was trained, and starts at the third frame", async () => {
    const { own, sessions } = mount();
    await own.choose(videoFile());
    const first = sessions.log.runs[0];
    expect(first?.channels).toBe(6);
    // Frame 2 first, then frame 0 behind it: the channel order of training.
    expect(first?.firsts[0]).toEqual([2, 2, 2, 0, 0, 0]);
    const stacks = sessions.log.runs.flatMap((run) => run.firsts.map((pixel) => pixel[0]));
    expect(stacks).toEqual(Array.from({ length: 18 }, (_, i) => i + 2));
  });

  it("shows the player: the wheel's estimate, where it is, a scrubber over the frames with an answer", async () => {
    const { own } = mount();
    await own.choose(videoFile());
    await settle();
    expect(shown("player")).toBe(true);
    expect(shown("drop")).toBe(false);
    expect(shown("error")).toBe(false);
    for (const name of ["play", "back", "forward", "scrub"]) expect(q<HTMLButtonElement>(name).disabled, name).toBe(false);
    const scrub = q<HTMLInputElement>("scrub");
    expect([scrub.min, scrub.max, scrub.value]).toEqual(["2", "19", "2"]);
    expect(text("angle")).toBe(steerParts(2).size);
    expect(text("side")).toBe("right");
    expect(text("when")).toBe("Frame 3 of 20, 0.1\u00A0s in");
    expect(scrub.getAttribute("aria-valuetext")).toBe(`Frame 3 of 20, 0.1\u00A0s in: the model estimates ${steerParts(2).size} right`);
    expect(text("note")).toBe("drive.mp4: 20 frames read, 18 answers.");
    expect(q("wheel").dataset["waiting"]).toBe("false");
    // The reading's second reader and its session are let go of; the video on
    // screen keeps the file to play it.
    expect(text("status")).toBe("");
  });

  it("moves to the frame asked for, and never before the first answer", async () => {
    const { own } = mount();
    await own.choose(videoFile());
    const scrub = q<HTMLInputElement>("scrub");
    scrub.value = "9";
    scrub.dispatchEvent(new Event("input"));
    expect(text("angle")).toBe(steerParts(9).size);
    q("back").click();
    expect(scrub.value).toBe("8");
    for (let i = 0; i < 12; i += 1) q("back").click();
    expect(scrub.value).toBe("2");
    expect(text("when")).toBe("Frame 3 of 20, 0.1\u00A0s in");
  });

  it("releases the session once read, and plays on its own unless motion is reduced", async () => {
    const still = mount({ reduced: true });
    await still.own.choose(videoFile());
    await settle();
    expect(still.sessions.log.released).toBe(1);
    expect(q<HTMLVideoElement>("video").play).not.toHaveBeenCalled();
    expect(text("play-label")).toBe("Play");

    document.body.innerHTML = "";
    const moving = mount({ reduced: false });
    await moving.own.choose(videoFile());
    await settle();
    expect(q<HTMLVideoElement>("video").play).toHaveBeenCalled();
    expect(text("play-label")).toBe("Pause");
    q("play").click();
    expect(text("play-label")).toBe("Play");
  });

  it("reads a long video up to the cap, from its start, and says so", async () => {
    const { own } = mount({ video: fakeVideo({ duration: 60 }) });
    await own.choose(videoFile("long.mp4"));
    expect(own.status()).toEqual({ phase: "done", name: "long.mp4", total: MAX_FRAMES, answerable: MAX_FRAMES - 2, capped: true, early: false });
    expect(text("note")).toBe(`long.mp4: ${MAX_FRAMES} frames read, ${MAX_FRAMES - 2} answers. The video is longer than the page reads, so these are its first 30 seconds.`);
  });

  it("stops where a file cut short stops giving new frames, says so, and keeps nothing past it", async () => {
    const { own, sessions } = mount({ video: fakeVideo({ duration: 1, endAt: 8 }) });
    await own.choose(videoFile("cut.mp4"));
    await settle();
    expect(own.status()).toEqual({ phase: "done", name: "cut.mp4", total: 8, answerable: 6, capped: false, early: true });
    expect(text("note")).toBe("cut.mp4: the file ends early. Only its first 0.4 seconds could be read, so these are the answers for that part: 8 frames read, 6 answers.");
    expect(q<HTMLInputElement>("scrub").max).toBe("7");
    expect(text("live")).toContain("the file ends early");
    // The player never goes past the end.
    for (let i = 0; i < 12; i += 1) q("forward").click();
    expect(q<HTMLInputElement>("scrub").value).toBe("7");
    expect(text("when")).toBe("Frame 8 of 8, 0.4\u00A0s in");
    // Answers given before the end was known are not shown for frames past it.
    expect(sessions.log.runs.flatMap((run) => run.firsts.map((pixel) => pixel[0])).filter((frame) => (frame as number) < 8)).toHaveLength(6);
  });

  it("calls a file that stops before one answer damaged", async () => {
    const { own } = mount({ video: fakeVideo({ duration: 1, endAt: 2 }) });
    await own.choose(videoFile("stub.mp4"));
    expect(own.status()).toMatchObject({ phase: "error", failure: "damaged" });
    expect(text("error-title")).toBe("That video looks damaged or incomplete");
  });

  it("reads a video of no stated length up to the cap, and its end there is no early end", async () => {
    const ended = mount({ video: fakeVideo({ duration: Number.POSITIVE_INFINITY, endAt: 30 }) });
    await ended.own.choose(videoFile("recorded.webm"));
    expect(ended.video.seen.planned).toBe(MAX_FRAMES);
    expect(ended.own.status()).toEqual({ phase: "done", name: "recorded.webm", total: 30, answerable: 28, capped: false, early: false });
    expect(text("note")).toBe("recorded.webm: 30 frames read, 28 answers.");

    document.body.innerHTML = "";
    const long = mount({ video: fakeVideo({ duration: Number.POSITIVE_INFINITY }) });
    await long.own.choose(videoFile("long.webm"));
    expect(long.own.status()).toMatchObject({ phase: "done", total: MAX_FRAMES, capped: true });
    expect(text("note")).toBe(`long.webm: ${MAX_FRAMES} frames read, ${MAX_FRAMES - 2} answers. The page reads at most a video’s first 30 seconds, and this one runs at least that long.`);
  });

  it("keeps the scrubber's spoken value on the frame on screen while the video plays", async () => {
    const { own } = mount({ reduced: false });
    await own.choose(videoFile());
    await settle();
    const video = q<HTMLVideoElement>("video");
    video.currentTime = (11 + 0.5) / 20;
    await new Promise((resolve) => setTimeout(resolve, 80));
    expect(q<HTMLInputElement>("scrub").getAttribute("aria-valuetext")).toMatch(/^Frame 12 of 20, 0\.6\u00A0s in: the model estimates /);
  });

  it("writes the rules in once the data is in: the crop, the pace from the measured gap, the cap", async () => {
    mount();
    await settle();
    expect(text("rule-crop")).toBe(
      "It takes the part of each frame with the shape of the camera it learned from (455 × 256), as large as fits: the whole frame of a video that shape, the band across the middle of a portrait one, the middle of a wider one. Of that part it reads the bottom 59%, across its whole width, as it did in training, so nothing is squeezed.",
    );
    expect(text("rule-pace")).toBe(
      "It reads a frame every 50 ms, so each answer compares a frame with one about 100 ms before it: the median gap between a frame and the one two frames before it on the 2018 drive, which it learned from. Its first answer is on the third frame it reads, as each one needs the frame two frames before.",
    );
    expect(text("rule-cap")).toBe("It reads up to 600 frames, so a longer video is read for its first 30 seconds.");
    expect(text("eye-caption")).toBe("What the model gets from this frame, 200 × 66 pixels, with where it looked.");
  });

  it("redraws the drawing of the rule to the camera the model learned from", async () => {
    mount();
    await settle();
    const box = (name: string) => {
      const rect = document.querySelector(`[data-ownclip-art="${name}"]`) as SVGRectElement;
      return ["x", "y", "width", "height"].map((key) => Number(rect.getAttribute(key))) as [number, number, number, number];
    };
    const [, y, , height] = box("land");
    // The landscape frame of the drawing is 90 tall from y = 25 (16:9, a hair
    // wider than the camera); the band is its bottom 150 / 256.
    expect(height).toBeCloseTo((90 * 150) / 256, 1);
    expect(y + height).toBeCloseTo(25 + 90, 1);
    // The wide frame (120 x 50 at x = 276): its centred camera-shaped part,
    // then that part's bottom share, in the camera's shape.
    const [wx, , ww, wh] = box("wide");
    expect(ww).toBeCloseTo((50 * 455) / 256, 1);
    expect(wx).toBeCloseTo(276 + (120 - (50 * 455) / 256) / 2, 1);
    expect(ww / wh).toBeCloseTo(455 / 150, 1);
  });
});

// -- What goes wrong ---------------------------------------------------------------------

describe("a file the page cannot read", () => {
  const expectError = (title: string) => {
    expect(shown("error")).toBe(true);
    expect(shown("player")).toBe(false);
    expect(text("error-title")).toBe(title);
    expect(text("error-text").length).toBeGreaterThan(40);
  };

  it("says an empty file is empty, without opening it or the model", async () => {
    const { own, video, sessions, urls } = mount();
    await own.choose(new File([], "empty.mp4", { type: "video/mp4" }));
    expect(own.status()).toEqual({ phase: "error", name: "empty.mp4", failure: "empty" });
    expectError("There is nothing to read in that file");
    expect(text("error-text")).toBe("It is empty, or too short for one answer, which needs two frames about 100 ms apart.");
    expect(video.seen.opened).toHaveLength(0);
    expect(sessions.log.opened).toBe(0);
    expect(urls.made).toHaveLength(0);
  });

  it("says a picture renamed as a video is not a video, from the bytes it opens with", async () => {
    const { own, video } = mount({ video: fakeVideo({ refuse: { mediaError: 4 } }) });
    await own.choose(videoFile("photo.mp4", PNG_HEAD));
    expect(video.seen.opened[0]?.evidence.container).toBeNull();
    expect(own.status()).toMatchObject({ phase: "error", failure: "not-a-video" });
    expectError("That file is not a video");
  });

  it("says a video it cannot decode is a codec this browser lacks", async () => {
    const { own } = mount({ video: fakeVideo({ refuse: { mediaError: 4 } }) });
    await own.choose(videoFile("prores.mov"));
    expect(own.status()).toMatchObject({ phase: "error", failure: "codec" });
    expectError("This browser cannot decode that video");
  });

  it("says a file with sound and no picture is not a video", async () => {
    const { own } = mount({ video: fakeVideo({ width: 0, height: 0, duration: 3 }) });
    await own.choose(videoFile("song.mp4"));
    expect(own.status()).toMatchObject({ phase: "error", failure: "not-a-video" });
  });

  it("says a video of no known length that cannot be stepped through cannot be placed in time", async () => {
    const endless = Number.POSITIVE_INFINITY;
    const { own } = mount({ video: fakeVideo({ duration: endless, failAt: { frame: 0, evidence: { seekTimedOut: true, duration: endless } } }) });
    await own.choose(videoFile("stream.webm"));
    expect(own.status()).toMatchObject({ phase: "error", failure: "too-long" });
    expectError("That video does not say how long it is");
  });

  it("says a video too short for one answer has nothing to read", async () => {
    const { own } = mount({ video: fakeVideo({ duration: 0.12 }) });
    await own.choose(videoFile("blink.mp4"));
    expect(own.status()).toMatchObject({ phase: "error", failure: "empty" });
  });

  it("stops at a frame it cannot seek to, and lets go of everything it held", async () => {
    const { own, video, sessions, urls } = mount({ video: fakeVideo({ failAt: { frame: 7, evidence: { seekTimedOut: true } } }) });
    await own.choose(videoFile());
    await settle();
    expect(own.status()).toMatchObject({ phase: "error", failure: "codec" });
    expect(video.seen.signals[0]?.aborted).toBe(true);
    expect(video.seen.closed).toBe(1);
    expect(sessions.log.released).toBe(1);
    expect(urls.revoked).toEqual(urls.made);
  });

  it("says when the model itself could not run, and lets the cockpit be", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const cockpit = document.createElement("section");
    cockpit.id = "cockpit";
    cockpit.innerHTML = `<div class="cockpit__console" data-state="ready"><button data-cockpit="play">Pause</button></div>`;
    document.body.append(cockpit);
    const before = cockpit.outerHTML;
    for (const sessions of [fakeSessions({ failOpen: true }), fakeSessions({ failRun: true })]) {
      document.querySelector("#own-video")?.remove();
      const { own } = mount({ sessions });
      await own.choose(videoFile());
      expect(own.status()).toMatchObject({ phase: "error", failure: "unknown" });
      expectError("The model could not read that video");
    }
    expect(cockpit.outerHTML).toBe(before);
  });

  it("takes another file after an error, from the same input", async () => {
    const { own } = mount({ video: fakeVideo({ duration: 1 }) });
    await own.choose(new File([], "empty.mp4"));
    expect(own.status().phase).toBe("error");
    const input = q<HTMLInputElement>("file");
    const clicked = vi.spyOn(input, "click").mockImplementation(() => undefined);
    q("retry").click();
    expect(clicked).toHaveBeenCalled();
    await own.choose(videoFile());
    expect(own.status().phase).toBe("done");
    expect(shown("error")).toBe(false);
  });
});

// -- Stopping and replacing ----------------------------------------------------------------

describe("a reading stopped or replaced", () => {
  it("cancels while sampling: back to the drop zone, with nothing left running or held", async () => {
    const { own, video, sessions, urls } = mount({ video: fakeVideo({ holdAt: 6 }) });
    const reading = own.choose(videoFile());
    await settle(30);
    expect(own.status()).toMatchObject({ phase: "sampling", read: 6 });
    expect(text("status")).toBe("Reading frame 7 of 20");
    expect(shown("cancel")).toBe(true);
    q("cancel").click();
    expect(own.status()).toEqual({ phase: "idle" });
    expect(shown("drop")).toBe(true);
    expect(shown("player")).toBe(false);
    expect(video.seen.signals[0]?.aborted).toBe(true);
    video.seen.release();
    await reading;
    await settle();
    expect(own.status()).toEqual({ phase: "idle" });
    expect(sessions.log.released).toBe(1);
    expect(urls.revoked).toEqual(urls.made);
    expect(document.activeElement).toBe(q("file"));
    expect(text("live")).toBe("Stopped reading your video.");
  });

  it("replaces a reading with a second file cleanly: the first lets go of its file and session, the second finishes", async () => {
    const first = fakeVideo({ holdAt: 4 });
    const second = fakeVideo();
    let calls = 0;
    const open: OpenVideo = (...args) => (calls++ === 0 ? first.open(...args) : second.open(...args));
    const { own, sessions, urls } = mount({ video: { open, seen: first.seen } });
    const a = own.choose(videoFile("first.mp4"));
    await settle(20);
    expect(own.status()).toMatchObject({ phase: "sampling", name: "first.mp4" });
    const b = own.choose(videoFile("second.mp4"));
    expect(first.seen.signals[0]?.aborted).toBe(true);
    expect(urls.revoked).toContain(urls.made[0]);
    first.seen.release();
    await Promise.all([a, b]);
    await settle();
    expect(own.status()).toMatchObject({ phase: "done", name: "second.mp4" });
    expect(sessions.log.opened).toBe(2);
    expect(sessions.log.released).toBe(2);
    expect(text("note")).toBe("second.mp4: 20 frames read, 18 answers.");
  });

  it("replaces a finished video: the player lets go of the first file", async () => {
    const { own, urls } = mount();
    await own.choose(videoFile("first.mp4"));
    expect(urls.revoked).toEqual([]);
    await own.choose(videoFile("second.mp4"));
    expect(urls.revoked).toEqual([urls.made[0]]);
    expect(own.status()).toMatchObject({ phase: "done", name: "second.mp4" });
  });
});

// -- What it never does ------------------------------------------------------------------

describe("the words", () => {
  it("serve the crop rule whole before the data is in: the camera-shaped part, then its bottom, with no number", () => {
    const served = document.createElement("div");
    served.innerHTML = ownclipHtml();
    const rule = served.querySelector('[data-ownclip="rule-crop"]')?.textContent ?? "";
    expect(rule).toMatch(/shape of the camera it learned from, as large as fits/);
    expect(rule).toMatch(/portrait/);
    expect(rule).toMatch(/wider/);
    expect(rule).toMatch(/bottom part/);
    expect(rule).not.toMatch(/full width/);
    expect(rule).not.toMatch(/\d/);
  });

  it("count one answer as one, not as answers", () => {
    const reading = context().reading;
    expect(doneNote("blink.mp4", 3, 1, { capped: false, early: false, endless: false }, reading)).toBe("blink.mp4: 3 frames read, 1 answer.");
    expect(answeredNote(1, 1, reading)).toBe("1 of 1 answer so far.");
    expect(answeredNote(2, 9, reading)).toBe("2 of 9 answers so far.");
  });

  it("say the demo above is not affected only while it is running", () => {
    expect(failureWords("unknown", null, true).text).toMatch(/the demo above is not affected/);
    expect(failureWords("unknown", null, false).text).not.toMatch(/demo above/);
  });
});

describe("the visitor's video stays on the device", () => {
  it("sends nothing anywhere while a video is read: the only request is the model's own download", async () => {
    const requests: Array<{ url: string; body: unknown }> = [];
    vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
      requests.push({ url: String(url), body: init?.body });
      return new Response(new Uint8Array([1, 2, 3]));
    });
    const beacon = vi.fn(() => true);
    Object.defineProperty(navigator, "sendBeacon", { value: beacon, configurable: true });
    const opened = vi.spyOn(XMLHttpRequest.prototype, "open");
    const { own } = mount({ openSession: shippedSession });
    await own.choose(videoFile());
    expect(own.status().phase).toBe("done");
    expect(requests).toEqual([{ url: "https://example.test/models/sightline.onnx", body: undefined }]);
    expect(beacon).not.toHaveBeenCalled();
    expect(opened).not.toHaveBeenCalled();
  });

  it("has no code that could send a file: no request, beacon, socket or form in the section's sources", () => {
    for (const source of [controllerSource, readerSource, planSource]) {
      const code = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
      expect(code).not.toMatch(/\bfetch\s*\(|XMLHttpRequest|sendBeacon|WebSocket|FormData|EventSource|postMessage/);
    }
  });
});

// -- The cockpit, unharmed ----------------------------------------------------------------

describe("the cockpit, beside a reading that fails", () => {
  it("runs on: the reading's session is its own, and the runtime's queue survives the failed run", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    vi.stubGlobal("fetch", async () => new Response(new Uint8Array([1, 2, 3])));
    const runner = await import("../model/runner.ts");
    const cockpitSession = await runner.createSession("models/sightline.onnx");
    runtime.failNext = true; // the next session created -- the reading's -- cannot run
    const { own } = mount({ openSession: shippedSession });
    await own.choose(videoFile());
    expect(own.status()).toMatchObject({ phase: "error", failure: "unknown" });
    expect(runtime.created).toHaveLength(2);
    expect(runtime.created[1]?.failing).toBe(true);
    // The reading let go of its own session, not of the cockpit's.
    await settle();
    expect(runtime.created[1]?.released).toBe(true);
    expect(runtime.created[0]?.released).toBe(false);
    const out = await runner.infer(cockpitSession, new Uint8Array(W * H * 6).fill(9), 1, 6);
    expect(Array.from(out.angles)).toEqual([9]);
  });
});
