import { describe, expect, it } from "vitest";

import { CROP_ROWS } from "../cockpit/geometry.ts";
import {
  ClipError,
  MAX_FRAMES,
  advance,
  classifyFailure,
  cropRect,
  exceedsCap,
  readingPace,
  sampleTimes,
  sniffContainer,
  type OwnClipState,
} from "./ownclip-plan.ts";

/** The camera frame the model was trained on, as the clips record it. */
const CAMERA = [455, 256] as const;

describe("cropRect", () => {
  it("is the bottom rows the model was trained on, at full width, on a frame of the camera's own size", () => {
    expect(cropRect(455, 256, CAMERA)).toEqual({ sx: 0, sy: 106, sw: 455, sh: 150 });
  });

  it("takes the same share of the height from a larger frame of nearly the same shape", () => {
    const r = cropRect(1920, 1080, CAMERA);
    expect(r.sx).toBe(0);
    expect(r.sw).toBe(1920);
    expect(r.sh).toBeCloseTo((1080 * CROP_ROWS) / 256, 9);
    expect(r.sy + r.sh).toBeCloseTo(1080, 9); // it runs to the bottom edge
  });

  it("keeps full width and the bottom share on a frame wider than the camera's", () => {
    const r = cropRect(2560, 1080, CAMERA);
    expect(r).toEqual({ sx: 0, sy: 1080 - (1080 * CROP_ROWS) / 256, sw: 2560, sh: (1080 * CROP_ROWS) / 256 });
  });

  it("from a portrait frame, takes the band across the middle with the camera's shape, then its bottom share", () => {
    const r = cropRect(720, 1280, CAMERA);
    const band = (720 * 256) / 455; // the camera's shape at the video's width
    const bandTop = (1280 - band) / 2;
    expect(r.sx).toBe(0);
    expect(r.sw).toBe(720);
    expect(r.sh).toBeCloseTo((band * CROP_ROWS) / 256, 9);
    expect(r.sy + r.sh).toBeCloseTo(bandTop + band, 9); // ends where the middle band ends, not at the frame's foot
    expect(r.sy).toBeGreaterThan(1280 / 2 - band / 2);
    // The read part keeps the training crop's shape, so the model's input is not squeezed.
    expect(r.sw / r.sh).toBeCloseTo(455 / CROP_ROWS, 9);
  });

  it("scales a portrait frame's rule with the frame: half the size, half of every edge", () => {
    const big = cropRect(1080, 1920, CAMERA);
    const small = cropRect(540, 960, CAMERA);
    for (const key of ["sx", "sy", "sw", "sh"] as const) expect(small[key]).toBeCloseTo(big[key] / 2, 9);
  });
});

describe("sampleTimes", () => {
  it("places frames one step apart, each in the middle of its step", () => {
    const t = sampleTimes(1, 20);
    expect(t).toHaveLength(20);
    expect(t[0]).toBeCloseTo(0.025, 12);
    for (let j = 1; j < t.length; j += 1) expect((t[j] as number) - (t[j - 1] as number)).toBeCloseTo(0.05, 12);
    expect(t.at(-1) as number).toBeLessThan(1);
  });

  it("counts only whole steps inside the video", () => {
    expect(sampleTimes(0.96, 20)).toHaveLength(19);
    expect(sampleTimes(0.05, 20)).toEqual([0.025]);
  });

  it("stops at the cap on a long video, and says the video went past it", () => {
    expect(sampleTimes(60, 20)).toHaveLength(MAX_FRAMES);
    expect(sampleTimes(60, 20, 7)).toHaveLength(7);
    expect(exceedsCap(60, 20)).toBe(true);
    expect(exceedsCap(MAX_FRAMES / 20, 20)).toBe(false);
    expect(exceedsCap(10, 20)).toBe(false);
  });

  it("has no frames for a video shorter than one step, an empty one or one of no known length", () => {
    expect(sampleTimes(0.04, 20)).toEqual([]);
    expect(sampleTimes(0, 20)).toEqual([]);
    expect(sampleTimes(Number.NaN, 20)).toEqual([]);
    expect(sampleTimes(Number.POSITIVE_INFINITY, 20)).toEqual([]);
    expect(sampleTimes(5, 0)).toEqual([]);
  });
});

describe("readingPace", () => {
  const gaps = new Map([[2, 0.1]]);

  it("steps so that a frame and its history frame are the measured gap apart", () => {
    const pace = readingPace([2], gaps, 17.5);
    expect(pace.lag).toBe(2);
    expect(pace.gapS).toBe(0.1);
    expect(pace.stepS).toBeCloseTo(0.05, 12);
    expect(pace.rate).toBeCloseTo(20, 12);
  });

  it("falls back to the clips' frame rate, counting frames only, where the gap was not measured", () => {
    const pace = readingPace([2], null, 17.5);
    expect(pace).toEqual({ lag: 2, gapS: null, stepS: 1 / 17.5, rate: 17.5 });
    expect(readingPace([2], new Map([[1, 0.05]]), 17.5).gapS).toBeNull();
  });

  it("paces a model with no history at the clips' rate", () => {
    expect(readingPace([], gaps, 17.5)).toEqual({ lag: 0, gapS: null, stepS: 1 / 17.5, rate: 17.5 });
  });
});

describe("sniffContainer", () => {
  const bytes = (...values: number[]) => Uint8Array.from(values);
  const ascii = (text: string) => Array.from(text, (c) => c.charCodeAt(0));

  it("knows the video containers by their first bytes", () => {
    expect(sniffContainer(bytes(0, 0, 0, 32, ...ascii("ftypisom")))).toBe("mp4");
    expect(sniffContainer(bytes(0, 0, 0, 20, ...ascii("ftypqt  ")))).toBe("mp4");
    expect(sniffContainer(bytes(0, 0, 0, 8, ...ascii("wide")))).toBe("mp4");
    expect(sniffContainer(bytes(0x1a, 0x45, 0xdf, 0xa3, 1, 0))).toBe("matroska");
    expect(sniffContainer(bytes(...ascii("RIFF"), 0, 0, 0, 0, ...ascii("AVI ")))).toBe("avi");
    expect(sniffContainer(bytes(...ascii("OggS"), 0))).toBe("ogg");
    expect(sniffContainer(bytes(0, 0, 1, 0xba, 0x44))).toBe("mpeg");
    const ts = new Uint8Array(400);
    ts[0] = 0x47;
    ts[188] = 0x47;
    expect(sniffContainer(ts)).toBe("mpeg");
  });

  it("knows nothing else: a picture, a text, a WAV file or too few bytes", () => {
    expect(sniffContainer(bytes(0x89, ...ascii("PNG\r\n"), 0x1a, 0x0a))).toBeNull();
    expect(sniffContainer(bytes(0xff, 0xd8, 0xff, 0xe0))).toBeNull();
    expect(sniffContainer(bytes(...ascii("hello, world")))).toBeNull();
    expect(sniffContainer(bytes(...ascii("RIFF"), 0, 0, 0, 0, ...ascii("WAVE")))).toBeNull();
    expect(sniffContainer(bytes())).toBeNull();
  });
});

describe("classifyFailure", () => {
  const fail = (evidence: ConstructorParameters<typeof ClipError>[1]) => classifyFailure(new ClipError("x", evidence));

  it("calls a file of no bytes empty", () => {
    expect(fail({ bytes: 0 })).toBe("empty");
  });

  it("calls a file the browser refused and whose bytes are no video container not a video", () => {
    expect(fail({ bytes: 1000, container: null, mediaError: 4 })).toBe("not-a-video");
  });

  it("calls a video container the browser cannot decode a codec failure", () => {
    expect(fail({ bytes: 1000, container: "mp4", mediaError: 4 })).toBe("codec");
    expect(fail({ bytes: 1000, container: "matroska", mediaError: 3 })).toBe("codec");
    expect(fail({ bytes: 1000, container: "mp4", seekTimedOut: true })).toBe("codec");
  });

  it("calls a video with no picture (sound alone) not a video", () => {
    expect(fail({ bytes: 1000, container: "mp4", width: 0, height: 0, duration: 3 })).toBe("not-a-video");
  });

  it("calls a video of endless or unknown length too long", () => {
    expect(fail({ bytes: 1000, container: "matroska", width: 640, height: 360, duration: Number.POSITIVE_INFINITY })).toBe("too-long");
    expect(fail({ bytes: 1000, container: "mp4", width: 640, height: 360, duration: Number.NaN })).toBe("too-long");
  });

  it("calls a video too short for one answer empty", () => {
    expect(fail({ bytes: 1000, container: "mp4", width: 640, height: 360, duration: 0.04, frames: 0, needed: 3 })).toBe("empty");
    expect(fail({ bytes: 1000, container: "mp4", width: 640, height: 360, duration: 0.1, frames: 2, needed: 3 })).toBe("empty");
  });

  it("calls anything else unknown", () => {
    expect(fail({ bytes: 1000, container: "mp4", width: 640, height: 360, duration: 3, frames: 60, needed: 3 })).toBe("unknown");
    expect(classifyFailure(new Error("the model could not be downloaded"))).toBe("unknown");
    expect(classifyFailure("a string")).toBe("unknown");
  });
});

describe("advance: the reading's states", () => {
  const idle: OwnClipState = { phase: "idle" };
  const loading = advance(idle, { type: "choose", name: "drive.mp4" });
  const opened = advance(loading, { type: "opened", total: 10, answerable: 8, capped: false });

  it("goes idle, loading, sampling with its progress, running, done", () => {
    expect(loading).toEqual({ phase: "loading", name: "drive.mp4" });
    expect(opened).toEqual({ phase: "sampling", name: "drive.mp4", read: 0, total: 10, answered: 0, answerable: 8, capped: false });
    const reading = advance(advance(opened, { type: "read", count: 4 }), { type: "answered", count: 2 });
    expect(reading).toMatchObject({ phase: "sampling", read: 4, answered: 2 });
    const running = advance(reading, { type: "read", count: 10 });
    expect(running).toEqual({ phase: "running", name: "drive.mp4", total: 10, answered: 2, answerable: 8, capped: false });
    expect(advance(running, { type: "answered", count: 7 })).toMatchObject({ phase: "running", answered: 7 });
    expect(advance(running, { type: "answered", count: 8 })).toEqual({ phase: "done", name: "drive.mp4", total: 10, answerable: 8, capped: false });
  });

  it("is done straight from sampling when the last read leaves nothing to answer", () => {
    const all = advance(advance(opened, { type: "answered", count: 8 }), { type: "read", count: 10 });
    expect(all.phase).toBe("done");
  });

  it("goes to error from any working state, and keeps the class of the failure", () => {
    for (const state of [loading, opened, advance(opened, { type: "read", count: 10 })]) {
      expect(advance(state, { type: "fail", failure: "codec" })).toEqual({ phase: "error", name: "drive.mp4", failure: "codec" });
    }
  });

  it("cancels while loading or sampling, back to idle", () => {
    expect(advance(loading, { type: "cancel" })).toEqual(idle);
    expect(advance(advance(opened, { type: "read", count: 3 }), { type: "cancel" })).toEqual(idle);
  });

  it("replaces whatever is on screen when another file is chosen", () => {
    const done = advance(advance(opened, { type: "answered", count: 8 }), { type: "read", count: 10 });
    const failed = advance(opened, { type: "fail", failure: "empty" });
    for (const state of [opened, done, failed]) {
      expect(advance(state, { type: "choose", name: "other.webm" })).toEqual({ phase: "loading", name: "other.webm" });
    }
  });

  it("ignores events that do not belong to the state it is in", () => {
    expect(advance(idle, { type: "read", count: 3 })).toBe(idle);
    expect(advance(idle, { type: "fail", failure: "codec" })).toBe(idle);
    expect(advance(idle, { type: "cancel" })).toBe(idle);
    expect(advance(loading, { type: "read", count: 3 })).toBe(loading);
    const done = advance(advance(opened, { type: "answered", count: 8 }), { type: "read", count: 10 });
    expect(advance(done, { type: "fail", failure: "codec" })).toBe(done);
    expect(advance(done, { type: "cancel" })).toBe(done);
    expect(advance(opened, { type: "opened", total: 3, answerable: 1, capped: false })).toBe(opened);
  });

  it("never counts backwards or past the end", () => {
    const reading = advance(opened, { type: "read", count: 5 });
    expect(advance(reading, { type: "read", count: 3 })).toBe(reading);
    expect(advance(reading, { type: "read", count: 11 })).toBe(reading);
    expect(advance(reading, { type: "answered", count: 9 })).toBe(reading);
  });
});
