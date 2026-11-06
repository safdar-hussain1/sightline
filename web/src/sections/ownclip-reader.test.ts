// @vitest-environment happy-dom
/**
 * The reader on fake video elements: how it opens a file (the element on
 * screen alone first, the second only after it, and never needed), how it
 * shares the frames between the two, how it recovers the length of a video
 * that does not state one, and how it finds where a file stops giving new
 * frames. Each fake element plays a script: when it fails, which source
 * frame a time lands on, and up to where the file can be decoded.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { ClipError, classifyFailure, type Evidence } from "./ownclip-plan.ts";
import { openVideo, type ReaderDeps } from "./ownclip-reader.ts";

interface Script {
  /** The file's length as the element first reports it. */
  readonly duration?: number;
  /** The length it reports after a seek far past the end. */
  readonly recovered?: number;
  readonly width?: number;
  readonly height?: number;
  /** The element refuses the file with this MediaError. */
  readonly refuse?: { readonly code: number; readonly message: string };
  /** The element fails on its n-th seek with this MediaError. */
  readonly failSeek?: { readonly at: number; readonly code: number };
  /** Source frames a second. */
  readonly fps?: number;
  /** The last time the file can be decoded to: later seeks show the frame
   * there again, and present nothing new. */
  readonly decodableTo?: number;
  /** A scene that never changes: every frame's pixels are the same. */
  readonly still?: boolean;
  /** A stretch with no frames at all (a variable frame rate's gap): the
   * picture there is the last frame before it. */
  readonly gap?: { readonly from: number; readonly to: number };
  /** The element refuses the file once with this MediaError, then opens it. */
  readonly refuseOnce?: { readonly code: number; readonly message: string };
  /** How far past the decodable end the decoded-frame count still rises
   * (a browser decoding what is left of a cut frame), seconds. */
  readonly countsPast?: number;
}

/** A fake video element following `script`; `log` records the order of
 * events across every element made. */
function fakeVideo(script: Script, log: string[], name: string) {
  const listeners = new Map<string, Set<() => void>>();
  const fire = (type: string) => queueMicrotask(() => [...(listeners.get(type) ?? [])].forEach((listener) => listener()));
  let time = 0;
  let seeks = 0;
  let duration = Number.NaN;
  let presented = -1;
  let loads = 0;
  const fps = script.fps ?? 30;
  const end = script.decodableTo ?? Number.POSITIVE_INFINITY;
  const watchers = new Set<(mediaTime: number) => void>();
  /** The source frame shown for a seek to `t`. */
  const frameAt = (t: number) => {
    let k = Math.min(Math.floor(t * fps + 1e-9), Math.floor(end * fps));
    if (script.gap && k / fps >= script.gap.from && k / fps < script.gap.to) k = Math.ceil(script.gap.from * fps) - 1;
    return k;
  };
  const video = {
    name,
    /** Frames decoded so far, as a browser counts them: none for a seek past
     * the decodable end. */
    decoded: 0,
    seeks: 0,
    muted: false,
    preload: "",
    error: null as { code: number; message: string } | null,
    readyState: 0,
    videoWidth: 0,
    videoHeight: 0,
    get duration() {
      return duration;
    },
    get currentTime() {
      return time;
    },
    set currentTime(t: number) {
      seeks += 1;
      if (script.failSeek && seeks === script.failSeek.at) {
        video.error = { code: script.failSeek.code, message: "PIPELINE_ERROR_DECODE" };
        fire("error");
        return;
      }
      if (t > 1e9) {
        if (script.recovered !== undefined) {
          duration = script.recovered;
          fire("durationchange");
        }
        fire("seeked");
        return;
      }
      time = t;
      video.seeks += 1;
      if (t <= end + (script.countsPast ?? 0)) video.decoded += 1;
      const frame = frameAt(t);
      // A frame past the decodable end, or in a gap, is the last one again,
      // and nothing new is presented.
      if (frame !== presented) {
        presented = frame;
        queueMicrotask(() => watchers.forEach((watch) => watch(frame / fps)));
      }
      fire("seeked");
    },
    /** The source frame on screen, for the fake drawing. */
    get frame() {
      return script.still ? 0 : presented;
    },
    addEventListener(type: string, listener: () => void) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)?.add(listener);
    },
    removeEventListener(type: string, listener: () => void) {
      listeners.get(type)?.delete(listener);
    },
    removeAttribute: vi.fn(),
    pause: vi.fn(),
    set src(_url: string) {
      log.push(`${name} src`);
    },
    load() {
      loads += 1;
      video.error = null;
      if (script.refuse || (script.refuseOnce && loads === 1)) {
        video.error = (script.refuse ?? script.refuseOnce) as { code: number; message: string };
        log.push(`${name} refused`);
        fire("error");
        return;
      }
      if (script.refuse) {
        video.error = script.refuse;
        log.push(`${name} refused`);
        fire("error");
        return;
      }
      duration = script.duration ?? 10;
      video.videoWidth = script.width ?? 640;
      video.videoHeight = script.height ?? 360;
      video.readyState = 1;
      fire("loadedmetadata");
      queueMicrotask(() => {
        video.readyState = 4;
        presented = 0;
        log.push(`${name} loadeddata`);
        fire("loadeddata");
      });
    },
    watchers,
  };
  return video;
}

type FakeVideo = ReturnType<typeof fakeVideo>;

/** What the fake browser can say of the frames it presents: their media
 * times (requestVideoFrameCallback), a count of frames decoded, or neither. */
interface Signals {
  readonly watch?: boolean;
  readonly counter?: boolean;
}

function deps(second: FakeVideo | (() => never), signals: Signals = { watch: true, counter: true }): ReaderDeps {
  return {
    create: () => (typeof second === "function" ? second() : second) as unknown as HTMLVideoElement,
    // Each frame's pixels say which source frame it is.
    grab: (video) => new Uint8Array(12).fill(((video as unknown as FakeVideo).frame + 1) % 256),
    watch: signals.watch
      ? (video, onFrame) => {
          const fake = video as unknown as FakeVideo;
          fake.watchers.add(onFrame);
          return () => fake.watchers.delete(onFrame);
        }
      : () => null,
    decoded: (video) => (signals.counter ? (video as unknown as FakeVideo).decoded : null),
    visible: () => true,
  };
}

const EVIDENCE: Evidence = { bytes: 5000, container: "matroska" };
const CROP = { sx: 0, sy: 0, sw: 640, sh: 360 };
const times = (n: number, rate = 20) => Array.from({ length: n }, (_, j) => (j + 0.5) / rate);

async function readAll(display: FakeVideo, second: FakeVideo, n: number, signals?: Signals) {
  const signal = new AbortController().signal;
  const source = await openVideo("blob:x", EVIDENCE, signal, display as unknown as HTMLVideoElement, deps(second, signals));
  const got: number[] = [];
  // How far the frames passed on fall behind the seeks made, at worst.
  let lag = 0;
  const outcome = await source.read(
    times(n),
    CROP,
    [2, 2],
    (j) => {
      got.push(j);
      lag = Math.max(lag, display.seeks + second.seeks - got.length);
    },
    signal,
  );
  return { source, got: got.sort((a, b) => a - b), outcome, lag };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("opening a file", () => {
  it("opens the element on screen alone first, and the second only once the first has its first frame", async () => {
    const log: string[] = [];
    const display = fakeVideo({}, log, "display");
    const second = fakeVideo({}, log, "second");
    await openVideo("blob:x", EVIDENCE, new AbortController().signal, display as unknown as HTMLVideoElement, deps(second));
    expect(log).toEqual(["display src", "display loadeddata", "second src", "second loadeddata"]);
  });

  it("reads with one element when the second refuses the file, and the reading completes", async () => {
    const log: string[] = [];
    const display = fakeVideo({}, log, "display");
    const second = fakeVideo({ refuse: { code: 3, message: "PIPELINE_ERROR_DECODE" } }, log, "second");
    const { got, outcome } = await readAll(display, second, 40);
    expect(got).toEqual(Array.from({ length: 40 }, (_, j) => j));
    expect(outcome.end).toBeNull();
  });

  it("carries on with one element when the second fails part way, reading the frame it dropped", async () => {
    const log: string[] = [];
    const display = fakeVideo({}, log, "display");
    const second = fakeVideo({ failSeek: { at: 4, code: 3 } }, log, "second");
    const { got, outcome } = await readAll(display, second, 30);
    expect(got).toEqual(Array.from({ length: 30 }, (_, j) => j));
    expect(outcome.end).toBeNull();
  });

  it("reads the frame the second dropped even when the first had finished its own", async () => {
    const log: string[] = [];
    const display = fakeVideo({}, log, "display");
    // The second fails on its very last seek, after the first has run out of frames.
    const second = fakeVideo({ failSeek: { at: 10, code: 3 } }, log, "second");
    const { got, outcome } = await readAll(display, second, 20);
    expect(got).toEqual(Array.from({ length: 20 }, (_, j) => j));
    expect(outcome.end).toBeNull();
  });

  it("re-opens the element on screen once when it refuses the file with a decoding error, and reads on", async () => {
    const log: string[] = [];
    const display = fakeVideo({ refuseOnce: { code: 3, message: "PIPELINE_ERROR_DECODE" } }, log, "display");
    const second = fakeVideo({}, log, "second");
    const { got, outcome } = await readAll(display, second, 20);
    expect(log.slice(0, 3)).toEqual(["display src", "display refused", "display src"]);
    expect(got).toEqual(Array.from({ length: 20 }, (_, j) => j));
    expect(outcome.end).toBeNull();
  });

  it("re-opens the element on screen once after a decoding error part way, when it reads alone, and reads on", async () => {
    const log: string[] = [];
    const display = fakeVideo({ failSeek: { at: 5, code: 3 } }, log, "display");
    const second = fakeVideo({ refuse: { code: 4, message: "busy" } }, log, "second");
    const { got, outcome } = await readAll(display, second, 30);
    expect(got).toEqual(Array.from({ length: 30 }, (_, j) => j));
    expect(outcome.end).toBeNull();
  });

  it("calls a file undecodable only when the element on screen, alone, refused it", async () => {
    const log: string[] = [];
    const display = fakeVideo({ refuse: { code: 4, message: "DEMUXER_ERROR_NO_SUPPORTED_STREAMS: no supported streams" } }, log, "display");
    const make = vi.fn(() => {
      throw new Error("the second element is never made");
    });
    const opening = openVideo("blob:x", EVIDENCE, new AbortController().signal, display as unknown as HTMLVideoElement, deps(make));
    const error = await opening.catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ClipError);
    expect(classifyFailure(error)).toBe("codec");
    expect(make).not.toHaveBeenCalled();
  });

  it("recovers the length of a video that does not state one, by a seek past its end", async () => {
    const log: string[] = [];
    const display = fakeVideo({ duration: Number.POSITIVE_INFINITY, recovered: 4.95 }, log, "display");
    const second = fakeVideo({ duration: Number.POSITIVE_INFINITY, recovered: 4.95 }, log, "second");
    const source = await openVideo("blob:x", EVIDENCE, new AbortController().signal, display as unknown as HTMLVideoElement, deps(second));
    expect(source.duration).toBe(4.95);
  });

  it("keeps a length it cannot recover unknown, for the caller to read up to its limit", async () => {
    const log: string[] = [];
    const display = fakeVideo({ duration: Number.POSITIVE_INFINITY }, log, "display");
    const second = fakeVideo({ duration: Number.POSITIVE_INFINITY }, log, "second");
    const source = await openVideo("blob:x", EVIDENCE, new AbortController().signal, display as unknown as HTMLVideoElement, deps(second));
    expect(source.duration).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("where a file stops giving new frames", () => {
  it("stops at the first frame of a file cut short, without passing on a single frozen frame", async () => {
    const log: string[] = [];
    // Decodable to 3 s; the planned frames run to 6 s.
    const display = fakeVideo({ decodableTo: 3 }, log, "display");
    const second = fakeVideo({ decodableTo: 3 }, log, "second");
    const { got, outcome } = await readAll(display, second, 120);
    // Frame j is at (j + 0.5) / 20 s: the last one before 3 s is j = 59; the
    // first that shows a frame already shown is j = 60.
    expect(outcome.end).not.toBeNull();
    const end = outcome.end as number;
    expect(end).toBeGreaterThanOrEqual(59);
    expect(end).toBeLessThanOrEqual(61);
    // Every frame before the end was passed on; past it, at most the one frame
    // each element showed new before the end was known (the caller drops it),
    // and never a repeated one.
    expect(got.filter((j) => j < end)).toEqual(Array.from({ length: end }, (_, j) => j));
    expect(got.filter((j) => j >= end).length).toBeLessThanOrEqual(2);
  });

  it("reads a scene that does not change to the end: the frames are new, only their pixels repeat", async () => {
    const log: string[] = [];
    const display = fakeVideo({ still: true }, log, "display");
    const second = fakeVideo({ still: true }, log, "second");
    const { got, outcome } = await readAll(display, second, 80);
    expect(outcome.end).toBeNull();
    expect(got).toHaveLength(80);
  });

  it("reads a slow frame rate to the end: a frame shown for a while is not a frozen file", async () => {
    const log: string[] = [];
    const display = fakeVideo({ fps: 2 }, log, "display");
    const second = fakeVideo({ fps: 2 }, log, "second");
    const { got, outcome } = await readAll(display, second, 80);
    expect(outcome.end).toBeNull();
    expect(got).toHaveLength(80);
  });

  it("reads through a gap in a variable frame rate: the picture stands still a while, then the file goes on", async () => {
    const log: string[] = [];
    const gap = { from: 2.95, to: 5.5 };
    const display = fakeVideo({ gap }, log, "display");
    const second = fakeVideo({ gap }, log, "second");
    const { got, outcome } = await readAll(display, second, 150);
    expect(outcome.end).toBeNull();
    expect(got).toEqual(Array.from({ length: 150 }, (_, j) => j));
  });

  it("passes the frames of a still scene on as it reads them, so the progress keeps moving", async () => {
    const log: string[] = [];
    const display = fakeVideo({ still: true }, log, "display");
    const second = fakeVideo({ still: true }, log, "second");
    const { got, lag } = await readAll(display, second, 120);
    expect(got).toHaveLength(120);
    expect(lag).toBeLessThanOrEqual(3);
  });

  it("never cuts a still scene where the browser says nothing of the frames it presents", async () => {
    const log: string[] = [];
    const display = fakeVideo({ still: true }, log, "display");
    const second = fakeVideo({ still: true }, log, "second");
    const { got, outcome } = await readAll(display, second, 100, { watch: false, counter: false });
    expect(outcome.end).toBeNull();
    expect(got).toHaveLength(100);
  });

  it("says nothing of a file's end where it cannot tell: no cut without a word on the frames presented", async () => {
    const log: string[] = [];
    const display = fakeVideo({ decodableTo: 3 }, log, "display");
    const second = fakeVideo({ decodableTo: 3 }, log, "second");
    const { got, outcome } = await readAll(display, second, 120, { watch: false, counter: false });
    expect(outcome.end).toBeNull();
    expect(got).toHaveLength(120);
  });

  it("with only a count of frames decoded, finds a file cut short and reads a still scene through", async () => {
    const log: string[] = [];
    const cut = await readAll(fakeVideo({ decodableTo: 3 }, log, "display"), fakeVideo({ decodableTo: 3 }, log, "second"), 120, { watch: false, counter: true });
    expect(cut.outcome.end).not.toBeNull();
    expect(cut.outcome.end).toBeGreaterThanOrEqual(59);
    expect(cut.outcome.end).toBeLessThanOrEqual(62);
    const still = await readAll(fakeVideo({ still: true }, log, "display"), fakeVideo({ still: true }, log, "second"), 100, { watch: false, counter: true });
    expect(still.outcome.end).toBeNull();
    expect(still.got).toHaveLength(100);
  });

  it("with only a count, puts the cut where the picture first stood, though the count rose a while past it", async () => {
    const log: string[] = [];
    const cut = await readAll(
      fakeVideo({ decodableTo: 3, countsPast: 0.8 }, log, "display"),
      fakeVideo({ decodableTo: 3, countsPast: 0.8 }, log, "second"),
      120,
      { watch: false, counter: true },
    );
    expect(cut.outcome.end).toBeGreaterThanOrEqual(59);
    expect(cut.outcome.end).toBeLessThanOrEqual(62);
  });

  it("finds the real end of a video of unknown length read up to the limit", async () => {
    const log: string[] = [];
    const display = fakeVideo({ duration: Number.POSITIVE_INFINITY, decodableTo: 2 }, log, "display");
    const second = fakeVideo({ duration: Number.POSITIVE_INFINITY, decodableTo: 2 }, log, "second");
    const { outcome, got } = await readAll(display, second, 200);
    const end = outcome.end as number;
    expect(end).toBeGreaterThanOrEqual(39);
    expect(end).toBeLessThanOrEqual(41);
    expect(got.filter((j) => j < end)).toEqual(Array.from({ length: end }, (_, j) => j));
    expect(got.filter((j) => j >= end).length).toBeLessThanOrEqual(2);
  });
});
