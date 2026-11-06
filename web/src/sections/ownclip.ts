/**
 * Try it on your own video: a file the visitor chooses or drops, read in this
 * browser, with the shipped model's estimate on every frame it reads.
 *
 * The video never leaves the device. The file is read through an object URL
 * (`blob:`), its frames are taken by seeking (ownclip-reader.ts), and the
 * model runs here, in WebAssembly; the page has no code that sends a file
 * anywhere. The only downloads are the files the cockpit already uses: the
 * model, and the clip files that name its input and history.
 *
 * A reading goes idle, loading, sampling, running, done -- or error -- as
 * ownclip-plan.ts's `advance` says, and this module draws each state. While
 * the frames are read the model answers in batches as soon as each frame and
 * its history are in, so the wheel turns, the model's own input fills in and
 * the trace draws itself while the video is still being read. Then a player:
 * the video, the heat over the part the model read, the wheel turned to the
 * model's estimate, the angle and a scrubber. There is no driver's wheel
 * and no error: nobody recorded the steering in a visitor's video.
 *
 * The reading has its own model session, apart from the cockpit's, so
 * nothing that goes wrong with a visitor's file can stop the cockpit: a
 * failure here is caught and said here. Inference on both sessions waits in
 * the runner's one queue, so the two never run at once. A new file, a
 * cancel or a failure ends the current reading cleanly: its seeks stop, its
 * session is released once its last run is over, and its object URL is
 * revoked.
 */

import { heatImageFromBytes, quantizeSaliency } from "../cockpit/saliency.ts";
import { frameIndexAt, seekTime } from "../cockpit/player.ts";
import { stackAt, type Bytes } from "../cockpit/sprites.ts";
import { renderWheel } from "../cockpit/wheel.ts";
import { CROP_ROWS } from "../cockpit/geometry.ts";
import { loadClip } from "../data/load.ts";
import { driveGaps } from "../data/timing.ts";
import type { Clip, Results } from "../data/types.ts";
import { NO_VALUE, steerParts } from "../format.ts";
import { prefersReducedMotion } from "./motion.ts";
import {
  OPENING,
  answeredNote,
  capRule,
  cropRule,
  doneNote,
  eyeCaptionText,
  failureWords,
  finishingStatus,
  frameWhen,
  paceRule,
  readingStatus,
  scrubText,
  type Reading,
} from "./ownclip-copy.ts";
import { ART_FRAMES, artBand, artMiddle } from "./ownclip-markup.ts";
import {
  ClipError,
  advance,
  classifyFailure,
  cropRect,
  exceedsCap,
  readingPace,
  sampleTimes,
  sniffContainer,
  type Evidence,
  type OwnClipEvent,
  type OwnClipState,
  type Rect,
} from "./ownclip-plan.ts";
import { emptyVideo, openVideo as browserOpenVideo, type OpenVideo, type VideoSource } from "./ownclip-reader.ts";
import { Trace } from "./ownclip-trace.ts";

/** The heat's opacity where the model looks hardest: the cockpit's own. */
const HEAT_ALPHA = 0.85;
/** How long one inference batch may hold the page, ms, and its bounds. */
const BATCH_BUDGET_MS = 40;
const MAX_BATCH = 8;
/** How many of a file's first bytes tell its container. */
const SNIFF_BYTES = 512;
/** A screen no taller than this shape (width over height): a portrait video
 * is drawn whole inside it, with the console's dark at its sides. */
const TALLEST_SCREEN = 3 / 4;

// -- What the section needs from outside ------------------------------------------------

/** One model session, as the reading uses it. */
export interface Session {
  run(frames: Uint8Array, n: number, channels: number): Promise<{ readonly angles: Float32Array; readonly saliency: Float32Array }>;
  release(): Promise<void>;
}

export type OpenSession = (modelUrl: string, inputW: number, inputH: number) => Promise<Session>;

/** What the page's data says about reading a video: the model, its input and
 * history, the camera it learned from and the pace to read at. */
export interface Context {
  readonly modelUrl: string;
  readonly reading: Reading;
}

export interface ObjectUrls {
  create(file: Blob): string;
  revoke(url: string): void;
}

export interface Deps {
  readonly context: Promise<Context>;
  readonly openSession?: OpenSession;
  readonly openVideo?: OpenVideo;
  readonly urls?: ObjectUrls;
  readonly reduced?: () => boolean;
}

/** A session on the shipped model, opened with the runtime the cockpit uses.
 * The runtime's chunk is imported here only once a video is chosen. */
export const shippedSession: OpenSession = async (modelUrl, inputW, inputH) => {
  const runner = await import("../model/runner.ts");
  if (runner.INPUT_W !== inputW || runner.INPUT_H !== inputH) {
    throw new Error(`the clips' frames are ${inputW} x ${inputH}, not the model's ${runner.INPUT_W} x ${runner.INPUT_H}`);
  }
  const session = await runner.createSession(modelUrl);
  return {
    run: (frames, n, channels) => runner.infer(session, frames, n, channels),
    release: () => session.release(),
  };
};

/** The reading's context from results.json and the clips it names: the
 * model's history and input size and the camera from the clip files, and the
 * pace from the measured gaps of the drive the shipped model learned from
 * (the clip whose drive has them). */
export async function loadContext(results: Promise<Results>): Promise<Context> {
  const loaded = await results;
  const clips = await Promise.all(Object.values(loaded.assets.clips).map((path) => loadClip(new URL(path, document.baseURI).href)));
  const gaps = driveGaps(loaded.raw);
  const timed = clips.find((clip) => gaps[clip.drive] != null);
  const clip: Clip | undefined = timed ?? clips[0];
  if (!clip) throw new Error("results.json names no clips");
  const drive = timed ? timed.drive : null;
  const pace = readingPace(clip.frame_lags, drive ? (gaps[drive] ?? null) : null, clip.fps);
  const model = loaded.assets.models.sightline;
  return {
    modelUrl: new URL(model, document.baseURI).href,
    reading: {
      pace,
      lags: clip.frame_lags,
      camera: clip.camera_size,
      inputW: clip.sprite.frame_w,
      inputH: clip.sprite.frame_h,
      drive,
      trainedOnDrive: drive !== null && loaded.shipped.run.startsWith(`${drive}_`),
    },
  };
}

// -- One reading ----------------------------------------------------------------------------

/** The frames of one video and the model's answers on them. */
interface Result {
  readonly name: string;
  readonly times: readonly number[];
  readonly crop: Rect;
  readonly width: number;
  readonly height: number;
  readonly frames: Array<Bytes | undefined>;
  readonly angles: Float32Array;
  /** One byte per pixel per frame (see `quantizeSaliency`). */
  readonly heat: Uint8Array;
  readonly done: Uint8Array;
  readonly capped: boolean;
}

/** A reading in progress, and what it holds that must be let go of. */
interface Run {
  readonly controller: AbortController;
  url: string | null;
  source: VideoSource | null;
  session: Promise<Session> | null;
  /** The last inference started, so the session is released after it. */
  pending: Promise<unknown>;
}

function stopped(): DOMException {
  return new DOMException("The reading was replaced or stopped", "AbortError");
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export class OwnClip {
  /** The section in the page, or null on a page without it. */
  static mount(results: Promise<Results>): OwnClip | null {
    const section = document.getElementById("own-video");
    if (!section) return null;
    const context = loadContext(results);
    context.catch(() => undefined); // said when a file is chosen
    return new OwnClip(section, { context });
  }

  private readonly section: HTMLElement;
  private readonly context: Promise<Context>;
  private readonly openSession: OpenSession;
  private readonly openVideo: OpenVideo;
  private readonly urls: ObjectUrls;
  private readonly reduced: () => boolean;
  private reading: Reading | null = null;

  private state: OwnClipState = { phase: "idle" };
  private run: Run | null = null;
  private result: Result | null = null;
  /** The object URL the player plays, once a reading is done. */
  private playerUrl: string | null = null;

  private readonly trace: Trace;
  private readonly heatContext: CanvasRenderingContext2D | null;
  private readonly eyeContext: CanvasRenderingContext2D | null;
  private frame = -1;
  private playing = false;
  /** The video should be playing (the opening play, or the visitor's). */
  private wanted = false;
  private offscreen = false;
  private stopLoop: (() => void) | null = null;

  constructor(section: HTMLElement, deps: Deps) {
    this.section = section;
    this.context = deps.context;
    this.openSession = deps.openSession ?? shippedSession;
    this.openVideo = deps.openVideo ?? browserOpenVideo;
    this.urls = deps.urls ?? { create: (file) => URL.createObjectURL(file), revoke: (url) => URL.revokeObjectURL(url) };
    this.reduced = deps.reduced ?? prefersReducedMotion;
    this.trace = new Trace(this.part<HTMLCanvasElement>("trace-canvas"));
    this.heatContext = this.part<HTMLCanvasElement>("heat").getContext("2d");
    this.eyeContext = this.part<HTMLCanvasElement>("eye").getContext("2d");
    this.wire();
    this.context.then(
      (context) => this.describe(context.reading),
      () => undefined,
    );
    this.show(null);
  }

  private part<T extends HTMLElement = HTMLElement>(name: string): T {
    const element = this.section.querySelector<T>(`[data-ownclip="${name}"]`);
    if (!element) throw new Error(`the own-video section has no ${name}`);
    return element;
  }

  /** What the section is doing, for whoever reads the page. */
  status(): OwnClipState {
    return this.state;
  }

  // -- The rules, in numbers ---------------------------------------------------------------

  private describe(reading: Reading): void {
    this.reading = reading;
    this.part("rule-crop").textContent = cropRule(reading);
    this.part("rule-pace").textContent = paceRule(reading);
    this.part("rule-cap").textContent = capRule(reading);
    this.part("eye-caption").textContent = eyeCaptionText(reading);
    for (const canvas of [this.part<HTMLCanvasElement>("heat"), this.part<HTMLCanvasElement>("eye")]) {
      canvas.width = reading.inputW;
      canvas.height = reading.inputH;
    }
    // The drawing of the rule, redrawn to the camera the model learned from.
    const [w, h] = reading.camera;
    const camera = { aspect: w / h, share: Math.min(CROP_ROWS, h) / h };
    const set = (name: string, r: Rect) => {
      const rect = this.section.querySelector(`[data-ownclip-art="${name}"]`);
      if (!rect) return;
      rect.setAttribute("x", String(r.sx));
      rect.setAttribute("y", String(r.sy));
      rect.setAttribute("width", String(r.sw));
      rect.setAttribute("height", String(r.sh));
    };
    set("land", artBand(ART_FRAMES.landscape, camera));
    set("port", artBand(ART_FRAMES.portrait, camera));
    set("middle", artMiddle(ART_FRAMES.portrait, camera));
  }

  // -- Input ------------------------------------------------------------------------------

  private wire(): void {
    const input = this.part<HTMLInputElement>("file");
    input.addEventListener("change", () => {
      const file = input.files?.[0];
      // Cleared, so choosing the same file again is a change too.
      input.value = "";
      if (file) void this.choose(file);
    });
    for (const name of ["retry", "again"]) this.part(name).addEventListener("click", () => input.click());
    this.part("cancel").addEventListener("click", () => this.cancel());

    // Dropping a file anywhere on the console: an enhancement over the input.
    const console = this.part("console");
    const carriesFiles = (event: DragEvent) => [...(event.dataTransfer?.types ?? [])].includes("Files");
    console.addEventListener("dragover", (event) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      console.dataset["drag"] = "true";
    });
    console.addEventListener("dragleave", (event) => {
      if (!(event.relatedTarget instanceof Node && console.contains(event.relatedTarget))) delete console.dataset["drag"];
    });
    console.addEventListener("drop", (event) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      delete console.dataset["drag"];
      const file = event.dataTransfer?.files[0];
      if (file) void this.choose(file);
    });

    this.part("play").addEventListener("click", () => {
      if (this.state.phase !== "done") return;
      if (this.playing) {
        this.wanted = false;
        this.pause();
      } else {
        this.wanted = true;
        void this.play();
      }
    });
    this.part("back").addEventListener("click", () => this.step(-1));
    this.part("forward").addEventListener("click", () => this.step(1));
    const scrub = this.part<HTMLInputElement>("scrub");
    scrub.addEventListener("input", () => this.seek(Number(scrub.value)));

    const video = this.part<HTMLVideoElement>("video");
    video.addEventListener("ended", () => {
      if (this.state.phase === "done" && this.playing) this.wrap();
    });

    // Out of sight, or behind another tab, the player pauses; it picks up
    // again on return if it was playing.
    if (typeof IntersectionObserver === "function") {
      new IntersectionObserver(
        (entries) => {
          this.offscreen = !entries.some((entry) => entry.isIntersecting);
          this.followSight();
        },
        { threshold: 0.2 },
      ).observe(console);
    }
    document.addEventListener("visibilitychange", () => this.followSight());
  }

  private followSight(): void {
    if (this.state.phase !== "done") return;
    const hidden = this.offscreen || document.visibilityState === "hidden";
    if (hidden && this.playing) this.pause();
    else if (!hidden && this.wanted && !this.playing) void this.play();
  }

  // -- A reading --------------------------------------------------------------------------

  /** Read `file`, replacing whatever the section holds. */
  async choose(file: File): Promise<void> {
    this.endRun(false);
    this.endPlayer();
    const run: Run = { controller: new AbortController(), url: null, source: null, session: null, pending: Promise.resolve() };
    this.run = run;
    this.apply({ type: "choose", name: file.name });
    const live = () => {
      if (this.run !== run || run.controller.signal.aborted) throw stopped();
    };
    try {
      const context = await this.context;
      live();
      const { reading } = context;
      if (file.size === 0) throw new ClipError("the file is empty", { bytes: 0 });
      const container = sniffContainer(new Uint8Array(await file.slice(0, SNIFF_BYTES).arrayBuffer()));
      live();
      const evidence: Evidence = { bytes: file.size, container };
      run.url = this.urls.create(file);
      run.session = this.openSession(context.modelUrl, reading.inputW, reading.inputH);
      run.session.catch(() => undefined); // awaited below, or let go of by endRun
      const source = await this.openVideo(run.url, evidence, run.controller.signal, this.part<HTMLVideoElement>("video"));
      run.source = source;
      live();

      const { pace } = reading;
      const times = sampleTimes(source.duration, pace.rate);
      const needed = pace.lag + 1;
      const seen: Evidence = { ...evidence, width: source.width, height: source.height, duration: source.duration, frames: times.length, needed };
      // The reader refuses these too; a picture of no size, or a video of no
      // known length or too short for one answer, never reaches the model.
      if (!(source.width > 0 && source.height > 0)) throw new ClipError("the video has no picture", seen);
      if (times.length < needed) throw new ClipError("the video is too short for one answer, or of no known length", seen);
      const total = times.length;
      const map = reading.inputW * reading.inputH;
      const result: Result = {
        name: file.name,
        times,
        crop: cropRect(source.width, source.height, reading.camera),
        width: source.width,
        height: source.height,
        frames: new Array<Bytes | undefined>(total),
        angles: new Float32Array(total).fill(Number.NaN),
        heat: new Uint8Array(total * map),
        done: new Uint8Array(total),
        capped: exceedsCap(source.duration, pace.rate),
      };
      this.result = result;
      this.layOut(result);
      this.apply({ type: "opened", total, answerable: total - pace.lag, capped: result.capped });

      const session = await run.session;
      live();
      await this.readAll(run, source, session, result, reading, live);
      live();
      // Done: the second reader and the session go; the video on screen
      // keeps the file, and the player plays it.
      this.playerUrl = run.url;
      run.url = null;
      this.endRun(true);
      await this.startPlayer(result, reading);
    } catch (error) {
      if (isAbort(error) || this.run !== run) return; // stopped, or replaced by another file
      const failure = classifyFailure(error);
      // The visitor's file being what it is is no fault of the page's; anything
      // else is worth its detail in the console.
      if (failure === "unknown") console.warn("Your video could not be read:", error);
      this.endRun(false);
      this.apply({ type: "fail", failure });
    }
  }

  /** Read every frame of `result` and run the model on each one that has
   * its history, a batch at a time, as soon as its frames are in. */
  private async readAll(run: Run, source: VideoSource, session: Session, result: Result, reading: Reading, live: () => void): Promise<void> {
    const { frames, times } = result;
    const total = times.length;
    const { lag } = reading.pace;
    const map = reading.inputW * reading.inputH;
    const channels = 3 * (1 + reading.lags.length);
    let read = 0;
    let finished = false;
    let failure: unknown = null;
    let wake: (() => void) | null = null;
    const nudge = () => {
      const waiting = wake;
      wake = null;
      waiting?.();
    };

    const reads = source
      .read(
        times,
        result.crop,
        [reading.inputW, reading.inputH],
        (j, rgb, picture) => {
          if (this.run !== run) return;
          frames[j] = rgb;
          if (j === 0 || !this.reduced()) this.paintStill(picture);
          const before = read;
          while (read < total && frames[read]) read += 1;
          if (read !== before) {
            this.apply({ type: "read", count: read });
            this.trace.update(read, result.angles);
          }
          nudge();
        },
        run.controller.signal,
      )
      .then(
        () => {
          finished = true;
          nudge();
        },
        (error: unknown) => {
          failure = error;
          nudge();
        },
      );

    let next = lag;
    let batch = 2;
    while (next < total) {
      const want = Math.min(total, next + batch);
      while (read < want && failure === null && !finished) await new Promise<void>((resolve) => (wake = resolve));
      if (failure !== null) throw failure;
      live();
      const upto = Math.min(read, want);
      if (upto <= next) throw new Error("the reader stopped short of the frames it was asked for");
      const count = upto - next;
      const stride = map * channels;
      const stacks = new Uint8Array(count * stride);
      for (let i = 0; i < count; i += 1) stacks.set(stackAt(frames as Bytes[], next + i, reading.lags), i * stride);
      const started = performance.now();
      const running = session.run(stacks, count, channels);
      run.pending = running.catch(() => undefined);
      const out = await running;
      live();
      for (let i = 0; i < count; i += 1) {
        const k = next + i;
        result.angles[k] = out.angles[i] as number;
        result.heat.set(quantizeSaliency(out.saliency.subarray(i * map, (i + 1) * map)), k * map);
        result.done[k] = 1;
      }
      const perFrame = (performance.now() - started) / count;
      batch = Math.max(1, Math.min(MAX_BATCH, Math.floor(BATCH_BUDGET_MS / Math.max(perFrame, 1))));
      next = upto;
      this.apply({ type: "answered", count: next - lag });
      this.trace.update(read, result.angles);
      this.showAnswer(result, next - 1, reading);
      // Let the page and the seeks through before the next batch.
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    await reads;
    if (failure !== null) throw failure;
  }

  /** Stop the stage on the last frame read while the video is read; under
   * reduced motion it holds its first frame. */
  private paintStill(picture: HTMLVideoElement): void {
    const still = this.part<HTMLCanvasElement>("still");
    const context = still.getContext("2d");
    if (!context || still.width === 0) return;
    try {
      context.drawImage(picture, 0, 0, still.width, still.height);
    } catch {
      // A frame the canvas cannot take is only a missed picture on screen.
    }
  }

  /** Cancel the reading in progress. */
  cancel(): void {
    if (!(this.state.phase === "loading" || this.state.phase === "sampling" || this.state.phase === "running")) return;
    this.endRun(false);
    this.endPlayer();
    this.apply({ type: "cancel" });
    this.part<HTMLInputElement>("file").focus();
  }

  /** Let go of the reading in progress: stop its seeks, close its readers,
   * release its session once its last run is over, and revoke its URL
   * unless the player has taken it (`keepDisplay`). */
  private endRun(keepDisplay: boolean): void {
    const run = this.run;
    if (!run) return;
    this.run = null;
    run.controller.abort();
    run.source?.close();
    if (!keepDisplay) emptyVideo(this.part<HTMLVideoElement>("video"));
    if (run.url !== null) this.urls.revoke(run.url);
    run.url = null;
    const pending = run.pending;
    void run.session
      ?.then(async (session) => {
        await pending;
        await session.release();
      })
      .catch(() => undefined);
  }

  // -- The player -------------------------------------------------------------------------

  /** Size the screen and the band to the video. */
  private layOut(result: Result): void {
    const { width, height, crop } = result;
    const frame = this.part("frame");
    const screen = this.part("screen");
    screen.style.setProperty("--screen-ar", String(Math.max(width / height, TALLEST_SCREEN)));
    frame.style.setProperty("--video-ar", String(width / height));
    const band = this.part("band").style;
    band.left = `${(100 * crop.sx) / width}%`;
    band.top = `${(100 * crop.sy) / height}%`;
    band.width = `${(100 * crop.sw) / width}%`;
    band.height = `${(100 * crop.sh) / height}%`;
    const still = this.part<HTMLCanvasElement>("still");
    const scale = Math.min(1, 960 / width);
    still.width = Math.max(1, Math.round(width * scale));
    still.height = Math.max(1, Math.round(height * scale));
    still.hidden = false;
    this.heatContext?.clearRect(0, 0, this.part<HTMLCanvasElement>("heat").width, this.part<HTMLCanvasElement>("heat").height);
    this.trace.reset(result.times.length);
    const scrub = this.part<HTMLInputElement>("scrub");
    scrub.min = String(this.reading?.pace.lag ?? 0);
    scrub.max = String(result.times.length - 1);
    scrub.value = scrub.min;
    this.clearReadout();
  }

  private clearReadout(): void {
    this.part("angle").textContent = NO_VALUE;
    this.part("side").textContent = "";
    this.part("when").textContent = "";
    this.part("wheel").dataset["waiting"] = "true";
    const wheel = this.part("wheel").querySelector("svg");
    if (wheel) renderWheel(wheel, 0, 0);
    const eye = this.part<HTMLCanvasElement>("eye");
    this.eyeContext?.clearRect(0, 0, eye.width, eye.height);
  }

  /** The wheel, the readout and the model's own input for frame `k`. */
  private showAnswer(result: Result, k: number, reading: Reading): void {
    const done = result.done[k] === 1;
    const angle = done ? (result.angles[k] as number) : Number.NaN;
    const wheel = this.part("wheel");
    wheel.dataset["waiting"] = String(!done);
    const svg = wheel.querySelector("svg");
    if (svg && done) renderWheel(svg, angle, angle);
    const parts = steerParts(angle);
    this.part("angle").textContent = parts.size;
    this.part("side").textContent = parts.side;
    this.part("when").textContent = frameWhen(k, result.times.length, reading.pace);
    this.paintEye(result, k, reading);
  }

  private paintEye(result: Result, k: number, reading: Reading): void {
    const context = this.eyeContext;
    const rgb = result.frames[k];
    if (!context || !rgb) return;
    const { inputW: w, inputH: h } = reading;
    const image = context.createImageData(w, h);
    for (let p = 0, from = 0, to = 0; p < w * h; p += 1, from += 3, to += 4) {
      image.data[to] = rgb[from] as number;
      image.data[to + 1] = rgb[from + 1] as number;
      image.data[to + 2] = rgb[from + 2] as number;
      image.data[to + 3] = 255;
    }
    context.putImageData(image, 0, 0);
    if (result.done[k]) {
      const heat = this.heatImage(result, k, reading);
      const scratch = document.createElement("canvas");
      scratch.width = w;
      scratch.height = h;
      scratch.getContext("2d")?.putImageData(heat, 0, 0);
      context.drawImage(scratch, 0, 0);
    }
  }

  private heatImage(result: Result, k: number, reading: Reading): ImageData {
    const map = reading.inputW * reading.inputH;
    return heatImageFromBytes(result.heat.subarray(k * map, (k + 1) * map), reading.inputW, reading.inputH, HEAT_ALPHA);
  }

  /** Show frame `k` of the read video in the player. */
  private render(k: number, announce: boolean): void {
    const result = this.result;
    const reading = this.reading;
    if (!result || !reading || this.state.phase !== "done") return;
    this.frame = k;
    if (this.heatContext) {
      if (result.done[k]) this.heatContext.putImageData(this.heatImage(result, k, reading), 0, 0);
      else this.heatContext.clearRect(0, 0, reading.inputW, reading.inputH);
    }
    this.showAnswer(result, k, reading);
    this.trace.setFrame(k);
    const scrub = this.part<HTMLInputElement>("scrub");
    scrub.value = String(k);
    if (announce) scrub.setAttribute("aria-valuetext", scrubText(k, result.times.length, reading.pace, steerParts(result.angles[k] as number)));
  }

  private async startPlayer(result: Result, reading: Reading): Promise<void> {
    const video = this.part<HTMLVideoElement>("video");
    const first = reading.pace.lag;
    await this.seekVideo(seekTime(first, reading.pace.rate));
    if (this.result !== result || this.state.phase !== "done") return;
    this.part("still").hidden = true;
    this.render(first, true);
    this.wanted = !this.reduced();
    video.loop = false;
    if (this.wanted && !this.offscreen) await this.play();
  }

  /** Seek the video on screen, waiting a moment at most. */
  private seekVideo(t: number): Promise<void> {
    const video = this.part<HTMLVideoElement>("video");
    return new Promise<void>((resolve) => {
      const done = () => {
        window.clearTimeout(timer);
        video.removeEventListener("seeked", done);
        resolve();
      };
      const timer = window.setTimeout(done, 3000);
      video.addEventListener("seeked", done);
      video.currentTime = t;
    });
  }

  private async play(): Promise<void> {
    const result = this.result;
    const reading = this.reading;
    if (!result || !reading || this.state.phase !== "done" || this.playing) return;
    const video = this.part<HTMLVideoElement>("video");
    try {
      await video.play();
    } catch {
      // Refused (a data-saving mode, a browser setting): stay paused.
      this.wanted = false;
      this.showPlaying(false);
      return;
    }
    if (this.result !== result || this.state.phase !== "done") return;
    this.playing = true;
    this.showPlaying(true);
    this.loop(result, reading);
  }

  private pause(): void {
    this.part<HTMLVideoElement>("video").pause();
    this.stopLoop?.();
    this.stopLoop = null;
    this.playing = false;
    this.showPlaying(false);
    if (this.frame >= 0) this.render(this.frame, true);
  }

  /** Follow the video as it plays: the frame on screen is the one whose step
   * holds the video's time; past the last frame read it starts over at the
   * first answer. */
  private loop(result: Result, reading: Reading): void {
    this.stopLoop?.();
    const video = this.part<HTMLVideoElement>("video");
    const { rate } = reading.pace;
    const total = result.times.length;
    const end = total / rate;
    let stopped = false;
    const follow = (time: number) => {
      if (time >= end) {
        this.wrap();
        return;
      }
      const k = Math.max(reading.pace.lag, frameIndexAt(time, rate, total));
      if (k !== this.frame) this.render(k, false);
    };
    const request = video.requestVideoFrameCallback?.bind(video);
    if (request) {
      let handle = 0;
      const presented = (_now: number, metadata: { mediaTime: number }) => {
        if (stopped) return;
        follow(metadata.mediaTime);
        handle = request(presented);
      };
      handle = request(presented);
      this.stopLoop = () => {
        stopped = true;
        video.cancelVideoFrameCallback?.(handle);
      };
      return;
    }
    let handle = 0;
    const tick = () => {
      if (stopped) return;
      follow(video.currentTime);
      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);
    this.stopLoop = () => {
      stopped = true;
      cancelAnimationFrame(handle);
    };
  }

  /** Back to the first answer, playing on. */
  private wrap(): void {
    const reading = this.reading;
    if (!reading) return;
    const video = this.part<HTMLVideoElement>("video");
    video.currentTime = seekTime(reading.pace.lag, reading.pace.rate);
    if (video.paused && this.playing) void video.play().catch(() => this.pause());
  }

  private seek(k: number): void {
    const result = this.result;
    const reading = this.reading;
    if (!result || !reading || this.state.phase !== "done") return;
    const target = Math.min(result.times.length - 1, Math.max(reading.pace.lag, Math.round(k)));
    this.part<HTMLVideoElement>("video").currentTime = seekTime(target, reading.pace.rate);
    this.render(target, true);
  }

  private step(by: number): void {
    if (this.state.phase !== "done") return;
    if (this.playing) {
      this.wanted = false;
      this.pause();
    }
    this.seek(this.frame + by);
  }

  private showPlaying(playing: boolean): void {
    const play = this.part("play");
    play.dataset["playing"] = String(playing);
    this.part("play-label").textContent = playing ? "Pause" : "Play";
  }

  /** Stop the player and let go of the file it plays. */
  private endPlayer(): void {
    this.stopLoop?.();
    this.stopLoop = null;
    this.playing = false;
    this.wanted = false;
    this.frame = -1;
    this.result = null;
    this.showPlaying(false);
    if (this.playerUrl !== null) {
      emptyVideo(this.part<HTMLVideoElement>("video"));
      this.urls.revoke(this.playerUrl);
      this.playerUrl = null;
    }
    this.trace.setFrame(-1);
  }

  // -- Drawing each state -------------------------------------------------------------------

  private apply(event: OwnClipEvent): void {
    const before = this.state;
    this.state = advance(before, event);
    if (this.state !== before) this.show(before);
  }

  /** Draw the state on screen; `before` is the state it replaced (null on
   * the first draw). */
  private show(before: OwnClipState | null): void {
    const state = this.state;
    const { phase } = state;
    const changed = before === null || before.phase !== phase;
    this.part("console").dataset["phase"] = phase;
    const working = phase === "loading" || phase === "sampling" || phase === "running";
    this.part("drop").hidden = phase !== "idle";
    this.part("player").hidden = !(working || phase === "done");
    this.part("error").hidden = phase !== "error";
    this.part("cancel").hidden = !working;
    this.part("again").hidden = phase !== "done";
    this.part<HTMLInputElement>("file").tabIndex = phase === "idle" ? 0 : -1;
    for (const name of ["play", "back", "forward", "scrub"]) this.part<HTMLButtonElement>(name).disabled = phase !== "done";

    const status = this.part("status");
    const note = this.part("note");
    const reading = this.reading;
    switch (state.phase) {
      case "idle":
        status.textContent = "";
        note.textContent = "";
        break;
      case "loading":
        status.textContent = OPENING;
        note.textContent = "";
        break;
      case "sampling":
        status.textContent = readingStatus(state.read, state.total);
        note.textContent = reading ? answeredNote(state.answered, state.answerable, reading) : "";
        break;
      case "running":
        status.textContent = finishingStatus(state.answered, state.answerable);
        note.textContent = reading ? answeredNote(state.answered, state.answerable, reading) : "";
        break;
      case "done":
        status.textContent = "";
        note.textContent = reading ? doneNote(state.name, state.total, state.answerable, state.capped, reading) : "";
        break;
      case "error": {
        const words = failureWords(state.failure, reading);
        this.part("error-title").textContent = words.title;
        this.part("error-text").textContent = words.text;
        status.textContent = "";
        note.textContent = "";
        break;
      }
    }
    status.hidden = status.textContent === "";
    if (!changed) return;

    // Said once per state, not on every frame read.
    const live = this.part("live");
    if (phase === "loading") live.textContent = `${OPENING}.`;
    else if (phase === "sampling") live.textContent = "Reading your video; the model answers as the frames come in.";
    else if (phase === "done") live.textContent = `Finished reading your video. ${note.textContent}`;
    else if (phase === "idle" && before !== null) live.textContent = "Stopped reading your video.";
    else live.textContent = "";
    if (phase === "error" && this.section.contains(document.activeElement)) this.part("retry").focus();
    if (phase === "done" && this.section.contains(document.activeElement)) this.part("play").focus();
  }
}
