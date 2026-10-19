/**
 * The cockpit: the model steering recorded footage, live, in this browser.
 *
 * The page loads results.json (which names every file), the hero clip and the
 * model, then the browser decodes the clip's sprite frames and runs the model
 * on them itself -- the wheel, the readouts and the heat on screen are what
 * this browser computed, frame by frame, not numbers shipped with the page.
 * When the decoded frames are checked to be the reference decoder's, the clip
 * file's own expected outputs draw the error strip the moment the clip is
 * ready; either way each column is overwritten as the model reaches it.
 *
 * Inference runs one batch at a time on the main thread (the runtime has one
 * thread), always starting from the frame on screen and working ahead of it,
 * so a playing clip finds its frames already computed. Every asynchronous
 * step checks it still belongs to the clip that started it (`generation`), so
 * switching clips mid-load never lets the old clip paint over the new one.
 *
 * A model reaches the screen only once its session exists. Choosing the twin
 * downloads it while Sightline keeps driving; if the twin cannot be had,
 * Sightline stays on screen and a note says so. Only Sightline failing stops
 * the demo, and even then the clip picker stays usable, so the other clip --
 * or the same one, picked again -- can be tried.
 */

import type { InferenceSession } from "onnxruntime-web/wasm";

import { AssetError, loadClip, loadResults } from "../data/load.ts";
import { MODEL_KEYS, type Clip, type ModelKey, type Results } from "../data/types.ts";
import { NO_VALUE, fmtDeg, steerParts } from "../format.ts";
import { INPUT_H, INPUT_W, createSession, infer } from "../model/runner.ts";
import { TWIN_UNAVAILABLE, decodeNote, eyeCaption, failureMessage, modelNote, splitTitle } from "./copy.ts";
import { cropBand, videoOverhang } from "./geometry.ts";
import { mountCockpitMarkup } from "./markup.ts";
import { seekTime, startFrameLoop, type FrameLoop } from "./player.ts";
import { heatImageFromBytes, quantizeSaliency } from "./saliency.ts";
import { checkDecoded, decodeSprites, stackAt, type Bytes, type DecodeCheck, type SpritePhase } from "./sprites.ts";
import { ErrorStrip, errorScale } from "./strip.ts";
import { renderWheel } from "./wheel.ts";

/** The clip the cockpit opens on: minutes of the training drive the shipped
 * model never trained on. */
const HERO_CLIP = "c2018";

/** The model the page ships. The cockpit starts on it, falls back to it, and
 * cannot run without it. */
const PRIMARY: ModelKey = "sightline";

/** Pixels in one saliency map. */
const MAP = INPUT_H * INPUT_W;

/** The heat's opacity where the model looks hardest, over the footage and over
 * its own input; where it does not look the heat is fainter (`HEAT_FLOOR`). */
const HEAT_ALPHA = 0.85;

/** How long one inference batch may hold the main thread, in milliseconds. */
const BATCH_BUDGET_MS = 24;
const MAX_BATCH = 8;

/** The frames a clip's model runs on before the clip is shown. */
const OPENING_FRAMES = [0, 1, 2, 3];

const MODEL_NAMES: Readonly<Record<ModelKey, string>> = { sightline: "Sightline", shuffled: "Its twin" };

const PHASE_STATUS: Readonly<Record<SpritePhase, string>> = {
  downloading: "Downloading the frames the model reads",
  decoding: "Decoding the frames the model reads",
};

type State = "loading" | "ready" | "unavailable";

/** The standing remarks under the strip, in the order they are read. */
const REMARKS = ["twin", "decode"] as const;
type Remark = (typeof REMARKS)[number];

/** One model's outputs on one clip, as the browser computes them. */
interface Track {
  readonly angles: Float32Array;
  readonly saliency: Uint8Array;
  readonly done: Uint8Array;
}

/** The clip on screen: its file, its decoded frames, the models' outputs on
 * them, and whether its published outputs may be shown before the model's. */
interface Loaded {
  readonly clip: Clip;
  readonly frames: Bytes[];
  readonly tracks: Partial<Record<ModelKey, Track>>;
  readonly scale: number;
  readonly check: DecodeCheck;
}

function newTrack(n: number): Track {
  return {
    angles: new Float32Array(n).fill(Number.NaN),
    saliency: new Uint8Array(n * MAP),
    done: new Uint8Array(n),
  };
}

/** The elements the cockpit drives, found once in its markup. */
class View {
  readonly console: HTMLElement;
  readonly clips: HTMLFieldSetElement;
  readonly models: HTMLFieldSetElement;
  readonly windshield: HTMLElement;
  readonly video: HTMLVideoElement;
  readonly heat: HTMLCanvasElement;
  readonly status: HTMLElement;
  readonly notice: HTMLElement;
  readonly play: HTMLButtonElement;
  readonly playLabel: HTMLElement;
  readonly back: HTMLButtonElement;
  readonly forward: HTMLButtonElement;
  readonly strip: HTMLElement;
  readonly stripCanvas: HTMLCanvasElement;
  readonly average: HTMLElement;
  readonly note: HTMLElement;
  readonly wheelBox: HTMLElement;
  readonly wheel: SVGSVGElement;
  readonly modelName: HTMLElement;
  readonly modelAngle: HTMLElement;
  readonly modelSide: HTMLElement;
  readonly driverAngle: HTMLElement;
  readonly driverSide: HTMLElement;
  readonly gap: HTMLElement;
  readonly eye: HTMLCanvasElement;
  readonly eyeCaption: HTMLElement;

  constructor(root: HTMLElement) {
    const find = <T extends Element>(name: string): T => {
      const element = root.querySelector<T>(`[data-cockpit="${name}"]`);
      if (!element) throw new Error(`the cockpit markup has no ${name}`);
      return element;
    };
    const screen = root.querySelector<HTMLElement>(".cockpit__console");
    if (!screen) throw new Error("the cockpit markup has no console");
    this.console = screen;
    this.clips = find("clips");
    this.models = find("models");
    this.windshield = find("windshield");
    this.video = find("video");
    this.heat = find("heat");
    this.status = find("status");
    this.notice = find("notice");
    this.play = find("play");
    this.playLabel = find("play-label");
    this.back = find("back");
    this.forward = find("forward");
    this.strip = find("strip");
    this.stripCanvas = find("strip-canvas");
    this.average = find("average");
    this.note = find("note");
    this.wheelBox = find("wheel");
    const wheel = this.wheelBox.querySelector("svg");
    if (!wheel) throw new Error("the cockpit markup has no wheel drawing");
    this.wheel = wheel;
    this.modelName = find("model-name");
    this.modelAngle = find("model-angle");
    this.modelSide = find("model-side");
    this.driverAngle = find("driver-angle");
    this.driverSide = find("driver-side");
    this.gap = find("gap");
    this.eye = find("eye");
    this.eyeCaption = find("eye-caption");
  }

  /** "loading", "ready" or "unavailable": the console's look follows it. */
  setState(state: State): void {
    this.console.dataset["state"] = state;
  }

  say(text: string): void {
    this.status.textContent = text;
    this.status.hidden = text === "";
  }

  /** The standing remarks about the clip and the model; empty clears them. */
  remark(text: string): void {
    this.notice.textContent = text;
    this.notice.hidden = text === "";
  }

  /** Play, pause and the frame steps. */
  enableTransport(on: boolean): void {
    for (const control of [this.play, this.back, this.forward]) control.disabled = !on;
  }

  /** The frame on screen has no output from the model yet: its wheel dims. */
  waiting(on: boolean): void {
    this.wheelBox.dataset["waiting"] = String(on);
  }

  /** Show a steering angle as its size and its side, `12.6°` `right`; NaN
   * shows as no value. */
  steer(size: HTMLElement, side: HTMLElement, angle: number): void {
    const parts = steerParts(angle);
    size.textContent = parts.size;
    side.textContent = parts.side;
  }

  showPlaying(playing: boolean): void {
    this.play.dataset["playing"] = String(playing);
    this.playLabel.textContent = playing ? "Pause" : "Play";
  }

  /** Whether any clip can be picked. */
  hasClipChoices(): boolean {
    return [...this.clips.querySelectorAll<HTMLInputElement>('input[name="cockpit-clip"]')].some((input) => !input.disabled);
  }

  checkModel(model: ModelKey): void {
    const input = this.models.querySelector<HTMLInputElement>(`input[value="${model}"]`);
    if (input) input.checked = true;
  }
}

export class Cockpit {
  private readonly view: View;
  private readonly strip: ErrorStrip;
  private results: Results | null = null;
  private readonly clipFiles = new Map<string, Promise<Clip>>();
  /** Each model's session while it downloads; a failed one is dropped, so the
   * next request tries again. */
  private readonly sessions: Partial<Record<ModelKey, Promise<InferenceSession>>> = {};
  /** The sessions that exist. Only these models are ever run. */
  private readonly ready = new Map<ModelKey, InferenceSession>();
  private state: State = "loading";
  /** The clip on screen, loading, or failed. */
  private clipId: string | null = null;
  private loaded: Loaded | null = null;
  /** The model on screen: its session exists before any frame is run on it. */
  private model: ModelKey = PRIMARY;
  /** The model the visitor picked; it differs from `model` while the picked
   * one downloads. */
  private chosen: ModelKey = PRIMARY;
  private check: DecodeCheck["status"] | null = null;
  private readonly remarks = new Map<Remark, string>();
  private frame = 0;
  private playing = false;
  /** The clip should be playing (the visitor's choice, or the opening play). */
  private wanted = false;
  /** The cockpit is scrolled out of view, or the tab is in the background:
   * either pauses the clip and the model, and both must clear to resume. */
  private offscreen = false;
  private tabHidden = false;
  private hidden = false;
  private loop: FrameLoop | null = null;
  private generation = 0;
  private queue: Promise<unknown> = Promise.resolve();
  private filling = false;
  /** A frame on screen still waiting for the model's output, drawn again when
   * the fill reaches it. */
  private awaiting: number | null = null;
  private msPerFrame = 12;
  private stripStale = false;
  /** Stops listening for the video of a clip another clip has replaced. */
  private dropVideo: (() => void) | null = null;
  private readonly heatContext: CanvasRenderingContext2D | null;
  private readonly eyeContext: CanvasRenderingContext2D | null;
  private readonly eyeImage: ImageData | null;

  constructor(root: HTMLElement) {
    this.view = new View(root);
    this.heatContext = this.view.heat.getContext("2d");
    this.eyeContext = this.view.eye.getContext("2d");
    this.eyeImage = this.eyeContext?.createImageData(INPUT_W, INPUT_H) ?? null;
    this.strip = new ErrorStrip(this.view.strip, this.view.stripCanvas, {
      onSeek: (k) => this.seek(k),
      describe: (k) => this.describe(k),
    });
    this.wire(root);
  }

  /** Whether this browser decoded the reference decoder's frames for the clip
   * on screen ("match", "differ" or "unverified"); null until a clip is
   * ready. Only "match" lets the page claim the clip's published outputs. */
  get frameCheck(): DecodeCheck["status"] | null {
    return this.check;
  }

  // -- Startup ------------------------------------------------------------------

  async start(): Promise<void> {
    try {
      this.enterLoading("Loading the model and the footage");
      const site = new URL("./", document.baseURI).href;
      this.results = await loadResults(site);
      void this.session(PRIMARY).catch(() => undefined); // start the download now; a failure surfaces when the clip needs it
      const ids = Object.keys(this.results.assets.clips);
      const ordered = [...ids.filter((id) => id === HERO_CLIP), ...ids.filter((id) => id !== HERO_CLIP)];
      const first = ordered[0];
      if (first === undefined) {
        throw new AssetError("results.json lists no clips", "invalid", `${site}data/results.json`);
      }
      this.buildClipPicker(ordered);
      this.wanted = !matchMedia("(prefers-reduced-motion: reduce)").matches;
      await this.showClip(first, this.wanted);
    } catch (error) {
      this.unavailable(error);
    }
  }

  private clipFile(id: string): Promise<Clip> {
    let file = this.clipFiles.get(id);
    if (!file) {
      const path = this.results?.assets.clips[id];
      file = path === undefined
        ? Promise.reject(new Error(`results.json names no clip ${id}`))
        : loadClip(new URL(path, document.baseURI).href);
      this.clipFiles.set(id, file);
    }
    return file;
  }

  /** The model's session, downloading it the first time it is asked for. Once
   * the promise settles the session is in `ready`. */
  private session(model: ModelKey): Promise<InferenceSession> {
    let session = this.sessions[model];
    if (!session) {
      const path = this.results?.assets.models[model];
      const created = path === undefined
        ? Promise.reject(new Error(`results.json names no ${model} model`))
        : createSession(new URL(path, document.baseURI).href);
      const settled = created.then((made) => {
        this.ready.set(model, made);
        return made;
      });
      this.sessions[model] = settled;
      // A failed download may be tried again on the next request.
      settled.catch(() => {
        if (this.sessions[model] === settled) delete this.sessions[model];
      });
      session = settled;
    }
    return session;
  }

  /** One radio button per clip, labelled from the clip's own title as each
   * clip file arrives. */
  private buildClipPicker(ids: readonly string[]): void {
    const legend = this.view.clips.querySelector("legend");
    this.view.clips.replaceChildren(...(legend ? [legend] : []));
    for (const id of ids) {
      const label = document.createElement("label");
      label.className = "picker__option";
      const input = document.createElement("input");
      input.className = "picker__input";
      input.type = "radio";
      input.name = "cockpit-clip";
      input.value = id;
      input.checked = id === ids[0];
      const lead = document.createElement("span");
      lead.className = "picker__lead";
      const sub = document.createElement("span");
      sub.className = "picker__sub";
      label.append(input, lead, sub);
      this.view.clips.append(label);
      this.clipFile(id).then(
        (clip) => {
          const [year, words] = splitTitle(clip.title);
          lead.textContent = year;
          sub.textContent = words;
        },
        () => {
          sub.textContent = "not available";
          input.disabled = true;
        },
      );
    }
    this.view.clips.disabled = !this.view.hasClipChoices();
  }

  // -- Loading a clip ------------------------------------------------------------

  private async showClip(id: string, play: boolean): Promise<void> {
    const generation = ++this.generation;
    const current = () => generation === this.generation;
    this.clipId = id;
    this.enterLoading("Loading the footage");
    try {
      const clip = await this.clipFile(id);
      if (!current()) return;
      const base = new URL("./", new URL(this.results?.assets.clips[id] ?? "", document.baseURI)).href;
      if (clip.sprite.frame_w !== INPUT_W || clip.sprite.frame_h !== INPUT_H) {
        throw new AssetError(`the clip's frames are ${clip.sprite.frame_w} x ${clip.sprite.frame_h}, not the model's ${INPUT_W} x ${INPUT_H}`, "invalid", base);
      }
      if (clip.video === null) {
        throw new AssetError("this clip has no video", "missing", base);
      }
      this.layOut(clip);
      const video = this.loadVideo(base + clip.video, clip.video);
      const frames = await decodeSprites(clip, base, (phase) => {
        if (current()) this.view.say(PHASE_STATUS[phase]);
      });
      if (!current()) return;
      // Before the clip's published numbers go on screen: are these the frames
      // they were computed on?
      const check = await checkDecoded(frames, clip.sprite.decoded_sha256);
      if (!current()) return;
      const loaded: Loaded = { clip, frames, tracks: {}, scale: errorScale(clip.labels_deg, clip.expected), check };
      this.loaded = loaded;
      this.view.say("Starting the model");
      if (!(await this.startModel(loaded, current))) return;
      await video;
      if (!current()) return;
      await this.seekVideo(0, clip.fps);
      if (!current()) return;
      this.enterReady(loaded);
      if (play) await this.play();
      void this.fill();
      this.warmTwin();
    } catch (error) {
      if (current()) this.unavailable(error);
    }
  }

  /** Run the model on screen on the clip's opening frames. A twin that cannot
   * be had gives way to Sightline; Sightline failing fails the clip. False if
   * another clip took over meanwhile. */
  private async startModel(loaded: Loaded, current: () => boolean): Promise<boolean> {
    for (;;) {
      const model = this.model;
      try {
        await this.session(model);
        if (!current()) return false;
        await this.compute(OPENING_FRAMES, model, loaded);
        return current();
      } catch (error) {
        if (!current()) return false;
        if (model === PRIMARY) throw error;
        this.fallBack(model, error);
      }
    }
  }

  /** Size the windshield and the band from the clip's own geometry. */
  private layOut(clip: Clip): void {
    const [w, h] = clip.camera_size;
    const band = cropBand(h);
    const style = this.view.windshield.style;
    style.setProperty("--camera-aspect", `${w} / ${h}`);
    style.setProperty("--video-overhang", String(videoOverhang(clip.camera_size, clip.video_size)));
    style.setProperty("--band-top", String(band.top));
    this.view.eyeCaption.textContent = eyeCaption(clip);
  }

  private loadVideo(src: string, name: string): Promise<void> {
    this.dropVideo?.();
    const video = this.view.video;
    const loading = new Promise<void>((resolve, reject) => {
      const stop = () => {
        video.removeEventListener("loadeddata", ready);
        video.removeEventListener("error", failed);
      };
      const ready = () => {
        stop();
        resolve();
      };
      const failed = () => {
        stop();
        reject(new AssetError(`this browser could not play ${name}`, "undecodable", src));
      };
      // When another clip takes the element, whoever waits on this load
      // belongs to the old clip and stops at its own generation check.
      this.dropVideo = () => {
        stop();
        resolve();
      };
      video.addEventListener("loadeddata", ready);
      video.addEventListener("error", failed);
      video.muted = true;
      video.preload = "auto";
      video.src = src;
      video.load();
    });
    // The clip waits for its video only once its frames are decoded and its
    // model has started. A failure before then -- or after another clip has
    // taken over -- is handled there or not at all, never left unhandled.
    loading.catch(() => undefined);
    return loading;
  }

  /** Seek the video to frame `k` and wait for it, at most a couple of seconds:
   * a server that cannot serve byte ranges cannot seek a video, and the demo
   * should start regardless. */
  private seekVideo(k: number, fps: number): Promise<void> {
    const video = this.view.video;
    return new Promise<void>((resolve) => {
      const done = () => {
        clearTimeout(timer);
        video.removeEventListener("seeked", done);
        resolve();
      };
      const timer = setTimeout(done, 2000);
      video.addEventListener("seeked", done);
      video.currentTime = seekTime(k, fps);
    });
  }

  // -- The cockpit's states ---------------------------------------------------------

  /** A clip is on its way: nothing of the last clip stays on screen, and only
   * the clip picker takes input. */
  private enterLoading(message: string): void {
    this.leaveRunning("loading");
    this.view.clips.disabled = !this.view.hasClipChoices();
    this.view.note.textContent = modelNote(this.model, null);
    this.remark("decode", "");
    this.view.say(message);
  }

  private enterReady(loaded: Loaded): void {
    this.state = "ready";
    this.check = loaded.check.status;
    this.view.setState("ready");
    this.view.say(this.chosen !== this.model ? "Loading the twin" : "");
    this.view.enableTransport(true);
    this.strip.setEnabled(true);
    this.view.clips.disabled = false;
    this.view.models.disabled = false;
    this.view.note.textContent = modelNote(this.model, loaded.clip);
    this.remark("decode", decodeNote(loaded.check, loaded.frames.length));
    this.strip.setClip(loaded.clip.labels_deg, loaded.clip.fps, loaded.scale);
    this.refreshStrip();
    this.render(0, true);
  }

  /** The demo cannot run this clip. The picker stays usable: the failure may
   * be this clip's alone, and choosing a clip -- another, or this one again --
   * starts over. */
  private unavailable(error: unknown): void {
    this.leaveRunning("unavailable");
    this.view.clips.disabled = !this.view.hasClipChoices();
    this.remarks.clear();
    this.view.remark("");
    this.view.say(failureMessage(error));
    // A copy of the site built without its data is expected to say so; anything
    // else is a fault worth its detail in the console.
    if (!(error instanceof AssetError && error.kind === "missing")) {
      console.warn("The demo could not start:", error);
    }
  }

  /** Stop the clip, drop it, disable every control that acts on it, and clear
   * every reading of it from the screen. */
  private leaveRunning(state: Exclude<State, "ready">): void {
    this.stopLoop();
    this.view.video.pause();
    this.playing = false;
    this.state = state;
    this.loaded = null;
    this.awaiting = null;
    this.check = null;
    this.view.setState(state);
    this.view.showPlaying(false);
    this.view.enableTransport(false);
    this.strip.setEnabled(false);
    this.view.models.disabled = true;
    this.clearReadings();
  }

  private clearReadings(): void {
    const { view } = this;
    view.steer(view.modelAngle, view.modelSide, Number.NaN);
    view.steer(view.driverAngle, view.driverSide, Number.NaN);
    view.gap.textContent = NO_VALUE;
    view.average.textContent = NO_VALUE;
    view.waiting(false);
    renderWheel(view.wheel, 0, 0);
    this.heatContext?.clearRect(0, 0, INPUT_W, INPUT_H);
    this.eyeContext?.clearRect(0, 0, INPUT_W, INPUT_H);
    this.strip.clear();
  }

  private remark(kind: Remark, text: string): void {
    if (text === "") this.remarks.delete(kind);
    else this.remarks.set(kind, text);
    this.view.remark(REMARKS.flatMap((key) => this.remarks.get(key) ?? []).join(" "));
  }

  // -- Running the model -----------------------------------------------------------

  /** Run `task` after every task queued before it: one inference at a time. */
  private exclusive<T>(task: () => Promise<T>): Promise<T> {
    const run = this.queue.then(task, task);
    this.queue = run.catch(() => undefined);
    return run;
  }

  private track(loaded: Loaded, model: ModelKey): Track {
    let track = loaded.tracks[model];
    if (!track) {
      track = newTrack(loaded.clip.n);
      loaded.tracks[model] = track;
    }
    return track;
  }

  /** Compute the model's outputs on the given frames of the loaded clip,
   * skipping any already done. Only a model whose session exists is run. */
  private compute(frames: readonly number[], model: ModelKey, loaded: Loaded): Promise<void> {
    return this.exclusive(async () => {
      const track = this.track(loaded, model);
      const todo = frames.filter((k) => k >= 0 && k < loaded.clip.n && !track.done[k]);
      if (todo.length === 0) return;
      const session = this.ready.get(model);
      if (!session) throw new Error(`the ${model} model is not loaded`);
      const { clip } = loaded;
      const channels = 3 * (1 + clip.frame_lags.length);
      const stride = MAP * channels;
      const batch = new Uint8Array(todo.length * stride);
      todo.forEach((k, i) => batch.set(stackAt(loaded.frames, clip.lead_in + k, clip.frame_lags), i * stride));
      const started = performance.now();
      const out = await infer(session, batch, todo.length, channels);
      this.msPerFrame = 0.7 * this.msPerFrame + (0.3 * (performance.now() - started)) / todo.length;
      todo.forEach((k, i) => {
        track.angles[k] = out.angles[i] as number;
        track.saliency.set(quantizeSaliency(out.saliency.subarray(i * MAP, (i + 1) * MAP)), k * MAP);
        track.done[k] = 1;
      });
      this.stripStale = true;
    });
  }

  /** The next frames without output, from the frame on screen onwards. */
  private nextMissing(track: Track, from: number, count: number): number[] {
    const n = track.done.length;
    const out: number[] = [];
    for (let step = 0; step < n && out.length < count; step += 1) {
      const k = (from + step) % n;
      if (!track.done[k]) out.push(k);
    }
    return out;
  }

  /** Work through the clip on screen, ahead of the frame on screen, until
   * every frame has the output of the model on screen, yielding to the page
   * between batches. One fill runs at a time, and it follows whatever clip and
   * model are on screen as they change. */
  private async fill(): Promise<void> {
    if (this.filling) return;
    this.filling = true;
    try {
      for (;;) {
        const loaded = this.loaded;
        const model = this.model;
        if (this.state !== "ready" || this.hidden || !loaded || !this.ready.has(model)) break;
        const size = Math.max(1, Math.min(MAX_BATCH, Math.floor(BATCH_BUDGET_MS / this.msPerFrame)));
        const next = this.nextMissing(this.track(loaded, model), this.frame, size);
        if (next.length === 0) break;
        try {
          await this.compute(next, model, loaded);
        } catch (error) {
          // The clip or the model changed meanwhile: carry on with what is on screen now.
          if (this.loaded !== loaded || this.model !== model) continue;
          if (model === PRIMARY) {
            this.unavailable(error);
            break;
          }
          this.fallBack(model, error);
          continue;
        }
        if (this.loaded === loaded) {
          if (this.stripStale) this.refreshStrip();
          const waiting = this.awaiting;
          if (waiting !== null && this.track(loaded, this.model).done[waiting]) {
            this.awaiting = null;
            if (this.frame === waiting) this.render(waiting, !this.playing);
          }
        }
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    } finally {
      this.filling = false;
    }
  }

  /** Download the twin once a clip is running, so switching to it is
   * immediate. A failure here is silent: it matters only if the visitor picks
   * the twin, and that tries again. */
  private warmTwin(): void {
    for (const model of MODEL_KEYS) {
      if (model === PRIMARY || this.ready.has(model) || this.sessions[model]) continue;
      const idle = (globalThis as { requestIdleCallback?: (cb: () => void) => void }).requestIdleCallback;
      const later = (cb: () => void) => (idle ? idle(cb) : setTimeout(cb, 2000));
      later(() => void this.session(model).catch(() => undefined));
    }
  }

  // -- Choosing the model ------------------------------------------------------------

  private selectModel(model: ModelKey): void {
    this.chosen = model;
    this.remark("twin", "");
    if (model === this.model) {
      // Back to the model on screen while another was downloading.
      if (this.state === "ready") this.view.say("");
      return;
    }
    if (this.ready.has(model)) {
      this.switchTo(model);
      return;
    }
    // Until it arrives the model on screen keeps driving.
    if (this.state === "ready") this.view.say("Loading the twin");
    this.session(model).then(
      () => {
        if (this.chosen !== model) return;
        if (this.state === "ready") this.view.say("");
        this.switchTo(model);
      },
      (error: unknown) => {
        if (this.chosen === model) this.fallBack(model, error);
      },
    );
  }

  /** Put `model`, whose session exists, on screen. */
  private switchTo(model: ModelKey): void {
    this.model = model;
    this.view.modelName.textContent = MODEL_NAMES[model];
    const loaded = this.state === "ready" ? this.loaded : null;
    this.view.note.textContent = modelNote(model, loaded?.clip ?? null);
    if (!loaded) return;
    this.refreshStrip();
    this.render(this.frame, true);
    void this.fill();
  }

  /** `model` (the twin) could not be downloaded or run: Sightline takes over
   * and the visitor is told, once. Picking the twin again starts afresh. */
  private fallBack(model: ModelKey, error: unknown): void {
    console.warn("The twin could not be used:", error);
    this.ready.delete(model);
    delete this.sessions[model];
    this.chosen = PRIMARY;
    this.view.checkModel(PRIMARY);
    if (this.state === "ready") this.view.say("");
    this.remark("twin", TWIN_UNAVAILABLE);
    this.switchTo(PRIMARY);
  }

  // -- Drawing ---------------------------------------------------------------------

  private refreshStrip(): void {
    const loaded = this.loaded;
    if (!loaded) return;
    const { clip } = loaded;
    const track = loaded.tracks[this.model];
    // The published outputs stand in for frames not yet computed only when
    // this browser's frames are the ones they were computed on.
    const published = loaded.check.status === "match";
    const values = Float32Array.from(clip.expected[this.model], (value, k) =>
      track?.done[k] ? (track.angles[k] as number) : published ? value : Number.NaN,
    );
    this.strip.setValues(values, clip.clip_mae_deg[this.model]);
    this.view.average.textContent = fmtDeg(clip.clip_mae_deg[this.model]);
    this.stripStale = false;
  }

  /** Show frame `k`: the wheel, the readouts, the heat and the model's input.
   * A frame the model has not reached yet shows the footage and the driver,
   * and nothing of the model's until its output arrives. */
  private render(k: number, announce: boolean): void {
    const loaded = this.loaded;
    if (!loaded || this.state !== "ready") return;
    this.frame = k;
    const { clip } = loaded;
    const human = clip.labels_deg[k] as number;
    const track = this.track(loaded, this.model);
    const done = track.done[k] === 1;
    const angle = done ? (track.angles[k] as number) : Number.NaN;
    this.view.steer(this.view.driverAngle, this.view.driverSide, human);
    this.view.steer(this.view.modelAngle, this.view.modelSide, angle);
    this.view.gap.textContent = done ? fmtDeg(Math.abs(angle - human)) : NO_VALUE;
    this.view.waiting(!done);
    renderWheel(this.view.wheel, done ? angle : null, human);
    this.paint(k, loaded, done ? track : null);
    if (!done) {
      this.awaiting = k;
      void this.fill();
    }
    if (this.stripStale) this.refreshStrip();
    this.strip.setFrame(k, announce);
  }

  /** The heat over the footage and the model's own input for frame `k`; with
   * no output (`track` null) the heat is cleared and the input shown bare. */
  private paint(k: number, loaded: Loaded, track: Track | null): void {
    const heat = this.heatContext;
    if (heat) {
      if (track) {
        heat.putImageData(heatImageFromBytes(track.saliency.subarray(k * MAP, (k + 1) * MAP), INPUT_W, INPUT_H, HEAT_ALPHA), 0, 0);
      } else {
        heat.clearRect(0, 0, INPUT_W, INPUT_H);
      }
    }
    const eye = this.eyeContext;
    const image = this.eyeImage;
    const frame = loaded.frames[loaded.clip.lead_in + k];
    if (eye && image && frame) {
      const data = image.data;
      for (let p = 0, from = 0, to = 0; p < MAP; p += 1, from += 3, to += 4) {
        data[to] = frame[from] as number;
        data[to + 1] = frame[from + 1] as number;
        data[to + 2] = frame[from + 2] as number;
        data[to + 3] = 255;
      }
      eye.putImageData(image, 0, 0);
      if (track) eye.drawImage(this.view.heat, 0, 0);
    }
  }

  /** The strip's spoken value at frame `k`. */
  private describe(k: number): string {
    const loaded = this.loaded;
    if (!loaded) return "";
    const { clip } = loaded;
    const seconds = (k / clip.fps).toFixed(1);
    const track = loaded.tracks[this.model];
    const angle = track?.done[k] ? (track.angles[k] as number) : Number.NaN;
    const human = clip.labels_deg[k] as number;
    const off = Number.isFinite(angle) ? `, off by ${fmtDeg(Math.abs(angle - human))}` : "";
    return `Frame ${k + 1} of ${clip.n}, ${seconds} seconds in${off}`;
  }

  // -- Playback ----------------------------------------------------------------------

  private stopLoop(): void {
    this.loop?.stop();
    this.loop = null;
  }

  private async play(): Promise<void> {
    const loaded = this.loaded;
    if (!loaded || this.state !== "ready") return;
    this.wanted = true;
    if (this.hidden) return;
    try {
      await this.view.video.play();
    } catch {
      // Another clip took the video meanwhile: its own start decides.
      if (this.loaded !== loaded) return;
      // Autoplay refused (a saved data mode, a browser setting): stay paused.
      this.wanted = false;
      this.view.showPlaying(false);
      return;
    }
    if (this.loaded !== loaded || this.state !== "ready") return;
    this.playing = true;
    this.view.showPlaying(true);
    this.stopLoop();
    this.loop = startFrameLoop(this.view.video, loaded.clip.fps, loaded.clip.n, (k) => this.render(k, false));
    void this.fill();
  }

  private pause(): void {
    this.view.video.pause();
    this.stopLoop();
    this.playing = false;
    this.view.showPlaying(false);
    this.render(this.frame, true);
  }

  private seek(k: number): void {
    const loaded = this.loaded;
    if (!loaded || this.state !== "ready") return;
    this.view.video.currentTime = seekTime(k, loaded.clip.fps);
    this.render(k, true);
    void this.fill();
  }

  private step(by: number): void {
    const loaded = this.loaded;
    if (!loaded || this.state !== "ready") return;
    if (this.playing) {
      this.wanted = false;
      this.pause();
    }
    this.seek(Math.min(loaded.clip.n - 1, Math.max(0, this.frame + by)));
  }

  /** Pause while the cockpit is out of sight, and pick up again on return. */
  private setHidden(): void {
    const hidden = this.offscreen || this.tabHidden;
    if (hidden === this.hidden) return;
    this.hidden = hidden;
    if (hidden) {
      if (this.playing) {
        this.view.video.pause();
        this.stopLoop();
        this.playing = false;
      }
    } else {
      if (this.wanted && this.state === "ready") void this.play();
      void this.fill();
    }
  }

  // -- Controls ----------------------------------------------------------------------

  private wire(root: HTMLElement): void {
    const { view } = this;
    view.play.addEventListener("click", () => {
      if (this.state !== "ready") return;
      if (this.playing) {
        this.wanted = false;
        this.pause();
      } else {
        void this.play();
      }
    });
    view.back.addEventListener("click", () => this.step(-1));
    view.forward.addEventListener("click", () => this.step(1));
    view.clips.addEventListener("change", (event) => {
      const input = event.target as HTMLInputElement;
      if (input.name === "cockpit-clip") void this.showClip(input.value, this.wanted);
    });
    // A radio button already checked sends no change: after a failure,
    // choosing the same clip again is a click, and it tries the clip again.
    view.clips.addEventListener("click", (event) => {
      const input = event.target as HTMLInputElement;
      if (input.name === "cockpit-clip" && this.state === "unavailable" && input.value === this.clipId) {
        void this.showClip(input.value, this.wanted);
      }
    });
    view.models.addEventListener("change", (event) => {
      const input = event.target as HTMLInputElement;
      if (MODEL_KEYS.includes(input.value as ModelKey)) this.selectModel(input.value as ModelKey);
    });
    view.note.textContent = modelNote(this.model, null);

    new IntersectionObserver(
      (entries) => {
        for (const entry of entries) this.offscreen = !entry.isIntersecting;
        this.setHidden();
      },
      { threshold: 0.2 },
    ).observe(root);
    document.addEventListener("visibilitychange", () => {
      this.tabHidden = document.visibilityState === "hidden";
      this.setHidden();
    });
  }
}

/** Mount the cockpit into the page and start it. Building it can throw (a
 * browser without what it needs); starting it reports its own failures on
 * screen. */
export function startCockpit(): Cockpit | null {
  const root = mountCockpitMarkup();
  if (!root) return null;
  const cockpit = new Cockpit(root);
  void cockpit.start();
  return cockpit;
}
