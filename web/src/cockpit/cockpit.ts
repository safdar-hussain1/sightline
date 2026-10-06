/**
 * The cockpit: the model steering recorded footage, live, in this browser.
 *
 * The page loads results.json (which names every file), the hero clip and the
 * model, then the browser decodes the clip's sprite frames and runs the model
 * on them itself -- the wheel, the readouts and the heat on screen are what
 * this browser computed, frame by frame, not numbers shipped with the page.
 * The clip file's own expected outputs only draw the error strip the moment
 * the clip loads; each column is overwritten as the model reaches it.
 *
 * Inference runs one batch at a time on the main thread (the runtime has one
 * thread), always starting from the frame on screen and working ahead of it,
 * so a playing clip finds its frames already computed. Every asynchronous
 * step checks it still belongs to the clip that started it (`generation`), so
 * switching clips mid-load never lets the old clip paint over the new one.
 */

import type { InferenceSession } from "onnxruntime-web/wasm";

import { AssetError, loadClip, loadResults } from "../data/load.ts";
import { MODEL_KEYS, type Clip, type ModelKey, type Results } from "../data/types.ts";
import { NO_VALUE, fmtDeg, steerParts } from "../format.ts";
import { INPUT_H, INPUT_W, createSession, infer } from "../model/runner.ts";
import { eyeCaption, failureMessage, modelNote, splitTitle } from "./copy.ts";
import { cropBand, videoOverhang } from "./geometry.ts";
import { mountCockpitMarkup } from "./markup.ts";
import { seekTime, startFrameLoop, type FrameLoop } from "./player.ts";
import { heatImageFromBytes, quantizeSaliency } from "./saliency.ts";
import { decodeSprites, mismatchedFrames, stackAt, type Bytes } from "./sprites.ts";
import { ErrorStrip, errorScale } from "./strip.ts";
import { renderWheel } from "./wheel.ts";

/** The clip the cockpit opens on: minutes of the training drive the shipped
 * model never trained on. */
const HERO_CLIP = "c2018";

/** Pixels in one saliency map. */
const MAP = INPUT_H * INPUT_W;

/** The heat's opacity where the model looks hardest, over the footage and over
 * its own input; where it does not look the heat is fainter (`HEAT_FLOOR`). */
const HEAT_ALPHA = 0.85;

/** How long one inference batch may hold the main thread, in milliseconds. */
const BATCH_BUDGET_MS = 24;
const MAX_BATCH = 8;

const MODEL_NAMES: Readonly<Record<ModelKey, string>> = { sightline: "Sightline", shuffled: "Its twin" };

/** One model's outputs on one clip, as the browser computes them. */
interface Track {
  readonly angles: Float32Array;
  readonly saliency: Uint8Array;
  readonly done: Uint8Array;
}

/** The clip on screen: its file, its decoded frames and the models' outputs
 * on them. */
interface Loaded {
  readonly clip: Clip;
  readonly frames: Bytes[];
  readonly tracks: Partial<Record<ModelKey, Track>>;
  readonly scale: number;
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
    this.console = root.querySelector<HTMLElement>(".cockpit__console") as HTMLElement;
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
    this.wheel = find<HTMLElement>("wheel").querySelector("svg") as SVGSVGElement;
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
  setState(state: "loading" | "ready" | "unavailable"): void {
    this.console.dataset["state"] = state;
  }

  say(text: string): void {
    this.status.textContent = text;
    this.status.hidden = text === "";
  }

  /** A standing remark about the clip on screen; empty clears it. */
  remark(text: string): void {
    this.notice.textContent = text;
    this.notice.hidden = text === "";
  }

  enable(on: boolean): void {
    for (const control of [this.play, this.back, this.forward]) control.disabled = !on;
    this.clips.disabled = !on;
    this.models.disabled = !on;
    this.strip.tabIndex = on ? 0 : -1;
    this.strip.setAttribute("aria-disabled", String(!on));
  }

  /** Show a steering angle as its size and its side, `12.6°` `right`. */
  steer(size: HTMLElement, side: HTMLElement, angle: number): void {
    const parts = steerParts(angle);
    size.textContent = parts.size;
    side.textContent = parts.side;
  }

  showPlaying(playing: boolean): void {
    this.play.dataset["playing"] = String(playing);
    this.playLabel.textContent = playing ? "Pause" : "Play";
  }
}

export class Cockpit {
  private readonly view: View;
  private readonly strip: ErrorStrip;
  private results: Results | null = null;
  private readonly clipFiles = new Map<string, Promise<Clip>>();
  private readonly sessions: Partial<Record<ModelKey, Promise<InferenceSession>>> = {};
  private loaded: Loaded | null = null;
  private model: ModelKey = "sightline";
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

  // -- Startup ------------------------------------------------------------------

  async start(): Promise<void> {
    this.view.setState("loading");
    this.view.say("Loading the model and the footage");
    const site = new URL("./", document.baseURI).href;
    try {
      this.results = await loadResults(site);
    } catch (error) {
      this.unavailable(error);
      return;
    }
    const results = this.results;
    void this.session("sightline").catch(() => undefined); // start the download now; errors surface below
    const ids = Object.keys(results.assets.clips);
    const ordered = [...ids.filter((id) => id === HERO_CLIP), ...ids.filter((id) => id !== HERO_CLIP)];
    this.buildClipPicker(ordered);
    const first = ordered[0];
    if (first === undefined) {
      this.unavailable(new Error("results.json lists no clips"));
      return;
    }
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    await this.showClip(first, !reduced);
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

  private session(model: ModelKey): Promise<InferenceSession> {
    let session = this.sessions[model];
    if (!session) {
      const path = this.results?.assets.models[model];
      session = path === undefined
        ? Promise.reject(new Error(`results.json names no ${model} model`))
        : createSession(new URL(path, document.baseURI).href);
      this.sessions[model] = session;
      // A failed download may be tried again on the next request.
      session.catch(() => {
        if (this.sessions[model] === session) delete this.sessions[model];
      });
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
  }

  // -- Loading a clip ------------------------------------------------------------

  private async showClip(id: string, play: boolean): Promise<void> {
    const generation = ++this.generation;
    const current = () => generation === this.generation;
    this.stopLoop();
    this.view.video.pause();
    this.playing = false;
    this.view.showPlaying(false);
    this.view.enable(false);
    this.view.setState("loading");
    this.view.say("Loading the footage");
    this.view.remark("");
    this.loaded = null;

    try {
      const clip = await this.clipFile(id);
      if (!current()) return;
      const base = new URL("./", new URL(this.results?.assets.clips[id] ?? "", document.baseURI)).href;
      this.layOut(clip);
      const video = this.loadVideo(clip, base);
      this.view.say("Decoding the frames the model reads");
      const frames = await decodeSprites(clip, base);
      if (!current()) return;
      if (clip.sprite.frame_w !== INPUT_W || clip.sprite.frame_h !== INPUT_H) {
        throw new AssetError(`the clip's frames are ${clip.sprite.frame_w} x ${clip.sprite.frame_h}, not the model's ${INPUT_W} x ${INPUT_H}`, "invalid", base);
      }
      // Before the clip's published numbers go on screen: are these the frames
      // they were computed on?
      const differing = await this.decodedDifferently(frames, clip);
      if (!current()) return;
      const loaded: Loaded = {
        clip,
        frames,
        tracks: {},
        scale: errorScale(clip.labels_deg, clip.expected),
      };
      this.loaded = loaded;
      this.strip.setClip(clip.labels_deg, clip.fps, loaded.scale);
      this.refreshStrip();
      this.view.say("Starting the model");
      await this.session(this.model);
      if (!current()) return;
      await this.compute([0, 1, 2, 3], this.model, loaded);
      if (!current()) return;
      await video;
      if (!current()) return;
      await this.seekVideo(0);
      if (!current()) return;
      this.view.say("");
      this.view.setState("ready");
      this.view.enable(true);
      this.render(0, true);
      if (differing > 0) {
        this.view.remark(
          `This browser decodes ${differing} of the clip’s ${frames.length} frames slightly differently from the reference decoder, so the model’s angles here can differ a little from the published ones.`,
        );
      }
      if (play) {
        await this.play();
      }
      void this.fill(generation);
      this.warmTwin();
    } catch (error) {
      if (current()) this.unavailable(error);
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

  private loadVideo(clip: Clip, base: string): Promise<void> {
    const video = this.view.video;
    if (clip.video === null) {
      return Promise.reject(new AssetError("this clip has no video", "missing", base));
    }
    const src = base + clip.video;
    return new Promise<void>((resolve, reject) => {
      const ready = () => {
        video.removeEventListener("error", failed);
        resolve();
      };
      const failed = () => {
        video.removeEventListener("loadeddata", ready);
        reject(new AssetError(`this browser could not play ${clip.video}`, "undecodable", src));
      };
      video.addEventListener("loadeddata", ready, { once: true });
      video.addEventListener("error", failed, { once: true });
      video.muted = true;
      video.preload = "auto";
      video.src = src;
      video.load();
    });
  }

  /** Seek the video to frame `k` and wait for it, at most a couple of seconds:
   * a server that cannot serve byte ranges cannot seek a video, and the demo
   * should start regardless. */
  private seekVideo(k: number): Promise<void> {
    const video = this.view.video;
    const clip = this.loaded?.clip;
    if (!clip) return Promise.resolve();
    return new Promise<void>((resolve) => {
      const done = () => {
        clearTimeout(timer);
        video.removeEventListener("seeked", done);
        resolve();
      };
      const timer = setTimeout(done, 2000);
      video.addEventListener("seeked", done);
      video.currentTime = seekTime(k, clip.fps);
    });
  }

  /** How many decoded frames differ from the reference decoder's, by their
   * sha256. On a difference the demo still runs, and says that its numbers can
   * differ a little from the published ones. */
  private async decodedDifferently(frames: readonly Bytes[], clip: Clip): Promise<number> {
    try {
      return (await mismatchedFrames(frames, clip.sprite.decoded_sha256)).length;
    } catch {
      // No SubtleCrypto (an insecure context): the check cannot run, and the demo runs regardless.
      return 0;
    }
  }

  private unavailable(error: unknown): void {
    this.stopLoop();
    this.view.video.pause();
    this.playing = false;
    this.view.showPlaying(false);
    this.view.enable(false);
    this.view.setState("unavailable");
    this.view.say(failureMessage(error));
    // A copy of the site built without its data is expected to say so; anything
    // else is a fault worth its detail in the console.
    if (!(error instanceof AssetError && error.kind === "missing")) {
      console.warn("The demo could not start:", error);
    }
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
   * skipping any already done. */
  private compute(frames: readonly number[], model: ModelKey, loaded: Loaded): Promise<void> {
    return this.exclusive(async () => {
      const track = this.track(loaded, model);
      const todo = frames.filter((k) => k >= 0 && k < loaded.clip.n && !track.done[k]);
      if (todo.length === 0) return;
      const session = await this.session(model);
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

  /** Work through the clip ahead of the frame on screen until every frame has
   * the selected model's output, yielding to the page between batches. */
  private async fill(generation: number): Promise<void> {
    if (this.filling) return;
    this.filling = true;
    try {
      while (generation === this.generation && !this.hidden) {
        const loaded = this.loaded;
        if (!loaded) break;
        const model = this.model;
        const size = Math.max(1, Math.min(MAX_BATCH, Math.floor(BATCH_BUDGET_MS / this.msPerFrame)));
        const next = this.nextMissing(this.track(loaded, model), this.frame, size);
        if (next.length === 0) break;
        await this.compute(next, model, loaded);
        if (this.stripStale) this.refreshStrip();
        const waiting = this.awaiting;
        if (waiting !== null && this.loaded === loaded && this.track(loaded, this.model).done[waiting]) {
          this.awaiting = null;
          if (this.frame === waiting) this.render(waiting, !this.playing);
        }
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    } catch (error) {
      if (generation === this.generation) this.unavailable(error);
    } finally {
      this.filling = false;
    }
  }

  /** Download the twin once the opening clip is running, so switching to it
   * is immediate. */
  private warmTwin(): void {
    const idle = (globalThis as { requestIdleCallback?: (cb: () => void) => void }).requestIdleCallback;
    const later = (cb: () => void) => (idle ? idle(cb) : setTimeout(cb, 2000));
    later(() => void this.session("shuffled").catch(() => undefined));
  }

  // -- Drawing ---------------------------------------------------------------------

  private refreshStrip(): void {
    const loaded = this.loaded;
    if (!loaded) return;
    const { clip } = loaded;
    const track = loaded.tracks[this.model];
    const expected = clip.expected[this.model];
    const values = Float32Array.from(expected, (value, k) => (track?.done[k] ? (track.angles[k] as number) : value));
    this.strip.setValues(values, clip.clip_mae_deg[this.model]);
    this.view.average.textContent = fmtDeg(clip.clip_mae_deg[this.model]);
    this.stripStale = false;
  }

  /** Show frame `k`: the wheel, the readouts, the heat and the model's input.
   * A frame the model has not reached yet is computed first. */
  private render(k: number, announce: boolean): void {
    const loaded = this.loaded;
    if (!loaded) return;
    this.frame = k;
    const { clip } = loaded;
    const human = clip.labels_deg[k] as number;
    const track = this.track(loaded, this.model);
    this.view.steer(this.view.driverAngle, this.view.driverSide, human);
    if (track.done[k]) {
      const angle = track.angles[k] as number;
      renderWheel(this.view.wheel, angle, human);
      this.view.steer(this.view.modelAngle, this.view.modelSide, angle);
      this.view.gap.textContent = fmtDeg(Math.abs(angle - human));
      this.paintHeat(track, k, loaded);
    } else {
      // The model's wheel waits where it was until its output arrives: the
      // fill works from the frame on screen, and draws it again when done.
      renderWheel(this.view.wheel, null, human);
      this.view.steer(this.view.modelAngle, this.view.modelSide, Number.NaN);
      this.view.gap.textContent = NO_VALUE;
      this.awaiting = k;
      void this.fill(this.generation);
    }
    if (this.stripStale) this.refreshStrip();
    this.strip.setFrame(k, announce);
  }

  private paintHeat(track: Track, k: number, loaded: Loaded): void {
    const heat = this.heatContext;
    if (heat) {
      heat.putImageData(heatImageFromBytes(track.saliency.subarray(k * MAP, (k + 1) * MAP), INPUT_W, INPUT_H, HEAT_ALPHA), 0, 0);
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
      eye.drawImage(this.view.heat, 0, 0);
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
    if (!loaded) return;
    this.wanted = true;
    if (this.hidden) return;
    try {
      await this.view.video.play();
    } catch {
      // Autoplay refused (a saved data mode, a browser setting): stay paused.
      this.wanted = false;
      this.view.showPlaying(false);
      return;
    }
    if (this.loaded !== loaded) return;
    this.playing = true;
    this.view.showPlaying(true);
    this.stopLoop();
    this.loop = startFrameLoop(this.view.video, loaded.clip.fps, loaded.clip.n, (k) => this.render(k, false));
    void this.fill(this.generation);
  }

  private pause(): void {
    this.view.video.pause();
    this.stopLoop();
    this.playing = false;
    this.view.showPlaying(false);
    this.render(this.frame, true);
  }

  private seek(k: number): void {
    if (!this.loaded) return;
    this.view.video.currentTime = seekTime(k, this.loaded.clip.fps);
    this.render(k, true);
    void this.fill(this.generation);
  }

  private step(by: number): void {
    const loaded = this.loaded;
    if (!loaded) return;
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
      if (this.wanted && this.loaded) void this.play();
      void this.fill(this.generation);
    }
  }

  private selectModel(model: ModelKey): void {
    this.model = model;
    this.view.modelName.textContent = MODEL_NAMES[model];
    this.view.note.textContent = modelNote(model, this.loaded?.clip ?? null);
    const loaded = this.loaded;
    if (!loaded) return;
    this.refreshStrip();
    if (!this.sessions[model]) this.view.say("Loading the twin");
    const generation = this.generation;
    this.session(model).then(
      () => {
        if (generation !== this.generation || this.model !== model) return;
        this.view.say("");
        this.render(this.frame, true);
        void this.fill(generation);
      },
      (error) => {
        if (generation !== this.generation || this.model !== model) return;
        // The other model still runs: go back to it and say what happened.
        console.warn("The twin model could not be loaded:", error);
        this.view.say("");
        this.view.remark("This browser could not load the twin model; Sightline is shown instead.");
        const fallback = this.view.models.querySelector<HTMLInputElement>('input[value="sightline"]');
        if (fallback) fallback.checked = true;
        this.selectModel("sightline");
      },
    );
  }

  // -- Controls ----------------------------------------------------------------------

  private wire(root: HTMLElement): void {
    const { view } = this;
    view.play.addEventListener("click", () => {
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
      if (input.name === "cockpit-clip") {
        void this.showClip(input.value, this.wanted).then(() => {
          view.note.textContent = modelNote(this.model, this.loaded?.clip ?? null);
        });
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

/** Mount the cockpit into the page and start it. */
export function startCockpit(): Cockpit | null {
  const root = mountCockpitMarkup();
  if (!root) return null;
  const cockpit = new Cockpit(root);
  void cockpit.start();
  return cockpit;
}
