/**
 * The page's self-test: open the page with `?selftest=1`, and the tab title
 * says whether this browser runs the published models and gets the published
 * outputs.
 *
 * It goes through the page's own code wherever the demo has code for the
 * step: the session settings of `createSession`, the sprite decoding of
 * `decodeSprites`, the frame stacking of `stackAt`, the model call of `infer`.
 * Six checks run in order, and each counts towards the verdict:
 *
 *   1. sessions: both models load, the shipped one and its shuffled-split
 *      twin, from the files results.json names;
 *   2. metadata: each file is the published model, by the run, the network
 *      and the frame lags written into the file itself;
 *   3. selftest-inputs: on the raw frame stacks of data/selftest/, both
 *      models' angles and saliency maps match the ones onnxruntime computed
 *      in Python, within the tolerances expected.json records;
 *   4. decode-hash: every sprite frame of every clip decodes here to the
 *      bytes Python decoded (each frame's sha256);
 *   5. clip-angles: on every scored frame of every clip, stacked by the page's
 *      code, both models' angles match the clip's published ones;
 *   6. clip-mae: results.json loads, the cockpit shows the published average
 *      error of the clip on screen, and every clip's published averages follow
 *      from its own published angles and labels. That last part checks the
 *      data for consistency: the averages were computed in Python, not here.
 *
 * The verdict is `PASS 6/6 threads=1` when every check passed with the
 * runtime on one thread, and otherwise `FAIL k/6 <the first failing check>:
 * <why>`, with k the checks that passed. A failing verdict never starts with
 * PASS, so a reader that matches the PASS prefix cannot take a failure for a
 * pass. Every check runs even after one fails, as far as what it needs could
 * be had, and the console lists what each one found.
 *
 * Nothing here runs, and the title is never written, unless the address asks
 * for the self-test: main.ts imports this module only then, and the module
 * asks again before it starts.
 */

import type { InferenceSession } from "onnxruntime-web/wasm";

// The cockpit's module, for the build's sake: imported here, the cockpit, the
// model runtime and the decoding the self-test shares with it stay in the one
// chunk the page already loads, and preloads from its head, rather than being
// split into a further chunk of the page's own load. Importing it starts
// nothing.
import "./cockpit/cockpit.ts";
import { COCKPIT_SELECTOR } from "./cockpit/boot.ts";
import { checkDecoded, decodeSprites, stackAt, type Bytes, type DecodeCheck } from "./cockpit/sprites.ts";
import { fetchJson, fetchOk, loadClip, loadResults } from "./data/load.ts";
import { MODEL_KEYS, type Clip, type ModelKey, type Results } from "./data/types.ts";
import { fmtDeg } from "./format.ts";
import { modelMetadata } from "./model/metadata.ts";
import { compareOutputs, compareWithin, describeComparison, parseSelftestInputs } from "./model/parity.ts";
import { INPUT_H, INPUT_W, createSession, infer, threadsInUse, type Outputs } from "./model/runner.ts";

export const CHECKS = ["sessions", "metadata", "selftest-inputs", "decode-hash", "clip-angles", "clip-mae"] as const;
export type CheckName = (typeof CHECKS)[number];

export interface CheckResult {
  readonly name: string;
  readonly ok: boolean;
  /** What the check found, or why it failed. */
  readonly detail: string;
}

/** Whether the address asks for the self-test. */
export function selftestRequested(search: string): boolean {
  return new URLSearchParams(search).get("selftest") === "1";
}

/** The longest reason a failing verdict carries. */
const REASON_CHARS = 160;

function oneLine(text: string): string {
  const line = text.replace(/\s+/g, " ").trim();
  return line.length > REASON_CHARS ? `${line.slice(0, REASON_CHARS - 1)}…` : line;
}

/** The verdict for the tab title: `PASS k/k threads=1`, or `FAIL k/m` with the
 * first failing check and why. */
export function formatVerdict(results: readonly CheckResult[], threads: number): string {
  const total = results.length;
  const passed = results.filter((result) => result.ok).length;
  const failed = results.find((result) => !result.ok);
  if (total > 0 && failed === undefined && threads === 1) {
    return `PASS ${passed}/${total} threads=1`;
  }
  const reason = failed ? oneLine(`${failed.name}: ${failed.detail}`) : total === 0 ? "no checks ran" : `threads=${threads}`;
  return `FAIL ${passed}/${total} ${reason}`;
}

// -- What the self-test reads of the page -------------------------------------------------

/** What the cockpit shows: the clip and model picked and the clip's average
 * error once it runs, or else its state and what its status line says. */
export type Shown =
  | { readonly state: "ready"; readonly clip: string; readonly model: string; readonly average: string }
  | { readonly state: "loading" | "unavailable" | "absent"; readonly status: string };

/** The cockpit's parts the self-test reads, in its markup (markup.ts and
 * cockpit.ts write them). */
const PARTS = {
  console: ".cockpit__console",
  status: '[data-cockpit="status"]',
  average: '[data-cockpit="average"]',
  clip: 'input[name="cockpit-clip"]:checked',
  model: 'input[name="cockpit-model"]:checked',
} as const;

/** What the cockpit shows right now. */
export function readCockpit(root: ParentNode = document): Shown {
  const cockpit = root.querySelector<HTMLElement>(COCKPIT_SELECTOR);
  if (!cockpit) return { state: "absent", status: "" };
  const state = cockpit.querySelector<HTMLElement>(PARTS.console)?.dataset["state"];
  if (state === "ready") {
    return {
      state,
      clip: cockpit.querySelector<HTMLInputElement>(PARTS.clip)?.value ?? "",
      model: cockpit.querySelector<HTMLInputElement>(PARTS.model)?.value ?? "",
      average: cockpit.querySelector(PARTS.average)?.textContent ?? "",
    };
  }
  const status = (cockpit.querySelector(PARTS.status)?.textContent ?? "").trim();
  return { state: state === "unavailable" ? "unavailable" : "loading", status };
}

/** How long the self-test waits for the cockpit to start, and how often it
 * looks. The cockpit loads its footage alongside the self-test's own work, so
 * it is usually running long before it is asked. */
const COCKPIT_WAIT_MS = 60_000;
const COCKPIT_POLL_MS = 250;

/** What the self-test needs of the page and its files: the page's own
 * functions in the browser (`browserPage`), stand-ins in tests. */
export interface Page<Session> {
  /** The site root, ending in a slash. */
  readonly base: string;
  results(base: string): Promise<Results>;
  clip(url: string): Promise<Clip>;
  bytes(url: string): Promise<Bytes>;
  json(url: string): Promise<unknown>;
  /** A session on exactly these model bytes. */
  open(model: Bytes): Promise<Session>;
  infer(session: Session, frames: Bytes, n: number, channels: number): Promise<Outputs>;
  /** The threads the runtime uses. */
  threads(): number;
  /** Every sprite frame of `clip`, whose files sit in `folder`. */
  decode(clip: Clip, folder: string): Promise<Bytes[]>;
  hashes(frames: readonly Bytes[], digests: readonly string[]): Promise<DecodeCheck>;
  /** What the cockpit shows once it has started, or why it has not. */
  cockpit(): Promise<Shown>;
  /** A moment for the page between batches. */
  idle(): Promise<void>;
}

// -- The checks --------------------------------------------------------------------------

/** A check failed, for the reason it gives. */
class Failure extends Error {}

const message = (error: unknown) => (error instanceof Error ? error.message : String(error));

/** Stacks per run of the model, as the cockpit batches them. */
const BATCH = 8;

/** How closely a clip's published average error must follow from its own
 * published angles and labels: both come out of the same float64 sums in
 * Python, so only the order of the additions differs. */
const MAE_RELATIVE = 1e-6;

/** Which run each model file must hold, and which network, by results.json. */
const RUN: Readonly<Record<ModelKey, (results: Results) => string>> = {
  sightline: (results) => results.shipped.run,
  shuffled: (results) => results.twin.run,
};
const NETWORK: Readonly<Record<ModelKey, (results: Results) => unknown>> = {
  sightline: (results) => results.shipped.model,
  shuffled: (results) => (results.raw["twin"] as Readonly<Record<string, unknown>> | undefined)?.["model"],
};

/** What the checks have loaded, for the checks after them. */
interface Loaded<Session> {
  results: Results | Error;
  clips: Map<string, Clip | Error>;
  models: Map<ModelKey, Bytes>;
  sessions: Map<ModelKey, Session>;
  frames: Map<string, Bytes[]>;
}

/** The result of one check: what it found when every part passed, or every
 * problem it found. */
function outcome(found: readonly string[], problems: readonly string[]): string {
  if (problems.length > 0) throw new Failure(problems.join("; "));
  return found.join("; ");
}

function listOf(indices: readonly number[]): string {
  const shown = indices.slice(0, 5).join(", ");
  return indices.length > 5 ? `${shown} and ${indices.length - 5} more` : shown;
}

function sameNumbers(a: unknown, b: readonly number[]): boolean {
  return Array.isArray(a) && a.length === b.length && a.every((value, i) => value === b[i]);
}

class Selftest<Session> {
  private readonly loaded: Loaded<Session> = {
    results: new Error("not loaded yet"),
    clips: new Map(),
    models: new Map(),
    sessions: new Map(),
    frames: new Map(),
  };

  constructor(private readonly page: Page<Session>) {}

  private url(path: string): string {
    return new URL(path, this.page.base).href;
  }

  /** results.json and every clip.json it names. A failure stays in `loaded`
   * for the checks that need the file. */
  async load(): Promise<void> {
    try {
      this.loaded.results = await this.page.results(this.page.base);
    } catch (error) {
      this.loaded.results = error instanceof Error ? error : new Error(String(error));
      return;
    }
    for (const [id, path] of Object.entries(this.loaded.results.assets.clips)) {
      try {
        this.loaded.clips.set(id, await this.page.clip(this.url(path)));
      } catch (error) {
        this.loaded.clips.set(id, error instanceof Error ? error : new Error(String(error)));
      }
    }
  }

  private results(): Results {
    const { results } = this.loaded;
    if (results instanceof Error) throw new Failure(`results.json did not load: ${results.message}`);
    return results;
  }

  private clips(): Clip[] {
    this.results();
    const clips = [...this.loaded.clips.entries()];
    if (clips.length === 0) throw new Failure("results.json names no clips");
    const missing = clips.flatMap(([id, clip]) => (clip instanceof Error ? [`${id}: ${clip.message}`] : []));
    if (missing.length > 0) throw new Failure(`a clip did not load: ${missing.join("; ")}`);
    return clips.map(([, clip]) => clip as Clip);
  }

  private folder(clip: Clip): string {
    return new URL("./", this.url(this.results().assets.clips[clip.id] ?? "")).href;
  }

  private session(model: ModelKey): Session {
    const session = this.loaded.sessions.get(model);
    if (session === undefined) throw new Failure("its session did not load");
    return session;
  }

  /** 1. Both models load, from the files results.json names. */
  async sessions(): Promise<string> {
    const results = this.results();
    const found: string[] = [];
    const problems: string[] = [];
    for (const model of MODEL_KEYS) {
      const path = results.assets.models[model];
      try {
        const bytes = await this.page.bytes(this.url(path));
        this.loaded.models.set(model, bytes);
        this.loaded.sessions.set(model, await this.page.open(bytes));
        found.push(`${model} ${path} (${bytes.length} bytes)`);
      } catch (error) {
        problems.push(`${model} ${path}: ${message(error)}`);
      }
    }
    return outcome(found, problems);
  }

  /** 2. Each file holds the published model: its run, its network and the
   * lags every clip stacks. */
  async metadata(): Promise<string> {
    const results = this.results();
    const clips = this.clips();
    const found: string[] = [];
    const problems: string[] = [];
    for (const model of MODEL_KEYS) {
      const bytes = this.loaded.models.get(model);
      if (bytes === undefined) {
        problems.push(`${model}: its file did not load`);
        continue;
      }
      let meta: ReadonlyMap<string, string>;
      try {
        meta = modelMetadata(bytes);
      } catch (error) {
        problems.push(`${model}: its file cannot be read as ONNX: ${message(error)}`);
        continue;
      }
      const run = meta.get("run");
      if (run !== RUN[model](results)) problems.push(`${model} is run ${JSON.stringify(run ?? null)}, not ${RUN[model](results)}`);
      const network = NETWORK[model](results);
      if (network !== undefined && meta.get("model") !== network) {
        problems.push(`${model} is network ${JSON.stringify(meta.get("model") ?? null)}, not ${String(network)}`);
      }
      const written = meta.get("frame_lags") ?? "";
      let lags: unknown = null;
      try {
        lags = JSON.parse(written);
      } catch {
        lags = null;
      }
      for (const clip of clips) {
        if (!sameNumbers(lags, clip.frame_lags)) {
          problems.push(`${model} stacks lags ${written || "(none)"}, but clip ${clip.id} stacks ${JSON.stringify(clip.frame_lags)}`);
        }
      }
      found.push(`${model}: run ${run}, lags ${written}`);
    }
    return outcome(found, problems);
  }

  /** 3. Both models on the raw self-test stacks against Python's outputs. */
  async selftestInputs(): Promise<string> {
    const files = (this.results().raw["assets"] as Readonly<Record<string, unknown>> | undefined)?.["selftest"] as
      | Readonly<Record<string, unknown>>
      | undefined;
    const expectedPath = files?.["expected"];
    const framesPath = files?.["frames"];
    if (typeof expectedPath !== "string" || typeof framesPath !== "string") {
      throw new Failure("results.json names no self-test files (assets.selftest.expected and .frames)");
    }
    const expectedUrl = this.url(expectedPath);
    const [json, frames] = await Promise.all([this.page.json(expectedUrl), this.page.bytes(this.url(framesPath))]);
    const inputs = parseSelftestInputs(json, frames, expectedUrl);
    const found: string[] = [];
    const problems: string[] = [];
    for (const model of MODEL_KEYS) {
      try {
        const outputs = await this.page.infer(this.session(model), inputs.frames, inputs.n, inputs.channels);
        const { angles, saliency } = compareOutputs(outputs, inputs, model);
        const said = `${model} angles ${describeComparison(angles, "°")}, saliency ${describeComparison(saliency, "")}`;
        if (angles.ok && saliency.ok) found.push(said);
        else problems.push(said);
      } catch (error) {
        problems.push(`${model}: ${message(error)}`);
      }
      await this.page.idle();
    }
    return outcome(found, problems);
  }

  /** 4. Every sprite frame decodes to the reference decoder's bytes. */
  async decodeHash(): Promise<string> {
    const found: string[] = [];
    const problems: string[] = [];
    for (const clip of this.clips()) {
      try {
        const frames = await this.page.decode(clip, this.folder(clip));
        this.loaded.frames.set(clip.id, frames);
        const check = await this.page.hashes(frames, clip.sprite.decoded_sha256);
        if (check.status === "match") {
          found.push(`${clip.id}: all ${frames.length} frames match`);
        } else if (check.status === "differ") {
          const one = check.frames.length === 1;
          problems.push(`${clip.id}: frame${one ? "" : "s"} ${listOf(check.frames)} of ${frames.length} decode${one ? "s" : ""} to other bytes than Python's`);
        } else {
          problems.push(`${clip.id}: this browser cannot compute sha256 here, so the frames cannot be checked`);
        }
      } catch (error) {
        problems.push(`${clip.id}: ${message(error)}`);
      }
    }
    return outcome(found, problems);
  }

  /** 5. Every scored frame of every clip, stacked by the page's own code,
   * against the clip's published angles, for both models. */
  async clipAngles(): Promise<string> {
    const found: string[] = [];
    const problems: string[] = [];
    for (const clip of this.clips()) {
      const frames = this.loaded.frames.get(clip.id);
      const tolerance = clip.tolerance?.angle_deg;
      if (frames === undefined) {
        problems.push(`${clip.id}: its frames did not decode`);
        continue;
      }
      if (tolerance === undefined) {
        problems.push(`${clip.id} records no angle tolerance`);
        continue;
      }
      const channels = 3 * (1 + clip.frame_lags.length);
      const stride = INPUT_H * INPUT_W * channels;
      for (const model of MODEL_KEYS) {
        try {
          const session = this.session(model);
          const angles: number[] = [];
          for (let start = 0; start < clip.n; start += BATCH) {
            const count = Math.min(BATCH, clip.n - start);
            const batch = new Uint8Array(count * stride);
            for (let i = 0; i < count; i += 1) {
              batch.set(stackAt(frames, clip.lead_in + start + i, clip.frame_lags), i * stride);
            }
            const outputs = await this.page.infer(session, batch, count, channels);
            angles.push(...outputs.angles);
            await this.page.idle();
          }
          const comparison = compareWithin(angles, clip.expected[model], tolerance);
          const said = `${clip.id} ${model} ${describeComparison(comparison, "°")}`;
          if (comparison.ok) found.push(said);
          else problems.push(said);
        } catch (error) {
          problems.push(`${clip.id} ${model}: ${message(error)}`);
        }
      }
    }
    return outcome(found, problems);
  }

  /** 6. results.json loaded, the cockpit shows the published average of the
   * clip on screen, and every clip's published averages follow from its own
   * angles and labels. */
  async clipMae(): Promise<string> {
    const clips = this.clips();
    const problems: string[] = [];
    let checked = 0;
    for (const clip of clips) {
      const labels = clip.labels_deg;
      const mean = (error: (k: number) => number) => labels.reduce((sum, _, k) => sum + error(k), 0) / labels.length;
      for (const key of [...MODEL_KEYS, "zero"] as const) {
        const recomputed =
          key === "zero"
            ? mean((k) => Math.abs(labels[k] as number))
            : mean((k) => Math.abs((clip.expected[key][k] as number) - (labels[k] as number)));
        const published = clip.clip_mae_deg[key];
        const agrees =
          Number.isFinite(recomputed) && Number.isFinite(published) && Math.abs(recomputed - published) <= MAE_RELATIVE * Math.abs(published);
        if (!agrees) problems.push(`${clip.id} ${key}: the published average ${published}° is not the ${recomputed}° its own angles and labels give`);
        checked += 1;
      }
    }
    const shown = await this.page.cockpit();
    let onScreen = "";
    if (shown.state !== "ready") {
      problems.push(`the cockpit is not running (${shown.state}${shown.status ? `: ${shown.status}` : ""})`);
    } else {
      const clip = clips.find((candidate) => candidate.id === shown.clip);
      const model = MODEL_KEYS.find((key) => key === shown.model);
      if (!clip || !model) {
        problems.push(`the cockpit shows clip ${JSON.stringify(shown.clip)} and model ${JSON.stringify(shown.model)}, which results.json does not name`);
      } else {
        const published = fmtDeg(clip.clip_mae_deg[model]);
        if (shown.average !== published) {
          problems.push(`the cockpit shows ${shown.average} for ${clip.id}, ${model}; the published figure is ${published}`);
        }
        onScreen = `the cockpit shows the published ${published} for ${clip.id}, ${model}`;
      }
    }
    return outcome(
      [
        "results.json loaded",
        onScreen,
        `${checked} published clip averages agree with their clips' own angles and labels to ${MAE_RELATIVE} relative ` +
          "(a consistency check of the data: they were computed in Python, not in this browser)",
      ],
      problems,
    );
  }
}

/** Run every check in order on `page`, telling `show` each step and then the
 * verdict. Resolves to the verdict and what each check found. */
export async function runSelftest<Session>(
  page: Page<Session>,
  show: (title: string) => void,
): Promise<{ verdict: string; results: readonly CheckResult[] }> {
  const results: CheckResult[] = [];
  const total = CHECKS.length;
  const test = new Selftest(page);
  const run: Readonly<Record<CheckName, () => Promise<string>>> = {
    sessions: () => test.sessions(),
    metadata: () => test.metadata(),
    "selftest-inputs": () => test.selftestInputs(),
    "decode-hash": () => test.decodeHash(),
    "clip-angles": () => test.clipAngles(),
    "clip-mae": () => test.clipMae(),
  };
  let verdict: string;
  try {
    show(`RUNNING 0/${total} loading`);
    await test.load();
    for (const [i, name] of CHECKS.entries()) {
      show(`RUNNING ${i + 1}/${total} ${name}`);
      try {
        results.push({ name, ok: true, detail: await run[name]() });
      } catch (error) {
        results.push({ name, ok: false, detail: message(error) });
      }
    }
    verdict = formatVerdict(results, page.threads());
  } catch (error) {
    // Whatever went wrong outside the checks, the title still ends on a verdict.
    verdict = formatVerdict([...results, { name: "self-test", ok: false, detail: message(error) }], Number.NaN);
  }
  for (const result of results) {
    (result.ok ? console.info : console.warn)(`Self-test ${result.ok ? "passed" : "FAILED"} ${result.name}: ${result.detail}`);
  }
  console.info(`Self-test verdict: ${verdict}`);
  show(verdict);
  return { verdict, results };
}

// -- In the browser -----------------------------------------------------------------------

/** A session on the very bytes the self-test read, made by the page's own
 * `createSession`, so with the demo's runtime settings. */
async function openSession(model: Bytes): Promise<InferenceSession> {
  const url = URL.createObjectURL(new Blob([model], { type: "application/octet-stream" }));
  try {
    return await createSession(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function waitForCockpit(): Promise<Shown> {
  const until = performance.now() + COCKPIT_WAIT_MS;
  for (;;) {
    const shown = readCockpit();
    if (shown.state !== "loading" || performance.now() >= until) return shown;
    await new Promise((resolve) => setTimeout(resolve, COCKPIT_POLL_MS));
  }
}

function browserPage(): Page<InferenceSession> {
  return {
    base: new URL("./", document.baseURI).href,
    results: loadResults,
    clip: loadClip,
    bytes: async (url) => new Uint8Array(await (await fetchOk(url)).arrayBuffer()),
    json: fetchJson,
    open: openSession,
    infer,
    threads: threadsInUse,
    decode: (clip, folder) => decodeSprites(clip, folder),
    hashes: checkDecoded,
    cockpit: waitForCockpit,
    idle: () => new Promise((resolve) => setTimeout(resolve, 0)),
  };
}

/** The verdict goes into the tab title, and only ever under ?selftest=1. */
function writeTitle(text: string): void {
  if (selftestRequested(location.search)) document.title = text;
}

if (typeof location !== "undefined" && selftestRequested(location.search)) {
  void runSelftest(browserPage(), writeTitle);
}
