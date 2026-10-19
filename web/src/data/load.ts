/**
 * Loading the site data, and checking it is what the page expects.
 *
 * A file the page needs can be missing (not built into this copy of the site),
 * unreachable (the network, or the server, failed), invalid (present, but not
 * the shape the page reads) or undecodable (an image or video this browser
 * cannot decode). The page says which, so `AssetError` carries the kind and
 * the URL. Parsing checks every key the page uses -- the types,
 * the array lengths that must agree, the lags the lead-in frames must cover --
 * and names the key that is wrong, so a page built against data of another
 * shape fails with a sentence rather than with a NaN on screen.
 */

import { MODEL_KEYS, type Clip, type ModelKey, type Results } from "./types.ts";

/** "stopped" is footage that loaded and then broke off. */
export type AssetErrorKind = "missing" | "unreachable" | "invalid" | "undecodable" | "stopped";

export class AssetError extends Error {
  constructor(
    message: string,
    readonly kind: AssetErrorKind,
    readonly url: string,
  ) {
    super(message);
    this.name = "AssetError";
  }
}

/** `fetch(url)`, failing with an `AssetError` unless the response is a 2xx. */
export async function fetchOk(url: string): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new AssetError(`could not reach ${url}: ${(error as Error).message}`, "unreachable", url);
  }
  if (response.status === 404) {
    throw new AssetError(`${url} is not on this server`, "missing", url);
  }
  if (!response.ok) {
    throw new AssetError(`${url} answered ${response.status}`, "unreachable", url);
  }
  return response;
}

export async function fetchJson(url: string): Promise<unknown> {
  const response = await fetchOk(url);
  // Some servers answer an unknown path with the site's own page instead of a
  // 404: a web page where data should be means the data is not there.
  if ((response.headers.get("content-type") ?? "").includes("text/html")) {
    throw new AssetError(`${url} answered with a web page, not the data`, "missing", url);
  }
  const text = await response.text();
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new AssetError(`${url} is not JSON`, "invalid", url);
  }
}

// -- Checking a parsed file ------------------------------------------------------

type Json = Readonly<Record<string, unknown>>;

class Checker {
  constructor(private readonly url: string) {}

  fail(key: string, wanted: string): never {
    throw new AssetError(`${this.url}: ${key} must be ${wanted}`, "invalid", this.url);
  }

  object(value: unknown, key: string): Json {
    if (typeof value !== "object" || value === null || Array.isArray(value)) this.fail(key, "an object");
    return value as Json;
  }

  string(value: unknown, key: string): string {
    if (typeof value !== "string" || value === "") this.fail(key, "a non-empty string");
    return value;
  }

  number(value: unknown, key: string, check: (v: number) => boolean = () => true, wanted = "a number"): number {
    if (typeof value !== "number" || !Number.isFinite(value) || !check(value)) this.fail(key, wanted);
    return value;
  }

  integer(value: unknown, key: string, min: number): number {
    return this.number(value, key, (v) => Number.isInteger(v) && v >= min, `an integer of at least ${min}`);
  }

  numbers(value: unknown, key: string, length?: number): number[] {
    const wanted = length === undefined ? "a list of numbers" : `a list of ${length} numbers`;
    if (!Array.isArray(value) || (length !== undefined && value.length !== length)) this.fail(key, wanted);
    return value.map((v, i) => this.number(v, `${key}[${i}]`));
  }

  strings(value: unknown, key: string, length?: number): string[] {
    const wanted = length === undefined ? "a non-empty list of strings" : `a list of ${length} strings`;
    if (!Array.isArray(value) || value.length === 0 || (length !== undefined && value.length !== length)) {
      this.fail(key, wanted);
    }
    return value.map((v, i) => this.string(v, `${key}[${i}]`));
  }

  size(value: unknown, key: string): [number, number] {
    const [w, h] = this.numbers(value, key, 2);
    if (!(w! > 0 && h! > 0)) this.fail(key, "a [width, height] pair");
    return [w!, h!];
  }

  share(value: unknown, key: string): number {
    return this.number(value, key, (v) => v >= 0 && v <= 1, "a share between 0 and 1");
  }
}

/** A clip.json, checked. Throws an `AssetError` of kind "invalid" naming the
 * first key that is missing or wrong. */
export function parseClip(json: unknown, url: string): Clip {
  const c = new Checker(url);
  const o = c.object(json, "the file");
  const n = c.integer(o["n"], "n", 1);
  const leadIn = c.integer(o["lead_in"], "lead_in", 0);
  const lags = c.numbers(o["frame_lags"], "frame_lags");
  lags.forEach((lag, i) => {
    if (!Number.isInteger(lag) || lag < 1) c.fail(`frame_lags[${i}]`, "a positive whole number of frames");
    if (lag > leadIn) c.fail(`frame_lags[${i}]`, `at most lead_in (${leadIn}), or the first frames have no history`);
  });
  const expected = c.object(o["expected"], "expected");
  const sprite = c.object(o["sprite"], "sprite");
  const mae = c.object(o["clip_mae_deg"], "clip_mae_deg");
  const video = o["video"] === null ? null : c.string(o["video"], "video");
  const tolerance = o["tolerance"] === undefined ? undefined : c.object(o["tolerance"], "tolerance");
  return {
    id: c.string(o["id"], "id"),
    drive: c.string(o["drive"], "drive"),
    title: c.string(o["title"], "title"),
    fps: c.number(o["fps"], "fps", (v) => v > 0, "a positive number"),
    n,
    lead_in: leadIn,
    first_frame: c.integer(o["first_frame"], "first_frame", 0),
    frame_lags: lags,
    rule: c.string(o["rule"], "rule"),
    labels_deg: c.numbers(o["labels_deg"], "labels_deg", n),
    expected: Object.fromEntries(
      MODEL_KEYS.map((key) => [key, c.numbers(expected[key], `expected.${key}`, n)]),
    ) as Record<ModelKey, number[]>,
    sprite: {
      files: c.strings(sprite["files"], "sprite.files"),
      frames_per_sheet: c.integer(sprite["frames_per_sheet"], "sprite.frames_per_sheet", 1),
      frame_w: c.integer(sprite["frame_w"], "sprite.frame_w", 1),
      frame_h: c.integer(sprite["frame_h"], "sprite.frame_h", 1),
      decoded_sha256: c.strings(sprite["decoded_sha256"], "sprite.decoded_sha256", leadIn + n),
    },
    video,
    video_size: o["video_size"] === null ? null : c.size(o["video_size"], "video_size"),
    camera_size: c.size(o["camera_size"], "camera_size"),
    clip_mae_deg: {
      sightline: c.number(mae["sightline"], "clip_mae_deg.sightline"),
      shuffled: c.number(mae["shuffled"], "clip_mae_deg.shuffled"),
      zero: c.number(mae["zero"], "clip_mae_deg.zero"),
    },
    ...(o["twin_trained_frac"] === undefined
      ? {}
      : { twin_trained_frac: c.share(o["twin_trained_frac"], "twin_trained_frac") }),
    ...(o["twin_neighbour_frac"] === undefined
      ? {}
      : { twin_neighbour_frac: c.share(o["twin_neighbour_frac"], "twin_neighbour_frac") }),
    ...(tolerance === undefined
      ? {}
      : {
          tolerance: {
            angle_deg: c.number(tolerance["angle_deg"], "tolerance.angle_deg", (v) => v > 0, "a positive number"),
          },
        }),
  };
}

/** results.json, checked for what the cockpit reads; everything else is kept
 * in `raw`. */
export function parseResults(json: unknown, url: string): Results {
  const c = new Checker(url);
  const o = c.object(json, "the file");
  const assets = c.object(o["assets"], "assets");
  const models = c.object(assets["models"], "assets.models");
  const clips = c.object(assets["clips"], "assets.clips");
  const shipped = c.object(o["shipped"], "shipped");
  const twin = c.object(o["twin"], "twin");
  return {
    assets: {
      models: {
        sightline: c.string(models["sightline"], "assets.models.sightline"),
        shuffled: c.string(models["shuffled"], "assets.models.shuffled"),
      },
      clips: Object.fromEntries(Object.entries(clips).map(([id, path]) => [id, c.string(path, `assets.clips.${id}`)])),
    },
    shipped: {
      run: c.string(shipped["run"], "shipped.run"),
      model: c.string(shipped["model"], "shipped.model"),
      protocol: c.string(shipped["protocol"], "shipped.protocol"),
    },
    twin: { run: c.string(twin["run"], "twin.run") },
    raw: o,
  };
}

/** results.json from the site root `base` (ending in a slash). */
export async function loadResults(base: string): Promise<Results> {
  const url = `${base}data/results.json`;
  return parseResults(await fetchJson(url), url);
}

/** A clip.json at `url`. */
export async function loadClip(url: string): Promise<Clip> {
  return parseClip(await fetchJson(url), url);
}
