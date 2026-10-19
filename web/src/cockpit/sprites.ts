/**
 * The model's input frames, as the browser decodes them.
 *
 * Each clip ships its model input as JPEG sprite sheets: 200 x 66 frames
 * stacked top to bottom, a fixed number to a sheet. Sprite frame i is clip
 * frame i - lead_in: the `lead_in` frames before the first scored frame are
 * there so that every scored frame has its history, and nothing is ever
 * clamped. The model reads raw RGB bytes, so the sheets are decoded with no
 * colour management (`colorSpaceConversion: "none"`), into a canvas meant to be
 * read back, and every frame's bytes can be checked against the sha256 the
 * clip records for the frames the reference decoder produced.
 *
 * Every failure here is an `AssetError` saying what went wrong with the
 * footage -- not reachable, not decodable, or not the shape the clip file
 * describes -- so the page never blames the model for a picture it could not
 * read.
 */

import { AssetError, fetchOk } from "../data/load.ts";

/** Bytes backed by a plain ArrayBuffer: what SubtleCrypto and onnxruntime take. */
export type Bytes = Uint8Array<ArrayBuffer>;

/** A decoded sheet: RGBA rows, top to bottom. */
export interface SheetPixels {
  readonly data: Uint8ClampedArray;
  readonly width: number;
  readonly height: number;
}

/** What `decodeSprites` reads from a clip. */
export interface SpriteSource {
  readonly lead_in: number;
  readonly n: number;
  readonly sprite: {
    readonly files: readonly string[];
    readonly frames_per_sheet: number;
    readonly frame_w: number;
    readonly frame_h: number;
  };
}

/**
 * The model input for sprite frame `t`: frame t's RGB, then the RGB of frame
 * t - lag for each lag, per pixel -- the channel order of the Python
 * `stack_frames` the model was trained with. `t - lag` is never negative for a
 * scored frame, because the clip starts `lead_in = max(lags)` frames early.
 */
export function stackAt(frames: readonly Bytes[], t: number, lags: readonly number[]): Bytes {
  const sources = [t, ...lags.map((lag) => t - lag)].map((i) => {
    const frame = frames[i];
    if (i < 0) {
      throw new Error(`frame ${t} needs frame ${i}, before the clip's lead-in frames`);
    }
    if (!frame) {
      throw new Error(`frame ${i} is not in this clip (${frames.length} frames)`);
    }
    return frame;
  });
  const pixels = (sources[0] as Bytes).length / 3;
  const channels = 3 * sources.length;
  const out = new Uint8Array(pixels * channels);
  sources.forEach((frame, s) => {
    for (let p = 0, from = 0, to = 3 * s; p < pixels; p += 1, from += 3, to += channels) {
      out[to] = frame[from] as number;
      out[to + 1] = frame[from + 1] as number;
      out[to + 2] = frame[from + 2] as number;
    }
  });
  return out;
}

/** Cut `count` frames of `w` x `h` out of a sheet stacked top to bottom, as
 * RGB (the alpha channel dropped). */
export function framesFromSheet(sheet: SheetPixels, w: number, h: number, count: number): Bytes[] {
  if (sheet.width !== w || sheet.height !== h * count) {
    throw new Error(`a sheet of ${count} frames of ${w} x ${h} is ${w} x ${h * count}, not ${sheet.width} x ${sheet.height}`);
  }
  const frames: Bytes[] = [];
  const pixels = w * h;
  for (let f = 0; f < count; f += 1) {
    const rgb = new Uint8Array(pixels * 3);
    for (let p = 0, from = f * pixels * 4, to = 0; p < pixels; p += 1, from += 4, to += 3) {
      rgb[to] = sheet.data[from] as number;
      rgb[to + 1] = sheet.data[from + 1] as number;
      rgb[to + 2] = sheet.data[from + 2] as number;
    }
    frames.push(rgb);
  }
  return frames;
}

/** What `decodeSprites` is busy with: fetching the sheets, then decoding them. */
export type SpritePhase = "downloading" | "decoding";

/** Whether this browser decoded the reference decoder's frames: every one
 * ("match"), not every one ("differ", with the frames that differ), or
 * unknown ("unverified": the page cannot hash here -- SubtleCrypto exists only
 * in a secure context -- so it claims nothing either way). */
export type DecodeCheck =
  | { readonly status: "match" }
  | { readonly status: "differ"; readonly frames: readonly number[] }
  | { readonly status: "unverified" };

/** The SHA-256 of `bytes` as lowercase hex. */
export async function sha256Hex(bytes: Bytes): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** The indices of the frames whose sha256 is not the recorded one (a frame
 * with no recorded digest counts as a mismatch). */
export async function mismatchedFrames(frames: readonly Bytes[], digests: readonly string[]): Promise<number[]> {
  const found = await Promise.all(frames.map((frame) => sha256Hex(frame)));
  return found.flatMap((digest, i) => (digest === digests[i] ? [] : [i]));
}

/**
 * Whether `frames` are the frames the reference decoder produced, by their
 * sha256 against `digests`. Never "match" unless every frame was hashed and
 * matched: where the page cannot hash, the answer is "unverified".
 */
export async function checkDecoded(frames: readonly Bytes[], digests: readonly string[]): Promise<DecodeCheck> {
  const subtle = (globalThis.crypto as { subtle?: SubtleCrypto } | undefined)?.subtle;
  if (subtle === undefined) {
    return { status: "unverified" };
  }
  let differing: number[];
  try {
    differing = await mismatchedFrames(frames, digests);
  } catch {
    return { status: "unverified" };
  }
  return differing.length === 0 ? { status: "match" } : { status: "differ", frames: differing };
}

/** How a sheet is fetched and decoded: the browser's own way unless a test
 * says otherwise. */
export interface SheetSteps {
  download(url: string): Promise<Blob>;
  decode(file: Blob, url: string): Promise<SheetPixels>;
}

async function downloadSheet(url: string): Promise<Blob> {
  const response = await fetchOk(url);
  try {
    return await response.blob();
  } catch (error) {
    throw new AssetError(`the download of ${url} broke off: ${(error as Error).message}`, "unreachable", url);
  }
}

/** Decode one sheet: no colour conversion, no premultiplied alpha, read back
 * from a canvas made for reading. */
async function decodeSheet(file: Blob, url: string): Promise<SheetPixels> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  } catch (error) {
    throw new AssetError(`this browser could not decode ${url}: ${(error as Error).message}`, "undecodable", url);
  }
  try {
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) {
      throw new Error("no 2D canvas to read the pixels back from");
    }
    context.drawImage(bitmap, 0, 0);
    const image = context.getImageData(0, 0, bitmap.width, bitmap.height);
    return { data: image.data, width: image.width, height: image.height };
  } catch (error) {
    throw new AssetError(`this browser could not read the pixels of ${url}: ${(error as Error).message}`, "undecodable", url);
  } finally {
    bitmap.close();
  }
}

const BROWSER: SheetSteps = { download: downloadSheet, decode: decodeSheet };

/**
 * Every sprite frame of `clip` (lead_in + n of them, in order) as RGB bytes,
 * frame_w x frame_h x 3 each. Sheet file names are appended to `base`, the
 * clip's folder with its trailing slash. Every sheet is downloaded before any
 * is decoded, and `onPhase` hears when each step starts, so the page can say
 * which one it is waiting for.
 */
export async function decodeSprites(
  clip: SpriteSource,
  base: string,
  onPhase?: (phase: SpritePhase) => void,
  steps: SheetSteps = BROWSER,
): Promise<Bytes[]> {
  const { files, frames_per_sheet: perSheet, frame_w: w, frame_h: h } = clip.sprite;
  const total = clip.lead_in + clip.n;
  const urls = files.map((file) => base + file);
  onPhase?.("downloading");
  const downloaded = await Promise.all(urls.map((url) => steps.download(url)));
  onPhase?.("decoding");
  const sheets = await Promise.all(downloaded.map((file, k) => steps.decode(file, urls[k] as string)));
  const frames = sheets.flatMap((sheet, k) => {
    const url = urls[k] as string;
    const count = sheet.height / h;
    if (sheet.width !== w || !Number.isInteger(count) || count < 1 || count > perSheet) {
      throw new AssetError(
        `${url} is ${sheet.width} x ${sheet.height}, not a sheet of up to ${perSheet} frames of ${w} x ${h}`,
        "invalid",
        url,
      );
    }
    return framesFromSheet(sheet, w, h, count);
  });
  if (frames.length !== total) {
    throw new AssetError(
      `the clip needs ${total} frames (${clip.lead_in} lead-in and ${clip.n} scored), and its sheets hold ${frames.length}`,
      "invalid",
      base,
    );
  }
  return frames;
}
