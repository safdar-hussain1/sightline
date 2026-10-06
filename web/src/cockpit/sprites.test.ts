import { describe, expect, it } from "vitest";

import { decodeSprites, framesFromSheet, mismatchedFrames, sha256Hex, stackAt, type SheetPixels } from "./sprites.ts";

const PIXELS = 200 * 66;

/** A frame whose every byte is `value`. */
const flat = (value: number) => new Uint8Array(PIXELS * 3).fill(value);

describe("stackAt", () => {
  it("puts the current frame first, then each lag's frame, per pixel (Python's stack_frames order)", () => {
    const frames = [flat(0), flat(1), flat(2)];
    const stack = stackAt(frames, 2, [2]);
    expect(stack).toHaveLength(PIXELS * 6);
    expect([...stack.slice(0, 6)]).toEqual([2, 2, 2, 0, 0, 0]);
    expect([...stack.slice(-6)]).toEqual([2, 2, 2, 0, 0, 0]);
    expect([...stackAt(frames, 2, [1, 2]).slice(0, 9)]).toEqual([2, 2, 2, 1, 1, 1, 0, 0, 0]);
    expect([...stackAt(frames, 1, [1]).slice(0, 6)]).toEqual([1, 1, 1, 0, 0, 0]);
  });

  it("keeps each pixel's own colour channels in place", () => {
    const current = new Uint8Array(PIXELS * 3).map((_, i) => i % 251);
    const before = new Uint8Array(PIXELS * 3).map((_, i) => (i * 7) % 253);
    const stack = stackAt([before, flat(9), current], 2, [2]);
    for (const p of [0, 1, 199, 200, PIXELS - 1]) {
      expect([...stack.slice(p * 6, p * 6 + 3)]).toEqual([...current.slice(p * 3, p * 3 + 3)]);
      expect([...stack.slice(p * 6 + 3, p * 6 + 6)]).toEqual([...before.slice(p * 3, p * 3 + 3)]);
    }
  });

  it("is the plain frame when there are no lags", () => {
    const frames = [flat(5)];
    expect([...stackAt(frames, 0, []).slice(0, 3)]).toEqual([5, 5, 5]);
  });

  it("never reaches before the first frame: the clip's lead-in frames are there for that", () => {
    expect(() => stackAt([flat(0), flat(1)], 1, [2])).toThrow(/lead-in/);
    expect(() => stackAt([flat(0)], 1, [])).toThrow(/frame 1/);
  });
});

describe("framesFromSheet", () => {
  it("cuts a sheet into frames stacked top to bottom, dropping alpha", () => {
    // A 2 x 6 sheet holding three 2 x 2 frames; every pixel of frame f is (f, 10f, 100 + f, 255).
    const w = 2;
    const rgba = new Uint8ClampedArray(w * 6 * 4);
    for (let y = 0; y < 6; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const f = Math.floor(y / 2);
        rgba.set([f, 10 * f, 100 + f, 255], (y * w + x) * 4);
      }
    }
    const frames = framesFromSheet({ data: rgba, width: w, height: 6 }, 2, 2, 3);
    expect(frames).toHaveLength(3);
    expect([...(frames[2] ?? [])]).toEqual([2, 20, 102, 2, 20, 102, 2, 20, 102, 2, 20, 102]);
    expect([...(frames[0] ?? [])].slice(0, 3)).toEqual([0, 0, 100]);
  });

  it("refuses a sheet of another size than the clip describes", () => {
    const sheet = { data: new Uint8ClampedArray(2 * 4 * 4), width: 2, height: 4 };
    expect(() => framesFromSheet(sheet, 2, 2, 3)).toThrow(/2 x 4/);
    expect(() => framesFromSheet(sheet, 3, 2, 2)).toThrow(/2 x 4/);
  });
});

describe("sha256Hex", () => {
  it("is the standard SHA-256 of the bytes, in lowercase hex", async () => {
    expect(await sha256Hex(new TextEncoder().encode("abc"))).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });
});

describe("mismatchedFrames", () => {
  it("lists the frames whose bytes do not hash to the recorded digest", async () => {
    const frames = [flat(1), flat(2), flat(3)];
    const digests = await Promise.all(frames.map((f) => sha256Hex(f)));
    expect(await mismatchedFrames(frames, digests)).toEqual([]);
    expect(await mismatchedFrames(frames, [digests[0] ?? "", "0".repeat(64), digests[2] ?? ""])).toEqual([1]);
    expect(await mismatchedFrames(frames, digests.slice(0, 2))).toEqual([2]); // a frame with no digest
  });
});

describe("decodeSprites", () => {
  const clip = {
    lead_in: 2,
    n: 3,
    sprite: { files: ["sprite-00.jpg", "sprite-01.jpg"], frames_per_sheet: 3, frame_w: 2, frame_h: 1 },
  };

  /** Sheets whose every pixel of frame f (counted across sheets) is (f, f, f). */
  function sheets(): Map<string, SheetPixels> {
    const make = (first: number, count: number): SheetPixels => {
      const data = new Uint8ClampedArray(2 * count * 4);
      for (let row = 0; row < count; row += 1) {
        for (let x = 0; x < 2; x += 1) data.set([first + row, first + row, first + row, 255], (row * 2 + x) * 4);
      }
      return { data, width: 2, height: count };
    };
    return new Map([
      ["base/sprite-00.jpg", make(0, 3)],
      ["base/sprite-01.jpg", make(3, 2)],
    ]);
  }

  it("decodes every sheet, in order, into lead_in + n frames", async () => {
    const loaded = sheets();
    const urls: string[] = [];
    const frames = await decodeSprites(clip, "base/", async (url) => {
      urls.push(url);
      const sheet = loaded.get(url);
      if (!sheet) throw new Error(`no ${url}`);
      return sheet;
    });
    expect(urls).toEqual(["base/sprite-00.jpg", "base/sprite-01.jpg"]);
    expect(frames.map((f) => f[0])).toEqual([0, 1, 2, 3, 4]);
    expect(frames.every((f) => f.length === 2 * 1 * 3)).toBe(true);
  });

  it("refuses sheets holding fewer frames than the clip needs", async () => {
    const loaded = sheets();
    const short = { ...clip, n: 4 };
    await expect(decodeSprites(short, "base/", async (url) => loaded.get(url) as SheetPixels)).rejects.toThrow(
      /6 frames/,
    );
  });
});
