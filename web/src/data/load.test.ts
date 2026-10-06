import { afterEach, describe, expect, it, vi } from "vitest";

import { AssetError, fetchJson, fetchOk, parseClip, parseResults } from "./load.ts";

/** A small clip.json as the site-data builder writes it: 3 scored frames, lag 2. */
function clipJson(overrides: Record<string, unknown> = {}) {
  return {
    id: "c2018",
    drive: "drive2018",
    title: "2018: minutes it never trained on",
    fps: 17.5,
    n: 3,
    lead_in: 2,
    first_frame: 120,
    frame_lags: [2],
    rule: "the 3-frame window ...",
    labels_deg: [1.5, -2.25, 30],
    expected: { sightline: [1, -2, 28], shuffled: [1.4, -2.2, 29.5] },
    sprite: {
      files: ["sprite-00.jpg"],
      frames_per_sheet: 60,
      frame_w: 200,
      frame_h: 66,
      decoded_sha256: Array.from({ length: 5 }, (_, i) => String(i).repeat(64)),
    },
    video: "clip.mp4",
    video_size: [456, 256],
    camera_size: [455, 256],
    clip_mae_deg: { sightline: 1.08, shuffled: 0.3, zero: 11.25 },
    ...overrides,
  };
}

describe("parseClip", () => {
  it("accepts a clip as the builder writes it", () => {
    const clip = parseClip(clipJson(), "c2018/clip.json");
    expect(clip.n).toBe(3);
    expect(clip.expected.shuffled).toEqual([1.4, -2.2, 29.5]);
    expect(clip.camera_size).toEqual([455, 256]);
    expect(clip.twin_trained_frac).toBeUndefined();
  });

  it("reads how much of the clip the twin trained on, when the clip says", () => {
    const clip = parseClip(
      clipJson({ twin_trained_frac: 0.7, twin_neighbour_frac: 0.96, tolerance: { angle_deg: 0.001 } }),
      "c2018/clip.json",
    );
    expect([clip.twin_trained_frac, clip.twin_neighbour_frac]).toEqual([0.7, 0.96]);
    expect(clip.tolerance?.angle_deg).toBe(0.001);
  });

  it("names the key that is missing or malformed", () => {
    const { labels_deg: _, ...noLabels } = clipJson();
    expect(() => parseClip(noLabels, "x/clip.json")).toThrow(/labels_deg/);
    expect(() => parseClip(clipJson({ labels_deg: [1, 2] }), "x/clip.json")).toThrow(/labels_deg.*3/);
    expect(() => parseClip(clipJson({ expected: { sightline: [1, 2, 3] } }), "x/clip.json")).toThrow(/shuffled/);
    expect(() => parseClip(clipJson({ fps: 0 }), "x/clip.json")).toThrow(/fps/);
    expect(() => parseClip(clipJson({ twin_trained_frac: 1.5 }), "x/clip.json")).toThrow(/twin_trained_frac/);
    expect(() => parseClip(null, "x/clip.json")).toThrow(/x\/clip\.json/);
  });

  it("refuses history the lead-in frames cannot supply", () => {
    expect(() => parseClip(clipJson({ frame_lags: [3] }), "x/clip.json")).toThrow(/lead_in/);
    expect(() => parseClip(clipJson({ frame_lags: [0] }), "x/clip.json")).toThrow(/frame_lags/);
  });

  it("needs a recorded digest for every sprite frame", () => {
    const sprite = { ...clipJson().sprite, decoded_sha256: ["a".repeat(64)] };
    expect(() => parseClip(clipJson({ sprite }), "x/clip.json")).toThrow(/decoded_sha256.*5/);
  });

  it("reports a bad clip as invalid data, not as a missing file", () => {
    try {
      parseClip(clipJson({ n: -1 }), "x/clip.json");
      expect.unreachable();
    } catch (error) {
      expect(error).toBeInstanceOf(AssetError);
      expect((error as AssetError).kind).toBe("invalid");
    }
  });
});

describe("parseResults", () => {
  const results = {
    assets: {
      models: { sightline: "models/sightline.onnx", shuffled: "models/sightline-shuffled.onnx" },
      clips: { c2017: "data/clips/c2017/clip.json", c2018: "data/clips/c2018/clip.json" },
    },
    shipped: { run: "drive2018_sightlinenet_blocked_s0", model: "sightlinenet", protocol: "blocked" },
    twin: { run: "drive2018_sightlinenet_leaky_s0" },
    ladder: {},
  };

  it("reads the files the page loads and the two runs", () => {
    const parsed = parseResults(results, "data/results.json");
    expect(parsed.assets.models.shuffled).toBe("models/sightline-shuffled.onnx");
    expect(Object.keys(parsed.assets.clips)).toEqual(["c2017", "c2018"]);
    expect(parsed.shipped.run).toBe("drive2018_sightlinenet_blocked_s0");
    expect(parsed.raw["ladder"]).toEqual({});
  });

  it("names what is missing", () => {
    expect(() => parseResults({ ...results, assets: { clips: {} } }, "r.json")).toThrow(/assets\.models/);
    expect(() => parseResults({ ...results, twin: {} }, "r.json")).toThrow(/twin\.run/);
  });
});

describe("fetching", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("tells a missing file from an unreachable one", async () => {
    vi.stubGlobal("fetch", async () => new Response("", { status: 404 }));
    await expect(fetchOk("data/results.json")).rejects.toMatchObject({ kind: "missing", url: "data/results.json" });
    vi.stubGlobal("fetch", async () => new Response("", { status: 503 }));
    await expect(fetchOk("data/results.json")).rejects.toMatchObject({ kind: "unreachable" });
    vi.stubGlobal("fetch", async () => {
      throw new TypeError("Failed to fetch");
    });
    await expect(fetchOk("data/results.json")).rejects.toMatchObject({ kind: "unreachable" });
  });

  it("reports a file that is not JSON as invalid", async () => {
    vi.stubGlobal("fetch", async () => new Response("{ not json", { status: 200 }));
    await expect(fetchJson("data/results.json")).rejects.toMatchObject({ kind: "invalid" });
    vi.stubGlobal("fetch", async () => new Response('{"a": 1}', { status: 200 }));
    await expect(fetchJson("data/results.json")).resolves.toEqual({ a: 1 });
  });

  it("counts a web page served in place of the data as missing", async () => {
    // Servers that answer every unknown path with the site's index page.
    vi.stubGlobal("fetch", async () =>
      new Response("<!doctype html><html></html>", { status: 200, headers: { "content-type": "text/html" } }),
    );
    await expect(fetchJson("data/results.json")).rejects.toMatchObject({ kind: "missing" });
  });
});
