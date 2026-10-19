import { describe, expect, it } from "vitest";

import { AssetError } from "../data/load.ts";
import type { Clip } from "../data/types.ts";
import { CANNOT_RUN } from "./boot.ts";
import { TWIN_UNAVAILABLE, decodeNote, eyeCaption, failureMessage, modelNote, splitTitle } from "./copy.ts";

/** The parts of a clip the copy reads. */
function clip(overrides: Partial<Clip> = {}): Clip {
  return {
    id: "c2018",
    drive: "drive2018",
    title: "2018: minutes it never trained on",
    fps: 17.5,
    n: 420,
    lead_in: 2,
    first_frame: 28752,
    frame_lags: [2],
    rule: "",
    labels_deg: [],
    expected: { sightline: [], shuffled: [] },
    sprite: { files: [], frames_per_sheet: 60, frame_w: 200, frame_h: 66, decoded_sha256: [] },
    video: "clip.mp4",
    video_size: [456, 256],
    camera_size: [455, 256],
    clip_mae_deg: { sightline: 14.26, shuffled: 5.63, zero: 46.7 },
    twin_trained_frac: 0.7095,
    twin_neighbour_frac: 0.9714,
    ...overrides,
  };
}

describe("modelNote", () => {
  it("says what Sightline is, whatever the clip", () => {
    expect(modelNote("sightline", clip())).toBe(modelNote("sightline", null));
    expect(modelNote("sightline", null)).toMatch(/held back/);
  });

  it("explains a twin that looks better here by the frames it trained on, from the clip's own numbers", () => {
    expect(modelNote("shuffled", clip())).toBe(
      "The same network, trained on shuffled frames. It looks better on this clip because it trained on 71% of these very frames, and 97% sit within one frame of one it trained on.",
    );
  });

  it("does not say the twin looks better on a clip where it does not", () => {
    const note = modelNote("shuffled", clip({ clip_mae_deg: { sightline: 5, shuffled: 6, zero: 40 } }));
    expect(note).not.toMatch(/better/);
    expect(note).toMatch(/71% of these very frames/);
  });

  it("says so when the twin never trained on the clip", () => {
    expect(
      modelNote("shuffled", clip({ twin_trained_frac: 0, twin_neighbour_frac: 0, clip_mae_deg: { sightline: 10.78, shuffled: 14.17, zero: 42.1 } })),
    ).toBe("The same network, trained on shuffled frames. It never trained on any frame of this clip.");
  });

  it("claims nothing about the clip when the clip file does not say", () => {
    const { twin_trained_frac: _, twin_neighbour_frac: __, ...older } = clip();
    expect(modelNote("shuffled", older)).toBe("The same network, trained on shuffled frames.");
  });
});

describe("splitTitle", () => {
  it("splits a clip title into its year and its description", () => {
    expect(splitTitle("2018: minutes it never trained on")).toEqual(["2018", "minutes it never trained on"]);
    expect(splitTitle("2017: another day")).toEqual(["2017", "another day"]);
  });

  it("treats a title without a colon as all description", () => {
    expect(splitTitle("a clip")).toEqual(["", "a clip"]);
  });
});

describe("eyeCaption", () => {
  const timed = { interval: 0.056999921798706055, reference: null };
  const untimed = { interval: null, reference: { drive: "drive2018", interval: 0.057 } };

  it("states the input's size and the history frame's age, from the drive's own timestamps", () => {
    expect(eyeCaption(clip(), timed)).toBe(
      "What the model gets: 200 × 66 pixels of this frame and of the one 114 ms before it, with where it looked.",
    );
  });

  it("counts frames, not milliseconds, on a drive without timestamps", () => {
    expect(eyeCaption(clip({ drive: "drive2017" }), untimed)).toBe(
      "What the model gets: 200 × 66 pixels of this frame and of the one two frames before it, with where it looked.",
    );
  });

  it("follows the clip's own lags", () => {
    expect(eyeCaption(clip({ frame_lags: [1, 3] }), { interval: 0.1, reference: null })).toMatch(
      /the ones 100 ms and 300 ms before it/,
    );
    expect(eyeCaption(clip({ frame_lags: [] }), timed)).toBe(
      "What the model gets: 200 × 66 pixels of this frame, with where it looked.",
    );
  });
});

describe("failureMessage", () => {
  it("says in plain words why the demo cannot run, by what went wrong", () => {
    expect(failureMessage(new AssetError("x", "missing", "u"))).toMatch(/not on this copy of the site/);
    expect(failureMessage(new AssetError("x", "unreachable", "u"))).toMatch(/could not download.*reload/);
    expect(failureMessage(new AssetError("x", "invalid", "u"))).toMatch(/not in the form this page reads/);
    expect(failureMessage(new AssetError("x", "undecodable", "u"))).toMatch(/could not decode the clip/);
    expect(failureMessage(new AssetError("x", "stopped", "u"))).toBe(
      "The clip’s footage stopped loading. Pick the clip again to reload it.",
    );
    expect(failureMessage(new Error("no available backend found"))).toMatch(/could not run the model/);
    expect(failureMessage(new Error("no available backend found"))).toBe(CANNOT_RUN);
  });

  it("never shows a URL or a stack of technical detail", () => {
    for (const error of [
      new AssetError("could not reach http://127.0.0.1/data/results.json: Failed to fetch", "unreachable", "http://127.0.0.1/data/results.json"),
      new Error("WebAssembly SIMD is not supported in the current environment."),
      "a string",
    ]) {
      const message = failureMessage(error);
      expect(message).not.toMatch(/https?:|Failed to fetch|WebAssembly SIMD/);
      expect(message.endsWith(".")).toBe(true);
    }
  });
});

describe("decodeNote", () => {
  it("says nothing when every frame is the reference decoder's", () => {
    expect(decodeNote({ status: "match" }, 422)).toBe("");
  });

  it("counts the frames this browser decodes differently, from the check itself", () => {
    expect(decodeNote({ status: "differ", frames: [3, 9, 10] }, 422)).toBe(
      "This browser decodes 3 of the clip’s 422 frames slightly differently from the reference decoder, so the model’s angles here can differ a little from the published ones.",
    );
  });

  it("never claims a match it could not check", () => {
    const note = decodeNote({ status: "unverified" }, 422);
    expect(note).toMatch(/could not check/);
    expect(note).toMatch(/may differ a little from the published ones\.$/);
  });
});

describe("TWIN_UNAVAILABLE", () => {
  it("says what the visitor now sees and what to do, in one sentence", () => {
    expect(TWIN_UNAVAILABLE).toMatch(/^The twin could not be loaded, so Sightline is shown instead;/);
    expect(TWIN_UNAVAILABLE.match(/\./g)).toHaveLength(1);
  });
});
