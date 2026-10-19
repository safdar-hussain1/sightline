import { describe, expect, it } from "vitest";

import type { Agreement } from "../cockpit/live.ts";
import { AssetError } from "../data/load.ts";
import type { Clip } from "../data/types.ts";
import {
  accuracyCaption,
  accuracyLabel,
  accuracySentence,
  anotherDayFigure,
  anotherDayNote,
  anotherDayRuler,
  framesAges,
  framesText,
  framesTitle,
  leakageSentence,
  measurementsFailure,
  memoriserNote,
  pairedNote,
  pairedRuler,
  parityText,
  resultsLede,
  splitsText,
  speedText,
  timesInWords,
} from "./copy.ts";
import { readFigures } from "./figures.ts";
import { patch, resultsFixture } from "./fixture.ts";

const figures = () => readFigures(resultsFixture());

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
    ...overrides,
  };
}

describe("the accuracy figure's sentences", () => {
  it("says how far off Sightline is on minutes it never trained on, against steering straight", () => {
    expect(accuracySentence(figures().heldOut)).toBe(
      "On minutes of driving it never trained on, Sightline is off by 10.1° on average. Always steering straight is off by 17.5° on the same frames.",
    );
  });

  it("names the frames the figure is measured on", () => {
    expect(accuracyCaption(figures().heldOut)).toBe("Mean error on the 6,000 frames its training held back.");
    expect(accuracyLabel(figures().heldOut)).toBe(
      "Mean error on the same 6,000 frames: Sightline 10.1°, always steering straight 17.5°.",
    );
  });

  it("says how much better a shuffled test makes the same kind of model look, in words", () => {
    expect(leakageSentence(figures().paired.sightlinenet.ratio.mean)).toBe(
      "Test it the usual way, on shuffled frames, and the same kind of model looks about three times better than it is on road it has not seen nearby.",
    );
  });

  it("rounds the ratio to the nearest half, and never claims a flattery the data does not show", () => {
    expect(timesInWords(3.0466)).toBe("three times");
    expect(timesInWords(3.1666)).toBe("three times");
    expect(timesInWords(2.6)).toBe("two and a half times");
    expect(timesInWords(1.3)).toBe("one and a half times");
    expect(timesInWords(14.2)).toBe("14 times");
    expect(leakageSentence(1.1)).toBe(
      "Test it the usual way, on shuffled frames, and the same kind of model looks a little better than it is on road it has not seen nearby.",
    );
    expect(leakageSentence(1)).toBe(
      "Tested the usual way, on shuffled frames, the same kind of model looks no better than it is on road it has not seen nearby.",
    );
    expect(leakageSentence(0.8)).toBe(leakageSentence(1));
  });
});

describe("when the measurements cannot be had", () => {
  it("says why in plain words", () => {
    expect(measurementsFailure(new AssetError("x", "missing", "data/results.json"))).toBe(
      "The measurements are not on this copy of the site.",
    );
    expect(measurementsFailure(new AssetError("x", "unreachable", "data/results.json"))).toBe(
      "The measurements could not be downloaded. Check the connection, then reload the page.",
    );
    expect(measurementsFailure(new AssetError("x", "invalid", "data/results.json"))).toBe(
      "The measurements file is not in the form this page reads, so its figures are not shown.",
    );
    expect(measurementsFailure(new Error("anything else"))).toBe(measurementsFailure(new AssetError("x", "invalid", "u")));
  });
});

describe("the Results view's figures", () => {
  it("counts the training runs behind them", () => {
    expect(resultsLede(figures().runs)).toBe(
      "How good a steering model looks depends on how it is tested. These are the headline numbers from 21 training runs.",
    );
  });

  it("explains the paired ratio's source and its spread over seeds", () => {
    expect(pairedNote(figures().paired)).toBe(
      "Mean of three trainings each way, each pair scored on the frames both of its splits held out. Spread ±0.6 for Sightline, ±0.1 for PilotNet.",
    );
  });

  it("states another day's skill with the two errors it comes from", () => {
    expect(anotherDayFigure(figures().anotherDay)).toEqual({
      value: 0.3145251314711981,
      text: "31%",
      label: "Better than always steering straight, on a drive it never saw",
    });
    expect(anotherDayNote(figures().anotherDay)).toBe(
      "Mean of three trainings, each scored against steering straight on the same frames of another day’s drive.",
    );
  });

  it("never calls a model better than steering straight when it is worse", () => {
    const json = patch(resultsFixture(), "groups.sightlinenet/blocked.cross_skill", { mean: -0.2423, sd: 0.05 });
    expect(anotherDayFigure(readFigures(json).anotherDay)).toEqual({
      value: 0.2423,
      text: "24%",
      label: "Worse than always steering straight, on a drive it never saw",
    });
  });

  it("describes the two errors behind each summary figure, from the same numbers the rulers draw", () => {
    expect(pairedRuler(figures().paired.sightlinenet)).toBe(
      "Sightline’s mean error on the frames both splits held out: 3.3° when trained on the shuffled split, 10.1° when trained on the blocked split.",
    );
    expect(anotherDayRuler(figures().anotherDay)).toBe(
      "Mean error on another day’s drive: Sightline 10.6°, always steering straight 15.5°.",
    );
  });

  it("explains the lookup table's score", () => {
    expect(memoriserNote(figures().memoriser)).toBe(
      "On 893 held-out frames. It learns nothing, so its small error on the shuffled split only shows how close the test frames sit to training frames.",
    );
  });

  it("says how close test frames sit to training frames under each split", () => {
    expect(splitsText(figures().gaps)).toEqual({
      shuffled: "Frames are dealt out at random, so 91% of test frames sit within one frame of a frame the model trained on.",
      blocked: "Whole stretches of road are held back, so no test frame comes closer than 501 frames to one the model trained on.",
    });
  });
});

describe("the live check against Python", () => {
  const agreement = (overrides: Partial<Agreement> = {}): Agreement => ({
    clip: clip(),
    model: "sightline",
    check: "match",
    compared: 420,
    total: 420,
    maxDiffDeg: 0.0000191,
    ...overrides,
  });

  it("states the largest difference once every frame is checked", () => {
    expect(parityText(agreement(), "ready")).toBe(
      "Checked live: on all 420 frames of the demo’s 2018 clip, this browser’s angles match the model run in Python to within 0.00002°.",
    );
  });

  it("says exactly when there is no difference at all", () => {
    expect(parityText(agreement({ maxDiffDeg: 0 }), "ready")).toBe(
      "Checked live: on all 420 frames of the demo’s 2018 clip, this browser’s angles match the model run in Python exactly.",
    );
  });

  it("states a bound, not a made-up digit, for a difference below what it prints", () => {
    expect(parityText(agreement({ maxDiffDeg: 2e-9 }), "ready")).toBe(
      "Checked live: on all 420 frames of the demo’s 2018 clip, this browser’s angles match the model run in Python to within 0.000001°.",
    );
  });

  it("counts the frames checked so far while the model is still running", () => {
    expect(parityText(agreement({ compared: 96 }), "ready")).toBe(
      "Checking live: 96 of 420 frames of the demo’s 2018 clip so far, every one within 0.00002° of the model run in Python.",
    );
    expect(parityText(agreement({ compared: 96, maxDiffDeg: 0 }), "ready")).toBe(
      "Checking live: 96 of 420 frames of the demo’s 2018 clip so far, every one identical to the model run in Python.",
    );
  });

  it("does not compare runtimes on frames this browser decoded differently", () => {
    expect(parityText(agreement({ check: "differ", maxDiffDeg: 0.4123 }), "ready")).toBe(
      "This browser decodes the demo’s 2018 clip a little differently, so its angles differ from the model run in Python by up to 0.41° on the 420 frames it has run.",
    );
    expect(parityText(agreement({ check: "unverified" }), "ready")).toBe(
      "This browser could not check its decoded frames, so its angles are not compared with the model run in Python.",
    );
  });

  it("never states a difference that is not a number", () => {
    expect(parityText(agreement({ maxDiffDeg: Number.POSITIVE_INFINITY }), "ready")).toBe(
      "This browser’s angles on the demo’s 2018 clip do not match the model run in Python.",
    );
  });

  it("says what it is waiting for, and why there is no check when the demo cannot run", () => {
    expect(parityText(null, "loading")).toBe("Checking this browser against the model run in Python…");
    expect(parityText(agreement({ compared: 0, maxDiffDeg: Number.NaN }), "loading")).toBe(
      "Checking this browser against the model run in Python…",
    );
    expect(parityText(null, "unavailable")).toBe("The live check needs the demo, which could not run here.");
  });
});

describe("how it works", () => {
  it("names the frames the model is shown from the clip's lags and frame rate", () => {
    expect(framesTitle(clip())).toBe("Two frames, 114 ms apart");
    expect(framesAges(clip())).toEqual(["now", "114 ms before"]);
    expect(framesText(clip())).toBe(
      "The frame on screen and the one 114 ms before it, cropped to the road and shrunk to 200 × 66 pixels. Two frames show which way the view is moving, which one frame cannot.",
    );
  });

  it("follows a clip with more history, or none", () => {
    const three = clip({ frame_lags: [1, 2], lead_in: 2 });
    expect(framesTitle(three)).toBe("Three frames, up to 114 ms apart");
    expect(framesAges(three)).toEqual(["now", "57 ms before", "114 ms before"]);
    const one = clip({ frame_lags: [] });
    expect(framesTitle(one)).toBe("One frame at a time");
    expect(framesAges(one)).toEqual(["now"]);
    expect(framesText(one)).toBe("The frame on screen, cropped to the road and shrunk to 200 × 66 pixels.");
  });

  it("states how long the model takes here only once it has been measured", () => {
    expect(speedText(2.5312)).toBe("Here it takes about 2.5 ms a frame.");
    expect(speedText(null)).toBe("");
  });
});
