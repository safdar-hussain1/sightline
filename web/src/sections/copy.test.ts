import { describe, expect, it } from "vitest";

import type { Agreement } from "../cockpit/live.ts";
import { AssetError } from "../data/load.ts";
import { frameTiming } from "../data/timing.ts";
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
  pairedLabel,
  pairedNote,
  pairedRuler,
  parityState,
  resultsLede,
  splitsText,
  speedText,
  timesInWords,
  upperBound,
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
    tolerance: { angle_deg: 0.001 },
    ...overrides,
  };
}

/** The drives as results.json records them: 2018 with timestamps, 2017 without. */
const INTERVALS = { drive2018: 0.056999921798706055, drive2017: null };

describe("the accuracy figure's sentences", () => {
  it("says how far off Sightline is on minutes it never trained on, against steering straight", () => {
    expect(accuracySentence(figures().heldOut)).toBe(
      "On minutes of driving it never trained on, Sightline is off by 10.1° on average. Always steering straight is off by 17.5° on the same frames.",
    );
  });

  it("names the frames the figure is measured on: the test frames, not everything the split held back", () => {
    expect(accuracyCaption(figures().heldOut)).toBe("Mean error on the 6,000 test frames held back from its training.");
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

  it("puts a ratio in words from 1.25 up, and calls anything between one and 1.25 a little", () => {
    expect(leakageSentence(1.24)).toMatch(/looks a little better/);
    expect(leakageSentence(1.25)).toMatch(/looks about one and a half times better/);
    expect(leakageSentence(1.3)).toMatch(/looks about one and a half times better/);
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
  it("says what the training runs found, true of the whole view", () => {
    expect(resultsLede(figures().runs)).toBe(
      "How good a steering model looks depends on how it is tested. Here is what 21 training runs found.",
    );
  });

  it("labels the paired ratio by what it shows, and only calls it flattery above one", () => {
    expect(pairedLabel(3.0466)).toBe("How much better testing on shuffled frames makes the same kind of model look");
    for (const ratio of [1, 0.8]) {
      expect(pairedLabel(ratio), String(ratio)).toBe(
        "The same kind of model’s error on held-back stretches, as a multiple of its error under a shuffled test",
      );
    }
  });

  it("explains the paired ratio's source and its spread over seeds", () => {
    expect(pairedNote(figures().paired)).toBe(
      "Mean of three trainings each way, each pair scored on the frames both of its splits held out. Spread ±0.6 for Sightline, ±0.1 for PilotNet.",
    );
  });

  it("states another day's skill, a mean of three trainings", () => {
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
      "Sightline’s mean error over three trainings on the frames both splits held out: 3.3° when trained on the shuffled split, 10.1° when trained on the blocked split.",
    );
    expect(anotherDayRuler(figures().anotherDay)).toBe(
      "Mean error over three trainings on another day’s drive: Sightline 10.6°, always steering straight 15.5°.",
    );
  });

  it("explains the lookup table's score without judging its size", () => {
    expect(memoriserNote(figures().memoriser)).toBe(
      "On 893 held-out frames. It learns nothing, so its error under each split only shows how close the test frames sit to training frames.",
    );
  });

  it("says how close test frames sit to training frames under each split", () => {
    expect(splitsText(figures().gaps)).toEqual({
      shuffled: "Frames are dealt out at random, so 91% of test frames sit within one frame of a frame the model trained on.",
      blocked: "Whole stretches of road are held back, so no test frame comes closer than 501 frames to one the model trained on.",
    });
  });
});

describe("upperBound", () => {
  it("rounds a difference up, never to nearest, at one significant figure", () => {
    expect(upperBound(0.0000249)).toBe("0.00003°");
    expect(upperBound(0.000030517578125)).toBe("0.00004°");
    expect(upperBound(0.0000381)).toBe("0.00004°");
    expect(upperBound(0.0004)).toBe("0.0004°");
    expect(upperBound(0.00099)).toBe("0.001°");
    expect(upperBound(0.5)).toBe("0.5°");
    expect(upperBound(124.8157)).toBe("124.9°");
  });

  it("states a floor of a millionth of a degree for anything smaller", () => {
    expect(upperBound(2e-9)).toBe("0.000001°");
  });

  it("is never below the difference it bounds", () => {
    for (let exponent = -6; exponent <= 2; exponent += 1) {
      for (const mantissa of [1, 1.0000001, 2.49, 3.0517578125, 4.9999, 9.99]) {
        const diff = mantissa * 10 ** exponent;
        expect(Number(upperBound(diff).replace("°", "")), String(diff)).toBeGreaterThanOrEqual(diff);
      }
    }
  });
});

describe("the live check against Python", () => {
  const agreement = (overrides: Partial<Agreement> = {}): Agreement => ({
    clip: clip(),
    model: "sightline",
    check: "match",
    compared: 420,
    total: 420,
    maxDiffDeg: 0.000030517578125,
    ...overrides,
  });

  it("passes when every frame is within the clip's tolerance, with a bound rounded up", () => {
    expect(parityState(agreement(), "ready")).toEqual({
      look: "checked",
      text: "Checked live: on all 420 frames of the demo’s 2018 clip, Sightline’s angles in this browser match the same model run in Python to within 0.00004°.",
    });
  });

  it("names the model it checked", () => {
    expect(parityState(agreement({ model: "shuffled" }), "ready").text).toMatch(/, the twin’s angles in this browser match/);
  });

  it("says exactly when there is no difference at all", () => {
    expect(parityState(agreement({ maxDiffDeg: 0 }), "ready").text).toBe(
      "Checked live: on all 420 frames of the demo’s 2018 clip, Sightline’s angles in this browser match the same model run in Python exactly.",
    );
  });

  it("fails as soon as a frame differs by more than the clip's tolerance", () => {
    expect(parityState(agreement({ maxDiffDeg: 124.8157 }), "ready")).toEqual({
      look: "mismatch",
      text: "Sightline’s angles in this browser do not match the same model run in Python: on the demo’s 2018 clip they differ by up to 124.9°, more than the 0.001° allowed.",
    });
    expect(parityState(agreement({ maxDiffDeg: 0.0011, compared: 12 }), "ready").look).toBe("mismatch");
  });

  it("passes right at the tolerance, and fails just above it", () => {
    expect(parityState(agreement({ maxDiffDeg: 0.001 }), "ready").look).toBe("checked");
    expect(parityState(agreement({ maxDiffDeg: 0.0010001 }), "ready").look).toBe("mismatch");
  });

  it("calls nothing a match when the clip records no tolerance", () => {
    const { tolerance: _, ...bare } = clip();
    expect(parityState(agreement({ clip: bare }), "ready")).toEqual({
      look: "note",
      text: "Sightline’s angles in this browser differ from the same model run in Python by at most 0.00004° on the 420 frames it has run; the clip records no tolerance to judge that by.",
    });
  });

  it("counts the frames checked so far while the model is still running", () => {
    expect(parityState(agreement({ compared: 96 }), "ready")).toEqual({
      look: "checking",
      text: "Checking live: 96 of 420 frames of the demo’s 2018 clip so far, every one within 0.00004° of the same model run in Python.",
    });
    expect(parityState(agreement({ compared: 96, maxDiffDeg: 0 }), "ready").text).toBe(
      "Checking live: 96 of 420 frames of the demo’s 2018 clip so far, every one identical to the same model run in Python.",
    );
  });

  it("never passes frames this browser decoded differently, or could not check", () => {
    expect(parityState(agreement({ check: "differ", maxDiffDeg: 0.4123 }), "ready")).toEqual({
      look: "note",
      text: "This browser decodes the demo’s 2018 clip differently from Python, so Sightline’s angles here differ from the same model run in Python by up to 0.5° on the 420 frames it has run.",
    });
    expect(parityState(agreement({ check: "differ", maxDiffDeg: 0 }), "ready").look).toBe("note");
    expect(parityState(agreement({ check: "unverified", maxDiffDeg: 0 }), "ready")).toEqual({
      look: "note",
      text: "This browser could not check that it decoded the clip’s frames as Python did, so its angles are not compared.",
    });
  });

  it("never states a difference that is not a number", () => {
    expect(parityState(agreement({ maxDiffDeg: Number.POSITIVE_INFINITY }), "ready")).toEqual({
      look: "mismatch",
      text: "Sightline’s angles in this browser do not match the same model run in Python on the demo’s 2018 clip.",
    });
  });

  it("says what it is waiting for, and why there is no check without a running demo", () => {
    const waiting = { look: "checking", text: "Checking this browser against the same model run in Python…" };
    expect(parityState(null, "loading")).toEqual(waiting);
    expect(parityState(agreement({ compared: 0, maxDiffDeg: Number.NaN }), "loading")).toEqual(waiting);
    expect(parityState(null, "unavailable")).toEqual({
      look: "note",
      text: "The live check needs the demo, and the demo is not running.",
    });
    expect(parityState(null, "absent")).toEqual({
      look: "note",
      text: "The live check needs the demo, which could not start in this browser.",
    });
  });
});

describe("how it works", () => {
  const timed = frameTiming(INTERVALS, "drive2018");
  const untimed = frameTiming(INTERVALS, "drive2017");

  it("names the frames the model is shown, timed from the drive's own timestamps", () => {
    expect(framesTitle(clip(), timed)).toBe("Two frames, 114 ms apart");
    expect(framesAges(clip(), timed)).toEqual(["now", "114 ms before"]);
    expect(framesText(clip(), timed)).toBe(
      "The frame on screen and the one 114 ms before it, cropped to the road and shrunk to 200 × 66 pixels. Two frames show which way the view is moving, which one frame cannot.",
    );
  });

  it("counts frames, not milliseconds, on a drive without timestamps, and quotes the other drive's measured spacing", () => {
    const another = clip({ id: "c2017", drive: "drive2017", title: "2017: another day" });
    expect(framesTitle(another, untimed)).toBe("This frame and the one two before it");
    expect(framesAges(another, untimed)).toEqual(["now", "two frames before"]);
    expect(framesText(another, untimed)).toBe(
      "The frame on screen and the one two frames before it, cropped to the road and shrunk to 200 × 66 pixels. This drive’s frames carry no timestamps; on the 2018 drive, whose frames do, two frames are 114 ms apart. Two frames show which way the view is moving, which one frame cannot.",
    );
    expect(framesText(another, frameTiming({ drive2017: null }, "drive2017"))).toBe(
      "The frame on screen and the one two frames before it, cropped to the road and shrunk to 200 × 66 pixels. This drive’s frames carry no timestamps, so their spacing is counted in frames. Two frames show which way the view is moving, which one frame cannot.",
    );
  });

  it("follows a clip with more history, or none", () => {
    const three = clip({ frame_lags: [1, 2], lead_in: 2 });
    expect(framesTitle(three, timed)).toBe("Three frames, up to 114 ms apart");
    expect(framesAges(three, timed)).toEqual(["now", "57 ms before", "114 ms before"]);
    expect(framesTitle(three, untimed)).toBe("This frame and the two before it");
    const one = clip({ frame_lags: [] });
    expect(framesTitle(one, timed)).toBe("One frame at a time");
    expect(framesAges(one, timed)).toEqual(["now"]);
    expect(framesText(one, timed)).toBe("The frame on screen, cropped to the road and shrunk to 200 × 66 pixels.");
  });

  it("states how long the model takes here only once it has been measured", () => {
    expect(speedText(2.5312)).toBe("Here it takes about 2.5 ms a frame.");
    expect(speedText(null)).toBe("");
  });
});
