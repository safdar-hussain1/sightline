import { describe, expect, it } from "vitest";

import type { Agreement } from "../cockpit/live.ts";
import { AssetError } from "../data/load.ts";
import { driveGaps, frameTiming } from "../data/timing.ts";
import type { Clip } from "../data/types.ts";
import {
  barWords,
  driveTitle,
  frameWords,
  framesAlt,
  framesTitleOf,
  YARDSTICK_LABELS,
  yardstickSetTitle,
  yardsticksAlt,
  yardsticksVerdict,
  ladderCaption,
  ladderLead,
  trainingsWord,
  turnWords,
  exactDeg,
  exportText,
  ladderAlt,
  legacyAlt,
  legacyText,
  pairedAlt,
  pairedFrames,
  pairedLead,
  pairedMeaning,
  pairedRatioLabel,
  pairedScale,
  patternWords,
  protocolTexts,
  recipeNote,
  regimeNames,
  regimesAlt,
  regimesText,
  rungCaption,
  rungNote,
  skillLabel,
  skillWords,
  twoFramesText,
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
import { readFigures, readFrames, readLadder, readLegacy, readPaired, readParity, readRegimes, readSplits, readYardsticks } from "./figures.ts";
import { patch, resultsFixture, setSkill } from "./fixture.ts";
import { networkName } from "./names.ts";

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

/** The drives' measured gaps as results.json records them: 2018 with timestamps, 2017 without. */
const GAPS = driveGaps(resultsFixture());

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
      "Mean of three trainings each way, each pair scored on the frames both of its splits held out. Spread ±0.6 for SightlineNet, ±0.1 for PilotNet.",
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
    const json = setSkill(resultsFixture(), "sightlinenet", "blocked", "cross_drive", -0.2423);
    const figure = anotherDayFigure(readFigures(json).anotherDay);
    expect(figure.value).toBeCloseTo(0.2423, 12);
    expect(figure).toMatchObject({ text: "24%", label: "Worse than always steering straight, on a drive it never saw" });
  });

  it("describes the two errors behind each summary figure, from the same numbers the rulers draw", () => {
    expect(pairedRuler(figures().paired.sightlinenet)).toBe(
      "SightlineNet’s mean error over three trainings on the frames both splits held out: 3.3° when trained on the shuffled split, 10.1° when trained on the blocked split.",
    );
    expect(anotherDayRuler(figures().anotherDay)).toBe(
      "Mean error over three trainings on another day’s drive: SightlineNet 10.6°, always steering straight 15.5°.",
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
  /** A state as the band shows it: the verdict it announces, the progress
   * line beside it, whether the check can still change, and both as read. */
  const state = (look: string, verdict: string, progress: string, done: boolean) => ({
    look,
    verdict,
    progress,
    done,
    text: [verdict, progress].filter((part) => part !== "").join(" "),
  });

  it("passes when every frame is within the clip's tolerance, with a bound rounded up", () => {
    const passed = "Checked live: on all 420 frames of the demo’s 2018 clip, Sightline’s angles in this browser match the same model run in Python to within 0.00004°.";
    expect(parityState(agreement(), "ready")).toEqual(state("checked", passed, "", true));
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
    expect(parityState(agreement({ maxDiffDeg: 124.8157 }), "ready")).toEqual(
      state(
        "mismatch",
        "Sightline’s angles in this browser do not match the same model run in Python: on the demo’s 2018 clip they differ by up to 124.9°, more than the 0.001° allowed.",
        "",
        true,
      ),
    );
    expect(parityState(agreement({ maxDiffDeg: 0.0011, compared: 12 }), "ready").look).toBe("mismatch");
  });

  it("passes right at the tolerance, and fails just above it", () => {
    expect(parityState(agreement({ maxDiffDeg: 0.001 }), "ready").look).toBe("checked");
    expect(parityState(agreement({ maxDiffDeg: 0.0010001 }), "ready").look).toBe("mismatch");
  });

  it("calls nothing a match when the clip records no tolerance", () => {
    const { tolerance: _, ...bare } = clip();
    expect(parityState(agreement({ clip: bare }), "ready")).toEqual(
      state(
        "note",
        "Sightline’s angles in this browser differ from the same model run in Python by at most 0.00004° on all 420 frames of the demo’s 2018 clip; the clip records no tolerance to judge that by.",
        "",
        true,
      ),
    );
    expect(parityState(agreement({ clip: bare, compared: 96 }), "ready")).toEqual(
      state(
        "note",
        "The clip records no tolerance to judge Sightline’s angles in this browser by, so they are not called a match.",
        "They differ from the same model run in Python by at most 0.00004° on the 96 frames run so far.",
        false,
      ),
    );
  });

  it("counts the frames checked so far while the model is still running, naming the model, and announces nothing yet", () => {
    expect(parityState(agreement({ compared: 96 }), "ready")).toEqual(
      state("checking", "", "Checking Sightline live: 96 of 420 frames of the demo’s 2018 clip so far, every one within 0.00004° of the same model run in Python.", false),
    );
    expect(parityState(agreement({ compared: 96, maxDiffDeg: 0 }), "ready").text).toBe(
      "Checking Sightline live: 96 of 420 frames of the demo’s 2018 clip so far, every one identical to the same model run in Python.",
    );
    expect(parityState(agreement({ compared: 96, model: "shuffled" }), "ready").text).toBe(
      "Checking the twin live: 96 of 420 frames of the demo’s 2018 clip so far, every one within 0.00004° of the same model run in Python.",
    );
  });

  it("announces a mismatch found while the model is still running in words that hold, and counts the frames so far beside it", () => {
    expect(parityState(agreement({ maxDiffDeg: 0.0012, compared: 12 }), "ready")).toEqual(
      state(
        "mismatch",
        "Sightline’s angles in this browser do not match the same model run in Python.",
        "On the 12 frames of the demo’s 2018 clip run so far, they differ by up to 0.002°, more than the 0.001° allowed.",
        false,
      ),
    );
    // More frames change the count, never the announced words.
    expect(parityState(agreement({ maxDiffDeg: 0.3, compared: 300 }), "ready").verdict).toBe(
      parityState(agreement({ maxDiffDeg: 0.0012, compared: 12 }), "ready").verdict,
    );
    expect(parityState(agreement({ maxDiffDeg: 0.0012, compared: 12, model: "shuffled" }), "ready").verdict).toBe(
      "The twin’s angles in this browser do not match the same model run in Python.",
    );
  });

  it("never passes frames this browser decoded differently, or could not check", () => {
    expect(parityState(agreement({ check: "differ", maxDiffDeg: 0.4123 }), "ready")).toEqual(
      state(
        "note",
        "This browser decodes the demo’s 2018 clip differently from Python, so Sightline’s angles here differ from the same model run in Python by up to 0.5° on all 420 frames.",
        "",
        true,
      ),
    );
    expect(parityState(agreement({ check: "differ", maxDiffDeg: 0.4123, compared: 96 }), "ready")).toEqual(
      state(
        "note",
        "This browser decodes the demo’s 2018 clip differently from Python, so Sightline’s angles here differ from the same model run in Python.",
        "They differ by up to 0.5° on the 96 frames run so far.",
        false,
      ),
    );
    expect(parityState(agreement({ check: "differ", maxDiffDeg: 0 }), "ready").look).toBe("note");
    expect(parityState(agreement({ check: "unverified", maxDiffDeg: 0 }), "ready")).toEqual(
      state("note", "This browser could not check that it decoded the clip’s frames as Python did, so its angles are not compared.", "", true),
    );
  });

  it("never states a difference that is not a number", () => {
    expect(parityState(agreement({ maxDiffDeg: Number.POSITIVE_INFINITY, compared: 12 }), "ready")).toEqual(
      state("mismatch", "Sightline’s angles in this browser do not match the same model run in Python on the demo’s 2018 clip.", "", true),
    );
  });

  it("says what it is waiting for, and why there is no check without a running demo", () => {
    expect(parityState(null, "loading")).toEqual(state("checking", "", "Checking this browser against the same model run in Python…", false));
    expect(parityState(agreement({ compared: 0, maxDiffDeg: Number.NaN }), "loading")).toEqual(
      state("checking", "", "Checking Sightline in this browser against the same model run in Python…", false),
    );
    expect(parityState(agreement({ compared: 0, maxDiffDeg: Number.NaN, model: "shuffled" }), "ready").text).toBe(
      "Checking the twin in this browser against the same model run in Python…",
    );
    expect(parityState(null, "unavailable")).toEqual(state("note", "The live check needs the demo, and the demo is not running.", "", true));
    expect(parityState(null, "absent")).toEqual(state("note", "The live check needs the demo, which could not start in this browser.", "", true));
  });
});

describe("how it works", () => {
  const timed = frameTiming(GAPS, "drive2018");
  const untimed = frameTiming(GAPS, "drive2017");

  it("names the frames the model is shown, with the drive's measured median gap said as the drive's", () => {
    expect(framesTitle(clip(), timed)).toBe("Two frames, a median 100\u00A0ms apart on the 2018 drive");
    expect(framesAges(clip(), timed)).toEqual(["now", "two frames before"]);
    expect(framesText(clip(), timed)).toBe(
      "The frame on screen and the one two frames before it, cropped to the road and shrunk to 200 × 66 pixels. Two frames show which way the view is moving, which one frame cannot.",
    );
  });

  it("counts frames, not milliseconds, on a drive without timestamps, and quotes the other drive's measured gap", () => {
    const another = clip({ id: "c2017", drive: "drive2017", title: "2017: another day" });
    expect(framesTitle(another, untimed)).toBe("This frame and the one two frames before it");
    expect(framesAges(another, untimed)).toEqual(["now", "two frames before"]);
    expect(framesText(another, untimed)).toBe(
      "The frame on screen and the one two frames before it, cropped to the road and shrunk to 200 × 66 pixels. This drive’s frames carry no timestamps; on the 2018 drive, whose frames do, a frame and the one two before it were taken a median 100\u00A0ms apart. Two frames show which way the view is moving, which one frame cannot.",
    );
    expect(framesText(another, frameTiming({ drive2017: null }, "drive2017"))).toBe(
      "The frame on screen and the one two frames before it, cropped to the road and shrunk to 200 × 66 pixels. This drive’s frames carry no timestamps, so their spacing is counted in frames. Two frames show which way the view is moving, which one frame cannot.",
    );
  });

  it("follows a clip with more history, or none", () => {
    const three = clip({ frame_lags: [1, 2], lead_in: 2 });
    expect(framesTitle(three, timed)).toBe("Three frames, the oldest a median 100\u00A0ms earlier on the 2018 drive");
    expect(framesAges(three, timed)).toEqual(["now", "one frame before", "two frames before"]);
    expect(framesTitle(three, untimed)).toBe("This frame and the two before it");
    // A lag the drive's timing does not measure gets no time.
    expect(framesTitle(clip({ frame_lags: [3] }), timed)).toBe("This frame and the one three frames before it");
    const one = clip({ frame_lags: [] });
    expect(framesTitle(one, timed)).toBe("One frame at a time");
    expect(framesAges(one, timed)).toEqual(["now"]);
    expect(framesText(one, timed)).toBe("The frame on screen, cropped to the road and shrunk to 200 × 66 pixels.");
  });

  it("states how long the model takes here only once it has been measured", () => {
    expect(speedText(2.5312)).toBe("Here it takes about 2.5\u00A0ms a frame.");
    expect(speedText(null)).toBe("");
  });
});

describe("the names and small words of the Results view", () => {
  it("names a drive by its year", () => {
    expect(driveTitle("drive2018")).toBe("the 2018 drive");
    expect(driveTitle("highway")).toBe("the drive highway");
  });

  it("prints an angle the data states exactly with only the decimals it has", () => {
    expect(exactDeg(5)).toBe("5°");
    expect(exactDeg(30)).toBe("30°");
    expect(exactDeg(2.5)).toBe("2.5°");
    expect(exactDeg(180)).toBe("180°");
  });

  it("says a skill as better, worse or level, never a negative better", () => {
    expect(skillWords(0.3145251314711981)).toBe("31% better than always steering straight");
    expect(skillWords(-0.4485684827442407)).toBe("45% worse than always steering straight");
    expect(skillWords(0)).toBe("level with always steering straight");
    expect(skillWords(0.0004, "it")).toBe("0.04% better than it");
    expect(skillLabel(-0.24229235843871808)).toBe("−24%");
    expect(skillLabel(0.7788633029556049)).toBe("78%");
  });
});

describe("the paired experiment's sentences", () => {
  const paired = () => readPaired(resultsFixture());

  it("says what each network scored, trained each way, and what the ratio means", () => {
    expect(pairedLead("SightlineNet", paired().sightlinenet)).toBe(
      "On the frames both splits held out, SightlineNet trained on shuffled frames is off by 3.3° on average. Trained with whole stretches held back, the same network is off by 10.1°.",
    );
    expect(pairedLead("PilotNet", paired().pilotnet)).toBe(
      "On the frames both splits held out, PilotNet trained on shuffled frames is off by 4.1° on average. Trained with whole stretches held back, the same network is off by 13.1°.",
    );
    expect(pairedMeaning("SightlineNet", paired().sightlinenet.ratio.mean)).toBe(
      "So a shuffled test makes SightlineNet look about three times better than it is on road it has not seen nearby.",
    );
  });

  it("never calls a ratio of one or less flattery, and calls one under 1.25 a little", () => {
    expect(pairedMeaning("PilotNet", 1)).toBe("On these frames, a shuffled test does not make PilotNet look any better than it is.");
    expect(pairedMeaning("PilotNet", 0.7)).toBe(pairedMeaning("PilotNet", 1));
    expect(pairedMeaning("PilotNet", 1.2)).toMatch(/look a little better/);
    expect(pairedMeaning("PilotNet", 1.25)).toMatch(/look about one and a half times better/);
  });

  it("gives steering straight on the same frames for scale, as a range over seeds", () => {
    expect(pairedScale(paired().sightlinenet)).toBe("Always steering straight is off by 16.2° to 18.8° on the same frames.");
    const json = resultsFixture();
    for (const i of [0, 1, 2]) {
      for (const network of ["sightlinenet", "pilotnet"]) patch(json, `paired_leakage.${network}.per_seed.${i}.zero_mae_deg`, 17.3);
    }
    expect(pairedScale(readPaired(json).sightlinenet)).toBe("Always steering straight is off by 17.3° on the same frames.");
  });

  it("names each seed's frames, never one count for all of them", () => {
    expect(pairedFrames(paired().sightlinenet)).toBe(
      "Each line is one training seed: the network trained on that seed’s shuffled split and on its blocked split, both scored on the frames both splits held out (893, 889 and 943 frames).",
    );
    expect(pairedRatioLabel(paired().sightlinenet.ratio.mean)).toBe("3.0× on average");
  });

  it("describes the chart with every number it draws", () => {
    expect(pairedAlt("SightlineNet", paired().sightlinenet)).toEqual({
      title: "SightlineNet’s mean error on the frames both splits held out, trained each way",
      desc:
        "Seed 0, 893 frames: 3.0° trained on shuffled frames, 11.2° trained with stretches held back, 3.7× as much; always steering straight 17.3°. " +
        "Seed 1, 889 frames: 3.3° trained on shuffled frames, 8.6° trained with stretches held back, 2.6× as much; always steering straight 16.2°. " +
        "Seed 2, 943 frames: 3.8° trained on shuffled frames, 10.5° trained with stretches held back, 2.8× as much; always steering straight 18.8°. " +
        "Mean of the three seeds: 3.3° and 10.1°; the held-back error is on average 3.0× the shuffled one.",
    });
  });
});

describe("the ladder's sentences", () => {
  const ladder = () => readLadder(resultsFixture());
  const splits = () => readSplits(resultsFixture());

  it("says what moved further from the training frames on each rung", () => {
    expect(ladder().rungs.map((rung) => rungCaption(rung, splits()))).toEqual([
      "Test frames are dealt out at random, so 91% of them sit within one frame of a frame the network trained on.",
      "Whole stretches of the drive are held back, so no test frame is closer than 501 frames to one the network trained on.",
      "It trains on the first 70% of the drive and is tested on the last 15%, never closer than 10,075 frames to anything it trained on.",
      "A drive recorded on another day: 45,404 frames of the 2017 drive, none of them seen in training.",
    ]);
  });

  it("gives steering straight on each rung, and names the network not trained this way", () => {
    expect([0, 1, 2, 3].map((i) => rungNote(ladder(), i))).toEqual([
      "Each training is tested on its own draw of shuffled frames; always steering straight is off by 13.1° on them, on average. SightlineNet with one frame was not trained this way.",
      "Always steering straight is off by 17.5° on these frames.",
      "Always steering straight is off by 9.4° on these frames. SightlineNet with one frame was not trained this way.",
      "Always steering straight is off by 15.5° on these frames.",
    ]);
  });

  it("puts each bar in words, in either view, and a missing one as not trained this way", () => {
    const { bars } = ladder();
    expect(barWords("pilotnet", bars.pilotnet[2] ?? null, "skill")).toBe("PilotNet: 45% worse than always steering straight (±44%)");
    expect(barWords("sightlinenet", bars.sightlinenet[0] ?? null, "skill")).toBe("SightlineNet: 78% better than always steering straight (±0.04%)");
    expect(barWords("pilotnet", bars.pilotnet[2] ?? null, "error")).toBe("PilotNet: 13.6° (each training: 15.1°, 9.0°, 16.9°)");
    expect(barWords("sightlinenet-1f", null, "skill")).toBe("SightlineNet, one frame: not trained this way");
  });

  it("describes the chart in the view on screen, with every number it draws", () => {
    const skill = ladderAlt(ladder(), "skill");
    expect(skill.title).toBe("How much better than always steering straight each network is, on each rung");
    expect(skill.desc).toBe(
      "Mean of three trainings, with the spread over them. " +
        "Shuffled: SightlineNet: 78% better than always steering straight (±0.04%); PilotNet: 70% better than always steering straight (±1%); SightlineNet, one frame: not trained this way. " +
        "Blocked: SightlineNet: 41% better than always steering straight (±4%); PilotNet: 26% better than always steering straight (±1%); SightlineNet, one frame: 35% better than always steering straight (±3%). " +
        "Later in the drive: SightlineNet: 34% better than always steering straight (±3%); PilotNet: 45% worse than always steering straight (±44%); SightlineNet, one frame: not trained this way. " +
        "Another day: SightlineNet: 31% better than always steering straight (±8%); PilotNet: 24% worse than always steering straight (±8%); SightlineNet, one frame: 2% better than always steering straight (±4%).",
    );
    const error = ladderAlt(ladder(), "error");
    expect(error.title).toBe("Each network’s mean error in degrees on each rung, beside always steering straight");
    expect(error.desc).toMatch(/^Mean of three trainings, with each training’s error\. Shuffled: SightlineNet: 2\.9° \(each training: 2\.8°, 2\.9°, 3\.0°\); /);
    expect(error.desc).toMatch(/Another day: .*SightlineNet, one frame: 15\.2° \(each training: 14\.5°, 15\.6°, 15\.6°\); always steering straight: 15\.5°\.$/);
  });
});

describe("the cards' sentences", () => {
  it("says what the second frame buys, calling a skill near zero level", () => {
    expect(twoFramesText(readLadder(resultsFixture()), readFrames(resultsFixture()))).toEqual({
      lead: "On another day’s drive, SightlineNet with one frame is about level with always steering straight (2% better than it). With two frames it is 31% better than always steering straight.",
      also: "On held-back stretches of its own drive: 41% better than steering straight with two frames, 35% better than it with one frame.",
    });
    const json = setSkill(resultsFixture(), "sightlinenet-1f", "blocked", "cross_drive", 0.12);
    expect(twoFramesText(readLadder(json), readFrames(json))?.lead).toMatch(/^On another day’s drive, SightlineNet with one frame is 12% better than always steering straight\./);
    const worse = setSkill(resultsFixture(), "sightlinenet-1f", "blocked", "cross_drive", -0.049);
    expect(twoFramesText(readLadder(worse), readFrames(worse))?.lead).toMatch(/about level with always steering straight \(4.9% worse than it\)|about level with always steering straight \(5% worse than it\)/);
  });

  it("names the regimes by the edges in the data", () => {
    expect(regimeNames([5, 30])).toEqual({ straight: "Straight, wheel under 5°", gentle: "Gentle curve, 5° to 30°", sharp: "Sharp curve, 30° or more" });
    expect(regimeNames([2.5, 45]).gentle).toBe("Gentle curve, 2.5° to 45°");
  });

  it("says where Sightline struggles, and that steering straight wins on straight road", () => {
    const regimes = readRegimes(resultsFixture());
    expect(regimesText(regimes)).toEqual({
      lead: "Sightline’s 6,000 test frames, sorted by how hard the driver was turning. It is furthest off on sharp curves: 36.0°, where always steering straight is off by 78.4°.",
      straight: "On straight road, always steering straight wins: it is off by 2.8°, Sightline by 6.1°.",
    });
    expect(regimesAlt(regimes)).toBe(
      "Mean error by how hard the driver was turning. Straight, wheel under 5° (2,731 frames): Sightline 6.1°, always steering straight 2.8°. " +
        "Gentle curve, 5° to 30° (2,416 frames): Sightline 5.5°, always steering straight 12.6°. " +
        "Sharp curve, 30° or more (853 frames): Sightline 36.0°, always steering straight 78.4°.",
    );
    const better = patch(resultsFixture(), "regimes.models.sightlinenet.regimes.straight.mae_deg", 2.1);
    expect(regimesText(readRegimes(better)).straight).toBe("On straight road Sightline beats always steering straight too: 2.1° against 2.8°.");
  });

  it("tells the standard recipe's story from its report", () => {
    const legacy = readLegacy(resultsFixture());
    expect(legacyText(legacy)).toEqual({
      lead: "On its own training drive, the 2017 drive, it is off by 4.5°: better than always steering straight (15.5°).",
      other:
        "On the 2018 drive, which it never saw, it does worse than always steering straight: 22.0° against 17.5° on the held-back stretches Sightline is tested on, " +
        "and 19.4° against 9.4° on the drive’s last part.",
      cap: "Its arctangent output can never pass 180° either way, half a turn of the wheel: its largest answer, either way, is 169.8°, yet the driver turned past 180° on 235 frames of the 2018 drive.",
    });
    expect(recipeNote(legacy.recipe)).toBe(
      "The recipe, as its report records it: PilotNet in TensorFlow 1: shuffled 80/20 frame split, MSE in radians plus L2, atan output head limited to ±180°.",
    );
    const mixed = readLegacy(patch(resultsFixture(), "legacy.eval.drive2018_forward_test.model.mae_deg", 8.9));
    expect(legacyText(mixed).other).toBe(
      "On the 2018 drive, which it never saw, it is off by 22.0° on the held-back stretches Sightline is tested on, worse than always steering straight (17.5°), " +
        "and by 8.9° on the drive’s last part, better than always steering straight (9.4°).",
    );
    expect(legacyAlt(legacy)).toBe(
      "Mean error of the standard-recipe PilotNet against always steering straight: on its own training drive (45,404 frames), 4.5° against 15.5°; " +
        "on the held-back stretches of the 2018 drive (6,000 frames), 22.0° against 17.5°; on the last part of the 2018 drive (9,324 frames), 19.4° against 9.4°.",
    );
  });

  it("states the export check against its gate, and a failed gate as a failure", () => {
    expect(exportText(readParity(resultsFixture()))).toBe(
      "When each model was exported, onnxruntime 1.28.0 reproduced PyTorch on 848 frame stacks: Sightline’s angles to within 0.0002°, its twin’s to within 0.0002°, under a gate of 0.0005°.",
    );
    const failed = patch(resultsFixture(), "parity.shuffled.max_abs_angle_deg", 0.0007);
    expect(exportText(readParity(failed))).toBe("When the models were exported, onnxruntime and PyTorch differed by up to 0.0007°, more than the 0.0005° allowed.");
  });

  it("states each protocol with its arithmetic", () => {
    expect(patternWords("TTVTTET")).toBe("train, train, validate, train, train, test, train");
    expect(protocolTexts(readSplits(resultsFixture()))).toEqual({
      shuffled: "The 63,825 frames of the 2018 drive are shuffled and dealt 70%, 15% and 15%: 44,677 to train, 9,574 to validate and 9,574 to test.",
      blocked:
        "The drive is cut into blocks of 1,500 frames, dealt train, train, validate, train, train, test, train, over and over. " +
        "At every boundary between a training block and a held-out one, 250 frames are dropped on each side: 24 boundaries × 500 frames = 12,000 frames, " +
        "so every test frame is at least 250 + 250 + 1 = 501 frames from training. That leaves 39,825 to train, 6,000 to validate and 6,000 to test.",
      later:
        "The first 70% of the drive trains, the next 15% validates and the last 15% tests, with 250 frames dropped on each side of both boundaries (1,000 in all): " +
        "44,427 to train, 9,074 to validate and 9,324 to test.",
      anotherDay: "The 2017 drive, recorded on another day: 45,404 of its frames, scored by the networks trained with stretches held back.",
    });
  });
});

describe("counts said from the data, never typed", () => {
  it("counts the trainings behind every bar, or says none when they differ", () => {
    expect(trainingsWord(readLadder(resultsFixture()))).toBe("three");
    const ladder = readLadder(resultsFixture());
    const fewer = { ...ladder, bars: { ...ladder.bars, pilotnet: ladder.bars.pilotnet.map((bar) => (bar ? { ...bar, seeds: [0, 1] } : bar)) } };
    expect(trainingsWord(fewer)).toBeNull();
    expect(ladderCaption(fewer, "skill")).toBe("Each bar is a mean over the network’s trainings; each whisker spans one standard deviation over them.");
    expect(ladderCaption(ladder, "skill")).toBe("Mean of three trainings; each whisker spans one standard deviation over them.");
    expect(ladderCaption(ladder, "error")).toBe("Mean of three trainings; each tick is one training. Each rung’s grey line is always steering straight on its frames.");
  });

  it("names a network's frames from its data: the shipped network's lags, the other's key", () => {
    const frames = readFrames(resultsFixture());
    expect(frames).toEqual({ sightlinenet: 2, "sightlinenet-1f": 1 });
    expect(networkName("sightlinenet-1f")).toEqual({ name: "SightlineNet, one frame", said: "SightlineNet with one frame" });
    expect(networkName("sightlinenet-3f")).toEqual({ name: "SightlineNet, three frames", said: "SightlineNet with three frames" });
    expect(networkName("pilotnet")).toEqual({ name: "PilotNet", said: "PilotNet" });
    expect(frameWords(frames)).toEqual({ more: "two frames", fewer: "one frame" });
    expect(framesTitleOf(frames)).toBe("What a second frame buys");
    expect(framesTitleOf({ sightlinenet: 3, "sightlinenet-1f": 1 })).toBe("What two more frames buy");
    expect(framesTitleOf(null)).toBe("What an earlier frame buys");
    const without = resultsFixture();
    delete without["timing"];
    expect(readFrames(without)).toBeNull();
  });

  it("says what the extra frame buys with the counts the data gives", () => {
    const ladder = readLadder(resultsFixture());
    expect(twoFramesText(ladder, { sightlinenet: 3, "sightlinenet-1f": 1 })?.lead).toBe(
      "On another day’s drive, SightlineNet with one frame is about level with always steering straight (2% better than it). With three frames it is 31% better than always steering straight.",
    );
  });

  it("says what the extra frame buys without a count when the data gives none", () => {
    const without = resultsFixture();
    delete without["timing"];
    expect(twoFramesText(readLadder(without), readFrames(without))).toEqual({
      lead: "On another day’s drive, SightlineNet with one frame is about level with always steering straight (2% better than it). With an earlier frame too it is 31% better than always steering straight.",
      also: "On held-back stretches of its own drive: 41% better than steering straight with an earlier frame too, 35% better than it with one frame.",
    });
  });

  it("never calls a clearly negative skill about level", () => {
    const json = setSkill(resultsFixture(), "sightlinenet-1f", "blocked", "cross_drive", -0.3);
    expect(twoFramesText(readLadder(json), readFrames(json))?.lead).toMatch(/^On another day’s drive, SightlineNet with one frame is 30% worse than always steering straight\./);
  });

  it("puts the head's limit in turns of the wheel only when it is a plain share of one", () => {
    expect(turnWords(180)).toBe("half a turn of the wheel");
    expect(turnWords(360)).toBe("a full turn of the wheel");
    expect(turnWords(90)).toBe("a quarter turn of the wheel");
    expect(turnWords(170)).toBe("");
  });

  it("says the head cannot pass its limit only while its largest answer is below it", () => {
    const over = readLegacy(patch(resultsFixture(), "legacy.atan_cap.max_abs_pred_deg", 185));
    expect(legacyText(over).cap).toBe("Its largest answer, either way, is 185.0°; the driver turned past 180° on 235 frames of the 2018 drive.");
  });
});

describe("the ladder's lead", () => {
  it("says what a bar's height means in the view on screen, never that a bar above the line wins in raw error", () => {
    expect(ladderLead("skill")).toMatch(/A bar above the line beats always steering straight on the same frames; a bar below it does worse\.$/);
    expect(ladderLead("error")).not.toMatch(/above the line beats/);
    expect(ladderLead("error")).toMatch(/A bar that stops short of its rung’s grey line beats always steering straight on the same frames; a bar that passes it does worse\.$/);
  });
});

describe("sentences that follow the data where it could change", () => {
  it("names the regime where the model is furthest off", () => {
    const json = patch(resultsFixture(), "regimes.models.sightlinenet.regimes.sharp.mae_deg", 3);
    expect(regimesText(readRegimes(json)).lead).toBe(
      "Sightline’s 6,000 test frames, sorted by how hard the driver was turning. It is furthest off on straight road: 6.1°, where always steering straight is off by 2.8°.",
    );
  });

  it("says steering straight on the shuffled rung is a mean over each training's own frames", () => {
    expect(rungNote(readLadder(resultsFixture()), 0)).toBe(
      "Each training is tested on its own draw of shuffled frames; always steering straight is off by 13.1° on them, on average. SightlineNet with one frame was not trained this way.",
    );
  });
});

describe("the baselines' sentences", () => {
  const sets = () => readYardsticks(resultsFixture());

  it("names each rule plainly", () => {
    expect(YARDSTICK_LABELS).toEqual({ zero: "steering straight", mean: "mean angle", zoh: "hold the last label", extrap: "linear extrapolation" });
  });

  it("titles each set of frames with its drive and its count", () => {
    expect(sets().map(yardstickSetTitle)).toEqual(["Held-back stretches of the 2018 drive, 6,000 frames", "Another day: the 2017 drive, 45,404 frames"]);
  });

  it("describes every number the rulers draw, the rules that read the answers named as doing so", () => {
    expect(yardsticksAlt(sets())).toBe(
      "Mean error on the held-back stretches of the 2018 drive (6,000 frames): always steering straight 17.5°, always steering the training frames’ mean angle 17.8°, " +
        "and, reading the answer stream the model never sees, holding the last label 0.6° and linear extrapolation 0.7°; Sightline 10.1°. " +
        "Mean error on another day’s drive, the 2017 drive (45,404 frames): always steering straight 15.5°, always steering the training frames’ mean angle 15.5°, " +
        "and, reading the answer stream the model never sees, holding the last label 0.5° and linear extrapolation 0.5°; Sightline 9.8°.",
    );
  });

  it("says what the rules that read the answers show: a bound on their mean error, rounded up, never one on every frame", () => {
    expect(yardsticksVerdict(sets())).toBe(
      "The rules that read the answers are off by 0.7° or less on average: the true angle barely changes from one frame to the next. They are yardsticks here, not rivals.",
    );
    // 0.64° rounds to 0.6° but is bounded by 0.7°.
    const json = resultsFixture();
    for (const split of ["test", "cross_drive"]) {
      patch(json, `baselines.${split}.baselines.zoh.mae_deg`, 0.5);
      patch(json, `baselines.${split}.baselines.extrap.mae_deg`, split === "test" ? 0.64 : 0.55);
    }
    expect(yardsticksVerdict(readYardsticks(json))).toMatch(/^The rules that read the answers are off by 0\.7° or less on average:/);
    expect(yardsticksVerdict(readYardsticks(json))).not.toMatch(/at most/);
  });
});

describe("the frames card's chart in words", () => {
  it("names both networks by their frames, and every number it draws", () => {
    const rows = [
      { title: "Another day’s drive", more: 0.3145251314711981, fewer: 0.0213 },
      { title: "Held-back stretches of its own drive", more: 0.41, fewer: 0.35 },
    ];
    expect(framesAlt(rows, readFrames(resultsFixture()))).toEqual({
      title: "SightlineNet with two frames and with one frame, against always steering straight",
      desc:
        "How much better than always steering straight SightlineNet is with two frames and with one frame: " +
        "Another day’s drive: two frames 31% better than always steering straight, one frame 2% better than always steering straight; " +
        "Held-back stretches of its own drive: two frames 41% better than always steering straight, one frame 35% better than always steering straight.",
    });
  });
});

// -- The frame pairs and the split timeline -------------------------------------------

import pairsFile from "../../public/data/leakage/pairs.json?raw";
import shippedFile from "../../public/data/results.json?raw";
import clipJson from "../../public/data/clips/c2018/clip.json?raw";
import {
  clockTime,
  gapWords,
  neighbourWords,
  pairAlt,
  pairsLead,
  timelineAlt,
  timelineClipNote,
  timelineLead,
  timelineNote,
} from "./copy.ts";
import { readClipWindow, readPairs, readTimeline } from "./figures.ts";

const realResults = () => JSON.parse(shippedFile) as Record<string, unknown>;
const thePairs = () => readPairs(JSON.parse(pairsFile) as Record<string, unknown>, realResults());

describe("the frame pairs' sentences", () => {
  it("says how far each split's closest training frame is, with a lower bound rounded down", () => {
    expect(pairsLead(thePairs())).toBe(
      "Six test frames, each beside the training frame that looks most like it. Under the shuffled split that frame is the next one along, " +
        "30 to 57 ms away. Under the blocked split it is 1,887 frames away or more, at least 1.7 minutes of driving.",
    );
  });

  it("rounds the blocked neighbours' lower bound down, never up", () => {
    const json = JSON.parse(pairsFile) as Record<string, unknown>;
    // The nearest blocked neighbour 119 s away: 1.98 minutes, so at least 1.9, not 2.0.
    const pairs = readPairs(patch(patch(json, "pairs.2.blocked.gap_s", 119), "pairs.3.blocked.gap_s", 119), realResults());
    expect(pairsLead(pairs)).toMatch(/at least 1\.9 minutes of driving\.$/);
  });

  it("puts a time apart in the unit that reads it best, from the timestamps, never a frame count times a typical interval", () => {
    expect(gapWords(0.041)).toBe("41 ms");
    expect(gapWords(0.999)).toBe("999 ms");
    expect(gapWords(12.34)).toBe("12.3 s");
    expect(gapWords(107.129)).toBe("1 min 47 s");
    expect(gapWords(1073.237)).toBe("17 min 53 s");
    // Whole seconds are rounded down: 1 min 59.9 s is never "2 min 0 s".
    expect(gapWords(119.9)).toBe("1 min 59 s");
    expect(gapWords(null)).toBe("");
  });

  it("names each neighbour by where it is, how alike it looks and what the driver steered", () => {
    const pair = thePairs().pairs[0];
    if (!pair) throw new Error("no pair");
    expect(neighbourWords(pair, "shuffled")).toEqual({
      where: "1 frame before it, 41 ms",
      likeness: "Pixel difference 8.8",
      steering: "Steering 0.5° right",
    });
    expect(neighbourWords(pair, "blocked")).toEqual({
      where: "20,302 frames after it, 17 min 53 s",
      likeness: "Pixel difference 28.5",
      steering: "Steering 1.4° right",
    });
    expect(pairAlt(pair)).toBe(
      "Test frame 7,917, the driver steering 0.4° right. Under the shuffled split its closest training frame is 1 frame before it, 41 ms, " +
        "pixel difference 8.8, steering 0.5° right. Under the blocked split it is 20,302 frames after it, 17 min 53 s, pixel difference 28.5, steering 1.4° right.",
    );
  });
});

describe("the split timeline's sentences", () => {
  const timeline = () => readTimeline(realResults());

  it("words the drive's length from its measured duration", () => {
    expect(timelineLead(timeline())).toBe(
      "The 55.9 minutes of the 2018 drive, from its first frame to its last, and how each split deals its 63,825 frames to training, validation and testing.",
    );
    expect(clockTime(1519.063)).toBe("25:19");
    expect(clockTime(3351.698)).toBe("55:51");
  });

  it("says what each split's drawing shows, and that the shuffled one is an illustration", () => {
    expect(timelineNote(timeline(), "shuffled")).toBe(
      "Every frame is dealt at random: 70% to train, 15% to validate and 15% to test. The band shows an even mix at those shares, an illustration: it does not say which frame went where.",
    );
    expect(timelineNote(timeline(), "blocked")).toMatch(/^Blocks of 1,500 frames, dealt train, train, validate, train, train, test, train, over and over, with 250 frames dropped on each side of every boundary between a training block and a held-out one\.$/);
    expect(timelineNote(timeline(), "later")).toBe(
      "The first 70% of the drive trains, the next 15% validates and the last 15% tests, with 250 frames dropped on each side of both boundaries.",
    );
  });

  it("places the demo's clip from its own timestamps, and says which stretch it sits in", () => {
    const clip = readClipWindow(JSON.parse(clipJson));
    expect(timelineClipNote(timeline(), clip)).toBe("The demo’s 2018 clip, from 25:19 to 25:41, sits inside one of the blocked split’s test stretches.");
    expect(timelineClipNote(timeline(), clip, "later")).toBe("The demo’s 2018 clip, from 25:19 to 25:41, sits inside one of the later-in-the-drive split’s training stretches.");
    expect(timelineClipNote(timeline(), clip, "shuffled")).toMatch(/dealt frame by frame/);
  });

  it("describes every split's drawing in words, with the frames it deals", () => {
    expect(timelineAlt(timeline(), "blocked")).toMatch(
      /^The 2018 drive, 55\.9 minutes, split the blocked way: 39,825 frames to train, 6,000 to validate, 6,000 to test and 12,000 dropped, in .* stretches\. Above it, how hard the driver was turning, on a square-root scale\.$/,
    );
    expect(timelineAlt(timeline(), "shuffled")).toMatch(/dealt at random/);
  });
});
