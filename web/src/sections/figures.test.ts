import { describe, expect, it } from "vitest";

import shipped from "../../public/data/results.json?raw";
import { AssetError } from "../data/load.ts";
import { readFigures, readFrames, readLadder, readLegacy, readPaired, readParity, readRegimes, readSplits, readYardsticks } from "./figures.ts";
import { patch, resultsFixture } from "./fixture.ts";

/** The AssetError a reader throws on `json`, or undefined. */
function refusal(read: (json: Record<string, unknown>) => unknown, json: Record<string, unknown>): AssetError | undefined {
  try {
    read(json);
  } catch (error) {
    expect(error).toBeInstanceOf(AssetError);
    expect((error as AssetError).kind).toBe("invalid");
    return error as AssetError;
  }
  return undefined;
}

describe("readFigures", () => {
  it("copies the figures the page quotes, value for value", () => {
    const figures = readFigures(resultsFixture());
    expect(figures.heldOut).toEqual({ frames: 6000, model: 10.12272448789344, straight: 17.479270016382138 });
    expect(figures.paired.sightlinenet).toEqual({
      ratio: { mean: 3.0465626341557503, sd: 0.6142642724803958, runs: 3 },
      shuffled: 3.3481055871804877,
      blocked: 10.086980146092957,
      frames: [893, 889, 943],
    });
    expect(figures.paired.pilotnet.ratio.mean).toBe(3.166602797825647);
    expect(figures.anotherDay).toEqual({
      skill: { mean: 0.3145251314711981, sd: 0.08401826957372512, runs: 3 },
      model: 10.640418845640982,
      straight: 15.52269723393061,
    });
    expect(figures.memoriser).toEqual({ shuffled: 0.6379842974604311, blocked: 21.867592464485867, frames: 893 });
    expect(figures.gaps).toEqual({ shuffledWithinOne: 0.9093377898475037, blockedMinFrames: 501 });
    expect(figures.runs).toBe(21);
  });

  it("refuses held-out figures from a run other than the shipped one", () => {
    const json = patch(resultsFixture(), "baselines.run", "drive2018_sightlinenet_blocked_s0");
    expect(() => readFigures(json)).toThrow(/baselines\.run must be the shipped run/);
  });

  it("refuses a comparison with steering straight scored on other frames", () => {
    const json = patch(resultsFixture(), "baselines.test.baselines.zero.n", 5999);
    expect(() => readFigures(json)).toThrow(/baselines\.test\.baselines\.zero\.n must be 6000/);
  });

  it("needs a frame count for every seed of the paired measurement", () => {
    const json = resultsFixture();
    patch(json, "paired_leakage.pilotnet.per_seed", [{ n: 893 }, { n: 889 }]);
    expect(() => readFigures(json)).toThrow(/paired_leakage\.pilotnet\.per_seed must be one entry per ratio value \(3\)/);
    patch(json, "paired_leakage.pilotnet.per_seed", [{ n: 893 }, { n: 889 }, {}]);
    expect(() => readFigures(json)).toThrow(/paired_leakage\.pilotnet\.per_seed\.2\.n must be a count/);
  });

  it("refuses a lookup table scored on different frames under the two splits", () => {
    const json = patch(resultsFixture(), "leakage.memoriser.blocked.n", 900);
    expect(() => readFigures(json)).toThrow(/leakage\.memoriser\.blocked\.n/);
  });

  it("names the key that is missing or malformed, as invalid data", () => {
    const cases: Array<[string, unknown, RegExp]> = [
      ["paired_leakage.sightlinenet.ratio.mean", undefined, /paired_leakage\.sightlinenet\.ratio\.mean must be a number/],
      ["groups.sightlinenet/blocked.cross_skill", undefined, /groups\.sightlinenet\/blocked\.cross_skill\.mean/],
      ["leakage.gaps.leaky.test.frac_within_1", 1.5, /frac_within_1 must be a share/],
      ["baselines.test.model.mae_deg", -1, /mae_deg must be an error/],
      ["baselines.test.n", 0, /baselines\.test\.n must be a count/],
      ["paired_leakage.pilotnet.ratio.sd", -0.1, /ratio\.sd must be a spread/],
    ];
    for (const [path, value, message] of cases) {
      const json = patch(resultsFixture(), path, value);
      let thrown: unknown;
      try {
        readFigures(json, "data/results.json");
      } catch (error) {
        thrown = error;
      }
      expect(thrown, path).toBeInstanceOf(AssetError);
      expect((thrown as AssetError).kind, path).toBe("invalid");
      expect((thrown as AssetError).message, path).toMatch(message);
    }
  });
});

describe("the readers of the Results view's sections", () => {
  it("all accept the results.json this site ships", () => {
    const raw = JSON.parse(shipped) as Record<string, unknown>;
    for (const read of [readFigures, readPaired, readLadder, readRegimes, readLegacy, readParity, readSplits, readYardsticks]) {
      expect(() => read(raw), read.name).not.toThrow();
    }
  });
});

describe("readPaired", () => {
  it("copies each seed's pair: its frames, both errors, their ratio and steering straight on the same frames", () => {
    const paired = readPaired(resultsFixture());
    expect(paired.sightlinenet.seeds).toEqual([
      { seed: 0, frames: 893, shuffled: 2.9803185991667553, blocked: 11.167367562601575, ratio: 3.747038174282365, straight: 17.308191551773707 },
      { seed: 1, frames: 889, shuffled: 3.28994326465276, blocked: 8.553125311204633, ratio: 2.5997789697772737, straight: 16.160995532040925 },
      { seed: 2, frames: 943, shuffled: 3.774054897721948, blocked: 10.540447564472663, ratio: 2.792870758407613, straight: 18.825535512678442 },
    ]);
    expect(paired.sightlinenet.ratio).toEqual({ mean: 3.0465626341557503, sd: 0.6142642724803958, runs: 3 });
    expect([paired.pilotnet.shuffled, paired.pilotnet.blocked]).toEqual([4.130170764717941, 13.07809728495099]);
  });

  it("refuses a ratio that is not the blocked error over the shuffled one", () => {
    const json = patch(resultsFixture(), "paired_leakage.sightlinenet.per_seed.1.ratio", 2.7);
    expect(refusal(readPaired, json)?.message).toMatch(/per_seed\.1\.ratio must be the blocked error over the shuffled one/);
  });

  it("refuses a statistic over seeds that is not made of the seeds beside it", () => {
    // Two seeds swapped: the mean and the spread still hold, the seeds do not.
    const json = patch(resultsFixture(), "paired_leakage.pilotnet.blocked_mae_deg.values", [11.91747827882825, 13.385377171316913, 13.931436404707808]);
    expect(refusal(readPaired, json)?.message).toMatch(/blocked_mae_deg\.values\.0 must be 13\.38/);
  });

  it("refuses a seed whose two networks were scored on different frames", () => {
    const json = patch(resultsFixture(), "paired_leakage.pilotnet.per_seed.2.n", 942);
    expect(refusal(readPaired, json)?.message).toMatch(/pilotnet\.per_seed\.2\.n must be 943, the frames SightlineNet's pair was scored on/);
  });
});

describe("readLadder", () => {
  it("reads the rungs in order, named as the data names them, with steering straight on each", () => {
    const ladder = readLadder(resultsFixture());
    expect(ladder.rungs.map(({ kind, name, drive, straight }) => [kind, name, drive, straight])).toEqual([
      ["shuffled", "shuffled", "drive2018", 13.13605197527287],
      ["blocked", "blocked", "drive2018", 17.479270016382138],
      ["later", "later in the drive", "drive2018", 9.406421568820829],
      ["another-day", "another day", "drive2017", 15.52269723393061],
    ]);
  });

  it("takes each bar's skill from its group, negative where the network is worse than steering straight", () => {
    const { bars } = readLadder(resultsFixture());
    expect(bars.sightlinenet.map((bar) => bar?.skill.mean)).toEqual([0.7788633029556049, 0.41426654247784617, 0.34295904005598987, 0.3145251314711981]);
    expect(bars.pilotnet.map((bar) => bar?.skill.mean)).toEqual([0.6951777264965134, 0.2598095345840305, -0.4485684827442407, -0.24229235843871808]);
    expect(bars.pilotnet[2]?.skill.sd).toBe(0.440446173159765);
    expect(bars.pilotnet[2]?.error.values).toEqual([15.065162773726238, 8.955122993239405, 16.857251693032822]);
  });

  it("leaves a gap where a network was never trained with a rung's split, and nowhere else", () => {
    const { bars } = readLadder(resultsFixture());
    expect(bars["sightlinenet-1f"].map((bar) => (bar === null ? null : bar.skill.mean))).toEqual([null, 0.3467922040093572, null, 0.01885671731341254]);
  });

  it("refuses a missing rung for a network that was trained with its split", () => {
    const json = resultsFixture();
    const ladder = json["ladder"] as Record<string, unknown[]>;
    ladder["pilotnet"] = (ladder["pilotnet"] as unknown[]).filter((_, i) => i !== 2);
    expect(refusal(readLadder, json)?.message).toMatch(/ladder\.pilotnet must be a later in the drive rung: groups holds pilotnet\/forward/);
  });

  it("refuses rungs out of order, or a rung it cannot name", () => {
    const json = resultsFixture();
    const ladder = json["ladder"] as Record<string, unknown[]>;
    ladder["sightlinenet"] = [...(ladder["sightlinenet"] as unknown[])].reverse();
    expect(refusal(readLadder, json)?.message).toMatch(/ladder\.sightlinenet\.1 must be the next rung away from the training frames/);
    const odd = patch(resultsFixture(), "ladder.pilotnet.1.split", "val");
    expect(refusal(readLadder, odd)?.message).toMatch(/ladder\.pilotnet\.1 must be a rung the page knows/);
  });

  it("refuses a bar whose group says otherwise, or steering straight scored on other frames", () => {
    const mean = patch(resultsFixture(), "groups.pilotnet/forward.test_mae_deg.mean", 13.6);
    expect(refusal(readLadder, mean)?.message).toMatch(/groups\.pilotnet\/forward\.test_mae_deg\.mean must be 13\.62/);
    const straight = patch(resultsFixture(), "ladder.pilotnet.3.zero_mae_deg", 15.6);
    expect(refusal(readLadder, straight)?.message).toMatch(/ladder\.pilotnet\.3\.zero_mae_deg must be 15\.52/);
  });
});

describe("readRegimes", () => {
  it("reads the edges, the frames in each regime and every error on them", () => {
    const regimes = readRegimes(resultsFixture());
    expect(regimes.edges).toEqual([5, 30]);
    expect(regimes.frames).toEqual({ straight: 2731, gentle: 2416, sharp: 853 });
    expect(regimes.total).toBe(6000);
    expect(regimes.straight.straight).toBe(2.763533517210879);
    expect(regimes.models.sightlinenet).toEqual({
      run: "drive2018_sightlinenet_blocked_s2",
      seed: 2,
      errors: { straight: 6.094077526435531, gentle: 5.523734191249877, sharp: 36.046986396958374 },
    });
  });

  it("refuses regimes scored on other frames, or not the shipped model's", () => {
    expect(refusal(readRegimes, patch(resultsFixture(), "regimes.models.pilotnet.regimes.sharp.n", 852))?.message).toMatch(
      /regimes\.models\.pilotnet\.regimes\.sharp\.n must be 853/,
    );
    expect(refusal(readRegimes, patch(resultsFixture(), "regimes.zero.straight.n", 2730))?.message).toMatch(/regimes\.zero must be regimes that hold the 6000 test frames/);
    expect(refusal(readRegimes, patch(resultsFixture(), "regimes.models.sightlinenet.run", "drive2018_sightlinenet_blocked_s0"))?.message).toMatch(
      /regimes\.models\.sightlinenet\.run must be drive2018_sightlinenet_blocked_s2, the shipped run/,
    );
    expect(refusal(readRegimes, patch(resultsFixture(), "cutoffs.regime_bins_deg", [30, 5]))?.message).toMatch(/regime_bins_deg\.1 must be an angle above 30/);
  });
});

describe("readLegacy", () => {
  it("reads its own drive, the other drive's test frames, its largest prediction and the labels past its limit", () => {
    expect(readLegacy(resultsFixture())).toEqual({
      recipe: "PilotNet in TensorFlow 1: shuffled 80/20 frame split, MSE in radians plus L2, atan output head limited to ±180°",
      own: { drive: "drive2017", frames: 45404, model: 4.480583195894129, straight: 15.52269723393061 },
      firstFrame: 2,
      blocked: { drive: "drive2018", frames: 6000, model: 21.97078830103064, straight: 17.479270016382138 },
      later: { drive: "drive2018", frames: 9324, model: 19.394295458684024, straight: 9.406421568820829 },
      largest: 169.84153747558594,
      cap: 180,
      beyond: 235,
    });
  });

  it("refuses a limit it cannot name, or scores it cannot place", () => {
    const json = resultsFixture();
    const cap = (json["legacy"] as Record<string, Record<string, unknown>>)["atan_cap"] as Record<string, unknown>;
    cap["labels_beyond_200"] = cap["labels_beyond_180"];
    expect(refusal(readLegacy, json)?.message).toMatch(/legacy\.atan_cap must be one count of labels beyond/);
    expect(refusal(readLegacy, patch(resultsFixture(), "legacy.trained_on", "drive2018"))?.message).toMatch(/legacy\.eval must be/);
  });
});

describe("readParity", () => {
  it("reads the export check of the page's model and of its twin", () => {
    const parity = readParity(resultsFixture());
    expect(parity.sightline).toEqual({
      run: "drive2018_sightlinenet_blocked_s2",
      stacks: 848,
      angle: 0.000179290771484375,
      angleGate: 0.0005,
      saliency: 7.450580596923828e-6,
      saliencyGate: 0.0001,
      onnxruntime: "1.28.0",
    });
    expect(parity.shuffled.angle).toBe(0.00012969970703125);
  });

  it("refuses a check of another run than the one the page ships", () => {
    expect(refusal(readParity, patch(resultsFixture(), "parity.shuffled.run", "drive2018_sightlinenet_leaky_s0"))?.message).toMatch(
      /parity\.shuffled\.run must be drive2018_sightlinenet_leaky_s2, the twin's run/,
    );
  });
});

describe("readSplits", () => {
  it("reads each split's frames and rules, with the firewall arithmetic the page states", () => {
    const splits = readSplits(resultsFixture());
    expect(splits.frames).toBe(63825);
    expect(splits.shuffled).toEqual({ label: "shuffled", train: 44677, val: 9574, test: 9574, dropped: 0, shares: [0.7, 0.15, 0.15], withinOne: 0.9093377898475037 });
    expect(splits.blocked).toEqual({
      label: "blocked", train: 39825, val: 6000, test: 6000, dropped: 12000, block: 1500, firewall: 250, pattern: "TTVTTET", nearest: 501, boundaries: 24,
    });
    expect(splits.later).toEqual({ label: "later in the drive", train: 44427, val: 9074, test: 9324, dropped: 1000, shares: [0.7, 0.15, 0.15], firewall: 250, nearest: 10075 });
    expect(splits.anotherDay).toEqual({ drive: "drive2017", frames: 45404 });
  });

  it("refuses sizes that do not add up, or a firewall the gaps do not show", () => {
    expect(refusal(readSplits, patch(resultsFixture(), "split_sizes.blocked.train", 39826))?.message).toMatch(/split_sizes\.blocked must be train, validation, test and dropped frames that add up to 63825/);
    expect(refusal(readSplits, patch(resultsFixture(), "leakage.gaps.blocked.test.min_frames", 500))?.message).toMatch(/min_frames must be 501, two firewalls and a frame/);
    expect(refusal(readSplits, patch(resultsFixture(), "split_rules.blocked.firewall", 300))?.message).toMatch(/must be 601|must be a whole number of boundaries|must be 300/);
    expect(refusal(readSplits, patch(resultsFixture(), "split_sizes.leaky.label", "random"))?.message).toMatch(/split_sizes\.leaky\.label must be shuffled/);
  });
});

describe("the readers' cross-checks of statistics over seeds", () => {
  it("refuse a paired ratio whose mean or spread is not made of its seeds' values", () => {
    expect(refusal(readFigures, patch(resultsFixture(), "paired_leakage.sightlinenet.ratio.mean", 5))?.message).toMatch(
      /paired_leakage\.sightlinenet\.ratio\.mean must be the mean of its values/,
    );
    expect(refusal(readPaired, patch(resultsFixture(), "paired_leakage.sightlinenet.ratio.mean", 5))?.message).toMatch(/ratio\.mean must be the mean of its values/);
    expect(refusal(readPaired, patch(resultsFixture(), "paired_leakage.pilotnet.shuffled_mae_deg.mean", 1))?.message).toMatch(
      /shuffled_mae_deg\.mean must be the mean of its values/,
    );
    expect(refusal(readFigures, patch(resultsFixture(), "paired_leakage.pilotnet.blocked_mae_deg.sd", 0.5))?.message).toMatch(
      /blocked_mae_deg\.sd must be the sample standard deviation of its values/,
    );
  });

  it("refuse another day's skill that is not one minus the error over steering straight's", () => {
    const json = patch(resultsFixture(), "groups.sightlinenet/blocked.cross_skill", { mean: 0.9, sd: 0.08401826957372512 });
    expect(refusal(readFigures, json)?.message).toMatch(/cross_skill\.mean must be 1 − cross_mae_deg\.mean \/ cross_zero_mae_deg/);
    expect(refusal(readLadder, json)?.message).toMatch(/cross_skill\.mean must be 1 − /);
    const sd = patch(resultsFixture(), "groups.sightlinenet/blocked.cross_skill.sd", 0.2);
    expect(refusal(readLadder, sd)?.message).toMatch(/cross_skill\.sd must be cross_mae_deg\.sd \/ cross_zero_mae_deg/);
  });

  it("refuse a ladder skill flipped against its errors, on a held-back rung and on the shuffled one", () => {
    const later = patch(resultsFixture(), "groups.pilotnet/forward.test_skill", { mean: 0.4485684827442407, sd: 0.440446173159765 });
    expect(refusal(readLadder, later)?.message).toMatch(/groups\.pilotnet\/forward\.test_skill\.mean must be 1 − /);
    // On the shuffled rung each training is tested on its own frames: only the side of the line is checked.
    const shuffled = patch(resultsFixture(), "groups.pilotnet/leaky.test_skill", { mean: -0.6951777264965134, sd: 0.009588771929433299 });
    expect(refusal(readLadder, shuffled)?.message).toMatch(/groups\.pilotnet\/leaky\.test_skill\.mean must be above 0, as its error is below steering straight’s/);
  });

  it("refuse a ladder error whose seed values no longer make its mean", () => {
    const json = patch(resultsFixture(), "ladder.sightlinenet.1.mae_deg.values", [11.9, 9.618937237003818, 10.12272448789344]);
    expect(refusal(readLadder, json)?.message).toMatch(/ladder\.sightlinenet\.1\.mae_deg\.mean must be the mean of its values/);
  });

  it("refuse steering straight on a rung that its group does not hold", () => {
    const json = patch(resultsFixture(), "groups.pilotnet/forward.test_zero_mae_deg", 9.5);
    expect(refusal(readLadder, json)?.message).toMatch(/groups\.pilotnet\/forward\.test_zero_mae_deg must be 9\.40/);
  });
});

describe("the readers' checks of the sentences built on them", () => {
  it("refuse a standard-recipe score that is not on the split's own test frames", () => {
    expect(refusal(readLegacy, patch(resultsFixture(), "legacy.eval.drive2018_blocked_test.model.n", 5000))?.message).toMatch(
      /drive2018_blocked_test\.model\.n must be 6000, the blocked split’s test frames/,
    );
    expect(refusal(readLegacy, patch(resultsFixture(), "legacy.eval.drive2018_forward_test.model.n", 9000))?.message).toMatch(
      /drive2018_forward_test\.model\.n must be 9324, the later-in-the-drive split’s test frames/,
    );
    expect(refusal(readLegacy, patch(resultsFixture(), "legacy.splits.firewall", 300))?.message).toMatch(/legacy\.splits\.firewall must be 250/);
  });

  it("count the boundaries from the pattern, and refuse drops that do not match them", () => {
    expect(readSplits(resultsFixture()).blocked.boundaries).toBe(24);
    const json = patch(resultsFixture(), "split_rules.blocked.pattern", "TTVTTETT");
    expect(refusal(readSplits, json)?.message).toMatch(/split_sizes\.blocked\.dropped must be/);
  });

  it("refuse shares that the sizes do not follow", () => {
    expect(refusal(readSplits, patch(resultsFixture(), "split_rules.forward.train_frac", 0.6))?.message).toMatch(/split_sizes\.forward\.train must be/);
    expect(refusal(readSplits, patch(resultsFixture(), "split_rules.leaky.val_frac", 0.2))?.message).toMatch(/split_sizes\.leaky\.val must be/);
  });
});

describe("readYardsticks", () => {
  it("reads every rule beside Sightline on the blocked test frames and on another day, each on the same frames", () => {
    const [held, day] = readYardsticks(resultsFixture());
    expect(held).toEqual({
      kind: "held-back",
      drive: "drive2018",
      frames: 6000,
      model: 10.12272448789344,
      errors: { zero: 17.479270016382138, mean: 17.84318643687274, zoh: 0.5927266640439629, extrap: 0.6901967614466945 },
    });
    expect(day).toEqual({
      kind: "another-day",
      drive: "drive2017",
      frames: 45404,
      model: 9.771067024628094,
      errors: { zero: 15.52269723393061, mean: 15.473928367779282, zoh: 0.5068826531183521, extrap: 0.48579622978518006 },
    });
  });

  it("refuses a rule scored on other frames, another run, or steering straight the ladder does not hold", () => {
    expect(refusal(readYardsticks, patch(resultsFixture(), "baselines.test.baselines.zoh.n", 5998))?.message).toMatch(/baselines\.test\.baselines\.zoh\.n must be 6000/);
    expect(refusal(readYardsticks, patch(resultsFixture(), "baselines.cross_drive.model.n", 45000))?.message).toMatch(/baselines\.cross_drive\.model\.n/);
    expect(refusal(readYardsticks, patch(resultsFixture(), "baselines.run", "drive2018_sightlinenet_blocked_s0"))?.message).toMatch(/baselines\.run/);
    expect(refusal(readYardsticks, patch(resultsFixture(), "baselines.test.n", 5000))?.message).toMatch(/baselines\.test/);
    expect(refusal(readYardsticks, patch(resultsFixture(), "baselines.cross_drive.baselines.zero.mae_deg", 15.6))?.message).toMatch(/cross_zero_mae_deg|baselines\.cross_drive\.baselines\.zero/);
    expect(refusal(readYardsticks, patch(resultsFixture(), "baselines.test.baselines.extrap.mae_deg", -1))?.message).toMatch(/extrap\.mae_deg/);
  });
});

describe("readFrames", () => {
  it("counts the shipped network's frames from its lags, and the other's from its key", () => {
    expect(readFrames(resultsFixture())).toEqual({ sightlinenet: 2, "sightlinenet-1f": 1 });
    expect(readFrames(patch(resultsFixture(), "timing.drive2018.lag_gap_s", [{ lag: 1, median: 0.05 }, { lag: 2, median: 0.1 }]))).toEqual({ sightlinenet: 3, "sightlinenet-1f": 1 });
    expect(readFrames(patch(resultsFixture(), "shipped.model", "pilotnet"))).toBeNull();
    expect(readFrames(patch(resultsFixture(), "timing.drive2018", null))).toBeNull();
  });
});

// -- The frame pairs and the split timeline -----------------------------------------

import pairsFile from "../../public/data/leakage/pairs.json?raw";
import clipFile from "../../public/data/clips/c2018/clip.json?raw";
import { readClipWindow, readPairs, readTimeline } from "./figures.ts";

const real = () => JSON.parse(shipped) as Record<string, unknown>;
const pairsJson = () => JSON.parse(pairsFile) as Record<string, unknown>;

describe("readPairs", () => {
  it("reads every pair: the test frame, and its closest training frame under each split, with their gaps, likeness and labels", () => {
    const pairs = readPairs(pairsJson(), real());
    expect(pairs.drive).toBe("drive2018");
    expect(pairs.pairs).toHaveLength(6);
    expect(pairs.pairs[0]).toEqual({
      frame: 7917,
      image: "frame-7917.jpg",
      angle: 0.4000000059604645,
      shuffled: { frame: 7916, image: "frame-7916.jpg", gapFrames: 1, gapS: 0.041, pixelRmse: 8.786114474962163, angle: 0.5 },
      blocked: { frame: 28219, image: "frame-28219.jpg", gapFrames: 20302, gapS: 1073.237, pixelRmse: 28.473862043437446, angle: 1.409999966621399 },
    });
    expect(pairs.nearestBlocked).toBe(501);
  });

  it("refuses a pair whose gap is not its frames' distance, a blocked neighbour nearer than the split allows, or an image outside its folder", () => {
    const bad = (path: string, value: unknown) => refusal((json) => readPairs(patch(json, path, value), real()), pairsJson());
    expect(bad("pairs.0.leaky.gap_frames", 2)?.message).toMatch(/pairs\.0\.leaky\.gap_frames/);
    expect(bad("pairs.0.blocked.nn_frame", 7917 + 400)?.message).toMatch(/pairs\.0\.blocked/);
    expect(bad("pairs.0.image", "../secret.jpg")?.message).toMatch(/pairs\.0\.image/);
    // A blocked neighbour nearer than any test frame comes to training under the blocked split.
    const near = patch(patch(pairsJson(), "pairs.0.blocked.nn_frame", 7917 + 400), "pairs.0.blocked.gap_frames", 400);
    expect(refusal((json) => readPairs(json, real()), near)?.message).toMatch(/pairs\.0\.blocked must be a neighbour at least 501 frames away/);
    expect(bad("pairs.1.blocked.gap_s", -1)?.message).toMatch(/pairs\.1\.blocked\.gap_s/);
    expect(bad("drive", "drive2017")?.message).toMatch(/drive/);
    expect(bad("seed", 4)?.message).toMatch(/seed/);
  });
});

describe("readTimeline", () => {
  it("lays the drive out in time from its own clock, with each split drawn exactly as the runs recorded it", () => {
    const t = readTimeline(real());
    expect(t.drive).toBe("drive2018");
    expect(t.frames).toBe(63825);
    expect(t.clock.durationS).toBe(3351.698);
    expect(t.clock.startS).toHaveLength(998);
    expect(t.protocols.blocked.sizes).toEqual({ train: 39825, val: 6000, test: 6000, dropped: 12000 });
    expect(t.protocols.later.sizes).toEqual({ train: 44427, val: 9074, test: 9324, dropped: 1000 });
    expect(t.protocols.shuffled.sizes).toEqual({ train: 44677, val: 9574, test: 9574, dropped: 0 });
    expect(t.protocols.later.segments.map((s) => [s.from, s.to])).toEqual([[0, 44427], [44427, 44927], [44927, 54001], [54001, 54501], [54501, 63825]]);
  });

  it("refuses rules that do not reproduce the recorded sizes, or a clock that runs backwards", () => {
    expect(refusal(readTimeline, patch(real(), "split_rules.blocked.firewall", 200))?.message).toMatch(/split_sizes\.blocked|firewall/);
    expect(refusal(readTimeline, patch(real(), "split_sizes.forward.val", 9075))?.message).toMatch(/split_sizes\.forward/);
    expect(refusal(readTimeline, patch(real(), "drive_profile.drives.drive2018.start_s.5", 1))?.message).toMatch(/start_s\.5/);
    expect(refusal(readTimeline, patch(real(), "drive_profile.drives.drive2018.start_frame.3", 190))?.message).toMatch(/start_frame\.3/);
    expect(refusal(readTimeline, patch(real(), "drive_profile.drives.drive2018.duration_s", 3000))?.message).toMatch(/duration_s/);
  });
});

describe("readClipWindow", () => {
  it("reads where the demo's clip sits in its drive, in frames and in seconds", () => {
    expect(readClipWindow(JSON.parse(clipFile))).toEqual({ drive: "drive2018", title: "2018: minutes it never trained on", firstFrame: 28752, frames: 420, firstS: 1519.063, lastS: 1541.774, aspect: 455 / 256 });
  });
});
