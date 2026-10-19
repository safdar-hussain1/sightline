import { describe, expect, it } from "vitest";

import { AssetError } from "../data/load.ts";
import { readFigures } from "./figures.ts";
import { patch, resultsFixture } from "./fixture.ts";

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
