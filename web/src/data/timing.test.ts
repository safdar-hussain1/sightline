import { describe, expect, it } from "vitest";

import copySource from "../sections/copy.ts?raw";
import { resultsFixture } from "../sections/fixture.ts";
import { driveGaps, driveName, frameTiming, framesBack, medianGap, timedLags } from "./timing.ts";
import timingSource from "./timing.ts?raw";

describe("driveGaps", () => {
  it("reads each drive's median gap at each lag the model uses, null for a drive without timestamps", () => {
    const gaps = driveGaps(resultsFixture());
    expect(gaps["drive2017"]).toBeNull();
    expect([...(gaps["drive2018"] ?? new Map())]).toEqual([[2, 0.1]]);
  });

  it("treats gaps it cannot read as no timestamps, so the drive is only counted in frames", () => {
    const timing = (lagGap: unknown) => driveGaps({ timing: { d: { lag_gap_s: lagGap } } })["d"];
    expect(timing([{ lag: 2, median: 0 }])).toBeNull();
    expect(timing([{ lag: 0, median: 0.1 }])).toBeNull();
    expect(timing([{ lag: 2, median: "0.1" }])).toBeNull();
    expect(timing([{ lag: 2, median: 0.1 }, { lag: 2, median: 0.2 }])).toBeNull();
    expect(timing([])).toBeNull();
    expect(timing(undefined)).toBeNull();
    expect(driveGaps({})).toEqual({});
    expect(driveGaps({ timing: [] })).toEqual({});
  });
});

describe("timedLags", () => {
  it("lists the lags every timed drive measured", () => {
    expect(timedLags(resultsFixture())).toEqual([2]);
    const lags = (...lists: number[][]) =>
      timedLags({ timing: Object.fromEntries(lists.map((list, i) => [`d${i}`, { lag_gap_s: list.map((lag) => ({ lag, median: 0.05 * lag })) }])) });
    expect(lags([2, 1], [1, 2])).toEqual([1, 2]);
    expect(lags([2], [1])).toBeNull();
    expect(timedLags({ timing: { d: null } })).toBeNull();
    expect(timedLags({})).toBeNull();
  });
});

describe("frameTiming", () => {
  const gaps = driveGaps(resultsFixture());

  it("times a drive with timestamps by its own gaps", () => {
    expect(frameTiming(gaps, "drive2018")).toEqual({ drive: "drive2018", gaps: gaps["drive2018"], reference: null });
  });

  it("gives a drive without them another drive's measured gaps to quote", () => {
    expect(frameTiming(gaps, "drive2017")).toEqual({ drive: "drive2017", gaps: null, reference: { drive: "drive2018", gaps: gaps["drive2018"] } });
    expect(frameTiming({ drive2017: null }, "drive2017")).toEqual({ drive: "drive2017", gaps: null, reference: null });
    expect(frameTiming({}, "drive2018")).toEqual({ drive: "drive2018", gaps: null, reference: null });
  });
});

describe("the words for a history frame", () => {
  it("counts how far back it is in frames, which is true of every frame of a clip", () => {
    expect(framesBack(2)).toBe("two frames");
    expect(framesBack(1)).toBe("one frame");
    expect(framesBack(1500)).toBe("1,500 frames");
  });

  it("quotes the drive's median gap at that lag, the measured time, never a multiple of the frame interval", () => {
    const gaps = driveGaps(resultsFixture())["drive2018"] ?? null;
    expect(medianGap(2, gaps)).toBe("a median 100\u00A0ms");
    expect(medianGap(1, gaps)).toBeNull();
    expect(medianGap(2, null)).toBeNull();
  });

  it("names a drive by its year", () => {
    expect(driveName("drive2018")).toBe("the 2018 drive");
    expect(driveName("highway")).toBe("the drive highway");
  });
});

describe("the number words", () => {
  it("come from the page's one list, not a copy of it", () => {
    for (const source of [timingSource, copySource]) expect(source).not.toMatch(/"eleven"/);
  });
});
