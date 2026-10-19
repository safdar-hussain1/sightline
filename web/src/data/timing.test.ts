import { describe, expect, it } from "vitest";

import { driveIntervals, frameTiming, lagText } from "./timing.ts";

describe("driveIntervals", () => {
  it("reads each drive's measured frame interval, null for a drive without timestamps", () => {
    const raw = { drives: { drive2018: { median_dt_s: 0.057, frames: 63825 }, drive2017: { median_dt_s: null, frames: 45406 } } };
    expect(driveIntervals(raw)).toEqual({ drive2018: 0.057, drive2017: null });
  });

  it("treats anything but a positive interval as no timestamps", () => {
    const raw = { drives: { a: { median_dt_s: 0 }, b: { median_dt_s: "0.05" }, c: {}, d: { median_dt_s: -1 } } };
    expect(driveIntervals(raw)).toEqual({ a: null, b: null, c: null, d: null });
    expect(driveIntervals({})).toEqual({});
  });
});

describe("frameTiming", () => {
  const intervals = { drive2018: 0.057, drive2017: null };

  it("times a drive with timestamps by its own interval", () => {
    expect(frameTiming(intervals, "drive2018")).toEqual({ interval: 0.057, reference: null });
  });

  it("gives a drive without them another drive's measured interval to quote", () => {
    expect(frameTiming(intervals, "drive2017")).toEqual({ interval: null, reference: { drive: "drive2018", interval: 0.057 } });
    expect(frameTiming({ drive2017: null }, "drive2017")).toEqual({ interval: null, reference: null });
    expect(frameTiming({}, "drive2018")).toEqual({ interval: null, reference: null });
  });
});

describe("lagText", () => {
  it("says how long before a history frame was taken: in milliseconds when measured, in frames otherwise", () => {
    expect(lagText(2, { interval: 0.056999921798706055, reference: null })).toBe("114 ms");
    expect(lagText(1, { interval: 0.056999921798706055, reference: null })).toBe("57 ms");
    expect(lagText(2, { interval: null, reference: null })).toBe("two frames");
    expect(lagText(1, { interval: null, reference: null })).toBe("one frame");
  });
});
