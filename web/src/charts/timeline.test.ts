import { describe, expect, it } from "vitest";

import { blockedSegments, ditherRoles, laterSegments, roleSizes, shuffledSizes, timeOfFrame } from "./timeline.ts";

/** The roles of frames 0..n-1, frame by frame, as the splits module deals them. */
function blockedByFrame(n: number, block: number, firewall: number, pattern: string): string[] {
  const labels = Array.from({ length: n }, (_, i) => pattern[Math.floor(i / block) % pattern.length] as string);
  const keep = labels.map(() => true);
  for (let edge = 1; edge < n; edge += 1) {
    if (labels[edge] !== labels[edge - 1] && (labels[edge] === "T" || labels[edge - 1] === "T")) {
      for (let i = Math.max(0, edge - firewall); i < Math.min(n, edge + firewall); i += 1) keep[i] = false;
    }
  }
  const role: Record<string, string> = { T: "train", V: "val", E: "test" };
  return labels.map((label, i) => (keep[i] ? (role[label] as string) : "dropped"));
}

const byFrame = (segments: ReturnType<typeof blockedSegments>, n: number) => {
  const out = new Array<string>(n).fill("?");
  for (const s of segments) for (let i = s.from; i < s.to; i += 1) out[i] = s.role;
  return out;
};

describe("the blocked split's layout", () => {
  it("is the splits module's, frame for frame", () => {
    for (const [n, block, firewall, pattern] of [
      [63825, 1500, 250, "TTVTTET"],
      [10000, 1500, 250, "TTVTTET"],
      [3100, 400, 120, "TVTE"],
      [5000, 1000, 0, "TTVTTET"],
    ] as const) {
      const segments = blockedSegments(n, block, firewall, pattern);
      expect(byFrame(segments, n)).toEqual(blockedByFrame(n, block, firewall, pattern));
      // In order, touching, never empty.
      segments.forEach((s, i) => {
        expect(s.to).toBeGreaterThan(s.from);
        if (i > 0) expect(s.from).toBe(segments[i - 1]?.to);
      });
    }
  });

  it("reproduces the sizes the runs recorded on the 2018 drive", () => {
    expect(roleSizes(blockedSegments(63825, 1500, 250, "TTVTTET"))).toEqual({ train: 39825, val: 6000, test: 6000, dropped: 12000 });
  });
});

describe("the later-in-the-drive split's layout", () => {
  it("is train, a firewall each side of the boundary, validation, the same again, test", () => {
    const segments = laterSegments(63825, 0.7, 0.15, 250);
    expect(segments).toEqual([
      { role: "train", from: 0, to: 44427 },
      { role: "dropped", from: 44427, to: 44927 },
      { role: "val", from: 44927, to: 54001 },
      { role: "dropped", from: 54001, to: 54501 },
      { role: "test", from: 54501, to: 63825 },
    ]);
    expect(roleSizes(segments)).toEqual({ train: 44427, val: 9074, test: 9324, dropped: 1000 });
  });
});

describe("the shuffled split", () => {
  it("deals its shares as the splits module slices them", () => {
    expect(shuffledSizes(63825, 0.7, 0.15)).toEqual({ train: 44677, val: 9574, test: 9574, dropped: 0 });
  });

  it("is drawn as an even dither at its shares: an illustration, the same every time", () => {
    const roles = ditherRoles(400, { train: 44677, val: 9574, test: 9574 });
    expect(roles).toEqual(ditherRoles(400, { train: 44677, val: 9574, test: 9574 }));
    const count = (role: string) => roles.filter((r) => r === role).length;
    expect(count("train")).toBeGreaterThan(275);
    expect(count("train")).toBeLessThan(285);
    expect(Math.abs(count("val") - count("test"))).toBeLessThanOrEqual(4);
    // Mixed, not in runs: no role holds more than a handful of cells in a row.
    let run = 1;
    let longest = 1;
    for (let i = 1; i < roles.length; i += 1) {
      run = roles[i] === roles[i - 1] ? run + 1 : 1;
      longest = Math.max(longest, run);
    }
    expect(longest).toBeLessThan(8);
  });
});

describe("the drive's clock", () => {
  // Three buckets of four frames, the last one short: its frames are uneven in time.
  const profile = { bucketFrames: 4, frames: 10, startFrame: [0, 4, 8], startS: [0, 2, 10], durationS: 11 };

  it("reads each frame's time from its bucket's start, never frames times a typical interval", () => {
    expect(timeOfFrame(profile, 0)).toBe(0);
    expect(timeOfFrame(profile, 2)).toBe(1);
    expect(timeOfFrame(profile, 4)).toBe(2);
    expect(timeOfFrame(profile, 6)).toBe(6);
    // The last bucket holds the two frames left over and runs to the drive's end.
    expect(timeOfFrame(profile, 9)).toBe(10.5);
    expect(timeOfFrame(profile, 10)).toBe(11);
  });
});
