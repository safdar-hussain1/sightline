import { describe, expect, it } from "vitest";

import { compareAngles } from "./live.ts";

describe("compareAngles", () => {
  it("compares only the frames this browser has run", () => {
    const angles = Float32Array.from([1.5, 99, -2.25, 0]);
    const done = Uint8Array.from([1, 0, 1, 0]);
    expect(compareAngles(angles, done, [1.5, 3, -2.0, 7])).toEqual({ compared: 2, maxDiff: 0.25 });
  });

  it("states no difference before any frame has run", () => {
    const result = compareAngles(new Float32Array(3), new Uint8Array(3), [1, 2, 3]);
    expect(result.compared).toBe(0);
    expect(result.maxDiff).toBeNaN();
  });

  it("says exactly zero when every answer is the published one, bit for bit", () => {
    const expected = [-12.135157585144043, 4.879011154174805];
    expect(compareAngles(Float32Array.from(expected), Uint8Array.from([1, 1]), expected)).toEqual({ compared: 2, maxDiff: 0 });
  });

  it("counts an answer that is not a number as an infinite difference, wherever it falls", () => {
    for (const angles of [[Number.NaN, 1, 1], [1, Number.NaN, 1], [1, 1, Number.NaN]]) {
      const result = compareAngles(Float32Array.from(angles), Uint8Array.from([1, 1, 1]), [1, 1, 1]);
      expect(result, String(angles)).toEqual({ compared: 3, maxDiff: Number.POSITIVE_INFINITY });
    }
  });
});
