import { describe, expect, it } from "vitest";

import { CROP_ROWS, cropBand, videoOverhang } from "./geometry.ts";

describe("cropBand", () => {
  it("is the bottom rows of the camera frame the model sees, as shares of its height", () => {
    const band = cropBand(256);
    expect(band.top).toBeCloseTo((256 - CROP_ROWS) / 256, 12);
    expect(band.height).toBeCloseTo(CROP_ROWS / 256, 12);
    expect(band.top + band.height).toBeCloseTo(1, 12); // it runs to the bottom edge
  });

  it("is the whole frame when the frame is shorter than the crop", () => {
    expect(cropBand(CROP_ROWS - 10)).toEqual({ top: 0, height: 1 });
  });
});

describe("videoOverhang", () => {
  it("is how much wider the video is than the camera picture in it", () => {
    expect(videoOverhang([455, 256], [456, 256])).toBeCloseTo(456 / 455, 12);
    expect(videoOverhang([456, 256], [456, 256])).toBe(1);
  });

  it("is one when the clip has no video", () => {
    expect(videoOverhang([455, 256], null)).toBe(1);
  });
});
