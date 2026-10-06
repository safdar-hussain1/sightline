import { describe, expect, it } from "vitest";

import { frameIndexAt, seekTime, startFrameLoop, type FrameClock } from "./player.ts";

describe("frameIndexAt", () => {
  it("maps media time to the frame on screen", () => {
    expect(frameIndexAt(0, 17.5, 420)).toBe(0);
    expect(frameIndexAt(1.0, 17.5, 420)).toBe(17);
    expect(frameIndexAt(0.0571, 17.5, 420)).toBe(0);
  });

  it("clamps to the clip: never before frame 0, never past frame n - 1", () => {
    expect(frameIndexAt(-0.5, 17.5, 420)).toBe(0);
    expect(frameIndexAt(24, 17.5, 420)).toBe(419);
    expect(frameIndexAt(1e9, 17.5, 420)).toBe(419);
    expect(frameIndexAt(Number.NaN, 17.5, 420)).toBe(0);
  });

  it("lands every frame's own start time on that frame, despite float error", () => {
    for (let k = 0; k < 420; k += 1) {
      expect(frameIndexAt(k / 17.5, 17.5, 420), `frame ${k}`).toBe(k);
      expect(frameIndexAt(k * (1 / 17.5), 17.5, 420), `frame ${k}`).toBe(k);
    }
  });
});

describe("seekTime", () => {
  it("aims at the middle of a frame, so a decoder rounding either way shows that frame", () => {
    for (let k = 0; k < 420; k += 1) {
      expect(frameIndexAt(seekTime(k, 17.5), 17.5, 420)).toBe(k);
      expect(frameIndexAt(seekTime(k, 17.5) - 0.4 / 17.5, 17.5, 420)).toBe(k);
      expect(frameIndexAt(seekTime(k, 17.5) + 0.4 / 17.5, 17.5, 420)).toBe(k);
    }
  });
});

describe("startFrameLoop", () => {
  it("reports each new frame the video presents, from its media time", () => {
    const pending: Array<(now: number, meta: { mediaTime: number }) => void> = [];
    const cancelled: number[] = [];
    const video: FrameClock = {
      currentTime: 0,
      requestVideoFrameCallback: (cb) => pending.push(cb),
      cancelVideoFrameCallback: (id) => void cancelled.push(id),
    };
    const seen: number[] = [];
    const loop = startFrameLoop(video, 17.5, 420, (k) => seen.push(k));
    for (const t of [0, 1 / 17.5, 1 / 17.5, 2 / 17.5, 1.0]) {
      const cb = pending.shift();
      cb?.(0, { mediaTime: t });
    }
    expect(seen).toEqual([0, 1, 2, 17]); // a repeated frame is reported once
    loop.stop();
    expect(cancelled).toEqual([pending.length]); // the one still pending
    const before = seen.length;
    pending.shift()?.(0, { mediaTime: 3 });
    expect(seen.length).toBe(before); // nothing after stop
  });

  it("falls back to polling currentTime once per animation frame", () => {
    const ticks: Array<() => void> = [];
    const video: FrameClock = { currentTime: 0 };
    const seen: number[] = [];
    let cancelledTick = -1;
    const loop = startFrameLoop(video, 17.5, 420, (k) => seen.push(k), {
      request: (cb) => ticks.push(cb),
      cancel: (id) => {
        cancelledTick = id;
      },
    });
    for (const t of [0, 0.01, 0.06, 0.06, 0.2]) {
      video.currentTime = t;
      ticks.shift()?.();
    }
    expect(seen).toEqual([0, 1, 3]);
    loop.stop();
    expect(cancelledTick).toBe(ticks.length);
  });
});
