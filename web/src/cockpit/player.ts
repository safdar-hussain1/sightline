/**
 * Which clip frame is on screen.
 *
 * The clip video holds exactly the clip's scored frames at a constant rate, so
 * video frame k is scored frame k and starts at k / fps seconds. While the
 * video plays, `requestVideoFrameCallback` reports the media time of every
 * frame the browser actually presents, which is what keeps the wheel and the
 * heat on the frame the visitor sees. Browsers without it are polled once per
 * animation frame through `currentTime` instead.
 */

/** The frame on screen at `mediaTime` seconds, clamped to the clip. The small
 * epsilon absorbs float error: k / fps * fps can come out a hair under k. */
export function frameIndexAt(mediaTime: number, fps: number, n: number): number {
  if (!Number.isFinite(mediaTime)) {
    return 0;
  }
  return Math.min(n - 1, Math.max(0, Math.floor(mediaTime * fps + 1e-6)));
}

/** Where to seek the video to show frame `k`: the middle of its time on
 * screen, so a decoder that rounds a seek either way still lands on it. */
export function seekTime(k: number, fps: number): number {
  return (k + 0.5) / fps;
}

/** The part of a video element the loop reads. */
export interface FrameClock {
  currentTime: number;
  requestVideoFrameCallback?: (callback: (now: number, metadata: { mediaTime: number }) => void) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
}

/** A scheduler for the fallback: requestAnimationFrame and its cancel. */
export interface Ticker {
  request(callback: () => void): number;
  cancel(handle: number): void;
}

const animationFrames: Ticker = {
  request: (callback) => requestAnimationFrame(() => callback()),
  cancel: (handle) => cancelAnimationFrame(handle),
};

export interface FrameLoop {
  stop(): void;
}

/** Call `onFrame(k)` whenever a new clip frame reaches the screen, until
 * `stop()`. A frame presented twice in a row is reported once. */
export function startFrameLoop(
  video: FrameClock,
  fps: number,
  n: number,
  onFrame: (k: number) => void,
  ticker: Ticker = animationFrames,
): FrameLoop {
  let last = -1;
  let stopped = false;
  const report = (mediaTime: number) => {
    const k = frameIndexAt(mediaTime, fps, n);
    if (k !== last) {
      last = k;
      onFrame(k);
    }
  };

  const request = video.requestVideoFrameCallback?.bind(video);
  const cancel = video.cancelVideoFrameCallback?.bind(video);
  if (request && cancel) {
    let handle = 0;
    const presented = (_now: number, metadata: { mediaTime: number }) => {
      if (stopped) return;
      report(metadata.mediaTime);
      handle = request(presented);
    };
    handle = request(presented);
    return {
      stop: () => {
        stopped = true;
        cancel(handle);
      },
    };
  }

  let handle = 0;
  const tick = () => {
    if (stopped) return;
    report(video.currentTime);
    handle = ticker.request(tick);
  };
  handle = ticker.request(tick);
  return {
    stop: () => {
      stopped = true;
      ticker.cancel(handle);
    },
  };
}
