/**
 * What the "own video" section says: how it reads a video, how far it has
 * got, and what went wrong. Every number in these sentences is worked out
 * from the data the page loads -- the clips' camera size, input size and
 * lags, the measured frame gaps in results.json -- or from the page's own
 * limit (`MAX_FRAMES`); none is typed in. The page is served with plain
 * versions of the rules, without numbers (ownclip-markup.ts), and these
 * replace them once the data has loaded.
 */

import { CROP_ROWS } from "../cockpit/geometry.ts";
import { driveName, framesBack } from "../data/timing.ts";
import { countWord, fmtCount, fmtFixed, fmtMs, fmtPct, ordinalWord } from "../format.ts";
import { MAX_FRAMES, type Failure, type Pace } from "./ownclip-plan.ts";

/** What the page knows about how it reads a video, once its data has loaded. */
export interface Reading {
  readonly pace: Pace;
  /** The model's history offsets, frames back. */
  readonly lags: readonly number[];
  /** The camera frame the model was trained on, [width, height]. */
  readonly camera: readonly [number, number];
  /** The model's input, in pixels. */
  readonly inputW: number;
  readonly inputH: number;
  /** The drive whose gaps set the pace, and whether the shipped model was
   * trained on it. */
  readonly drive: string | null;
  readonly trainedOnDrive: boolean;
}

/** A gap in seconds as the page prints it, "100 ms". */
const ms = (seconds: number) => fmtMs(seconds * 1000, 0);

/** Seconds of video as a sentence says them: "30 seconds", "34.3 seconds". */
export function secondsText(seconds: number): string {
  const whole = Math.abs(seconds - Math.round(seconds)) < 0.05;
  const text = fmtFixed(seconds, whole ? 0 : 1);
  return `${text} second${text === "1" ? "" : "s"}`;
}

/** How much video the page reads at most, in seconds. */
export function capSeconds(pace: Pace): number {
  return MAX_FRAMES * pace.stepS;
}

/** How many frames one answer reads: the frame itself and its history. */
const framesPerAnswer = (reading: Reading) => reading.lags.length + 1;

// -- How it reads a video -------------------------------------------------------------

export function cropRule(reading: Reading): string {
  const [w, h] = reading.camera;
  const share = fmtPct(Math.min(CROP_ROWS, h) / h);
  return (
    `It reads the bottom ${share} of each frame, at full width, as it did in training. ` +
    `From a portrait video it first takes the band across the middle with the shape of the camera it learned from ` +
    `(${w} × ${h}), then the bottom ${share} of that.`
  );
}

export function paceRule(reading: Reading): string {
  const { pace } = reading;
  if (pace.lag === 0) {
    return `It reads ${fmtFixed(pace.rate, 1)} frames a second and answers every one.`;
  }
  const first = `Its first answer is on the ${ordinalWord(pace.lag + 1)} frame it reads, as each one needs the frame ${framesBack(pace.lag)} before.`;
  if (pace.gapS === null) {
    return `It reads ${fmtFixed(pace.rate, 1)} frames a second, the clips’ own rate, and each answer compares a frame with the one ${framesBack(pace.lag)} before it. ${first}`;
  }
  const where = reading.drive === null ? "" : ` on ${driveName(reading.drive)}${reading.trainedOnDrive ? ", which it learned from" : ""}`;
  return (
    `It reads a frame every ${ms(pace.stepS)}, so each answer compares a frame with the one ${ms(pace.gapS)} before it: ` +
    `the median gap between a frame and the one ${framesBack(pace.lag)} before it${where}. ${first}`
  );
}

export function capRule(reading: Reading): string {
  return `It reads up to ${fmtCount(MAX_FRAMES)} frames, so a longer video is read for its first ${secondsText(capSeconds(reading.pace))}.`;
}

export function eyeCaptionText(reading: Reading): string {
  return `What the model gets from this frame, ${reading.inputW} × ${reading.inputH} pixels, with where it looked.`;
}

// -- How far it has got ------------------------------------------------------------------

export const OPENING = "Opening your video";

export function readingStatus(read: number, total: number): string {
  return `Reading frame ${fmtCount(Math.min(total, read + 1))} of ${fmtCount(total)}`;
}

export function answeredNote(answered: number, answerable: number, reading: Reading): string {
  if (answered > 0) return `${fmtCount(answered)} of ${fmtCount(answerable)} answers so far.`;
  return `The model answers as soon as it has ${countWord(framesPerAnswer(reading))} frames to compare.`;
}

export function finishingStatus(answered: number, answerable: number): string {
  return `Finishing the answers: ${fmtCount(answered)} of ${fmtCount(answerable)}`;
}

/** The line under the player once a video has been read: the file's own
 * name, so the visitor knows what is playing, and what was read of it. */
export function doneNote(name: string, total: number, answerable: number, capped: boolean, reading: Reading): string {
  const read = `${name}: ${fmtCount(total)} frames read, ${fmtCount(answerable)} answers.`;
  return capped ? `${read} The video is longer than the page reads, so these are its first ${secondsText(capSeconds(reading.pace))}.` : read;
}

/** Where the player is: "Frame 34 of 600, 1.7 s in". */
export function frameWhen(k: number, total: number, pace: Pace): string {
  return `Frame ${fmtCount(k + 1)} of ${fmtCount(total)}, ${fmtFixed((k + 0.5) * pace.stepS, 1)} s in`;
}

/** The scrubber's spoken value at frame `k`, with the model's angle there
 * as `steerParts` gives it. */
export function scrubText(k: number, total: number, pace: Pace, angle: { size: string; side: string }): string {
  const said = angle.side === "" ? angle.size : `${angle.size} ${angle.side}`;
  return `${frameWhen(k, total, pace)}: the model estimates ${said}`;
}

// -- What went wrong ---------------------------------------------------------------------

export interface FailureWords {
  readonly title: string;
  readonly text: string;
}

/** The message for each kind of failure; `reading` (null until the data has
 * loaded) lets the one for a too-short video say what one answer needs. */
export function failureWords(failure: Failure, reading: Reading | null): FailureWords {
  switch (failure) {
    case "not-a-video":
      return {
        title: "That file is not a video",
        text: "The browser found no moving picture in it. Choose a video file, such as an MP4 from a phone or a dashcam.",
      };
    case "codec":
      return {
        title: "This browser cannot decode that video",
        text: "Its picture is in a format this browser does not play. Try another browser, or save the video as an MP4 (H.264), which most browsers play.",
      };
    case "empty": {
      const needs =
        reading === null || reading.pace.lag === 0 ? "too short for the model to read one frame"
        : reading.pace.gapS === null ? `too short for one answer, which needs a frame and the one ${framesBack(reading.pace.lag)} before it`
        : `too short for one answer, which needs ${countWord(framesPerAnswer(reading))} frames ${ms(reading.pace.gapS)} apart`;
      return { title: "There is nothing to read in that file", text: `It is empty, or ${needs}.` };
    }
    case "too-long":
      return {
        title: "That video does not say how long it is",
        text: "The browser reports no end to it, which a live recording can do, so its frames cannot be placed in time. Saving it again as an ordinary video file gives it a length.",
      };
    case "unknown":
      return {
        title: "The model could not read that video",
        text: "Something went wrong part way, so it stopped. Choose the video again, or try another; the demo above is not affected.",
      };
  }
}
