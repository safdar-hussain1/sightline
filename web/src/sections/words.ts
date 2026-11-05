/**
 * The sentences of the demo view: the accuracy figure's and How it works',
 * and the few small words every section shares.
 *
 * Every number in them is read from results.json, a clip.json, or what this
 * browser measured itself: a ratio is put in words only to the nearest half,
 * and a time between frames is the drive's measured median, said as the
 * drive's. The Results view's sentences live in copy.ts, which the page loads
 * with that view.
 */

import { AssetError } from "../data/load.ts";
import { driveName, framesBack, medianGap, type FrameTiming } from "../data/timing.ts";
import type { Clip } from "../data/types.ts";
import { countWord, fmtCount, fmtDeg, fmtMs } from "../format.ts";
import type { Figures } from "./reader.ts";

export const capitalised = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

/** "about three times" for a ratio of 3.05: the ratio to the nearest half, in
 * words while it is small enough to say. Meant for ratios of 1.25 or more. */
export function timesInWords(ratio: number): string {
  const half = Math.round(ratio * 2) / 2;
  const whole = Math.floor(half);
  if (whole > 12) {
    return `${fmtCount(Math.round(ratio))} times`;
  }
  return half === whole ? `${countWord(whole)} times` : `${countWord(whole)} and a half times`;
}

/** Why the measurements are not shown, for the visitor: what went wrong in
 * plain words, never the URL (the console has it). */
export function measurementsFailure(error: unknown): string {
  if (error instanceof AssetError) {
    if (error.kind === "missing") return "The measurements are not on this copy of the site.";
    if (error.kind === "unreachable") return "The measurements could not be downloaded. Check the connection, then reload the page.";
  }
  return "The measurements file is not in the form this page reads, so its figures are not shown.";
}

// -- The accuracy figure ----------------------------------------------------------

export function accuracySentence(heldOut: Figures["heldOut"]): string {
  return (
    `On minutes of driving it never trained on, Sightline is off by ${fmtDeg(heldOut.model)} on average. ` +
    `Always steering straight is off by ${fmtDeg(heldOut.straight)} on the same frames.`
  );
}

/** The frames are the split's test frames: the split also holds back
 * validation frames and the gaps around them, which this figure does not use. */
export function accuracyCaption(heldOut: Figures["heldOut"]): string {
  return `Mean error on the ${fmtCount(heldOut.frames)} test frames held back from its training.`;
}

/** The figure's text alternative, from the same numbers it draws. */
export function accuracyLabel(heldOut: Figures["heldOut"]): string {
  return `Mean error on the same ${fmtCount(heldOut.frames)} frames: Sightline ${fmtDeg(heldOut.model)}, always steering straight ${fmtDeg(heldOut.straight)}.`;
}

/** What the paired measurement means, from its ratio: the blocked-trained
 * model's error over the shuffled-trained one's on the same held-out frames. */
export function leakageSentence(ratio: number): string {
  const tail = "better than it is on road it has not seen nearby.";
  if (!(ratio > 1)) {
    return "Tested the usual way, on shuffled frames, the same kind of model looks no better than it is on road it has not seen nearby.";
  }
  const how = ratio < 1.25 ? "a little" : `about ${timesInWords(ratio)}`;
  return `Test it the usual way, on shuffled frames, and the same kind of model looks ${how} ${tail}`;
}

// -- How it works ------------------------------------------------------------------------

/** The first step's title: how many frames the model is shown, and, on a
 * drive whose frames carry timestamps, the drive's median time between the
 * oldest of them and the frame on screen -- said as the drive's, since the
 * clip's own frames have their own spread. Elsewhere it counts frames. */
export function framesTitle(clip: Clip, timing: FrameTiming): string {
  const lags = clip.frame_lags;
  if (lags.length === 0) return "One frame at a time";
  const oldest = Math.max(...lags);
  const gap = medianGap(oldest, timing.gaps);
  if (gap === null) {
    return lags.length === 1 ? `This frame and the one ${framesBack(oldest)} before it` : `This frame and the ${countWord(lags.length)} before it`;
  }
  const count = capitalised(countWord(lags.length + 1));
  const drive = driveName(timing.drive);
  return lags.length === 1 ? `${count} frames, ${gap} apart on ${drive}` : `${count} frames, the oldest ${gap} earlier on ${drive}`;
}

/** The labels of the frames shown in the first step, the frame on screen
 * first: how many frames back each is, which is true of every frame of the clip. */
export function framesAges(clip: Clip, _timing?: FrameTiming): string[] {
  return ["now", ...clip.frame_lags.map((lag) => `${framesBack(lag)} before`)];
}

export function framesText(clip: Clip, timing: FrameTiming): string {
  const size = `${clip.sprite.frame_w} × ${clip.sprite.frame_h} pixels`;
  const lags = clip.frame_lags;
  if (lags.length === 0) {
    return `The frame on screen, cropped to the road and shrunk to ${size}.`;
  }
  const history = lags.length === 1 ? `the one ${framesBack(lags[0] as number)} before it` : `the ones ${lags.map(framesBack).join(" and ")} before it`;
  const oldest = Math.max(...lags);
  const quoted = timing.gaps === null && timing.reference !== null ? medianGap(oldest, timing.reference.gaps) : null;
  const spacing =
    timing.gaps !== null
      ? ""
      : quoted === null || timing.reference === null
        ? " This drive’s frames carry no timestamps, so their spacing is counted in frames."
        : ` This drive’s frames carry no timestamps; on ${driveName(timing.reference.drive)}, whose frames do, ` +
          `a frame and the one ${countWord(oldest)} before it were taken ${quoted} apart.`;
  return (
    `The frame on screen and ${history}, cropped to the road and shrunk to ${size}.${spacing} ` +
    `${capitalised(countWord(lags.length + 1))} frames show which way the view is moving, which one frame cannot.`
  );
}

export function speedText(msPerFrame: number | null): string {
  return msPerFrame === null ? "" : `Here it takes about ${fmtMs(msPerFrame)} a frame.`;
}
