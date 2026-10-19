/**
 * The sentences the page's sections say about the measurements and the clip.
 *
 * Every number in them is read from results.json, a clip.json, or what this
 * browser measured itself, and each sentence is built so it cannot say more
 * than its numbers do: a ratio is put in words only to the nearest half, a
 * model is called better than steering straight only when its skill is above
 * zero, and this browser's angles are compared with Python's only where it
 * decoded the very pixels Python ran on.
 */

import type { Agreement, CockpitState } from "../cockpit/live.ts";
import { splitTitle } from "../cockpit/copy.ts";
import { AssetError } from "../data/load.ts";
import type { Clip } from "../data/types.ts";
import { fmtCount, fmtDeg, fmtMs, fmtPct, fmtSpread } from "../format.ts";
import type { Figures } from "./figures.ts";

const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];

/** A small count in words ("three"); larger ones in figures. */
export function countWord(n: number): string {
  return WORDS[n] ?? fmtCount(n);
}

const capitalised = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

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

export function accuracyCaption(heldOut: Figures["heldOut"]): string {
  return `Mean error on the ${fmtCount(heldOut.frames)} frames its training held back.`;
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

// -- The Results view ---------------------------------------------------------------

export function resultsLede(runs: number): string {
  return `How good a steering model looks depends on how it is tested. These are the headline numbers from ${fmtCount(runs)} training runs.`;
}

export function pairedNote(paired: Figures["paired"]): string {
  const { sightlinenet, pilotnet } = paired;
  const runs =
    sightlinenet.ratio.runs === pilotnet.ratio.runs
      ? `${countWord(sightlinenet.ratio.runs)} trainings`
      : `${countWord(sightlinenet.ratio.runs)} and ${countWord(pilotnet.ratio.runs)} trainings`;
  return (
    `Mean of ${runs} each way, each pair scored on the frames both of its splits held out. ` +
    `Spread ${fmtSpread(sightlinenet.ratio.sd)} for Sightline, ${fmtSpread(pilotnet.ratio.sd)} for PilotNet.`
  );
}

/** Another day's skill as a share, and which way it points: a negative skill
 * is a model worse than steering straight, and is said so. */
export function anotherDayFigure(anotherDay: Figures["anotherDay"]): { value: number; text: string; label: string } {
  const mean = anotherDay.skill.mean;
  const value = Math.abs(mean);
  const side = mean >= 0 ? "Better" : "Worse";
  return { value, text: fmtPct(value), label: `${side} than always steering straight, on a drive it never saw` };
}

export function anotherDayNote(anotherDay: Figures["anotherDay"]): string {
  return `Mean of ${countWord(anotherDay.skill.runs)} trainings, each scored against steering straight on the same frames of another day’s drive.`;
}

/** The text alternative of the paired figure's ruler: Sightline's network's
 * mean error on each seed's paired frames, trained each way. */
export function pairedRuler(sightline: Figures["paired"]["sightlinenet"]): string {
  return (
    `Sightline’s mean error on the frames both splits held out: ${fmtDeg(sightline.shuffled)} when trained on the shuffled split, ` +
    `${fmtDeg(sightline.blocked)} when trained on the blocked split.`
  );
}

/** The text alternative of the another-day ruler. */
export function anotherDayRuler(anotherDay: Figures["anotherDay"]): string {
  return `Mean error on another day’s drive: Sightline ${fmtDeg(anotherDay.model)}, always steering straight ${fmtDeg(anotherDay.straight)}.`;
}

export function memoriserNote(memoriser: Figures["memoriser"]): string {
  return (
    `On ${fmtCount(memoriser.frames)} held-out frames. It learns nothing, so its small error on the shuffled split ` +
    "only shows how close the test frames sit to training frames."
  );
}

export function splitsText(gaps: Figures["gaps"]): { shuffled: string; blocked: string } {
  return {
    shuffled: `Frames are dealt out at random, so ${fmtPct(gaps.shuffledWithinOne)} of test frames sit within one frame of a frame the model trained on.`,
    blocked: `Whole stretches of road are held back, so no test frame comes closer than ${fmtCount(gaps.blockedMinFrames)} frames to one the model trained on.`,
  };
}

// -- The live check -------------------------------------------------------------------

const PYTHON = "the model run in Python";

/** "to within 0.00002°", or "exactly"; a difference below what four decimals
 * (and two more) can show is stated as the bound it is under. */
function within(diff: number): string {
  if (diff === 0) return "exactly";
  return `to within ${fmtDeg(diff, 4).replace(/^</, "")}`;
}

function clipName(clip: Clip): string {
  const [year] = splitTitle(clip.title);
  return year ? `the demo’s ${year} clip` : "the demo clip";
}

/** The trust chip: this browser's angles on the loaded clip against the
 * published ones, or what the check is waiting for. */
export function parityText(agreement: Agreement | null, state: CockpitState): string {
  if (state === "unavailable") {
    return "The live check needs the demo, which could not run here.";
  }
  if (agreement === null || agreement.compared === 0) {
    return `Checking this browser against ${PYTHON}…`;
  }
  const { check, compared, total, maxDiffDeg } = agreement;
  const name = clipName(agreement.clip);
  if (check === "unverified") {
    return `This browser could not check its decoded frames, so its angles are not compared with ${PYTHON}.`;
  }
  if (!Number.isFinite(maxDiffDeg)) {
    return `This browser’s angles on ${name} do not match ${PYTHON}.`;
  }
  if (check === "differ") {
    return `This browser decodes ${name} a little differently, so its angles differ from ${PYTHON} by up to ${fmtDeg(maxDiffDeg, 2)} on the ${fmtCount(compared)} frames it has run.`;
  }
  if (compared < total) {
    const each = maxDiffDeg === 0 ? `identical to ${PYTHON}` : `${within(maxDiffDeg).replace(/^to /, "")} of ${PYTHON}`;
    return `Checking live: ${fmtCount(compared)} of ${fmtCount(total)} frames of ${name} so far, every one ${each}.`;
  }
  return `Checked live: on all ${fmtCount(total)} frames of ${name}, this browser’s angles match ${PYTHON} ${within(maxDiffDeg)}.`;
}

// -- How it works ------------------------------------------------------------------------

/** How long before the frame on screen each history frame was taken. */
function lagMs(clip: Clip, lag: number): string {
  return fmtMs((lag / clip.fps) * 1000, 0);
}

export function framesTitle(clip: Clip): string {
  const lags = clip.frame_lags;
  if (lags.length === 0) return "One frame at a time";
  const count = capitalised(countWord(lags.length + 1));
  if (lags.length === 1) return `${count} frames, ${lagMs(clip, lags[0] as number)} apart`;
  return `${count} frames, up to ${lagMs(clip, Math.max(...lags))} apart`;
}

/** The labels of the frames shown in the first step, the frame on screen first. */
export function framesAges(clip: Clip): string[] {
  return ["now", ...clip.frame_lags.map((lag) => `${lagMs(clip, lag)} before`)];
}

export function framesText(clip: Clip): string {
  const size = `${clip.sprite.frame_w} × ${clip.sprite.frame_h} pixels`;
  const lags = clip.frame_lags;
  if (lags.length === 0) {
    return `The frame on screen, cropped to the road and shrunk to ${size}.`;
  }
  const history =
    lags.length === 1
      ? `the one ${lagMs(clip, lags[0] as number)} before it`
      : `the ones ${lags.map((lag) => lagMs(clip, lag)).join(" and ")} before it`;
  return (
    `The frame on screen and ${history}, cropped to the road and shrunk to ${size}. ` +
    `${capitalised(countWord(lags.length + 1))} frames show which way the view is moving, which one frame cannot.`
  );
}

export function speedText(msPerFrame: number | null): string {
  return msPerFrame === null ? "" : `Here it takes about ${fmtMs(msPerFrame)} a frame.`;
}
