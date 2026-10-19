/**
 * The sentences the page's sections say about the measurements and the clip.
 *
 * Every number in them is read from results.json, a clip.json, or what this
 * browser measured itself, and each sentence is built so it cannot say more
 * than its numbers do: a ratio is put in words only to the nearest half, a
 * model is called better than steering straight only when its skill is above
 * zero, a bound on a difference is rounded up, never to nearest, and this
 * browser's angles pass the live check only within the clip's own tolerance,
 * on the very pixels Python ran on.
 */

import type { Agreement, CockpitState } from "../cockpit/live.ts";
import { splitTitle } from "../cockpit/copy.ts";
import { AssetError } from "../data/load.ts";
import { lagText, type FrameTiming } from "../data/timing.ts";
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

// -- The Results view ---------------------------------------------------------------

export function resultsLede(runs: number): string {
  return `How good a steering model looks depends on how it is tested. Here is what ${fmtCount(runs)} training runs found.`;
}

/** What the paired ratio measures: flattery only when it is above one. */
export function pairedLabel(ratio: number): string {
  return ratio > 1
    ? "How much better testing on shuffled frames makes the same kind of model look"
    : "The same kind of model’s error on held-back stretches, as a multiple of its error under a shuffled test";
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
    `Sightline’s mean error over ${countWord(sightline.ratio.runs)} trainings on the frames both splits held out: ` +
    `${fmtDeg(sightline.shuffled)} when trained on the shuffled split, ${fmtDeg(sightline.blocked)} when trained on the blocked split.`
  );
}

/** The text alternative of the another-day ruler. */
export function anotherDayRuler(anotherDay: Figures["anotherDay"]): string {
  return (
    `Mean error over ${countWord(anotherDay.skill.runs)} trainings on another day’s drive: ` +
    `Sightline ${fmtDeg(anotherDay.model)}, always steering straight ${fmtDeg(anotherDay.straight)}.`
  );
}

export function memoriserNote(memoriser: Figures["memoriser"]): string {
  return (
    `On ${fmtCount(memoriser.frames)} held-out frames. It learns nothing, so its error under each split ` +
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

const PYTHON = "the same model run in Python";

/** The smallest figure of one significant digit at or above `diff`
 * (degrees): a bound is rounded up, so it is never below what it bounds.
 * Anything under a millionth of a degree is stated as that. */
export function upperBound(diff: number): string {
  const floor = 1e-6;
  if (!(diff > floor)) return "0.000001°";
  const decimals = Math.max(1, Math.ceil(-Math.log10(diff)));
  const scale = 10 ** decimals;
  let units = Math.ceil(diff * scale);
  // Float error can carry the product a hair past a whole number.
  if ((units - 1) / scale >= diff) units -= 1;
  return `${String(Number((units / scale).toFixed(decimals)))}°`;
}

function clipName(clip: Clip): string {
  const [year] = splitTitle(clip.title);
  return year ? `the demo’s ${year} clip` : "the demo clip";
}

const MODEL_NAMES = { sightline: "Sightline", shuffled: "the twin" } as const;

/** How the check looks: still running, passed, failed, or not possible. */
export type ParityLook = "checking" | "checked" | "mismatch" | "note";

/**
 * The trust chip: the model's angles in this browser on the loaded clip
 * against the clip's published ones, computed in Python by the same runtime on
 * the reference decoder's frames. It passes only when this browser decoded
 * those very frames (`check` "match") and every frame run so far is within
 * the clip's `tolerance`; one frame beyond it fails the check. `state` is the
 * cockpit's, or "absent" when the demo's code never started.
 */
export function parityState(agreement: Agreement | null, state: CockpitState | "absent"): { look: ParityLook; text: string } {
  if (state === "absent") {
    return { look: "note", text: "The live check needs the demo, which could not start in this browser." };
  }
  if (state === "unavailable") {
    return { look: "note", text: "The live check needs the demo, and the demo is not running." };
  }
  if (agreement === null || agreement.compared === 0) {
    return { look: "checking", text: `Checking this browser against ${PYTHON}…` };
  }
  const { check, compared, total, maxDiffDeg, clip } = agreement;
  const name = clipName(clip);
  const model = MODEL_NAMES[agreement.model];
  const angles = `${model.charAt(0).toUpperCase()}${model.slice(1)}’s angles`;
  if (check === "unverified") {
    return {
      look: "note",
      text: "This browser could not check that it decoded the clip’s frames as Python did, so its angles are not compared.",
    };
  }
  if (!Number.isFinite(maxDiffDeg)) {
    return { look: "mismatch", text: `${angles} in this browser do not match ${PYTHON} on ${name}.` };
  }
  if (check === "differ") {
    return {
      look: "note",
      text:
        `This browser decodes ${name} differently from Python, so ${model}’s angles here differ from ${PYTHON} ` +
        `by up to ${upperBound(maxDiffDeg)} on the ${fmtCount(compared)} frames it has run.`,
    };
  }
  const tolerance = clip.tolerance?.angle_deg;
  if (tolerance === undefined) {
    return {
      look: "note",
      text:
        `${angles} in this browser differ from ${PYTHON} by at most ${upperBound(maxDiffDeg)} on the ` +
        `${fmtCount(compared)} frames it has run; the clip records no tolerance to judge that by.`,
    };
  }
  if (maxDiffDeg > tolerance) {
    return {
      look: "mismatch",
      text:
        `${angles} in this browser do not match ${PYTHON}: on ${name} they differ by up to ` +
        `${upperBound(maxDiffDeg)}, more than the ${upperBound(tolerance)} allowed.`,
    };
  }
  const within = maxDiffDeg === 0 ? null : upperBound(maxDiffDeg);
  if (compared < total) {
    const each = within === null ? `identical to ${PYTHON}` : `within ${within} of ${PYTHON}`;
    return {
      look: "checking",
      text: `Checking live: ${fmtCount(compared)} of ${fmtCount(total)} frames of ${name} so far, every one ${each}.`,
    };
  }
  return {
    look: "checked",
    text:
      `Checked live: on all ${fmtCount(total)} frames of ${name}, ${model}’s angles in this browser match ` +
      `${PYTHON} ${within === null ? "exactly" : `to within ${within}`}.`,
  };
}

// -- How it works ------------------------------------------------------------------------

/** "the 2018 drive" for "drive2018". */
function driveName(drive: string): string {
  const year = /(\d{4})$/.exec(drive)?.[1];
  return year ? `the ${year} drive` : "the other drive";
}

export function framesTitle(clip: Clip, timing: FrameTiming): string {
  const lags = clip.frame_lags;
  if (lags.length === 0) return "One frame at a time";
  if (timing.interval === null) {
    // Without timestamps the spacing is a number of frames, not a time.
    return lags.length === 1
      ? `This frame and the one ${countWord(lags[0] as number)} before it`
      : `This frame and the ${countWord(lags.length)} before it`;
  }
  const count = capitalised(countWord(lags.length + 1));
  if (lags.length === 1) return `${count} frames, ${lagText(lags[0] as number, timing)} apart`;
  return `${count} frames, up to ${lagText(Math.max(...lags), timing)} apart`;
}

/** The labels of the frames shown in the first step, the frame on screen first. */
export function framesAges(clip: Clip, timing: FrameTiming): string[] {
  return ["now", ...clip.frame_lags.map((lag) => `${lagText(lag, timing)} before`)];
}

export function framesText(clip: Clip, timing: FrameTiming): string {
  const size = `${clip.sprite.frame_w} × ${clip.sprite.frame_h} pixels`;
  const lags = clip.frame_lags;
  if (lags.length === 0) {
    return `The frame on screen, cropped to the road and shrunk to ${size}.`;
  }
  const history =
    lags.length === 1
      ? `the one ${lagText(lags[0] as number, timing)} before it`
      : `the ones ${lags.map((lag) => lagText(lag, timing)).join(" and ")} before it`;
  const spacing =
    timing.interval !== null
      ? ""
      : timing.reference === null
        ? " This drive’s frames carry no timestamps, so their spacing is counted in frames."
        : ` This drive’s frames carry no timestamps; on ${driveName(timing.reference.drive)}, whose frames do, ` +
          `${lagText(Math.max(...lags), { interval: null, reference: null })} are ` +
          `${lagText(Math.max(...lags), { interval: timing.reference.interval, reference: null })} apart.`;
  return (
    `The frame on screen and ${history}, cropped to the road and shrunk to ${size}.${spacing} ` +
    `${capitalised(countWord(lags.length + 1))} frames show which way the view is moving, which one frame cannot.`
  );
}

export function speedText(msPerFrame: number | null): string {
  return msPerFrame === null ? "" : `Here it takes about ${fmtMs(msPerFrame)} a frame.`;
}
