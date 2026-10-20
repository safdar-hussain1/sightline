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
import { countWord, fmtCount, fmtDeg, fmtMs, fmtPct, fmtRatio, fmtSpread } from "../format.ts";
import type {
  ExportCheck,
  Figures,
  Ladder,
  LadderBar,
  LadderModel,
  Legacy,
  PairedNetwork,
  Regime,
  Regimes,
  Rung,
  Splits,
} from "./figures.ts";

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
    `Spread ${fmtSpread(sightlinenet.ratio.sd)} for SightlineNet, ${fmtSpread(pilotnet.ratio.sd)} for PilotNet.`
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

/** The text alternative of the paired figure's ruler: SightlineNet's
 * mean error on each seed's paired frames, trained each way. */
export function pairedRuler(sightline: Figures["paired"]["sightlinenet"]): string {
  return (
    `SightlineNet’s mean error over ${countWord(sightline.ratio.runs)} trainings on the frames both splits held out: ` +
    `${fmtDeg(sightline.shuffled)} when trained on the shuffled split, ${fmtDeg(sightline.blocked)} when trained on the blocked split.`
  );
}

/** The text alternative of the another-day ruler. */
export function anotherDayRuler(anotherDay: Figures["anotherDay"]): string {
  return (
    `Mean error over ${countWord(anotherDay.skill.runs)} trainings on another day’s drive: ` +
    `SightlineNet ${fmtDeg(anotherDay.model)}, always steering straight ${fmtDeg(anotherDay.straight)}.`
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

/** The smallest figure of one significant digit at or above `diff`, in
 * `unit` (degrees unless said): a bound is rounded up, so it is never below
 * what it bounds. Anything under a millionth is stated as that. */
export function upperBound(diff: number, unit = "°"): string {
  const floor = 1e-6;
  if (!(diff > floor)) return `0.000001${unit}`;
  const decimals = Math.max(1, Math.ceil(-Math.log10(diff)));
  const scale = 10 ** decimals;
  let units = Math.ceil(diff * scale);
  // Float error can carry the product a hair past a whole number.
  if ((units - 1) / scale >= diff) units -= 1;
  return `${String(Number((units / scale).toFixed(decimals)))}${unit}`;
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
  if (agreement === null) {
    return { look: "checking", text: `Checking this browser against ${PYTHON}…` };
  }
  const { check, compared, total, maxDiffDeg, clip } = agreement;
  const model = MODEL_NAMES[agreement.model];
  if (compared === 0) {
    return { look: "checking", text: `Checking ${model} in this browser against ${PYTHON}…` };
  }
  const name = clipName(clip);
  const angles = `${capitalised(model)}’s angles`;
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
    // While frames are still running, the largest difference is only the
    // largest so far, so the sentence says which frames it covers.
    const where = compared < total ? `on the ${fmtCount(compared)} frames of ${name} run so far, they` : `on ${name} they`;
    return {
      look: "mismatch",
      text:
        `${angles} in this browser do not match ${PYTHON}: ${where} differ by up to ` +
        `${upperBound(maxDiffDeg)}, more than the ${upperBound(tolerance)} allowed.`,
    };
  }
  const within = maxDiffDeg === 0 ? null : upperBound(maxDiffDeg);
  if (compared < total) {
    const each = within === null ? `identical to ${PYTHON}` : `within ${within} of ${PYTHON}`;
    return {
      look: "checking",
      text: `Checking ${model} live: ${fmtCount(compared)} of ${fmtCount(total)} frames of ${name} so far, every one ${each}.`,
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
      ? `This frame and the one ${lagText(lags[0] as number, timing)} before it`
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

// -- Names ----------------------------------------------------------------------------

/** Each network as a legend names it, and as a sentence does. */
export const NETWORK_NAMES: Readonly<Record<LadderModel, { readonly name: string; readonly said: string }>> = {
  sightlinenet: { name: "SightlineNet", said: "SightlineNet" },
  pilotnet: { name: "PilotNet", said: "PilotNet" },
  "sightlinenet-1f": { name: "SightlineNet, one frame", said: "SightlineNet with one frame" },
};

/** What the page calls always steering straight. */
export const STRAIGHT = "Always steering straight";

/** "the 2018 drive" for "drive2018". */
export function driveTitle(drive: string): string {
  const year = /(\d{4})$/.exec(drive)?.[1];
  return year ? `the ${year} drive` : `the drive ${drive}`;
}

/** An angle the data states exactly (a cutoff, a limit), with only the
 * decimals it has: 5 is "5°", 2.5 is "2.5°". */
export function exactDeg(x: number): string {
  const decimals = Math.min(2, (String(x).split(".")[1] ?? "").length);
  return fmtDeg(x, decimals);
}

/** A skill (1 - error / steering straight's) in words: "31% better than
 * always steering straight", "24% worse than ...", or level with it. */
export function skillWords(skill: number, than = "always steering straight"): string {
  if (skill === 0) return `level with ${than}`;
  return `${fmtPct(Math.abs(skill))} ${skill > 0 ? "better" : "worse"} than ${than}`;
}

/** A skill as the chart labels it: "78%", "−45%". */
export function skillLabel(skill: number): string {
  return fmtPct(skill);
}

// -- The paired experiment ------------------------------------------------------------

/** What one network scored on the frames both splits held out, trained each way. */
export function pairedLead(name: string, network: PairedNetwork): string {
  return (
    `On the frames both splits held out, ${name} trained on shuffled frames is off by ${fmtDeg(network.shuffled)} on average. ` +
    `Trained with whole stretches held back, the same network is off by ${fmtDeg(network.blocked)}.`
  );
}

/** What the ratio means, in words, for one network. */
export function pairedMeaning(name: string, ratio: number): string {
  if (!(ratio > 1)) return `On these frames, a shuffled test does not make ${name} look any better than it is.`;
  const how = ratio < 1.25 ? "a little" : `about ${timesInWords(ratio)}`;
  return `So a shuffled test makes ${name} look ${how} better than it is on road it has not seen nearby.`;
}

/** Always steering straight on the same frames, for scale: one figure when
 * every seed's frames give the same, else the range. */
export function pairedScale(network: PairedNetwork): string {
  const values = network.seeds.map((seed) => seed.straight);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const by = fmtDeg(low) === fmtDeg(high) ? fmtDeg(low) : `${fmtDeg(low)} to ${fmtDeg(high)}`;
  return `${STRAIGHT} is off by ${by} on the same frames.`;
}

/** The frames each seed's pair was scored on: a different set per seed. */
export function pairedFrames(network: PairedNetwork): string {
  const counts = network.seeds.map((seed) => fmtCount(seed.frames));
  const list = counts.length < 2 ? counts.join("") : `${counts.slice(0, -1).join(", ")} and ${counts.at(-1) as string}`;
  return (
    `Each line is one training seed: the network trained on that seed’s shuffled split and on its blocked split, ` +
    `both scored on the frames the two splits both held out (${list} frames).`
  );
}

/** The chart's ratio, as its annotation says it. */
export function pairedRatioLabel(ratio: number): string {
  return `${fmtRatio(ratio)} on average`;
}

/** The slope chart's text alternative: a title and every number it draws. */
export function pairedAlt(name: string, network: PairedNetwork): { title: string; desc: string } {
  const seeds = network.seeds.map(
    (seed) =>
      `Seed ${seed.seed}, ${fmtCount(seed.frames)} frames: ${fmtDeg(seed.shuffled)} trained on shuffled frames, ` +
      `${fmtDeg(seed.blocked)} trained with stretches held back, ${fmtRatio(seed.ratio)} as much; always steering straight ${fmtDeg(seed.straight)}.`,
  );
  return {
    title: `${name}’s mean error on the frames both splits held out, trained each way`,
    desc:
      `${seeds.join(" ")} ` +
      `Mean of the ${countWord(network.seeds.length)} seeds: ${fmtDeg(network.shuffled)} and ${fmtDeg(network.blocked)}; ` +
      `the held-back error is on average ${fmtRatio(network.ratio.mean)} the shuffled one.`,
  };
}

// -- The ladder ---------------------------------------------------------------------------

/** What moved further from the training frames on each rung. */
export function rungCaption(rung: Rung, splits: Splits): string {
  switch (rung.kind) {
    case "shuffled":
      return `Test frames are dealt out at random, so ${fmtPct(splits.shuffled.withinOne)} of them sit within one frame of a frame the network trained on.`;
    case "blocked":
      return `Whole stretches of the drive are held back, so no test frame is closer than ${fmtCount(splits.blocked.nearest)} frames to one the network trained on.`;
    case "later":
      return (
        `It trains on the first ${fmtPct(splits.later.shares[0])} of the drive and is tested on the last ${fmtPct(splits.later.shares[2])}, ` +
        `never closer than ${fmtCount(splits.later.nearest)} frames to anything it trained on.`
      );
    case "another-day":
      return `A drive recorded on another day: ${fmtCount(splits.anotherDay.frames)} frames of ${driveTitle(rung.drive)}, none of them seen in training.`;
  }
}

/** The rung's second line: steering straight on its frames, and any network
 * that was not trained this way. */
export function rungNote(ladder: Ladder, index: number): string {
  const rung = ladder.rungs[index] as Rung;
  const missing = (Object.keys(ladder.bars) as LadderModel[]).filter((model) => ladder.bars[model][index] === null);
  const parts = [`${STRAIGHT} is off by ${fmtDeg(rung.straight)} on these frames.`];
  for (const model of missing) parts.push(`${NETWORK_NAMES[model].said} was not trained this way.`);
  return parts.join(" ");
}

export type LadderView = "skill" | "error";

/** One bar in words, for its tooltip and the chart's text alternative. */
export function barWords(model: LadderModel, bar: LadderBar | null, view: LadderView): string {
  const name = NETWORK_NAMES[model].name;
  if (bar === null) return `${name}: not trained this way`;
  if (view === "skill") return `${name}: ${skillWords(bar.skill.mean)} (±${fmtPct(bar.skill.sd)})`;
  return `${name}: ${fmtDeg(bar.error.mean)} (each training: ${bar.error.values.map((v) => fmtDeg(v)).join(", ")})`;
}

/** The ladder's text alternative in the view on screen. */
export function ladderAlt(ladder: Ladder, view: LadderView): { title: string; desc: string } {
  const models = Object.keys(ladder.bars) as LadderModel[];
  const rungs = ladder.rungs.map((rung, i) => {
    const bars = models.map((model) => barWords(model, ladder.bars[model][i] ?? null, view)).join("; ");
    const straight = view === "error" ? `; ${STRAIGHT.toLowerCase()}: ${fmtDeg(rung.straight)}` : "";
    return `${capitalised(rung.name)}: ${bars}${straight}.`;
  });
  return view === "skill"
    ? {
        title: "How much better than always steering straight each network is, on each rung",
        desc: `Mean of three trainings, with the spread over them. ${rungs.join(" ")}`,
      }
    : {
        title: "Each network’s mean error in degrees on each rung, beside always steering straight",
        desc: `Mean of three trainings, with each training’s error. ${rungs.join(" ")}`,
      };
}

// -- The cards -------------------------------------------------------------------------------

/** The level band: a skill this close to zero is "about level". */
const LEVEL = 0.05;

/** What the second frame buys: SightlineNet with one frame and with two, on
 * another day and on held-back stretches of its own drive. */
export function twoFramesText(ladder: Ladder): { lead: string; also: string } | null {
  const day = ladder.rungs.findIndex((rung) => rung.kind === "another-day");
  const held = ladder.rungs.findIndex((rung) => rung.kind === "blocked");
  const two = (i: number) => (i < 0 ? null : ladder.bars.sightlinenet[i] ?? null);
  const one = (i: number) => (i < 0 ? null : ladder.bars["sightlinenet-1f"][i] ?? null);
  const twoDay = two(day);
  const oneDay = one(day);
  if (!twoDay || !oneDay) return null;
  const single = Math.abs(oneDay.skill.mean) < LEVEL
    ? `about level with always steering straight (${skillWords(oneDay.skill.mean, "it")})`
    : skillWords(oneDay.skill.mean);
  const lead = `On another day’s drive, SightlineNet with one frame is ${single}. With two frames it is ${skillWords(twoDay.skill.mean)}.`;
  const twoHeld = two(held);
  const oneHeld = one(held);
  const also =
    twoHeld && oneHeld
      ? `On held-back stretches of its own drive: ${skillWords(twoHeld.skill.mean, "steering straight")} with two frames, ${skillWords(oneHeld.skill.mean, "it")} with one.`
      : "";
  return { lead, also };
}

/** The regimes' names, with their edges from the data. */
export function regimeNames(edges: readonly [number, number]): Record<Regime, string> {
  const [low, high] = edges;
  return {
    straight: `Straight, wheel under ${exactDeg(low)}`,
    gentle: `Gentle curve, ${exactDeg(low)} to ${exactDeg(high)}`,
    sharp: `Sharp curve, ${exactDeg(high)} or more`,
  };
}

/** Where the shipped model struggles, and where steering straight beats it. */
export function regimesText(regimes: Regimes): { lead: string; straight: string } {
  const model = regimes.models.sightlinenet.errors;
  const zero = regimes.straight;
  const lead =
    `Sightline’s ${fmtCount(regimes.total)} test frames, sorted by how hard the driver was turning. ` +
    `It is furthest off on sharp curves: ${fmtDeg(model.sharp)}, where always steering straight is off by ${fmtDeg(zero.sharp)}.`;
  const straight =
    zero.straight < model.straight
      ? `On straight road, always steering straight wins: it is off by ${fmtDeg(zero.straight)}, Sightline by ${fmtDeg(model.straight)}.`
      : `On straight road Sightline beats always steering straight too: ${fmtDeg(model.straight)} against ${fmtDeg(zero.straight)}.`;
  return { lead, straight };
}

/** The regimes chart's text alternative. */
export function regimesAlt(regimes: Regimes): string {
  const names = regimeNames(regimes.edges);
  const rows = (["straight", "gentle", "sharp"] as const).map(
    (regime) =>
      `${names[regime]} (${fmtCount(regimes.frames[regime])} frames): Sightline ${fmtDeg(regimes.models.sightlinenet.errors[regime])}, ` +
      `always steering straight ${fmtDeg(regimes.straight[regime])}.`,
  );
  return `Mean error by how hard the driver was turning. ${rows.join(" ")}`;
}

/** Better or worse than steering straight, for a score of the standard recipe. */
function againstStraight(score: { model: number; straight: number }): string {
  if (score.model === score.straight) return `level with always steering straight (${fmtDeg(score.straight)})`;
  return `${score.model < score.straight ? "better" : "worse"} than always steering straight (${fmtDeg(score.straight)})`;
}

/** The standard-recipe PilotNet's card. */
export function legacyText(legacy: Legacy): { lead: string; other: string; cap: string } {
  const other = driveTitle(legacy.blocked.drive);
  const { blocked, later } = legacy;
  const worse = blocked.model > blocked.straight && later.model > later.straight;
  return {
    lead: `On its own training drive, ${driveTitle(legacy.own.drive)}, it is off by ${fmtDeg(legacy.own.model)}: ${againstStraight(legacy.own)}.`,
    other: worse
      ? `On ${other}, which it never saw, it does worse than always steering straight: ${fmtDeg(blocked.model)} against ${fmtDeg(blocked.straight)} ` +
        `on the held-back stretches Sightline is tested on, and ${fmtDeg(later.model)} against ${fmtDeg(later.straight)} on the drive’s last part.`
      : `On ${other}, which it never saw, it is off by ${fmtDeg(blocked.model)} on the held-back stretches Sightline is tested on, ` +
        `${againstStraight(blocked)}, and by ${fmtDeg(later.model)} on the drive’s last part, ${againstStraight(later)}.`,
    cap:
      legacy.largest < legacy.cap
        ? `Its arctangent output can never pass ${exactDeg(legacy.cap)}, half a turn of the wheel: its largest answer is ${fmtDeg(legacy.largest)}, ` +
          `yet the driver turned past ${exactDeg(legacy.cap)} on ${fmtCount(legacy.beyond)} frames of ${other}.`
        : `Its largest answer is ${fmtDeg(legacy.largest)}; the driver turned past ${exactDeg(legacy.cap)} on ${fmtCount(legacy.beyond)} frames of ${other}.`,
  };
}

/** The recipe, as the report records it. */
export function recipeNote(recipe: string): string {
  return `The recipe, as its report records it: ${recipe}.`;
}

/** The standard-recipe chart's text alternative. */
export function legacyAlt(legacy: Legacy): string {
  const rows = [
    [`its own training drive (${fmtCount(legacy.own.frames)} frames)`, legacy.own],
    [`the held-back stretches of ${driveTitle(legacy.blocked.drive)} (${fmtCount(legacy.blocked.frames)} frames)`, legacy.blocked],
    [`the last part of ${driveTitle(legacy.later.drive)} (${fmtCount(legacy.later.frames)} frames)`, legacy.later],
  ] as const;
  return `Mean error of the standard-recipe PilotNet against always steering straight: ${rows
    .map(([name, score]) => `on ${name}, ${fmtDeg(score.model)} against ${fmtDeg(score.straight)}`)
    .join("; ")}.`;
}

/** The export check, in one sentence. */
export function exportText(parity: { sightline: ExportCheck; shuffled: ExportCheck }): string {
  const { sightline, shuffled } = parity;
  const gate = sightline.angleGate === shuffled.angleGate ? `, under a gate of ${upperBound(sightline.angleGate)}` : "";
  const passed = sightline.angle <= sightline.angleGate && shuffled.angle <= shuffled.angleGate;
  const stacks = sightline.stacks === shuffled.stacks ? `${fmtCount(sightline.stacks)} frame stacks` : "the frame stacks the page uses";
  return passed
    ? `When each model was exported, onnxruntime ${sightline.onnxruntime} reproduced PyTorch on ${stacks}: ` +
        `Sightline’s angles to within ${upperBound(sightline.angle)}, its twin’s to within ${upperBound(shuffled.angle)}${gate}.`
    : `When the models were exported, onnxruntime and PyTorch differed by up to ${upperBound(Math.max(sightline.angle, shuffled.angle))}, ` +
        `more than the ${upperBound(Math.min(sightline.angleGate, shuffled.angleGate))} allowed.`;
}

/** The blocked split's pattern in words: TTVTTET as "train, train,
 * validate, train, train, test, train". */
export function patternWords(pattern: string): string {
  const words: Record<string, string> = { T: "train", V: "validate", E: "test" };
  return [...pattern].map((letter) => words[letter] ?? letter).join(", ");
}

/** The protocols, each with its arithmetic. */
export function protocolTexts(splits: Splits): { shuffled: string; blocked: string; later: string; anotherDay: string } {
  const { shuffled, blocked, later } = splits;
  const drive = `the ${fmtCount(splits.frames)} frames of ${driveTitle(splits.drive)}`;
  const counts = (s: { train: number; val: number; test: number }) =>
    `${fmtCount(s.train)} to train, ${fmtCount(s.val)} to validate and ${fmtCount(s.test)} to test`;
  const fw = fmtCount(blocked.firewall);
  return {
    shuffled:
      `${capitalised(drive)} are shuffled and dealt ${fmtPct(shuffled.shares[0])}, ${fmtPct(shuffled.shares[1])} and ${fmtPct(shuffled.shares[2])}: ` +
      `${counts(shuffled)}.`,
    blocked:
      `The drive is cut into blocks of ${fmtCount(blocked.block)} frames, dealt ${patternWords(blocked.pattern)}, over and over. ` +
      `At every boundary between a training block and a held-out one, ${fw} frames are dropped on each side: ` +
      `${fmtCount(blocked.boundaries)} boundaries × ${fmtCount(2 * blocked.firewall)} frames = ${fmtCount(blocked.dropped)} frames, ` +
      `so every test frame is at least ${fw} + ${fw} + 1 = ${fmtCount(blocked.nearest)} frames from training. That leaves ${counts(blocked)}.`,
    later:
      `The first ${fmtPct(later.shares[0])} of the drive trains, the next ${fmtPct(later.shares[1])} validates and the last ${fmtPct(later.shares[2])} tests, ` +
      `with ${fmtCount(later.firewall)} frames dropped on each side of both boundaries (${fmtCount(later.dropped)} in all): ${counts(later)}.`,
    anotherDay: `${capitalised(driveTitle(splits.anotherDay.drive))}, recorded on another day: ${fmtCount(splits.anotherDay.frames)} of its frames, scored by the networks trained with stretches held back.`,
  };
}
