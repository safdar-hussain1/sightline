/**
 * The sentences the Results view says about the measurements and the clip
 * (the demo view's are in words.ts, re-exported here).
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
import { driveName } from "../data/timing.ts";
import type { Clip } from "../data/types.ts";
import { countWord, fmtCount, fmtDeg, fmtFixed, fmtPct, fmtRatio, fmtSpread, ordinalWord, steerParts } from "../format.ts";
import {
  LADDER_MODELS,
  ORACLES,
  REGIMES,
  type ExportCheck,
  type Figures,
  type FrameCounts,
  type Ladder,
  type LadderBar,
  type LadderModel,
  type Legacy,
  type PairedNetwork,
  type Regime,
  type Regimes,
  type Rung,
  type Splits,
  type YardstickSet,
  type ClipWindow,
  type FramePair,
  type FramePairs,
  type Timeline,
} from "./figures.ts";
import type { Role } from "../charts/timeline.ts";
import { framesInKey, networkName } from "./names.ts";

export {
  accuracyCaption,
  accuracyLabel,
  accuracySentence,
  capitalised,
  framesAges,
  framesText,
  framesTitle,
  leakageSentence,
  measurementsFailure,
  speedText,
  timesInWords,
} from "./words.ts";
import { capitalised, timesInWords } from "./words.ts";

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

/** The live check as the page shows it. `verdict` is what a screen reader is
 * told: nothing while the check simply runs, words that hold for as long as
 * the look does while it is under way, and the full finding once it is
 * `done` -- so it changes only with the look, or on completion. `progress`
 * is the line beside it that counts the frames run so far; `text` is both,
 * as read on screen. */
export interface ParityState {
  readonly look: ParityLook;
  readonly verdict: string;
  readonly progress: string;
  /** Nothing more the check reads can change it. */
  readonly done: boolean;
  readonly text: string;
}

function parity(look: ParityLook, verdict: string, progress: string, done: boolean): ParityState {
  return { look, verdict, progress, done, text: [verdict, progress].filter((part) => part !== "").join(" ") };
}

/**
 * The trust chip: the model's angles in this browser on the loaded clip
 * against the clip's published ones, computed in Python by the same runtime on
 * the reference decoder's frames. It passes only when this browser decoded
 * those very frames (`check` "match") and every frame run so far is within
 * the clip's `tolerance`; one frame beyond it fails the check. `state` is the
 * cockpit's, or "absent" when the demo's code never started.
 */
export function parityState(agreement: Agreement | null, state: CockpitState | "absent"): ParityState {
  if (state === "absent") {
    return parity("note", "The live check needs the demo, which could not start in this browser.", "", true);
  }
  if (state === "unavailable") {
    return parity("note", "The live check needs the demo, and the demo is not running.", "", true);
  }
  if (agreement === null) {
    return parity("checking", "", `Checking this browser against ${PYTHON}…`, false);
  }
  const { check, compared, total, maxDiffDeg, clip } = agreement;
  const model = MODEL_NAMES[agreement.model];
  if (compared === 0) {
    return parity("checking", "", `Checking ${model} in this browser against ${PYTHON}…`, false);
  }
  const name = clipName(clip);
  const angles = `${capitalised(model)}’s angles`;
  const done = compared >= total;
  // While frames are still running, the largest difference is only the
  // largest so far, so the line beside the verdict says which frames it covers.
  const soFar = `on the ${fmtCount(compared)} frames run so far`;
  if (check === "unverified") {
    return parity("note", "This browser could not check that it decoded the clip’s frames as Python did, so its angles are not compared.", "", true);
  }
  if (!Number.isFinite(maxDiffDeg)) {
    return parity("mismatch", `${angles} in this browser do not match ${PYTHON} on ${name}.`, "", true);
  }
  const by = upperBound(maxDiffDeg);
  if (check === "differ") {
    const decodes = `This browser decodes ${name} differently from Python, so ${model}’s angles here differ from ${PYTHON}`;
    return done
      ? parity("note", `${decodes} by up to ${by} on all ${fmtCount(total)} frames.`, "", true)
      : parity("note", `${decodes}.`, `They differ by up to ${by} ${soFar}.`, false);
  }
  const tolerance = clip.tolerance?.angle_deg;
  if (tolerance === undefined) {
    return done
      ? parity(
          "note",
          `${angles} in this browser differ from ${PYTHON} by at most ${by} on all ${fmtCount(total)} frames of ${name}; ` +
            "the clip records no tolerance to judge that by.",
          "",
          true,
        )
      : parity(
          "note",
          `The clip records no tolerance to judge ${model}’s angles in this browser by, so they are not called a match.`,
          `They differ from ${PYTHON} by at most ${by} ${soFar}.`,
          false,
        );
  }
  const allowed = `more than the ${upperBound(tolerance)} allowed`;
  if (maxDiffDeg > tolerance) {
    return done
      ? parity("mismatch", `${angles} in this browser do not match ${PYTHON}: on ${name} they differ by up to ${by}, ${allowed}.`, "", true)
      : parity(
          "mismatch",
          `${angles} in this browser do not match ${PYTHON}.`,
          `On the ${fmtCount(compared)} frames of ${name} run so far, they differ by up to ${by}, ${allowed}.`,
          false,
        );
  }
  const within = maxDiffDeg === 0 ? null : upperBound(maxDiffDeg);
  if (!done) {
    const each = within === null ? `identical to ${PYTHON}` : `within ${within} of ${PYTHON}`;
    return parity("checking", "", `Checking ${model} live: ${fmtCount(compared)} of ${fmtCount(total)} frames of ${name} so far, every one ${each}.`, false);
  }
  return parity(
    "checked",
    `Checked live: on all ${fmtCount(total)} frames of ${name}, ${model}’s angles in this browser match ` +
      `${PYTHON} ${within === null ? "exactly" : `to within ${within}`}.`,
    "",
    true,
  );
}

// -- Names ----------------------------------------------------------------------------

/** Each network as a legend names it, and as a sentence does, from its key. */
export const NETWORK_NAMES = Object.fromEntries(LADDER_MODELS.map((model) => [model, networkName(model)])) as unknown as Readonly<
  Record<LadderModel, { readonly name: string; readonly said: string }>
>;

/** What the page calls always steering straight. */
export const STRAIGHT = "Always steering straight";

/** "the 2018 drive" for "drive2018". */
export const driveTitle = driveName;

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
    `both scored on the frames both splits held out (${list} frames).`
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
 * that was not trained this way. On the shuffled rung each training was
 * tested on its own draw of frames, so steering straight there is a mean over
 * those draws, and is said so. */
export function rungNote(ladder: Ladder, index: number): string {
  const rung = ladder.rungs[index] as Rung;
  const missing = (Object.keys(ladder.bars) as LadderModel[]).filter((model) => ladder.bars[model][index] === null);
  const parts = [
    rung.kind === "shuffled"
      ? `Each training is tested on its own draw of shuffled frames; always steering straight is off by ${fmtDeg(rung.straight)} on them, on average.`
      : `${STRAIGHT} is off by ${fmtDeg(rung.straight)} on these frames.`,
  ];
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

/** How many trainings every bar of the ladder is the mean of, in words; null
 * when the bars are not all the mean of as many. */
export function trainingsWord(ladder: Ladder): string | null {
  const counts = new Set(Object.values(ladder.bars).flatMap((bars) => bars.flatMap((bar) => (bar ? [bar.seeds.length] : []))));
  const [only] = counts;
  return counts.size === 1 && only !== undefined ? countWord(only) : null;
}

/** "Mean of three trainings", or, when the bars differ, what each is. */
function meanOf(ladder: Ladder | null): string {
  const trainings = ladder ? trainingsWord(ladder) : null;
  return trainings === null ? "Each bar is a mean over the network’s trainings" : `Mean of ${trainings} trainings`;
}

/** The chart's caption in the view on screen. */
export function ladderCaption(ladder: Ladder | null, view: LadderView): string {
  return view === "skill"
    ? `${meanOf(ladder)}; each whisker spans one standard deviation over them.`
    : `${meanOf(ladder)}; each tick is one training. Each rung’s grey line is always steering straight on its frames.`;
}

/** What a bar's height says, in the view on screen: in the skill view bars
 * stand on steering straight's line, in the error view each rung's line is
 * steering straight's own error. */
export function ladderLead(view: LadderView): string {
  const rungs = "Each rung tests on frames further from anything the network trained on.";
  return view === "skill"
    ? `${rungs} A bar above the line beats always steering straight on the same frames; a bar below it does worse.`
    : `${rungs} A bar that stops short of its rung’s grey line beats always steering straight on the same frames; a bar that passes it does worse.`;
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
        desc: `${meanOf(ladder)}, with the spread over them. ${rungs.join(" ")}`,
      }
    : {
        title: "Each network’s mean error in degrees on each rung, beside always steering straight",
        desc: `${meanOf(ladder)}, with each training’s error. ${rungs.join(" ")}`,
      };
}

// -- The cards -------------------------------------------------------------------------------

/** The level band: a skill this close to zero is "about level". */
const LEVEL = 0.05;

/** What each SightlineNet sees, in words: "two frames" and "one frame".
 * Without the shipped network's count, the first is "an earlier frame too". */
export function frameWords(frames: FrameCounts | null): { readonly more: string; readonly fewer: string } {
  const fewer = frames?.["sightlinenet-1f"] ?? framesInKey("sightlinenet-1f") ?? 1;
  return {
    more: frames ? `${countWord(frames.sightlinenet)} frames` : "an earlier frame too",
    fewer: `${countWord(fewer)} frame${fewer === 1 ? "" : "s"}`,
  };
}

/** The frames card's title, from the counts: "What a second frame buys". */
export function framesTitleOf(frames: FrameCounts | null): string {
  if (!frames) return "What an earlier frame buys";
  const extra = frames.sightlinenet - frames["sightlinenet-1f"];
  if (extra === 1) return `What a ${ordinalWord(frames.sightlinenet)} frame buys`;
  return extra > 1 ? `What ${countWord(extra)} more frames buy` : "What the frames buy";
}

/** What the extra frames buy: SightlineNet with fewer frames and with more,
 * on another day and on held-back stretches of its own drive. A skill near
 * zero either way is "about level"; one clearly below zero is worse. */
export function twoFramesText(ladder: Ladder, frames: FrameCounts | null): { lead: string; also: string } | null {
  const day = ladder.rungs.findIndex((rung) => rung.kind === "another-day");
  const held = ladder.rungs.findIndex((rung) => rung.kind === "blocked");
  const two = (i: number) => (i < 0 ? null : ladder.bars.sightlinenet[i] ?? null);
  const one = (i: number) => (i < 0 ? null : ladder.bars["sightlinenet-1f"][i] ?? null);
  const twoDay = two(day);
  const oneDay = one(day);
  if (!twoDay || !oneDay) return null;
  const words = frameWords(frames);
  const single = Math.abs(oneDay.skill.mean) < LEVEL
    ? `about level with always steering straight (${skillWords(oneDay.skill.mean, "it")})`
    : skillWords(oneDay.skill.mean);
  const lead = `On another day’s drive, ${NETWORK_NAMES["sightlinenet-1f"].said} is ${single}. With ${words.more} it is ${skillWords(twoDay.skill.mean)}.`;
  const twoHeld = two(held);
  const oneHeld = one(held);
  const also =
    twoHeld && oneHeld
      ? `On held-back stretches of its own drive: ${skillWords(twoHeld.skill.mean, "steering straight")} with ${words.more}, ${skillWords(oneHeld.skill.mean, "it")} with ${words.fewer}.`
      : "";
  return { lead, also };
}

/** The frames card's chart in words: its title and every number it draws. */
export function framesAlt(
  rows: ReadonlyArray<{ readonly title: string; readonly more: number; readonly fewer: number }>,
  frames: FrameCounts | null,
): { title: string; desc: string } {
  const { more, fewer } = frameWords(frames);
  return {
    title: `SightlineNet with ${more} and with ${fewer}, against always steering straight`,
    desc:
      `How much better than always steering straight SightlineNet is with ${more} and with ${fewer}: ` +
      `${rows.map((row) => `${row.title}: ${more} ${skillWords(row.more)}, ${fewer} ${skillWords(row.fewer)}`).join("; ")}.`,
  };
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

/** Each regime as a place on the road, for a sentence. */
const REGIME_PLACES: Readonly<Record<Regime, string>> = { straight: "straight road", gentle: "gentle curves", sharp: "sharp curves" };

/** Where the shipped model struggles -- the regime where its error is
 * largest -- and where steering straight beats it. */
export function regimesText(regimes: Regimes): { lead: string; straight: string } {
  const model = regimes.models.sightlinenet.errors;
  const zero = regimes.straight;
  const worst = REGIMES.reduce((a, b) => (model[b] > model[a] ? b : a));
  const lead =
    `Sightline’s ${fmtCount(regimes.total)} test frames, sorted by how hard the driver was turning. ` +
    `It is furthest off on ${REGIME_PLACES[worst]}: ${fmtDeg(model[worst])}, where always steering straight is off by ${fmtDeg(zero[worst])}.`;
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

/** The shares of a turn the page puts in words. */
const TURNS: ReadonlyArray<readonly [share: number, words: string]> = [
  [1 / 4, "a quarter turn"],
  [1 / 2, "half a turn"],
  [3 / 4, "three quarters of a turn"],
  [1, "a full turn"],
];

/** An angle of the wheel as a turn ("half a turn of the wheel" for 180°),
 * when it is a plain share of one; otherwise nothing. */
export function turnWords(angle: number): string {
  const found = TURNS.find(([share]) => Math.abs(share * 360 - angle) < 1e-9);
  return found ? `${found[1]} of the wheel` : "";
}

/** The standard-recipe PilotNet's card. */
export function legacyText(legacy: Legacy): { lead: string; other: string; cap: string } {
  const other = driveTitle(legacy.blocked.drive);
  const { blocked, later } = legacy;
  const worse = blocked.model > blocked.straight && later.model > later.straight;
  const turn = turnWords(legacy.cap);
  return {
    lead: `On its own training drive, ${driveTitle(legacy.own.drive)}, it is off by ${fmtDeg(legacy.own.model)}: ${againstStraight(legacy.own)}.`,
    other: worse
      ? `On ${other}, which it never saw, it does worse than always steering straight: ${fmtDeg(blocked.model)} against ${fmtDeg(blocked.straight)} ` +
        `on the held-back stretches Sightline is tested on, and ${fmtDeg(later.model)} against ${fmtDeg(later.straight)} on the drive’s last part.`
      : `On ${other}, which it never saw, it is off by ${fmtDeg(blocked.model)} on the held-back stretches Sightline is tested on, ` +
        `${againstStraight(blocked)}, and by ${fmtDeg(later.model)} on the drive’s last part, ${againstStraight(later)}.`,
    cap:
      legacy.largest < legacy.cap
        ? `Its arctangent output can never pass ${exactDeg(legacy.cap)} either way${turn ? `, ${turn}` : ""}: its largest answer, either way, is ${fmtDeg(legacy.largest)}, ` +
          `yet the driver turned past ${exactDeg(legacy.cap)} on ${fmtCount(legacy.beyond)} frames of ${other}.`
        : `Its largest answer, either way, is ${fmtDeg(legacy.largest)}; the driver turned past ${exactDeg(legacy.cap)} on ${fmtCount(legacy.beyond)} frames of ${other}.`,
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

// -- The baselines ----------------------------------------------------------------------------

/** Each rule as its mark on a ruler names it. */
export const YARDSTICK_LABELS = {
  zero: "steering straight",
  mean: "mean angle",
  zoh: "hold the last label",
  extrap: "linear extrapolation",
} as const;

/** Each rule as a sentence names it. */
const YARDSTICK_SAID = {
  zero: "always steering straight",
  mean: "always steering the training frames’ mean angle",
  zoh: "holding the last label",
  extrap: "linear extrapolation",
} as const;

/** A set of frames by name: "Held-back stretches of the 2018 drive". */
export function yardstickSetName(set: YardstickSet): string {
  return set.kind === "held-back" ? `Held-back stretches of ${driveTitle(set.drive)}` : `Another day: ${driveTitle(set.drive)}`;
}

/** A set of frames, as its ruler is titled: its name and its count. */
export function yardstickSetTitle(set: YardstickSet): string {
  return `${yardstickSetName(set)}, ${fmtCount(set.frames)} frames`;
}

/** The rulers' text alternative: every number they draw. */
export function yardsticksAlt(sets: readonly YardstickSet[]): string {
  return sets
    .map((set) => {
      const where =
        set.kind === "held-back"
          ? `Mean error on the held-back stretches of ${driveTitle(set.drive)} (${fmtCount(set.frames)} frames)`
          : `Mean error on another day’s drive, ${driveTitle(set.drive)} (${fmtCount(set.frames)} frames)`;
      const e = set.errors;
      return (
        `${where}: ${YARDSTICK_SAID.zero} ${fmtDeg(e.zero)}, ${YARDSTICK_SAID.mean} ${fmtDeg(e.mean)}, ` +
        `and, reading the answer stream the model never sees, ${YARDSTICK_SAID.zoh} ${fmtDeg(e.zoh)} and ${YARDSTICK_SAID.extrap} ${fmtDeg(e.extrap)}; ` +
        `Sightline ${fmtDeg(set.model)}.`
      );
    })
    .join(" ");
}

/** What the rules that read the answers show: a bound on their MEAN error,
 * rounded up, never to nearest. On a few frames they are far off, so it is
 * never said of every frame. */
export function yardsticksVerdict(sets: readonly YardstickSet[]): string {
  const largest = Math.max(...sets.flatMap((set) => ORACLES.map((name) => set.errors[name])));
  return (
    `The rules that read the answers are off by ${upperBound(largest)} or less on average: the true angle barely changes from one frame to the next. ` +
    "They are yardsticks here, not rivals."
  );
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

// -- The frame pairs ------------------------------------------------------------------

/** How far apart two frames were taken, from their timestamps, in the unit
 * that reads best: milliseconds under a second, seconds under a minute, then
 * minutes and whole seconds (rounded down). Empty without timestamps. */
export function gapWords(seconds: number | null): string {
  if (seconds === null) return "";
  if (seconds < 1) return `${fmtCount(Math.round(seconds * 1000))} ms`;
  if (seconds < 60) return `${fmtFixed(seconds)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${fmtCount(minutes)} min ${fmtCount(Math.floor(seconds - minutes * 60))} s`;
}

/** A lower bound, rounded down to one decimal: never above what it bounds. */
function atLeast(value: number): string {
  return fmtFixed(Math.floor(value * 10) / 10);
}

/** What the pairs show, from all of them: the shuffled split's closest
 * training frame is next door, the blocked split's far away. */
export function pairsLead(pairs: FramePairs): string {
  const list = pairs.pairs;
  const near = list.map((pair) => pair.shuffled);
  const far = list.map((pair) => pair.blocked);
  const nearFrames = near.map((side) => side.gapFrames);
  const where = nearFrames.every((gap) => gap === 1) ? "the next one along" : `${fmtCount(Math.min(...nearFrames))} to ${fmtCount(Math.max(...nearFrames))} frames away`;
  const nearTimes = near.flatMap((side) => (side.gapS === null ? [] : [side.gapS]));
  let nearTime = "";
  if (nearTimes.length === near.length) {
    const [low, high] = [Math.min(...nearTimes), Math.max(...nearTimes)];
    nearTime = high < 1 ? `, ${fmtCount(Math.round(low * 1000))} to ${gapWords(high)} away` : `, ${gapWords(low)} to ${gapWords(high)} away`;
  }
  const farTimes = far.flatMap((side) => (side.gapS === null ? [] : [side.gapS]));
  const farTime = farTimes.length === far.length ? `, at least ${atLeast(Math.min(...farTimes) / 60)} minutes of driving` : "";
  return (
    `${capitalised(countWord(list.length))} test frames, each beside the training frame that looks most like it. ` +
    `Under the shuffled split that frame is ${where}${nearTime}. ` +
    `Under the blocked split it is ${fmtCount(Math.min(...far.map((side) => side.gapFrames)))} frames away or more${farTime}.`
  );
}

const steering = (angle: number) => {
  const { size, side } = steerParts(angle);
  return side ? `${size} ${side}` : size;
};

/** One neighbour, in the words of its panel: where it is, how alike it
 * looks, and what the driver steered. */
export function neighbourWords(pair: FramePair, split: "shuffled" | "blocked"): { where: string; likeness: string; steering: string } {
  const side = pair[split];
  const direction = side.frame < pair.frame ? "before" : "after";
  const time = gapWords(side.gapS);
  return {
    where: `${fmtCount(side.gapFrames)} frame${side.gapFrames === 1 ? "" : "s"} ${direction} it${time ? `, ${time}` : ""}`,
    likeness: `Pixel difference ${fmtFixed(side.pixelRmse)}`,
    steering: `Steering ${steering(side.angle)}`,
  };
}

/** A test frame's own label, as its panel says it. */
export function testFrameWords(pair: FramePair): { frame: string; steering: string } {
  return { frame: `Frame ${fmtCount(pair.frame)}`, steering: `Steering ${steering(pair.angle)}` };
}

/** The picker's legend: the order of the pairs, when they come in one. */
export function pickerLegend(pairs: FramePairs): string {
  const sizes = pairs.pairs.map((pair) => Math.abs(pair.angle));
  const rising = sizes.every((size, i) => i === 0 || size >= (sizes[i - 1] as number));
  return rising && sizes.length > 1 ? "Pick a test frame, from the gentlest steering to the hardest" : "Pick a test frame";
}

/** A pair, in one passage: the text alternative of its three pictures. */
export function pairAlt(pair: FramePair): string {
  const [near, far] = [neighbourWords(pair, "shuffled"), neighbourWords(pair, "blocked")];
  const lower = (words: string) => words.charAt(0).toLowerCase() + words.slice(1);
  return (
    `Test frame ${fmtCount(pair.frame)}, the driver steering ${steering(pair.angle)}. ` +
    `Under the shuffled split its closest training frame is ${near.where}, ${lower(near.likeness)}, ${lower(near.steering)}. ` +
    `Under the blocked split it is ${far.where}, ${lower(far.likeness)}, ${lower(far.steering)}.`
  );
}

// -- The split timeline ------------------------------------------------------------

/** Seconds as minutes and seconds on the drive's clock, "25:19" (rounded down). */
export function clockTime(seconds: number): string {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

const minutesOf = (seconds: number) => fmtFixed(seconds / 60);

export type TimelineProtocol = "shuffled" | "blocked" | "later";

export function timelineLead(timeline: Timeline): string {
  return (
    `The ${minutesOf(timeline.clock.durationS)} minutes of ${driveTitle(timeline.drive)}, from its first frame to its last, ` +
    `and how each split deals its ${fmtCount(timeline.frames)} frames to training, validation and testing.`
  );
}

/** What the band shows for one split, with the rules results.json states. */
export function timelineNote(timeline: Timeline, protocol: TimelineProtocol): string {
  const { shuffled, blocked, later } = timeline.protocols;
  if (protocol === "shuffled") {
    const shares = shuffled.shares;
    return (
      `Every frame is dealt at random: ${fmtPct(shares[0])} to train, ${fmtPct(shares[1])} to validate and ${fmtPct(shares[2])} to test. ` +
      "The band shows an even mix at those shares, an illustration: it does not say which frame went where."
    );
  }
  if (protocol === "blocked") {
    return (
      `Blocks of ${fmtCount(blocked.block)} frames, dealt ${patternWords(blocked.pattern)}, over and over, ` +
      `with ${fmtCount(blocked.firewall)} frames dropped on each side of every boundary between a training block and a held-out one.`
    );
  }
  return (
    `The first ${fmtPct(later.shares[0])} of the drive trains, the next ${fmtPct(later.shares[1])} validates and the last ${fmtPct(later.shares[2])} tests, ` +
    `with ${fmtCount(later.firewall)} frames dropped on each side of both boundaries.`
  );
}

/** What each split does with a frame, as the band's key names it. */
export const ROLE_NAMES: Readonly<Record<Role, string>> = { train: "Train", val: "Validate", test: "Test", dropped: "Dropped" };

const ROLE_STRETCH: Readonly<Record<Role, string>> = { train: "training", val: "validation", test: "test", dropped: "dropped" };

/** Where the demo's clip sits: its times on the drive's clock, and the
 * stretch of the picked split it falls in, when it falls in one. */
export function timelineClipNote(timeline: Timeline, clip: ClipWindow, protocol: TimelineProtocol = "blocked"): string {
  const [year] = splitTitle(clip.title);
  const name = year ? `the demo’s ${year} clip` : "the demo clip";
  const when = clip.firstS !== null && clip.lastS !== null ? `, from ${clockTime(clip.firstS)} to ${clockTime(clip.lastS)},` : "";
  if (protocol === "shuffled") return `${capitalised(name)}${when} is dealt frame by frame like the rest of the drive: any of its frames may train.`;
  const way = protocol === "blocked" ? "blocked" : "later-in-the-drive";
  const end = clip.firstFrame + clip.frames;
  const inside = timeline.protocols[protocol].segments.find((s) => s.from <= clip.firstFrame && end <= s.to);
  return inside
    ? `${capitalised(name)}${when} sits inside one of the ${way} split’s ${ROLE_STRETCH[inside.role]} stretches.`
    : `${capitalised(name)}${when} spans more than one of the ${way} split’s stretches.`;
}

/** The band's text alternative for one split. */
export function timelineAlt(timeline: Timeline, protocol: TimelineProtocol): string {
  const head = `${capitalised(driveTitle(timeline.drive))}, ${minutesOf(timeline.clock.durationS)} minutes, `;
  const trace = " Above it, how hard the driver was turning, on a square-root scale.";
  if (protocol === "shuffled") {
    const s = timeline.protocols.shuffled.sizes;
    return `${head}split the shuffled way: ${fmtCount(s.train)} frames to train, ${fmtCount(s.val)} to validate and ${fmtCount(s.test)} to test, every frame dealt at random.${trace}`;
  }
  const layout = protocol === "blocked" ? timeline.protocols.blocked : timeline.protocols.later;
  const s = layout.sizes;
  const stretches = layout.segments.filter((segment) => segment.role !== "dropped").length;
  const name = protocol === "blocked" ? "the blocked way" : "later in the drive";
  return (
    `${head}split ${name}: ${fmtCount(s.train)} frames to train, ${fmtCount(s.val)} to validate, ${fmtCount(s.test)} to test ` +
    `and ${fmtCount(s.dropped)} dropped, in ${fmtCount(stretches)} stretches.${trace}`
  );
}
