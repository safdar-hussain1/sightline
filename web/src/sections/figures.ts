/**
 * The measured figures the page quotes, read out of results.json.
 *
 * Every figure is a value the reports wrote, copied, never worked out again
 * here: the shipped model's error on the frames of its own drive it never
 * trained on, the paired leakage measurement for both networks, the shipped
 * network's skill on another day, the lookup table's score under both splits,
 * and how close test frames sit to training frames under each -- and, for the
 * Results view's sections, each seed's pair of the paired measurement, the
 * generalisation ladder, the steering regimes, the standard-recipe PilotNet,
 * the export check and the split rules. Reading checks that the pairs the page
 * compares really were measured on the same frames, and that figures the data
 * holds twice agree, and names the key that is missing or wrong, so data of
 * another shape fails with a sentence rather than with a NaN on screen. Each
 * section has its own reader, so a malformed key costs only the section that
 * reads it.
 */

import { timedLags } from "../data/timing.ts";
import { Reader, type Json, type Network, type Spread, type Seeded } from "./reader.ts";
import { framesInKey } from "./names.ts";

export { NETWORKS, Reader, readFigures } from "./reader.ts";
export type { Figures, Json, Network, Seeded, Spread } from "./reader.ts";

// -- The Results view's sections -------------------------------------------------------

/** One seed of the paired measurement: the network trained on that seed's
 * shuffled split and on its blocked split, both scored on the frames both
 * splits held out (`frames` of them), and always steering straight on the
 * same frames, in degrees. */
export interface PairedSeed {
  readonly seed: number;
  readonly frames: number;
  readonly shuffled: number;
  readonly blocked: number;
  /** The blocked error over the shuffled one. */
  readonly ratio: number;
  readonly straight: number;
}

export interface PairedNetwork {
  readonly ratio: Spread;
  /** The mean errors over the seeds. */
  readonly shuffled: number;
  readonly blocked: number;
  readonly seeds: readonly PairedSeed[];
}

/** The paired measurement, seed by seed, for both networks. Each statistic
 * over seeds must be made of the per-seed figures beside it, and a seed's
 * paired frames are the same for both networks: as many, and steering
 * straight scoring the same on them. */
export function readPaired(raw: Json, url = "data/results.json"): Readonly<Record<Network, PairedNetwork>> {
  // Annotated, so a failed check narrows what follows it.
  const r: Reader = new Reader(raw, url);
  const read = (network: Network): PairedNetwork => {
    const base = `paired_leakage.${network}`;
    const list = r.list(`${base}.per_seed`);
    const seeds = list.map((_, i): PairedSeed => {
      const at = `${base}.per_seed.${i}`;
      const shuffled = r.number(`${at}.shuffled_mae_deg`, (v) => v > 0, "an error above 0 degrees");
      const blocked = r.error(`${at}.blocked_mae_deg`);
      const ratio = r.number(`${at}.ratio`, (v) => v >= 0, "a ratio of at least 0");
      if (Math.abs(ratio - blocked / shuffled) > 1e-9 * Math.max(1, ratio)) {
        r.fail(`${at}.ratio`, "the blocked error over the shuffled one");
      }
      return { seed: r.index(`${at}.seed`), frames: r.count(`${at}.n`), shuffled, blocked, ratio, straight: r.error(`${at}.zero_mae_deg`) };
    });
    const ratio = r.seeded(`${base}.ratio`, seeds.length, (at) => r.number(at, (v) => v >= 0, "a ratio of at least 0"));
    const shuffled = r.seeded(`${base}.shuffled_mae_deg`, seeds.length);
    const blocked = r.seeded(`${base}.blocked_mae_deg`, seeds.length);
    seeds.forEach((seed, i) => {
      r.same(`${base}.ratio.values.${i}`, seed.ratio, `seed ${seed.seed}'s ratio`);
      r.same(`${base}.shuffled_mae_deg.values.${i}`, seed.shuffled, `seed ${seed.seed}'s shuffled error`);
      r.same(`${base}.blocked_mae_deg.values.${i}`, seed.blocked, `seed ${seed.seed}'s blocked error`);
    });
    return { ratio: { mean: ratio.mean, sd: ratio.sd, runs: seeds.length }, shuffled: shuffled.mean, blocked: blocked.mean, seeds };
  };
  const paired = { sightlinenet: read("sightlinenet"), pilotnet: read("pilotnet") };
  const [first, second] = [paired.sightlinenet.seeds, paired.pilotnet.seeds];
  if (first.length !== second.length) r.fail("paired_leakage.pilotnet.per_seed", `as many seeds as SightlineNet's (${first.length})`);
  first.forEach((seed, i) => {
    const at = `paired_leakage.pilotnet.per_seed.${i}`;
    r.same(`${at}.seed`, seed.seed, "the same seed as SightlineNet's");
    r.same(`${at}.n`, seed.frames, "the frames SightlineNet's pair was scored on");
    r.same(`${at}.zero_mae_deg`, seed.straight, "steering straight on the same frames");
  });
  return paired;
}

/** The networks the ladder compares, by their keys in the data, in the order
 * their bars stand in each rung. */
export const LADDER_MODELS = ["sightlinenet", "pilotnet", "sightlinenet-1f"] as const;
export type LadderModel = (typeof LADDER_MODELS)[number];

/** The ladder's rungs, nearest the training frames first: each a protocol's
 * test frames, or another day's whole drive scored by the blocked models. */
export const RUNG_KINDS = ["shuffled", "blocked", "later", "another-day"] as const;
export type RungKind = (typeof RUNG_KINDS)[number];

const RUNG_OF: Readonly<Record<string, RungKind>> = {
  "leaky/test": "shuffled",
  "blocked/test": "blocked",
  "forward/test": "later",
  "blocked/cross_drive": "another-day",
};

export interface Rung {
  readonly kind: RungKind;
  /** The rung's name, as the data gives it ("later in the drive"). */
  readonly name: string;
  readonly protocol: string;
  readonly split: "test" | "cross_drive";
  /** The drive its frames come from. */
  readonly drive: string;
  /** Always steering straight on the rung's frames, degrees. */
  readonly straight: number;
}

/** One network on one rung, over its trainings: its error and its skill,
 * how much better it is than always steering straight on the same frames
 * (1 - error / steering straight's), negative when it is worse. */
export interface LadderBar {
  readonly error: Seeded;
  readonly skill: Omit<Spread, "runs">;
  readonly seeds: readonly number[];
}

export interface Ladder {
  readonly rungs: readonly Rung[];
  /** Each network's bar on each rung, in rung order: null where the network
   * was never trained with that rung's split. */
  readonly bars: Readonly<Record<LadderModel, ReadonlyArray<LadderBar | null>>>;
}

/** The generalisation ladder, with each bar's skill from its group: the
 * group's own mean of each training's skill, never worked out here from the
 * mean error. A rung a network lacks must be one whose split it was never
 * trained with (it has no group for it); a network trained that way but not
 * on the ladder is data the page cannot draw honestly, and is refused. */
export function readLadder(raw: Json, url = "data/results.json"): Ladder {
  // Annotated, so a failed check narrows what follows it.
  const r: Reader = new Reader(raw, url);
  const groups = r.object("groups");
  const rungs = new Map<RungKind, Rung>();
  const entries = new Map<LadderModel, Map<RungKind, LadderBar>>();
  for (const model of LADDER_MODELS) {
    const list = r.list(`ladder.${model}`);
    const own = new Map<RungKind, LadderBar>();
    let last = -1;
    list.forEach((_, i) => {
      const at = `ladder.${model}.${i}`;
      const protocol = r.string(`${at}.protocol`);
      const split = r.string(`${at}.split`);
      const kind = RUNG_OF[`${protocol}/${split}`];
      if (kind === undefined) r.fail(at, "a rung the page knows: a protocol's test frames or another day");
      const order = RUNG_KINDS.indexOf(kind);
      if (order <= last) r.fail(at, "the next rung away from the training frames");
      last = order;
      const rung: Rung = {
        kind,
        name: r.string(`${at}.rung`),
        protocol,
        split: split as Rung["split"],
        drive: r.string(`${at}.drive`),
        straight: r.error(`${at}.zero_mae_deg`),
      };
      const known = rungs.get(kind);
      if (known === undefined) rungs.set(kind, rung);
      else {
        r.same(`${at}.rung`, known.name, "the rung's name for every network");
        r.same(`${at}.drive`, known.drive, "the rung's drive for every network");
        r.same(`${at}.zero_mae_deg`, known.straight, "steering straight on the same frames for every network");
      }
      const seeds = r.list(`${at}.seeds`).map((_, k) => r.index(`${at}.seeds.${k}`));
      const error = r.seeded(`${at}.mae_deg`, seeds.length);
      const group = `groups.${model}/${protocol}`;
      const prefix = split === "test" ? "test" : "cross";
      r.same(`${group}.${prefix}_mae_deg.mean`, error.mean, "the ladder's mean error");
      r.same(`${group}.${prefix}_mae_deg.sd`, error.sd, "the ladder's spread of the error");
      r.same(`${group}.${prefix}_zero_mae_deg`, rung.straight, "the ladder's steering straight");
      // On the shuffled split each training is tested on its own draw of frames,
      // so steering straight there is a mean over different frames.
      const skill = r.skill(`${group}.${prefix}_skill`, error, rung.straight, kind === "shuffled", [`${prefix}_mae_deg`, `${prefix}_zero_mae_deg`]);
      own.set(kind, { error, skill, seeds });
    });
    entries.set(model, own);
  }
  const ordered = RUNG_KINDS.filter((kind) => rungs.has(kind)).map((kind) => rungs.get(kind) as Rung);
  const bars = Object.fromEntries(
    LADDER_MODELS.map((model) => {
      const own = entries.get(model) as Map<RungKind, LadderBar>;
      return [
        model,
        ordered.map((rung) => {
          const bar = own.get(rung.kind);
          if (bar) return bar;
          if (`${model}/${rung.protocol}` in groups) {
            r.fail(`ladder.${model}`, `a ${rung.name} rung: groups holds ${model}/${rung.protocol}`);
          }
          return null;
        }),
      ];
    }),
  ) as unknown as Ladder["bars"];
  return { rungs: ordered, bars };
}

/** How many frames each SightlineNet is shown. */
export type FrameCounts = Readonly<Record<"sightlinenet" | "sightlinenet-1f", number>>;

/** The frames each SightlineNet sees: the shipped network the frame on screen
 * and one more for each frame lag its drives' timing lists (the shipped run's
 * own lags), the other as many as its key names. Null when the shipped run is
 * not a SightlineNet or no drive's timing lists its lags. */
export function readFrames(raw: Json): FrameCounts | null {
  const shipped = raw["shipped"];
  const model = typeof shipped === "object" && shipped !== null ? (shipped as Json)["model"] : undefined;
  const lags = timedLags(raw);
  const fewer = framesInKey("sightlinenet-1f");
  if (model !== "sightlinenet" || lags === null || fewer === null) return null;
  return { sightlinenet: 1 + lags.length, "sightlinenet-1f": fewer };
}

/** The steering regimes: frames binned by how hard the driver was turning. */
export const REGIMES = ["straight", "gentle", "sharp"] as const;
export type Regime = (typeof REGIMES)[number];

export interface Regimes {
  /** The edges between straight and gentle, and gentle and sharp, degrees of
   * the driver's wheel either way. */
  readonly edges: readonly [number, number];
  /** The test frames in each regime, and in all. */
  readonly frames: Readonly<Record<Regime, number>>;
  readonly total: number;
  /** Each network's chosen run (the shipped one for SightlineNet) on them. */
  readonly models: Readonly<Record<LadderModel, { readonly run: string; readonly seed: number; readonly errors: Readonly<Record<Regime, number>> }>>;
  /** Always steering straight on the same frames. */
  readonly straight: Readonly<Record<Regime, number>>;
}

/** The regimes of the blocked split's test frames, the shipped run's for
 * SightlineNet. Every network and steering straight must be scored on the
 * same frames of each regime, and the regimes together on all the test frames
 * the shipped run was scored on. */
export function readRegimes(raw: Json, url = "data/results.json"): Regimes {
  // Annotated, so a failed check narrows what follows it.
  const r: Reader = new Reader(raw, url);
  const edges = r.list("cutoffs.regime_bins_deg");
  if (edges.length !== 2) r.fail("cutoffs.regime_bins_deg", "two edges");
  const low = r.number("cutoffs.regime_bins_deg.0", (v) => v > 0, "an angle above 0 degrees");
  const high = r.number("cutoffs.regime_bins_deg.1", (v) => v > low, `an angle above ${low} degrees`);
  r.same("regimes.models.sightlinenet.run", r.at("shipped.run"), "the shipped run");
  const frames = Object.fromEntries(REGIMES.map((regime) => [regime, r.count(`regimes.zero.${regime}.n`)])) as Record<Regime, number>;
  const total = r.count("baselines.test.n");
  if (REGIMES.reduce((sum, regime) => sum + frames[regime], 0) !== total) {
    r.fail("regimes.zero", `regimes that hold the ${total} test frames between them`);
  }
  const errors = (base: string) => Object.fromEntries(REGIMES.map((regime) => [regime, r.error(`${base}.${regime}.mae_deg`)])) as Record<Regime, number>;
  const models = Object.fromEntries(
    LADDER_MODELS.map((model) => {
      const base = `regimes.models.${model}`;
      for (const regime of REGIMES) r.same(`${base}.regimes.${regime}.n`, frames[regime], `the ${regime} frames steering straight was scored on`);
      return [model, { run: r.string(`${base}.run`), seed: r.index(`${base}.seed`), errors: errors(`${base}.regimes`) }];
    }),
  ) as unknown as Regimes["models"];
  return { edges: [low, high], frames, total, models, straight: errors("regimes.zero") };
}

/** One set of frames the standard-recipe PilotNet was scored on. */
export interface LegacyScore {
  readonly drive: string;
  readonly frames: number;
  readonly model: number;
  readonly straight: number;
}

export interface Legacy {
  readonly recipe: string;
  /** Its own training drive, from frame `firstFrame` on. */
  readonly own: LegacyScore;
  readonly firstFrame: number;
  /** The other drive's blocked and later-in-the-drive test frames. */
  readonly blocked: LegacyScore;
  readonly later: LegacyScore;
  /** The largest angle it predicted, either way, on every frame it was
   * scored on, and the angle its head can never pass. */
  readonly largest: number;
  readonly cap: number;
  /** The other drive's labels steered more than `cap` either way. */
  readonly beyond: number;
}

/** The standard-recipe PilotNet. The angle its head can never pass is the one
 * the report names its count of labels beyond by. */
export function readLegacy(raw: Json, url = "data/results.json"): Legacy {
  // Annotated, so a failed check narrows what follows it.
  const r: Reader = new Reader(raw, url);
  const own = r.string("legacy.trained_on");
  const scores = r.object("legacy.eval");
  const find = (suffix: string, drive?: string): LegacyScore => {
    const keys = Object.keys(scores).filter((key) => key.endsWith(suffix) && (drive === undefined || key === `${drive}${suffix}`));
    if (keys.length !== 1) r.fail("legacy.eval", `one score named *${suffix}${drive ? ` for ${drive}` : ""}`);
    const key = keys[0] as string;
    const at = `legacy.eval.${key}`;
    return { drive: key.slice(0, -suffix.length), frames: r.count(`${at}.model.n`), model: r.error(`${at}.model.mae_deg`), straight: r.error(`${at}.zero_mae_deg`) };
  };
  const blocked = find("_blocked_test");
  const later = find("_forward_test", blocked.drive);
  if (blocked.drive === own) r.fail("legacy.eval", `scores on a drive other than ${own}, its training drive`);
  // Scored on the very test frames the page's own models are scored on: the
  // same drive, split with the same block and firewall, as many frames.
  r.same("split_sizes.blocked.drive", blocked.drive, "the drive the standard recipe was scored on");
  r.same("legacy.splits.block", r.at("split_rules.blocked.block"), "the blocked split's block");
  r.same("legacy.splits.firewall", r.at("split_rules.blocked.firewall"), "the blocked split's firewall");
  r.same(`legacy.eval.${blocked.drive}_blocked_test.model.n`, r.at("split_sizes.blocked.test"), "the blocked split’s test frames");
  r.same(`legacy.eval.${later.drive}_forward_test.model.n`, r.at("split_sizes.forward.test"), "the later-in-the-drive split’s test frames");
  const caps = Object.keys(r.object("legacy.atan_cap")).flatMap((key) => {
    const found = /^labels_beyond_(\d+(?:\.\d+)?)$/.exec(key);
    return found ? [Number(found[1])] : [];
  });
  if (caps.length !== 1) r.fail("legacy.atan_cap", "one count of labels beyond the head's limit");
  const cap = caps[0] as number;
  return {
    recipe: r.string("legacy.recipe"),
    own: find("_all_seen", own),
    firstFrame: r.index("legacy.seen_first_frame"),
    blocked,
    later,
    largest: r.error("legacy.atan_cap.max_abs_pred_deg"),
    cap,
    beyond: r.index(`legacy.atan_cap.labels_beyond_${cap}.${blocked.drive}.n`),
  };
}

/** The rules the shipped model is measured against, by their keys in the
 * data: always steering straight, always steering the training frames' mean
 * angle, and two that read the answer stream -- the driver's true angles on
 * the frames just before -- which the model never sees: holding the last
 * one, and extending the line through the last two. */
export const YARDSTICKS = ["zero", "mean", "zoh", "extrap"] as const;
export type Yardstick = (typeof YARDSTICKS)[number];
/** The rules that read the answer stream. */
export const ORACLES: readonly Yardstick[] = ["zoh", "extrap"];

/** The rules and the shipped model on one set of frames. */
export interface YardstickSet {
  /** The blocked split's test frames, or another day's drive. */
  readonly kind: "held-back" | "another-day";
  readonly drive: string;
  readonly frames: number;
  /** The shipped model's own error on them, degrees. */
  readonly model: number;
  readonly errors: Readonly<Record<Yardstick, number>>;
}

/** The shipped run's report: every rule and the model on the blocked split's
 * test frames and on another day's drive, each scored on the very frames the
 * model was, with steering straight the same figure the ladder holds. */
export function readYardsticks(raw: Json, url = "data/results.json"): readonly YardstickSet[] {
  // Annotated, so a failed check narrows what follows it.
  const r: Reader = new Reader(raw, url);
  r.same("baselines.run", r.at("shipped.run"), "the shipped run");
  const read = (split: "test" | "cross_drive", kind: YardstickSet["kind"], drive: string): YardstickSet => {
    const at = `baselines.${split}`;
    const frames = r.count(`${at}.n`);
    r.same(`${at}.model.n`, frames, "the frames the rules were scored on");
    const errors = Object.fromEntries(
      YARDSTICKS.map((name) => {
        r.same(`${at}.baselines.${name}.n`, frames, "the frames the model was scored on");
        return [name, r.error(`${at}.baselines.${name}.mae_deg`)];
      }),
    ) as Record<Yardstick, number>;
    const prefix = split === "test" ? "test" : "cross";
    r.same(`groups.sightlinenet/blocked.${prefix}_zero_mae_deg`, errors.zero, "steering straight on the same frames");
    return { kind, drive, frames, model: r.error(`${at}.model.mae_deg`), errors };
  };
  r.same("baselines.test.n", r.at("split_sizes.blocked.test"), "the blocked split’s test frames");
  return [read("test", "held-back", r.string("split_sizes.blocked.drive")), read("cross_drive", "another-day", r.string("baselines.cross_drive.drive"))];
}

/** How closely onnxruntime reproduced PyTorch when a page model was exported. */
export interface ExportCheck {
  readonly run: string;
  readonly stacks: number;
  readonly angle: number;
  readonly angleGate: number;
  readonly saliency: number;
  readonly saliencyGate: number;
  readonly onnxruntime: string;
}

/** The export check of the page's model and of its twin, each the run the
 * page says it is. */
export function readParity(raw: Json, url = "data/results.json"): { readonly sightline: ExportCheck; readonly shuffled: ExportCheck } {
  // Annotated, so a failed check narrows what follows it.
  const r: Reader = new Reader(raw, url);
  const read = (key: "sightline" | "shuffled", run: unknown): ExportCheck => {
    const at = `parity.${key}`;
    r.same(`${at}.run`, run, key === "sightline" ? "the shipped run" : "the twin's run");
    return {
      run: r.string(`${at}.run`),
      stacks: r.count(`${at}.stacks`),
      angle: r.error(`${at}.max_abs_angle_deg`),
      angleGate: r.number(`${at}.angle_tolerance_deg`, (v) => v > 0, "a tolerance above 0"),
      saliency: r.number(`${at}.max_abs_saliency`, (v) => v >= 0, "a difference of at least 0"),
      saliencyGate: r.number(`${at}.saliency_tolerance`, (v) => v > 0, "a tolerance above 0"),
      onnxruntime: r.string(`${at}.onnxruntime`),
    };
  };
  return { sightline: read("sightline", r.at("shipped.run")), shuffled: read("shuffled", r.at("twin.run")) };
}

/** How one protocol cut the drive: its frames, and the share of the drive it
 * trains, validates and tests on where it deals by shares. */
export interface SplitSize {
  readonly label: string;
  readonly train: number;
  readonly val: number;
  readonly test: number;
  readonly dropped: number;
}

export interface Splits {
  readonly drive: string;
  readonly frames: number;
  readonly shuffled: SplitSize & { readonly shares: readonly [number, number, number]; readonly withinOne: number };
  readonly blocked: SplitSize & { readonly block: number; readonly firewall: number; readonly pattern: string; readonly nearest: number; readonly boundaries: number };
  readonly later: SplitSize & { readonly shares: readonly [number, number, number]; readonly firewall: number; readonly nearest: number };
  /** Another day: the other drive, scored from its frame `firstFrame` on. */
  readonly anotherDay: { readonly drive: string; readonly frames: number };
}

/** The split rules and sizes, with the arithmetic the page states checked
 * against the data: each protocol's frames add up to the drive's, the blocked
 * split drops 2 x firewall frames at each boundary between a training block
 * and a held-out one, and so keeps every test frame at least 2 x firewall + 1
 * frames from training, as the leakage report measured. */
export function readSplits(raw: Json, url = "data/results.json"): Splits {
  // Annotated, so a failed check narrows what follows it.
  const r: Reader = new Reader(raw, url);
  const drive = r.string("split_sizes.blocked.drive");
  const frames = r.count("split_sizes.blocked.frames");
  const size = (protocol: string): SplitSize => {
    const at = `split_sizes.${protocol}`;
    r.same(`${at}.drive`, drive, "the drive every split cuts");
    r.same(`${at}.frames`, frames, "the drive's frames");
    r.same(`${at}.label`, r.at(`protocol_names.${protocol}`), "the protocol's name");
    const counts = { label: r.string(`${at}.label`), train: r.count(`${at}.train`), val: r.count(`${at}.val`), test: r.count(`${at}.test`), dropped: r.index(`${at}.dropped`) };
    if (counts.train + counts.val + counts.test + counts.dropped !== frames) r.fail(at, `train, validation, test and dropped frames that add up to ${frames}`);
    return counts;
  };
  const shares = (protocol: string): [number, number, number] => [
    r.share(`split_rules.${protocol}.train_frac`),
    r.share(`split_rules.${protocol}.val_frac`),
    r.share(`split_rules.${protocol}.test_frac`),
  ];
  const firewall = r.count("split_rules.blocked.firewall");
  r.same("split_rules.forward.firewall", firewall, "the blocked split's firewall");
  const blocked = size("blocked");
  const nearest = r.count("leakage.gaps.blocked.test.min_frames");
  if (nearest !== 2 * firewall + 1) r.fail("leakage.gaps.blocked.test.min_frames", `${2 * firewall + 1}, two firewalls and a frame`);
  // The boundaries between a training block and a held-out one, counted from
  // the pattern laid along the drive; the firewalls drop both sides of each.
  const block = r.count("split_rules.blocked.block");
  const pattern = r.string("split_rules.blocked.pattern");
  if (!/^[TVE]+$/.test(pattern)) r.fail("split_rules.blocked.pattern", "a pattern of T, V and E blocks");
  const blocks = Math.ceil(frames / block);
  let boundaries = 0;
  for (let i = 1; i < blocks; i += 1) {
    if ((pattern[(i - 1) % pattern.length] === "T") !== (pattern[i % pattern.length] === "T")) boundaries += 1;
  }
  if (blocked.dropped !== boundaries * 2 * firewall) {
    r.fail("split_sizes.blocked.dropped", `${boundaries * 2 * firewall}: ${firewall} frames each side of the pattern’s ${boundaries} boundaries`);
  }
  const later = size("forward");
  if (later.dropped !== 4 * firewall) r.fail("split_sizes.forward.dropped", `${4 * firewall}: a firewall each side of its two boundaries`);
  // Each split's sizes follow its shares of the drive, to the frame.
  const shuffled = size("leaky");
  const shuffledShares = shares("leaky");
  const near = (path: string, value: number, expected: number) => {
    if (Math.abs(value - expected) > 1) r.fail(path, `about ${Math.round(expected)}, its share of the drive`);
  };
  near("split_sizes.leaky.train", shuffled.train, frames * shuffledShares[0]);
  near("split_sizes.leaky.val", shuffled.val, frames * shuffledShares[1]);
  const laterShares = shares("forward");
  near("split_sizes.forward.train", later.train + firewall, frames * laterShares[0]);
  near("split_sizes.forward.val", later.train + later.val + 3 * firewall, frames * (laterShares[0] + laterShares[1]));
  return {
    drive,
    frames,
    shuffled: { ...shuffled, shares: shuffledShares, withinOne: r.share("leakage.gaps.leaky.test.frac_within_1") },
    blocked: { ...blocked, block, firewall, pattern, nearest, boundaries },
    later: { ...later, shares: laterShares, firewall, nearest: r.count("leakage.gaps.forward.test.min_frames") },
    anotherDay: { drive: r.string("baselines.cross_drive.drive"), frames: r.count("baselines.cross_drive.n") },
  };
}
