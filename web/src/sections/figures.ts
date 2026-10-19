/**
 * The measured figures the page quotes, read out of results.json.
 *
 * Every figure is a value the reports wrote, copied, never worked out again
 * here: the shipped model's error on the frames of its own drive it never
 * trained on, the paired leakage measurement for both networks, the shipped
 * network's skill on another day, the lookup table's score under both splits,
 * and how close test frames sit to training frames under each. Reading checks
 * that the pairs the page compares really were measured on the same frames,
 * and names the key that is missing or wrong, so data of another shape fails
 * with a sentence rather than with a NaN on screen.
 */

import { AssetError } from "../data/load.ts";

/** The two networks the paired measurement compares, by their keys in the data. */
export const NETWORKS = ["sightlinenet", "pilotnet"] as const;
export type Network = (typeof NETWORKS)[number];

/** A figure measured over several training runs. */
export interface Spread {
  readonly mean: number;
  readonly sd: number;
  /** How many runs it is the mean of. */
  readonly runs: number;
}

export interface Figures {
  /** The shipped model on the frames of its training drive held back from
   * training, and always steering straight on the same frames, degrees. */
  readonly heldOut: { readonly frames: number; readonly model: number; readonly straight: number };
  /** For each seed, the error of the network trained with stretches held back
   * over the error of the one trained on shuffled frames, both scored on the
   * frames that seed's two splits hold out (`frames`, one count per seed). */
  readonly paired: Readonly<
    Record<
      Network,
      { readonly ratio: Spread; readonly shuffled: number; readonly blocked: number; readonly frames: readonly number[] }
    >
  >;
  /** Sightline's network, trained with stretches held back, on another day's
   * drive: its skill (1 - its error / steering straight's) and both errors. */
  readonly anotherDay: { readonly skill: Spread; readonly model: number; readonly straight: number };
  /** Copying the angle of the most similar training frame, under each split. */
  readonly memoriser: { readonly shuffled: number; readonly blocked: number; readonly frames: number };
  /** How close test frames sit to training frames: the share within one frame
   * under the shuffled split, and the nearest any comes under the blocked one. */
  readonly gaps: { readonly shuffledWithinOne: number; readonly blockedMinFrames: number };
  /** The training runs the figures come from. */
  readonly runs: number;
}

type Json = Readonly<Record<string, unknown>>;

/** Walks the file by dotted paths and says which one is wrong. */
class Reader {
  constructor(
    private readonly root: Json,
    private readonly url: string,
  ) {}

  fail(path: string, wanted: string): never {
    throw new AssetError(`${this.url}: ${path} must be ${wanted}`, "invalid", this.url);
  }

  /** The value at a dotted path; a list is stepped into by index. Group keys
   * hold a slash ("sightlinenet/blocked"), never a dot, so they are one step. */
  at(path: string): unknown {
    let value: unknown = this.root;
    for (const key of path.split(".")) {
      if (Array.isArray(value) && /^\d+$/.test(key)) {
        value = value[Number(key)];
      } else if (typeof value === "object" && value !== null && !Array.isArray(value)) {
        value = (value as Json)[key];
      } else {
        this.fail(path, "present");
      }
    }
    return value;
  }

  object(path: string): Json {
    const value = this.at(path);
    if (typeof value !== "object" || value === null || Array.isArray(value)) this.fail(path, "an object");
    return value as Json;
  }

  list(path: string): unknown[] {
    const value = this.at(path);
    if (!Array.isArray(value) || value.length === 0) this.fail(path, "a non-empty list");
    return value;
  }

  number(path: string, check: (v: number) => boolean = () => true, wanted = "a number"): number {
    const value = this.at(path);
    if (typeof value !== "number" || !Number.isFinite(value) || !check(value)) this.fail(path, wanted);
    return value;
  }

  error(path: string): number {
    return this.number(path, (v) => v >= 0, "an error of at least 0 degrees");
  }

  count(path: string): number {
    return this.number(path, (v) => Number.isInteger(v) && v >= 1, "a count of at least 1");
  }

  spread(path: string): Omit<Spread, "runs"> {
    return { mean: this.number(`${path}.mean`), sd: this.number(`${path}.sd`, (v) => v >= 0, "a spread of at least 0") };
  }
}

/** The figures in a parsed results.json (`raw`), checked. Throws an
 * `AssetError` of kind "invalid" naming the first key that is wrong. */
export function readFigures(raw: Json, url = "data/results.json"): Figures {
  const r = new Reader(raw, url);

  // The shipped model's own report holds its held-out error and the baselines
  // scored on the same frames.
  const shipped = r.at("shipped.run");
  if (r.at("baselines.run") !== shipped) r.fail("baselines.run", `the shipped run (${String(shipped)})`);
  const frames = r.count("baselines.test.n");
  for (const path of ["baselines.test.model.n", "baselines.test.baselines.zero.n"]) {
    if (r.count(path) !== frames) r.fail(path, `${frames}, the frames the model was scored on`);
  }
  const heldOut = {
    frames,
    model: r.error("baselines.test.model.mae_deg"),
    straight: r.error("baselines.test.baselines.zero.mae_deg"),
  };

  const paired = Object.fromEntries(
    NETWORKS.map((network) => {
      const base = `paired_leakage.${network}`;
      const values = r.list(`${base}.ratio.values`);
      const seeds = r.list(`${base}.per_seed`);
      if (seeds.length !== values.length) r.fail(`${base}.per_seed`, `one entry per ratio value (${values.length})`);
      return [
        network,
        {
          ratio: { ...r.spread(`${base}.ratio`), runs: values.length },
          shuffled: r.error(`${base}.shuffled_mae_deg.mean`),
          blocked: r.error(`${base}.blocked_mae_deg.mean`),
          // Each seed's shuffled and blocked splits hold out different frames,
          // so each seed has its own paired set.
          frames: seeds.map((_, i) => r.count(`${base}.per_seed.${i}.n`)),
        },
      ];
    }),
  ) as unknown as Figures["paired"];

  const group = "groups.sightlinenet/blocked";
  const anotherDay = {
    skill: { ...r.spread(`${group}.cross_skill`), runs: r.list(`${group}.runs`).length },
    model: r.error(`${group}.cross_mae_deg.mean`),
    straight: r.error(`${group}.cross_zero_mae_deg`),
  };

  const memoriser = {
    shuffled: r.error("leakage.memoriser.leaky.mae_deg"),
    blocked: r.error("leakage.memoriser.blocked.mae_deg"),
    frames: r.count("leakage.memoriser.leaky.n"),
  };
  if (r.count("leakage.memoriser.blocked.n") !== memoriser.frames) {
    r.fail("leakage.memoriser.blocked.n", `${memoriser.frames}, the frames scored under the shuffled split`);
  }

  const gaps = {
    shuffledWithinOne: r.number("leakage.gaps.leaky.test.frac_within_1", (v) => v >= 0 && v <= 1, "a share between 0 and 1"),
    blockedMinFrames: r.count("leakage.gaps.blocked.test.min_frames"),
  };

  const groups = r.object("groups");
  const runs = Object.keys(groups).reduce((sum, key) => sum + r.list(`groups.${key}.runs`).length, 0);

  return { heldOut, paired, anotherDay, memoriser, gaps, runs };
}
