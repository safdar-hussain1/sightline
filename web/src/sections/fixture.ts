/**
 * A results.json for tests: the keys the page reads, in the shape the
 * site-data builder writes them, with values like the reports'. Each call
 * returns a fresh copy, so a test may break one key without touching another.
 * Nothing in the page imports this file.
 */

const RUNS = (model: string, protocol: string) => [0, 1, 2].map((seed) => `drive2018_${model}_${protocol}_s${seed}`);

export function resultsFixture(): Record<string, unknown> {
  return {
    assets: {
      models: { sightline: "models/sightline.onnx", shuffled: "models/sightline-shuffled.onnx" },
      clips: { c2017: "data/clips/c2017/clip.json", c2018: "data/clips/c2018/clip.json" },
    },
    shipped: { run: "drive2018_sightlinenet_blocked_s2", model: "sightlinenet", protocol: "blocked", seed: 2 },
    twin: { run: "drive2018_sightlinenet_leaky_s2" },
    baselines: {
      run: "drive2018_sightlinenet_blocked_s2",
      test: {
        n: 6000,
        model: { mae_deg: 10.12272448789344, n: 6000 },
        baselines: { zero: { mae_deg: 17.479270016382138, n: 6000 } },
      },
    },
    paired_leakage: {
      pilotnet: {
        ratio: { mean: 3.166602797825647, sd: 0.13786572388950946, values: [3.3195783025911982, 3.051959717769372, 3.1282703731163717] },
        shuffled_mae_deg: { mean: 4.130170764717941, sd: 0.28707933891936716 },
        blocked_mae_deg: { mean: 13.07809728495099, sd: 1.0415481408090872 },
        per_seed: [893, 889, 943].map((n, seed) => ({ seed, n })),
      },
      sightlinenet: {
        ratio: { mean: 3.0465626341557503, sd: 0.6142642724803958, values: [3.747038174282365, 2.5997789697772737, 2.792870758407613] },
        shuffled_mae_deg: { mean: 3.3481055871804877, sd: 0.40005183380705683 },
        blocked_mae_deg: { mean: 10.086980146092957, sd: 1.3648407093489716 },
        per_seed: [893, 889, 943].map((n, seed) => ({ seed, n })),
      },
    },
    groups: {
      "pilotnet/blocked": { runs: RUNS("pilotnet", "blocked") },
      "pilotnet/forward": { runs: RUNS("pilotnet", "forward") },
      "pilotnet/leaky": { runs: RUNS("pilotnet", "leaky") },
      "sightlinenet-1f/blocked": { runs: RUNS("sightlinenet-1f", "blocked") },
      "sightlinenet/blocked": {
        runs: RUNS("sightlinenet", "blocked"),
        cross_mae_deg: { mean: 10.640418845640982, sd: 1.304190160711699 },
        cross_skill: { mean: 0.3145251314711981, sd: 0.08401826957372512 },
        cross_zero_mae_deg: 15.52269723393061,
      },
      "sightlinenet/forward": { runs: RUNS("sightlinenet", "forward") },
      "sightlinenet/leaky": { runs: RUNS("sightlinenet", "leaky") },
    },
    leakage: {
      gaps: {
        leaky: { test: { frac_within_1: 0.9093377898475037, min_frames: 1 } },
        blocked: { test: { frac_within_1: 0, min_frames: 501 } },
      },
      memoriser: {
        leaky: { mae_deg: 0.6379842974604311, n: 893 },
        blocked: { mae_deg: 21.867592464485867, n: 893 },
      },
    },
  };
}

/** Set the value at a dotted path of `json` (group keys hold a slash, never a
 * dot); `undefined` removes the key. Returns `json`. */
export function patch(json: Record<string, unknown>, path: string, value: unknown): Record<string, unknown> {
  const keys = path.split(".");
  let at: Record<string, unknown> = json;
  for (const key of keys.slice(0, -1)) at = at[key] as Record<string, unknown>;
  const last = keys[keys.length - 1] as string;
  if (value === undefined) delete at[last];
  else at[last] = value;
  return json;
}
