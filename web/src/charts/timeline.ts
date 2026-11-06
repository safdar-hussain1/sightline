/**
 * How each split lays out the drive, as geometry: the stretches of frames it
 * trains, validates and tests on and the ones it drops, and the drive's own
 * clock to place them in time. Pure, like the other charts' layouts.
 *
 * The blocked and the later-in-the-drive layouts are the splits module's,
 * frame for frame, from the rules results.json states; the reader checks they
 * reproduce the sizes the runs recorded. The shuffled split deals each frame
 * at random with a seed, so it has no layout to draw: `ditherRoles` gives an
 * even mix at its shares, an illustration of the idea, never of which frame
 * went where.
 */

/** What a split does with a frame. */
export type Role = "train" | "val" | "test" | "dropped";
export const ROLES: readonly Role[] = ["train", "val", "test", "dropped"];

/** Frames `from` up to `to`, all given the same role. */
export interface Segment {
  readonly role: Role;
  readonly from: number;
  readonly to: number;
}

const CODE_ROLE: Readonly<Record<string, Role>> = { T: "train", V: "val", E: "test" };

/** Touching segments of one role made one. */
function merged(segments: readonly Segment[]): Segment[] {
  const out: Segment[] = [];
  for (const s of segments) {
    if (s.to <= s.from) continue;
    const last = out[out.length - 1];
    if (last && last.role === s.role && last.to === s.from) out[out.length - 1] = { role: s.role, from: last.from, to: s.to };
    else out.push(s);
  }
  return out;
}

/**
 * The blocked split of `n` frames: blocks of `block` frames dealt by repeating
 * `pattern` (T train, V validate, E test), with `firewall` frames dropped on
 * each side of every boundary between a training block and a held-out one.
 * Validation and test blocks meeting each other are not firewalled.
 */
export function blockedSegments(n: number, block: number, firewall: number, pattern: string): Segment[] {
  // The pattern's runs along the drive: consecutive blocks with one label are one run.
  const runs: Array<{ code: string; from: number; to: number }> = [];
  for (let start = 0, k = 0; start < n; start += block, k += 1) {
    const code = pattern[k % pattern.length] as string;
    const to = Math.min(n, start + block);
    const last = runs[runs.length - 1];
    if (last && last.code === code) last.to = to;
    else runs.push({ code, from: start, to });
  }
  const drops: Array<[number, number]> = [];
  for (let i = 1; i < runs.length; i += 1) {
    const edge = (runs[i] as { from: number }).from;
    if ((runs[i - 1] as { code: string }).code === "T" || (runs[i] as { code: string }).code === "T") {
      drops.push([Math.max(0, edge - firewall), Math.min(n, edge + firewall)]);
    }
  }
  const segments: Segment[] = [];
  for (const run of runs) {
    let cursor = run.from;
    for (const [a, b] of drops) {
      if (b <= run.from || a >= run.to) continue;
      const cut = Math.max(a, run.from);
      segments.push({ role: CODE_ROLE[run.code] ?? "dropped", from: cursor, to: cut });
      segments.push({ role: "dropped", from: cut, to: Math.min(b, run.to) });
      cursor = Math.min(b, run.to);
    }
    segments.push({ role: CODE_ROLE[run.code] ?? "dropped", from: cursor, to: run.to });
  }
  return merged(segments);
}

/**
 * The later-in-the-drive split of `n` frames: the first `trainFrac` of the
 * drive trains and the next `valFrac` validates, the rest tests, each cut
 * where the splits module cuts it (the whole frames below each share), with
 * `firewall` frames dropped on each side of both boundaries.
 */
export function laterSegments(n: number, trainFrac: number, valFrac: number, firewall: number): Segment[] {
  const trainEnd = Math.floor(n * trainFrac);
  const valEnd = Math.floor(n * (trainFrac + valFrac));
  return merged([
    { role: "train", from: 0, to: trainEnd - firewall },
    { role: "dropped", from: trainEnd - firewall, to: trainEnd + firewall },
    { role: "val", from: trainEnd + firewall, to: valEnd - firewall },
    { role: "dropped", from: valEnd - firewall, to: valEnd + firewall },
    { role: "test", from: valEnd + firewall, to: n },
  ]);
}

/** The frames of each role in `segments`. */
export function roleSizes(segments: readonly Segment[]): Record<Role, number> {
  const sizes: Record<Role, number> = { train: 0, val: 0, test: 0, dropped: 0 };
  for (const s of segments) sizes[s.role] += s.to - s.from;
  return sizes;
}

/** The shuffled split's sizes: the frames dealt to each role, as the splits
 * module slices its shuffled frames. It drops none. */
export function shuffledSizes(n: number, trainFrac: number, valFrac: number): Record<Role, number> {
  const trainEnd = Math.floor(n * trainFrac);
  const valEnd = Math.floor(n * (trainFrac + valFrac));
  return { train: trainEnd, val: valEnd - trainEnd, test: n - valEnd, dropped: 0 };
}

/** The fractional part of i times the golden ratio: an even scatter on [0, 1). */
const scatter = (i: number) => {
  const x = (i + 1) * 0.6180339887498949;
  return x - Math.floor(x);
};

/** `cells` cells, each given a role so that the roles come at the shares of
 * `sizes` and stay mixed along the band: the shuffled split, drawn as an
 * illustration. The same cells always get the same roles. */
export function ditherRoles(cells: number, sizes: Readonly<Record<"train" | "val" | "test", number>>): Role[] {
  const total = sizes.train + sizes.val + sizes.test;
  const trainShare = sizes.train / total;
  const valShare = sizes.val / total;
  return Array.from({ length: cells }, (_, i) => {
    const u = scatter(i);
    return u < trainShare ? "train" : u < trainShare + valShare ? "val" : "test";
  });
}

/** A drive's clock, from results.json's `drive_profile`: the first frame of
 * each bucket of `bucketFrames` frames and its time in seconds since the
 * drive's first frame, and the drive's duration. */
export interface DriveClock {
  readonly bucketFrames: number;
  readonly frames: number;
  readonly startFrame: readonly number[];
  readonly startS: readonly number[];
  readonly durationS: number;
}

/** The time of frame `frame` (up to `frames`, the drive's end), seconds: its
 * bucket's start, and the way through the bucket, the bucket running to the
 * next one's start (the last bucket, which may be shorter, to the drive's
 * end). Frames are never taken as evenly spaced over the whole drive. */
export function timeOfFrame(clock: DriveClock, frame: number): number {
  const last = clock.startFrame.length - 1;
  const k = Math.max(0, Math.min(last, Math.floor(frame / clock.bucketFrames)));
  const start = clock.startFrame[k] as number;
  const length = Math.min(clock.bucketFrames, clock.frames - start);
  const t0 = clock.startS[k] as number;
  const t1 = k < last ? (clock.startS[k + 1] as number) : clock.durationS;
  return t0 + ((frame - start) / length) * (t1 - t0);
}

/** The bucket holding the moment `seconds` into the drive. */
export function bucketAtTime(clock: DriveClock, seconds: number): number {
  let low = 0;
  let high = clock.startS.length - 1;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if ((clock.startS[mid] as number) <= seconds) low = mid;
    else high = mid - 1;
  }
  return low;
}

/** How long the buckets run, in seconds: the shortest and the longest, the
 * last (which holds what is left over) not counted. */
export function bucketSpan(clock: DriveClock): { readonly shortest: number; readonly longest: number } {
  const lengths = clock.startS.slice(1).map((start, i) => start - (clock.startS[i] as number));
  const whole = lengths.length > 0 ? lengths : [clock.durationS];
  return { shortest: Math.min(...whole), longest: Math.max(...whole) };
}

/** The steering trace in a few figures: the middle bucket's mean size, the
 * largest and where it is, and the share of buckets above each edge. */
export function traceStats(
  meanAbs: readonly number[],
  edges: readonly number[],
): { readonly median: number; readonly largest: number; readonly largestAt: number; readonly shareAbove: readonly number[] } {
  const sorted = [...meanAbs].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 1 ? (sorted[mid] as number) : ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2;
  const largest = Math.max(...meanAbs);
  return {
    median,
    largest,
    largestAt: meanAbs.indexOf(largest),
    shareAbove: edges.map((edge) => meanAbs.filter((v) => v > edge).length / meanAbs.length),
  };
}

/** How wide a 12 px label is, for placing it clear of other marks: seven
 * pixels a character, a little over what the page's text face takes. */
export const labelWidth = (text: string): number => text.length * 7;

/** How far the label that names the peak keeps from the centre of the
 * peak's dot, px: the dot's radius and a clear pixel. */
export const PEAK_CLEAR = 4.5;

/**
 * Which of the label's wordings to use and where it starts. The label stands
 * beside the dot that marks the peak, `gap` pixels from it (less, down to
 * `PEAK_CLEAR`, where the chart's edge is near), on the side away from
 * `avoidX` -- the demo clip's column, which would otherwise read as the
 * peak's -- or, with nothing to avoid, on the side with more room. `widths`
 * are the wordings' widths, longest first: the longest that fits on that side
 * is used; when none does, the longest that fits on the other side; and when
 * none fits anywhere, the shortest, kept inside the chart.
 */
export function peakLabelPlace(peakX: number, width: number, widths: readonly number[], avoidX: number | null, gap = 6): { readonly wording: number; readonly start: number } {
  const onLeft = (w: number) => {
    const start = Math.max(0, peakX - gap - w);
    return start + w <= peakX - PEAK_CLEAR ? start : null;
  };
  const onRight = (w: number) => {
    const start = Math.min(width - w, peakX + gap);
    return start >= peakX + PEAK_CLEAR ? start : null;
  };
  const wantLeft = avoidX === null ? peakX > width / 2 : avoidX > peakX;
  const sides = wantLeft ? [onLeft, onRight] : [onRight, onLeft];
  for (const side of sides) {
    for (let i = 0; i < widths.length; i += 1) {
      const start = side(widths[i] as number);
      if (start !== null) return { wording: i, start };
    }
  }
  const last = widths.length - 1;
  return { wording: last, start: Math.max(0, Math.min(peakX - (widths[last] as number) / 2, width - (widths[last] as number))) };
}

/** One point per column of `step` pixels: of the buckets whose x falls in a
 * column, the one with the largest mean size of angle, with its own signed
 * mean, so a spike is never averaged away. Columns come in order of x. */
export function columnPeaks(
  xs: readonly number[],
  meanAbs: readonly number[],
  meanSigned: readonly number[],
  step: number,
): Array<{ readonly x: number; readonly abs: number; readonly signed: number }> {
  const out: Array<{ x: number; abs: number; signed: number }> = [];
  let column = -1;
  xs.forEach((x, k) => {
    const c = Math.floor(x / step);
    const abs = meanAbs[k] as number;
    if (c !== column) {
      column = c;
      out.push({ x, abs, signed: meanSigned[k] as number });
    } else if (abs > (out[out.length - 1] as { abs: number }).abs) {
      out[out.length - 1] = { x, abs, signed: meanSigned[k] as number };
    }
  });
  return out;
}
