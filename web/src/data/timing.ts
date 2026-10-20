/**
 * How far apart in time a clip's frames are.
 *
 * A clip's history frames are named by how many frames back they are, which is
 * true of every frame of it. Only a drive whose frames carry timestamps has a
 * measured time to add: results.json's `timing.<drive>.lag_gap_s`, for each
 * frame lag the shipped model uses, the spread of the time between frames that
 * far apart over the whole drive. The page quotes its median as the drive's,
 * never as the clip's: a clip's own frames have their own spread. A drive
 * without timestamps has null, and is counted in frames only.
 */

import { countWord, fmtMs } from "../format.ts";

/** A drive's measured gaps: for each frame lag, the median time in seconds
 * between frames that far apart. */
export type Gaps = ReadonlyMap<number, number>;

const isLag = (lag: unknown): lag is number => typeof lag === "number" && Number.isInteger(lag) && lag >= 1;
const isTime = (s: unknown): s is number => typeof s === "number" && Number.isFinite(s) && s > 0;

/** Each drive's gaps from results.json's `timing` (`raw`); null for a drive
 * without timestamps, or whose gaps are not in the form this page reads, so it
 * is only ever counted in frames. */
export function driveGaps(raw: Readonly<Record<string, unknown>>): Record<string, Gaps | null> {
  const timing = raw["timing"];
  if (typeof timing !== "object" || timing === null || Array.isArray(timing)) return {};
  return Object.fromEntries(
    Object.entries(timing as Record<string, unknown>).map(([drive, info]) => {
      const list = typeof info === "object" && info !== null ? (info as Record<string, unknown>)["lag_gap_s"] : undefined;
      if (!Array.isArray(list) || list.length === 0) return [drive, null];
      const gaps = new Map<number, number>();
      for (const entry of list) {
        const { lag, median } = (typeof entry === "object" && entry !== null ? entry : {}) as Record<string, unknown>;
        if (!isLag(lag) || !isTime(median) || gaps.has(lag)) return [drive, null];
        gaps.set(lag, median);
      }
      return [drive, gaps];
    }),
  );
}

/** The frame lags the shipped model uses, as every timed drive lists them;
 * null when no drive is timed, or two disagree. */
export function timedLags(raw: Readonly<Record<string, unknown>>): readonly number[] | null {
  const lists = Object.values(driveGaps(raw)).flatMap((gaps) => (gaps ? [[...gaps.keys()].sort((a, b) => a - b)] : []));
  const [first] = lists;
  if (!first) return null;
  return lists.every((list) => list.join() === first.join()) ? first : null;
}

export interface FrameTiming {
  /** The clip's drive. */
  readonly drive: string;
  /** Its measured gaps; null when its frames carry no timestamps. */
  readonly gaps: Gaps | null;
  /** For a drive without timestamps, a drive with them, to quote. */
  readonly reference: { readonly drive: string; readonly gaps: Gaps } | null;
}

/** The timing of `drive`'s frames, from `gaps` (see `driveGaps`). */
export function frameTiming(gaps: Readonly<Record<string, Gaps | null>>, drive: string): FrameTiming {
  const own = gaps[drive] ?? null;
  if (own !== null) return { drive, gaps: own, reference: null };
  const other = Object.entries(gaps).find(([name, measured]) => name !== drive && measured !== null);
  return { drive, gaps: null, reference: other ? { drive: other[0], gaps: other[1] as Gaps } : null };
}

/** How far back a history frame is: "two frames", "one frame". */
export function framesBack(lag: number): string {
  return `${countWord(lag)} frame${lag === 1 ? "" : "s"}`;
}

/** The median time between frames `lag` apart, "a median 100 ms"; null when
 * it was not measured. */
export function medianGap(lag: number, gaps: Gaps | null): string | null {
  const median = gaps?.get(lag);
  return median === undefined ? null : `a median ${fmtMs(median * 1000, 0)}`;
}

/** "the 2018 drive" for "drive2018". */
export function driveName(drive: string): string {
  const year = /(\d{4})$/.exec(drive)?.[1];
  return year ? `the ${year} drive` : `the drive ${drive}`;
}
