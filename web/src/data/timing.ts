/**
 * How far apart in time a clip's frames are.
 *
 * Only a drive whose frames carry timestamps has a measured spacing:
 * results.json's `drives.<drive>.median_dt_s`. The other drive's clip plays at
 * the same rate, but how far apart its frames really were is not known, so the
 * page counts them in frames, and quotes a time only as the measured drive's.
 */

import { countWord, fmtMs } from "../format.ts";

/** Each drive's median frame interval in seconds, from results.json's
 * `drives`; null for a drive without timestamps. */
export function driveIntervals(raw: Readonly<Record<string, unknown>>): Record<string, number | null> {
  const drives = raw["drives"];
  if (typeof drives !== "object" || drives === null) return {};
  return Object.fromEntries(
    Object.entries(drives as Record<string, unknown>).map(([drive, info]) => {
      const dt = typeof info === "object" && info !== null ? (info as Record<string, unknown>)["median_dt_s"] : undefined;
      return [drive, typeof dt === "number" && Number.isFinite(dt) && dt > 0 ? dt : null];
    }),
  );
}

export interface FrameTiming {
  /** This clip's drive's median frame interval, seconds; null without timestamps. */
  readonly interval: number | null;
  /** For a drive without timestamps, a drive with them, to quote its spacing. */
  readonly reference: { readonly drive: string; readonly interval: number } | null;
}

/** The timing of `drive`'s frames, from `intervals` (see `driveIntervals`). */
export function frameTiming(intervals: Readonly<Record<string, number | null>>, drive: string): FrameTiming {
  const interval = intervals[drive] ?? null;
  if (interval !== null) return { interval, reference: null };
  const other = Object.entries(intervals).find(([name, dt]) => name !== drive && dt !== null);
  return { interval: null, reference: other ? { drive: other[0], interval: other[1] as number } : null };
}

/** How long before the frame on screen a frame `lag` frames back was taken:
 * "114 ms" when the drive's frames are timed, "two frames" when they are not. */
export function lagText(lag: number, timing: FrameTiming): string {
  if (timing.interval !== null) return fmtMs(lag * timing.interval * 1000, 0);
  return `${countWord(lag)} frame${lag === 1 ? "" : "s"}`;
}
