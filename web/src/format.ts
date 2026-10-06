/**
 * Every number the page prints goes through here.
 *
 * Two rules hold for all of them. A measured non-zero never prints as zero:
 * fixed decimals would turn a 0.04° error into "0.0°", a perfect score nobody
 * measured, so a value that would round away gets more decimals, and below a
 * thousandth of the unit the page states the bound instead of inventing digits.
 * And a negative number carries a true minus sign (U+2212), which reads as a
 * minus at display sizes where a hyphen reads as a dash.
 *
 * Where there is no measurement at all, every formatter prints an em dash.
 */

/** What the page prints where it has no number. */
export const NO_VALUE = "—";

const MINUS = "\u2212";

/** How many decimals beyond the requested ones a formatter may add before it
 * states a bound instead. Two keeps 0.04° and 0.004° readable without printing
 * digits the measurement does not have. */
const MAX_EXTRA_DECIMALS = 2;

/**
 * `magnitude` (≥ 0) with `digits` decimals, widened one decimal at a time, up
 * to `MAX_EXTRA_DECIMALS`, for as long as `needsMore` says the text would
 * misstate the value. Returns the widest attempt if none satisfied it.
 */
function widen(magnitude: number, digits: number, needsMore: (text: string) => boolean): string {
  let text = magnitude.toFixed(digits);
  for (let extra = 1; extra <= MAX_EXTRA_DECIMALS && needsMore(text); extra += 1) {
    text = magnitude.toFixed(digits + extra);
  }
  return text;
}

/**
 * `value` in `unit`, never rounded to zero unless it is zero.
 *
 * A non-zero too small for `digits + MAX_EXTRA_DECIMALS` decimals prints as a
 * bound, e.g. `<0.001°`, with its sign in front.
 */
function signedNonZero(value: number, digits: number, unit: string, guard?: (text: string) => boolean): string {
  if (!Number.isFinite(value)) {
    return NO_VALUE;
  }
  if (value === 0) {
    return `${(0).toFixed(digits)}${unit}`;
  }
  const sign = value < 0 ? MINUS : "";
  const magnitude = Math.abs(value);
  const text = widen(magnitude, digits, (t) => Number(t) === 0 || (guard?.(t) ?? false));
  if (Number(text) === 0) {
    const bound = (10 ** -(digits + MAX_EXTRA_DECIMALS)).toFixed(digits + MAX_EXTRA_DECIMALS);
    return `${sign}<${bound}${unit}`;
  }
  return `${sign}${text}${unit}`;
}

/**
 * A steering angle, or an error between two angles, in degrees.
 *
 * The true angle is printed whatever its size: labels on these drives pass
 * half a turn (501.8° on one of them), and the page shows them as they are.
 */
export function fmtDeg(x: number, digits = 1): string {
  return signedNonZero(x, digits, "°");
}

/**
 * How many times larger one figure is than another, e.g. `4.3×`.
 *
 * One decimal, except that a ratio that differs from one never prints as an
 * even `1.0×`: that would claim two figures are equal when they are not.
 */
export function fmtRatio(x: number): string {
  return signedNonZero(x, 1, "×", (text) => Number(text) === 1 && Math.abs(x) !== 1);
}

/**
 * A fraction as a percentage: `fmtPct(0.6431)` is `64%`.
 *
 * Whole percent by default. A share that is not all of it never prints as
 * `100%`, for the same reason a non-zero never prints as zero.
 */
export function fmtPct(x: number, digits = 0): string {
  const percent = x * 100;
  return signedNonZero(percent, digits, "%", (text) => Number(text) === 100 && Math.abs(percent) !== 100);
}
