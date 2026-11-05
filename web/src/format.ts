/**
 * Every number the page prints goes through here.
 *
 * Two rules hold for all of them. A measured value never prints as a round
 * figure it is not: fixed decimals would turn a 0.04° error into "0.0°", a
 * perfect score nobody measured, or a ratio of 0.9999 into "1.000×", a tie
 * nobody measured. So a value that would round onto such a landmark gets more
 * decimals, and where even those are not enough the page states a bound
 * ("<0.001°", "<1.000×", ">99.99%") instead of inventing digits. And a
 * negative number carries a true minus sign (U+2212), which reads as a minus
 * at display sizes where a hyphen reads as a dash.
 *
 * Where there is no measurement at all, every formatter prints an em dash.
 * Nothing here uses the thin or narrow no-break spaces: the page's font subset
 * has no glyph for them.
 */

/** What the page prints where it has no number. */
export const NO_VALUE = "—";

const MINUS = "\u2212";

/** How many decimals beyond the requested ones a formatter may add before it
 * states a bound instead. Two keeps 0.04° and 0.004° readable without printing
 * digits the measurement does not have. */
const MAX_EXTRA_DECIMALS = 2;

/**
 * A round figure a formatter must not print unless the value is exactly it,
 * besides zero, and how to state the bound when the value sits too close to it.
 *
 * `inner: false` states the landmark itself as the bound: 0.9999 as `<1.000`.
 * `inner: true` never prints the landmark for a value short of it: 99.999 as
 * `>99.99`, the nearest figure below it.
 */
interface Landmark {
  readonly value: number;
  readonly inner: boolean;
}

/**
 * `magnitude` (≥ 0) with `digits` decimals, widened one decimal at a time, up
 * to `MAX_EXTRA_DECIMALS`, for as long as `misstates` says the text would
 * misstate the value. Returns the widest attempt if none satisfied it.
 */
function widen(magnitude: number, digits: number, misstates: (text: string) => boolean): string {
  let text = magnitude.toFixed(digits);
  for (let extra = 1; extra <= MAX_EXTRA_DECIMALS && misstates(text); extra += 1) {
    text = magnitude.toFixed(digits + extra);
  }
  return text;
}

/** `<x` or `>x` for a signed value, with the comparison the right way round
 * for a negative number: −0.0002 lies above −0.001, so it is `>−0.001`. */
function comparison(value: number, magnitudeIsBelow: boolean, magnitudeText: string, unit: string): string {
  const negative = value < 0;
  const operator = magnitudeIsBelow !== negative ? "<" : ">";
  return `${operator}${negative ? MINUS : ""}${magnitudeText}${unit}`;
}

/**
 * `value` in `unit` with `digits` decimals, never printed as zero unless it is
 * zero, nor as `landmark` unless it is exactly that.
 */
function measured(value: number, digits: number, unit: string, landmark?: Landmark): string {
  if (!Number.isFinite(value)) {
    return NO_VALUE;
  }
  if (value === 0) {
    return `${(0).toFixed(digits)}${unit}`;
  }
  const magnitude = Math.abs(value);
  const onLandmark = (text: string) =>
    landmark !== undefined && Number(text) === landmark.value && magnitude !== landmark.value;
  const misstates = (text: string) => Number(text) === 0 || onLandmark(text);
  const text = widen(magnitude, digits, misstates);
  if (!misstates(text)) {
    return `${value < 0 ? MINUS : ""}${text}${unit}`;
  }

  const places = digits + MAX_EXTRA_DECIMALS;
  const step = 10 ** -places;
  if (Number(text) === 0) {
    return comparison(value, true, step.toFixed(places), unit);
  }
  // The value rounds onto the landmark even at the widest: say which side of
  // it the value lies.
  const mark = landmark as Landmark;
  const below = magnitude < mark.value;
  if (mark.inner && below) {
    return comparison(value, false, (mark.value - step).toFixed(places), unit);
  }
  return comparison(value, below, mark.value.toFixed(places), unit);
}

/**
 * A steering angle, or an error between two angles, in degrees.
 *
 * The true angle is printed whatever its size: labels on these drives pass
 * half a turn (501.8° on one of them), and the page shows them as they are.
 */
export function fmtDeg(x: number, digits = 1): string {
  return measured(x, digits, "°");
}

/**
 * A steering angle as its size and the side the wheel is turned to:
 * `{ size: "12.6°", side: "right" }`. The labels count clockwise as positive,
 * so a positive angle is a turn to the right. A wheel that is not turned has
 * no side, and a missing angle prints as an em dash.
 */
export function steerParts(x: number, digits = 1): { size: string; side: "left" | "right" | "" } {
  if (!Number.isFinite(x)) {
    return { size: NO_VALUE, side: "" };
  }
  return { size: fmtDeg(Math.abs(x), digits), side: x > 0 ? "right" : x < 0 ? "left" : "" };
}

/**
 * How many times larger one figure is than another, e.g. `4.3×`.
 *
 * One decimal, except that a ratio that differs from one never prints as an
 * even `1.0×`: that would claim two figures are equal when they are not.
 */
export function fmtRatio(x: number): string {
  return measured(x, 1, "×", { value: 1, inner: false });
}

/**
 * A fraction as a percentage: `fmtPct(0.6431)` is `64%`.
 *
 * Whole percent by default. A share that is not all of it never prints as
 * `100%`, for the same reason a non-zero never prints as zero.
 */
export function fmtPct(x: number, digits = 0): string {
  return measured(x * 100, digits, "%", { value: 100, inner: true });
}

/**
 * A count of things -- frames, runs -- with its thousands grouped by commas:
 * `fmtCount(6000)` is `6,000`. Only whole counts are counts; anything else
 * prints as an em dash rather than as a count nobody made.
 */
export function fmtCount(n: number): string {
  if (!Number.isInteger(n)) {
    return NO_VALUE;
  }
  const digits = String(Math.abs(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return n < 0 ? `${MINUS}${digits}` : digits;
}

/**
 * The spread of a figure over repeated runs (a standard deviation) as
 * `±0.6`, with the figure's own decimals. A spread is never negative; one
 * that was measured is never printed as none.
 */
export function fmtSpread(sd: number, digits = 1): string {
  if (!Number.isFinite(sd) || sd < 0) {
    return NO_VALUE;
  }
  return `±${measured(sd, digits, "")}`;
}

/** The words for the small counts the page spells out. */
const NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];

/** A count as the page's sentences say it: in words while it is small
 * ("three"), in grouped figures beyond ("1,500"). */
export function countWord(n: number): string {
  return NUMBER_WORDS[n] ?? fmtCount(n);
}

const ORDINAL_WORDS = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"];

/** A position as a sentence says it while it is small ("second"); beyond,
 * "number 13". */
export function ordinalWord(n: number): string {
  return ORDINAL_WORDS[n] ?? `number ${fmtCount(n)}`;
}

/** A measured value with no unit of its own -- a pixel difference, a time
 * in seconds or minutes the caller names -- with `digits` decimals, never
 * printed as zero unless it is zero. */
export function fmtFixed(x: number, digits = 1): string {
  return measured(x, digits, "");
}

/** A time in milliseconds: `fmtMs(2.53)` is `2.5 ms`, its number and unit
 * joined by a no-break space (U+00A0, which the page's fonts carry), so a
 * line never ends between them. */
export function fmtMs(x: number, digits = 1): string {
  if (!Number.isFinite(x) || x < 0) {
    return NO_VALUE;
  }
  return measured(x, digits, "\u00A0ms");
}
