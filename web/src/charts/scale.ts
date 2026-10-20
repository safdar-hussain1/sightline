/**
 * Scales for the Results view's charts: a value to a position, and the round
 * figures an axis is marked at.
 *
 * Every chart draws from zero or through it -- an error starts at zero, and a
 * skill is measured from always steering straight, its zero -- so the extent
 * of an axis always holds zero, and is widened to round figures so its ticks
 * land on its ends.
 */

/** A straight map from `domain` to `range`. A domain of no width maps to the
 * middle of the range. */
export function linear(domain: readonly [number, number], range: readonly [number, number]): (value: number) => number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  if (d1 === d0) return () => (r0 + r1) / 2;
  const k = (r1 - r0) / (d1 - d0);
  return (value) => r0 + (value - d0) * k;
}

/** The round step for about `count` intervals across `span`: 1, 2, 2.5 or
 * 5 times a power of ten. */
export function niceStep(span: number, count: number): number {
  if (!(span > 0) || !(count > 0)) return 1;
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  const multiple = unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 2.5 ? 2.5 : unit <= 5 ? 5 : 10;
  return multiple * power;
}

/** The axis for values from `low` to `high`: zero always inside, both ends
 * widened to a whole number of round steps. Returns the ends and the ticks,
 * from the bottom up. */
export function niceAxis(low: number, high: number, count = 4): { readonly extent: readonly [number, number]; readonly ticks: readonly number[] } {
  const lo = Math.min(0, low);
  const hi = Math.max(0, high);
  const step = niceStep(hi - lo || 1, count);
  // A hair of tolerance, so a value that is a round figure is not widened past it.
  const bottom = Math.floor(lo / step + 1e-9) * step;
  const top = Math.ceil(hi / step - 1e-9) * step;
  const ticks: number[] = [];
  for (let i = 0; bottom + i * step <= top + step * 1e-9; i += 1) {
    // Rounded at the step's own decimals, so 0.1 * 3 is 0.3, not 0.30000000000000004.
    const decimals = Math.max(0, -Math.floor(Math.log10(step)) + 1);
    ticks.push(Number((bottom + i * step).toFixed(decimals)) + 0);
  }
  return { extent: [ticks[0] ?? bottom, ticks.at(-1) ?? top], ticks };
}
