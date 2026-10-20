import { describe, expect, it } from "vitest";

import css from "./styles.css?raw";

/** The declarations of every rule whose selector list is exactly `selector`. */
function rules(selector: string): string[] {
  const out: string[] = [];
  const pattern = /([^{}]+)\{([^{}]*)\}/g;
  for (const match of css.matchAll(pattern)) {
    if ((match[1] ?? "").trim() === selector) out.push(match[2] ?? "");
  }
  return out;
}

describe("the cockpit's note", () => {
  it("states the floor of its slot once", () => {
    expect(css.match(/min-height:\s*6em/g) ?? []).toHaveLength(1);
  });

  it("stacks the note and the unseen copies of every note it can show in one cell, so the slot fits the longest", () => {
    expect(rules(".model-note > *").join(" ")).toMatch(/grid-area:\s*1\s*\/\s*1/);
    expect(rules(".model-note__sizer").join(" ")).toMatch(/visibility:\s*hidden/);
  });
});

// -- The charts' colours -------------------------------------------------------

const channel = (hex: string, i: number) => parseInt(hex.slice(1 + 2 * i, 3 + 2 * i), 16) / 255;
const linear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const gamma = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

/** A colour in OKLab, and back: `color-mix(in oklab, …)` mixes there. */
function oklab(hex: string): [number, number, number] {
  const [r, g, b] = [0, 1, 2].map((i) => linear(channel(hex, i))) as [number, number, number];
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}

function rgb([L, a, b]: [number, number, number]): [number, number, number] {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ].map((c) => Math.min(1, Math.max(0, gamma(c)))) as [number, number, number];
}

const luminance = ([r, g, b]: [number, number, number]) => 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);
const contrast = (a: [number, number, number], b: [number, number, number]) => {
  const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p) as [number, number];
  return (x + 0.05) / (y + 0.05);
};
const solid = (hex: string): [number, number, number] => [0, 1, 2].map((i) => channel(hex, i)) as [number, number, number];

/** The value of `--name` in the block that starts at `from` in the stylesheet. */
function token(name: string, from: number): string {
  const found = new RegExp(`--${name}:\\s*(#[0-9a-f]{6})`).exec(css.slice(from));
  if (!found?.[1]) throw new Error(`no --${name} after ${from}`);
  return found[1];
}

describe("the charts' colours", () => {
  const themes = {
    light: css.indexOf(":root {"),
    "dark by preference": css.indexOf(':root:not([data-theme="light"])'),
    "dark by choice": css.indexOf(':root[data-theme="dark"]'),
  };

  it.each(Object.entries(themes))("give every network 3:1 or more on the ladder's spotlight, %s", (_, at) => {
    expect(at).toBeGreaterThanOrEqual(0);
    // The spotlight: the accent, 7% of the way in from the surface, mixed in OKLab.
    expect(css).toMatch(/--spot:\s*color-mix\(in oklab, var\(--accent\) 7%, var\(--surface\)\)/);
    const accent = oklab(token("accent", themes.light));
    const surface = oklab(token("surface", at));
    const spot = rgb([0, 1, 2].map((i) => 0.07 * (accent[i] ?? 0) + 0.93 * (surface[i] ?? 0)) as [number, number, number]);
    for (const series of ["sightlinenet", "pilotnet", "oneframe", "straight"]) {
      expect(contrast(solid(token(`model-${series}`, at)), spot), series).toBeGreaterThanOrEqual(3);
      expect(contrast(solid(token(`model-${series}`, at)), solid(token("surface", at))), series).toBeGreaterThanOrEqual(3);
    }
  });
});

describe("the charts' words", () => {
  it("never sit on a rule: the slope chart's note and every value wear a halo of what is behind them", () => {
    const halo = [...css.matchAll(/([^{}]+)\{([^{}]*paint-order:\s*stroke[^{}]*)\}/g)];
    const selectors = halo.map((m) => m[1] ?? "").join(" ");
    for (const part of [".chart__value", ".slope .chart__note", ".card-chart__label"]) expect(selectors).toContain(part);
    expect(halo.map((m) => m[2]).join(" ")).toMatch(/stroke:\s*var\(--halo, var\(--surface\)\)/);
    expect(rules(".ladder-chart .is-lit").join(" ")).toMatch(/--halo:\s*var\(--spot\)/);
  });

  it("put away a whole part of the sources card that cannot be shown, its heading with it", () => {
    expect(css).toMatch(/\.sources-card\[data-state="unavailable"\] :is\(\[data-card="live"\], \.sources-card__part--splits\)/);
  });
});
