/**
 * Where the model looks, as heat.
 *
 * The network returns a saliency map with every frame: a 200 x 66 grid over
 * its own input, scaled to [0, 1] per frame (VisualBackProp: the regions that
 * stay active through every convolution layer). The page paints it with one
 * ramp -- night, violet, ember, amber, pale sun -- whose five stops are the
 * `--heat-0..4` tokens in styles.css (a test keeps the two equal). Between the
 * stops the ramp is interpolated in OKLab, so equal steps in attention look
 * like equal steps in colour, and it rises in luminance throughout, so "where
 * it looks" reads as brightness even without colour vision.
 */

/** The display transfer from attention to the ramp: attention^0.6. The maps are
 * sparse -- on the shipped model's self-test frames half the pixels sit under
 * 0.04 and nine in ten under 0.15 -- so a linear ramp would draw nearly all of
 * the road in its darkest stop; the power spreads the ramp over the values
 * that occur, and still maps 0 to the first stop and 1 to the last. */
export const HEAT_GAMMA = 0.6;

/** The share of the full opacity that a pixel of no attention gets: the heat
 * dims the road it ignores a little and covers what it attends to most, so the
 * road stays readable under it. */
export const HEAT_FLOOR = 0.35;

/** The ramp's stops, darkest first. styles.css declares the same five. */
export const HEAT_STOPS = ["#0e0b1d", "#9333ea", "#f26a2e", "#fdb44b", "#fff3c4"] as const;

const LUT_SIZE = 256;

/** The LUT entry that holds stop `k` exactly. */
export function stopIndex(k: number): number {
  return Math.round((k * (LUT_SIZE - 1)) / (HEAT_STOPS.length - 1));
}

// -- Colour arithmetic ---------------------------------------------------------

type Triple = [number, number, number];

function hexToRgb(hex: string): Triple {
  return [1, 3, 5].map((at) => Number.parseInt(hex.slice(at, at + 2), 16)) as Triple;
}

function toLinear(channel: number): number {
  const c = channel / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function fromLinear(value: number): number {
  const v = Math.min(1, Math.max(0, value));
  const c = v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055;
  return Math.round(c * 255);
}

/** WCAG relative luminance of an sRGB colour, 0 (black) to 1 (white). */
export function relativeLuminance(r: number, g: number, b: number): number {
  return 0.2126 * toLinear(r) + 0.7152 * toLinear(g) + 0.0722 * toLinear(b);
}

// OKLab (Ottosson 2020), from and to linear sRGB.
function rgbToOklab([r, g, b]: Triple): Triple {
  const [lr, lg, lb] = [toLinear(r), toLinear(g), toLinear(b)];
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

function oklabToRgb([L, a, b]: Triple): Triple {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [
    fromLinear(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    fromLinear(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    fromLinear(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
  ];
}

// -- The lookup table ------------------------------------------------------------

let shared: Uint8ClampedArray | null = null;

/** The heat ramp as 256 opaque RGBA entries, index = attention x 255. Built
 * once and shared: every caller gets the same array, so none may write to it. */
export function heatLUT(): Uint8ClampedArray {
  if (shared) {
    return shared;
  }
  const lut = new Uint8ClampedArray(LUT_SIZE * 4);
  const stops = HEAT_STOPS.map((hex) => rgbToOklab(hexToRgb(hex)));
  for (let k = 0; k < stops.length - 1; k += 1) {
    const from = stopIndex(k);
    const to = stopIndex(k + 1);
    const a = stops[k] as Triple;
    const b = stops[k + 1] as Triple;
    for (let i = from; i <= to; i += 1) {
      const t = (i - from) / (to - from);
      const rgb =
        i === from ? hexToRgb(HEAT_STOPS[k] as string)
        : i === to ? hexToRgb(HEAT_STOPS[k + 1] as string)
        : oklabToRgb([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
      lut.set([...rgb, 255], i * 4);
    }
  }
  shared = lut;
  return lut;
}

// -- Saliency maps to pixels -------------------------------------------------------

/** A saliency map as one byte per pixel, its place on the ramp: attention
 * through `HEAT_GAMMA`, x 255, rounded. How the cockpit keeps every frame's map
 * without four bytes per pixel. A value outside [0, 1] is clamped, and a
 * missing one (NaN) is no attention. */
export function quantizeSaliency(sal: Float32Array): Uint8Array {
  const out = new Uint8Array(sal.length);
  for (let i = 0; i < sal.length; i += 1) {
    const v = sal[i] as number;
    out[i] = v > 0 ? Math.round(Math.min(v, 1) ** HEAT_GAMMA * 255) : 0;
  }
  return out;
}

/** A byte-per-pixel saliency map (see `quantizeSaliency`) as heat whose
 * opacity rises with attention, from `HEAT_FLOOR` x `alpha` to `alpha`. */
export function heatImageFromBytes(bytes: Uint8Array, w = 200, h = 66, alpha = 0.6): ImageData {
  if (bytes.length !== w * h) {
    throw new Error(`a ${w} x ${h} saliency map has ${w * h} values, not ${bytes.length}`);
  }
  const lut = heatLUT();
  const data = new Uint8ClampedArray(w * h * 4);
  const full = Math.min(1, Math.max(0, alpha)) * 255;
  for (let i = 0; i < bytes.length; i += 1) {
    const place = bytes[i] as number;
    const at = place * 4;
    const out = i * 4;
    data[out] = lut[at] as number;
    data[out + 1] = lut[at + 1] as number;
    data[out + 2] = lut[at + 2] as number;
    data[out + 3] = Math.round(full * (HEAT_FLOOR + (1 - HEAT_FLOOR) * (place / 255)));
  }
  return new ImageData(data, w, h);
}

/** A saliency map, values in [0, 1], as heat: 0 is the ramp's first entry, 1
 * its last, and between them `HEAT_GAMMA` sets the place; `alpha` is the
 * opacity of full attention. */
export function saliencyToImageData(sal: Float32Array, w = 200, h = 66, alpha = 0.6): ImageData {
  if (sal.length !== w * h) {
    throw new Error(`a ${w} x ${h} saliency map has ${w * h} values, not ${sal.length}`);
  }
  return heatImageFromBytes(quantizeSaliency(sal), w, h, alpha);
}
