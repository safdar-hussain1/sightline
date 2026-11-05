/**
 * The markup of the sections around the cockpit -- How it works and the
 * accuracy figure on the demo view -- before any data has loaded; the Results
 * view's is in results-markup.ts.
 *
 * Plain HTML with no behaviour, so the build writes it into index.html (see
 * inject.ts, called from vite.config.ts): the first paint has each
 * section's final layout, and a crawler reads its words. Nothing here is a
 * measurement. Where a sentence will carry numbers, the page is served with a
 * plain version of it, without them, and the section writes the numbers in
 * once the data has loaded; the illustrations stand in for the live frames
 * until the clip's frames are decoded, so no box is ever empty.
 */

import { wheelSvg } from "../cockpit/wheel.ts";

/** The empty elements index.html holds for the sections. */
export const HOW_ROOT = `<section class="how" id="how-it-works" aria-labelledby="how-title"></section>`;
export const ACCURACY_ROOT = `<section class="accuracy" id="accuracy" aria-labelledby="accuracy-title"></section>`;
export const RESULTS_ROOT = `<section class="results" id="results-view" aria-labelledby="results-title"></section>`;

// -- Drawings ---------------------------------------------------------------------

/** The road as the model's crop shows it, drawn: the stand-in for a frame of
 * the clip until its frames are decoded. 200 x 66, the model's input. */
const ROAD = `<svg class="road" viewBox="0 0 200 66" preserveAspectRatio="none" aria-hidden="true" focusable="false">
            <rect class="road__ground" width="200" height="66"/>
            <path class="road__verge" d="M0 18h200v6H0z"/>
            <path class="road__surface" d="M58 18h84l58 34v14H0V52z"/>
            <path class="road__line" d="M99 19 82 66M101 19l17 47"/>
            <path class="road__edge" d="M60 19 0 54M140 19l60 35"/>
          </svg>`;

/** The road with heat where a steering model tends to look: the stand-in for
 * the third step's map. */
const ROAD_HEAT = ROAD.replace(
  "</svg>",
  `<defs><radialGradient id="how-heat-glow"><stop offset="0" class="road__heat-core"/><stop offset="0.55" class="road__heat-mid"/><stop offset="1" class="road__heat-edge"/></radialGradient></defs>
            <ellipse cx="96" cy="34" rx="44" ry="15" fill="url(#how-heat-glow)"/>
            <ellipse cx="150" cy="30" rx="26" ry="9" fill="url(#how-heat-glow)" opacity="0.6"/>
          </svg>`,
);

/** A slab of the network: a box seen from above-right, its front face
 * `w` x `h` with its top-left corner at (`x`, `y`). */
function slab(x: number, y: number, w: number, h: number): string {
  const dx = 7;
  const dy = 5;
  return (
    `<g class="graph__slab">` +
    `<path class="graph__top" d="M${x} ${y}l${dx} -${dy}h${w}l-${dx} ${dy}z"/>` +
    `<path class="graph__side" d="M${x + w} ${y}l${dx} -${dy}v${h}l-${dx} ${dy}z"/>` +
    `<rect class="graph__front" x="${x}" y="${y}" width="${w}" height="${h}"/>` +
    `<rect class="graph__glow" x="${x}" y="${y}" width="${w}" height="${h}"/>` +
    `</g>`
  );
}

/** The network as one block: its convolution slabs narrowing towards the
 * dense layers, the angle coming out at the right, and, underneath, the path
 * back to the picture that the map of where it looked is drawn along. The
 * slabs follow the network's shape (five convolutions, three dense layers),
 * not its scale. */
function graphSvg(): string {
  const cy = 64;
  const convs = [
    [36, 7, 78],
    [66, 9, 60],
    [98, 11, 44],
    [132, 13, 30],
    [166, 13, 22],
  ] as const;
  const dense = [
    [202, 40],
    [212, 26],
    [222, 12],
  ] as const;
  const slabs = convs.map(([x, w, h]) => slab(x, cy - h / 2 + 2, w, h)).join("");
  const bars = dense
    .map(([x, h]) => {
      const box = `x="${x}" y="${cy - h / 2}" width="4" height="${h}" rx="2"`;
      return `<rect class="graph__dense" ${box}/><rect class="graph__glow" ${box}/>`;
    })
    .join("");
  return `<svg class="graph" viewBox="0 0 264 120" aria-hidden="true" focusable="false">
            <rect class="graph__case" x="1" y="1" width="262" height="118" rx="14"/>
            <path class="graph__wire" d="M12 ${cy}H36M50 ${cy}H66M82 ${cy}H98M116 ${cy}H132M152 ${cy}H166M186 ${cy}H202M216 ${cy}H222M226 ${cy}H244"/>
            <path class="graph__back" pathLength="1" d="M176 ${cy + 18}C176 108 52 110 42 ${cy + 42}"/>
            <path class="graph__back-tip" d="m46 ${cy + 36} -4 ${7} -5 -6"/>
            ${slabs}
            ${bars}
            <circle class="graph__out" cx="248" cy="${cy}" r="6"/>
            <circle class="graph__glow graph__glow--out" cx="248" cy="${cy}" r="6"/>
          </svg>`;
}

const PAUSE_ICON = `<svg class="how__toggle-icon how__toggle-icon--pause" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/></svg>`;
const PLAY_ICON = `<svg class="how__toggle-icon how__toggle-icon--play" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 5.5v13l10.5-6.5z"/></svg>`;

/** The joint between two steps: the data passing from one to the next. */
const WIRE = `<span class="how-step__wire" aria-hidden="true"><span class="how-step__pulse"></span></span>`;

// -- How it works -----------------------------------------------------------------

export function howHtml(): string {
  return `
        <div class="how__inner wrap">
          <div class="how__head">
            <h2 class="how__title" id="how-title">How it works</h2>
            <p class="how__lede">Each turn of the wheel above is one answer from the model. This is how it gets there, on the clip above.</p>
            <button class="how__toggle" type="button" data-how="toggle" data-playing="false" hidden>${PAUSE_ICON}${PLAY_ICON}<span data-how="toggle-label">Play</span><span class="visually-hidden"> the steps</span></button>
          </div>
          <ol class="how__steps" data-how="steps" data-live="false">
            <li class="how-step how-step--frames">
              <div class="how-step__visual">
                <div class="how-frames" data-how="frames">
                  <figure class="how-frame how-frame--back">
                    <span class="how-frame__picture">${ROAD}<canvas class="how-frame__canvas" data-how="frame-1" width="200" height="66"></canvas></span>
                    <figcaption class="how-frame__age" data-how="age-1">just before</figcaption>
                  </figure>
                  <figure class="how-frame how-frame--front">
                    <span class="how-frame__picture">${ROAD}<canvas class="how-frame__canvas" data-how="frame-0" width="200" height="66"></canvas></span>
                    <figcaption class="how-frame__age" data-how="age-0">now</figcaption>
                  </figure>
                </div>
                ${WIRE}
              </div>
              <div class="how-step__text">
                <h3 class="how-step__title" data-how="frames-title">Frames of the road</h3>
                <p data-how="frames-text">The frame on screen and one taken just before it, cropped to the road and shrunk to the size the model reads. Together they show which way the view is moving.</p>
              </div>
            </li>
            <li class="how-step how-step--graph">
              <div class="how-step__visual">
                <div class="how-graph" data-how="graph">${graphSvg()}</div>
                ${WIRE}
              </div>
              <div class="how-step__text">
                <h3 class="how-step__title">One ONNX graph, in your browser</h3>
                <p>The trained network is a single ONNX file. It runs on this device, in WebAssembly, and no frame leaves the page. <span data-how="speed"></span></p>
              </div>
            </li>
            <li class="how-step how-step--answer">
              <div class="how-step__visual">
                <div class="how-answer">
                  <div class="how-answer__dial">
                    <div class="how-answer__wheel" data-how="wheel">${wheelSvg()}</div>
                    <p class="how-answer__angle" data-how="readout" hidden><span class="how-answer__size" data-how="angle"></span> <span class="how-answer__side" data-how="side"></span></p>
                  </div>
                  <div class="how-answer__map">
                    <span class="how-frame__picture">${ROAD_HEAT}<canvas class="how-frame__canvas" data-how="heat" width="200" height="66"></canvas></span>
                    <p class="heat-key" aria-hidden="true"><span>less</span><span class="heat-key__ramp"></span><span>more</span></p>
                  </div>
                </div>
              </div>
              <div class="how-step__text">
                <h3 class="how-step__title">An angle, and where it looked</h3>
                <p>Out come a steering angle and a heat map of where it looked: the brighter the spot, the more that part of the road counted.</p>
              </div>
            </li>
          </ol>
          <p class="how__caveat">It estimates the steering angle being applied from what it sees. It does not drive.</p>
        </div>
      `;
}

// -- The accuracy figure ----------------------------------------------------------

export function accuracyHtml(): string {
  return `
        <div class="accuracy__inner wrap" data-accuracy="root" data-state="loading">
          <h2 class="accuracy__title" id="accuracy-title">How accurate it is</h2>
          <p class="accuracy__lead" data-accuracy="sentence">Sightline is scored on minutes of driving it never trained on, against always steering straight on the same frames.</p>
          <figure class="bars" data-accuracy="figure">
            <div class="bars__plot" role="img" data-accuracy="plot" aria-label="Mean error of Sightline and of always steering straight, on the same frames.">
              <div class="bars__row bars__row--model">
                <span class="bars__name">Sightline</span>
                <span class="bars__track"><span class="bars__bar" data-accuracy="bar-model"></span><span class="bars__value" data-accuracy="value-model"></span></span>
              </div>
              <div class="bars__row bars__row--straight">
                <span class="bars__name">Always steering straight</span>
                <span class="bars__track"><span class="bars__bar" data-accuracy="bar-straight"></span><span class="bars__value" data-accuracy="value-straight"></span></span>
              </div>
            </div>
            <figcaption class="bars__caption" data-accuracy="caption">Mean error on the frames its training held back, in degrees of steering-wheel angle.</figcaption>
          </figure>
          <p class="accuracy__leak" data-accuracy="leak">Test it the usual way, on shuffled frames, and the same kind of model looks better than it is on road it has not seen nearby.</p>
          <p class="accuracy__links"><a class="accuracy__more" href="#results">See what the measurements found</a><a href="model-card.md">Read the model card</a></p>
        </div>
      `;
}

// -- Writing a section in ----------------------------------------------------------

/** `root` (an empty element) with `inner` written into it. */
export function filled(root: string, inner: string): string {
  const close = root.lastIndexOf("></");
  return `${root.slice(0, close + 1)}${inner}${root.slice(close + 1)}`;
}

/** Write a section in if the page arrived without it (served from the
 * template rather than the build). The markup is the app's own constant text,
 * with nothing from outside interpolated. */
export function mountMarkup(root: string, inner: () => string): void {
  const id = /id="([^"]+)"/.exec(root)?.[1];
  const element = id ? document.getElementById(id) : null;
  if (element && element.childElementCount === 0) element.innerHTML = inner();
}

/** The demo view's sections, written in if the page arrived without them. */
export function mountDemoMarkup(): void {
  mountMarkup(HOW_ROOT, howHtml);
  mountMarkup(ACCURACY_ROOT, accuracyHtml);
}
