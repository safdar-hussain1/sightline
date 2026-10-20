/**
 * The markup of the sections around the cockpit -- How it works and the
 * accuracy figure on the demo view, and the Results view -- before any data
 * has loaded.
 *
 * Plain HTML with no behaviour, so the build writes it into index.html (see
 * `injectSections`, called from vite.config.ts): the first paint has each
 * section's final layout, and a crawler reads its words. Nothing here is a
 * measurement. Where a sentence will carry numbers, the page is served with a
 * plain version of it, without them, and the section writes the numbers in
 * once the data has loaded; the illustrations stand in for the live frames
 * until the clip's frames are decoded, so no box is ever empty.
 */

import { wheelSvg } from "../cockpit/wheel.ts";
import { networkName } from "./names.ts";

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

// -- The Results view -------------------------------------------------------------

/** A small two-bar ruler of the errors behind a figure, on the band's shared
 * degree scale: each row its name, its bar and its value. `rows` are
 * [modifier, name, part] for each bar. */
function ruler(name: string, title: string, rows: ReadonlyArray<readonly [string, string, string]>): string {
  const bars = rows
    .map(
      ([modifier, label, part]) =>
        `<div class="ruler__row ruler__row--${modifier}"><span class="ruler__name">${label}</span><span class="ruler__track"><span class="ruler__bar" data-results="bar-${part}"></span><span class="ruler__value" data-results="value-${part}"></span></span></div>`,
    )
    .join("");
  return `<div class="ruler" role="img" data-results="${name}" aria-label="${title}"><p class="ruler__title" aria-hidden="true">${title}</p>${bars}</div>`;
}

/** A figure that counts up: the final text sets the width; the counting text
 * is drawn over it; assistive technology reads only the final text. */
function count(name: string): string {
  return `<span class="count" data-count="${name}"><span class="count__final" aria-hidden="true"></span><span class="count__shown" aria-hidden="true"></span><span class="count__text visually-hidden"></span></span>`;
}

export function resultsHtml(): string {
  return `
        <div class="results__inner wrap" data-results="root" data-state="loading">
          <div class="results__head">
            <h2 class="results__title" id="results-title">What the measurements found</h2>
            <p class="results__lede" data-results="lede">How good a steering model looks depends on how it is tested. Here is what its training runs found.</p>
          </div>
          <ul class="headline" data-results="band" role="list">
            <li class="finding finding--hero">
              <p class="finding__values finding__values--pair"><span class="finding__pair">${count("ratio")}<span class="finding__split">SightlineNet</span></span><span class="finding__pair finding__pair--also">${count("ratio-pilotnet")}<span class="finding__split">PilotNet</span></span></p>
              <p class="finding__label" data-results="paired-label">How much better testing on shuffled frames makes the same kind of model look</p>
              ${ruler("ruler-paired", "SightlineNet’s mean error on the same frames", [
                ["reported", "shuffled split", "paired-shuffled"],
                ["model", "blocked split", "paired-blocked"],
              ])}
              <p class="finding__note" data-results="paired-note">Each network trained both ways and scored on the same held-out frames.</p>
            </li>
            <li class="finding">
              <p class="finding__values">${count("another-day")}</p>
              <p class="finding__label" data-results="another-day-label">Better than always steering straight, on a drive it never saw</p>
              ${ruler("ruler-another-day", "Mean error on another day’s drive", [
                ["model", "SightlineNet", "day-model"],
                ["straight", "steering straight", "day-straight"],
              ])}
              <p class="finding__note" data-results="another-day-note">SightlineNet on another day’s drive, against steering straight on the same frames.</p>
            </li>
            <li class="finding">
              <p class="finding__values finding__values--pair"><span class="finding__pair">${count("memoriser-shuffled")}<span class="finding__split">shuffled split</span></span><span class="finding__pair">${count("memoriser-blocked")}<span class="finding__split">blocked split</span></span></p>
              <p class="finding__label">A lookup table that copies the angle of the most similar training frame</p>
              <p class="finding__note" data-results="memoriser-note">It learns nothing, so its error only shows how close the test frames sit to training frames.</p>
            </li>
          </ul>
          <p class="trust" data-results="trust"><svg class="trust__icon" viewBox="0 0 20 20" aria-hidden="true" focusable="false"><circle cx="10" cy="10" r="8.25"/><path class="trust__tick" d="m6.2 10.3 2.5 2.5 5.1-5.4"/><path class="trust__alert" d="M10 5.6v5.6M10 14.1v.3"/></svg><span class="trust__text"><span class="trust__verdict" data-results="trust-verdict" role="status"></span><span class="trust__progress" data-results="trust-text">Checking this browser against the same model run in Python…</span></span></p>
          ${pairedHtml()}
          ${ladderHtml()}
          ${cardsHtml()}
        </div>
      `;
}

// -- The paired experiment ----------------------------------------------------------

/** The frames of a stretch of drive in the explainer's sketch, and which of
 * them each split holds out to test: the shuffled split's scattered, the
 * blocked split's in stretches. A drawing of the idea, not of the data. */
const SKETCH_FRAMES = 30;
const SKETCH_SHUFFLED = [2, 6, 11, 14, 19, 24, 27];
const SKETCH_BLOCKED = [9, 10, 11, 12, 13, 14, 15, 22, 23, 24, 25, 26, 27];

function sketchRow(name: string, out: readonly number[], kind: string): string {
  const cells = Array.from({ length: SKETCH_FRAMES }, (_, i) => `<span class="sketch__frame${out.includes(i) ? ` sketch__frame--${kind}` : ""}"></span>`).join("");
  return `<div class="sketch__row"><span class="sketch__name">${name}</span><span class="sketch__frames">${cells}</span></div>`;
}

function sketchHtml(): string {
  const both = SKETCH_SHUFFLED.filter((i) => SKETCH_BLOCKED.includes(i));
  return `<div class="sketch__drawing" role="img" aria-label="A sketch of a stretch of the drive, frame by frame: the shuffled split holds out frames scattered at random, the blocked split holds out whole stretches, and only a few frames are held out by both.">
              ${sketchRow("shuffled split", SKETCH_SHUFFLED, "out")}
              ${sketchRow("blocked split", SKETCH_BLOCKED, "out")}
              ${sketchRow("held out by both", both, "both")}
            </div>`;
}

export function pairedHtml(): string {
  return `<div class="study study--paired paired" data-paired="root" data-state="loading">
            <div class="paired__text">
              <h3 class="study__title" id="paired-title">Trained two ways, scored on the same frames</h3>
              <p class="study__lead" data-paired="lead">Each network is trained twice, once on shuffled frames and once with whole stretches of the drive held back, and both are scored on the very same frames.</p>
              <p class="study__meaning" data-paired="meaning">A shuffled test makes the same kind of model look better than it is on road it has not seen nearby.</p>
              <p class="study__scale" data-paired="scale"></p>
              <figure class="sketch">
                ${sketchHtml()}
                <figcaption class="sketch__caption">
                  <dl class="sketch__splits">
                    <div><dt>Shuffled split</dt><dd data-paired="shuffled">Frames are dealt out at random, so most test frames sit right beside a frame the model trained on.</dd></div>
                    <div><dt>Blocked split</dt><dd data-paired="blocked">Whole stretches of road are held back, so every test frame is far from the frames the model trained on.</dd></div>
                  </dl>
                  <p class="sketch__note">A sketch, not to scale: the frames both splits hold out are scored by a network trained each way.</p>
                </figcaption>
              </figure>
            </div>
            <figure class="chart chart--slope" data-paired="figure">
              <div class="chart__head">
                <p class="chart__title">Mean error on the frames both splits held out</p>
                <fieldset class="segmented" data-paired="networks">
                  <legend class="visually-hidden">Network</legend>
                  <label class="segmented__option"><input class="segmented__input" type="radio" name="paired-network" value="sightlinenet" checked /><span class="segmented__swatch segmented__swatch--sightlinenet" aria-hidden="true"></span>SightlineNet</label>
                  <label class="segmented__option"><input class="segmented__input" type="radio" name="paired-network" value="pilotnet" /><span class="segmented__swatch segmented__swatch--pilotnet" aria-hidden="true"></span>PilotNet</label>
                </fieldset>
              </div>
              <div class="chart__plot chart__plot--slope" data-paired="plot">
                ${SLOPE_PLACEHOLDER}
                <div class="chart__tip" data-paired="tip" aria-hidden="true" hidden></div>
              </div>
              <ul class="legend" role="list">
                <li><span class="legend__dot legend__dot--hollow" aria-hidden="true"></span>trained on shuffled frames</li>
                <li><span class="legend__dot" aria-hidden="true"></span>trained with stretches held back</li>
                <li><span class="legend__line" aria-hidden="true"></span>always steering straight</li>
              </ul>
              <figcaption class="chart__caption" data-paired="caption">Each line is one training seed: the network trained on that seed’s shuffled split and on its blocked split, both scored on the frames both splits held out.</figcaption>
              <details class="numbers" data-paired="numbers">
                <summary>All the numbers in this chart</summary>
                <div class="numbers__scroll"><table class="numbers__table" data-paired="table"><caption>Each seed’s pair, on the frames both of its splits held out</caption></table></div>
              </details>
            </figure>
          </div>`;
}

/** The slope chart before the data has loaded: its frame, in quiet grey. */
const SLOPE_PLACEHOLDER = `<svg class="chart__svg slope slope--waiting" viewBox="0 0 600 340" preserveAspectRatio="none" aria-hidden="true" focusable="false">
                  <path class="placeholder__grid" d="M38 30H596M38 96H596M38 162H596M38 228H596M38 294H596"/>
                  <path class="placeholder__mark" d="M114 254 520 150M114 250 520 176M114 246 520 160"/>
                </svg>`;

// -- The ladder -------------------------------------------------------------------------

/** The rungs as the reports name them; the data's own names replace these
 * once it has loaded. */
const RUNG_NAMES = ["shuffled", "blocked", "later in the drive", "another day"];

const LADDER_PLACEHOLDER = `<svg class="chart__svg ladder-chart ladder-chart--waiting" viewBox="0 0 1000 300" preserveAspectRatio="none" aria-hidden="true" focusable="false">
                  <path class="placeholder__grid" d="M42 18H1000M42 84H1000M42 149H1000M42 215H1000M42 280H1000"/>
                  <path class="placeholder__mark" d="M101 149V48h26v101zM134 149V60h26v89zM351 149V95h26v54zM384 149v-34h26v34zM417 149v-46h26v46zM601 149v-44h26v44zM634 149v59h26v-59zM851 149v-41h26v41zM884 149v32h26v-32zM917 149v-4h26v4z"/>
                </svg>`;

export function ladderHtml(): string {
  const steps = RUNG_NAMES.map(
    (name, i) =>
      `<label class="stepper__step"><input class="stepper__input" type="radio" name="ladder-rung" value="${i}"${i === RUNG_NAMES.length - 1 ? " checked" : ""} /><span class="stepper__mark" aria-hidden="true"></span><span class="stepper__name">${name}</span></label>`,
  ).join("");
  return `<div class="study study--ladder ladder" data-ladder="root" data-state="loading">
            <div class="ladder__head">
              <h3 class="study__title" id="ladder-title">Further from the training frames</h3>
              <p class="study__lead" data-ladder="lead">Each rung tests on frames further from anything the network trained on. A bar above the line beats always steering straight on the same frames; a bar below it does worse.</p>
            </div>
            <figure class="chart chart--ladder" data-ladder="figure">
              <div class="chart__head">
                <p class="chart__title" data-ladder="chart-title">How much better than always steering straight</p>
                <fieldset class="segmented" data-ladder="views">
                  <legend class="visually-hidden">Show</legend>
                  <label class="segmented__option"><input class="segmented__input" type="radio" name="ladder-view" value="skill" checked />Better than steering straight</label>
                  <label class="segmented__option"><input class="segmented__input" type="radio" name="ladder-view" value="error" />Error in degrees</label>
                </fieldset>
              </div>
              <fieldset class="stepper" data-ladder="stepper">
                <legend class="visually-hidden">Rung of the ladder</legend>
                ${steps}
              </fieldset>
              <div class="ladder__caption" data-ladder="captions">
                <p class="ladder__said" id="ladder-caption" data-ladder="caption"><span class="ladder__rung" data-ladder="caption-rung">Another day</span> <span data-ladder="caption-text">A drive recorded on another day, none of it seen in training.</span></p>
              </div>
              <div class="chart__plot chart__plot--ladder" data-ladder="plot">
                ${LADDER_PLACEHOLDER}
                <div class="chart__tip" data-ladder="tip" aria-hidden="true" hidden></div>
              </div>
              <ul class="legend" role="list">
                <li><span class="legend__swatch legend__swatch--sightlinenet" aria-hidden="true"></span>SightlineNet</li>
                <li><span class="legend__swatch legend__swatch--pilotnet" aria-hidden="true"></span>PilotNet</li>
                <li><span class="legend__swatch legend__swatch--oneframe" aria-hidden="true"></span>${networkName("sightlinenet-1f").name}</li>
                <li><span class="legend__line" aria-hidden="true"></span>always steering straight</li>
              </ul>
              <figcaption class="chart__caption" data-ladder="chart-caption">Each bar is a mean over the network’s trainings; each whisker spans one standard deviation over them.</figcaption>
              <details class="numbers" data-ladder="numbers">
                <summary>All the numbers in this chart</summary>
                <div class="numbers__scroll"><table class="numbers__table" data-ladder="table"><caption>Each network on each rung, over its trainings</caption></table></div>
              </details>
            </figure>
          </div>`;
}

// -- The cards --------------------------------------------------------------------------

/** A card's chart before the data has loaded: quiet bars. */
const CARD_PLACEHOLDER = `<svg class="chart__svg card-chart card-chart--waiting" viewBox="0 0 360 150" preserveAspectRatio="none" aria-hidden="true" focusable="false">
                  <path class="placeholder__mark" d="M8 16h220v14H8zM8 36h120v14H8zM8 86h260v14H8zM8 106h200v14H8z"/>
                </svg>`;

/** The repository's files the numbers come from, linked where they live. */
const SOURCES: ReadonlyArray<readonly [path: string, what: string]> = [
  ["reports/summary.json", "the paired measurement, the ladder, the regimes and each network’s trainings"],
  ["reports/runs", "each training run’s own report"],
  ["reports/leakage.json", "the lookup table, and how close test frames sit to training frames"],
  ["reports/legacy.json", "the standard-recipe PilotNet"],
  ["reports/export.json", "the export check of the models the page runs"],
  ["src/sightline/data/splits.py", "how each split cuts the drive"],
];

const REPOSITORY = "https://github.com/safdar-hussain1/sightline";

/** A file's page in the repository; a folder's is its tree. */
function sourceUrl(path: string): string {
  return `${REPOSITORY}/${path.includes(".") ? "blob" : "tree"}/main/${path}`;
}

export function cardsHtml(): string {
  const sources = SOURCES.map(([path, what]) => `<li><a href="${sourceUrl(path)}">${path}</a> <span class="sources__what">${what}</span></li>`).join("");
  return `<div class="cards">
            <article class="card card--baselines" data-card="baselines" data-state="loading">
              <h3 class="card__title">Baselines, on the same frames</h3>
              <p class="card__lead" data-card="baselines-lead">Simple rules, scored on the very frames Sightline is tested on. Always steering straight, and always steering the training frames’ mean angle, never look at the road. Holding the last label and linear extrapolation read the answer stream, the driver’s own angle on the frames just before, which the model never sees.</p>
              <figure class="card__figure"><div class="chart__plot" data-card="baselines-plot">${CARD_PLACEHOLDER}</div></figure>
              <p class="card__verdict" data-card="baselines-verdict">The rules that read the answers are yardsticks here, not rivals.</p>
            </article>
            <article class="card card--frames" data-card="frames" data-state="loading">
              <h3 class="card__title" data-card="frames-title">What an earlier frame buys</h3>
              <p class="card__lead" data-card="frames-lead">SightlineNet sees the frame on screen and an earlier one. The same network trained on the frame on screen alone shows what the earlier frame is worth.</p>
              <figure class="card__figure"><div class="chart__plot" data-card="frames-plot">${CARD_PLACEHOLDER}</div></figure>
              <p class="card__verdict" data-card="frames-also">An earlier frame shows which way the view is moving.</p>
            </article>
            <article class="card card--regimes" data-card="regimes" data-state="loading">
              <h3 class="card__title">Where it struggles</h3>
              <p class="card__lead" data-card="regimes-lead">The test frames sorted by how hard the driver was turning: straight road, gentle curves and sharp ones.</p>
              <figure class="card__figure"><div class="chart__plot" data-card="regimes-plot">${CARD_PLACEHOLDER}</div></figure>
              <p class="card__verdict" data-card="regimes-straight">On straight road, always steering straight is hard to beat.</p>
            </article>
            <article class="card card--recipe" data-card="recipe" data-state="loading">
              <h3 class="card__title">The standard recipe</h3>
              <p class="card__lead">A PilotNet trained the widely copied way, scored here from its own weights. <span data-card="recipe-lead"></span> <span data-card="recipe-other"></span></p>
              <p class="card__note" data-card="recipe-recipe"></p>
              <figure class="card__figure"><div class="chart__plot" data-card="recipe-plot">${CARD_PLACEHOLDER}</div></figure>
              <p class="card__verdict" data-card="recipe-cap">Its arctangent output has a limit, and the drivers turned the wheel past it.</p>
            </article>
          </div>
          <div class="sources-card" data-card="trust" data-state="loading">
            <h3 class="study__title" id="sources-title">Where the numbers come from</h3>
            <div class="sources-card__grid">
              <div class="sources-card__part sources-card__part--checks">
                <h4 class="sources-card__heading">Checked at export, and live</h4>
                <p data-card="export">When each model was exported, onnxruntime was checked against PyTorch on every frame the page uses.</p>
                <p data-card="live" data-state="checking">This browser’s answers are checked live against the same model run in Python.</p>
              </div>
              <div class="sources-card__part sources-card__part--splits">
                <h4 class="sources-card__heading">How each split cuts the drive</h4>
                <dl class="protocols">
                  <div><dt>Shuffled</dt><dd data-card="protocol-shuffled">Every frame is dealt at random to training, validation or test.</dd></div>
                  <div><dt>Blocked</dt><dd data-card="protocol-blocked">The drive is cut into blocks, and frames are dropped either side of every boundary between a training block and a held-out one.</dd></div>
                  <div><dt>Later in the drive</dt><dd data-card="protocol-later">The drive’s first part trains, the next validates and the last part tests.</dd></div>
                  <div><dt>Another day</dt><dd data-card="protocol-another-day">Another drive, recorded on another day, scored by the networks trained with stretches held back.</dd></div>
                </dl>
              </div>
              <div class="sources-card__part sources-card__part--files">
                <h4 class="sources-card__heading">The files</h4>
                <ul class="sources" role="list"><li><a href="${REPOSITORY}#readme">README</a> <span class="sources__what">what the repository holds, and how to run it</span></li>${sources}</ul>
                <p class="sources-card__note">Every number in this view is read from <a href="data/results.json">results.json</a>, which the repository generates from those reports. The <a href="model-card.md">model card</a> says how the shipped model was trained, what it is for and where it falls short.</p>
              </div>
            </div>
          </div>`;
}

// -- Writing the sections in ------------------------------------------------------

const SECTIONS: ReadonlyArray<readonly [root: string, html: () => string, name: string]> = [
  [HOW_ROOT, howHtml, "How it works"],
  [ACCURACY_ROOT, accuracyHtml, "accuracy"],
  [RESULTS_ROOT, resultsHtml, "results"],
];

/** `root` (an empty element) with `inner` written into it. */
function filled(root: string, inner: string): string {
  const close = root.lastIndexOf("></");
  return `${root.slice(0, close + 1)}${inner}${root.slice(close + 1)}`;
}

/** index.html with every section written into its empty element. Throws
 * unless each has exactly one, so a renamed or doubled element fails the
 * build instead of shipping a page without it. */
export function injectSections(html: string): string {
  let out = html;
  for (const [root, inner, name] of SECTIONS) {
    const parts = out.split(root);
    if (parts.length !== 2) {
      throw new Error(`index.html must contain exactly one empty ${name} section: ${root}`);
    }
    out = parts.join(filled(root, inner()));
  }
  return out;
}

/** Write each section in if the page arrived without it (served from the
 * template rather than the build). The markup is this module's own constant
 * text, with nothing from outside interpolated. */
export function mountSectionsMarkup(): void {
  for (const [root, inner] of SECTIONS) {
    const id = /id="([^"]+)"/.exec(root)?.[1];
    const element = id ? document.getElementById(id) : null;
    if (element && element.childElementCount === 0) element.innerHTML = inner();
  }
}
