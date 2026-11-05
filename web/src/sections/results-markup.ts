/**
 * The markup of the Results view before any data has loaded.
 *
 * Plain HTML with no behaviour, so the build writes it into index.html (see
 * inject.ts): the first paint of the view has its final layout, and a crawler
 * reads its words. Nothing here is a measurement: where a sentence will carry
 * numbers, the page is served with a plain version of it, and the view writes
 * the numbers in once the data has loaded. The page loads this module with the
 * view, never with the demo.
 */

import { RESULTS_ROOT, mountMarkup } from "./markup.ts";
import { networkName } from "./names.ts";

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
          ${pairsHtml()}
          ${timelineHtml()}
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

// -- The frame next door ---------------------------------------------------------------

/** A photo's frame before the data has loaded: the road drawn in quiet grey. */
const PHOTO_PLACEHOLDER = `<svg class="pair-photo__waiting" viewBox="0 0 160 90" preserveAspectRatio="none" aria-hidden="true" focusable="false"><path class="placeholder__mark" d="M0 52h160v38H0z"/><path class="placeholder__grid" d="M80 52 58 90M80 52l22 38"/></svg>`;

function pairPanel(split: "test" | "shuffled" | "blocked", title: string): string {
  return `<div class="pair-panel pair-panel--${split}" data-pairs="panel-${split}">
                <p class="pair-panel__title">${title}</p>
                <div class="pair-photo" data-pairs="photo-${split}">${PHOTO_PLACEHOLDER}</div>
                <ul class="pair-panel__facts" data-pairs="facts-${split}" role="list"></ul>
              </div>`;
}

export function pairsHtml(): string {
  return `<div class="study study--pairs pairs" data-pairs="root" data-state="loading">
            <div class="pairs__head">
              <h3 class="study__title" id="pairs-title">The frame next door</h3>
              <p class="study__lead" data-pairs="lead">Test frames, each beside the training frame that looks most like it under each split.</p>
            </div>
            <figure class="pairs__figure" data-pairs="figure">
              <fieldset class="pairs__picker" data-pairs="picker">
                <legend class="pairs__legend">Pick a test frame</legend>
              </fieldset>
              <div class="pairs__panels" data-pairs="panels">
              <p class="visually-hidden" data-pairs="alt"></p>
              ${pairPanel("test", "The test frame")}
              ${pairPanel("shuffled", "Shuffled split: its closest training frame")}
              ${pairPanel("blocked", "Blocked split: its closest training frame")}
              </div>
              <div class="compare" data-pairs="compare">
                <div class="compare__frame" data-pairs="compare-frame">
                  <div class="compare__under" data-pairs="compare-under">${PHOTO_PLACEHOLDER}</div>
                  <div class="compare__over" data-pairs="compare-over"></div>
                  <span class="compare__handle" data-pairs="compare-handle" aria-hidden="true"></span>
                  <span class="compare__tag compare__tag--left" aria-hidden="true">Test frame</span>
                  <span class="compare__tag compare__tag--right" aria-hidden="true">Shuffled neighbour</span>
                  <input class="compare__range" type="range" min="0" max="100" value="50" step="1" data-pairs="compare-range" aria-label="Slide between the test frame and its closest training frame under the shuffled split" />
                </div>
                <div class="compare__words">
                  <p class="compare__title">Slide between them</p>
                  <p class="compare__text" data-pairs="compare-text">The test frame and the shuffled split’s closest training frame, one over the other.</p>
                  <p class="compare__note" data-pairs="compare-note"></p>
                </div>
              </div>
              <details class="numbers pairs__details" data-pairs="numbers">
                <summary>All the numbers in these pairs</summary>
                <div class="numbers__scroll"><table class="numbers__table" data-pairs="table"><caption>Each test frame and its closest training frame under each split</caption></table></div>
              </details>
            </figure>
          </div>`;
}

// -- How the drive was split ------------------------------------------------------------

const TIMELINE_PLACEHOLDER = `<svg class="chart__svg timeline-chart timeline-chart--waiting" viewBox="0 0 1000 150" preserveAspectRatio="none" aria-hidden="true" focusable="false">
                  <path class="placeholder__grid" d="M0 58H1000M0 124H1000"/>
                  <path class="placeholder__mark" d="M0 70h1000v36H0z"/>
                </svg>`;

export function timelineHtml(): string {
  const role = (key: string, name: string) =>
    `<li class="timeline__role"><span class="legend__swatch role--${key}" aria-hidden="true"></span>${name} <span class="timeline__count" data-timeline="count-${key}"></span></li>`;
  return `<div class="study study--timeline timeline" data-timeline="root" data-state="loading">
            <div class="timeline__head">
              <h3 class="study__title" id="timeline-title">How the drive was split</h3>
              <p class="study__lead" data-timeline="lead">The whole drive, from its first frame to its last, and how each split deals its frames to training, validation and testing.</p>
            </div>
            <figure class="chart chart--timeline" data-timeline="figure">
              <div class="chart__head">
                <p class="chart__title">The drive in time, and what each split does with it</p>
                <fieldset class="segmented" data-timeline="protocols">
                  <legend class="visually-hidden">Split</legend>
                  <label class="segmented__option"><input class="segmented__input" type="radio" name="timeline-split" value="shuffled" />Shuffled</label>
                  <label class="segmented__option"><input class="segmented__input" type="radio" name="timeline-split" value="blocked" checked />Blocked</label>
                  <label class="segmented__option"><input class="segmented__input" type="radio" name="timeline-split" value="later" />Later in the drive</label>
                </fieldset>
              </div>
              <ul class="timeline__trace-key" role="list">
                <li><span class="timeline__trace-swatch timeline__trace-swatch--turn" aria-hidden="true"></span>How hard the driver was turning</li>
                <li><span class="timeline__trace-swatch timeline__trace-swatch--way" aria-hidden="true"></span>Which way, on average</li>
                <li>Square-root scale, so gentle driving still shows</li>
              </ul>
              <div class="chart__plot chart__plot--timeline" data-timeline="plot">
                ${TIMELINE_PLACEHOLDER}
                <div class="chart__tip" data-timeline="tip" aria-hidden="true" hidden></div>
              </div>
              <ul class="legend timeline__key" role="list">
                ${role("train", "Train")}
                ${role("val", "Validate")}
                ${role("test", "Test")}
                ${role("dropped", "Dropped")}
              </ul>
              <figcaption class="chart__caption" data-timeline="note">Each split deals the drive’s frames to training, validation and testing; the blocked and later-in-the-drive splits drop the frames either side of a boundary between training and held-out road.</figcaption>
              <p class="timeline__clip" data-timeline="clip"></p>
              <details class="numbers" data-timeline="numbers">
                <summary>All the numbers in this chart</summary>
                <div class="numbers__scroll"><table class="numbers__table" data-timeline="table"><caption>The frames each split deals to each part</caption></table></div>
              </details>
            </figure>
          </div>`;
}

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
              <figure class="card__figure"><p class="chart__title card__chart-title">Mean error, in degrees of steering-wheel angle</p><div class="chart__plot" data-card="baselines-plot">${CARD_PLACEHOLDER}</div></figure>
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

/** The Results view, written in if the page arrived without it. */
export function mountResultsMarkup(): void {
  mountMarkup(RESULTS_ROOT, resultsHtml);
}
