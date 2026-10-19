/**
 * The cockpit's markup, before anything has loaded.
 *
 * Plain HTML with no behaviour, so the build writes it into index.html (see
 * `injectCockpit`, called from vite.config.ts) and the first paint already has
 * the cockpit's final layout: the windshield, the wheel, the readouts and the
 * controls, waiting, with nothing below them moving when the demo starts.
 * `cockpit.ts` fills it in; it renders this markup itself only if the page
 * arrived without it. Nothing here is a measurement: every number appears
 * once the data has loaded.
 */

import { NO_CLIP_TEXT } from "./strip.ts";
import { wheelSvg } from "./wheel.ts";

export const COCKPIT_ID = "cockpit";

/** The empty element index.html holds for the cockpit. */
export const COCKPIT_ROOT = `<section class="cockpit" id="${COCKPIT_ID}" aria-label="Live demo"></section>`;

const PLAY_ICON = `<svg class="play__icon play__icon--play" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 5.5v13l10.5-6.5z"/></svg>`;
const PAUSE_ICON = `<svg class="play__icon play__icon--pause" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/></svg>`;
/* The readouts' keys repeat the wheel's own marks: the model's violet pointer
   reaching up from its rim, the driver's pale one reaching down from outside,
   and the arc between them. */
const KEY_MODEL = `<svg class="key key--model" viewBox="0 0 14 12" aria-hidden="true" focusable="false"><path d="M7 1.5 12.5 10.5h-11z"/></svg>`;
const KEY_DRIVER = `<svg class="key key--driver" viewBox="0 0 14 12" aria-hidden="true" focusable="false"><path d="M7 10.5 1.5 1.5h11z"/></svg>`;
const KEY_GAP = `<svg class="key key--gap" viewBox="0 0 14 12" aria-hidden="true" focusable="false"><path d="M1.75 9a7 7 0 0 1 10.5 0"/></svg>`;
const BACK_ICON = `<svg class="step__icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M14.5 6.5 9 12l5.5 5.5"/></svg>`;
const FORWARD_ICON = `<svg class="step__icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m9.5 6.5 5.5 5.5-5.5 5.5"/></svg>`;

/** The cockpit's contents, written into `COCKPIT_ROOT`. */
export function cockpitHtml(): string {
  return `
        <div class="cockpit__console wrap" data-state="loading">
          <div class="cockpit__choices">
            <fieldset class="picker picker--clips" data-cockpit="clips" disabled>
              <legend class="visually-hidden">Footage</legend>
            </fieldset>
            <fieldset class="picker picker--models" data-cockpit="models" disabled>
              <legend class="visually-hidden">Model</legend>
              <label class="picker__option"><input class="picker__input" type="radio" name="cockpit-model" value="sightline" checked /><span class="picker__lead">Sightline</span><span class="picker__sub">blocked split</span></label>
              <label class="picker__option"><input class="picker__input" type="radio" name="cockpit-model" value="shuffled" /><span class="picker__lead">Its twin</span><span class="picker__sub">shuffled split</span></label>
            </fieldset>
          </div>

          <figure class="windshield" data-cockpit="windshield">
            <div class="windshield__camera">
              <video class="windshield__video" data-cockpit="video" muted playsinline loop preload="none" disablepictureinpicture aria-hidden="true" tabindex="-1"></video>
              <div class="windshield__band" data-cockpit="band" aria-hidden="true">
                <canvas class="windshield__heat" data-cockpit="heat" width="200" height="66"></canvas>
                <span class="windshield__tag">What the model sees</span>
              </div>
            </div>
            <p class="windshield__status" data-cockpit="status" role="status">Loading the model and the footage</p>
          </figure>

          <div class="transport">
            <button class="play" data-cockpit="play" type="button" disabled>${PLAY_ICON}${PAUSE_ICON}<span class="play__label" data-cockpit="play-label">Play</span></button>
            <button class="step" data-cockpit="back" type="button" aria-label="Back one frame" disabled>${BACK_ICON}</button>
            <button class="step" data-cockpit="forward" type="button" aria-label="Forward one frame" disabled>${FORWARD_ICON}</button>
            <div class="strip" data-cockpit="strip" role="slider" tabindex="-1" aria-label="Position in the clip" aria-disabled="true" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0" aria-valuetext="${NO_CLIP_TEXT}">
              <canvas class="strip__canvas" data-cockpit="strip-canvas" aria-hidden="true"></canvas>
            </div>
          </div>

          <div class="strip-key">
            <p class="strip-key__title">How far off, frame by frame</p>
            <ul class="strip-key__regimes" aria-label="Column colour: how hard the driver was turning">
              <li><span class="swatch swatch--straight" aria-hidden="true"></span>straight</li>
              <li><span class="swatch swatch--gentle" aria-hidden="true"></span>gentle curve</li>
              <li><span class="swatch swatch--sharp" aria-hidden="true"></span>sharp curve</li>
            </ul>
            <p class="strip-key__average"><span class="swatch swatch--average" aria-hidden="true"></span>This clip, on average: <span class="tnum" data-cockpit="average">—</span></p>
            <p class="cockpit__notice" data-cockpit="notice" role="status" hidden></p>
          </div>

          <div class="cockpit__side">
            <p class="model-note" data-cockpit="note"></p>

            <div class="instruments">
              <div class="wheel-box" data-cockpit="wheel">${wheelSvg()}</div>
              <dl class="readouts">
                <div class="readout readout--model"><dt>${KEY_MODEL}<span data-cockpit="model-name">Sightline</span></dt><dd><span class="readout__size tnum" data-cockpit="model-angle">—</span> <span class="readout__side" data-cockpit="model-side"></span></dd></div>
                <div class="readout readout--driver"><dt>${KEY_DRIVER}Driver</dt><dd><span class="readout__size tnum" data-cockpit="driver-angle">—</span> <span class="readout__side" data-cockpit="driver-side"></span></dd></div>
                <div class="readout readout--gap"><dt>${KEY_GAP}Off by</dt><dd><span class="readout__size tnum" data-cockpit="gap">—</span></dd></div>
              </dl>
            </div>

            <figure class="eye">
              <canvas class="eye__canvas" data-cockpit="eye" width="200" height="66" aria-hidden="true"></canvas>
              <figcaption class="eye__caption" data-cockpit="eye-caption">The model’s own input, with where it looked</figcaption>
            </figure>
          </div>

          <noscript><p class="cockpit__noscript">The live demo needs JavaScript.</p></noscript>
        </div>
      `;
}

/** index.html with the cockpit written into its empty section. Throws unless
 * there is exactly one, so a renamed or duplicated element fails the build. */
export function injectCockpit(html: string): string {
  const parts = html.split(COCKPIT_ROOT);
  if (parts.length !== 2) {
    throw new Error(`index.html must contain exactly one empty cockpit section: ${COCKPIT_ROOT}`);
  }
  return parts.join(COCKPIT_ROOT.replace("></section>", `>${cockpitHtml()}</section>`));
}

/** Render the cockpit if the page arrived without it (served from the
 * template rather than the build). The markup is this module's own constant
 * text, with nothing from outside interpolated. */
export function mountCockpitMarkup(): HTMLElement | null {
  const root = document.getElementById(COCKPIT_ID);
  if (root && !root.querySelector(".cockpit__console")) {
    root.innerHTML = cockpitHtml();
  }
  return root;
}
