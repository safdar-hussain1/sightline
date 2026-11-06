/**
 * The markup of "Try it on your own video", before anything has loaded.
 *
 * Plain HTML with no behaviour, so the build writes it into index.html (see
 * inject.ts) and the first paint has the section's layout and words. Nothing
 * here is a measurement: the rules are served in plain words, without their
 * numbers, and ownclip.ts writes the numbers in once the data has loaded. The
 * drawing of where the model reads a landscape and a portrait frame is drawn
 * to a nominal shape here and redrawn to the clips' own camera once it is
 * known.
 */

import { wheelSvg } from "../cockpit/wheel.ts";
import { readBand, type Rect } from "./ownclip-plan.ts";

/** The empty element index.html holds for the section. */
export const OWNCLIP_ROOT = `<section class="ownclip" id="own-video" aria-labelledby="ownclip-title"></section>`;

/** The id of the section's file input, which its labels name. */
export const FILE_INPUT_ID = "ownclip-file";

// -- The drawing of the rule ------------------------------------------------------------

/** The two frames of the drawing, in its viewBox: a landscape one and a
 * portrait one, each [x, y, width, height]. */
export const ART_FRAMES = {
  landscape: [10, 22, 176, 99],
  portrait: [226, 6, 72, 128],
} as const;

/** The camera the served drawing uses until the clips' own is known: a
 * nominal dashcam shape (width over height), and a nominal share of its height
 * read. */
const NOMINAL: Camera = { aspect: 16 / 9, share: 0.6 };

/** A camera's shape and the share of its height the model reads. */
export interface Camera {
  readonly aspect: number;
  readonly share: number;
}

type Frame = readonly [number, number, number, number];

const r2 = (v: number) => Math.round(v * 100) / 100;

/** The read part of `frame` for `camera`, in the drawing's own coordinates. */
export function artBand(frame: Frame, camera: Camera): Rect {
  const [x, y, w, h] = frame;
  const band = readBand(w, h, camera.aspect, camera.share);
  return { sx: r2(x + band.sx), sy: r2(y + band.sy), sw: r2(band.sw), sh: r2(band.sh) };
}

/** The band across a portrait frame's middle with the camera's shape. */
export function artMiddle(frame: Frame, camera: Camera): Rect {
  const [x, y, w, h] = frame;
  const band = Math.min(h, w / camera.aspect);
  return { sx: x, sy: r2(y + (h - band) / 2), sw: w, sh: r2(band) };
}

const rectAttrs = (r: Rect) => `x="${r.sx}" y="${r.sy}" width="${r.sw}" height="${r.sh}"`;

/** A road seen through the windscreen, drawn inside `frame`, its horizon at
 * `vanish` (a share of the frame's height). */
function road([x, y, w, h]: Frame, vanish: number): string {
  const vx = x + w / 2;
  const vy = y + h * vanish;
  const foot = y + h;
  const near = (share: number) => r2(x + w * share);
  return (
    `<path class="art__ground" d="M${x} ${r2(vy)}H${x + w}V${foot}H${x}Z"/>` +
    `<path class="art__road" d="M${r2(vx - w * 0.03)} ${r2(vy)}H${r2(vx + w * 0.03)}L${near(0.9)} ${foot}H${near(0.1)}Z"/>` +
    `<path class="art__edge" d="M${r2(vx - w * 0.03)} ${r2(vy)}L${near(0.1)} ${foot}M${r2(vx + w * 0.03)} ${r2(vy)}L${near(0.9)} ${foot}"/>` +
    `<path class="art__lane" d="M${r2(vx)} ${r2(vy + 1)}V${foot}"/>` +
    `<path class="art__horizon" d="M${x} ${r2(vy)}H${x + w}"/>`
  );
}

function artSvg(camera: Camera): string {
  const land = ART_FRAMES.landscape;
  const port = ART_FRAMES.portrait;
  const frame = ([x, y, w, h]: Frame) => `x="${x}" y="${y}" width="${w}" height="${h}"`;
  const middle = artMiddle(port, camera);
  // A phone held upright sees the horizon across the middle of its picture
  // and the road running on below it, to the foot of the frame.
  const horizon = (middle.sy + middle.sh * 0.42 - port[1]) / port[3];
  return `<svg class="art" viewBox="0 0 308 140" aria-hidden="true" focusable="false">
            <defs>
              <clipPath id="ownclip-art-land"><rect ${frame(land)} rx="6"/></clipPath>
              <clipPath id="ownclip-art-port"><rect ${frame(port)} rx="6"/></clipPath>
            </defs>
            <g clip-path="url(#ownclip-art-land)">
              <rect class="art__sky" ${frame(land)}/>
              ${road(land, 0.42)}
              <rect class="art__band" data-ownclip-art="land" ${rectAttrs(artBand(land, camera))}/>
            </g>
            <rect class="art__frame" ${frame(land)} rx="6"/>
            <g clip-path="url(#ownclip-art-port)">
              <rect class="art__sky" ${frame(port)}/>
              ${road(port, horizon)}
              <rect class="art__middle" data-ownclip-art="middle" ${rectAttrs(middle)}/>
              <rect class="art__band" data-ownclip-art="port" ${rectAttrs(artBand(port, camera))}/>
            </g>
            <rect class="art__frame" ${frame(port)} rx="6"/>
          </svg>`;
}

// -- Icons ------------------------------------------------------------------------------

const FILM_ICON = `<svg class="ownclip__icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M7.5 4.5v15M16.5 4.5v15M3.5 9h4M3.5 15h4M16.5 9h4M16.5 15h4"/></svg>`;
const LOCK_ICON = `<svg class="ownclip__icon ownclip__icon--lock" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="5" y="10.5" width="14" height="10" rx="2"/><path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5"/></svg>`;
const ALERT_ICON = `<svg class="ownclip__icon ownclip__icon--alert" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M9 9.5l6 6M15 9.5l-6 6"/></svg>`;
const PLAY_ICON = `<svg class="play__icon play__icon--play" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 5.5v13l10.5-6.5z"/></svg>`;
const PAUSE_ICON = `<svg class="play__icon play__icon--pause" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/></svg>`;
const BACK_ICON = `<svg class="step__icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M14.5 6.5 9 12l5.5 5.5"/></svg>`;
const FORWARD_ICON = `<svg class="step__icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m9.5 6.5 5.5 5.5-5.5 5.5"/></svg>`;

// -- The section ---------------------------------------------------------------------------

export function ownclipHtml(): string {
  return `
        <div class="ownclip__inner wrap">
          <div class="ownclip__head">
            <h2 class="ownclip__title" id="ownclip-title">Try it on your own video</h2>
            <p class="ownclip__lede">Choose a video filmed from a car, facing the road, and the same model estimates a steering angle for it, frame by frame, here in this browser.</p>
          </div>

          <div class="ownclip__console" data-ownclip="console" data-phase="idle">
            <div class="ownclip__bar">
              <p class="ood"><span class="ood__badge">Out of distribution</span> <span class="ood__text">Your video almost certainly shows a car, camera and road the model never trained on, and it carries no record of how the wheel was turned. The angle is the model’s estimate, not a measurement of anything.</span></p>
              <p class="ownclip__private">${LOCK_ICON}<span>Your video stays on your device: it is read here, in this browser, and never uploaded.</span></p>
            </div>

            <input class="ownclip__file" id="${FILE_INPUT_ID}" data-ownclip="file" type="file" accept="video/*" aria-label="Choose a video" />

            <div class="drop" data-ownclip="drop">
              <figure class="drop__figure">
                <div class="drop__art">${artSvg(NOMINAL)}</div>
                <figcaption class="drop__key"><span class="drop__swatch" aria-hidden="true"></span>The violet band is the part of each frame the model reads, from a landscape video and from a portrait one.</figcaption>
              </figure>
              <div class="drop__act">
                <p class="drop__title">Drop a video here, or</p>
                <label class="ownclip__choose" for="${FILE_INPUT_ID}">${FILM_ICON}<span>Choose a video</span></label>
                <p class="drop__hint">Footage through a windscreen, facing the road, is what it knows.</p>
              </div>
            </div>

            <div class="ownclip__player" data-ownclip="player" hidden>
              <div class="ownclip__screen" data-ownclip="screen">
                <div class="ownclip__frame" data-ownclip="frame">
                  <video class="ownclip__video" data-ownclip="video" muted playsinline preload="auto" disablepictureinpicture aria-hidden="true" tabindex="-1"></video>
                  <canvas class="ownclip__still" data-ownclip="still" aria-hidden="true"></canvas>
                  <div class="ownclip__band" data-ownclip="band" aria-hidden="true">
                    <canvas class="ownclip__heat" data-ownclip="heat" width="200" height="66"></canvas>
                    <span class="ownclip__tag">What the model sees</span>
                  </div>
                </div>
                <p class="ownclip__status" data-ownclip="status"></p>
              </div>

              <div class="ownclip__side">
                <div class="ownclip__instruments">
                  <div class="ownclip__wheel" data-ownclip="wheel" data-waiting="true">${wheelSvg()}</div>
                  <div class="ownclip__readout">
                    <p class="ownclip__readout-label"><svg class="key key--model" viewBox="0 0 14 12" aria-hidden="true" focusable="false"><path d="M7 1.5 12.5 10.5h-11z"/></svg>The model’s estimate</p>
                    <p class="ownclip__angle"><span class="readout__size tnum" data-ownclip="angle">—</span> <span class="readout__side" data-ownclip="side"></span></p>
                    <p class="ownclip__when tnum" data-ownclip="when"></p>
                  </div>
                </div>
                <figure class="eye ownclip__eye">
                  <canvas class="eye__canvas" data-ownclip="eye" width="200" height="66" aria-hidden="true"></canvas>
                  <figcaption class="eye__caption" data-ownclip="eye-caption">What the model gets from this frame, with where it looked.</figcaption>
                </figure>
              </div>

              <div class="ownclip__transport">
                <button class="play" data-ownclip="play" type="button" disabled>${PLAY_ICON}${PAUSE_ICON}<span class="play__label" data-ownclip="play-label">Play</span></button>
                <button class="step" data-ownclip="back" type="button" aria-label="Back one frame" disabled>${BACK_ICON}</button>
                <button class="step" data-ownclip="forward" type="button" aria-label="Forward one frame" disabled>${FORWARD_ICON}</button>
                <div class="trace" data-ownclip="trace">
                  <canvas class="trace__canvas" data-ownclip="trace-canvas" aria-hidden="true"></canvas>
                  <span class="trace__side trace__side--right" aria-hidden="true">right</span>
                  <span class="trace__side trace__side--left" aria-hidden="true">left</span>
                  <input class="trace__scrub" data-ownclip="scrub" type="range" min="0" max="0" step="1" value="0" aria-label="Position in your video" disabled />
                </div>
              </div>

              <div class="ownclip__foot">
                <p class="ownclip__note" data-ownclip="note"></p>
                <button class="ownclip__action" data-ownclip="cancel" type="button" hidden>Stop reading</button>
                <button class="ownclip__action" data-ownclip="again" type="button" hidden>${FILM_ICON}<span>Choose another video</span></button>
              </div>
            </div>

            <div class="ownclip__error" data-ownclip="error" hidden>
              ${ALERT_ICON}
              <div class="ownclip__error-words" role="alert">
                <p class="ownclip__error-title" data-ownclip="error-title"></p>
                <p class="ownclip__error-text" data-ownclip="error-text"></p>
              </div>
              <button class="ownclip__choose" data-ownclip="retry" type="button">${FILM_ICON}<span>Choose another video</span></button>
            </div>

            <p class="visually-hidden" data-ownclip="live" role="status"></p>
            <noscript><p class="ownclip__noscript">Reading a video needs JavaScript.</p></noscript>
          </div>

          <div class="ownclip__notes">
            <h3 class="ownclip__notes-title">How it reads your video</h3>
            <ul class="ownclip__rules">
              <li data-ownclip="rule-crop">It reads the bottom of each frame, at full width, the share of the picture it was trained on. From a portrait video it first takes the band across the middle with the shape of the camera it learned from, then the bottom of that.</li>
              <li data-ownclip="rule-pace">It steps through your video so that the frames each answer compares are as far apart in time as the ones it learned from. Its first answer comes once it has a frame from just before.</li>
              <li data-ownclip="rule-cap">A long video is read from its start, up to a limit.</li>
            </ul>
          </div>
        </div>
      `;
}
