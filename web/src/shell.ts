/**
 * The masthead: the mark, the page's one <h1>, the lede and the theme toggle.
 *
 * Plain HTML with no behaviour, which is what lets the build write it into
 * index.html (see `injectShell`, called from vite.config.ts): the served page
 * carries its heading and its explanation before any script runs, so a crawler
 * that does not run scripts still reads them. The app uses the same function,
 * so the two can never drift apart.
 */

import { THEME_TOGGLE_ID } from "./theme.ts";

export const MASTHEAD_ID = "masthead";

/** The empty element index.html holds for the shell. */
export const SHELL_ROOT = `<header class="masthead" id="${MASTHEAD_ID}"></header>`;

/** A steering wheel turned a little to the left, with the centre marker a rally
 * wheel carries at twelve o'clock: the marker is where the wheel points. The
 * favicon is the same drawing. */
const MARK = `<svg class="mark" viewBox="0 0 32 32" aria-hidden="true" focusable="false">
            <rect class="mark__tile" width="32" height="32" rx="8" />
            <g transform="rotate(-14 16 16)" fill="none" stroke-linecap="round">
              <circle class="mark__wheel" cx="16" cy="16" r="10.5" stroke-width="3" />
              <path class="mark__wheel" d="M5.8 16.6h7.4M18.8 16.6h7.4M16 19.6v6.6" stroke-width="3" />
              <circle class="mark__wheel" cx="16" cy="16.8" r="2.9" stroke-width="2.4" />
              <path class="mark__marker" d="M14.2 5.66a10.5 10.5 0 0 1 3.6 0" stroke-width="3.4" />
            </g>
          </svg>`;

const MOON = `<svg class="theme-toggle__icon theme-toggle__icon--moon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M20.4 14.3A8.4 8.4 0 0 1 9.7 3.6a8.4 8.4 0 1 0 10.7 10.7Z" /></svg>`;

const SUN = `<svg class="theme-toggle__icon theme-toggle__icon--sun" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="12" cy="12" r="4.2" /><path d="M12 2.6v2.2M12 19.2v2.2M2.6 12h2.2M19.2 12h2.2M5.35 5.35l1.55 1.55M17.1 17.1l1.55 1.55M5.35 18.65l1.55-1.55M17.1 6.9l1.55-1.55" /></svg>`;

/** The masthead's contents, written into `SHELL_ROOT`. */
export function shellHtml(): string {
  return `
      <div class="masthead__inner wrap">
        <div class="wordmark">
          ${MARK}
          <h1 class="wordmark__name">Sightline</h1>
        </div>
        <p class="lede">A steering model you can watch drive. It runs in your browser on recorded road footage, turns its wheel beside the real driver’s, and shows where on the road it looks.</p>
        <button class="theme-toggle" id="${THEME_TOGGLE_ID}" type="button" aria-label="Switch between the light and dark themes">${MOON}${SUN}</button>
      </div>
    `;
}

/** index.html with the shell written into its empty masthead. Throws unless
 * there is exactly one, so a renamed or duplicated element fails the build
 * instead of shipping a page without its heading. */
export function injectShell(html: string): string {
  const parts = html.split(SHELL_ROOT);
  if (parts.length !== 2) {
    throw new Error(`index.html must contain exactly one empty masthead: ${SHELL_ROOT}`);
  }
  return parts.join(SHELL_ROOT.replace("></header>", `>${shellHtml()}</header>`));
}

/** Render the shell if the page arrived without it (served from the template
 * rather than the build). On the built page it is already there. The markup is
 * this module's own constant text, with nothing from outside interpolated. */
export function mountShell(): void {
  const masthead = document.getElementById(MASTHEAD_ID);
  if (masthead && !masthead.querySelector("h1")) {
    masthead.innerHTML = shellHtml();
  }
}
