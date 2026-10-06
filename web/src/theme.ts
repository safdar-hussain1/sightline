/**
 * Light and dark themes: the system's by default, overridable, remembered.
 *
 * The page follows `prefers-color-scheme` until the visitor presses the toggle.
 * From then on their choice is set as `data-theme` on <html> and kept in
 * localStorage. A stored choice is first applied by a short inline script in
 * index.html, before the first paint: applying it from this module, after the
 * bundle loads, would flash the other theme on every visit. `initTheme` applies
 * it again, which costs nothing and covers a page served without that script.
 *
 * Storage can throw (private browsing, blocked site data), so every access is
 * wrapped: the toggle then still works for the visit, and the next visit starts
 * from the system theme again.
 */

export type Theme = "light" | "dark";

/** The localStorage key. index.html's inline script reads the same one. */
export const THEME_KEY = "sightline-theme";

/** The id of the toggle button in the masthead. */
export const THEME_TOGGLE_ID = "theme-toggle";

const DARK_QUERY = "(prefers-color-scheme: dark)";

function isTheme(value: unknown): value is Theme {
  return value === "light" || value === "dark";
}

/** The visitor's stored choice, or null if they never made one or storage is
 * unavailable. */
export function storedTheme(): Theme | null {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return isTheme(value) ? value : null;
  } catch {
    return null;
  }
}

function systemTheme(): Theme {
  return typeof matchMedia === "function" && matchMedia(DARK_QUERY).matches ? "dark" : "light";
}

/** The theme on screen: the visitor's choice on <html> if there is one, else
 * the system's. */
export function currentTheme(): Theme {
  const chosen = document.documentElement.getAttribute("data-theme");
  return isTheme(chosen) ? chosen : systemTheme();
}

/** The toggle's label names what pressing it does, which depends on the theme
 * on screen -- and that follows the system theme until the visitor chooses. */
function labelToggle(): void {
  const button = document.getElementById(THEME_TOGGLE_ID);
  if (button) {
    const other: Theme = currentTheme() === "dark" ? "light" : "dark";
    button.setAttribute("aria-label", `Switch to the ${other} theme`);
  }
}

/** Switch to the other theme and remember the choice. */
export function toggleTheme(): void {
  const next: Theme = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch {
    // Storage is unavailable: the choice holds for this visit only.
  }
  labelToggle();
}

/** Apply a stored choice, wire the masthead's toggle and keep its label true. */
export function initTheme(): void {
  const stored = storedTheme();
  if (stored) {
    document.documentElement.setAttribute("data-theme", stored);
  }
  document.getElementById(THEME_TOGGLE_ID)?.addEventListener("click", toggleTheme);
  if (typeof matchMedia === "function") {
    matchMedia(DARK_QUERY).addEventListener("change", labelToggle);
  }
  labelToggle();
}
