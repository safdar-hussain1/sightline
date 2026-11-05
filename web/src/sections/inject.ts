/**
 * Writes every section into index.html at build time (vite.config.ts calls
 * `injectSections`), from the same functions the page uses, so the served page
 * carries each section's words and layout before any script runs.
 */

import { ACCURACY_ROOT, HOW_ROOT, RESULTS_ROOT, accuracyHtml, filled, howHtml } from "./markup.ts";
import { resultsHtml } from "./results-markup.ts";

const SECTIONS: ReadonlyArray<readonly [root: string, html: () => string, name: string]> = [
  [HOW_ROOT, howHtml, "How it works"],
  [ACCURACY_ROOT, accuracyHtml, "accuracy"],
  [RESULTS_ROOT, resultsHtml, "results"],
];

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
