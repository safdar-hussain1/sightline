/**
 * The page's entry: the stylesheet, the masthead, the theme, the demo view and
 * its sections, the cockpit, and the Results view when it is wanted.
 *
 * The masthead and every section's markup are already in the served HTML (the
 * build writes them in); `mountShell` and `mountDemoMarkup` only fill them if
 * the page arrived without them. results.json is asked for once, here: the
 * accuracy figure reads its numbers from it, and the cockpit and the Results
 * view start on the same request. The cockpit's code, the model runtime and
 * everything they download come in a chunk of their own, so the page is drawn
 * before them; if that chunk cannot be had, `bootCockpit` says the demo cannot
 * run, and the sections that read the cockpit keep their drawings. The
 * Results view's code is a chunk of its own too (results.ts), loaded the first
 * time the view is shown, or ahead of that once the demo has loaded and the
 * browser is idle.
 */

import "./styles.css";

import { bootCockpit } from "./cockpit/boot.ts";
import { loadResults } from "./data/load.ts";
import { driveGaps } from "./data/timing.ts";
import type { ResultsView } from "./results.ts";
import { Router, type ViewName } from "./router.ts";
import { Accuracy } from "./sections/accuracy.ts";
import { HowItWorks } from "./sections/how.ts";
import { mountDemoMarkup } from "./sections/markup.ts";
import { readFigures } from "./sections/reader.ts";
import { mountShell } from "./shell.ts";
import { markResultsUnavailable } from "./sections/unavailable.ts";
import { initTheme } from "./theme.ts";

console.info("Sightline — built by Safdar Hussain · https://github.com/safdar-hussain1/sightline");

mountShell();
mountDemoMarkup();
initTheme();

const how = HowItWorks.mount();
const accuracy = Accuracy.mount();

const results = loadResults(new URL("./", document.baseURI).href);
const cockpit = bootCockpit(() => import("./cockpit/cockpit.ts"), results);

/** The Results view's code, asked for once. */
let resultsCode: Promise<typeof import("./results.ts")> | null = null;
const loadResultsCode = () => (resultsCode ??= import("./results.ts"));
let resultsView: Promise<ResultsView> | null = null;

/** Start the Results view the first time it is shown; after that, tell it
 * which view is on screen. */
function resultsShown(view: ViewName): void {
  if (view === "results" && resultsView === null) {
    resultsView = loadResultsCode().then(
      (code) => code.startResults(results, cockpit, view),
      (error: unknown) => {
        console.warn("The Results view's code could not be loaded:", error);
        markResultsUnavailable();
        return { viewShown: () => undefined };
      },
    );
    return;
  }
  void resultsView?.then((started) => started.viewShown(view));
}

new Router({
  focus: { demo: "#cockpit", results: "#results-title" },
  onShow: (view) => {
    how?.viewShown(view);
    resultsShown(view);
  },
}).start();

results.then(
  (loaded) => {
    how?.setGaps(driveGaps(loaded.raw));
    try {
      accuracy?.show(readFigures(loaded.raw));
    } catch (error) {
      console.warn("The measurements could not be read:", error);
      accuracy?.fail(error);
    }
  },
  // The cockpit reports the same failure for itself, with its detail.
  (error: unknown) => accuracy?.fail(error),
);

void cockpit.then((live) => how?.attach(live));

// Fetch the Results view's code ahead of a visit, once the demo has loaded
// and the browser has nothing better to do.
addEventListener("load", () => {
  const idle = window.requestIdleCallback ?? ((callback: () => void) => window.setTimeout(callback, 1500));
  idle(() => void loadResultsCode().catch(() => undefined), { timeout: 4000 });
});

// The self-test runs only when the address asks for it; without ?selftest=1 its
// code is never fetched.
if (new URLSearchParams(location.search).get("selftest") === "1") void import("./selftest.ts");
