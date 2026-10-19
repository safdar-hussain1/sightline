/**
 * The page's entry: the stylesheet, the masthead, the theme, the two views
 * and their sections, and the cockpit.
 *
 * The masthead and every section's markup are already in the served HTML (the
 * build writes them in); `mountShell` and `mountSectionsMarkup` only fill them
 * if the page arrived without them. results.json is asked for once, here: the
 * accuracy figure and the Results view read their numbers from it, and the
 * cockpit starts on the same request. The cockpit's code, the model runtime
 * and everything they download come in a separate chunk, so the page is drawn
 * before them; if that chunk cannot be had, `bootCockpit` says the demo cannot
 * run, and the sections that read the cockpit keep their drawings.
 */

import "./styles.css";

import { bootCockpit } from "./cockpit/boot.ts";
import { loadResults } from "./data/load.ts";
import { driveIntervals } from "./data/timing.ts";
import { Router } from "./router.ts";
import { Accuracy } from "./sections/accuracy.ts";
import { readFigures } from "./sections/figures.ts";
import { Headline } from "./sections/headline.ts";
import { HowItWorks } from "./sections/how.ts";
import { mountSectionsMarkup } from "./sections/markup.ts";
import { mountShell } from "./shell.ts";
import { initTheme } from "./theme.ts";

console.info("Sightline — built by Safdar Hussain · https://github.com/safdar-hussain1/sightline");

mountShell();
mountSectionsMarkup();
initTheme();

const how = HowItWorks.mount();
const accuracy = Accuracy.mount();
const headline = Headline.mount();

new Router({
  focus: { demo: "#cockpit", results: "#results-title" },
  onShow: (view) => {
    how?.viewShown(view);
    headline?.viewShown(view);
  },
}).start();

const results = loadResults(new URL("./", document.baseURI).href);
results.then(
  (loaded) => {
    how?.setIntervals(driveIntervals(loaded.raw));
    try {
      const figures = readFigures(loaded.raw);
      accuracy?.show(figures);
      headline?.show(figures);
    } catch (error) {
      console.warn("The measurements could not be read:", error);
      accuracy?.fail(error);
      headline?.fail(error);
    }
  },
  (error: unknown) => {
    // The cockpit reports the same failure for itself, with its detail.
    accuracy?.fail(error);
    headline?.fail(error);
  },
);

void bootCockpit(() => import("./cockpit/cockpit.ts"), results).then((cockpit) => {
  how?.attach(cockpit);
  headline?.attach(cockpit);
});
