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
import { driveGaps } from "./data/timing.ts";
import { Router } from "./router.ts";
import { Accuracy } from "./sections/accuracy.ts";
import { Cards } from "./sections/cards.ts";
import { readFigures, readFrames, readLadder, readLegacy, readPaired, readParity, readRegimes, readSplits, readYardsticks, type Figures } from "./sections/figures.ts";
import { Headline } from "./sections/headline.ts";
import { HowItWorks } from "./sections/how.ts";
import { LadderSection } from "./sections/ladder.ts";
import { mountSectionsMarkup } from "./sections/markup.ts";
import { PairedSection } from "./sections/paired.ts";
import { mountShell } from "./shell.ts";
import { initTheme } from "./theme.ts";

console.info("Sightline — built by Safdar Hussain · https://github.com/safdar-hussain1/sightline");

mountShell();
mountSectionsMarkup();
initTheme();

const how = HowItWorks.mount();
const accuracy = Accuracy.mount();
const headline = Headline.mount();
const paired = PairedSection.mount();
const ladder = LadderSection.mount();
const cards = Cards.mount();
headline?.onCheck((look, text) => cards?.live(look, text));

new Router({
  focus: { demo: "#cockpit", results: "#results-title" },
  onShow: (view) => {
    how?.viewShown(view);
    headline?.viewShown(view);
    paired?.viewShown(view);
    ladder?.viewShown(view);
    cards?.viewShown(view);
  },
}).start();

/** Show one part of the Results view from the data, or say why it cannot be
 * shown: each part reads its own keys, so a malformed one costs only that part. */
function part(show: () => void, fail: (error: unknown) => void): void {
  try {
    show();
  } catch (error) {
    console.warn("A part of the measurements could not be read:", error);
    fail(error);
  }
}

/** Every part of the Results view, from the parsed results.json. */
function showResults(raw: Readonly<Record<string, unknown>>, figures: Figures | null): void {
  part(
    () => paired?.show(readPaired(raw), (figures ?? readFigures(raw)).gaps),
    (error) => paired?.fail(error),
  );
  part(
    () => ladder?.show(readLadder(raw), readSplits(raw)),
    (error) => ladder?.fail(error),
  );
  part(() => cards?.showBaselines(readYardsticks(raw)), (error) => cards?.fail("baselines", error));
  part(() => cards?.showFrames(readLadder(raw), readFrames(raw)), (error) => cards?.fail("frames", error));
  part(() => cards?.showRegimes(readRegimes(raw)), (error) => cards?.fail("regimes", error));
  part(() => cards?.showRecipe(readLegacy(raw)), (error) => cards?.fail("recipe", error));
  part(() => cards?.showTrust(readParity(raw), readSplits(raw)), (error) => cards?.fail("trust", error));
}

/** The Results view without its data: every part says why. */
function failResults(error: unknown): void {
  paired?.fail(error);
  ladder?.fail(error);
  for (const card of ["baselines", "frames", "regimes", "recipe", "trust"] as const) cards?.fail(card, error);
}

const results = loadResults(new URL("./", document.baseURI).href);
results.then(
  (loaded) => {
    how?.setGaps(driveGaps(loaded.raw));
    let figures: Figures | null = null;
    try {
      figures = readFigures(loaded.raw);
      accuracy?.show(figures);
      headline?.show(figures);
    } catch (error) {
      console.warn("The measurements could not be read:", error);
      accuracy?.fail(error);
      headline?.fail(error);
    }
    showResults(loaded.raw, figures);
  },
  (error: unknown) => {
    // The cockpit reports the same failure for itself, with its detail.
    accuracy?.fail(error);
    headline?.fail(error);
    failResults(error);
  },
);

void bootCockpit(() => import("./cockpit/cockpit.ts"), results).then((cockpit) => {
  how?.attach(cockpit);
  headline?.attach(cockpit);
});
