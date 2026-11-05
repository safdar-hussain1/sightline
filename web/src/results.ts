/**
 * The Results view: the headline band and its live check, the paired
 * experiment, the ladder and the cards, with the data they read.
 *
 * The page imports this module the first time the view is shown -- at once on
 * a deep link to `#results` -- or, ahead of that, once the demo has loaded and
 * the browser is idle, so the demo's first screen never waits on it.
 * Importing it does nothing by itself: `startResults` writes the view's
 * markup in if the page arrived without it, mounts its sections and fills them
 * from results.json, the same request the demo started on.
 */

import type { LiveCockpit } from "./cockpit/live.ts";
import type { Results } from "./data/types.ts";
import type { ViewName } from "./router.ts";
import { Cards } from "./sections/cards.ts";
import {
  readFigures,
  readFrames,
  readLadder,
  readLegacy,
  readPaired,
  readParity,
  readRegimes,
  readSplits,
  readYardsticks,
  type Figures,
} from "./sections/figures.ts";
import { Headline } from "./sections/headline.ts";
import { LadderSection } from "./sections/ladder.ts";
import { PairedSection } from "./sections/paired.ts";
import { mountResultsMarkup } from "./sections/results-markup.ts";

export interface ResultsView {
  /** The router switched views. */
  viewShown(view: ViewName): void;
}

/** Show one part of the view from the data, or say why it cannot be shown:
 * each part reads its own keys, so a malformed one costs only that part. */
function part(show: () => void, fail: (error: unknown) => void): void {
  try {
    show();
  } catch (error) {
    console.warn("A part of the measurements could not be read:", error);
    fail(error);
  }
}

/** Mount the view's sections and fill them in as `results` and `cockpit`
 * arrive. `view` is the view on screen now. */
export function startResults(results: Promise<Results>, cockpit: Promise<LiveCockpit | null>, view: ViewName): ResultsView {
  mountResultsMarkup();
  const headline = Headline.mount();
  const paired = PairedSection.mount();
  const ladder = LadderSection.mount();
  const cards = Cards.mount();
  headline?.onCheck((look, text) => cards?.live(look, text));
  const sections = [headline, paired, ladder, cards];
  const shown = (name: ViewName) => {
    for (const section of sections) section?.viewShown(name);
  };

  results.then(
    ({ raw }) => {
      let figures: Figures | null = null;
      part(
        () => {
          figures = readFigures(raw);
          headline?.show(figures);
        },
        (error) => headline?.fail(error),
      );
      part(() => paired?.show(readPaired(raw), (figures ?? readFigures(raw)).gaps), (error) => paired?.fail(error));
      part(() => ladder?.show(readLadder(raw), readSplits(raw)), (error) => ladder?.fail(error));
      part(() => cards?.showBaselines(readYardsticks(raw)), (error) => cards?.fail("baselines", error));
      part(() => cards?.showFrames(readLadder(raw), readFrames(raw)), (error) => cards?.fail("frames", error));
      part(() => cards?.showRegimes(readRegimes(raw)), (error) => cards?.fail("regimes", error));
      part(() => cards?.showRecipe(readLegacy(raw)), (error) => cards?.fail("recipe", error));
      part(() => cards?.showTrust(readParity(raw), readSplits(raw)), (error) => cards?.fail("trust", error));
    },
    (error: unknown) => {
      // The cockpit reports the same failure for itself, with its detail.
      headline?.fail(error);
      paired?.fail(error);
      ladder?.fail(error);
      for (const card of ["baselines", "frames", "regimes", "recipe", "trust"] as const) cards?.fail(card, error);
    },
  );
  void cockpit.then((live) => headline?.attach(live));
  shown(view);
  return { viewShown: shown };
}
