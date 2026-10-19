/**
 * Starting the demo, and what the page says if it cannot start at all.
 *
 * The cockpit's code, the model runtime and everything they download arrive
 * in a chunk of their own (see main.ts). If that chunk cannot be fetched, or
 * the cockpit cannot be built in this browser, the page must not go on saying
 * it is loading. This module travels with the page's first chunk, so it can
 * still put the served cockpit markup into its failed state, with a plain
 * sentence, without any of the cockpit's code.
 */

/** What the demo says when it cannot start and there is no more specific
 * reason to give the visitor. */
export const CANNOT_RUN = "This browser could not run the model, so the demo cannot start.";

/** The cockpit's section in the served page (markup.ts names it too; a test
 * keeps the two equal). */
export const COCKPIT_SELECTOR = "#cockpit";

/** Show the cockpit's failed state with `CANNOT_RUN`, on the served markup. */
export function showCannotRun(): void {
  const root = document.querySelector<HTMLElement>(COCKPIT_SELECTOR);
  const screen = root?.querySelector<HTMLElement>(".cockpit__console");
  const status = root?.querySelector<HTMLElement>('[data-cockpit="status"]');
  if (screen) screen.dataset["state"] = "unavailable";
  if (status) {
    status.textContent = CANNOT_RUN;
    status.hidden = false;
  }
}

/** Load the cockpit's chunk with `load` and start it on `results` (the page's
 * own request for results.json, so the file is fetched once); if either step
 * fails, say that the demo cannot run. Resolves to the running cockpit, or
 * null when there is none. */
export async function bootCockpit<Results, Cockpit>(
  load: () => Promise<{ startCockpit(results?: Results): Cockpit | null }>,
  results?: Results,
): Promise<Cockpit | null> {
  try {
    const { startCockpit } = await load();
    return startCockpit(results);
  } catch (error) {
    console.warn("The demo could not start:", error);
    showCannotRun();
    return null;
  }
}
