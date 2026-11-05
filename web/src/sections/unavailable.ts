/**
 * The Results view when its own code could not be had: the view says so once,
 * in plain words, in its lede, and every part that was waiting for figures
 * drops what it was waiting to draw, so nothing is left on screen reading
 * "Checking" for good. When the view's code is there, its parts say their own
 * failures.
 */

/** What a visitor is told when the Results view's code could not be loaded. */
export function resultsCodeFailure(): string {
  return "The results could not be shown: this part of the page did not load. Check the connection, then reload the page.";
}

/** Mark every waiting part of the Results view unavailable, and say why once, in its lede. */
export function markResultsUnavailable(root: ParentNode = document): void {
  for (const part of root.querySelectorAll<HTMLElement>('#results-view [data-state="loading"]')) part.dataset["state"] = "unavailable";
  for (const words of root.querySelectorAll<HTMLElement>("#results-view .results__lede")) words.textContent = resultsCodeFailure();
}
