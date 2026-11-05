/**
 * The Results view when its own code could not be had: every part that was
 * waiting for figures says so in plain words and drops what it was waiting
 * to draw, so nothing is left on screen reading "Checking" for good. The
 * parts that run when the view's code is there say their own failures.
 */

/** What a visitor is told when the Results view's code could not be loaded. */
export function resultsCodeFailure(): string {
  return "The results could not be shown: this part of the page did not load. Check the connection, then reload the page.";
}

/** Mark every waiting part of the Results view unavailable, with the message. */
export function markResultsUnavailable(root: ParentNode = document): void {
  const message = resultsCodeFailure();
  for (const part of root.querySelectorAll<HTMLElement>('#results-view [data-state="loading"]')) part.dataset["state"] = "unavailable";
  for (const words of root.querySelectorAll<HTMLElement>('#results-view :is(.results__lede, .study__lead, [data-card$="-lead"], [data-card="export"])')) {
    words.textContent = message;
  }
}
