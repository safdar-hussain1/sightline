/**
 * The cockpit's sentences that depend on the data.
 *
 * Each is built from the clip file it describes, so a sentence cannot say
 * something its numbers do not: the twin is only said to look better on a clip
 * where its own error there is smaller, and how much of a clip it trained on
 * is quoted only where the clip file records it.
 */

import { AssetError } from "../data/load.ts";
import type { Clip, ModelKey } from "../data/types.ts";
import { fmtPct } from "../format.ts";

/** What the visitor reads about the selected model on the clip on screen. */
export function modelNote(model: ModelKey, clip: Clip | null): string {
  if (model === "sightline") {
    return "Trained with whole stretches of the drive held back for testing, so it is judged on road it never saw.";
  }
  const lead = "The same network, trained on shuffled frames.";
  if (!clip || clip.twin_trained_frac === undefined) {
    return lead;
  }
  if (clip.twin_trained_frac === 0) {
    return `${lead} It never trained on any frame of this clip.`;
  }
  const frames = `${fmtPct(clip.twin_trained_frac)} of these very frames`;
  const near = clip.twin_neighbour_frac === undefined
    ? ""
    : `, and ${fmtPct(clip.twin_neighbour_frac)} sit within one frame of one it trained on`;
  return clip.clip_mae_deg.shuffled < clip.clip_mae_deg.sightline
    ? `${lead} It looks better on this clip because it trained on ${frames}${near}.`
    : `${lead} It trained on ${frames}${near}.`;
}

/** "2018: minutes it never trained on" as its year and its description; a
 * title without a colon is all description. */
export function splitTitle(title: string): [string, string] {
  const at = title.indexOf(": ");
  return at > 0 ? [title.slice(0, at), title.slice(at + 2)] : ["", title];
}

/** The caption under the model's own input: its size, and how old the history
 * frames stacked with it are, from the clip's lags and frame rate. */
export function eyeCaption(clip: Clip): string {
  const size = `${clip.sprite.frame_w} × ${clip.sprite.frame_h} pixels`;
  const ages = clip.frame_lags.map((lag) => `${Math.round((lag / clip.fps) * 1000)} ms`);
  const history =
    ages.length === 0 ? ""
    : ages.length === 1 ? ` and of the one ${ages[0]} before it`
    : ` and of the ones ${ages.join(" and ")} before it`;
  return `What the model gets: ${size} of this frame${history}, with where it looked.`;
}

/** Why the demo cannot run, for the visitor: what went wrong in plain words,
 * never the URL or the runtime's own message (the console has those). */
export function failureMessage(error: unknown): string {
  if (error instanceof AssetError) {
    switch (error.kind) {
      case "missing":
        return "The demo’s footage and model are not on this copy of the site, so it cannot run here.";
      case "unreachable":
        return "The demo could not download its footage and model. Check the connection, then reload the page.";
      case "invalid":
        return "The demo’s data is not in the form this page reads, so it cannot run.";
      case "undecodable":
        return "This browser could not decode the clip’s footage, so the demo cannot run here.";
    }
  }
  return "This browser could not run the model, so the demo cannot start.";
}
