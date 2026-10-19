/**
 * The cockpit's sentences: what it says about the data, and what it says when
 * something cannot be had.
 *
 * Each sentence about the data is built from the clip file it describes, so it
 * cannot say something its numbers do not: the twin is only said to look
 * better on a clip where its own error there is smaller, how much of a clip it
 * trained on is quoted only where the clip file records it, and the footage is
 * never said to match the published frames unless every frame was checked.
 */

import { AssetError } from "../data/load.ts";
import type { Clip, ModelKey } from "../data/types.ts";
import { fmtPct } from "../format.ts";
import { CANNOT_RUN } from "./boot.ts";
import type { DecodeCheck } from "./sprites.ts";

/** Said once when the visitor chose the twin and it could not be downloaded
 * or run: the demo carries on with Sightline. */
export const TWIN_UNAVAILABLE = "The twin could not be loaded, so Sightline is shown instead; pick the twin again to retry.";

/** The standing remark about how this browser decoded the clip's frames:
 * nothing when they are the reference decoder's, otherwise why the angles on
 * screen may stray a little from the published ones. `total` is the number of
 * frames checked. */
export function decodeNote(check: DecodeCheck, total: number): string {
  switch (check.status) {
    case "match":
      return "";
    case "differ":
      return `This browser decodes ${check.frames.length} of the clip’s ${total} frames slightly differently from the reference decoder, so the model’s angles here can differ a little from the published ones.`;
    case "unverified":
      return "This browser could not check its decoded frames against the reference decoder’s, so the model’s angles here may differ a little from the published ones.";
  }
}

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
      case "stopped":
        return "The clip’s footage stopped loading. Pick the clip again to reload it.";
    }
  }
  return CANNOT_RUN;
}
