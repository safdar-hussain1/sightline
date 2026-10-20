/**
 * What the page calls each network, from its key in the data.
 *
 * A key that ends in a frame count ("sightlinenet-1f") names a variant of the
 * network that sees that many frames, and its name says so in words; the count
 * comes from the key, never from a word typed here. Nothing else is imported,
 * so the served markup can use the same names.
 */

import { countWord } from "../format.ts";

/** Each network's own name, by the key its runs carry. */
const BASE_NAMES: Readonly<Record<string, string>> = { sightlinenet: "SightlineNet", pilotnet: "PilotNet" };

/** The frames a variant's key names ("sightlinenet-1f": 1), or null for a key
 * that names none. */
export function framesInKey(key: string): number | null {
  const found = /-(\d+)f$/.exec(key);
  return found ? Number(found[1]) : null;
}

/** A network as a legend names it ("SightlineNet, one frame") and as a
 * sentence does ("SightlineNet with one frame"). */
export function networkName(key: string): { readonly name: string; readonly said: string } {
  const frames = framesInKey(key);
  const base = frames === null ? key : key.slice(0, key.lastIndexOf("-"));
  const name = BASE_NAMES[base] ?? base;
  if (frames === null) return { name, said: name };
  const count = `${countWord(frames)} frame${frames === 1 ? "" : "s"}`;
  return { name: `${name}, ${count}`, said: `${name} with ${count}` };
}
