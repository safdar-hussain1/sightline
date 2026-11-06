/**
 * Bringing in "Try it on your own video" when it is wanted.
 *
 * The section sits below the fold, so its code is a chunk of its own,
 * imported as the section nears the screen, when a visitor reaches its file
 * input first, or at once on a page served without the section's markup.
 * Until the code has arrived a file chosen or dropped on the section is held
 * and handed over, so none is lost, and a drop never opens the file in the
 * tab. This module travels with the page's first chunk and holds nothing
 * else.
 */

import type { LiveCockpit } from "../cockpit/live.ts";
import type { Results } from "../data/types.ts";

/** The section's id (ownclip-markup.ts names it too; a test keeps the two
 * equal). */
export const SECTION_ID = "own-video";

/** How far below the screen the section may be when its code is asked for. */
const NEAR = "0px 0px 900px 0px";

/** What the boot needs of the section's code. */
export interface OwnClipCode {
  readonly OwnClip: {
    mount(results: Promise<Results>, cockpit: Promise<LiveCockpit | null>): { choose(file: File): Promise<void> } | null;
  };
}

const carriesFiles = (event: DragEvent) => [...(event.dataTransfer?.types ?? [])].includes("Files");

/** Start the section on `results` and `cockpit` once it is wanted. Resolves
 * once its code is running (or could not be had), for whoever waits. */
export function bootOwnClip(
  results: Promise<Results>,
  cockpit: Promise<LiveCockpit | null>,
  load: () => Promise<OwnClipCode> = () => import("./ownclip.ts"),
): { started(): Promise<void> | null } {
  const section = document.getElementById(SECTION_ID);
  let starting: Promise<void> | null = null;
  if (!section) return { started: () => starting };
  let held: File | null = null;
  let ready: { choose(file: File): Promise<void> } | null = null;

  const start = () =>
    (starting ??= load().then(
      (code) => {
        ready = code.OwnClip.mount(results, cockpit);
        const file = held;
        held = null;
        if (ready && file) void ready.choose(file);
      },
      (error: unknown) => console.warn("The own-video section's code could not be loaded:", error),
    ));

  const hold = (file: File | undefined) => {
    if (!file) return;
    held = file;
    void start();
  };
  // Seen before the section's own listeners, which take over once it runs.
  section.addEventListener(
    "change",
    (event) => {
      const input = event.target;
      if (ready || !(input instanceof HTMLInputElement) || input.type !== "file") return;
      hold(input.files?.[0]);
      input.value = "";
    },
    true,
  );
  section.addEventListener("dragover", (event) => {
    if (!ready && carriesFiles(event)) event.preventDefault();
  });
  section.addEventListener("drop", (event) => {
    if (ready || !carriesFiles(event)) return;
    event.preventDefault();
    hold(event.dataTransfer?.files[0]);
  });
  section.addEventListener("focusin", () => void start(), { once: true });

  if (section.childElementCount === 0 || typeof IntersectionObserver !== "function") {
    void start();
  } else {
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        observer.disconnect();
        void start();
      },
      { rootMargin: NEAR },
    );
    observer.observe(section);
  }
  return { started: () => starting };
}
