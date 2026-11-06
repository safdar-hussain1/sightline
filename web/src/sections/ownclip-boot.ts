/**
 * Bringing in "Try it on your own video" when it is wanted.
 *
 * The section sits below the fold, so its code is a chunk of its own,
 * imported as the section nears the screen, when a visitor reaches its file
 * input first, or at once on a page served without the section's markup.
 * Until the code has arrived a file chosen, or dropped on the console (the
 * same drop target the code takes over), is held and handed over, so none
 * is lost, and a drop never opens the file in the tab. The console says at
 * once that the video is opening, however long the code takes on a slow
 * connection. If the code cannot be had, the section says so in plain words
 * and keeps the file; the next choice, or a press on the button, tries again.
 * This module travels with the page's first chunk and holds nothing else.
 */

import type { LiveCockpit } from "../cockpit/live.ts";
import type { Results } from "../data/types.ts";

/** The section's id (ownclip-markup.ts names it too; a test keeps the two
 * equal). */
export const SECTION_ID = "own-video";

/** What the console says while a video opens; the section's code says the
 * same (ownclip-copy.ts takes it from here). */
export const OPENING = "Opening your video";

/** What the section says when its code cannot be had. */
export const CODE_MISSING = {
  title: "This part of the page could not be loaded",
  text: "Check the connection, then choose the video again: it is still only on your device.",
} as const;

/** How far below the screen the section may be when its code is asked for. */
const NEAR = "0px 0px 900px 0px";

/** What the boot needs of the section's code. */
export interface OwnClipCode {
  readonly OwnClip: {
    mount(results: Promise<Results>, cockpit: Promise<LiveCockpit | null>): { choose(file: File): Promise<void> } | null;
  };
}

const carriesFiles = (event: DragEvent) => [...(event.dataTransfer?.types ?? [])].includes("Files");

/** Start the section on `results` and `cockpit` once it is wanted. */
export function bootOwnClip(
  results: Promise<Results>,
  cockpit: Promise<LiveCockpit | null>,
  load: () => Promise<OwnClipCode> = () => import("./ownclip.ts"),
): { started(): Promise<void> | null } {
  const section = document.getElementById(SECTION_ID);
  let starting: Promise<void> | null = null;
  if (!section) return { started: () => starting };
  const part = (name: string) => section.querySelector<HTMLElement>(`[data-ownclip="${name}"]`);
  let held: File | null = null;
  let ready: { choose(file: File): Promise<void> } | null = null;

  /** Put the served markup into one of the states the code would draw. */
  const draw = (phase: "loading" | "error") => {
    const screen = part("console");
    if (screen) screen.dataset["phase"] = phase;
    const shown: Record<string, boolean> = { drop: false, player: phase === "loading", error: phase === "error" };
    for (const [name, on] of Object.entries(shown)) {
      const element = part(name);
      if (element) element.hidden = !on;
    }
    const status = part("status");
    if (status) {
      status.textContent = phase === "loading" ? OPENING : "";
      status.hidden = phase !== "loading";
    }
    const live = part("live");
    if (live && phase === "loading") live.textContent = `${OPENING}.`;
    if (phase === "error") {
      const title = part("error-title");
      const text = part("error-text");
      if (title) title.textContent = CODE_MISSING.title;
      if (text) text.textContent = CODE_MISSING.text;
    }
  };

  const start = () =>
    (starting ??= load().then(
      (code) => {
        ready = code.OwnClip.mount(results, cockpit);
        const file = held;
        held = null;
        if (ready && file) void ready.choose(file);
      },
      (error: unknown) => {
        console.warn("The own-video section's code could not be loaded:", error);
        // The next choice tries again; a file already chosen is kept for it.
        starting = null;
        if (held) draw("error");
      },
    ));

  const hold = (file: File | undefined) => {
    if (!file) return;
    held = file;
    draw("loading");
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
  section.addEventListener("click", (event) => {
    if (ready || !(event.target instanceof Element) || !event.target.closest('[data-ownclip="retry"], [data-ownclip="again"]')) return;
    (part("file") as HTMLInputElement | null)?.click();
  });
  // The console is the drop target, before the code arrives as after.
  const screen = part("console");
  screen?.addEventListener("dragover", (event) => {
    if (ready || !carriesFiles(event)) return;
    event.preventDefault();
    screen.dataset["drag"] = "true";
  });
  screen?.addEventListener("dragleave", (event) => {
    if (!ready && !(event.relatedTarget instanceof Node && screen.contains(event.relatedTarget))) delete screen.dataset["drag"];
  });
  screen?.addEventListener("drop", (event) => {
    if (ready || !carriesFiles(event)) return;
    event.preventDefault();
    delete screen.dataset["drag"];
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
