// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Router, viewForHash, type ViewName } from "./router.ts";

const TITLE = "Sightline — a steering model that shows where it looks";

/** The page's skeleton: the nav, and the two views with what each focuses. */
function page(): void {
  document.title = TITLE;
  document.body.innerHTML = `
    <header><nav class="views-nav" data-views-nav>
      <a href="#demo" data-view-link="demo" aria-current="page">Demo</a>
      <a href="#results" data-view-link="results">Results</a>
    </nav></header>
    <main>
      <div class="view" data-view="demo">
        <section id="cockpit" aria-label="Live demo">cockpit</section>
        <section id="how-it-works"><h2>How it works</h2></section>
        <a id="to-results" href="#results">See what the measurements found</a>
      </div>
      <div class="view" data-view="results" hidden>
        <h2 id="results-title">What the measurements found</h2>
      </div>
    </main>`;
}

const q = (selector: string) => document.querySelector(selector) as HTMLElement;
const view = (name: ViewName) => document.querySelector<HTMLElement>(`.view[data-view="${name}"]`) as HTMLElement;
const link = (name: ViewName) => document.querySelector<HTMLAnchorElement>(`[data-view-link="${name}"]`) as HTMLAnchorElement;
const shown = () => (["demo", "results"] as const).filter((name) => !view(name).hidden);
const ticks = async (count = 3) => {
  for (let i = 0; i < count; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
};

let animations: Array<{ el: Element }> = [];
let scrolls: unknown[][] = [];
let titleWrites = 0;

let router: Router | null = null;

function start(reduced = false): Router {
  router = new Router({
    focus: { demo: "#cockpit", results: "#results-title" },
    reducedMotion: () => reduced,
    animate: (el) => void animations.push({ el }),
  });
  router.start();
  return router;
}

beforeEach(() => {
  page();
  animations = [];
  scrolls = [];
  titleWrites = 0;
  history.replaceState(null, "", "/");
  vi.spyOn(window, "scrollTo").mockImplementation((...args: unknown[]) => void scrolls.push(args));
  let proto: object | null = Object.getPrototypeOf(document);
  while (proto && !Object.getOwnPropertyDescriptor(proto, "title")) proto = Object.getPrototypeOf(proto);
  const descriptor = proto ? Object.getOwnPropertyDescriptor(proto, "title") : undefined;
  Object.defineProperty(document, "title", {
    configurable: true,
    get: () => descriptor?.get?.call(document),
    set: (value: string) => {
      titleWrites += 1;
      descriptor?.set?.call(document, value);
    },
  });
});

afterEach(() => {
  router?.stop();
  router = null;
  vi.restoreAllMocks();
  delete (document as { title?: string }).title;
  delete document.documentElement.dataset["view"];
});

describe("viewForHash", () => {
  it("shows the results for #results, and the demo for anything else", () => {
    expect(viewForHash("#results")).toBe("results");
    expect(viewForHash("#RESULTS")).toBe("results");
    for (const hash of ["", "#", "#demo", "#how-it-works", "#resultsx", "#cockpit"]) {
      expect(viewForHash(hash), hash).toBe("demo");
    }
  });
});

describe("the router", () => {
  it("opens on the demo, as the page is served, without moving focus or animating", () => {
    start();
    expect(shown()).toEqual(["demo"]);
    expect(document.documentElement.dataset["view"]).toBe("demo");
    expect(link("demo").getAttribute("aria-current")).toBe("page");
    expect(link("results").hasAttribute("aria-current")).toBe(false);
    expect(document.activeElement).toBe(document.body);
    expect(animations).toEqual([]);
  });

  it("opens straight on the results from a deep link", () => {
    history.replaceState(null, "", "/#results");
    start();
    expect(shown()).toEqual(["results"]);
    expect(document.documentElement.dataset["view"]).toBe("results");
    expect(link("results").getAttribute("aria-current")).toBe("page");
    expect(link("demo").hasAttribute("aria-current")).toBe(false);
    expect(document.activeElement).toBe(document.body);
    expect(scrolls).toEqual([]);
  });

  it("switches views from the nav: to the top of the new view, focus on its heading, with a short fade", async () => {
    start();
    link("results").click();
    await ticks();
    expect(location.hash).toBe("#results");
    expect(shown()).toEqual(["results"]);
    expect(scrolls).toEqual([[0, 0]]);
    expect(document.activeElement?.id).toBe("results-title");
    expect(animations.map(({ el }) => el)).toEqual([view("results")]);

    link("demo").click();
    await ticks();
    expect(shown()).toEqual(["demo"]);
    expect(document.activeElement?.id).toBe("cockpit");
    expect(link("demo").getAttribute("aria-current")).toBe("page");
  });

  it("follows a link inside a view to the other view", async () => {
    start();
    document.querySelector<HTMLAnchorElement>("#to-results")?.click();
    await ticks();
    expect(shown()).toEqual(["results"]);
    expect(document.activeElement?.id).toBe("results-title");
  });

  it("goes back and forward with the browser, leaving the scroll position and focus to it", async () => {
    start();
    link("results").click();
    await ticks();
    expect(document.activeElement?.id).toBe("results-title");
    scrolls.length = 0;
    (document.activeElement as HTMLElement).blur();
    history.back();
    await ticks(6);
    expect(location.hash).toBe("");
    expect(shown()).toEqual(["demo"]);
    expect(scrolls).toEqual([]);
    // Focus is not pulled to the top of a view the browser is scrolling down.
    expect(document.activeElement).toBe(document.body);
    history.forward();
    await ticks(6);
    expect(shown()).toEqual(["results"]);
    expect(scrolls).toEqual([]);
    expect(document.activeElement).toBe(document.body);
  });

  it("switches instantly under reduced motion", async () => {
    start(true);
    link("results").click();
    await ticks();
    expect(shown()).toEqual(["results"]);
    expect(animations).toEqual([]);
  });

  it("shows the demo for a hash naming a place in it, and takes the visitor there", async () => {
    history.replaceState(null, "", "/#results");
    start();
    const target = document.querySelector<HTMLElement>("#how-it-works") as HTMLElement;
    const into = vi.fn();
    target.scrollIntoView = into;
    location.hash = "#how-it-works";
    await ticks();
    expect(shown()).toEqual(["demo"]);
    expect(into).toHaveBeenCalledOnce();
  });

  it("puts focus on the place a followed link names, not on the top of its view", async () => {
    history.replaceState(null, "", "/#results");
    start();
    const target = document.querySelector<HTMLElement>("#how-it-works") as HTMLElement;
    target.scrollIntoView = vi.fn();
    const to = document.createElement("a");
    to.href = "#how-it-works";
    q("main").append(to);
    to.click();
    await ticks();
    expect(shown()).toEqual(["demo"]);
    expect(document.activeElement?.id).toBe("how-it-works");
  });

  it("takes the visitor to the top when they pick the view they are on", async () => {
    history.replaceState(null, "", "/#results");
    start();
    link("results").click();
    await ticks();
    expect(shown()).toEqual(["results"]);
    expect(scrolls.length).toBe(1);
  });

  it("leaves a click with a modifier key to the browser", async () => {
    start();
    link("results").dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, metaKey: true }));
    await ticks();
    expect(scrolls).toEqual([]);
  });

  it("never touches the page's title", async () => {
    start();
    link("results").click();
    await ticks();
    history.back();
    await ticks(6);
    expect(titleWrites).toBe(0);
    expect(document.title).toBe(TITLE);
  });
});
