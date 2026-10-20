/**
 * The page's two views -- the live demo and the results -- in one document,
 * chosen by the location's hash.
 *
 * `#results` shows the results; anything else shows the demo, so a link to a
 * place in the demo (`#how-it-works`) still lands there. Every switch is an
 * ordinary history entry: the browser's Back and Forward move between the
 * views, and a deep link opens on the right one (the page's first script marks
 * it on <html> before the first paint, so the demo never flashes first).
 *
 * The router shows one view and hides the other, marks the nav link of the
 * view on screen, and fades the view in -- except under reduced motion, where
 * the switch is instant. Following a link starts the new view at its top, or
 * at the place the link names, and moves focus there. Going back and forward
 * leaves the scroll position to the browser, which restores where the visitor
 * was; once it has, focus moves, without scrolling, to the heading or control
 * nearest the top of the screen in the view now shown -- or a labelled region
 * whose top is on screen, when none of its own headings or controls is -- so
 * the next Tab carries on from where the visitor is rather than from a place
 * in the view just hidden. Focus already on something still on screen stays
 * where it is. It
 * never changes the document's title: the title is what search engines index,
 * and it names the page, not the view.
 */

export type ViewName = "demo" | "results";

export const VIEW_NAMES: readonly ViewName[] = ["demo", "results"];

/** The view a location hash shows. */
export function viewForHash(hash: string): ViewName {
  return hash.toLowerCase() === "#results" ? "results" : "demo";
}

export interface RouterOptions {
  /** For each view, the element focus moves to when the view is switched to. */
  readonly focus: Readonly<Record<ViewName, string>>;
  readonly reducedMotion?: () => boolean;
  /** Fades a view in; the browser's own `animate` unless a test says otherwise. */
  readonly animate?: (el: HTMLElement) => void;
  /** Hears each switch, and the view the page opens on. */
  readonly onShow?: (view: ViewName) => void;
  /** Calls back once the browser has put back the scroll position it
   * restores on Back and Forward: after two animation frames, unless a test
   * says otherwise. */
  readonly afterRestore?: (callback: () => void) => void;
}

/** What focus may land on after Back or Forward: the view's headings, the
 * controls a visitor tabs between, and its labelled regions. */
const HEADINGS = "h2, h3";
const CONTROLS =
  'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), ' +
  'textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';
const REGIONS = "section[aria-label], section[aria-labelledby]";

/** A place focus may land on, by what it is: a heading says the most, then a
 * control, then a region. */
export type PlaceKind = "heading" | "control" | "region";
const RANK: Readonly<Record<PlaceKind, number>> = { heading: 0, control: 1, region: 2 };

/** Of `boxes`, the index of the one whose top is nearest the top of a
 * viewport `height` tall, among those at least partly in view -- a region
 * only while its own top is on screen, since one that began far above says
 * nothing of where the visitor is. A heading wins a tie with a control, a
 * control with a region. Null when none qualifies. */
export function nearestToTop(
  boxes: ReadonlyArray<{ readonly top: number; readonly bottom: number; readonly kind: PlaceKind }>,
  height: number,
): number | null {
  let best: number | null = null;
  boxes.forEach((box, i) => {
    if (box.bottom <= 0 || box.top >= height) return;
    if (box.kind === "region" && box.top < 0) return;
    const current = best === null ? undefined : boxes[best];
    const distance = Math.abs(box.top);
    if (
      current === undefined ||
      distance < Math.abs(current.top) ||
      (distance === Math.abs(current.top) && RANK[box.kind] < RANK[current.kind])
    ) {
      best = i;
    }
  });
  return best;
}

/** Whether a radio button is the one of its group that Tab reaches: the
 * checked one, or the first when none is. */
function tabbable(element: HTMLElement): boolean {
  if (!(element instanceof HTMLInputElement) || element.type !== "radio" || element.name === "") return true;
  if (element.checked) return true;
  const group = [...(element.form ?? element.ownerDocument).querySelectorAll<HTMLInputElement>(`input[type="radio"]`)].filter(
    (other) => other.name === element.name,
  );
  return !group.some((other) => other.checked) && group[0] === element;
}

const afterTwoFrames = (callback: () => void) => {
  requestAnimationFrame(() => requestAnimationFrame(callback));
};

const fadeIn = (el: HTMLElement) => {
  el.animate?.(
    [
      { opacity: 0, transform: "translateY(0.5rem)" },
      { opacity: 1, transform: "none" },
    ],
    { duration: 280, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)" },
  );
};

const prefersReducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export class Router {
  private current: ViewName | null = null;
  /** The hash of the in-page link the visitor just followed, if any: a switch
   * it causes starts at the top of the new view. */
  private followed: string | null = null;
  private readonly reduced: () => boolean;
  private readonly animate: (el: HTMLElement) => void;
  private readonly afterRestore: (callback: () => void) => void;
  private readonly listening = new AbortController();

  constructor(private readonly options: RouterOptions) {
    this.reduced = options.reducedMotion ?? prefersReducedMotion;
    this.animate = options.animate ?? fadeIn;
    this.afterRestore = options.afterRestore ?? afterTwoFrames;
  }

  /** The view on screen. */
  get view(): ViewName {
    return this.current ?? viewForHash(location.hash);
  }

  /** Show the view the location names and follow every change after it. */
  start(): void {
    this.sync(true);
    // Back and Forward fire popstate before the browser restores the scroll
    // position, so the view is switched in time for it; a followed link may
    // fire only hashchange. Showing a view twice is harmless.
    const { signal } = this.listening;
    addEventListener("popstate", () => this.sync(false), { signal });
    addEventListener("hashchange", () => this.sync(false), { signal });
    document.addEventListener("click", (event) => this.clicked(event), { signal });
  }

  /** Stop following the location. */
  stop(): void {
    this.listening.abort();
  }

  private clicked(event: MouseEvent): void {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
      return;
    }
    const link = (event.target as Element | null)?.closest?.<HTMLAnchorElement>('a[href^="#"]');
    if (!link) return;
    if (viewForHash(link.hash) === this.current && link.dataset["viewLink"] !== undefined) {
      // The nav link of the view on screen: back to its top.
      event.preventDefault();
      scrollTo({ top: 0, behavior: this.reduced() ? "auto" : "smooth" });
      return;
    }
    this.followed = link.hash;
  }

  private sync(initial: boolean): void {
    const hash = location.hash;
    const view = viewForHash(hash);
    const followed = this.followed === hash;
    this.followed = null;
    if (view === this.current) return;
    this.current = view;
    this.show(view);
    this.options.onShow?.(view);
    if (initial) return;

    const panel = this.panel(view);
    const target = hash.length > 1 ? document.getElementById(decodeURIComponent(hash.slice(1))) : null;
    const place = target && panel?.contains(target) ? target : null;
    if (place) {
      // A place in the view: the browser could not scroll to it while the view
      // was hidden.
      place.scrollIntoView({ block: "start" });
    } else if (followed) {
      scrollTo(0, 0);
    }
    if (followed) {
      const focus = place ?? document.querySelector<HTMLElement>(this.options.focus[view]);
      if (focus) this.focusOn(focus);
    } else {
      // Back, Forward, or a hash typed in: the browser puts the visitor back
      // where they were, then focus follows them there.
      this.afterRestore(() => this.refocus(view));
    }
    if (panel && !this.reduced()) this.animate(panel);
  }

  /** Focus `element` without scrolling; a heading or region is made
   * focusable for it, a control keeps its own place in the Tab order. */
  private focusOn(element: HTMLElement): void {
    if (!element.matches(CONTROLS) && !element.hasAttribute("tabindex")) element.tabIndex = -1;
    element.focus({ preventScroll: true });
  }

  /** Focus, without scrolling, the heading or control nearest the top of the
   * screen in `view` (see `nearestToTop`) -- unless the visitor has moved on,
   * or focus is on something still on screen. */
  private refocus(view: ViewName): void {
    if (this.current !== view) return;
    const panel = this.panel(view);
    if (!panel) return;
    const active = document.activeElement;
    const lost = active === null || active === document.body || !active.isConnected || active.closest(".view[hidden]") !== null;
    if (!lost) return;
    const found = (selector: string, kind: PlaceKind) =>
      [...panel.querySelectorAll<HTMLElement>(selector)].flatMap((element) => {
        const box = element.getBoundingClientRect();
        // Nothing drawn (hidden, or in a closed disclosure) is a place to be.
        if (box.width === 0 && box.height === 0) return [];
        if (kind === "control" && !tabbable(element)) return [];
        return [{ element, kind, top: box.top, bottom: box.bottom }];
      });
    const inView = (place: { top: number; bottom: number }) => place.bottom > 0 && place.top < innerHeight;
    const inner = [...found(HEADINGS, "heading"), ...found(CONTROLS, "control")];
    // A region is only a place of its own when nothing in it is on screen.
    const regions = found(REGIONS, "region").filter((region) => !inner.some((place) => inView(place) && region.element.contains(place.element)));
    const places = [...inner, ...regions];
    const nearest = nearestToTop(places, innerHeight);
    const target = nearest === null ? document.querySelector<HTMLElement>(this.options.focus[view]) : places[nearest]?.element;
    if (target) this.focusOn(target);
  }

  private panel(view: ViewName): HTMLElement | null {
    return document.querySelector<HTMLElement>(`.view[data-view="${view}"]`);
  }

  private show(view: ViewName): void {
    document.documentElement.dataset["view"] = view;
    for (const name of VIEW_NAMES) {
      const panel = this.panel(name);
      if (panel) panel.hidden = name !== view;
    }
    for (const link of document.querySelectorAll<HTMLAnchorElement>("[data-view-link]")) {
      if (link.dataset["viewLink"] === view) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
    const nav = document.querySelector<HTMLElement>("[data-views-nav]");
    if (nav) nav.dataset["active"] = view;
  }
}
