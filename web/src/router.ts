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
 * at the place the link names, and moves focus there; going back and forward
 * leaves the scroll position and focus to the browser, which restores where
 * the visitor was (focus pulled to a view's top would send the next Tab away
 * from it). It
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
}

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
  private readonly listening = new AbortController();

  constructor(private readonly options: RouterOptions) {
    this.reduced = options.reducedMotion ?? prefersReducedMotion;
    this.animate = options.animate ?? fadeIn;
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
      if (focus) {
        if (!focus.hasAttribute("tabindex")) focus.tabIndex = -1;
        focus.focus({ preventScroll: true });
      }
    }
    if (panel && !this.reduced()) this.animate(panel);
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
