/**
 * How the drive was split: the drive the runs trained on as a band in time,
 * from its first frame to its last, with how hard the driver was turning
 * drawn above it, and what each split does with every stretch of it -- the
 * blocked and the later-in-the-drive splits exactly as the splits module cuts
 * them, the shuffled split as an even mix at its shares, an illustration. The
 * demo's 2018 clip is marked where it was recorded.
 *
 * The band is drawn the moment the data is in, on the blocked split (or the
 * split the browser restored); the first time it nears the screen in the
 * Results view it draws itself in from the drive's start. Picking another
 * split sweeps the new layout across the old one from the start of the drive
 * to its end, the counts in the key moving in step; under reduced motion both
 * are instant. Times come from the drive's own timestamps, bucket by bucket,
 * never from a frame count times a typical interval.
 */

import { attrs, easeInOut, easeOut, local, svg, tween, watchWidth } from "../charts/draw.ts";
import { ROLES, ditherRoles, timeOfFrame, type Role, type Segment } from "../charts/timeline.ts";
import { niceStep } from "../charts/scale.ts";
import { fmtCount } from "../format.ts";
import type { ViewName } from "../router.ts";
import {
  ROLE_NAMES,
  capitalised,
  clockTime,
  driveTitle,
  exactDeg,
  measurementsFailure,
  timelineAlt,
  timelineClipNote,
  timelineLead,
  timelineNote,
  type TimelineProtocol,
} from "./copy.ts";
import { textAt, type Frames } from "./countup.ts";
import type { ClipWindow, Timeline } from "./figures.ts";
import { onScreen, prefersReducedMotion } from "./motion.ts";

const PROTOCOLS: readonly TimelineProtocol[] = ["shuffled", "blocked", "later"];
const DRAW_MS = 1200;
const MORPH_MS = 720;
const LEAD_PX = 80;
/** One cell of the shuffled split's mix, px. */
const CELL_PX = 3;

interface Geometry {
  readonly width: number;
  readonly height: number;
  readonly x: (seconds: number) => number;
  readonly traceTop: number;
  readonly traceBottom: number;
  readonly bandTop: number;
  readonly bandBottom: number;
}

interface Drawing {
  readonly root: SVGSVGElement;
  readonly desc: SVGDescElement;
  readonly traceClip: SVGRectElement;
  readonly layers: ReadonlyMap<TimelineProtocol, { readonly group: SVGGElement; readonly clip: SVGRectElement }>;
  readonly geometry: Geometry;
}

const r2 = (n: number) => Math.round(n * 100) / 100;

export class TimelineSection {
  /** The section in the page, or null on a page without it. */
  static mount(): TimelineSection | null {
    const root = document.querySelector<HTMLElement>('[data-timeline="root"]');
    return root ? new TimelineSection(root) : null;
  }

  private data: Timeline | null = null;
  private clip: ClipWindow | null = null;
  private protocol: TimelineProtocol = "blocked";
  private drawing: Drawing | null = null;
  private drawn: boolean;
  private onView: boolean;
  private stop: () => void = () => undefined;
  private static count = 0;
  private readonly id = `timeline-${(TimelineSection.count += 1)}`;

  constructor(
    private readonly root: HTMLElement,
    private readonly reduced: () => boolean = prefersReducedMotion,
    private readonly frames?: Frames,
  ) {
    this.onView = document.documentElement.dataset["view"] === "results";
    this.drawn = reduced();
    this.part("protocols").addEventListener("change", (event) => {
      const value = (event.target as HTMLInputElement).value;
      if (value === "shuffled" || value === "blocked" || value === "later") this.choose(value);
    });
    if (typeof IntersectionObserver === "function") {
      new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) this.maybeDraw();
      }, { rootMargin: `0px 0px ${LEAD_PX}px 0px` }).observe(this.part("plot"));
    }
    const plot = this.part("plot");
    plot.addEventListener("pointermove", (event) => this.point(event));
    plot.addEventListener("pointerleave", () => this.point(null));
  }

  private part<T extends HTMLElement = HTMLElement>(name: string): T {
    const element = this.root.querySelector<T>(`[data-timeline="${name}"]`);
    if (!element) throw new Error(`the split timeline has no ${name}`);
    return element;
  }

  /** Write the timeline in and draw it, final at once, on the split picked
   * (the browser may have restored a pick). `clip` marks the demo's clip. */
  show(timeline: Timeline, clip: ClipWindow | null): void {
    this.data = timeline;
    this.clip = clip !== null && clip.drive === timeline.drive ? clip : null;
    const picked = this.root.querySelector<HTMLInputElement>('input[name="timeline-split"]:checked')?.value;
    if (picked === "shuffled" || picked === "blocked" || picked === "later") this.protocol = picked;
    this.part("lead").textContent = timelineLead(timeline);
        this.fillTable(timeline);
    this.describe(this.protocol);
    this.counts(this.protocol, null, 1);
    this.root.dataset["state"] = "ready";
    watchWidth(this.part("plot"), (width) => this.build(width));
  }

  fail(error: unknown): void {
    this.part("lead").textContent = measurementsFailure(error);
    this.root.dataset["state"] = "unavailable";
  }

  viewShown(view: ViewName): void {
    this.onView = view === "results";
    this.maybeDraw();
  }

  private describe(protocol: TimelineProtocol): void {
    if (!this.data) return;
    this.part("note").textContent = timelineNote(this.data, protocol);
    this.part("clip").textContent = this.clip ? timelineClipNote(this.data, this.clip, protocol) : "";
    if (this.drawing) this.drawing.desc.textContent = timelineAlt(this.data, protocol);
  }

  /** The key's counts for `protocol`, at `t` of the way from `from`'s. */
  private counts(protocol: TimelineProtocol, from: TimelineProtocol | null, t: number): void {
    if (!this.data) return;
    const sizes = this.data.protocols[protocol].sizes;
    const before = from === null ? null : this.data.protocols[from].sizes;
    for (const role of ROLES) {
      const final = `${fmtCount(sizes[role])} frames`;
      this.part(`count-${role}`).textContent = before === null ? final : textAt(final, sizes[role], before[role], t);
    }
  }

  private geometry(width: number): Geometry {
    const narrow = width < 560;
    const traceTop = 20;
    const traceBottom = traceTop + (narrow ? 46 : 62);
    const bandTop = traceBottom + 30;
    const bandBottom = bandTop + (narrow ? 30 : 38);
    const duration = this.data?.clock.durationS ?? 1;
    return { width, height: bandBottom + 30, x: (s) => (s / duration) * width, traceTop, traceBottom, bandTop, bandBottom };
  }

  /** The drawing at `width`, every layer laid out, the picked one shown. */
  private build(width: number): void {
    const timeline = this.data;
    if (!timeline || width <= 0) return;
    this.stop();
    const g = this.geometry(width);
    const root = svg("svg", { class: "chart__svg timeline-chart", role: "img", "aria-labelledby": `${this.id}-title ${this.id}-desc`, viewBox: `0 0 ${width} ${g.height}`, width, height: g.height });
    svg("title", { id: `${this.id}-title` }, root).textContent = `${capitalised(driveTitle(timeline.drive))}, and how the picked split deals its frames`;
    const desc = svg("desc", { id: `${this.id}-desc` }, root);
    const defs = svg("defs", {}, root);
    const clipRect = (name: string) => {
      const clip = svg("clipPath", { id: `${this.id}-${name}` }, defs);
      return svg("rect", { x: 0, y: 0, width, height: g.height }, clip);
    };

    // How hard the driver was turning, on a square-root scale, with the edges
    // of the regimes marked.
    const trace = svg("g", { class: "timeline-chart__trace", "clip-path": `url(#${this.id}-trace)` }, root);
    const traceClip = clipRect("trace");
    const vmax = Math.max(...timeline.meanAbsDeg, ...timeline.turnEdges);
    const y = (v: number) => g.traceBottom - Math.sqrt(Math.max(0, v) / vmax) * (g.traceBottom - g.traceTop);
    for (const edge of timeline.turnEdges) {
      svg("line", { class: "timeline-chart__rule", x1: 0, x2: width, y1: r2(y(edge)), y2: r2(y(edge)) }, trace);
    }
    svg("line", { class: "timeline-chart__rule timeline-chart__rule--zero", x1: 0, x2: width, y1: g.traceBottom, y2: g.traceBottom }, trace);
    const { clock } = timeline;
    const points = timeline.meanAbsDeg.map((v, k) => {
      const t0 = clock.startS[k] as number;
      const t1 = k + 1 < clock.startS.length ? (clock.startS[k + 1] as number) : clock.durationS;
      return `${r2(g.x((t0 + t1) / 2))} ${r2(y(v))}`;
    });
    svg("path", { class: "timeline-chart__area", d: `M0 ${g.traceBottom}L${points.join("L")}L${width} ${g.traceBottom}Z` }, trace);
    const traceLabel = svg("text", { class: "card-chart__note", x: 0, y: 12 }, root);
    traceLabel.textContent = "How hard the driver was turning (square-root scale)";
    // Each edge labelled where there is room for its label above the line.
    let above = -Infinity;
    for (const edge of [...timeline.turnEdges].sort((a, b) => b - a)) {
      if (y(edge) - above < 14) continue;
      const label = svg("text", { class: "card-chart__tick", x: 2, y: r2(y(edge) - 3) }, trace);
      label.textContent = exactDeg(edge);
      above = y(edge);
    }

    // Each split's layout, one layer apiece.
    const band = svg("g", { class: "timeline-chart__band" }, root);
    const layers = new Map<TimelineProtocol, { group: SVGGElement; clip: SVGRectElement }>();
    const xFrame = (frame: number) => g.x(timeOfFrame(clock, frame));
    for (const protocol of PROTOCOLS) {
      const group = svg("g", { class: "timeline-chart__layer", "data-split": protocol, "clip-path": `url(#${this.id}-${protocol})` }, band);
      const clip = clipRect(protocol);
      if (protocol === "shuffled") {
        const cells = Math.max(1, Math.floor(width / CELL_PX));
        const { train, val, test } = timeline.protocols.shuffled.sizes;
        ditherRoles(cells, { train, val, test }).forEach((role, i) => {
          svg("rect", { class: `timeline-chart__cell role--${role}`, x: r2((i * width) / cells), y: g.bandTop, width: r2(width / cells - 1), height: g.bandBottom - g.bandTop }, group);
        });
      } else {
        for (const segment of timeline.protocols[protocol].segments) {
          const x1 = xFrame(segment.from);
          const x2 = xFrame(segment.to);
          svg("rect", { class: `timeline-chart__segment role--${segment.role}`, x: r2(x1), y: g.bandTop, width: r2(Math.max(0.5, x2 - x1)), height: g.bandBottom - g.bandTop }, group);
        }
      }
      layers.set(protocol, { group, clip });
    }

    // The demo's clip, where it was recorded.
    if (this.clip && this.clip.firstS !== null && this.clip.lastS !== null) {
      const x1 = g.x(this.clip.firstS);
      const x2 = Math.max(x1 + 2, g.x(this.clip.lastS));
      const mark = svg("g", { class: "timeline-chart__clip" }, root);
      svg("rect", { class: "timeline-chart__clip-window", x: r2(x1), y: g.bandTop - 4, width: r2(x2 - x1), height: g.bandBottom - g.bandTop + 8, rx: 1.5 }, mark);
      const mid = (x1 + x2) / 2;
      const anchor = mid < width * 0.2 ? "start" : mid > width * 0.8 ? "end" : "middle";
      const label = svg("text", { class: "card-chart__note timeline-chart__clip-label", x: r2(anchor === "start" ? x1 : anchor === "end" ? x2 : mid), y: g.bandTop - 10, "text-anchor": anchor }, mark);
      label.textContent = `the demo’s clip, ${clockTime(this.clip.firstS)}`;
    }

    // The drive's clock, at round minutes.
    const step = niceStep(clock.durationS / 60, 6);
    const axis = svg("g", { class: "timeline-chart__axis" }, root);
    for (let m = 0; m * 60 <= clock.durationS; m += step) {
      const x = g.x(m * 60);
      svg("line", { class: "timeline-chart__tick", x1: r2(x), x2: r2(x), y1: g.bandBottom + 2, y2: g.bandBottom + 7 }, axis);
      const text = svg("text", { class: "card-chart__tick", x: r2(x), y: g.bandBottom + 20, "text-anchor": m === 0 ? "start" : "middle" }, axis);
      text.textContent = m === 0 ? "0" : `${fmtCount(m)} min`;
    }

    const old = this.part("plot").querySelector("svg");
    if (old) old.replaceWith(root);
    else this.part("plot").prepend(root);
    this.drawing = { root, desc, traceClip, layers, geometry: g };
    this.describe(this.protocol);
    this.reveal(this.protocol, 1);
    attrs(traceClip, { width });
    this.maybeDraw();
  }

  /** Show `protocol`'s layer revealed `t` of the way across, the others hidden
   * except `under`, which stays whole beneath it. */
  private reveal(protocol: TimelineProtocol, t: number, under: TimelineProtocol | null = null): void {
    const d = this.drawing;
    if (!d) return;
    for (const [name, layer] of d.layers) {
      const shown = name === protocol ? t : name === under ? 1 : 0;
      attrs(layer.clip, { width: r2(d.geometry.width * shown) });
      layer.group.style.display = shown > 0 ? "" : "none";
    }
    // The layer arriving lies over the one it replaces.
    const arriving = d.layers.get(protocol)?.group;
    if (arriving && arriving.nextSibling) arriving.parentNode?.append(arriving);
  }

  /** Draw the timeline in, once, the first time it nears the screen. */
  private maybeDraw(): void {
    if (this.drawn || !this.drawing || !this.onView) return;
    if (!onScreen(this.part("plot"), LEAD_PX)) return;
    this.drawn = true;
    const d = this.drawing;
    const protocol = this.protocol;
    this.stop = tween(
      DRAW_MS,
      (p) => {
        attrs(d.traceClip, { width: r2(d.geometry.width * easeOut(local(p, 0, 0.75))) });
        this.reveal(protocol, easeOut(local(p, 0.2, 0.8)));
      },
      { reduced: this.reduced(), frames: this.frames },
    );
  }

  /** Sweep `protocol`'s layout across the one on screen. */
  private choose(protocol: TimelineProtocol): void {
    if (protocol === this.protocol) return;
    const from = this.protocol;
    this.protocol = protocol;
    this.describe(protocol);
    this.point(null);
    if (!this.drawing) {
      this.counts(protocol, null, 1);
      return;
    }
    this.stop();
    this.drawn = true;
    this.stop = tween(
      MORPH_MS,
      (p) => {
        const t = easeInOut(p);
        this.reveal(protocol, t, p < 1 ? from : null);
        this.counts(protocol, p < 1 ? from : null, t);
      },
      { reduced: this.reduced(), frames: this.frames },
    );
  }

  /** The stretch under the pointer, in words; null puts it away. */
  private point(event: PointerEvent | null): void {
    const tip = this.part("tip");
    const d = this.drawing;
    const timeline = this.data;
    if (!event || !d || !timeline) {
      tip.hidden = true;
      return;
    }
    const box = d.root.getBoundingClientRect();
    const px = event.clientX - box.left;
    const py = event.clientY - box.top;
    const g = d.geometry;
    if (px < 0 || px > g.width || py < g.bandTop - 6 || py > g.bandBottom + 6) {
      tip.hidden = true;
      return;
    }
    const seconds = (px / g.width) * timeline.clock.durationS;
    if (this.protocol === "shuffled") {
      tip.textContent = `At ${clockTime(seconds)}: every frame here was dealt at random`;
    } else {
      const segment = this.segmentAt(seconds);
      if (!segment) {
        tip.hidden = true;
        return;
      }
      tip.textContent =
        `${ROLE_NAMES[segment.role]}: ${clockTime(timeOfFrame(timeline.clock, segment.from))} to ${clockTime(timeOfFrame(timeline.clock, segment.to))}, ` +
        `frames ${fmtCount(segment.from)} to ${fmtCount(segment.to - 1)}`;
    }
    tip.hidden = false;
    const left = Math.min(Math.max(8, px + 14), Math.max(8, box.width - tip.offsetWidth - 8));
    tip.style.transform = `translate(${Math.round(left)}px, ${Math.round(Math.max(4, g.bandTop - tip.offsetHeight - 8))}px)`;
  }

  /** The picked split's stretch at `seconds` into the drive. */
  private segmentAt(seconds: number): Segment | null {
    const timeline = this.data;
    if (!timeline || this.protocol === "shuffled") return null;
    const segments = timeline.protocols[this.protocol].segments;
    return segments.find((s) => timeOfFrame(timeline.clock, s.from) <= seconds && seconds < timeOfFrame(timeline.clock, s.to)) ?? segments[segments.length - 1] ?? null;
  }

  private fillTable(timeline: Timeline): void {
    const table = this.part<HTMLTableElement>("table");
    const caption = table.querySelector("caption");
    table.replaceChildren(...(caption ? [caption] : []));
    const head = table.createTHead().insertRow();
    for (const heading of ["Split", ...ROLES.map((role: Role) => ROLE_NAMES[role]), "Stretches"]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = heading;
      head.append(cell);
    }
    const body = table.createTBody();
    const names: Record<TimelineProtocol, string> = { shuffled: "Shuffled", blocked: "Blocked", later: "Later in the drive" };
    for (const protocol of PROTOCOLS) {
      const row = body.insertRow();
      const th = document.createElement("th");
      th.scope = "row";
      th.textContent = names[protocol];
      row.append(th);
      const sizes = timeline.protocols[protocol].sizes;
      for (const role of ROLES) row.insertCell().textContent = fmtCount(sizes[role]);
      row.insertCell().textContent =
        protocol === "shuffled" ? "every frame dealt at random" : fmtCount(timeline.protocols[protocol].segments.filter((s) => s.role !== "dropped").length);
    }
  }
}
