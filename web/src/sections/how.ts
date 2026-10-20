/**
 * How it works: one of the model's answers, step by step, drawn from the clip
 * the cockpit has loaded -- the frames it is shown, the network they pass
 * through, and the angle and heat that come out.
 *
 * The section reads the cockpit (live.ts); it never runs the model itself.
 * As soon as the clip's frames are decoded it shows the first of them, and
 * each answer once the model has given it, with no interaction. While the
 * section is on screen it walks through the clip, one answer every few
 * seconds: while the footage plays it takes the frame on screen, and while
 * the footage is paused it moves along the clip on its own. Out of sight it
 * draws only a new clip's first frame and that frame's answer, not every
 * frame the cockpit shows, and catches up as it comes back into view.
 * Each answer is animated in the order it is computed -- the frames slide
 * into their stack, a pulse runs through the network, then the wheel turns
 * and the heat comes up. The walk can be paused; under reduced motion it
 * starts paused and every change is instant. Until the clip's frames are in,
 * and whenever the cockpit has no clip to give, the steps show their
 * drawings.
 */

import type { LiveCockpit, LiveFrame } from "../cockpit/live.ts";
import { heatImageFromBytes } from "../cockpit/saliency.ts";
import { renderWheel } from "../cockpit/wheel.ts";
import { frameTiming, type Gaps } from "../data/timing.ts";
import type { Clip } from "../data/types.ts";
import { steerParts } from "../format.ts";
import type { ViewName } from "../router.ts";
import { framesAges, framesText, framesTitle, speedText } from "./copy.ts";
import { EASE_OUT, prefersReducedMotion } from "./motion.ts";

/** One answer, from the frames arriving to the heat settling, and a rest. */
const CYCLE_MS = 3800;
/** How often the section looks at the cockpit, ms. */
const POLL_MS = 250;
/** Frames between two answers when the section walks the clip on its own:
 * about half a second of road at the clips' frame rate, so each answer is a
 * new place on the road. */
const STRIDE = 9;
/** The heat's opacity where the model looked hardest (the cockpit's own). */
const HEAT_ALPHA = 0.85;

/** When each part of an answer's animation starts, ms into the cycle. */
const BEATS = { frames: 0, wire1: 600, graph: 860, back: 1760, wire2: 1900, answer: 2200 } as const;

export class HowItWorks {
  /** The section in the page, or null on a page without it. */
  static mount(): HowItWorks | null {
    const section = document.getElementById("how-it-works");
    return section ? new HowItWorks(section) : null;
  }

  private readonly section: HTMLElement;
  private cockpit: LiveCockpit | null = null;
  /** The clip and frame drawn, and whether the model's answer was drawn with it. */
  private clipId: string | null = null;
  private k = -1;
  private answered = false;
  /** The walk through the clip plays (the visitor can pause it). */
  private walking: boolean;
  private inView = false;
  private onView = true;
  private lastAnswer = 0;
  private timers: number[] = [];
  private animations: Animation[] = [];
  /** Each drive's measured frame interval, from results.json: until it is
   * known, no clip's frames are timed. */
  private gaps: Readonly<Record<string, Gaps | null>> = {};
  private readonly scratch: HTMLCanvasElement;

  constructor(
    section: HTMLElement,
    private readonly reduced: () => boolean = prefersReducedMotion,
  ) {
    this.section = section;
    this.walking = !reduced();
    this.scratch = document.createElement("canvas");
    this.scratch.width = 200;
    this.scratch.height = 66;
    this.part<HTMLButtonElement>("toggle").addEventListener("click", () => this.toggle());
    this.showToggle();
    if (typeof IntersectionObserver === "function") {
      new IntersectionObserver(
        (entries) => {
          const was = this.inView;
          this.inView = entries.some((entry) => entry.isIntersecting);
          if (this.inView && !was) {
            // Coming into sight: catch up with the cockpit now, and the next
            // answer is not far off.
            this.lastAnswer = Math.min(this.lastAnswer, performance.now() - CYCLE_MS + 700);
            this.poll();
          }
        },
        { threshold: 0.2 },
      ).observe(this.part("steps"));
    }
  }

  private part<T extends HTMLElement = HTMLElement>(name: string): T {
    const element = this.section.querySelector<T>(`[data-how="${name}"]`);
    if (!element) throw new Error(`How it works has no ${name}`);
    return element;
  }

  /** Start reading `cockpit`; null when the demo could not start, and the
   * steps keep their drawings. */
  attach(cockpit: LiveCockpit | null): void {
    this.cockpit = cockpit;
    if (!cockpit) return;
    this.poll();
    window.setInterval(() => this.poll(), POLL_MS);
  }

  /** Each drive's measured gaps (`driveGaps` of results.json), so the first
   * step can say how far apart in time the drive's frames were taken. */
  setGaps(gaps: Readonly<Record<string, Gaps | null>>): void {
    this.gaps = gaps;
    const clip = this.clipId === null ? null : this.cockpit?.status().clip;
    if (clip && clip.id === this.clipId) this.describe(clip);
  }

  /** The router switched views: the walk stops behind the other view, and
   * catches up at once on coming back. */
  viewShown(view: ViewName): void {
    this.onView = view === "demo";
    if (this.onView) this.poll();
  }

  // -- Following the cockpit ------------------------------------------------------

  private poll(): void {
    const cockpit = this.cockpit;
    if (!cockpit || !this.onView || document.visibilityState === "hidden") return;
    const status = cockpit.status();
    const clip = status.clip;
    if (status.state === "unavailable" || clip === null) {
      // Between clips, or with no demo: the drawings, never another clip's
      // frames or answer.
      if (this.clipId !== null) this.rest();
      return;
    }
    if (clip.id !== this.clipId) {
      // A clip's frames have just been decoded: its frame on screen, at once.
      this.settle();
      this.clipId = clip.id;
      this.describe(clip);
      this.draw(status.frame, false);
      this.lastAnswer = performance.now();
      return;
    }
    // The cost of a frame here, measured as the model runs, can arrive while
    // the steps rest.
    const speed = speedText(cockpit.msPerFrame());
    if (this.part("speed").textContent !== speed) this.part("speed").textContent = speed;
    if (!this.inView) {
      // Out of sight the frame drawn stays; only its answer, once the model
      // has given it, is filled in.
      if (!this.answered && this.timers.length === 0) this.draw(this.k, false);
      return;
    }
    const now = performance.now();
    if (this.walking && now - this.lastAnswer >= CYCLE_MS) {
      this.lastAnswer = now;
      this.draw(this.nextFrame(status.playing ? status.frame : null, clip), !this.reduced());
      return;
    }
    if (this.timers.length > 0) return; // an answer is still being animated
    if (!this.walking) {
      // Not walking: show what the cockpit shows.
      if (status.frame !== this.k || !this.answered) this.draw(status.frame, false);
    } else if (!this.answered) {
      this.draw(this.k, false); // its answer has arrived since
    }
  }

  /** Back to the drawings: nothing of the last clip -- frames, angle, turned
   * wheel -- stays behind them. */
  private rest(): void {
    this.settle();
    this.clipId = null;
    this.k = -1;
    this.answered = false;
    for (const name of ["frame-0", "frame-1", "heat"]) {
      const canvas = this.part<HTMLCanvasElement>(name);
      canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    }
    const wheel = this.part("wheel").querySelector("svg");
    if (wheel) renderWheel(wheel, 0, 0);
    this.part("readout").hidden = true;
    this.live(false);
  }

  /** The next frame to answer: the frame on screen while the footage plays,
   * otherwise `STRIDE` frames on from the last, at a frame the model has run. */
  private nextFrame(onScreen: number | null, clip: Clip): number {
    if (onScreen !== null) return onScreen;
    const cockpit = this.cockpit as LiveCockpit;
    for (let step = 0; step < clip.n; step += 1) {
      const k = (Math.max(this.k, 0) + STRIDE + step) % clip.n;
      if (cockpit.frameAt(k)?.angle !== null) return k;
    }
    return (Math.max(this.k, 0) + STRIDE) % clip.n;
  }

  // -- Drawing ----------------------------------------------------------------------

  private live(on: boolean): void {
    this.part("steps").dataset["live"] = String(on);
    this.showToggle();
  }

  private showToggle(): void {
    const toggle = this.part<HTMLButtonElement>("toggle");
    toggle.hidden = this.part("steps").dataset["live"] !== "true";
    toggle.dataset["playing"] = String(this.walking);
    this.part("toggle-label").textContent = this.walking ? "Pause" : "Play";
  }

  private toggle(): void {
    this.walking = !this.walking;
    if (this.walking) this.lastAnswer = performance.now() - CYCLE_MS + 300;
    else this.settle();
    this.showToggle();
  }

  /** The words that depend on the clip: how many frames, how far apart. */
  private describe(clip: Clip): void {
    const timing = frameTiming(this.gaps, clip.drive);
    this.part("frames-title").textContent = framesTitle(clip, timing);
    this.part("frames-text").textContent = framesText(clip, timing);
    const ages = framesAges(clip, timing);
    this.part("age-0").textContent = ages[0] ?? "";
    this.part("age-1").textContent = ages[ages.length - 1] ?? "";
  }

  private draw(k: number, animate: boolean): void {
    const frame = this.cockpit?.frameAt(k);
    if (!frame) return;
    this.settle();
    this.k = k;
    this.answered = frame.angle !== null;
    this.part("speed").textContent = speedText(this.cockpit?.msPerFrame() ?? null);
    this.paintInputs(frame);
    this.live(true);
    if (animate) {
      this.choreograph(frame);
    } else {
      this.paintAnswer(frame);
    }
  }

  private paintInputs(frame: LiveFrame): void {
    const current = frame.inputs[0];
    const oldest = frame.inputs[frame.inputs.length - 1];
    if (current) paint(this.part<HTMLCanvasElement>("frame-0"), current);
    if (oldest) paint(this.part<HTMLCanvasElement>("frame-1"), oldest);
  }

  /** The third step: the frame with the heat over it, the wheel and the angle. */
  private paintAnswer(frame: LiveFrame): void {
    const canvas = this.part<HTMLCanvasElement>("heat");
    const current = frame.inputs[0];
    if (current) paint(canvas, current);
    if (frame.heat) {
      const scratch = this.scratch.getContext("2d");
      scratch?.putImageData(heatImageFromBytes(frame.heat, 200, 66, HEAT_ALPHA), 0, 0);
      canvas.getContext("2d")?.drawImage(this.scratch, 0, 0);
    }
    const wheel = this.part("wheel").querySelector("svg");
    const readout = this.part("readout");
    if (frame.angle === null) {
      // No answer for this frame yet: the wheel waits at rest, without a number.
      if (wheel) renderWheel(wheel, 0, 0);
      readout.hidden = true;
      return;
    }
    if (wheel) renderWheel(wheel, frame.angle, frame.angle);
    const { size, side } = steerParts(frame.angle);
    this.part("angle").textContent = size;
    this.part("side").textContent = side;
    readout.hidden = false;
  }

  // -- One answer, animated ---------------------------------------------------------

  private later(ms: number, run: () => void): void {
    const timer = window.setTimeout(() => {
      this.timers = this.timers.filter((t) => t !== timer);
      run();
    }, ms);
    this.timers.push(timer);
  }

  private play(el: Element | null, keyframes: Keyframe[], options: KeyframeAnimationOptions): void {
    const animation = el?.animate?.(keyframes, { easing: EASE_OUT, ...options });
    if (animation) this.animations.push(animation);
  }

  /** The new frames slide into their stack, a pulse runs to the network and
   * through it, and the answer comes out at the end: the order the browser
   * computed it in. */
  private choreograph(frame: LiveFrame): void {
    const back = this.section.querySelector(".how-frame--back");
    const front = this.section.querySelector(".how-frame--front");
    this.play(back, [{ transform: "translate(-1.75rem, 0.5rem)", opacity: 0 }, { transform: "none", opacity: 1 }], {
      duration: 640,
      fill: "backwards",
    });
    this.play(front, [{ transform: "translate(-2.5rem, 0.75rem)", opacity: 0 }, { transform: "none", opacity: 1 }], {
      duration: 680,
      delay: 120,
      fill: "backwards",
    });

    const wires = [...this.section.querySelectorAll<HTMLElement>(".how-step__pulse")];
    this.pulse(wires[0], BEATS.wire1);
    const glows = [...this.section.querySelectorAll<SVGElement>(".graph__glow")];
    glows.forEach((glow, i) =>
      this.play(glow, [{ opacity: 0 }, { opacity: 0.9, offset: 0.3 }, { opacity: 0 }], {
        duration: 760,
        delay: BEATS.graph + i * 110,
        easing: "ease-in-out",
      }),
    );
    this.play(this.section.querySelector(".graph__back"), [{ strokeDashoffset: 1 }, { strokeDashoffset: 0 }], {
      duration: 700,
      delay: BEATS.back,
      fill: "backwards",
    });
    this.pulse(wires[1], BEATS.wire2);

    this.later(BEATS.answer, () => {
      this.paintAnswer(frame);
      this.play(this.part("heat"), [{ opacity: 0.2 }, { opacity: 1 }], { duration: 650 });
      this.play(this.part("readout"), [{ opacity: 0, transform: "translateY(0.25rem)" }, { opacity: 1, transform: "none" }], {
        duration: 450,
      });
    });
  }

  /** A pulse along the joint between two steps, across or down it. */
  private pulse(pulse: HTMLElement | undefined, delay: number): void {
    const wire = pulse?.parentElement;
    if (!pulse || !wire) return;
    const across = wire.offsetWidth >= wire.offsetHeight;
    const run = across ? `translateX(${wire.offsetWidth}px)` : `translateY(${wire.offsetHeight}px)`;
    const start = across ? "translateX(-100%)" : "translateY(-100%)";
    this.play(pulse, [{ transform: start, opacity: 0 }, { opacity: 1, offset: 0.25 }, { transform: run, opacity: 0 }], {
      duration: 520,
      delay,
      easing: "ease-in-out",
    });
  }

  /** Stop the answer being animated, and draw it as it ends. */
  private settle(): void {
    for (const timer of this.timers) window.clearTimeout(timer);
    this.timers = [];
    for (const animation of this.animations) animation.cancel();
    this.animations = [];
    const frame = this.k >= 0 ? this.cockpit?.frameAt(this.k) : null;
    if (frame && this.clipId === frame.clip.id) this.paintAnswer(frame);
  }
}

/** Draw a frame of RGB bytes (200 x 66) on a canvas of its size. */
function paint(canvas: HTMLCanvasElement, rgb: Uint8Array): void {
  const context = canvas.getContext("2d");
  if (!context) return;
  const image = context.createImageData(canvas.width, canvas.height);
  const data = image.data;
  const pixels = Math.min(rgb.length / 3, data.length / 4);
  for (let p = 0, from = 0, to = 0; p < pixels; p += 1, from += 3, to += 4) {
    data[to] = rgb[from] as number;
    data[to + 1] = rgb[from + 1] as number;
    data[to + 2] = rgb[from + 2] as number;
    data[to + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}
