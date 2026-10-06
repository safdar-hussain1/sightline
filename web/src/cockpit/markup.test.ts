import { describe, expect, it } from "vitest";

import page from "../../index.html?raw";
import controller from "./cockpit.ts?raw";
import { COCKPIT_ROOT, cockpitHtml, injectCockpit } from "./markup.ts";

/** The text a visitor reads in some markup: no tags, no attribute values. */
const visibleText = (html: string) =>
  html
    .replace(/<svg\b[\s\S]*?<\/svg>/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();

describe("the cockpit's markup", () => {
  it("prints no number of its own: every figure on screen comes from the data", () => {
    const text = visibleText(cockpitHtml());
    expect(text.length).toBeGreaterThan(100);
    expect(text).not.toMatch(/\d/);
  });

  it("holds every element the cockpit's code looks up", () => {
    const names = [...controller.matchAll(/find(?:<[^>]+>)?\("([\w-]+)"\)/g)].map((m) => m[1]);
    expect(names.length).toBeGreaterThan(15);
    for (const name of names) {
      expect(cockpitHtml(), name).toContain(`data-cockpit="${name}"`);
    }
  });

  it("starts with its controls disabled, until the model and the clip are ready", () => {
    const html = cockpitHtml();
    for (const name of ["play", "back", "forward"]) {
      expect(html).toMatch(new RegExp(`data-cockpit="${name}"[^>]*\\bdisabled\\b`));
    }
    expect(html).toMatch(/data-cockpit="strip"[^>]*aria-disabled="true"/);
    // A slider has a value even before there is a clip to scrub.
    for (const attribute of ["aria-valuemin", "aria-valuemax", "aria-valuenow", "aria-valuetext"]) {
      expect(html).toMatch(new RegExp(`data-cockpit="strip"[^>]*${attribute}="`));
    }
    expect(html).toMatch(/data-cockpit="clips"[^>]*\bdisabled\b/);
    expect(html).toMatch(/data-cockpit="models"[^>]*\bdisabled\b/);
  });

  it("keeps the video silent and out of the tab order", () => {
    expect(cockpitHtml()).toMatch(/<video\b[^>]*\bmuted\b[^>]*\bplaysinline\b[^>]*aria-hidden="true"[^>]*tabindex="-1"/);
  });
});

describe("the template the cockpit is written into", () => {
  it("has exactly one empty cockpit section", () => {
    expect(page.split(COCKPIT_ROOT)).toHaveLength(2);
  });

  it("gains exactly the cockpit when the build writes it in", () => {
    const built = injectCockpit(page);
    expect(built).toContain(cockpitHtml());
    expect(built.replace(cockpitHtml(), "")).toBe(page);
  });

  it("refuses a template without a place for the cockpit, or with two", () => {
    expect(() => injectCockpit("<main></main>")).toThrow(/cockpit/);
    expect(() => injectCockpit(COCKPIT_ROOT + COCKPIT_ROOT)).toThrow(/cockpit/);
  });
});
