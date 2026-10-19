import { describe, expect, it } from "vitest";

import page from "../../index.html?raw";
import { injectCockpit } from "../cockpit/markup.ts";
import { injectShell } from "../shell.ts";
import accuracySource from "./accuracy.ts?raw";
import headlineSource from "./headline.ts?raw";
import howSource from "./how.ts?raw";
import {
  ACCURACY_ROOT,
  HOW_ROOT,
  RESULTS_ROOT,
  accuracyHtml,
  howHtml,
  injectSections,
  resultsHtml,
} from "./markup.ts";

/** The text a visitor reads in some markup: no tags, no attribute values,
 * no drawings, no scripts. */
const visibleText = (html: string) =>
  html
    .replace(/<script\b[\s\S]*?<\/script>/g, " ")
    .replace(/<svg\b[\s\S]*?<\/svg>/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const SECTIONS = [
  ["How it works", howHtml, HOW_ROOT],
  ["the accuracy figure", accuracyHtml, ACCURACY_ROOT],
  ["the Results view", resultsHtml, RESULTS_ROOT],
] as const;

describe("the sections' markup", () => {
  it.each(SECTIONS)("%s prints no number of its own: every figure comes from the data", (_, html) => {
    const text = visibleText(html());
    expect(text.length).toBeGreaterThan(150);
    expect(text).not.toMatch(/\d/);
  });

  it("leaves no number anywhere in the body of the page as served", () => {
    const built = injectSections(injectCockpit(injectShell(page)));
    const body = built.slice(built.indexOf("<body>"));
    expect(visibleText(body)).not.toMatch(/\d/);
  });

  it("holds every element the sections' code looks up", () => {
    const lookups: Array<[string, string, string]> = [
      [howSource, "how", howHtml()],
      [accuracySource, "accuracy", accuracyHtml()],
      [headlineSource, "results", resultsHtml()],
    ];
    for (const [source, attribute, html] of lookups) {
      const names = [...source.matchAll(/\.part(?:<[^>]+>)?\("([\w-]+)"\)/g)].map((m) => m[1]);
      expect(names.length, attribute).toBeGreaterThan(3);
      for (const name of names) {
        expect(html, `${attribute}: ${name}`).toContain(`data-${attribute}="${name}"`);
      }
    }
    const figures = [...headlineSource.matchAll(/\bname: "([\w-]+)"/g)].map((m) => m[1]);
    expect(figures.length).toBeGreaterThanOrEqual(3);
    for (const name of figures) {
      expect(resultsHtml(), name).toContain(`data-count="${name}"`);
    }
  });

  it("links the results and the model card from the demo, and the data file from the results", () => {
    expect(accuracyHtml()).toMatch(/<a\b[^>]*href="#results"/);
    expect(accuracyHtml()).toMatch(/<a\b[^>]*href="model-card\.md"/);
    expect(resultsHtml()).toMatch(/<a\b[^>]*href="model-card\.md"/);
    expect(resultsHtml()).toMatch(/<a\b[^>]*href="data\/results\.json"/);
  });

  it("uses headings below the page's one h1", () => {
    for (const [name, html] of SECTIONS) {
      expect(html(), name).not.toMatch(/<h1\b/);
      expect(html(), name).toMatch(/<h2\b/);
    }
  });

  it("keeps the step-by-step pause control and the angle out of sight until there is something to show", () => {
    expect(howHtml()).toMatch(/<button\b[^>]*data-how="toggle"[^>]*\bhidden\b/);
    expect(howHtml()).toMatch(/<p\b[^>]*data-how="readout"[^>]*\bhidden\b/);
  });
});

describe("the template the sections are written into", () => {
  it("has exactly one empty element for each section, in the views they belong to", () => {
    for (const [, , root] of SECTIONS) {
      expect(page.split(root), root).toHaveLength(2);
    }
    const demo = page.slice(page.indexOf('data-view="demo"'), page.indexOf('data-view="results"'));
    expect(demo).toContain(HOW_ROOT);
    expect(demo).toContain(ACCURACY_ROOT);
    expect(page.slice(page.indexOf('data-view="results"'))).toContain(RESULTS_ROOT);
  });

  it("serves the results view hidden: the demo is the page until the router says otherwise", () => {
    expect(page).toMatch(/<div class="view" data-view="results" hidden>/);
    expect(page).toMatch(/<div class="view" data-view="demo">/);
  });

  it("gains exactly the sections when the build writes them in", () => {
    const built = injectSections(page);
    let rest = built;
    for (const [, html] of SECTIONS) {
      expect(built).toContain(html());
      rest = rest.replace(html(), "");
    }
    expect(rest).toBe(page);
  });

  it("refuses a template without a place for a section, or with two", () => {
    expect(() => injectSections(page.replace(HOW_ROOT, ""))).toThrow(/How it works/);
    expect(() => injectSections(page.replace(RESULTS_ROOT, RESULTS_ROOT + RESULTS_ROOT))).toThrow(/results/);
  });
});
