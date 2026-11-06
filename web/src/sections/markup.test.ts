import { describe, expect, it } from "vitest";

import page from "../../index.html?raw";
import { injectCockpit } from "../cockpit/markup.ts";
import { injectShell } from "../shell.ts";
import accuracySource from "./accuracy.ts?raw";
import cardsSource from "./cards.ts?raw";
import headlineSource from "./headline.ts?raw";
import howSource from "./how.ts?raw";
import ladderSource from "./ladder.ts?raw";
import pairedSource from "./paired.ts?raw";
import {
  ACCURACY_ROOT,
  HOW_ROOT,
  RESULTS_ROOT,
  accuracyHtml,
  howHtml,
} from "./markup.ts";
import { injectSections } from "./inject.ts";
import { OWNCLIP_ROOT, ownclipHtml } from "./ownclip-markup.ts";
import { resultsHtml } from "./results-markup.ts";
import { networkName } from "./names.ts";

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
  ["your own video", ownclipHtml, OWNCLIP_ROOT],
  ["the Results view", resultsHtml, RESULTS_ROOT],
] as const;

describe("the sections' markup", () => {
  it.each(SECTIONS)("%s prints no number of its own: every figure comes from the data", (_, html) => {
    const text = visibleText(html());
    expect(text.length).toBeGreaterThan(150);
    expect(text).not.toMatch(/\d/);
  });

  it("serves the Results view without a count typed in words: counts of trainings and frames come with the data", () => {
    const html = resultsHtml();
    const said = [visibleText(html), ...[...html.matchAll(/aria-label="([^"]*)"/g)].map((m) => m[1] ?? "")].join(" ");
    const counts = "two|three|four|five|six|seven|eight|nine|ten|eleven|twelve";
    expect(said).not.toMatch(new RegExp(`\\b(${counts})\\s+(trainings?|frames?|seeds?|runs?|models?|drives?)\\b`, "i"));
    expect(said).not.toMatch(/\b(half a turn|quarter turn|full turn|a second frame|a second drive|checked twice)\b/i);
  });

  it("heads the baselines' rulers with what they measure: the mean error", () => {
    expect(resultsHtml()).toMatch(/data-card="baselines"[\s\S]*?<p class="chart__title card__chart-title">Mean error, in degrees of steering-wheel angle<\/p>[\s\S]*?data-card="baselines-plot"/);
  });

  it("names the one-frame network by its key, as the data does", () => {
    expect(resultsHtml()).toContain(`</span>${networkName("sightlinenet-1f").name}</li>`);
  });

  it("leaves no number anywhere in the body of the page as served", () => {
    const built = injectSections(injectCockpit(injectShell(page)));
    const body = built.slice(built.indexOf("<body>"));
    expect(visibleText(body)).not.toMatch(/\d/);
    // Nor in a drawing's own text: the charts' placeholders say nothing at all.
    for (const [, inner] of body.matchAll(/<text\b[^>]*>([\s\S]*?)<\/text>/g)) expect(inner).not.toMatch(/\d/);
    for (const [, label] of body.matchAll(/aria-label="([^"]*)"/g)) expect(label).not.toMatch(/\d/);
  });

  it("serves every chart of the Results view as a quiet placeholder, never as an empty box", () => {
    const html = resultsHtml();
    for (const plot of ["paired", "ladder"]) {
      expect(html).toMatch(new RegExp(`data-${plot}="plot">\\s*<svg[^>]*class="[^"]*--waiting`));
    }
    expect(html.match(/data-card="(?:baselines|frames|regimes|recipe)-plot"><svg[^>]*card-chart--waiting/g)).toHaveLength(4);
  });

  it("holds every element the sections' code looks up", () => {
    const lookups: Array<[string, string, string]> = [
      [howSource, "how", howHtml()],
      [accuracySource, "accuracy", accuracyHtml()],
      [headlineSource, "results", resultsHtml()],
      [pairedSource, "paired", resultsHtml()],
      [ladderSource, "ladder", resultsHtml()],
      [cardsSource, "card", resultsHtml()],
    ];
    for (const [source, attribute, html] of lookups) {
      const names = [...source.matchAll(/\.part(?:<[^>]+>)?\("([\w-]+)"\)/g)].map((m) => m[1]);
      // The cards look up some parts by a name made from the card's.
      for (const card of ["baselines", "frames", "regimes", "recipe"]) if (attribute === "card") names.push(`${card}-plot`, `${card}-lead`);
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
    // the model card as the repository renders it: the copy beside the page is served as plain text
    const card = /<a\b[^>]*href="https:\/\/github\.com\/safdar-hussain1\/sightline\/blob\/main\/web\/public\/model-card\.md"/;
    expect(accuracyHtml()).toMatch(card);
    expect(resultsHtml()).toMatch(card);
    expect(accuracyHtml()).not.toMatch(/href="model-card\.md"/);
    expect(resultsHtml()).not.toMatch(/href="model-card\.md"/);
    expect(resultsHtml()).toMatch(/<a\b[^>]*href="data\/results\.json"/);
  });

  it("links every report the Results view's numbers come from, where it lives in the repository", () => {
    const links = [...resultsHtml().matchAll(/href="(https:\/\/github\.com\/safdar-hussain1\/sightline\/[^"]+)"/g)].map((m) => m[1]);
    const base = "https://github.com/safdar-hussain1/sightline";
    for (const file of ["reports/summary.json", "reports/leakage.json", "reports/legacy.json", "reports/export.json", "src/sightline/data/splits.py"]) {
      expect(links).toContain(`${base}/blob/main/${file}`);
    }
    expect(links).toContain(`${base}/tree/main/reports/runs`);
    // And the README, beside them.
    expect(resultsHtml()).toMatch(/<a\b[^>]*href="https:\/\/github\.com\/safdar-hussain1\/sightline#readme"/);
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

  it("asks for results.json in the head, so its download starts with the page's own", () => {
    const head = page.slice(0, page.indexOf("</head>"));
    expect(head).toMatch(/<link rel="preload" href="\/data\/results\.json" as="fetch" type="application\/json" crossorigin \/>/);
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
