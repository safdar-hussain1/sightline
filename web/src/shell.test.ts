import { describe, expect, it } from "vitest";

import page from "../index.html?raw";
import { SHELL_ROOT, injectShell, shellHtml } from "./shell.ts";
import { THEME_TOGGLE_ID } from "./theme.ts";

const H1 = /<h1\b[^>]*>([\s\S]*?)<\/h1>/g;

describe("the static shell", () => {
  it("carries the page's one h1, which names the product", () => {
    const headings = [...shellHtml().matchAll(H1)].map((match) => match[1]);
    expect(headings).toEqual(["Sightline"]);
  });

  it("holds the theme toggle the theme module wires up", () => {
    expect(shellHtml()).toContain(`id="${THEME_TOGGLE_ID}"`);
    expect(shellHtml()).toMatch(/<button\b[^>]*type="button"/);
  });

  it("explains the product in a sentence or two of plain text", () => {
    const lede = /<p class="lede">([\s\S]*?)<\/p>/.exec(shellHtml());
    expect(lede).not.toBeNull();
    const text = (lede?.[1] ?? "").replace(/\s+/g, " ").trim();
    expect(text).not.toMatch(/<|lorem/i);
    const sentences = text.split(/(?<=\.)\s/).length;
    expect(sentences).toBeGreaterThanOrEqual(1);
    expect(sentences).toBeLessThanOrEqual(2);
  });
});

describe("the template the shell is written into", () => {
  it("has exactly one empty masthead and no heading of its own", () => {
    expect(page.split(SHELL_ROOT)).toHaveLength(2);
    expect(page).not.toMatch(/<h1\b/);
  });

  it("gains exactly the shell when the build writes it in", () => {
    const built = injectShell(page);
    expect([...built.matchAll(H1)]).toHaveLength(1);
    expect(built).toContain(shellHtml());
    expect(built.replace(shellHtml(), "")).toBe(page);
  });

  it("refuses a template without a place for the shell", () => {
    expect(() => injectShell("<body></body>")).toThrow(/masthead/);
    expect(() => injectShell(SHELL_ROOT + SHELL_ROOT)).toThrow(/masthead/);
  });
});
