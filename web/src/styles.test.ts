import { describe, expect, it } from "vitest";

import css from "./styles.css?raw";

/** The declarations of every rule whose selector list is exactly `selector`. */
function rules(selector: string): string[] {
  const out: string[] = [];
  const pattern = /([^{}]+)\{([^{}]*)\}/g;
  for (const match of css.matchAll(pattern)) {
    if ((match[1] ?? "").trim() === selector) out.push(match[2] ?? "");
  }
  return out;
}

describe("the cockpit's note", () => {
  it("states the floor of its slot once", () => {
    expect(css.match(/min-height:\s*6em/g) ?? []).toHaveLength(1);
  });

  it("stacks the note and the unseen copies of every note it can show in one cell, so the slot fits the longest", () => {
    expect(rules(".model-note > *").join(" ")).toMatch(/grid-area:\s*1\s*\/\s*1/);
    expect(rules(".model-note__sizer").join(" ")).toMatch(/visibility:\s*hidden/);
  });
});
