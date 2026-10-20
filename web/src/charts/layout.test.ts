import { describe, expect, it } from "vitest";

import { readLadder, readPaired } from "../sections/figures.ts";
import { patch, resultsFixture } from "../sections/fixture.ts";
import { barsLayout } from "./bars.ts";
import { LADDER_GRID, ladderLayout, lerpLadder } from "./ladder.ts";
import { SLOPE_NARROW, lerpSlope, nearestLine, slopeLayout, tickDecimals } from "./slope.ts";

const paired = () => readPaired(resultsFixture());
const ladder = () => readLadder(resultsFixture());

describe("the slope chart's layout", () => {
  it("draws each seed from its shuffled error on the left to its held-back error on the right, higher meaning worse", () => {
    const { sightlinenet, pilotnet } = paired();
    const layout = slopeLayout(sightlinenet, [sightlinenet, pilotnet], 600);
    expect(layout.lines.map((line) => line.seed)).toEqual([0, 1, 2]);
    for (const line of layout.lines) {
      expect(line.x1).toBe(layout.x.shuffled);
      expect(line.x2).toBe(layout.x.blocked);
      expect(line.y2).toBeLessThan(line.y1);
    }
    expect(layout.x.shuffled).toBeLessThan(layout.x.blocked);
  });

  it("puts both networks on one scale, from zero to a round figure above every value", () => {
    const { sightlinenet, pilotnet } = paired();
    const both = [sightlinenet, pilotnet];
    const a = slopeLayout(sightlinenet, both, 600);
    const b = slopeLayout(pilotnet, both, 600);
    expect(a.ticks).toEqual(b.ticks);
    expect(a.ticks.map((tick) => tick.text)).toEqual(["0°", "5°", "10°", "15°", "20°"]);
    expect(a.ticks[0]?.y).toBe(a.plot.bottom);
    expect(a.ticks.at(-1)?.y).toBe(a.plot.top);
  });

  it("labels the means and the mean ratio from the data, and steering straight as its range over seeds", () => {
    const layout = slopeLayout(paired().sightlinenet, [paired().sightlinenet], 600);
    expect([layout.means.shuffled.text, layout.means.blocked.text, layout.ratio.text]).toEqual(["3.3°", "10.1°", "3.0×"]);
    expect(layout.straightLabel.text).toBe("Always steering straight: 16.2° to 18.8°");
    expect(layout.straight.map((line) => line.seed)).toEqual([0, 1, 2]);
  });

  it("keeps everything inside its width, wide and narrow", () => {
    for (const width of [320, 358, SLOPE_NARROW, 560, 900]) {
      const layout = slopeLayout(paired().pilotnet, [paired().sightlinenet, paired().pilotnet], width);
      expect(layout.narrow).toBe(width < SLOPE_NARROW);
      const xs = [layout.x.shuffled, layout.x.blocked, layout.means.shuffled.x, layout.means.blocked.x, layout.straightLabel.x];
      for (const x of xs) {
        expect(x, String(width)).toBeGreaterThan(0);
        expect(x, String(width)).toBeLessThanOrEqual(width);
      }
      const ys = [...layout.lines.flatMap((line) => [line.y1, line.y2]), ...layout.straight.map((line) => line.y), layout.straightLabel.y];
      for (const y of ys) {
        expect(y).toBeGreaterThan(0);
        expect(y).toBeLessThan(layout.height);
      }
    }
  });

  it("morphs from one network to the other: halfway is halfway, and the end is the other network", () => {
    const { sightlinenet, pilotnet } = paired();
    const both = [sightlinenet, pilotnet];
    const a = slopeLayout(sightlinenet, both, 600);
    const b = slopeLayout(pilotnet, both, 600);
    const half = lerpSlope(a, b, 0.5);
    expect(half.lines[0]?.y2).toBeCloseTo(((a.lines[0]?.y2 ?? 0) + (b.lines[0]?.y2 ?? 0)) / 2, 9);
    expect(half.means.blocked.value).toBeCloseTo((sightlinenet.blocked + pilotnet.blocked) / 2, 9);
    expect(half.means.blocked.text).toBe("13.1°");
    expect(lerpSlope(a, b, 1)).toEqual(b);
    expect(lerpSlope(a, b, 0).lines).toEqual(a.lines);
  });

  it("finds the seed line nearest a point", () => {
    const layout = slopeLayout(paired().sightlinenet, [paired().sightlinenet], 600);
    const second = layout.lines[1];
    expect(second).toBeDefined();
    const mid = { x: ((second?.x1 ?? 0) + (second?.x2 ?? 0)) / 2, y: ((second?.y1 ?? 0) + (second?.y2 ?? 0)) / 2 };
    expect(nearestLine(layout, mid.x, mid.y)?.index).toBe(1);
    expect(nearestLine(layout, mid.x, mid.y)?.distance).toBeCloseTo(0, 9);
  });

  it("prints ticks with the decimals their step needs", () => {
    expect(tickDecimals([0, 5, 10])).toBe(0);
    expect(tickDecimals([0, 2.5, 5])).toBe(1);
    expect(tickDecimals([-1, -0.5, 0, 0.5, 1])).toBe(1);
  });
});

describe("the ladder's layout", () => {
  it("stands every bar of the skill view on one zero line, steering straight itself, in every panel", () => {
    const layout = ladderLayout(ladder(), "skill", 1000, 3);
    const zero = layout.baselines[0]?.y;
    for (const line of layout.baselines) expect(line.y).toBe(zero);
    for (const bar of layout.bars) expect(bar.base).toBe(zero);
    expect(layout.ticks.map((tick) => tick.text)).toEqual(["−100%", "−50%", "0%", "50%", "100%"]);
  });

  it("draws a network worse than steering straight below the zero line, labelled below its whisker", () => {
    const layout = ladderLayout(ladder(), "skill", 1000, 3);
    const later = layout.bars.find((bar) => bar.model === "pilotnet" && bar.rung === 2);
    expect(later?.text).toBe("−45%");
    expect(later?.end).toBeGreaterThan(later?.base ?? Infinity);
    const whisker = layout.whiskers.find((w) => w.model === "pilotnet" && w.rung === 2);
    expect(later?.labelY).toBeGreaterThan(whisker?.y2 ?? Infinity);
    const better = layout.bars.find((bar) => bar.model === "sightlinenet" && bar.rung === 2);
    expect(better?.end).toBeLessThan(better?.base ?? -Infinity);
    expect(better?.labelY).toBeLessThan(layout.whiskers.find((w) => w.model === "sightlinenet" && w.rung === 2)?.y1 ?? -Infinity);
  });

  it("says in words where a network was not trained this way, and draws no bar, whisker or zero there", () => {
    for (const view of ["skill", "error"] as const) {
      const layout = ladderLayout(ladder(), view, 1000, 0);
      expect(layout.missing.map(({ model, rung }) => `${model}/${rung}`)).toEqual(["sightlinenet-1f/0", "sightlinenet-1f/2"]);
      const drawn = new Set(layout.bars.map(({ model, rung }) => `${model}/${rung}`));
      expect(drawn.has("sightlinenet-1f/0")).toBe(false);
      expect(drawn.has("sightlinenet-1f/2")).toBe(false);
      expect(layout.bars).toHaveLength(10);
    }
  });

  it("keeps each network in the same place in every panel", () => {
    const layout = ladderLayout(ladder(), "skill", 1000, 0);
    for (const model of ["sightlinenet", "pilotnet"] as const) {
      const offsets = layout.bars.filter((bar) => bar.model === model).map((bar) => bar.x - (layout.panels[bar.rung]?.x ?? 0));
      for (const offset of offsets) expect(offset).toBeCloseTo(offsets[0] ?? 0, 9);
    }
  });

  it("stands the error view's bars on zero degrees, with each rung's own steering straight, at its own height", () => {
    const layout = ladderLayout(ladder(), "error", 1000, 0);
    const bottom = layout.ticks[0]?.y;
    for (const bar of layout.bars) expect(bar.base).toBe(bottom);
    expect(layout.baselines.map((line) => line.text)).toEqual(["13.1°", "17.5°", "9.4°", "15.5°"]);
    const heights = layout.baselines.map((line) => line.y);
    expect(new Set(heights).size).toBe(4);
    expect(layout.dots.filter((dot) => dot.model === "pilotnet" && dot.rung === 2)).toHaveLength(3);
    expect(layout.whiskers).toEqual([]);
  });

  it("stands the panels in a row when wide and two by two when narrow, naming the rung in the panel only then", () => {
    const wide = ladderLayout(ladder(), "skill", LADDER_GRID, 1);
    expect(wide.grid).toBe(false);
    expect(new Set(wide.panels.map((panel) => panel.y)).size).toBe(1);
    expect(wide.panels.every((panel) => panel.title === null)).toBe(true);
    const narrow = ladderLayout(ladder(), "skill", LADDER_GRID - 1, 1);
    expect(narrow.grid).toBe(true);
    expect(new Set(narrow.panels.map((panel) => panel.y)).size).toBe(2);
    expect(narrow.panels.map((panel) => panel.title)).toEqual(["shuffled", "blocked", "later in the drive", "another day"]);
    expect(narrow.spotlight.x).toBe((narrow.panels[1]?.x ?? 0) + 2);
  });

  it("keeps every mark and label inside the chart at phone and desktop widths", () => {
    for (const width of [300, 358, 520, 700, 1100]) {
      for (const view of ["skill", "error"] as const) {
        const layout = ladderLayout(ladder(), view, width, 2);
        for (const bar of layout.bars) {
          expect(bar.x).toBeGreaterThanOrEqual(0);
          expect(bar.x + bar.w).toBeLessThanOrEqual(width);
          expect(bar.labelY).toBeGreaterThan(0);
          expect(bar.labelY).toBeLessThan(layout.height);
        }
        for (const note of layout.missing) expect(note.x + 60).toBeLessThanOrEqual(width);
      }
    }
  });

  it("morphs the bars and the always-straight lines from one view to the other", () => {
    const skill = ladderLayout(ladder(), "skill", 1000, 0);
    const error = ladderLayout(ladder(), "error", 1000, 0);
    expect(lerpLadder(skill, error, 1).bars).toEqual(error.bars);
    expect(lerpLadder(skill, error, 1).baselines).toEqual(error.baselines);
    const half = lerpLadder(skill, error, 0.5);
    expect(half.baselines[2]?.y).toBeCloseTo(((skill.baselines[2]?.y ?? 0) + (error.baselines[2]?.y ?? 0)) / 2, 9);
    expect(half.bars[0]?.end).toBeCloseTo(((skill.bars[0]?.end ?? 0) + (error.bars[0]?.end ?? 0)) / 2, 9);
  });

  it("follows the data: a network that beats steering straight on a rung stands above the line", () => {
    const json = patch(resultsFixture(), "groups.pilotnet/forward.test_skill", { mean: 0.12, sd: 0.02 });
    const layout = ladderLayout(readLadder(json), "skill", 1000, 0);
    const bar = layout.bars.find((b) => b.model === "pilotnet" && b.rung === 2);
    expect(bar?.end).toBeLessThan(bar?.base ?? 0);
    expect(bar?.text).toBe("12%");
  });
});

describe("the cards' bar layout", () => {
  const rows = [
    { title: "Another day", bars: [{ series: "sightlinenet", value: 0.31, text: "31%" }, { series: "oneframe", value: 0.02, text: "2%" }] },
    { title: "Held back", bars: [{ series: "sightlinenet", value: 0.41, text: "41%" }, { series: "pilotnet", value: -0.24, text: "−24%" }] },
  ];

  it("stands every bar on one zero line, a negative one running left of it and labelled at its left", () => {
    const layout = barsLayout(rows, 340, 48);
    const zero = layout.zero[0]?.x;
    for (const segment of layout.zero) expect(segment.x).toBe(zero);
    for (const bar of layout.bars) expect(bar.x0).toBe(zero);
    const worse = layout.bars.find((bar) => bar.value < 0);
    expect(worse?.x1).toBeLessThan(zero ?? 0);
    expect(worse?.anchor).toBe("end");
    expect(worse?.labelX).toBeLessThan(worse?.x1 ?? 0);
    const better = layout.bars.find((bar) => bar.value === 0.41);
    expect(better?.anchor).toBe("start");
    expect(better?.labelX).toBeGreaterThan(better?.x1 ?? Infinity);
  });

  it("titles each row above its bars and leaves room for every label inside its width", () => {
    const layout = barsLayout(rows, 340, 48);
    expect(layout.titles.map((title) => title.text)).toEqual(["Another day", "Held back"]);
    layout.bars.forEach((bar) => {
      const title = layout.titles[bar.row];
      expect(bar.y).toBeGreaterThan(title?.y ?? Infinity);
      expect(Math.max(bar.x0, bar.x1, bar.labelX)).toBeLessThanOrEqual(340 - 48 + 6);
      expect(Math.min(bar.x0, bar.x1)).toBeGreaterThanOrEqual(0);
    });
    // The zero line runs behind the bars of a row, never through its title.
    layout.zero.forEach((segment, r) => expect(segment.y1).toBeGreaterThan(layout.titles[r]?.y ?? Infinity));
    expect(layout.height).toBeGreaterThan(layout.bars.at(-1)?.y ?? Infinity);
  });

  it("draws errors from zero degrees on a scale that ends at a round figure past the largest", () => {
    const layout = barsLayout([{ title: "Sharp", bars: [{ series: "straight", value: 78.4, text: "78.4°" }, { series: "sightlinenet", value: 36, text: "36.0°" }] }], 360, 48);
    const [straight, model] = layout.bars;
    expect(straight?.x0).toBe(2);
    expect((model?.x1 ?? 0) - 2).toBeCloseTo((((straight?.x1 ?? 0) - 2) * 36) / 78.4, 9);
  });
});
