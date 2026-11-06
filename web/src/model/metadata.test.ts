import { describe, expect, it } from "vitest";

import { modelMetadata } from "./metadata.ts";

/** Node runs these tests; the page's own TypeScript knows nothing of it. */
declare const process: {
  getBuiltinModule(id: "node:fs"): {
    readFileSync(path: URL): Uint8Array;
    readFileSync(path: URL, encoding: "utf8"): string;
  };
};
const fs = process.getBuiltinModule("node:fs");
const published = (path: string) => new URL(`../../public/${path}`, import.meta.url);
const bytesOf = (path: string) => new Uint8Array(fs.readFileSync(published(path)));
const jsonOf = (path: string) => JSON.parse(fs.readFileSync(published(path), "utf8")) as Record<string, any>;

// -- A protobuf writer, for files built here ------------------------------------------

function varint(value: number): number[] {
  const out: number[] = [];
  let rest = value;
  while (rest >= 0x80) {
    out.push((rest % 0x80) | 0x80);
    rest = Math.floor(rest / 0x80);
  }
  out.push(rest);
  return out;
}

const tag = (field: number, wire: number) => varint(field * 8 + wire);
const text = (value: string) => [...new TextEncoder().encode(value)];
const delimited = (field: number, payload: number[]) => [...tag(field, 2), ...varint(payload.length), ...payload];
const entry = (key: string, value: string) => delimited(14, [...delimited(1, text(key)), ...delimited(2, text(value))]);
const file = (...fields: number[][]) => new Uint8Array(fields.flat());

describe("modelMetadata on the published models", () => {
  const results = jsonOf("data/results.json");
  const clips = Object.values(results["assets"]["clips"] as Record<string, string>).map((path) => jsonOf(path));

  it("names the shipped model's run, its network and its frame lags", () => {
    const meta = modelMetadata(bytesOf(results["assets"]["models"]["sightline"]));
    expect(meta.get("run")).toBe(results["shipped"]["run"]);
    expect(meta.get("model")).toBe(results["shipped"]["model"]);
    expect(JSON.parse(meta.get("frame_lags") ?? "null")).toEqual([2]);
  });

  it("names the twin's own run", () => {
    const meta = modelMetadata(bytesOf(results["assets"]["models"]["shuffled"]));
    expect(meta.get("run")).toBe(results["twin"]["run"]);
    expect(meta.get("run")).not.toBe(results["shipped"]["run"]);
    expect(meta.get("model")).toBe(results["twin"]["model"]);
  });

  it("carries the lags every clip stacks", () => {
    expect(clips.length).toBeGreaterThan(0);
    for (const key of ["sightline", "shuffled"]) {
      const lags = JSON.parse(modelMetadata(bytesOf(results["assets"]["models"][key])).get("frame_lags") ?? "null");
      for (const clip of clips) expect(lags).toEqual(clip["frame_lags"]);
    }
  });

  it("holds exactly the three entries the export writes", () => {
    const meta = modelMetadata(bytesOf(results["assets"]["models"]["sightline"]));
    expect([...meta.keys()].sort()).toEqual(["frame_lags", "model", "run"]);
  });
});

describe("modelMetadata on files built here", () => {
  it("reads the entries wherever they sit, stepping over every other field by its wire type", () => {
    const bytes = file(
      [...tag(1, 0), ...varint(8)], // ir_version, a varint
      entry("run", "first"),
      delimited(7, [1, 2, 3, 4, 5]), // a graph: a message it never opens
      [...tag(9, 1), 1, 2, 3, 4, 5, 6, 7, 8], // a 64-bit field
      [...tag(10, 5), 1, 2, 3, 4], // a 32-bit field
      [...tag(11, 0), ...[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]], // a negative int64: ten bytes
      entry("frame_lags", "[2]"),
    );
    expect(Object.fromEntries(modelMetadata(bytes))).toEqual({ run: "first", frame_lags: "[2]" });
  });

  it("skips fields inside an entry that are neither its key nor its value", () => {
    const bytes = file(delimited(14, [...delimited(3, text("noise")), ...delimited(1, text("run")), ...tag(4, 0), 7, ...delimited(2, text("r"))]));
    expect(Object.fromEntries(modelMetadata(bytes))).toEqual({ run: "r" });
  });

  it("reads lengths that take several bytes to write", () => {
    const long = "x".repeat(20_000);
    const bytes = file(delimited(7, new Array<number>(300).fill(9)), entry("run", long));
    expect(modelMetadata(bytes).get("run")).toBe(long);
  });

  it("reads text as UTF-8", () => {
    expect(modelMetadata(file(entry("note", "drive 2018 — blocked"))).get("note")).toBe("drive 2018 — blocked");
  });

  it("finds no entries in a model without any", () => {
    expect(modelMetadata(file([...tag(1, 0), 8], delimited(7, [1, 2]))).size).toBe(0);
    expect(modelMetadata(new Uint8Array(0)).size).toBe(0);
  });

  it("refuses a file cut off inside a field", () => {
    const whole = file([...tag(1, 0), 8], delimited(7, [1, 2, 3, 4]), entry("run", "drive2018"));
    const cuts = [1, 3, 5, whole.length - 1, whole.length - 4];
    for (const cut of cuts) expect(() => modelMetadata(whole.subarray(0, cut)), `cut at ${cut}`).toThrow(/ends inside/);
  });

  it("refuses a length that runs past the end of the file", () => {
    expect(() => modelMetadata(file([...tag(7, 2), ...varint(1000), 1, 2]))).toThrow(/ends inside/);
  });

  it("refuses a number longer than protobuf allows", () => {
    expect(() => modelMetadata(file([...tag(1, 0), ...new Array<number>(11).fill(0x80), 0]))).toThrow(/ten bytes/);
  });

  it("refuses a key named twice: which value would count is not for the reader to guess", () => {
    expect(() => modelMetadata(file(entry("run", "a"), entry("run", "b")))).toThrow(/run.*twice/);
  });

  it("refuses wire types no ONNX file uses, and an entry that is not a message", () => {
    expect(() => modelMetadata(file([...tag(9, 3)]))).toThrow(/wire type 3/);
    expect(() => modelMetadata(file([...tag(14, 0), 1]))).toThrow(/metadata/);
    expect(() => modelMetadata(file(delimited(14, [...tag(1, 0), 1])))).toThrow(/key/);
    expect(() => modelMetadata(file([0]))).toThrow(/field 0/);
  });

  it("refuses text that is not UTF-8", () => {
    expect(() => modelMetadata(file(delimited(14, [...delimited(1, [0xff, 0xfe]), ...delimited(2, text("v"))])))).toThrow();
  });
});
