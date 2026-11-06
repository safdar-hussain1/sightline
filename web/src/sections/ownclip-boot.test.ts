// @vitest-environment happy-dom
/**
 * The own-video section's code arrives when it is wanted, and nothing a
 * visitor does before then is lost.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SECTION_ID, bootOwnClip, type OwnClipCode } from "./ownclip-boot.ts";
import { OWNCLIP_ID, ownclipHtml } from "./ownclip-markup.ts";

/** An IntersectionObserver the test triggers by hand. */
const observers: Array<(entries: Array<{ isIntersecting: boolean }>) => void> = [];

function fakeCode() {
  const chosen: File[] = [];
  const mounted = vi.fn(() => ({ choose: async (file: File) => void chosen.push(file) }));
  const code: OwnClipCode = { OwnClip: { mount: mounted } };
  return { code, chosen, mounted };
}

function section(withMarkup = true): HTMLElement {
  const element = document.createElement("section");
  element.id = SECTION_ID;
  if (withMarkup) element.innerHTML = ownclipHtml();
  document.body.append(element);
  return element;
}

const none = Promise.resolve(null);
const results = new Promise<never>(() => undefined);
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

beforeEach(() => {
  document.body.innerHTML = "";
  observers.length = 0;
  vi.stubGlobal(
    "IntersectionObserver",
    class {
      constructor(callback: (entries: Array<{ isIntersecting: boolean }>) => void) {
        observers.push(callback);
      }
      observe() {}
      disconnect() {}
    },
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("the own-video section's code", () => {
  it("names the section the markup writes", () => {
    expect(SECTION_ID).toBe(OWNCLIP_ID);
  });

  it("is not asked for until the section nears the screen", async () => {
    section();
    const { code, mounted } = fakeCode();
    const load = vi.fn(async () => code);
    bootOwnClip(results, none, load);
    expect(load).not.toHaveBeenCalled();
    observers[0]?.([{ isIntersecting: false }]);
    expect(load).not.toHaveBeenCalled();
    observers[0]?.([{ isIntersecting: true }]);
    await settle();
    expect(load).toHaveBeenCalledTimes(1);
    expect(mounted).toHaveBeenCalledTimes(1);
  });

  it("is asked for when a visitor reaches the section's controls first", async () => {
    const root = section();
    const { code } = fakeCode();
    const load = vi.fn(async () => code);
    bootOwnClip(results, none, load);
    root.querySelector<HTMLInputElement>('[data-ownclip="file"]')?.dispatchEvent(new FocusEvent("focusin", { bubbles: true }));
    await settle();
    expect(load).toHaveBeenCalledTimes(1);
  });

  it("is asked for at once on a page served without the section's markup", async () => {
    section(false);
    const { code } = fakeCode();
    const load = vi.fn(async () => code);
    bootOwnClip(results, none, load);
    await settle();
    expect(load).toHaveBeenCalledTimes(1);
  });

  it("is handed a file chosen before it arrived", async () => {
    const root = section();
    const { code, chosen } = fakeCode();
    let arrive: (value: OwnClipCode) => void = () => undefined;
    const load = vi.fn(() => new Promise<OwnClipCode>((resolve) => (arrive = resolve)));
    bootOwnClip(results, none, load);
    const input = root.querySelector<HTMLInputElement>('[data-ownclip="file"]') as HTMLInputElement;
    const file = new File([new Uint8Array(8)], "early.mp4", { type: "video/mp4" });
    Object.defineProperty(input, "files", { value: [file], configurable: true });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    expect(load).toHaveBeenCalledTimes(1);
    arrive(code);
    await settle();
    expect(chosen).toEqual([file]);
  });

  it("keeps a file dropped before it arrived from opening in the tab, and is handed it", async () => {
    const root = section();
    const { code, chosen } = fakeCode();
    bootOwnClip(results, none, async () => code);
    const file = new File([new Uint8Array(8)], "dropped.webm", { type: "video/webm" });
    const drop = new Event("drop", { bubbles: true, cancelable: true }) as DragEvent;
    Object.defineProperty(drop, "dataTransfer", { value: { types: ["Files"], files: [file] } });
    root.querySelector(".drop")?.dispatchEvent(drop);
    expect(drop.defaultPrevented).toBe(true);
    await settle();
    expect(chosen).toEqual([file]);
  });

  it("does nothing on a page without the section", () => {
    const load = vi.fn();
    bootOwnClip(results, none, load);
    expect(load).not.toHaveBeenCalled();
  });
});
