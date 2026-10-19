// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CANNOT_RUN, COCKPIT_SELECTOR, bootCockpit } from "./boot.ts";
import { COCKPIT_ID, mountCockpitMarkup } from "./markup.ts";

const status = () => document.querySelector<HTMLElement>('[data-cockpit="status"]') as HTMLElement;
const state = () => document.querySelector<HTMLElement>(".cockpit__console")?.dataset["state"];

beforeEach(() => {
  document.body.innerHTML = `<section class="cockpit" id="cockpit" aria-label="Live demo"></section>`;
  mountCockpitMarkup();
  vi.spyOn(console, "warn").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("bootCockpit", () => {
  it("says the demo cannot run when the cockpit's code cannot be fetched, instead of loading for ever", async () => {
    expect(state()).toBe("loading");
    await bootCockpit(() => Promise.reject(new TypeError("Failed to fetch dynamically imported module")));
    expect(state()).toBe("unavailable");
    expect(status().textContent).toBe(CANNOT_RUN);
    expect(status().hidden).toBe(false);
    expect(console.warn).toHaveBeenCalledOnce();
  });

  it("says the same when the cockpit cannot be built in this browser", async () => {
    await bootCockpit(async () => ({
      startCockpit: () => {
        throw new ReferenceError("IntersectionObserver is not defined");
      },
    }));
    expect(state()).toBe("unavailable");
    expect(status().textContent).toBe(CANNOT_RUN);
  });

  it("leaves the cockpit to speak for itself once it has started", async () => {
    let started = false;
    await bootCockpit(async () => ({ startCockpit: () => (started = true) }));
    expect(started).toBe(true);
    expect(state()).toBe("loading");
    expect(status().textContent).not.toBe(CANNOT_RUN);
    expect(console.warn).not.toHaveBeenCalled();
  });

  it("finds the cockpit by the id its markup gives it", () => {
    expect(COCKPIT_SELECTOR).toBe(`#${COCKPIT_ID}`);
  });
});
