import { afterEach, describe, expect, it, vi } from "vitest";

import page from "../index.html?raw";
import { THEME_KEY, THEME_TOGGLE_ID, currentTheme, initTheme, storedTheme, toggleTheme } from "./theme.ts";

/** Just enough of a browser for the theme module: <html>'s attributes, one
 * toggle button, localStorage and the colour-scheme media query. */
function fakeBrowser({ systemDark = false, stored = null as string | null, storageThrows = false } = {}) {
  const rootAttributes = new Map<string, string>();
  const buttonAttributes = new Map<string, string>();
  const clickHandlers: Array<() => void> = [];
  const store = new Map<string, string>();
  if (stored !== null) store.set(THEME_KEY, stored);

  const button = {
    getAttribute: (name: string) => buttonAttributes.get(name) ?? null,
    setAttribute: (name: string, value: string) => void buttonAttributes.set(name, value),
    addEventListener: (type: string, handler: () => void) => {
      if (type === "click") clickHandlers.push(handler);
    },
  };
  vi.stubGlobal("document", {
    documentElement: {
      getAttribute: (name: string) => rootAttributes.get(name) ?? null,
      setAttribute: (name: string, value: string) => void rootAttributes.set(name, value),
    },
    getElementById: (id: string) => (id === THEME_TOGGLE_ID ? button : null),
  });
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => {
      if (storageThrows) throw new Error("SecurityError");
      return store.get(key) ?? null;
    },
    setItem: (key: string, value: string) => {
      if (storageThrows) throw new Error("QuotaExceededError");
      store.set(key, value);
    },
  });
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: query === "(prefers-color-scheme: dark)" && systemDark,
    addEventListener: () => undefined,
  }));

  return {
    theme: () => rootAttributes.get("data-theme") ?? null,
    stored: () => store.get(THEME_KEY) ?? null,
    label: () => buttonAttributes.get("aria-label") ?? null,
    click: () => clickHandlers.forEach((handler) => handler()),
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("theme", () => {
  it("follows the system theme until the visitor chooses one", () => {
    const browser = fakeBrowser({ systemDark: true });
    initTheme();
    expect(browser.theme()).toBeNull();
    expect(currentTheme()).toBe("dark");
  });

  it("applies a stored choice over the system theme", () => {
    const browser = fakeBrowser({ systemDark: true, stored: "light" });
    initTheme();
    expect(browser.theme()).toBe("light");
    expect(currentTheme()).toBe("light");
  });

  it("ignores a stored value that is not a theme", () => {
    const browser = fakeBrowser({ stored: "sepia" });
    expect(storedTheme()).toBeNull();
    initTheme();
    expect(browser.theme()).toBeNull();
  });

  it("flips the theme on screen and remembers the choice", () => {
    const browser = fakeBrowser({ systemDark: false });
    initTheme();
    toggleTheme();
    expect(browser.theme()).toBe("dark");
    expect(browser.stored()).toBe("dark");
    toggleTheme();
    expect(browser.theme()).toBe("light");
    expect(browser.stored()).toBe("light");
  });

  it("still switches when storage is unavailable, for the visit only", () => {
    const browser = fakeBrowser({ systemDark: true, storageThrows: true });
    expect(() => initTheme()).not.toThrow();
    expect(storedTheme()).toBeNull();
    expect(() => toggleTheme()).not.toThrow();
    expect(browser.theme()).toBe("light");
  });

  it("wires the masthead button, whose label names the theme it switches to", () => {
    const browser = fakeBrowser({ systemDark: false });
    initTheme();
    expect(browser.label()).toBe("Switch to the dark theme");
    browser.click();
    expect(browser.theme()).toBe("dark");
    expect(browser.label()).toBe("Switch to the light theme");
  });

  it("reads the same storage key in the inline script that applies it before first paint", () => {
    expect(page).toContain(`localStorage.getItem("${THEME_KEY}")`);
  });
});
