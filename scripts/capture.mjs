// Screenshots of the page, taken in the Chrome that scripts/capture.sh opens.
//
//   node scripts/capture.mjs <devtools-port> <page-url> <out-dir>
//
// It writes into <out-dir>:
//
//   og-image.png      the picture a shared link shows: the demo's first
//                     screen, 1200 x 630, with the cockpit paused on a frame
//                     the model has answered, its heat over the road;
//   readme-hero.png   the same screen, smaller, for the README;
//   <view>-<theme>-<width>.png
//                     the whole of each view (demo, results) in the dark and
//                     the light theme at 1512 px, and in the dark at 390 px,
//                     for reading the page over.
//
// The two pictures of the first screen are laid out at a laptop's width and
// drawn at less than one device pixel per CSS pixel: the whole of the first
// screen fits, at the layout a visitor sees, and the PNG comes out at the
// size it is published at. Under reduced motion the demo starts paused, so
// the frame shown is chosen here, not by when the shot happens to be taken.
// Nothing is injected into the page but the press on the strip that picks the
// frame and the scrolls that bring each section into view.
//
// Node 22 or later (it uses the built-in WebSocket); no packages.

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const [port, url, out] = process.argv.slice(2);
if (!port || !url || !out) {
  console.error("usage: node scripts/capture.mjs <devtools-port> <page-url> <out-dir>");
  process.exit(2);
}
mkdirSync(out, { recursive: true });

/** The frame of the 2018 clip the first-screen pictures show, counted from
 * zero: a bend to the left, with the heat on the lane line and the guardrail. */
const SHARE_FRAME = 47;
const LIMIT_MS = 90_000;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const target = targets.find((t) => t.type === "page");
if (!target) throw new Error("Chrome has no page open");
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});

let nextId = 0;
const pending = new Map();
socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  const waiting = message.id !== undefined ? pending.get(message.id) : undefined;
  if (!waiting) return;
  pending.delete(message.id);
  if (message.error) waiting.reject(new Error(`${waiting.method}: ${message.error.message}`));
  else waiting.resolve(message.result);
});

function send(method, params = {}) {
  nextId += 1;
  const id = nextId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, method });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(expression) {
  const result = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description ?? "the page threw");
  return result.result.value;
}

async function until(expression, what) {
  const deadline = Date.now() + LIMIT_MS;
  while (Date.now() < deadline) {
    if (await evaluate(expression)) return;
    await sleep(200);
  }
  throw new Error(`timed out waiting for ${what}`);
}

const DEMO_READY = `document.querySelector(".cockpit__console")?.dataset.state === "ready"`;
const RESULTS_READY = `["results", "paired", "pairs", "timeline"].every((k) => {
  const el = document.querySelector('[data-' + k + '="root"]');
  return el && el.dataset.state !== "loading";
})`;

/** Open `hash` on the page with the viewport, scale, theme and motion asked for. */
async function open({ width, height, scale, theme, reduced, hash = "" }) {
  await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: scale, mobile: width < 500 });
  await send("Emulation.setEmulatedMedia", {
    features: [
      { name: "prefers-color-scheme", value: theme },
      { name: "prefers-reduced-motion", value: reduced ? "reduce" : "no-preference" },
    ],
  });
  // A blank page first, so a change of hash alone is a fresh load.
  await send("Page.navigate", { url: "about:blank" });
  await sleep(200);
  const page = new URL(url);
  page.hash = hash;
  await send("Page.navigate", { url: page.href });
  await until(`document.readyState === "complete"`, "the page to load");
}

async function shoot(file, params = {}) {
  const { data } = await send("Page.captureScreenshot", { format: "png", ...params });
  writeFileSync(join(out, file), Buffer.from(data, "base64"));
  console.log(join(out, file));
}

/** The demo paused on SHARE_FRAME, its answer and heat drawn. */
async function firstScreen(file, width, height, scale) {
  await open({ width, height, scale, theme: "dark", reduced: true });
  await until(DEMO_READY, "the demo to start");
  await evaluate(`(() => {
    const strip = document.querySelector('[data-cockpit="strip"]');
    const box = strip.getBoundingClientRect();
    const frames = Number(strip.getAttribute("aria-valuemax")) + 1;
    const x = box.left + ((${SHARE_FRAME} + 0.5) / frames) * box.width;
    strip.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, clientX: x, clientY: box.top + box.height / 2, pointerId: 1 }));
    strip.dispatchEvent(new PointerEvent("pointerup", { bubbles: true, clientX: x, clientY: box.top + box.height / 2, pointerId: 1 }));
  })()`);
  await until(
    `document.querySelector('[data-cockpit="strip"]').getAttribute("aria-valuenow") === "${SHARE_FRAME}"
      && document.querySelector('[data-cockpit="wheel"]').dataset.waiting !== "true"
      && !document.querySelector('[data-cockpit="video"]').seeking`,
    "the frame and its answer",
  );
  // The heat and the wheel settle.
  await sleep(1200);
  await shoot(file);
}

/** A whole view, every section drawn as a visitor scrolling down would see it. */
async function wholeView(view, theme, width) {
  const height = width < 500 ? 844 : 841;
  await open({ width, height, scale: 1, theme, reduced: false, hash: view === "results" ? "#results" : "" });
  await until(view === "results" ? RESULTS_READY : DEMO_READY, `the ${view} view`);
  const total = await evaluate("document.documentElement.scrollHeight");
  for (let y = 0; y < total; y += Math.floor(height * 0.8)) {
    await evaluate(`window.scrollTo(0, ${y})`);
    await sleep(300);
  }
  await sleep(2000);
  await evaluate("window.scrollTo(0, 0)");
  await sleep(500);
  const { cssContentSize } = await send("Page.getLayoutMetrics");
  await shoot(`${view}-${theme}-${width}.png`, {
    captureBeyondViewport: true,
    clip: { x: 0, y: 0, width, height: Math.ceil(cssContentSize.height), scale: 1 },
  });
}

try {
  await send("Page.enable");
  await send("Runtime.enable");
  // Laid out at 1600 x 840 and drawn at three quarters: 1200 x 630.
  await firstScreen("og-image.png", 1600, 840, 0.75);
  // The laptop screen the page is designed for, drawn at five eighths.
  await firstScreen("readme-hero.png", 1512, 841, 0.625);
  for (const view of ["demo", "results"]) {
    await wholeView(view, "dark", 1512);
    await wholeView(view, "light", 1512);
    await wholeView(view, "dark", 390);
  }
} finally {
  socket.close();
}
