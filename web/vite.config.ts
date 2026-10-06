import { defineConfig, type Plugin } from "vite";

import { injectShell } from "./src/shell.ts";

/**
 * Writes the masthead -- the mark, the one <h1> and the lede -- into
 * index.html, so the served page carries its heading before any script runs.
 * The markup comes from the same `shellHtml()` the app uses.
 */
function staticShell(): Plugin {
  return {
    name: "sightline-static-shell",
    transformIndexHtml: (html) => injectShell(html),
  };
}

// docs/ is the GitHub Pages site root, so the build lands there, and every
// asset URL is relative: the site is served from a project subpath, where an
// absolute "/assets/..." would 404.
export default defineConfig({
  base: "./",
  plugins: [staticShell()],
  build: {
    outDir: "../docs",
    emptyOutDir: true,
    target: "es2022",
    // Sourcemaps embed the absolute path of every source file on the machine
    // that built them, which would publish the checkout's location.
    sourcemap: false,
  },
});
