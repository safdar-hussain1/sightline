import { defineConfig, type Plugin } from "vite";

import { injectCockpit } from "./src/cockpit/markup.ts";
import { injectSections } from "./src/sections/markup.ts";
import { injectShell } from "./src/shell.ts";

/**
 * Writes the masthead -- the mark, the one <h1>, the lede and the nav -- the
 * cockpit's markup and the markup of the sections around it into index.html,
 * so the served page carries its heading and its words before any script
 * runs, and its first paint already has the page's layout. The markup comes
 * from the same `shellHtml()`, `cockpitHtml()` and section functions the app
 * uses.
 */
function staticShell(): Plugin {
  return {
    name: "sightline-static-shell",
    transformIndexHtml: (html) => injectSections(injectCockpit(injectShell(html))),
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
