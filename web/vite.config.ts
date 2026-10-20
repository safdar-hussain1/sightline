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

/**
 * The demo's own code -- the cockpit and the model runtime -- is a chunk of its
 * own, imported once the page's first chunk runs. A modulepreload in the head
 * lets the browser fetch both chunks side by side. The chunk's file name
 * carries its content hash, so the same sources give the same tag.
 */
function preloadCockpit(): Plugin {
  return {
    name: "sightline-preload-cockpit",
    transformIndexHtml: {
      order: "post",
      handler: (html, ctx) => {
        const chunk = Object.values(ctx.bundle ?? {}).find(
          (item) => item.type === "chunk" && item.isDynamicEntry && (item.facadeModuleId ?? "").endsWith("/src/cockpit/cockpit.ts"),
        );
        // The dev server has no bundle, and fetches modules as asked.
        if (!chunk) return html;
        return [{ tag: "link", attrs: { rel: "modulepreload", crossorigin: true, href: `./${chunk.fileName}` }, injectTo: "head" }];
      },
    },
  };
}

// docs/ is the GitHub Pages site root, so the build lands there, and every
// asset URL is relative: the site is served from a project subpath, where an
// absolute "/assets/..." would 404.
const config = {
  base: "./",
  plugins: [staticShell(), preloadCockpit()],
  build: {
    outDir: "../docs",
    emptyOutDir: true,
    target: "es2022",
    // Sourcemaps embed the absolute path of every source file on the machine
    // that built them, which would publish the checkout's location.
    sourcemap: false,
  },
  // For the test runner, which reads this file too: the stylesheet's tests
  // read it as text, which the runner allows only for a stylesheet it
  // processes. The build ignores this key.
  test: { css: { include: [/\/src\/styles\.css/] } },
};

export default defineConfig(config);
