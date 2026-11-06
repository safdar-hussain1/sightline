// Write `docs/build-manifest.json`: what `docs/` contains, and what it was
// built from.
//
// `docs/` is the published site, and it is a build output that lives in the
// repository. Without a record of what it was built from, it could drift from
// the sources for any length of time: a stale bundle served next to a green
// test suite, which tests `web/src` while visitors run `docs/`. The manifest
// records a digest of every file the build reads and every file it writes,
// and `tests/test_docs_build_manifest.py` recomputes both sides from the
// working tree. Edit a source without building again, edit a file in `docs/`
// by hand, or delete one, and that test fails.
//
// Run by the last step of `npm run build`, never by hand: a manifest written
// by hand would describe a build nobody ran.
//
// The file is a pure function of the tree: paths relative to the repository
// root with forward slashes, sorted, sha256 in hex, and no timestamp, host,
// user or tool version. So the same sources give the same bytes on every
// machine and every Node version, and CI can require that the build leaves
// `docs/` exactly as committed.

import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const docs = join(root, "docs");
const web = join(root, "web");

const MANIFEST = join(docs, "build-manifest.json");

/** Test files are left out of the inputs on purpose: Vitest specs are not in
 * the bundler's module graph, so editing one cannot change `docs/`, and
 * counting them would make every test edit look like a stale build. */
const EXCLUDED_INPUT = /\.test\.ts$/;

/** Every file under `directory`, at any depth, but those `skip` names. */
function walk(directory, skip = () => false) {
  const out = [];
  for (const entry of readdirSync(directory)) {
    const full = join(directory, entry);
    if (skip(full)) continue;
    if (statSync(full).isDirectory()) {
      out.push(...walk(full, skip));
    } else {
      out.push(full);
    }
  }
  return out;
}

function digest(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

/** `{path: sha256}` for each of `paths`, keyed by the path from the
 * repository root with forward slashes, in code-unit order: an order that
 * depends on neither the file system nor the locale. */
function digestsOf(paths) {
  const keyed = paths.map((path) => [relative(root, path).split(sep).join("/"), path]);
  keyed.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return Object.fromEntries(keyed.map(([key, path]) => [key, digest(path)]));
}

// What the build reads: the page, its sources, its static files, and
// everything that decides how they are compiled and bundled -- the Vite
// config (which also writes the page's markup into index.html), the
// TypeScript config, the package file and the lockfile that pins every
// tool's exact version.
const inputs = [
  join(web, "index.html"),
  join(web, "vite.config.ts"),
  join(web, "tsconfig.json"),
  join(web, "package.json"),
  join(web, "package-lock.json"),
  ...walk(join(web, "src"), (path) => EXCLUDED_INPUT.test(path)),
  ...walk(join(web, "public")),
];

// What the build wrote: everything in docs/ but this file.
const outputs = walk(docs, (path) => path === MANIFEST);

const manifest = {
  note: "Written by scripts/build_manifest.mjs during `npm run build`. Do not edit.",
  inputs: digestsOf(inputs),
  outputs: digestsOf(outputs),
};

writeFileSync(MANIFEST, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(
  `build manifest: ${Object.keys(manifest.inputs).length} inputs, ` +
    `${Object.keys(manifest.outputs).length} outputs`,
);
