/**
 * Ambient declarations for the non-TypeScript things the bundle imports.
 *
 * Written out rather than pulled in with `"types": ["vite/client"]`: tsconfig
 * keeps `types` empty, so what the compiler believes about the world is what is
 * written here, not whatever a toolchain package declares globally.
 */

/** `import "./styles.css"`: a side-effect import the bundler turns into a
 * stylesheet link. It has no value. */
declare module "*.css";

/** `import page from "../index.html?raw"`: the file's text, as the bundler
 * reads it from disk. Tests use it to check the page template against the code
 * that depends on it. */
declare module "*?raw" {
  const text: string;
  export default text;
}

/** `import url from "...?url"`: the URL the bundler gives the file in the
 * build -- for the runtime's wasm binary, a hashed file beside the app. */
declare module "*?url" {
  const url: string;
  export default url;
}
