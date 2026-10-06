/**
 * The page's entry: the stylesheet, the masthead and the theme.
 *
 * The masthead is already in the served HTML (the build writes it in);
 * `mountShell` only fills it if the page arrived without it.
 */

import "./styles.css";

import { mountShell } from "./shell.ts";
import { initTheme } from "./theme.ts";

console.info("Sightline — built by Safdar Hussain · https://github.com/safdar-hussain1/sightline");

mountShell();
initTheme();
