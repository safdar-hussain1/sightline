/**
 * The page's entry: the stylesheet, the masthead, the theme and the cockpit.
 *
 * The masthead and the cockpit's markup are already in the served HTML (the
 * build writes them in); `mountShell` only fills the masthead if the page
 * arrived without it. The cockpit's code, the model runtime and everything
 * they download come in a separate chunk, so the page is drawn before them.
 */

import "./styles.css";

import { mountShell } from "./shell.ts";
import { initTheme } from "./theme.ts";

console.info("Sightline — built by Safdar Hussain · https://github.com/safdar-hussain1/sightline");

mountShell();
initTheme();

void import("./cockpit/cockpit.ts").then(({ startCockpit }) => startCockpit());
