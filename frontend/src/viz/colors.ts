import * as d3 from "d3";

// Dark-surface roles (reference palette, dark mode).
export const SURFACE = "#1a1a19";
export const TEXT_PRIMARY = "#ffffff";
export const TEXT_SECONDARY = "#c3c2b7";
export const TEXT_MUTED = "#8b8a83";
export const GUIDE = "#3a3a37";

// Categorical slots 1-2 (dark).
export const SERIES_1 = "#3987e5";
export const SERIES_2 = "#d95926";

// Diverging (weights): blue = negative, gray = 0, red = positive.
export const WEIGHT_NEG = "#5598e7";
export const WEIGHT_MID = "#383835";
export const WEIGHT_POS = "#e66767";

const negArm = d3.interpolateLab(WEIGHT_MID, WEIGHT_NEG);
const posArm = d3.interpolateLab(WEIGHT_MID, WEIGHT_POS);

/** t in [-1, 1]. */
export function weightColor(t: number): string {
  const c = Math.max(-1, Math.min(1, t));
  return c < 0 ? negArm(-c) : posArm(c);
}

// Sequential (activations): one hue (aqua), receding into the surface at zero.
const ACT_STOPS = [SURFACE, "#0f4a37", "#199e70", "#4fd1a0", "#c9f7e4"];
const actInterp = d3.interpolateRgbBasis(ACT_STOPS);

/** t in [0, 1]. */
export function actColor(t: number): string {
  return actInterp(Math.max(0, Math.min(1, t)));
}

// Input pixels are shown as what they are: grayscale ink.
const inputInterp = d3.interpolateRgb("#111110", "#ffffff");
export function inputColor(v: number): string {
  return inputInterp(Math.max(0, Math.min(1, v)));
}

export const cssGradient = (fn: (t: number) => string, from: number, to: number, n = 12) =>
  `linear-gradient(to right, ${d3.range(n).map((i) => fn(from + ((to - from) * i) / (n - 1))).join(", ")})`;
