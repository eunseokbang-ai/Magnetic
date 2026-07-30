// Small dependency-free colormaps. Names match valid matplotlib colormap
// names 1:1 so the same choice can be sent as the backend's `cmap` request
// field for grid/transform PNG rendering and stay visually consistent with
// the point layer's client-side coloring.

const VIRIDIS_STOPS = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
  [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
  [180, 222, 44], [253, 231, 37],
];

const PLASMA_STOPS = [
  [13, 8, 135], [84, 2, 163], [139, 10, 165], [185, 50, 137],
  [219, 92, 104], [244, 136, 73], [254, 188, 43], [240, 249, 33],
];

const TURBO_STOPS = [
  [48, 18, 59], [65, 69, 171], [70, 117, 237], [57, 162, 237],
  [24, 199, 197], [63, 220, 140], [146, 231, 73], [216, 215, 44],
  [247, 161, 38], [231, 90, 15], [122, 4, 3],
];

// low -> high already in "reversed" (red = high) orientation
const RDYLBU_R_STOPS = [
  [49, 54, 149], [69, 117, 180], [116, 173, 209], [171, 217, 233],
  [224, 243, 248], [255, 255, 191], [254, 224, 144], [253, 174, 97],
  [244, 109, 67], [215, 48, 39], [165, 0, 38],
];

const RDBU_R_STOPS = [
  [5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222],
  [209, 229, 240], [253, 219, 199], [244, 165, 130], [214, 96, 77],
  [178, 24, 43], [103, 0, 31],
];

const SPECTRAL_R_STOPS = [
  [94, 79, 162], [50, 136, 189], [102, 194, 165], [171, 221, 164],
  [230, 245, 152], [255, 255, 191], [254, 224, 139], [253, 174, 97],
  [244, 109, 67], [213, 62, 79], [158, 1, 66],
];

const GRAY_STOPS = [
  [20, 20, 20], [255, 255, 255],
];

// Approximates the classic Geosoft Oasis Montaj default grid color table
// (dark blue -> blue -> cyan -> green -> yellow -> orange -> red -> magenta
// -> pink). Control points are unevenly spaced, unlike the other palettes
// above, so they're resampled into an evenly-spaced array once at load time
// (matches backend/app/processing/colormaps.py's control points).
const GEOSOFT_RAINBOW_CONTROL_POINTS = [
  [0.0, [10, 10, 120]],
  [0.1, [20, 60, 200]],
  [0.2, [0, 160, 220]],
  [0.3, [0, 210, 190]],
  [0.4, [0, 200, 90]],
  [0.5, [140, 220, 40]],
  [0.58, [255, 255, 0]],
  [0.66, [255, 180, 0]],
  [0.74, [255, 90, 0]],
  [0.82, [230, 20, 20]],
  [0.9, [200, 20, 160]],
  [1.0, [255, 200, 235]],
];

function resampleControlPoints(controlPoints, n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    let j = 0;
    while (j < controlPoints.length - 2 && controlPoints[j + 1][0] < t) j++;
    const [t0, c0] = controlPoints[j];
    const [t1, c1] = controlPoints[j + 1];
    const frac = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
    out.push([
      Math.round(c0[0] + (c1[0] - c0[0]) * frac),
      Math.round(c0[1] + (c1[1] - c0[1]) * frac),
      Math.round(c0[2] + (c1[2] - c0[2]) * frac),
    ]);
  }
  return out;
}

const GEOSOFT_RAINBOW_STOPS = resampleControlPoints(GEOSOFT_RAINBOW_CONTROL_POINTS, 64);

const COLORMAPS = {
  viridis: VIRIDIS_STOPS,
  plasma: PLASMA_STOPS,
  turbo: TURBO_STOPS,
  RdYlBu_r: RDYLBU_R_STOPS,
  RdBu_r: RDBU_R_STOPS,
  Spectral_r: SPECTRAL_R_STOPS,
  gray: GRAY_STOPS,
  geosoft_rainbow: GEOSOFT_RAINBOW_STOPS,
};

export const COLORMAP_OPTIONS = [
  { value: "viridis", label: "Viridis (연속)" },
  { value: "plasma", label: "Plasma (연속)" },
  { value: "turbo", label: "Turbo / Rainbow (연속)" },
  { value: "RdYlBu_r", label: "Red-Yellow-Blue (발산, 기본 이상값)" },
  { value: "RdBu_r", label: "Red-Blue (발산)" },
  { value: "Spectral_r", label: "Spectral (발산)" },
  { value: "gray", label: "Grayscale" },
  { value: "geosoft_rainbow", label: "Geosoft Rainbow (물리탐사 표준)" },
];

function interpolateStops(stops, t) {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (stops.length - 1);
  const i0 = Math.floor(scaled);
  const i1 = Math.min(stops.length - 1, i0 + 1);
  const frac = scaled - i0;
  const c0 = stops[i0];
  const c1 = stops[i1];
  const r = Math.round(c0[0] + (c1[0] - c0[0]) * frac);
  const g = Math.round(c0[1] + (c1[1] - c0[1]) * frac);
  const b = Math.round(c0[2] + (c1[2] - c0[2]) * frac);
  return `rgb(${r},${g},${b})`;
}

export function getColorFn(cmapName) {
  const stops = COLORMAPS[cmapName] || COLORMAPS.viridis;
  return (t) => interpolateStops(stops, t);
}

export function makeColorScale(cmapName, vmin, vmax) {
  const fn = getColorFn(cmapName);
  const span = vmax - vmin || 1;
  return (value) => fn((value - vmin) / span);
}
