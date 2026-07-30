// Small dependency-free colormaps matching the backend's default choices
// closely enough for consistent point/grid coloring.

const VIRIDIS_STOPS = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
  [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
  [180, 222, 44], [253, 231, 37],
];

const RDYLBU_STOPS = [
  [165, 0, 38], [215, 48, 39], [244, 109, 67], [253, 174, 97],
  [254, 224, 144], [255, 255, 191], [224, 243, 248], [171, 217, 233],
  [116, 173, 209], [69, 117, 180], [49, 54, 149],
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

export function viridis(t) {
  return interpolateStops(VIRIDIS_STOPS, t);
}

// RdYlBu_r (reversed: red=high, blue=low), matching backend's anomaly cmap.
export function rdylbuReversed(t) {
  return interpolateStops(RDYLBU_STOPS.slice().reverse(), t);
}

export function makeColorScale(cmapName, vmin, vmax) {
  const fn = cmapName === "anomaly" ? rdylbuReversed : viridis;
  const span = vmax - vmin || 1;
  return (value) => fn((value - vmin) / span);
}
