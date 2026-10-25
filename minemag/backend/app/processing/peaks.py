"""Where the targets are: peaks of the total horizontal derivative (THD)
or the analytic signal amplitude (ASA) of the anomaly grid, kept when
their prominence exceeds K times the grid's MAD - sections 2.6.4-2.6.5
of Bang, Jeong & Yim (2026).

Why a derivative and not the anomaly itself: at Korea's inclination the
anomaly of a mine is a dipole whose positive and negative lobes sit
north and south of the object, so the maximum of the anomaly is not
over the object. The THD peaks over it. The paper compared THD and ASA
against surveyed mine positions and found THD the better locator for
these clearly dipolar targets (ASA's vertical-derivative term is pulled
by the two lobes): on the KIGAM lawn the THD error was 0.20 m (drone)
and 0.23 m (robot) for M15 anti-tank mines, 0.13 and 0.24 m for M16
anti-personnel mines; on the riverside site 0.21 and 0.24 m.

Prominence, not height: a peak's prominence is its height above the
highest saddle that connects it to any higher peak (the grid edge
counts as a saddle at the grid minimum), so a bump riding on a bigger
anomaly's flank does not count, while an isolated small anomaly does.
The threshold K x MAD scales with the survey's own background: the
paper's reference values at 0.5 m probe height were K = 65-71 for
anti-tank mines and K = 19-20 for anti-personnel mines on both
platforms, and it notes K has to be calibrated per site. Each detected
peak therefore reports the K it would need, so the operator sees the
margin, not just pass/fail.

Type hint from the anomaly's peak-to-peak amplitude at 0.5 m probe
height, from the paper's tables (M15: 512-1,631 nT; M16: 186-329 nT):
>= 500 nT anti-tank class, 150-500 nT anti-personnel class, below that
small metal or unknown. It is a hint for that height and filter, not an
identification - the paper is explicit that magnetics alone cannot tell
a mine from a fragment of the same metal mass.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from magcore.transforms import analytic_signal, total_horizontal_derivative

DEFAULT_K = 20.0                  # the paper's anti-personnel reference
# The THD of one dipole has a weaker secondary maximum at the edge of its
# far lobe, 0.7-1.0 m away for a mine 0.5-1.1 m below the probe; anything
# closer than this is the same source. Two mines 2 m apart (the paper's
# closest anti-personnel pair) remain separate.
DEFAULT_MIN_SEPARATION_M = 1.2
_PROMINENCE_LEVELS = 64
_P2P_RADIUS_M = 1.5
TYPE_CLASSES = [
    (500.0, "대전차지뢰급", "M15 참고 512~1,631 nT (탐침 0.5 m)"),
    (150.0, "대인지뢰급", "M16 참고 186~329 nT (탐침 0.5 m)"),
    (0.0, "소형 금속/미상", "150 nT 미만"),
]


def derivative_grid(grid_values: np.ndarray, cell_size_m: float, kind: str) -> np.ndarray:
    if kind == "thd":
        return total_horizontal_derivative(grid_values, cell_size_m)
    if kind == "asa":
        return analytic_signal(grid_values, cell_size_m)
    if kind == "anomaly":
        return grid_values
    raise ValueError(f"알 수 없는 파생 격자: {kind}")


def mad(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return 0.0
    return float(np.median(np.abs(v - np.median(v))))


def _prominences(grid: np.ndarray, peaks: list[tuple[int, int]]) -> np.ndarray:
    """Prominence of each (row, col) peak: the peak minus the highest
    level at which the connected region above that level containing the
    peak also contains a higher cell. Levels are swept from the top down
    with one labelling per level, so the cost is _PROMINENCE_LEVELS
    labellings whatever the number of peaks."""
    finite = np.isfinite(grid)
    filled = np.where(finite, grid, -np.inf)
    heights = np.array([grid[r, c] for r, c in peaks])
    out = np.full(len(peaks), np.nan)
    gmin = float(np.nanmin(grid))
    top = float(np.nanmax(grid))
    if top <= gmin:
        return np.zeros(len(peaks))
    levels = np.linspace(top, gmin, _PROMINENCE_LEVELS, endpoint=False)[1:]
    unresolved = np.ones(len(peaks), dtype=bool)
    for level in levels:
        if not unresolved.any():
            break
        labels, n = ndimage.label(filled >= level)
        if n == 0:
            continue
        region_max = ndimage.maximum(filled, labels, index=np.arange(1, n + 1))
        for i in np.flatnonzero(unresolved):
            if heights[i] < level:
                continue
            r, c = peaks[i]
            lab = labels[r, c]
            if lab == 0:
                continue
            if region_max[lab - 1] > heights[i] + 1e-9:
                out[i] = heights[i] - level
                unresolved[i] = False
    out[unresolved] = heights[unresolved] - gmin
    return out


def find_peaks(
    derivative: np.ndarray,
    anomaly: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    cell_size_m: float,
    k: float = DEFAULT_K,
    min_separation_m: float = DEFAULT_MIN_SEPARATION_M,
) -> tuple[list[dict], dict]:
    """Local maxima of `derivative` whose prominence is >= k x MAD, with
    the anomaly's peak-to-peak amplitude and extent around each."""
    finite = np.isfinite(derivative)
    scale = mad(derivative)
    if scale <= 0 or finite.sum() < 9:
        return [], {"mad": scale, "k": k, "threshold": None, "n_local_maxima": 0}
    size = max(3, int(round(min_separation_m / cell_size_m)))
    if size % 2 == 0:
        size += 1
    filled = np.where(finite, derivative, -np.inf)
    local = (filled == ndimage.maximum_filter(filled, size=size, mode="nearest")) & finite
    # a plateau of equal values would give several maxima: keep one per label
    labels, n = ndimage.label(local)
    peaks = []
    for lab in range(1, n + 1):
        rr, cc = np.nonzero(labels == lab)
        peaks.append((int(rr[len(rr) // 2]), int(cc[len(cc) // 2])))
    prom = _prominences(derivative, peaks)
    threshold = k * scale
    found = []
    anom_med = float(np.nanmedian(anomaly)) if np.isfinite(anomaly).any() else 0.0
    r_px = max(1, int(round(_P2P_RADIUS_M / cell_size_m)))
    # A peak with empty cells within half the separation is on the edge of
    # the coverage: the anomaly may be cut, and the peak may be the cut.
    edge_px = max(1, size // 2)
    eroded = ndimage.binary_erosion(finite, structure=np.ones((2 * edge_px + 1, 2 * edge_px + 1)), border_value=0)
    for (r, c), p in zip(peaks, prom):
        if not np.isfinite(p) or p < threshold:
            continue
        r0, r1 = max(0, r - r_px), min(anomaly.shape[0], r + r_px + 1)
        c0, c1 = max(0, c - r_px), min(anomaly.shape[1], c + r_px + 1)
        win = anomaly[r0:r1, c0:c1]
        win = win[np.isfinite(win)]
        p2p = float(win.max() - win.min()) if len(win) else float("nan")
        found.append({
            "x": float(easting[c]), "y": float(northing[r]),
            "value": float(derivative[r, c]),
            "prominence": float(p),
            "k_needed": float(p / scale),
            "p2p_nt": round(p2p, 1),
            "extent_m": round(_extent_m(derivative, r, c, cell_size_m), 2),
            "type_class": type_class(p2p),
            "at_edge": bool(not eroded[r, c]),
        })
    found.sort(key=lambda d: -d["prominence"])
    return found, {"mad": round(scale, 4), "k": k, "threshold": round(threshold, 4), "n_local_maxima": int(n)}


def _extent_m(derivative: np.ndarray, r: int, c: int, cell: float) -> float:
    """Equivalent diameter of the connected region above half the peak."""
    half = derivative[r, c] * 0.5
    labels, _ = ndimage.label(np.where(np.isfinite(derivative), derivative, -np.inf) >= half)
    n_px = int((labels == labels[r, c]).sum()) if labels[r, c] else 1
    return 2.0 * np.sqrt(n_px * cell * cell / np.pi)


def type_class(p2p_nt: float) -> str:
    if not np.isfinite(p2p_nt):
        return "미상"
    for lower, name, _ref in TYPE_CLASSES:
        if p2p_nt >= lower:
            return name
    return TYPE_CLASSES[-1][1]
