"""Quick, grid-based depth-to-source estimation methods that complement
the full 3D inversion (processing/inversion.py) and Euler deconvolution
(processing/euler_deconvolution.py) with much faster estimates that need
no per-run structural-index tuning - standard first-pass tools in
potential-field interpretation:

- Tilt-depth (Salem et al., 2007, "Tilt-depth method: a simple depth
  estimation method using first-order magnetic derivatives"): for a
  vertical contact/thin-sheet source, tilt(x) crosses zero directly over
  the source and its slope there is 1/depth, so depth = 1 / |horizontal
  gradient of tilt| at each zero-crossing cell - no contour tracing
  needed.
- Analytic-signal depth (Nabighian, 1972; Roest et al., 1992 half-width
  method): for a 2D contact/thin sheet, the analytic-signal amplitude
  peaks directly over the source with a half-width-at-half-maximum
  (HWHM) approximately equal to depth.
- Spectral depth (Spector & Grant, 1970): a single characteristic
  ensemble-average depth from the radial log-power-spectrum slope -
  the same estimate already used internally to size the 3D inversion
  mesh (processing/inversion_auto.py:estimate_source_depth_m), exposed
  here with the underlying spectrum points for its own diagnostic plot.

All three assume simple source geometries (a 2D contact/thin sheet, or an
ensemble of point/line sources for the spectral method) - like Euler
deconvolution, they are fast reconnaissance estimates, not a substitute
for the full 3D inversion when a body's true geometry matters.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer
from scipy import ndimage

from .inversion_auto import spectral_depth_diagnostics
from .transforms import analytic_signal, derivative_easting, derivative_northing, tilt_angle

_MIN_FINITE_CELLS = 20
_FOUR_CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_MAX_TILT_DEPTH_POINTS = 1500
_MAX_AS_PEAKS = 500


def _to_latlon(x, y, utm_epsg: int):
    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon


def _subsample_indices(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.linspace(0, n - 1, cap).round().astype(int)


def tilt_depth_estimates(
    grid_values: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    cell_size_m: float,
    utm_epsg: int,
    min_depth_m: float = 1.0,
    max_depth_m: float = 500.0,
) -> dict:
    """Depth picks at every tilt-angle zero-crossing cell (Salem et al.
    2007) - see module docstring. available=False (with a reason) when
    the grid is too small or no zero-crossing survives the plausible
    depth range, rather than silently returning an empty-looking success."""
    finite_mask = np.isfinite(grid_values)
    if finite_mask.sum() < _MIN_FINITE_CELLS:
        return {"available": False, "reason": "유효한 격자 셀이 너무 적습니다.", "points": []}

    tilt_rad = np.radians(tilt_angle(grid_values, cell_size_m))
    tilt_rad = np.where(finite_mask, tilt_rad, np.nan)

    sign = np.sign(tilt_rad)
    crossing = np.zeros(tilt_rad.shape, dtype=bool)
    cross_h = (sign[:, :-1] * sign[:, 1:]) < 0
    crossing[:, :-1] |= cross_h
    crossing[:, 1:] |= cross_h
    cross_v = (sign[:-1, :] * sign[1:, :]) < 0
    crossing[:-1, :] |= cross_v
    crossing[1:, :] |= cross_v
    crossing &= finite_mask
    if not crossing.any():
        return {"available": False, "reason": "틸트각(tilt angle) 0도 교차점을 찾지 못했습니다.", "points": []}

    d_east = derivative_easting(tilt_rad, cell_size_m)
    d_north = derivative_northing(tilt_rad, cell_size_m)
    grad_mag = np.hypot(d_east, d_north)

    with np.errstate(divide="ignore", invalid="ignore"):
        depth = 1.0 / grad_mag
    valid = crossing & np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    if not valid.any():
        return {
            "available": False,
            "reason": f"교차점은 있으나 타당한 심도 범위({min_depth_m}~{max_depth_m}m) 내의 추정값이 없습니다.",
            "points": [],
        }

    rows, cols = np.nonzero(valid)
    x2d, y2d = np.meshgrid(easting, northing)
    idx = _subsample_indices(rows.size, _MAX_TILT_DEPTH_POINTS)
    rows, cols = rows[idx], cols[idx]
    lat, lon = _to_latlon(x2d[rows, cols], y2d[rows, cols], utm_epsg)
    depths = depth[rows, cols]

    points = [{"lat": float(la), "lon": float(lo), "depth_m": float(d)} for la, lo, d in zip(lat, lon, depths)]
    return {
        "available": True,
        "n_points": len(points),
        "n_points_before_subsample": int(rows.size) if idx.size == _MAX_TILT_DEPTH_POINTS else len(points),
        "depth_stats": {"min": float(np.min(depths)), "max": float(np.max(depths)), "median": float(np.median(depths))},
        "points": points,
    }


def analytic_signal_depth_estimates(
    grid_values: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    cell_size_m: float,
    utm_epsg: int,
    percentile_threshold: float = 90.0,
    search_radius_cells: int = 15,
) -> dict:
    """Depth picks at each analytic-signal amplitude peak, via the
    half-width-at-half-maximum method (Roest et al. 1992) - see module
    docstring."""
    finite_mask = np.isfinite(grid_values)
    if finite_mask.sum() < _MIN_FINITE_CELLS:
        return {"available": False, "reason": "유효한 격자 셀이 너무 적습니다.", "points": []}

    asa = analytic_signal(grid_values, cell_size_m)
    asa = np.where(finite_mask, asa, np.nan)
    finite_asa = asa[np.isfinite(asa)]
    threshold = float(np.percentile(finite_asa, percentile_threshold))

    padded = np.pad(np.where(np.isfinite(asa), asa, -np.inf), 1, mode="constant", constant_values=-np.inf)
    is_max = np.ones(asa.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor = padded[1 + dr : 1 + dr + asa.shape[0], 1 + dc : 1 + dc + asa.shape[1]]
            is_max &= asa >= neighbor
    peaks = is_max & np.isfinite(asa) & (asa >= threshold)
    if not peaks.any():
        return {"available": False, "reason": "임계값을 넘는 analytic signal 극댓값이 없습니다.", "points": []}

    ny, nx = asa.shape
    rows, cols = np.nonzero(peaks)
    idx = _subsample_indices(rows.size, _MAX_AS_PEAKS)
    rows, cols = rows[idx], cols[idx]

    x2d, y2d = np.meshgrid(easting, northing)
    results_lat, results_lon, results_depth, results_peak = [], [], [], []
    r = search_radius_cells
    for row, col in zip(rows, cols):
        r0, r1 = max(0, row - r), min(ny, row + r + 1)
        c0, c1 = max(0, col - r), min(nx, col + r + 1)
        window = asa[r0:r1, c0:c1]
        peak_val = asa[row, col]
        half_max = peak_val / 2.0
        win_x, win_y = x2d[r0:r1, c0:c1], y2d[r0:r1, c0:c1]
        dist = np.hypot(win_x - x2d[row, col], win_y - y2d[row, col])
        below = np.isfinite(window) & (window <= half_max)
        if not below.any():
            continue  # window too small to see the half-max fall-off - skip rather than guess
        hwhm = float(np.min(dist[below]))
        if hwhm <= 0:
            continue
        lat, lon = _to_latlon(x2d[row, col], y2d[row, col], utm_epsg)
        results_lat.append(lat)
        results_lon.append(lon)
        results_depth.append(hwhm)
        results_peak.append(float(peak_val))

    if not results_depth:
        return {
            "available": False,
            "reason": "탐색 반경 안에서 절반 진폭 지점을 찾지 못했습니다 - 탐색 반경을 넓혀보세요.",
            "points": [],
        }

    points = [
        {"lat": la, "lon": lo, "depth_m": d, "peak_value": p}
        for la, lo, d, p in zip(results_lat, results_lon, results_depth, results_peak)
    ]
    depths = np.array(results_depth)
    return {
        "available": True,
        "n_points": len(points),
        "depth_stats": {"min": float(np.min(depths)), "max": float(np.max(depths)), "median": float(np.median(depths))},
        "points": points,
    }


def spectral_depth_diagnostic(grid_values: np.ndarray, cell_size_m: float) -> dict:
    """Spector & Grant (1970) ensemble-average depth, plus the underlying
    radial power-spectrum points and fitted line for a QC plot - a thin
    dict-shaping wrapper over inversion_auto.py:spectral_depth_diagnostics,
    the same computation already used to size the 3D inversion mesh."""
    diag = spectral_depth_diagnostics(grid_values, cell_size_m)
    if diag is None:
        return {
            "available": False,
            "reason": "격자가 너무 작거나(4x4셀 미만) 방사평균 파워스펙트럼 기울기를 추정할 수 없습니다.",
        }
    return {
        "available": True,
        "depth_m": diag["depth_m"],
        "wavenumbers_rad_per_m": [float(v) for v in diag["wavenumbers_rad_per_m"]],
        "ln_power": [float(v) for v in diag["ln_power"]],
        "fit_band_mask": [bool(v) for v in diag["fit_band_mask"]],
        "fit_slope": diag["fit_slope"],
        "fit_intercept": diag["fit_intercept"],
    }
