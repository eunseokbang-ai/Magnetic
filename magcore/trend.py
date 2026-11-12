"""Regional trend (background) removal: fits a low-order polynomial
surface to a gridded field by least squares and subtracts it, leaving
the "residual" - the standard way to separate a slowly-varying regional
field (deep/broad sources, or leveling drift) from the shallower local
anomalies actually of interest, when no independent estimate of the
regional field is available.
"""
from __future__ import annotations

import numpy as np


def _design_matrix(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Columns are every monomial x^i * y^j with i+j <= order."""
    terms = []
    for total in range(order + 1):
        for i in range(total + 1):
            j = total - i
            terms.append((x**i) * (y**j))
    return np.column_stack(terms)


def remove_regional_trend(
    grid_values: np.ndarray, easting: np.ndarray, northing: np.ndarray, order: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an order-N polynomial surface (order=1: plane, 2: quadratic,
    3: cubic) to the finite cells of grid_values and return
    (residual, trend), both shaped like grid_values with NaN preserved
    outside coverage. Coordinates are centered/scaled before fitting so
    the (typically large, e.g. UTM-easting-scale) coordinate values don't
    ill-condition the least-squares solve."""
    finite = np.isfinite(grid_values)
    if finite.sum() < 3:
        raise ValueError("추세면을 맞추기 위한 유효 셀이 부족합니다.")

    x2d, y2d = np.meshgrid(easting, northing)
    x0, y0 = float(np.mean(easting)), float(np.mean(northing))
    scale = max(float(easting.max() - easting.min()), float(northing.max() - northing.min()), 1.0)
    xs = (x2d - x0) / scale
    ys = (y2d - y0) / scale

    A = _design_matrix(xs[finite], ys[finite], order)
    b = grid_values[finite]
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)

    A_full = _design_matrix(xs.ravel(), ys.ravel(), order)
    trend = (A_full @ coeffs).reshape(grid_values.shape)
    trend = np.where(finite, trend, np.nan)
    residual = grid_values - trend
    return residual, trend


REGIONAL_METHODS = ("none", "level", "plane", "quadratic")
_REGIONAL_ORDER = {"plane": 1, "quadratic": 2}


def robust_trend_surface(grid_values: np.ndarray, easting: np.ndarray, northing: np.ndarray, order: int = 1) -> np.ndarray:
    """An order-N polynomial surface through the background of a gridded
    field: fitted with Tukey biweights, iteratively reweighted, from the
    median level, so that it follows the bulk of the cells and is not
    pulled toward the anomalies standing on them. NaN where grid_values
    is. See separate_regional for why a plain least-squares surface will
    not do for this."""
    values = np.asarray(grid_values, dtype=float)
    finite = np.isfinite(values)
    x2d, y2d = np.meshgrid(easting, northing)
    x0, y0 = float(np.mean(easting)), float(np.mean(northing))
    scale = max(float(np.ptp(easting)), float(np.ptp(northing)), 1.0)
    A = _design_matrix(((x2d - x0) / scale)[finite], ((y2d - y0) / scale)[finite], order)
    b = values[finite]
    if len(b) <= A.shape[1]:
        raise ValueError("추세면을 맞추기 위한 유효 셀이 부족합니다.")
    # start from the median level rather than from least squares, so the
    # first set of weights is not already tilted toward the anomalies
    coeffs = np.zeros(A.shape[1])
    coeffs[0] = float(np.median(b))
    for _ in range(30):
        resid = b - A @ coeffs
        spread = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        if spread <= 0:
            break
        u = resid / (4.685 * spread)
        w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        sw = np.sqrt(w)
        new, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        done = np.max(np.abs(A @ (new - coeffs))) < 1e-6 * max(spread, 1e-12)
        coeffs = new
        if done:
            break
    surface = np.full(values.shape, np.nan)
    surface[finite] = A @ coeffs
    return surface


def separate_regional(
    grid_values: np.ndarray, easting: np.ndarray, northing: np.ndarray, method: str = "level"
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split a gridded anomaly into the part a 3D inversion should explain
    and a background it should not: (residual, regional, info), both grids
    shaped like grid_values with NaN preserved.

    The inversion has no unknown for a background. Its model is
    susceptibility between 0 and a ceiling, cell by cell, and whatever
    level the data sit at has to be produced by that: a negative level by
    bodies placed so that their side lobes cover the block, a positive one
    by a blanket of susceptibility. It fits - and the susceptibilities are
    well above what the anomalies themselves call for (GaSaDo 2026-11,
    41 flights: the anomaly sits at -100 nT; left in, the model averages
    0.16 SI over its cells with 5.4 % of them at the 1 SI ceiling, with
    the level out 0.09 SI and 1.7 %, at 3.9 and 4.8 nT misfit).

    method:
      "none"       nothing removed.
      "level"      the median. One number, and the choice that assumes
                   least: that half the block is above background.
      "plane"      a first-order surface, for a block on a gradient.
      "quadratic"  a second-order surface. It can swallow an anomaly as
                   wide as the block; look at what it removed.

    The surfaces are fitted robustly (robust_trend_surface), unlike
    remove_regional_trend above, a plain least-squares surface: the fit
    must follow the background, not be pulled toward the anomalies
    standing on it. A least-squares plane under one strong positive anomaly tilts up
    toward it and takes part of it away; the median does not move, and
    neither should the plane.

    info: method, the removed surface's median/min/max (nT), and the
    data's median before and after.
    """
    if method not in REGIONAL_METHODS:
        raise ValueError(f"unknown regional method: {method}")
    values = np.asarray(grid_values, dtype=float)
    finite = np.isfinite(values)
    if method == "none" or finite.sum() < 3:
        regional = np.where(finite, 0.0, np.nan)
    elif method == "level":
        regional = np.where(finite, float(np.median(values[finite])), np.nan)
    else:
        regional = robust_trend_surface(values, easting, northing, _REGIONAL_ORDER[method])
    residual = values - regional
    has = bool(finite.any())
    info = {
        "method": method,
        "level_nt": float(np.median(regional[finite])) if has else 0.0,
        "min_nt": float(np.min(regional[finite])) if has else 0.0,
        "max_nt": float(np.max(regional[finite])) if has else 0.0,
        "data_median_before_nt": float(np.median(values[finite])) if has else 0.0,
        "data_median_after_nt": float(np.median(residual[finite])) if has else 0.0,
        "data_std_nt": float(np.std(residual[finite])) if has else 0.0,
    }
    return residual, regional, info
