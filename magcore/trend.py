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
