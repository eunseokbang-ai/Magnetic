"""Multi-scale edge detection ("worming"): the UAV magnetics guidelines
list this (after Archibald et al., 1999) among the standard grid-based
interpretation tools, complementary to Euler deconvolution - total
horizontal derivative (THDR) ridges are traced at a series of upward-
continued heights, and the resulting edge points (one cloud per height)
give a quick-look picture of geological contact/structure locations and
how they persist or migrate with depth (a ridge that stays in roughly the
same place across many heights indicates a steep, laterally-continuous
contact; one that only appears at low continuation heights indicates a
shallow, localised source).

This implements the simplified, commonly-used form of the technique: THDR
ridge points are just local maxima of the upward-continued THDR grid above
a percentile threshold, rather than the full multi-directional edge
tracing of the original algorithm - adequate for the "quick structural
overview" role the guidelines describe it filling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

from .transforms import total_horizontal_derivative, upward_continuation

_MIN_FINITE_CELLS = 20


@dataclass
class EdgePoint:
    lat: float
    lon: float
    height_m: float
    thdr_value: float


def _local_maxima_mask(grid: np.ndarray) -> np.ndarray:
    """True where a finite cell is >= all 8 finite neighbours (edge cells
    of the array can never qualify, matching the array's own padding)."""
    padded = np.pad(grid, 1, mode="constant", constant_values=-np.inf)
    is_max = np.ones(grid.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor = padded[1 + dr : 1 + dr + grid.shape[0], 1 + dc : 1 + dc + grid.shape[1]]
            is_max &= grid >= neighbor
    return is_max & np.isfinite(grid)


def run_multiscale_edges(
    grid_values: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    cell_size_m: float,
    utm_epsg: int,
    heights_m: list[float],
    percentile: float = 80.0,
) -> list[EdgePoint]:
    finite_mask = np.isfinite(grid_values)
    if finite_mask.sum() < _MIN_FINITE_CELLS:
        return []

    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    x2d, y2d = np.meshgrid(easting, northing)

    points: list[EdgePoint] = []
    for height_m in heights_m:
        continued = grid_values if height_m == 0 else upward_continuation(grid_values, cell_size_m, height_m)
        thdr = total_horizontal_derivative(continued, cell_size_m)
        thdr = np.where(finite_mask, thdr, np.nan)
        finite_thdr = thdr[np.isfinite(thdr)]
        if finite_thdr.size < _MIN_FINITE_CELLS:
            continue
        threshold = float(np.percentile(finite_thdr, percentile))

        maxima = _local_maxima_mask(np.where(np.isfinite(thdr), thdr, -np.inf)) & (thdr >= threshold)
        rows, cols = np.nonzero(maxima)
        if rows.size == 0:
            continue

        xs, ys = x2d[rows, cols], y2d[rows, cols]
        lons, lats = transformer.transform(xs, ys)
        values = thdr[rows, cols]
        points.extend(
            EdgePoint(lat=float(lat), lon=float(lon), height_m=float(height_m), thdr_value=float(v))
            for lat, lon, v in zip(lats, lons, values)
        )

    return points
