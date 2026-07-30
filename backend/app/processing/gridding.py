"""Grid scattered magnetic point data onto a regular metric grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import verde as vd


@dataclass
class GridResult:
    values: np.ndarray  # shape (n_north, n_east), NaN outside data coverage
    easting: np.ndarray  # 1D, x/easting coordinate of each column
    northing: np.ndarray  # 1D, y/northing coordinate of each row
    cell_size_m: float
    region: tuple  # (west, east, south, north) in local meters


def grid_points(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    cell_size_m: float,
    max_distance_m: float | None = None,
) -> GridResult:
    """Block-mean reduce then bi-harmonic spline-interpolate scattered
    (x, y, values) onto a regular grid at cell_size_m spacing. Cells farther
    than max_distance_m from any input point are masked to NaN so the grid
    doesn't extrapolate far beyond the flown lines (defaults to 2 cells)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x, y, values = x[finite], y[finite], values[finite]
    if len(x) < 4:
        raise ValueError("그리딩을 위한 유효 포인트가 부족합니다 (최소 4개 필요).")

    region = vd.get_region((x, y))
    reducer = vd.BlockReduce(reduction="mean", spacing=cell_size_m)
    (x_r, y_r), values_r = reducer.filter((x, y), values)

    spline = vd.Spline()
    spline.fit((x_r, y_r), values_r)

    shape_coords = vd.grid_coordinates(region, spacing=cell_size_m)
    easting_2d, northing_2d = shape_coords
    grid_values = spline.predict((easting_2d, northing_2d))

    if max_distance_m is None:
        max_distance_m = 2.0 * cell_size_m
    tree_dist = _nearest_distance(easting_2d, northing_2d, x, y)
    grid_values = np.where(tree_dist <= max_distance_m, grid_values, np.nan)

    return GridResult(
        values=grid_values,
        easting=easting_2d[0, :],
        northing=northing_2d[:, 0],
        cell_size_m=cell_size_m,
        region=tuple(region),
    )


def _nearest_distance(grid_x: np.ndarray, grid_y: np.ndarray, pts_x: np.ndarray, pts_y: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([pts_x, pts_y]))
    dist, _ = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    return dist.reshape(grid_x.shape)
