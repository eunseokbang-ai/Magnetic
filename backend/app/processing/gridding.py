"""Grid scattered magnetic point data onto a regular metric grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import verde as vd
from scipy.interpolate import griddata


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
    method: str = "nearest",
    max_distance_m: float | None = None,
) -> GridResult:
    """Block-mean reduce then interpolate scattered (x, y, values) onto a
    regular grid at cell_size_m spacing.

    method: "nearest" (fast KD-tree nearest-value fill, blocky "raw cell"
    look - default), "linear" or "cubic" (scipy.interpolate.griddata,
    smoother but slower), "spline" (verde bi-harmonic spline, smoothest
    but solves a dense linear system - can be slow with many points), or
    "minimum_curvature" (Briggs 1974 minimum-curvature relaxation, the
    algorithm behind Golden Software Surfer's default gridder - see
    _grid_minimum_curvature below).

    Cells farther than max_distance_m from any input point are masked to
    NaN so the grid doesn't extrapolate far beyond the flown lines. The
    caller should pass a value based on the actual line spacing (see
    processing.lines.estimate_line_spacing_m) - a plain multiple of
    cell_size_m is usually far smaller than the gap between adjacent lines
    and leaves most of the survey block masked out. Defaults to 2 cells
    when not provided, which only fills a narrow band along each line."""
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

    shape_coords = vd.grid_coordinates(region, spacing=cell_size_m)
    easting_2d, northing_2d = shape_coords

    if method == "nearest":
        grid_values = griddata((x_r, y_r), values_r, (easting_2d, northing_2d), method="nearest")
    elif method == "spline":
        spline = vd.Spline()
        spline.fit((x_r, y_r), values_r)
        grid_values = spline.predict((easting_2d, northing_2d))
    elif method in ("linear", "cubic"):
        grid_values = griddata((x_r, y_r), values_r, (easting_2d, northing_2d), method=method)
        # griddata leaves NaN outside the convex hull; fall back to nearest
        # so the max_distance_m mask below is the only thing trimming edges.
        nan_mask = np.isnan(grid_values)
        if nan_mask.any():
            nearest = griddata((x_r, y_r), values_r, (easting_2d, northing_2d), method="nearest")
            grid_values = np.where(nan_mask, nearest, grid_values)
    elif method == "minimum_curvature":
        grid_values = _grid_minimum_curvature(x_r, y_r, values_r, easting_2d, northing_2d)
    else:
        raise ValueError(f"알 수 없는 보간 방법입니다: {method}")

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


def _nearest_axis_index(axis_1d: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Index of the closest node in a sorted 1D coordinate axis to each of
    `values` (nearest, not just left/right-of)."""
    idx = np.clip(np.searchsorted(axis_1d, values), 1, len(axis_1d) - 1)
    left, right = axis_1d[idx - 1], axis_1d[idx]
    return np.where((values - left) <= (right - values), idx - 1, idx)


def _minimum_curvature_relax(
    shape: tuple[int, int],
    constrained_mask: np.ndarray,
    constrained_values: np.ndarray,
    max_iterations: int = 1500,
    tol: float = 1e-4,
    relaxation: float = 0.5,
) -> np.ndarray:
    """Briggs (1974) minimum-curvature grid relaxation: iteratively solves
    the discrete biharmonic equation (nabla^4 z = 0) everywhere except at
    grid nodes with an actual data constraint, which are clamped to their
    (block-averaged) data value every pass - the standard algorithm behind
    Golden Software Surfer's default "Minimum Curvature" gridder, chosen
    there specifically for potential-field data because it produces the
    smoothest possible surface consistent with the data (no artificial
    faceting/blockiness the way nearest-neighbor gridding does).

    Uses a 13-point finite-difference biharmonic stencil, vectorized over
    the whole grid each iteration (edge-replicate padding avoids periodic
    wraparound at the grid boundary, acting as a free/zero-curvature
    boundary condition there). Unlike the 5-point Laplacian, this
    stencil's off-diagonal coefficients (44 in magnitude: 8*4 + 2*4 + 1*4)
    outweigh its diagonal one (20), so it is not diagonally dominant and
    plain (relaxation=1.0) Jacobi iteration reliably diverges - verified
    empirically here. A damped update (relaxation=0.5 by default, i.e.
    each pass only moves halfway from the current value toward the
    biharmonic estimate) stays stable and still converges well within
    max_iterations for the grid sizes this app produces.
    """
    ny, nx = shape
    if constrained_mask.any():
        fill_value = float(np.mean(constrained_values[constrained_mask]))
    else:
        fill_value = 0.0
    z = np.where(constrained_mask, constrained_values, fill_value).astype(float)

    for _ in range(max_iterations):
        padded = np.pad(z, 2, mode="edge")
        n_up = padded[1 : 1 + ny, 2 : 2 + nx]
        n_down = padded[3 : 3 + ny, 2 : 2 + nx]
        n_left = padded[2 : 2 + ny, 1 : 1 + nx]
        n_right = padded[2 : 2 + ny, 3 : 3 + nx]
        n_ul = padded[1 : 1 + ny, 1 : 1 + nx]
        n_ur = padded[1 : 1 + ny, 3 : 3 + nx]
        n_dl = padded[3 : 3 + ny, 1 : 1 + nx]
        n_dr = padded[3 : 3 + ny, 3 : 3 + nx]
        n_up2 = padded[0:ny, 2 : 2 + nx]
        n_down2 = padded[4 : 4 + ny, 2 : 2 + nx]
        n_left2 = padded[2 : 2 + ny, 0:nx]
        n_right2 = padded[2 : 2 + ny, 4 : 4 + nx]

        biharmonic = (
            8 * (n_up + n_down + n_left + n_right)
            - 2 * (n_ul + n_ur + n_dl + n_dr)
            - (n_up2 + n_down2 + n_left2 + n_right2)
        ) / 20.0

        z_new = np.where(constrained_mask, constrained_values, z + relaxation * (biharmonic - z))
        if not np.all(np.isfinite(z_new)):
            # damped update should not diverge, but bail out to the last
            # finite state rather than propagate NaN/inf if it somehow does
            break
        delta = float(np.max(np.abs(z_new - z)))
        z = z_new
        if delta < tol:
            break

    return z


def _grid_minimum_curvature(
    x_r: np.ndarray, y_r: np.ndarray, values_r: np.ndarray, easting_2d: np.ndarray, northing_2d: np.ndarray
) -> np.ndarray:
    """Assign each (already block-averaged) data point to its nearest grid
    node as a hard constraint, then relax the rest of the grid to minimum
    curvature around those constraints."""
    ny, nx = easting_2d.shape
    easting_1d, northing_1d = easting_2d[0, :], northing_2d[:, 0]
    col_idx = _nearest_axis_index(easting_1d, x_r)
    row_idx = _nearest_axis_index(northing_1d, y_r)

    sum_grid = np.zeros((ny, nx))
    count_grid = np.zeros((ny, nx))
    np.add.at(sum_grid, (row_idx, col_idx), values_r)
    np.add.at(count_grid, (row_idx, col_idx), 1)

    constrained_mask = count_grid > 0
    constrained_values = np.zeros((ny, nx))
    constrained_values[constrained_mask] = sum_grid[constrained_mask] / count_grid[constrained_mask]

    return _minimum_curvature_relax((ny, nx), constrained_mask, constrained_values)
