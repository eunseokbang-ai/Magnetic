"""Grid scattered magnetic point data onto a regular metric grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import verde as vd
from scipy.interpolate import griddata
from scipy.ndimage import zoom as ndi_zoom

# verde.Spline's fit cost grows roughly O(n^2.5-3) in the number of control
# points (it solves a dense n x n linear system) - empirically, ~4000
# points already takes several seconds and it gets unusable well before
# the tens of thousands of points a real multi-line survey block-reduces
# to. Capped via extra coarsening purely for the fit step - see the
# "spline" branch in grid_points.
_MAX_SPLINE_CONTROL_POINTS = 3000

# Hard cap on total output grid cells, checked before doing any work -
# even "nearest" (the fastest method) gets slow (multi-second to a
# minute+) once a large-area survey is gridded at a very fine cell size,
# and scipy.interpolate.griddata's "linear"/"cubic" (Qhull-based) get
# considerably worse. Failing fast with a clear message beats a request
# that silently takes minutes.
_MAX_GRID_CELLS = 3_000_000


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
    approx_nx = int(round((region[1] - region[0]) / cell_size_m)) + 1
    approx_ny = int(round((region[3] - region[2]) / cell_size_m)) + 1
    if approx_nx * approx_ny > _MAX_GRID_CELLS:
        raise ValueError(
            f"셀 크기({cell_size_m:.2f}m)가 너무 작아 격자가 지나치게 촘촘해집니다 "
            f"(약 {approx_nx}x{approx_ny}={approx_nx * approx_ny:,}셀, 최대 {_MAX_GRID_CELLS:,}셀) - "
            "셀 크기를 늘리거나 관심 영역만 남기고 나머지 측선을 수동 제외한 뒤 다시 시도하세요."
        )

    reducer = vd.BlockReduce(reduction="mean", spacing=cell_size_m)
    (x_r, y_r), values_r = reducer.filter((x, y), values)

    shape_coords = vd.grid_coordinates(region, spacing=cell_size_m)
    easting_2d, northing_2d = shape_coords

    if method == "nearest":
        grid_values = griddata((x_r, y_r), values_r, (easting_2d, northing_2d), method="nearest")
    elif method == "spline":
        control_x, control_y, control_v = x_r, y_r, values_r
        if len(control_x) > _MAX_SPLINE_CONTROL_POINTS:
            # Re-reduce at a coarser spacing just for the points used to
            # *fit* the spline (its cost is what's actually unbounded);
            # the fitted surface is still *evaluated*/predicted on the
            # full fine-resolution target grid below, so output
            # resolution is unaffected - only how smooth/damped the
            # surface is between real data points.
            #
            # Survey data lies along thin flight lines, not filling the
            # plane, so populated-block count scales roughly linearly
            # with spacing (not quadratically as area/spacing^2 would
            # suggest) - an analytic one-shot factor consistently
            # undershoots. Grown geometrically and re-checked instead.
            coarse_spacing = cell_size_m
            for _ in range(12):
                coarse_spacing *= 1.4
                coarse_reducer = vd.BlockReduce(reduction="mean", spacing=coarse_spacing)
                (control_x, control_y), control_v = coarse_reducer.filter((x_r, y_r), values_r)
                if len(control_x) <= _MAX_SPLINE_CONTROL_POINTS:
                    break
        spline = vd.Spline()
        spline.fit((control_x, control_y), control_v)
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
    z_init: np.ndarray | None = None,
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

    z_init seeds the starting surface instead of the flat data-mean fill
    (see _minimum_curvature_multilevel, which passes in a coarse-grid
    solution upsampled to this level) - a much better starting point
    converges in far fewer iterations than starting flat everywhere.
    """
    ny, nx = shape
    if z_init is not None:
        z = np.where(constrained_mask, constrained_values, z_init).astype(float)
    else:
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


# Below this cell count, a single relaxation pass (fully converged) is
# already fast - no need to pay for building a coarse pyramid.
_MULTILEVEL_BASE_CELLS = 3000
_MULTILEVEL_DOWNSAMPLE_FACTOR = 3.0
# The damped relaxation above propagates information roughly one cell per
# iteration; once a level starts from a good (upsampled coarse-solution)
# guess it only needs to resolve local, level-scale detail, not the whole
# domain, so far fewer iterations are needed than the base level's full
# solve.
_MULTILEVEL_FINE_ITERATIONS = 300


def _downsample_constraints(
    mask: np.ndarray, values: np.ndarray, factor: float
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Block-average a (mask, values) constraint grid down by `factor`
    along each axis (integer cell grouping), for building a coarser level
    of the multilevel solve below."""
    ny, nx = mask.shape
    f = max(1, int(round(factor)))
    coarse_ny = max(1, -(-ny // f))  # ceil division
    coarse_nx = max(1, -(-nx // f))

    row_idx = np.repeat(np.arange(coarse_ny), f)[:ny]
    col_idx = np.repeat(np.arange(coarse_nx), f)[:nx]
    RR, CC = np.meshgrid(row_idx, col_idx, indexing="ij")

    flat_mask = mask.ravel()
    sum_grid = np.zeros((coarse_ny, coarse_nx))
    count_grid = np.zeros((coarse_ny, coarse_nx))
    if flat_mask.any():
        np.add.at(sum_grid, (RR.ravel()[flat_mask], CC.ravel()[flat_mask]), values.ravel()[flat_mask])
        np.add.at(count_grid, (RR.ravel()[flat_mask], CC.ravel()[flat_mask]), 1)

    coarse_mask = count_grid > 0
    coarse_values = np.zeros((coarse_ny, coarse_nx))
    coarse_values[coarse_mask] = sum_grid[coarse_mask] / count_grid[coarse_mask]
    return coarse_mask, coarse_values, (coarse_ny, coarse_nx)


def _upsample_to_shape(z_coarse: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Bilinear-ish upsample (via scipy.ndimage.zoom) of a coarse solved
    surface to a finer grid shape, used only as a relaxation starting
    guess - exact interpolation quality doesn't matter since the
    subsequent relaxation pass (with hard constraints re-applied) refines
    it, so any minor size-rounding mismatch from zoom's own shape
    computation is just clamped/padded to the exact target shape."""
    cny, cnx = z_coarse.shape
    ny, nx = target_shape
    if (cny, cnx) == (ny, nx):
        return z_coarse.copy()
    up = ndi_zoom(z_coarse, (ny / cny, nx / cnx), order=1, mode="nearest")
    out = np.empty((ny, nx), dtype=float)
    uy, ux = up.shape
    cy, cx = min(uy, ny), min(ux, nx)
    out[:cy, :cx] = up[:cy, :cx]
    if cy < ny:
        out[cy:, :cx] = out[cy - 1 : cy, :cx]
    if cx < nx:
        out[:, cx:] = out[:, cx - 1 : cx]
    return out


def _minimum_curvature_multilevel(
    shape: tuple[int, int],
    constrained_mask: np.ndarray,
    constrained_values: np.ndarray,
) -> np.ndarray:
    """Coarse-to-fine warm start for the relaxation solver above: build a
    pyramid of progressively coarser constraint grids, fully solve the
    coarsest (cheap - few cells), then use each solved level (upsampled)
    as the starting guess for the next finer level instead of a flat
    fill.

    This matters because damped local relaxation propagates information
    only ~1 cell per iteration, so a single-level solve needs iteration
    counts on the order of the grid's own diameter (in cells) to converge
    - fine for a few thousand cells, but a multi-hundred-thousand-cell
    grid (a real multi-line survey at a few meters' cell size) either
    times out well before converging or takes tens of seconds even when
    it does. Seeding each level from the previous, already-converged
    coarse level means only local (level-scale) detail needs resolving at
    each step, cutting total work by roughly an order of magnitude for
    large grids while landing on the same fixed point (the constraints
    and the biharmonic stencil are unchanged, only the starting guess and
    per-level iteration budget are).
    """
    ny, nx = shape
    if ny * nx <= _MULTILEVEL_BASE_CELLS:
        return _minimum_curvature_relax((ny, nx), constrained_mask, constrained_values)

    coarse_mask, coarse_values, coarse_shape = _downsample_constraints(
        constrained_mask, constrained_values, _MULTILEVEL_DOWNSAMPLE_FACTOR
    )
    coarse_z = _minimum_curvature_multilevel(coarse_shape, coarse_mask, coarse_values)
    z_init = _upsample_to_shape(coarse_z, (ny, nx))
    return _minimum_curvature_relax(
        (ny, nx), constrained_mask, constrained_values, max_iterations=_MULTILEVEL_FINE_ITERATIONS, z_init=z_init
    )


def _grid_minimum_curvature(
    x_r: np.ndarray, y_r: np.ndarray, values_r: np.ndarray, easting_2d: np.ndarray, northing_2d: np.ndarray
) -> np.ndarray:
    """Assign each (already block-averaged) data point to its nearest grid
    node as a hard constraint, then relax the rest of the grid to minimum
    curvature around those constraints (via a coarse-to-fine multilevel
    warm start for large grids - see _minimum_curvature_multilevel)."""
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

    return _minimum_curvature_multilevel((ny, nx), constrained_mask, constrained_values)
