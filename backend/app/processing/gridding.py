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

# How far (in cells) auto-mode is allowed to extrapolate past the convex
# hull of the actual survey points - see _hull_extrapolation_mask.
_HULL_BUFFER_CELLS = 2.0

# Auto-mode interior fill reach, as a fraction of the local gap between the
# two nearest flight lines (see _local_line_gap_m). At the exact midpoint
# between two straight, evenly-spaced lines the nearest single point is
# local_gap/2 away, so anything above 0.5 covers that ideal case - but real
# lines bow, drift, have along-track point spacing/dropouts (QC/despike
# exclusion, turns), and rarely all start/end at exactly the same along-
# track position, which locally pushes the true nearest-point distance a
# bit past that ideal half-gap, especially near where a shorter line ends
# but its neighbors keep going. A coefficient of 0.6 left too thin a margin
# for that: fine cell sizes (which sample the threshold at many more
# discrete points across the same interior region than a coarse grid does)
# would occasionally land a cell just past the 0.6 line and mask it out,
# producing small "중간중간" gaps between lines that a coarser grid simply
# never happened to sample. Tuned against a synthetic irregular multi-line
# survey (line spacing 50m, bowed/jittered lines with staggered start/end
# points): 0.6-1.0 all still left one or more interior cells unfilled
# somewhere between 15m and 5m cell size, while 1.2 left none at any of
# 15/10/5m - so gridding down to a small fraction of the line spacing (as
# recommended for detailed work, e.g. 1/10) stays fully filled between
# lines even on realistically messy data, not just clean synthetic ones.
# Still fundamentally bounded, same as before: _hull_extrapolation_mask
# below is a separate, independent hard cap that this coefficient cannot
# push past no matter how large - see test_hull_cap_prevents_unbounded_
# fan_extrapolation_past_line_endpoints in test_local_line_gap.py.
_INTERIOR_FILL_FRACTION = 1.2

# _local_line_gap_m's per-line cKDTree query is the dominant cost of auto
# masking on a fine/large grid; below this cell count a single exact pass
# is already fast enough that the downsampling machinery isn't worth it.
_LOCAL_GAP_DOWNSAMPLE_MIN_CELLS = 4000
_LOCAL_GAP_DOWNSAMPLE_FACTOR = 4.0


@dataclass
class GridResult:
    values: np.ndarray  # shape (n_north, n_east), NaN outside data coverage
    easting: np.ndarray  # 1D, x/easting coordinate of each column
    northing: np.ndarray  # 1D, y/northing coordinate of each row
    cell_size_m: float
    region: tuple  # (west, east, south, north) in local meters


def _along_line_lowpass(
    x: np.ndarray, y: np.ndarray, values: np.ndarray, line_id: np.ndarray, wavelength_m: float
) -> np.ndarray:
    """Low-pass each flight line's values along its own along-line arc
    length before gridding, at a cutoff wavelength tied to the *cross*-line
    spacing (resolved by the caller).

    Why this is needed: BlockReduce (see grid_points) already collapses
    along-line point density down to roughly one value per grid cell, but
    that alone does not fix the underlying anisotropy - each line's
    block-reduced value sequence still carries real signal/noise detail at
    wavelengths the *cross*-line direction has no way to resolve at all
    (adjacent lines are typically 5-50x farther apart than the along-line
    sample spacing). Whichever interpolation method is used then
    reproduces that along-line-only detail faithfully while necessarily
    smoothing heavily across lines - which is exactly what shows up as
    fine ridges/corrugation running parallel to the flight lines in the
    gridded surface, most visibly in derivative-based transforms
    (RTP/1VD/tilt/etc.) that amplify high-wavenumber content. Smoothing
    along-line detail down to the same wavelength the cross-line direction
    can resolve removes that fabricated anisotropy before it ever reaches
    the grid, instead of trying to filter it back out afterward.

    Order-independent: each line's own principal direction is found via
    PCA and used to project/sort points, rather than assuming the input
    array order is already along-line-sequential.
    """
    out = values.copy()
    for lid in np.unique(line_id):
        mask = line_id == lid
        n = int(mask.sum())
        if n < 3:
            continue
        xi, yi, vi = x[mask], y[mask], values[mask]
        pts = np.column_stack([xi, yi])
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, int(np.argmax(eigvals))]
        s = centered @ direction  # along-line position (arbitrary origin/sign)
        order = np.argsort(s)
        s_sorted = s[order]
        v_sorted = vi[order]

        half = wavelength_m / 2.0
        csum = np.concatenate([[0.0], np.cumsum(v_sorted)])
        lo = np.searchsorted(s_sorted, s_sorted - half, side="left")
        hi = np.searchsorted(s_sorted, s_sorted + half, side="right")
        smoothed_sorted = (csum[hi] - csum[lo]) / (hi - lo)

        smoothed = np.empty_like(smoothed_sorted)
        smoothed[order] = smoothed_sorted
        out[np.flatnonzero(mask)] = smoothed
    return out


def grid_points(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    cell_size_m: float,
    method: str = "nearest",
    max_distance_m: float | None = None,
    line_id: np.ndarray | None = None,
    along_line_smooth_wavelength_m: float | None = None,
    typical_line_spacing_m: float | None = None,
) -> GridResult:
    """Block-mean reduce then interpolate scattered (x, y, values) onto a
    regular grid at cell_size_m spacing.

    method: "nearest" (fast KD-tree nearest-value fill, blocky "raw cell"
    look - default), "linear" or "cubic" (scipy.interpolate.griddata,
    smoother but slower), "spline" (verde bi-harmonic spline, smoothest
    but solves a dense linear system - can be slow with many points),
    "minimum_curvature" (Briggs 1974 minimum-curvature relaxation, the
    algorithm behind Golden Software Surfer's default gridder - see
    _grid_minimum_curvature below), or "boxing" (each cell's value is
    simply the average of the raw points that fall inside it, per the
    UAV magnetics guidelines' "Boxing" gridding method - cells with no
    data stay NaN rather than being filled in from a neighbor, unlike
    every other method here; the most literal, assumption-free option
    when you specifically don't want any interpolation to happen).

    Cells farther than max_distance_m from any input point are masked to
    NaN so the grid doesn't extrapolate far beyond the flown lines. If
    max_distance_m is given explicitly, it's used as-is everywhere (a
    plain multiple of cell_size_m is usually far smaller than the gap
    between adjacent lines and leaves most of the survey block masked
    out). If left None (auto) and line_id is provided, the threshold is
    computed per grid cell instead of as one project-wide constant - 60%
    of the actual local gap between the two nearest distinct flight lines
    at that exact location (see _local_line_gap_m) - so bowed/skewed
    lines or a stretch flown with locally wider spacing than the survey
    average still get filled correctly instead of leaving a gap in the
    middle of an otherwise-covered block. Falls back to a flat 2 cells
    when line_id isn't given (or there's only one line), same as before.
    In auto mode, cells are additionally never filled past the convex hull
    of the actual survey points, buffered outward by the greater of
    _HULL_BUFFER_CELLS cells or (a fraction of) typical_line_spacing_m,
    regardless of what the local-gap heuristic computes there (see
    _hull_extrapolation_mask) - the local-gap distance is unbounded far
    outside the hull (both "nearest" lines become roughly equidistant,
    growing with distance from the survey), so without this cap auto mode
    extrapolates without limit in a triangular fan past line endpoints.
    typical_line_spacing_m (a single project-wide robust estimate, e.g.
    store.py's self.line_spacing_m - deliberately *not* the per-cell,
    unbounded local_gap array) sizes that buffer generously enough that a
    flight path which bows/curves rather than running in dead-straight
    parallel lines - and so traces out a footprint that's itself slightly
    concave - doesn't get a real interior gap near the bend clipped back
    out by too tight a hull buffer. Left None, only the small flat
    _HULL_BUFFER_CELLS buffer applies.

    line_id + along_line_smooth_wavelength_m: if both are given, each
    line's values are along-line low-passed (see _along_line_lowpass)
    before block-reducing/interpolating, to prevent flight-line-parallel
    corrugation - see that function's docstring for why this is necessary
    regardless of which interpolation method is chosen below."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x, y, values = x[finite], y[finite], values[finite]
    if line_id is not None:
        line_id = np.asarray(line_id)[finite]
    if len(x) < 4:
        raise ValueError("그리딩을 위한 유효 포인트가 부족합니다 (최소 4개 필요).")

    if line_id is not None and along_line_smooth_wavelength_m and along_line_smooth_wavelength_m > 0:
        values = _along_line_lowpass(x, y, values, line_id, along_line_smooth_wavelength_m)

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
    elif method == "boxing":
        grid_values = _grid_boxing(x_r, y_r, values_r, easting_2d, northing_2d)
    else:
        raise ValueError(f"알 수 없는 보간 방법입니다: {method}")

    tree_dist = _nearest_distance(easting_2d, northing_2d, x, y)
    if max_distance_m is None:
        # Auto: fill each cell out to _INTERIOR_FILL_FRACTION of the
        # *local* gap between the two nearest actual flight lines at that
        # specific location (bowed/skewed lines and locally wider-than-
        # average spacing are accounted for directly), not a single
        # survey-wide average - see _local_line_gap_m. Falls back to the
        # old flat 2 cells when there's no line grouping to work with (e.g.
        # a single line).
        local_gap = _local_line_gap_m(easting_2d, northing_2d, x, y, line_id) if line_id is not None else None
        if local_gap is not None:
            max_distance_grid = np.maximum(2.0 * cell_size_m, _INTERIOR_FILL_FRACTION * local_gap)
        else:
            max_distance_grid = 2.0 * cell_size_m
        # Hard cap, independent of the heuristic above: never extrapolate
        # past the actual survey footprint - see _hull_extrapolation_mask
        # docstring for why local_gap alone doesn't bound this. The buffer
        # itself needs to be more than a token couple of cells, though: a
        # flight path that bows/curves (not perfectly straight parallel
        # lines) traces out a *concave* footprint, and a flat, tiny buffer
        # around the strict convex hull would then re-cut real interior
        # gap-fill area near those bends/curves right back out. Sizing the
        # buffer off typical_line_spacing_m (a single robust, bounded,
        # project-wide statistic - see this function's docstring) instead
        # of anything derived from local_gap's own distribution keeps this
        # safe: local_gap spans a huge, heavily skewed range from ~0 right
        # at a data point up to arbitrarily large near a hull edge/corner,
        # so any statistic pulled from it (even "just" the values already
        # inside the hull) risks being dragged right back up toward the
        # same unbounded blowup this cap exists to prevent.
        buffer_m = _HULL_BUFFER_CELLS * cell_size_m
        if typical_line_spacing_m:
            buffer_m = max(buffer_m, _INTERIOR_FILL_FRACTION * typical_line_spacing_m)
        hull_mask = _hull_extrapolation_mask(easting_2d, northing_2d, x, y, buffer_m)
    else:
        max_distance_grid = max_distance_m
        hull_mask = True
    grid_values = np.where((tree_dist <= max_distance_grid) & hull_mask, grid_values, np.nan)

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


def _hull_extrapolation_mask(
    grid_x: np.ndarray, grid_y: np.ndarray, pts_x: np.ndarray, pts_y: np.ndarray, buffer_m: float
) -> np.ndarray:
    """Boolean mask, True for grid cells within buffer_m of the convex hull
    of the actual survey points.

    This is the hard cap that keeps auto-mode masking bounded: the
    local-gap heuristic in _local_line_gap_m has no upper limit for a
    query point beyond a flight line's endpoint (its two nearest distinct
    lines end up roughly equidistant, both growing proportionally with how
    far outside the survey the point is, so the "60% of local gap"
    threshold never actually excludes it) - which without this cap
    produces unbounded triangular extrapolation fans radiating outward
    past line endpoints/corners. Bounding fill to the convex hull (plus a
    couple of cells of slack) is the same constraint Oasis
    Montaj/Surfer/ArcGIS-style grid engines apply by default, and it does
    not affect the actual Phase-4 gap-filling use case (filling real gaps
    *between* lines) since those gaps already lie inside the hull.

    buffer_m works via shapely's buffer() regardless of point-set shape
    (degenerate/collinear point sets still produce a valid buffered
    polygon - a "stadium" shape around a line, or a circle around a single
    point - rather than needing special-casing here).
    """
    from shapely import contains_xy
    from shapely.geometry import MultiPoint

    hull = MultiPoint(np.column_stack([pts_x, pts_y])).convex_hull.buffer(buffer_m)
    return contains_xy(hull, grid_x.ravel(), grid_y.ravel()).reshape(grid_x.shape)


def _local_line_gap_m(
    grid_x: np.ndarray, grid_y: np.ndarray, pts_x: np.ndarray, pts_y: np.ndarray, line_id: np.ndarray
) -> np.ndarray | None:
    """Per grid-cell estimate of the real cross-line spacing at that exact
    location: distance to the nearest point of the closest flight line,
    plus distance to the nearest point of the second-closest flight line.

    A single project-wide average line spacing (self.line_spacing_m in
    store.py) understates the true gap wherever lines are locally bowed,
    skewed, or simply flown farther apart than the average that day -
    exactly where a fixed global max_distance_m then masks out cells the
    interpolation would otherwise have filled in perfectly reasonably,
    leaving real gaps in the middle of an otherwise-covered block. This is
    computed fresh per grid cell from the actual nearest point on each
    actual line, so it tracks the true local gap instead of one constant
    everywhere. Returns None if there are fewer than 2 distinct lines (no
    "spacing between lines" is defined with only one).

    On a large/fine grid with many lines, computing this exactly (a
    separate cKDTree query per line, over every single grid cell) is the
    dominant cost of auto masking. local_gap is a spacing-scale quantity
    that varies smoothly over the survey, not per-cell detail, so above
    _LOCAL_GAP_DOWNSAMPLE_MIN_CELLS it's computed on a coarser grid
    (_LOCAL_GAP_DOWNSAMPLE_FACTOR cells per axis) and upsampled back to
    full resolution instead, cutting the per-line query cost by roughly
    _LOCAL_GAP_DOWNSAMPLE_FACTOR**2 with no visible change in the result.
    """
    lines = np.unique(line_id)
    if len(lines) < 2:
        return None

    ny, nx = grid_x.shape
    if ny * nx <= _LOCAL_GAP_DOWNSAMPLE_MIN_CELLS:
        return _local_line_gap_m_exact(grid_x, grid_y, pts_x, pts_y, line_id, lines)

    coarse_ny = max(2, int(round(ny / _LOCAL_GAP_DOWNSAMPLE_FACTOR)))
    coarse_nx = max(2, int(round(nx / _LOCAL_GAP_DOWNSAMPLE_FACTOR)))
    coarse_x = np.linspace(grid_x[0, 0], grid_x[0, -1], coarse_nx)
    coarse_y = np.linspace(grid_y[0, 0], grid_y[-1, 0], coarse_ny)
    coarse_xx, coarse_yy = np.meshgrid(coarse_x, coarse_y)
    coarse_gap = _local_line_gap_m_exact(coarse_xx, coarse_yy, pts_x, pts_y, line_id, lines)
    return _upsample_to_shape(coarse_gap, (ny, nx))


def _local_line_gap_m_exact(
    grid_x: np.ndarray, grid_y: np.ndarray, pts_x: np.ndarray, pts_y: np.ndarray, line_id: np.ndarray, lines: np.ndarray
) -> np.ndarray:
    from scipy.spatial import cKDTree

    query_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    per_line_dist = np.empty((len(lines), query_pts.shape[0]))
    for i, lid in enumerate(lines):
        mask = line_id == lid
        tree = cKDTree(np.column_stack([pts_x[mask], pts_y[mask]]))
        per_line_dist[i], _ = tree.query(query_pts)
    nearest_two = np.partition(per_line_dist, 1, axis=0)[:2]
    local_gap = nearest_two[0] + nearest_two[1]
    return local_gap.reshape(grid_x.shape)


def _nearest_axis_index(axis_1d: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Index of the closest node in a sorted 1D coordinate axis to each of
    `values` (nearest, not just left/right-of)."""
    idx = np.clip(np.searchsorted(axis_1d, values), 1, len(axis_1d) - 1)
    left, right = axis_1d[idx - 1], axis_1d[idx]
    return np.where((values - left) <= (right - values), idx - 1, idx)


def _grid_boxing(
    x_r: np.ndarray, y_r: np.ndarray, values_r: np.ndarray, easting_2d: np.ndarray, northing_2d: np.ndarray
) -> np.ndarray:
    """"Boxing" gridding (see grid_points docstring): each cell gets the
    plain average of whichever already block-reduced points snap to it -
    no interpolation into empty cells at all, unlike every other method
    here."""
    ny, nx = easting_2d.shape
    easting_1d, northing_1d = easting_2d[0, :], northing_2d[:, 0]
    col_idx = _nearest_axis_index(easting_1d, x_r)
    row_idx = _nearest_axis_index(northing_1d, y_r)

    sum_grid = np.zeros((ny, nx))
    count_grid = np.zeros((ny, nx))
    np.add.at(sum_grid, (row_idx, col_idx), values_r)
    np.add.at(count_grid, (row_idx, col_idx), 1)

    out = np.full((ny, nx), np.nan)
    mask = count_grid > 0
    out[mask] = sum_grid[mask] / count_grid[mask]
    return out


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
