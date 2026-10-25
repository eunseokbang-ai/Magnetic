"""Automatic mesh/inversion parameter suggestion from the survey's own
data, so a user can run a first-pass 3D inversion without having to guess
a cell size or investigation depth.

Depth: uses the classic Spector & Grant (1970) statistical spectral
method - for an ensemble of magnetic sources at a characteristic depth
d, the radially-averaged power spectrum of the field decays as
exp(-2*k*d), so a linear fit of ln(radial power) against wavenumber k
gives depth = -slope/2. This is a standard, widely-used depth-to-source
estimate from a gridded anomaly map (not a per-body precision estimate,
but a good first-pass characteristic depth for sizing the mesh).

Horizontal cell size: bounded below by half the flight line spacing
(the survey's own crosstrack resolution limit) and above by whatever
keeps the observation grid and mesh within the size caps enforced in
store.run_inversion.

Depth layers: graded (see processing/inversion.py:build_mesh's
growth_factor) with a near-surface thickness independent of the
horizontal cell size - vertical and horizontal resolution are different
design choices, and picking the near-surface layer thickness as some
factor of the (often much coarser, cap-limited) horizontal cell size
under-resolves depth for no reason. n_layers is instead solved from the
geometric-series relationship between a target near-surface thickness,
DEFAULT_DEPTH_GROWTH_FACTOR, and the requested depth_extent_m.
"""
from __future__ import annotations

import numpy as np

from .gridding import grid_points

# Per-layer thickness grows by this ratio with each layer down (see
# processing/inversion.py:build_mesh) - 1.1-1.3 is the standard UBC-GIF/
# SimPEG "core mesh" grading range; 1.15 is a moderate middle-of-the-road
# default (about 4x thicker at layer 10 than at the surface).
DEFAULT_DEPTH_GROWTH_FACTOR = 1.15


def spectral_depth_diagnostics(grid_values: np.ndarray, cell_size_m: float) -> dict | None:
    """Spector & Grant (1970) radially-averaged spectral depth estimate,
    with the underlying radial power-spectrum points and fitted band
    included - the shared implementation behind estimate_source_depth_m
    below (which just wants the scalar depth, for mesh auto-sizing) and
    processing/depth_estimation.py:spectral_depth_diagnostic (which wants
    the full diagnostic for a QC plot). Returns None if the grid is too
    small/uniform for a meaningful fit."""
    values = np.asarray(grid_values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 16:
        return None
    fill_value = float(np.nanmean(values[finite]))
    filled = np.where(finite, values, fill_value) - fill_value

    ny, nx = filled.shape
    if ny < 4 or nx < 4:
        return None
    window = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    tapered = filled * window

    spectrum = np.fft.fft2(tapered)
    power = np.abs(spectrum) ** 2
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=cell_size_m)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=cell_size_m)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k = np.sqrt(kx_grid**2 + ky_grid**2)

    k_flat = k.ravel()
    p_flat = power.ravel()
    n_bins = int(np.clip(min(nx, ny) // 2, 5, 30))
    bin_edges = np.linspace(0.0, k_flat.max(), n_bins + 1)
    bin_idx = np.digitize(k_flat, bin_edges)

    radial_k, radial_p = [], []
    for b in range(1, n_bins + 1):
        mask = bin_idx == b
        if mask.sum() < 3:
            continue
        kb = float(k_flat[mask].mean())
        pb = float(p_flat[mask].mean())
        if kb > 0 and pb > 0:
            radial_k.append(kb)
            radial_p.append(pb)

    if len(radial_k) < 4:
        return None
    radial_k = np.array(radial_k)
    ln_p = np.log(np.array(radial_p))

    # Mid-band of the spectrum: the lowest wavenumbers are dominated by
    # regional/edge effects and the highest by grid-sampling noise, so
    # the depth-controlled linear trend is clearest in between.
    lo, hi = np.percentile(radial_k, [15, 70])
    band = (radial_k >= lo) & (radial_k <= hi)
    if band.sum() < 3:
        band = np.ones_like(radial_k, dtype=bool)

    slope, intercept = np.polyfit(radial_k[band], ln_p[band], 1)
    depth = -slope / 2.0
    if not np.isfinite(depth) or depth <= 0:
        return None
    return {
        "depth_m": float(depth),
        "wavenumbers_rad_per_m": radial_k,
        "ln_power": ln_p,
        "fit_band_mask": band,
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
    }


def estimate_source_depth_m(grid_values: np.ndarray, cell_size_m: float) -> float | None:
    """Spector & Grant (1970) radially-averaged spectral depth estimate.
    Returns None if the grid is too small/uniform for a meaningful fit."""
    diag = spectral_depth_diagnostics(grid_values, cell_size_m)
    return diag["depth_m"] if diag is not None else None


def _grid_shape(width: float, height: float, cell_size_m: float) -> tuple[int, int]:
    """Number of grid nodes verde's grid_coordinates (grid-line
    registration) actually produces for a given span/spacing - each axis
    is round(span/spacing) + 1, matching processing/gridding.py's use of
    verde underneath. Needed here because the "+1 per axis" endpoint
    inclusion means a plain area/cell_size^2 estimate systematically
    *undercounts* the real node count - most severely for small, close-
    to-square grids (few tens of cells per axis), where +1 per axis is a
    several-percent effect - so a naive analytic cell size can land
    just over a hard node-count cap after the real grid is built."""
    nx = int(round(width / cell_size_m)) + 1
    ny = int(round(height / cell_size_m)) + 1
    return nx, ny


def suggest_mesh_params(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    line_spacing_m: float | None,
    n_obs_cap: int,
    n_active_cap: int,
) -> dict:
    """Suggest obs_cell_size_m, depth_extent_m, and n_layers from the
    survey's own point cloud, staying within the given size caps."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    width = float(x.max() - x.min())
    height = float(y.max() - y.min())
    area = max(width * height, 1.0)

    half_line_spacing = (line_spacing_m or 20.0) / 2.0
    # Margin under the cap so the analytic area/cell^2 estimate below
    # still respects n_obs_cap once the "+1 per axis" rounding effect
    # from the real discrete grid is accounted for (see _grid_shape).
    obs_cap_margin = 0.85
    cell_for_obs_cap = float(np.sqrt(area / (n_obs_cap * obs_cap_margin)))
    obs_cell_size_m = max(half_line_spacing, cell_for_obs_cap)

    # Grid at that spacing to run the spectral depth estimate on
    # something close to what the inversion will actually observe.
    spectral_cell = max(obs_cell_size_m, 5.0)
    try:
        grid = grid_points(x, y, values, spectral_cell, method="nearest", max_distance_m=max(2 * spectral_cell, line_spacing_m or 0.0))
        depth_estimate = estimate_source_depth_m(grid.values, spectral_cell)
    except ValueError:
        depth_estimate = None

    # The upper bound here is a fixed, physically-motivated ceiling for
    # near-surface drone magnetic surveys (typical flight AGL is tens of
    # meters; sources deeper than this produce anomalies too broad/weak
    # for a local block survey to resolve well anyway) - it must NOT
    # scale with the survey's horizontal footprint. An earlier version
    # used `4 * max(width, height)` as the ceiling, which is physically
    # meaningless (investigation depth has nothing to do with how wide
    # the survey block is) and let large-area surveys balloon into a
    # multi-kilometer "depth_extent_m", which then forced the cell size
    # up (sometimes past the line spacing) just to keep the resulting
    # mesh under the active-cell cap. Fixed to a flat 600 m ceiling.
    if depth_estimate is None:
        depth_extent_m = float(np.clip(3.0 * (line_spacing_m or 50.0), 60.0, 500.0))
    else:
        depth_extent_m = float(np.clip(4.0 * depth_estimate, 60.0, 600.0))

    # Graded layer count (see module docstring): pick a near-surface
    # thickness target independent of the horizontal cell size, then
    # solve how many layers a growth_factor schedule needs to reach
    # depth_extent_m from that starting thickness - the geometric-series
    # inverse of build_mesh's own thickness_k = t0 * growth_factor^k.
    target_top_thickness_m = max(2.0, depth_extent_m / 60.0)
    r = DEFAULT_DEPTH_GROWTH_FACTOR
    n_layers_graded = np.log1p(depth_extent_m * (r - 1.0) / target_top_thickness_m) / np.log(r)
    n_layers = int(np.clip(round(n_layers_graded), 4, 80))

    # cell size floor so nx*ny*n_layers stays within the active-cell cap
    # (nx*ny from area/cell^2, times the now depth-independent n_layers
    # picked above): cell^2 >= area * n_layers / cap.
    cap_margin = 0.75  # safety margin under the hard cap enforced in store.py
    cell_for_active_cap = np.sqrt(area * n_layers / (n_active_cap * cap_margin))
    obs_cell_size_m = max(obs_cell_size_m, cell_for_active_cap)

    # Belt-and-suspenders: verify against the actual discrete grid shape
    # (not just the analytic area/cell^2 approximation above) and nudge
    # the cell size up if it would still land over the cap - guarantees
    # store.run_inversion's own nx*ny <= n_obs_cap check never fails on a
    # freshly auto-suggested cell size, regardless of the survey's aspect
    # ratio or how the "+1 per axis" rounding happens to fall.
    for _ in range(8):
        nx, ny = _grid_shape(width, height, obs_cell_size_m)
        if nx * ny <= n_obs_cap:
            break
        obs_cell_size_m *= 1.05

    return {
        "obs_cell_size_m": float(obs_cell_size_m),
        "depth_extent_m": float(depth_extent_m),
        "n_layers": n_layers,
        "source_depth_estimate_m": depth_estimate,
    }
