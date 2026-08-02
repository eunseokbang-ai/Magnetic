"""3D magnetic susceptibility inversion, given information 없이 (no prior
geological model) - deployed as an experimental analysis tab alongside the
2D grid/derivative workflow.

Method: Li & Oldenburg (1996) depth-weighted, Tikhonov-regularized linear
inversion of induced-magnetization susceptibility on a rectangular prism
mesh, combined with the Portniaguine & Zhdanov (2002) IRLS "minimum
support" (compact/focusing) reweighting scheme so the recovered anomaly
concentrates into smooth, blocky bodies rather than a diffuse halo. This
is the same family of methods underlying UBC-GIF MAG3D and SimPEG's
magnetic inversion module - the de facto standard approach for this
problem.

Forward model: choclo's numba-JIT rectangular-prism magnetic field kernel
(Fatiando a Terra project - peer reviewed, unit-tested against analytic
solutions), wrapped in a hand-written @njit(parallel=True) double loop to
build the sensitivity/Jacobian matrix (induced magnetization only; no
remanence).

Terrain handling: "active cell" masking (the standard UBC/SimPEG
technique for terrain without a fully terrain-draped mesh) - a flat
tensor mesh spans from the highest ground elevation in the survey area
down by depth_extent_m, and any cell whose center lies above the locally
interpolated ground elevation at that x/y column is marked inactive
(susceptibility fixed at 0, excluded from the inversion unknowns).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO

import matplotlib
import matplotlib.patches
import numpy as np
from choclo.prism import magnetic_field
from numba import njit, prange
from scipy.ndimage import zoom as ndi_zoom

MU0 = 4.0 * np.pi * 1e-7


class InversionError(ValueError):
    pass


@dataclass
class InversionMesh:
    x_centers: np.ndarray  # (nx,) easting, ascending, local UTM meters
    y_centers: np.ndarray  # (ny,) northing, ascending, local UTM meters
    z_centers: np.ndarray  # (nz,) elevation, index 0 = shallowest layer
    cell_size_m: float
    layer_thickness_m: float
    ground_elevation: np.ndarray  # (ny, nx)
    active: np.ndarray  # (ny, nx, nz) bool - False = air/inactive cell


@dataclass
class InversionResult:
    mesh: InversionMesh
    susceptibility: np.ndarray  # (ny, nx, nz) SI, 0 for inactive/air cells
    predicted_nt: np.ndarray  # (n_obs,)
    observed_nt: np.ndarray  # (n_obs,)
    rms_misfit_nt: float
    n_active_cells: int
    n_obs: int
    iterations: int
    regularization_strength_used: float = 1.0
    depth_resolution: dict | None = None  # see resolution_diagnostics()


def build_mesh(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    ground_elevation: np.ndarray,
    cell_size_m: float,
    depth_extent_m: float,
    n_layers: int,
) -> InversionMesh:
    if ground_elevation.shape != (len(y_centers), len(x_centers)):
        raise InversionError("지형 격자와 역산 격자의 크기가 일치하지 않습니다.")
    if not np.isfinite(ground_elevation).any():
        raise InversionError("유효한 지형 고도 값이 없습니다.")

    z_top = float(np.nanmax(ground_elevation))
    layer_thickness = float(depth_extent_m) / int(n_layers)
    z_centers = z_top - layer_thickness * (np.arange(n_layers) + 0.5)

    ny, nx = ground_elevation.shape
    active = np.empty((ny, nx, n_layers), dtype=bool)
    ground_filled = np.where(np.isfinite(ground_elevation), ground_elevation, z_top)
    for k in range(n_layers):
        active[:, :, k] = z_centers[k] < ground_filled

    return InversionMesh(
        x_centers=np.asarray(x_centers, dtype=float),
        y_centers=np.asarray(y_centers, dtype=float),
        z_centers=z_centers,
        cell_size_m=float(cell_size_m),
        layer_thickness_m=layer_thickness,
        ground_elevation=ground_filled,
        active=active,
    )


def _active_prism_bounds(mesh: InversionMesh):
    """Flatten active cells into (n_active, 6) prism bounds
    (west, east, south, north, bottom, top) plus their (row, col, layer)
    grid indices for reassembling results back into the 3D array."""
    dx = mesh.cell_size_m / 2.0
    dz = mesh.layer_thickness_m / 2.0
    rows, cols, layers = np.nonzero(mesh.active)
    x_c = mesh.x_centers[cols]
    y_c = mesh.y_centers[rows]
    z_c = mesh.z_centers[layers]
    bounds = np.column_stack([x_c - dx, x_c + dx, y_c - dx, y_c + dx, z_c - dz, z_c + dz])
    return bounds, rows, cols, layers


@njit(parallel=True, cache=True)
def _build_sensitivity_numba(obs_x, obs_y, obs_z, bounds, mu_e, mu_n, mu_u, f_e, f_n, f_u):
    n_obs = obs_x.shape[0]
    n_active = bounds.shape[0]
    G = np.empty((n_obs, n_active), dtype=np.float64)
    for j in prange(n_active):
        w = bounds[j, 0]
        e = bounds[j, 1]
        s = bounds[j, 2]
        n = bounds[j, 3]
        b = bounds[j, 4]
        t = bounds[j, 5]
        for i in range(n_obs):
            bx, by, bz = magnetic_field(obs_x[i], obs_y[i], obs_z[i], w, e, s, n, b, t, mu_e, mu_n, mu_u)
            G[i, j] = (bx * f_e + by * f_n + bz * f_u) * 1e9
    return G


def build_sensitivity_matrix(
    obs_x: np.ndarray,
    obs_y: np.ndarray,
    obs_z: np.ndarray,
    mesh: InversionMesh,
    inclination_deg: float,
    declination_deg: float,
    field_intensity_nt: float,
):
    """Sensitivity/Jacobian matrix G (n_obs, n_active): the induced-TMI
    response (nT) at each observation for unit (chi=1) susceptibility in
    each active voxel. Returns (G, rows, cols, layers) where the index
    arrays map each G column back to its (row, col, layer) mesh cell."""
    bounds, rows, cols, layers = _active_prism_bounds(mesh)
    if bounds.shape[0] == 0:
        raise InversionError("역산 메쉬에 활성(지하) 셀이 없습니다 - 지형/깊이 설정을 확인하세요.")

    inc = np.radians(inclination_deg)
    dec = np.radians(declination_deg)
    f_e, f_n, f_u = np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)

    b0_tesla = field_intensity_nt * 1e-9
    m_unit = b0_tesla / MU0  # magnetization per unit susceptibility (A/m)
    mu_e, mu_n, mu_u = m_unit * f_e, m_unit * f_n, m_unit * f_u

    G = _build_sensitivity_numba(
        np.ascontiguousarray(obs_x, dtype=np.float64),
        np.ascontiguousarray(obs_y, dtype=np.float64),
        np.ascontiguousarray(obs_z, dtype=np.float64),
        np.ascontiguousarray(bounds, dtype=np.float64),
        mu_e, mu_n, mu_u, f_e, f_n, f_u,
    )
    return G, rows, cols, layers


def resolution_diagnostics(G: np.ndarray, layers: np.ndarray, n_layers: int) -> dict:
    """Cheap per-depth-layer sensitivity summary (mean column L2 norm of
    the already-built, unweighted sensitivity matrix, grouped by layer) -
    not a full model resolution matrix (that would cost an n_active x
    n_active computation this mesh scale can't afford), but a real,
    physically meaningful signal computed from the same G already built
    for the solve: since the raw magnetic response of a unit-susceptibility
    cell genuinely falls off with distance from the sensors, a layer's
    mean sensitivity here is a direct, honest measure of how much the
    survey's own geometry actually constrains that depth - not an
    artifact of the depth-weighting/regularization choices used to
    compensate for it in the solve itself. Surfaced to the user as a
    reliability caveat: depths where this has dropped to a small fraction
    of the shallowest layer's value should be trusted much less than the
    shallow, well-constrained part of the model."""
    col_norm = np.sqrt(np.sum(G**2, axis=0))
    per_layer = np.zeros(n_layers)
    n_cells = np.zeros(n_layers, dtype=int)
    for layer_idx in range(n_layers):
        mask = layers == layer_idx
        if mask.any():
            per_layer[layer_idx] = float(np.mean(col_norm[mask]))
            n_cells[layer_idx] = int(mask.sum())
    peak = float(per_layer.max()) if per_layer.max() > 0 else 1.0
    relative = per_layer / peak
    return {
        "per_layer_mean_sensitivity": per_layer.tolist(),
        "per_layer_relative_sensitivity": relative.tolist(),
        "n_cells_per_layer": n_cells.tolist(),
        # layers whose relative sensitivity has fallen under 5% of the
        # best-resolved layer - a common rule-of-thumb cutoff for "this
        # part of the model is barely constrained by the data at all".
        "poorly_resolved_layers": [i for i, r in enumerate(relative) if r < 0.05 and n_cells[i] > 0],
    }


def _ridge_solve(Gw: np.ndarray, dw: np.ndarray, p: np.ndarray, alpha0: float, identity_obs: np.ndarray, chi_max: float) -> np.ndarray:
    """One regularized normal-equations solve in data space (see invert's
    docstring for the Woodbury/dual reformulation this implements),
    non-negativity clipped. p = 1/(depth_weight^2 * compact_weight)."""
    GP = Gw * p[np.newaxis, :]
    A_reduced = GP @ Gw.T + alpha0 * identity_obs
    lam = np.linalg.solve(A_reduced, dw)
    m = p * (Gw.T @ lam)
    return np.clip(m, 0.0, chi_max)


def _select_regularization_strength(
    Gw: np.ndarray, dw: np.ndarray, p0: np.ndarray, alpha_scale: float, chi_max: float, target_rms_nt: float
) -> float:
    """Discrepancy-principle style search for the regularization_strength
    multiplier that brings the misfit under weighting p0 close to a
    target RMS level derived from the survey's own assumed noise floor -
    the standard, principled way to pick "how much" Tikhonov
    regularization to apply (UBC-GIF/SimPEG "beta search"), instead of
    leaving it purely to the user's own trial-and-error multiplier. p0
    should already fold in a representative compact-weight snapshot (see
    invert(), which builds one before calling this) - searching only
    against the plain depth-weighted (compact_weight=1) misfit
    systematically undershoots because the IRLS focusing that follows
    concentrates the model further and increases the final misfit several-
    fold past what the flat first pass alone predicts (confirmed
    empirically). Only cheap n_obs x n_obs solves are used for the search
    itself - misfit increases monotonically with regularization strength,
    so log-space bisection converges in a fixed, small number of solves
    regardless of mesh size (the expensive part, building G, already
    happened once before this is called)."""
    n_obs = Gw.shape[0]
    identity_obs = np.eye(n_obs)

    def misfit_at(reg_strength: float) -> float:
        alpha0 = alpha_scale * reg_strength
        m = _ridge_solve(Gw, dw, p0, alpha0, identity_obs, chi_max)
        predicted = Gw @ m
        return float(np.sqrt(np.mean((predicted - dw) ** 2)))

    # Misfit-vs-regularization is *not* globally monotonic here, unlike
    # plain (unclipped) Tikhonov: at very low regularization the
    # non-negativity clip on m starts binding hard (the near-unregularized
    # solve wants large positive/negative oscillations to chase noise,
    # which get clipped to 0), which makes the *clipped* misfit rise again
    # instead of continuing to fall toward an unregularized fit - verified
    # empirically (misfit can be several times *worse* at reg_strength
    # 1e-4 than at a moderate 0.3-1). The curve is instead roughly
    # U-shaped: misfit falls from a very regularized/oversmoothed high,
    # bottoms out at some "natural" regularization level, then rises
    # again toward the clipped-noise regime at very low regularization.
    # So a plain bisection (which assumes one monotonic crossing) isn't
    # safe over the whole range - first locate the minimum with a coarse
    # log-spaced scan, then bisect only within the reliably-monotonic
    # increasing branch from there upward.
    grid = np.geomspace(1e-3, 1e3, num=13)
    misfits = np.array([misfit_at(float(g)) for g in grid])
    i_min = int(np.argmin(misfits))

    if misfits[i_min] > target_rms_nt:
        # Even the best achievable point on this grid can't reach the
        # target (noise floor set unrealistically tight, or the survey
        # geometry just can't fit the data that closely) - this is the
        # closest achievable regularization strength.
        return float(grid[i_min])

    if i_min == len(grid) - 1 or misfits[i_min] >= target_rms_nt:
        # Minimum sits at (or past) the top of the grid, or already meets
        # the target - nothing to bisect, just use it.
        return float(grid[i_min])

    lo, hi = float(grid[i_min]), float(grid[-1])
    for _ in range(8):
        if misfit_at(hi) >= target_rms_nt:
            break
        hi *= 10.0

    for _ in range(20):
        mid = float(np.sqrt(lo * hi))
        if misfit_at(mid) > target_rms_nt:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


def invert(
    G: np.ndarray,
    data_nt: np.ndarray,
    mesh: InversionMesh,
    rows: np.ndarray,
    cols: np.ndarray,
    layers: np.ndarray,
    regularization_strength: float = 1.0,
    n_irls_iterations: int = 5,
    assumed_noise_nt: float | None = None,
) -> InversionResult:
    """Depth-weighted Tikhonov inversion with IRLS compact/focusing
    reweighting (Li & Oldenburg 1996 + Portniaguine & Zhdanov 2002).

    Each iteration solves the regularized normal equations
        (GtG + alpha * diag(depth_weight^2 * compact_weight)) m = Gt d
    with a non-negativity clip, then recomputes the compact ("minimum
    support") weight from the new model before the next pass. A fixed
    IRLS iteration count is used (keeps run time predictable at this mesh
    scale) - but the regularization strength itself can now be chosen
    automatically: if assumed_noise_nt is given, regularization_strength
    is *overridden* by a discrepancy-principle search (see
    _select_regularization_strength) targeting that noise level instead
    of using the passed-in value directly.

    The mesh routinely has far more voxels than there are observations
    (n_active >> n_obs), so the normal equations are solved in "data
    space" via the standard Woodbury/dual reformulation instead of
    forming and factorizing the dense n_active x n_active system
    directly: for P = diag(1/(depth_weight^2 * compact_weight)),
        (G P Gt + alpha*I) lambda = d,   m = P Gt lambda
    which only requires an n_obs x n_obs solve - at this problem's scale
    (thousands of voxels, ~1000 observations) that is orders of
    magnitude cheaper and gives the identical m (verified against the
    direct n_active x n_active solve on a small synthetic case).
    """
    n_obs, n_active = G.shape
    if n_active == 0:
        raise InversionError("역산 메쉬에 활성(지하) 셀이 없습니다.")

    Gw = np.asarray(G, dtype=float)
    dw = np.asarray(data_nt, dtype=float)

    ground_at_cell = mesh.ground_elevation[rows, cols]
    z_center = mesh.z_centers[layers]
    depth_below_surface = np.clip(ground_at_cell - z_center, 0.0, None)
    z0 = 0.5 * mesh.cell_size_m
    depth_weight = 1.0 / np.power(depth_below_surface + z0, 1.5)
    depth_weight = depth_weight / np.max(depth_weight)

    # trace(Gt G) == trace(G Gt) == sum of squared entries - avoids ever
    # forming either dense n_active x n_active or n_obs x n_obs product
    # just for this scalar. alpha scales with the data's own units (via
    # sum(G^2)) so regularization_strength alone is a unit-free knob.
    alpha_scale = float(np.sum(Gw ** 2)) / n_active

    # Physically-plausible soft ceiling: even iron-rich rocks rarely exceed
    # a few SI units of susceptibility; this only guards against numerical
    # runaway, it does not otherwise constrain the solution.
    chi_max = 1.0

    if assumed_noise_nt is not None and assumed_noise_nt > 0:
        # A plain depth-weighted-only search systematically undershoots
        # the target (see _select_regularization_strength) because the
        # IRLS compact/focusing passes that follow concentrate the model
        # further and raise the final misfit well past what that flat
        # first pass alone predicts. One quick snapshot solve at a
        # neutral (reg_strength=1) alpha stands in for "what will the
        # compact weighting roughly look like", so the search targets a
        # weighting shaped like what the full IRLS run will actually use.
        identity_obs_snapshot = np.eye(n_obs)
        p_flat = 1.0 / (depth_weight**2)
        m_snapshot = _ridge_solve(Gw, dw, p_flat, alpha_scale, identity_obs_snapshot, chi_max)
        positive_snapshot = m_snapshot[m_snapshot > 0]
        rms_m_snapshot = float(np.sqrt(np.mean(positive_snapshot**2))) if positive_snapshot.size else 1e-6
        eps_snapshot = max(1e-6, 0.05 * rms_m_snapshot)
        compact_weight_snapshot = 1.0 / (m_snapshot**2 + eps_snapshot**2)
        compact_weight_snapshot = compact_weight_snapshot / np.max(compact_weight_snapshot)
        compact_weight_snapshot = np.clip(compact_weight_snapshot, 0.02, 1.0)
        p_snapshot = 1.0 / (depth_weight**2 * compact_weight_snapshot)

        regularization_strength = _select_regularization_strength(
            Gw, dw, p_snapshot, alpha_scale, chi_max, float(assumed_noise_nt)
        )

    alpha0 = alpha_scale * float(regularization_strength)

    m = np.zeros(n_active, dtype=float)
    compact_weight = np.ones(n_active, dtype=float)
    eps = None
    identity_obs = np.eye(n_obs)
    n_iter = max(1, int(n_irls_iterations))
    for it in range(n_iter):
        wm2 = (depth_weight ** 2) * compact_weight
        p = 1.0 / wm2
        GP = Gw * p[np.newaxis, :]
        A_reduced = GP @ Gw.T + alpha0 * identity_obs
        lam = np.linalg.solve(A_reduced, dw)
        m = p * (Gw.T @ lam)
        m = np.clip(m, 0.0, chi_max)

        # The stabilizing epsilon is fixed from the first (plain
        # depth-weighted L2) solve and held constant afterwards - if it
        # were recomputed from each iteration's own model, cells that grew
        # large would keep shrinking their own regularization, creating an
        # unstable positive-feedback loop (verified empirically: rms
        # misfit diverged and chi blew past physical bounds within a few
        # iterations without this fix).
        if eps is None:
            positive = m[m > 0]
            rms_m = float(np.sqrt(np.mean(positive ** 2))) if positive.size else 1e-6
            eps = max(1e-6, 0.05 * rms_m)

        compact_weight = 1.0 / (m ** 2 + eps ** 2)
        compact_weight = compact_weight / np.max(compact_weight)
        compact_weight = np.clip(compact_weight, 0.02, 1.0)

    predicted = G @ m
    residual = predicted - np.asarray(data_nt, dtype=float)
    rms_misfit = float(np.sqrt(np.mean(residual ** 2)))

    susceptibility = np.zeros(mesh.active.shape, dtype=float)
    susceptibility[rows, cols, layers] = m

    n_layers = mesh.active.shape[2]
    depth_resolution = resolution_diagnostics(Gw, layers, n_layers)

    return InversionResult(
        mesh=mesh,
        susceptibility=susceptibility,
        predicted_nt=predicted,
        observed_nt=np.asarray(data_nt, dtype=float),
        rms_misfit_nt=rms_misfit,
        n_active_cells=n_active,
        n_obs=n_obs,
        iterations=n_iter,
        regularization_strength_used=float(regularization_strength),
        depth_resolution=depth_resolution,
    )


def upsample_susceptibility(
    mesh: InversionMesh,
    chi: np.ndarray,
    target_points: int = 120_000,
    max_factor: float = 6.0,
):
    """Resample the (blocky, cell-sized) susceptibility array onto a much
    finer regular grid via trilinear interpolation, purely for a smoother
    3D isosurface render - the inversion itself still solves on the
    original coarse mesh; this only affects how the result looks.

    Order-1 (trilinear) interpolation is used deliberately over a cubic
    spline: it can't overshoot past the local min/max, so it can't invent
    isosurface lobes that aren't supported by the actual solved model.
    Returns (x_centers, y_centers, z_centers, fine_chi) on the finer grid.
    """
    ny, nx, nz = chi.shape
    total_cells = max(ny * nx * nz, 1)
    factor = float(np.clip((target_points / total_cells) ** (1.0 / 3.0), 1.0, max_factor))

    if factor <= 1.0:
        return mesh.x_centers, mesh.y_centers, mesh.z_centers, chi

    fine_chi = ndi_zoom(chi, zoom=factor, order=1, mode="nearest")
    fine_ny, fine_nx, fine_nz = fine_chi.shape
    x_centers = np.linspace(mesh.x_centers[0], mesh.x_centers[-1], fine_nx)
    y_centers = np.linspace(mesh.y_centers[0], mesh.y_centers[-1], fine_ny)
    z_centers = np.linspace(mesh.z_centers[0], mesh.z_centers[-1], fine_nz)
    return x_centers, y_centers, z_centers, fine_chi


def _apply_range(values: np.ndarray, threshold: float | None, threshold_max: float | None) -> np.ndarray:
    """Keep only cells within [threshold, threshold_max] (either bound
    optional), NaN-ing the rest - lets a user isolate one SI band/"덩어리"
    instead of always seeing everything above a single floor."""
    out = values
    if threshold is not None:
        out = np.where(out >= threshold, out, np.nan)
    if threshold_max is not None:
        out = np.where(out <= threshold_max, out, np.nan)
    return out


def horizontal_slice(
    result: InversionResult,
    layer_index: int,
    threshold: float | None = None,
    threshold_max: float | None = None,
) -> np.ndarray:
    """2D (ny, nx) susceptibility slice at the given depth layer (0 =
    shallowest). Air/inactive cells, and cells outside [threshold,
    threshold_max] if given, are set to NaN so
    processing.render.grid_to_png_overlay renders them transparent."""
    n_layers = result.mesh.active.shape[2]
    layer_index = int(np.clip(layer_index, 0, n_layers - 1))
    slice_2d = result.susceptibility[:, :, layer_index].copy()
    active_2d = result.mesh.active[:, :, layer_index]
    slice_2d[~active_2d] = np.nan
    slice_2d = _apply_range(slice_2d, threshold, threshold_max)
    return slice_2d


def vertical_section(
    result: InversionResult,
    path_x: np.ndarray,
    path_y: np.ndarray,
    threshold: float | None = None,
    threshold_max: float | None = None,
):
    """Sample the susceptibility model at every depth layer along an
    arbitrary path (already densified to the desired along-path
    resolution, in local UTM meters). Returns (section, distance_m,
    ground_elevation_m) where section has shape (n_layers, n_samples),
    row 0 = shallowest, and ground_elevation_m is the terrain elevation
    sampled along the same path (for drawing a ground-surface line on
    the rendered section)."""
    mesh = result.mesh
    path_x = np.asarray(path_x, dtype=float)
    path_y = np.asarray(path_y, dtype=float)
    if len(path_x) < 2:
        raise InversionError("수직 단면을 위해서는 2개 이상의 경로 점이 필요합니다.")

    seg_len = np.hypot(np.diff(path_x), np.diff(path_y))
    distance = np.concatenate([[0.0], np.cumsum(seg_len)])

    col_idx = np.argmin(np.abs(mesh.x_centers[None, :] - path_x[:, None]), axis=1)
    row_idx = np.argmin(np.abs(mesh.y_centers[None, :] - path_y[:, None]), axis=1)

    section = result.susceptibility[row_idx, col_idx, :].T  # (n_layers, n_samples)
    active = mesh.active[row_idx, col_idx, :].T
    section = np.where(active, section, np.nan)
    section = _apply_range(section, threshold, threshold_max)
    ground_elevation = mesh.ground_elevation[row_idx, col_idx]

    return section, distance, ground_elevation


def box_faces(result: InversionResult, top_layer_index: int = 0) -> dict:
    """Assemble a "fence diagram" style box for 3D display: the top
    horizontal slice plus susceptibility on the 4 vertical boundary
    walls of the inversion mesh (south/north/west/east), all continuous
    (no threshold gating) and sharing one SI colorscale - mirrors the
    standard published-figure style of a colored box with a distinct
    top surface and 4 colored side faces, as an alternative to the
    single-threshold isosurface "blob" view."""
    mesh = result.mesh
    chi = result.susceptibility
    active = mesh.active
    x, y, z = mesh.x_centers, mesh.y_centers, mesh.z_centers
    ny, nx, nz = chi.shape
    top_layer_index = int(np.clip(top_layer_index, 0, nz - 1))

    def masked(values, mask):
        out = values.astype(float).copy()
        out[~mask] = np.nan
        return out

    x_grid, y_grid = np.meshgrid(x, y)  # (ny, nx)
    top = {
        "x": x_grid.tolist(),
        "y": y_grid.tolist(),
        "z": np.full_like(x_grid, float(z[top_layer_index])).tolist(),
        "value": masked(chi[:, :, top_layer_index], active[:, :, top_layer_index]).tolist(),
    }

    def wall_along_x(row_idx: int) -> dict:
        # a vertical wall at fixed y (south/north boundary): distance runs
        # east-west, so the coordinate grid is (x, z).
        x_grid_w, z_grid_w = np.meshgrid(x, z)  # (nz, nx)
        val = masked(chi[row_idx, :, :].T, active[row_idx, :, :].T)  # (nz, nx)
        return {
            "x": x_grid_w.tolist(),
            "y": np.full_like(x_grid_w, float(y[row_idx])).tolist(),
            "z": z_grid_w.tolist(),
            "value": val.tolist(),
        }

    def wall_along_y(col_idx: int) -> dict:
        # a vertical wall at fixed x (west/east boundary): distance runs
        # north-south, so the coordinate grid is (y, z).
        y_grid_w, z_grid_w = np.meshgrid(y, z)  # (nz, ny)
        val = masked(chi[:, col_idx, :].T, active[:, col_idx, :].T)  # (nz, ny)
        return {
            "x": np.full_like(y_grid_w, float(x[col_idx])).tolist(),
            "y": y_grid_w.tolist(),
            "z": z_grid_w.tolist(),
            "value": val.tolist(),
        }

    active_chi = chi[chi > 0]
    return {
        "top": top,
        "south": wall_along_x(0),
        "north": wall_along_x(ny - 1),
        "west": wall_along_y(0),
        "east": wall_along_y(nx - 1),
        "vmin": 0.0,
        "vmax": float(active_chi.max()) if active_chi.size else 1.0,
        "top_layer_index": top_layer_index,
        "top_elevation_m": float(z[top_layer_index]),
        "n_layers": int(nz),
    }


_PROFILE_LABELS = {"custom": "자유선", "ew": "동서", "ns": "남북"}


def render_section_png(
    section: np.ndarray,
    distance_m: np.ndarray,
    z_centers: np.ndarray,
    ground_elevation_m: np.ndarray,
    path_x: np.ndarray,
    path_y: np.ndarray,
    mesh_x_centers: np.ndarray,
    mesh_y_centers: np.ndarray,
    profile: str = "custom",
    cmap_name: str = "geosoft_rainbow",
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict:
    """Render a (n_layers, n_samples) vertical-section array as a proper
    labeled figure: real distance (m) / elevation (m) axes, a terrain
    line marking the ground surface, a colorbar, and a small locator
    inset showing where this profile sits within the full inversion
    mesh extent (project area context) - addresses the request to show
    depth/x/y coordinates and the section's location on the map."""
    finite = section[np.isfinite(section)]
    if finite.size == 0:
        raise InversionError("표시할 유효한 역산 값이 없는 단면입니다 (임계값을 낮춰보세요).")
    if vmin is None:
        vmin = float(np.nanmin(finite))
    if vmax is None:
        vmax = float(np.nanmax(finite))
    if vmin == vmax:
        vmin, vmax = vmin - 1e-6, vmax + 1e-6

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    dist_max = float(distance_m[-1]) if len(distance_m) else 1.0
    z_lo, z_hi = float(z_centers[-1]), float(z_centers[0])  # ascending: deepest -> shallowest

    fig = Figure(figsize=(11.0, 6.5), dpi=110)
    FigureCanvasAgg(fig)
    fig.set_facecolor("white")
    matplotlib.rcParams["font.family"] = "NanumGothic"  # Hangul labels need a CJK-capable font
    matplotlib.rcParams["axes.unicode_minus"] = False  # NanumGothic lacks the U+2212 minus glyph
    ax = fig.add_axes([0.09, 0.13, 0.78, 0.75])

    cmap = matplotlib.colormaps[cmap_name]
    im = ax.imshow(
        section, extent=[0.0, dist_max, z_lo, z_hi], origin="upper", aspect="auto",
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax.plot(distance_m, ground_elevation_m, color="black", linewidth=1.2, label="지표면")
    ax.set_xlabel("측선을 따른 거리 (m)")
    ax.set_ylabel("고도 (m)")
    profile_label = _PROFILE_LABELS.get(profile, profile)
    ax.set_title(f"수직 단면 - {profile_label} (자화율, SI)")
    ax.set_xlim(0.0, dist_max)
    ax.set_ylim(z_lo, z_hi)
    ax.text(0.01, 1.02, "A", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.text(0.99, 1.02, "A'", transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right")

    cbar_ax = fig.add_axes([0.885, 0.13, 0.02, 0.75])
    fig.colorbar(im, cax=cbar_ax, label="자화율 (SI)")

    # Locator inset: full mesh (project inversion) extent as a light-gray
    # box, this profile's path drawn on top, so the user can see where
    # the cross-section sits within the overall surveyed area.
    loc_ax = ax.inset_axes([0.01, 0.62, 0.32, 0.36])
    x_lo, x_hi = float(np.min(mesh_x_centers)), float(np.max(mesh_x_centers))
    y_lo, y_hi = float(np.min(mesh_y_centers)), float(np.max(mesh_y_centers))
    pad_x = 0.05 * max(x_hi - x_lo, 1.0)
    pad_y = 0.05 * max(y_hi - y_lo, 1.0)
    loc_ax.add_patch(
        matplotlib.patches.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo, facecolor="#e5e7eb", edgecolor="#9ca3af", linewidth=0.8)
    )
    loc_ax.plot(path_x, path_y, color="#dc2626", linewidth=1.8)
    loc_ax.text(path_x[0], path_y[0], "A", fontsize=8, fontweight="bold", color="#dc2626")
    loc_ax.text(path_x[-1], path_y[-1], "A'", fontsize=8, fontweight="bold", color="#dc2626", ha="right")
    loc_ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    loc_ax.set_ylim(y_lo - pad_y, y_hi + pad_y)
    loc_ax.set_aspect("equal")
    loc_ax.set_title("위치 (전체 조사구역 내)", fontsize=7.5)
    loc_ax.tick_params(labelsize=6)
    loc_ax.set_facecolor("white")
    for spine in loc_ax.spines.values():
        spine.set_linewidth(0.6)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "image_data_url": f"data:image/png;base64,{png_b64}",
        "distance_m": [float(d) for d in distance_m],
        "elevation_m": [float(z) for z in z_centers],
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cmap_name,
    }
