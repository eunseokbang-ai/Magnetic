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
import numpy as np
from choclo.prism import magnetic_field
from numba import njit, prange
from PIL import Image

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


def invert(
    G: np.ndarray,
    data_nt: np.ndarray,
    mesh: InversionMesh,
    rows: np.ndarray,
    cols: np.ndarray,
    layers: np.ndarray,
    regularization_strength: float = 1.0,
    n_irls_iterations: int = 5,
) -> InversionResult:
    """Depth-weighted Tikhonov inversion with IRLS compact/focusing
    reweighting (Li & Oldenburg 1996 + Portniaguine & Zhdanov 2002).

    Each iteration solves the regularized normal equations
        (GtG + alpha * diag(depth_weight^2 * compact_weight)) m = Gt d
    with a non-negativity clip, then recomputes the compact ("minimum
    support") weight from the new model before the next pass. No
    discrepancy-principle beta cooling - a fixed iteration count and a
    user-adjustable regularization multiplier are used instead, which
    keeps run time predictable at this mesh scale.

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
    alpha0 = float(np.sum(Gw ** 2)) / n_active * float(regularization_strength)

    # Physically-plausible soft ceiling: even iron-rich rocks rarely exceed
    # a few SI units of susceptibility; this only guards against numerical
    # runaway, it does not otherwise constrain the solution.
    chi_max = 1.0

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

    return InversionResult(
        mesh=mesh,
        susceptibility=susceptibility,
        predicted_nt=predicted,
        observed_nt=np.asarray(data_nt, dtype=float),
        rms_misfit_nt=rms_misfit,
        n_active_cells=n_active,
        n_obs=n_obs,
        iterations=n_iter,
    )


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
    resolution, in local UTM meters). Returns (section, distance_m)
    where section has shape (n_layers, n_samples), row 0 = shallowest."""
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

    return section, distance


def render_section_png(
    section: np.ndarray,
    distance_m: np.ndarray,
    z_centers: np.ndarray,
    cmap_name: str = "geosoft_rainbow",
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict:
    """Render a (n_layers, n_samples) vertical-section array (distance vs.
    elevation, not geo-referenced) as a plain PNG data URL."""
    finite = section[np.isfinite(section)]
    if finite.size == 0:
        raise InversionError("표시할 유효한 역산 값이 없는 단면입니다 (임계값을 낮춰보세요).")
    if vmin is None:
        vmin = float(np.nanmin(finite))
    if vmax is None:
        vmax = float(np.nanmax(finite))
    if vmin == vmax:
        vmin, vmax = vmin - 1e-6, vmax + 1e-6

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]
    rgba = (cmap(norm(section)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(section), 255, 0).astype(np.uint8)

    image = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    image.save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "image_data_url": f"data:image/png;base64,{png_b64}",
        "distance_m": [float(d) for d in distance_m],
        "elevation_m": [float(z) for z in z_centers],
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cmap_name,
    }
