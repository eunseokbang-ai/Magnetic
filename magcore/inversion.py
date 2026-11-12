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

from . import progress
from choclo.prism import magnetic_field
from numba import njit, prange
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import distance_transform_edt

MU0 = 4.0 * np.pi * 1e-7
# Newton steps the bounded solve may take. It ends by itself in the step
# after the set of cells at a bound stops changing: 30 steps from a cold
# start on GaSaDo, 301 on Andong with its level removed (13,046 of 15,395
# cells end at zero there, and the deep ones, whose weights are a million
# times the shallow ones', come off their bound a few per step). The
# limit is an order of magnitude above that; it only keeps a pathological
# case from running on.
_BOUNDED_SOLVE_MAX_STEPS = 3000
# The regularization search (_select_regularization_strength) stops
# weakening when half a decade more buys less than this fraction of
# misfit, or at this strength.
_SEARCH_MIN_GAIN = 0.10
_SEARCH_WEAKEST = 1e-3


class InversionError(ValueError):
    pass


@dataclass
class InversionMesh:
    x_centers: np.ndarray  # (nx,) easting, ascending, local UTM meters
    y_centers: np.ndarray  # (ny,) northing, ascending, local UTM meters
    z_centers: np.ndarray  # (nz,) elevation, index 0 = shallowest layer, descending
    cell_size_m: float
    layer_thickness_m: np.ndarray  # (nz,) per-layer thickness, index 0 = shallowest (see build_mesh's growth_factor)
    ground_elevation: np.ndarray  # (ny, nx)
    active: np.ndarray  # (ny, nx, nz) bool - False = air/inactive cell
    # (ny, nx) bool - True where the column has data above it. The mesh is
    # the survey's bounding rectangle and a survey seldom fills it; the
    # columns outside are either left out of the solve or kept as padding
    # (see build_mesh), and in neither case read as a result - see
    # interpretable_susceptibility. None = every column counts (a model
    # from before this was recorded, or one built without it).
    data_support: np.ndarray | None = None


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
    growth_factor: float = 1.0,
    data_support: np.ndarray | None = None,
    padding_cells: float | None = None,
) -> InversionMesh:
    """data_support marks the columns that have data above them, and
    padding_cells how far beyond them (in cells) the model reaches: the
    columns farther out are inactive. The columns within reach but without
    data are padding - cells the solve may fill to explain what sources
    outside the survey do to the data at its edge, and which nothing
    constrains from above. They are part of the model and its predicted
    data, and are not shown (interpretable_susceptibility). Without
    data_support every column of the rectangle is active and counts, as
    before.

    growth_factor grades the per-layer thickness geometrically with
    depth (thickness_k = thickness_0 * growth_factor^k, k=0 = shallowest)
    instead of dividing depth_extent_m into n_layers equal slabs - the
    standard UBC-GIF/SimPEG "core mesh" scheme (Li & Oldenburg's own
    published meshes use exactly this grading): finer cells near the
    surface, where the survey geometry can actually resolve detail and
    where the recovered top-of-body shape matters most for reading off
    real terrain/near-surface structure, coarser cells at depth, where
    potential-field data's inherent falloff with distance means no
    survey resolves fine structure there regardless of how small the
    cells are made. growth_factor=1.0 (the default) reproduces the
    original uniform-thickness mesh exactly."""
    if ground_elevation.shape != (len(y_centers), len(x_centers)):
        raise InversionError("지형 격자와 역산 격자의 크기가 일치하지 않습니다.")
    if not np.isfinite(ground_elevation).any():
        raise InversionError("유효한 지형 고도 값이 없습니다.")

    z_top = float(np.nanmax(ground_elevation))
    n_layers = int(n_layers)
    growth_factor = float(growth_factor)
    if abs(growth_factor - 1.0) < 1e-9:
        layer_thickness = np.full(n_layers, float(depth_extent_m) / n_layers)
    else:
        # Geometric series: depth_extent_m = t0 * (r^n - 1) / (r - 1),
        # solved for t0 given the requested growth ratio r and layer count n.
        t0 = float(depth_extent_m) * (growth_factor - 1.0) / (growth_factor**n_layers - 1.0)
        layer_thickness = t0 * growth_factor ** np.arange(n_layers)
    layer_bottom_depth = np.cumsum(layer_thickness)
    layer_top_depth = np.concatenate([[0.0], layer_bottom_depth[:-1]])
    z_centers = z_top - 0.5 * (layer_top_depth + layer_bottom_depth)

    ny, nx = ground_elevation.shape
    active = np.empty((ny, nx, n_layers), dtype=bool)
    ground_filled = np.where(np.isfinite(ground_elevation), ground_elevation, z_top)
    for k in range(n_layers):
        active[:, :, k] = z_centers[k] < ground_filled

    support = None
    if data_support is not None:
        support = np.asarray(data_support, dtype=bool)
        if support.shape != (ny, nx):
            raise InversionError("자료 범위 격자와 역산 격자의 크기가 일치하지 않습니다.")
        if not support.any():
            raise InversionError("자료가 있는 격자 칸이 없습니다.")
        if padding_cells is not None:
            within_reach = distance_transform_edt(~support) <= float(padding_cells) + 1e-9
            active &= within_reach[:, :, None]

    return InversionMesh(
        x_centers=np.asarray(x_centers, dtype=float),
        y_centers=np.asarray(y_centers, dtype=float),
        z_centers=z_centers,
        cell_size_m=float(cell_size_m),
        layer_thickness_m=layer_thickness,
        ground_elevation=ground_filled,
        active=active,
        data_support=support,
    )


def _active_prism_bounds(mesh: InversionMesh):
    """Flatten active cells into (n_active, 6) prism bounds
    (west, east, south, north, bottom, top) plus their (row, col, layer)
    grid indices for reassembling results back into the 3D array."""
    dx = mesh.cell_size_m / 2.0
    rows, cols, layers = np.nonzero(mesh.active)
    x_c = mesh.x_centers[cols]
    y_c = mesh.y_centers[rows]
    z_c = mesh.z_centers[layers]
    dz = mesh.layer_thickness_m[layers] / 2.0  # per-layer thickness (graded mesh - see build_mesh)
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
    col_norm = np.sqrt(np.einsum("ij,ij->j", G, G))
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


class _FreeGram:
    """G_F P_F G_F^T for a set F of mesh cells, kept up to date as F changes.

    The bounded solve (_ridge_solve) needs this n_obs x n_obs product for
    the cells that are strictly between their bounds, and that set changes
    from one Newton step to the next - by a few cells late on, by half the
    mesh at the start. Forming the product is the expensive part of the
    whole inversion (n_obs^2 * n_cells), so it is updated by the columns
    that entered or left instead of being rebuilt, whenever those are the
    smaller number. Valid for one (G, p); p changes with every IRLS pass,
    so each pass has its own."""

    def __init__(self, Gw: np.ndarray, p: np.ndarray):
        self.Gw = Gw
        self.sqrt_p = np.sqrt(p)
        self.free: np.ndarray | None = None
        self.A: np.ndarray | None = None

    def _gram(self, mask: np.ndarray) -> np.ndarray:
        if mask.all():
            Gs = self.Gw * self.sqrt_p[np.newaxis, :]
        else:
            cols = np.flatnonzero(mask)
            if len(cols) == 0:
                return np.zeros((self.Gw.shape[0], self.Gw.shape[0]))
            Gs = self.Gw[:, cols] * self.sqrt_p[np.newaxis, cols]
        return Gs @ Gs.T        # numpy does a @ a.T as a symmetric rank-k update: half a general product

    def get(self, free: np.ndarray) -> np.ndarray:
        if self.A is None:
            self.A = self._gram(free)
        else:
            entered = free & ~self.free
            left = self.free & ~free
            n_changed = int(entered.sum()) + int(left.sum())
            if n_changed > 0.5 * int(free.sum()):
                self.A = self._gram(free)
            elif n_changed:
                if entered.any():
                    self.A += self._gram(entered)
                if left.any():
                    self.A -= self._gram(left)
        self.free = free.copy()
        return self.A


def _ridge_solve(
    Gw: np.ndarray, dw: np.ndarray, p: np.ndarray, alpha0: float, chi_max: float, m_ref: np.ndarray,
    gram: _FreeGram | None = None, m_start: np.ndarray | None = None,
) -> np.ndarray:
    """_bounded_solve, or an InversionError if it did not converge - an
    unconverged model is feasible but can be far from fitting anything,
    and must not be handed on as a result."""
    m, converged = _bounded_solve(Gw, dw, p, alpha0, chi_max, m_ref, gram=gram, m_start=m_start)
    if not converged:
        raise InversionError(
            "역산 풀이가 수렴하지 않았습니다 - 정규화 강도가 너무 약합니다. 정규화 강도를 높이거나 "
            "'정규화 강도 자동 선택'을 사용하세요."
        )
    return m


def _bounded_solve(
    Gw: np.ndarray, dw: np.ndarray, p: np.ndarray, alpha0: float, chi_max: float, m_ref: np.ndarray,
    gram: _FreeGram | None = None, m_start: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """The regularized least-squares model with every cell between 0 and
    chi_max:

        minimize  ||G m - d||^2 + alpha0 * sum((m - m_ref)^2 / p)
        subject to 0 <= m <= chi_max

    p = 1/(depth_weight^2 * compact_weight); m_ref is the reference model
    the solution is pulled toward (zeros when there is none).

    Until 2026-11 this solved the problem without the bounds and then
    clipped the negative cells to zero. That is not the constrained
    solution, and it is not close to it: the unconstrained model fits the
    data *with* its negative cells, so removing them leaves a model whose
    field overshoots everywhere they were compensating. On the GaSaDo
    survey (strong negative anomalies) the clipped model missed the data
    by 436 nT rms where the data themselves vary by 64 nT; on Andong by
    708 nT against 438 nT.

    How it is solved. With lambda = (d - G m) / alpha0, the optimality
    conditions say m = clip(m_ref + p * G^T lambda, 0, chi_max): each cell
    is where the unconstrained formula puts it, or at the bound it ran
    into. Substituting gives n_obs equations in lambda,

        alpha0 * lambda + G clip(m_ref + p G^T lambda, 0, chi_max) = d,

    which are the gradient of a strongly convex function of lambda, so
    there is exactly one solution and Newton's method with a line search
    on that function reaches it from anywhere. Each step is one solve of

        (G_F P_F G_F^T + alpha0 I) delta = -(residual of the equations),

    F being the cells currently between their bounds - the same
    data-space (Woodbury) solve as before, restricted to the free cells.
    The equations are linear wherever F does not change, so the iteration
    ends exactly, in the step after F settles. With no cell at a bound
    the first step *is* the old unconstrained solve, and the result is
    identical to it.

    The line search is exact, and has to be. Along a Newton direction the
    function is piecewise quadratic with a kink wherever a cell reaches a
    bound - thousands of them - and its slope there is cheap to evaluate
    (no product with G), so the minimum along the line is found by
    bisection on the slope. With plain backtracking the same iteration
    was still 70 nT from the answer after 25 steps on the GaSaDo survey
    (1,884 data, 14,274 cells, 5,546 of them ending at zero); with the
    exact search it is done in 30, and reaches a lower objective than
    6,000 iterations of projected FISTA do.

    The weaker the regularization, the more steps it takes (same survey,
    plain depth weighting: 11 steps at strength 1000, 28 at 1, 90 at 0.01,
    250 at 0.001), and so do data that leave most of the mesh at zero
    (Andong: 301 at strength 1). Far outside the useful range it runs
    into _BOUNDED_SOLVE_MAX_STEPS. Returns (m, converged); m is within its
    bounds either way, but only a converged m is the solution.

    gram: the free-cell product cache (_FreeGram) to use and update, for a
    caller making several solves with the same p (the regularization
    search). m_start: a model to start from (the previous IRLS pass, the
    previous probe of the search) - it only saves steps, the answer does
    not depend on it."""
    if gram is None:
        gram = _FreeGram(Gw, p)
    n_obs = Gw.shape[0]
    lam = np.zeros(n_obs) if m_start is None else (dw - Gw @ m_start) / alpha0
    gtl = Gw.T @ lam
    # Done when the equations hold to a part in 10^7 of the data (or of the
    # reference model's field, when the data are zero). Far inside any
    # noise; it is there for the degenerate case of a cell that belongs
    # exactly on its bound and flickers across it in the last bits, which
    # the free-set test below would otherwise never see settle.
    scale = max(float(np.sqrt(np.mean(dw ** 2))), float(np.sqrt(np.mean((Gw @ np.clip(m_ref, 0.0, chi_max)) ** 2))))
    tol = 1e-7 * max(scale, 1e-30)

    z = m_ref + p * gtl
    m = np.clip(z, 0.0, chi_max)
    converged = False
    for _ in range(_BOUNDED_SOLVE_MAX_STEPS):
        progress.check()        # a long solve stays cancellable
        equations = alpha0 * lam + Gw @ m - dw
        if float(np.sqrt(np.mean(equations ** 2))) <= tol:
            converged = True
            break
        free = (z >= 0.0) & (z <= chi_max)
        J = gram.get(free).copy()
        J[np.diag_indices_from(J)] += alpha0
        delta = cho_solve(cho_factor(J, lower=True, overwrite_a=True, check_finite=False), -equations, check_finite=False)
        gtd = Gw.T @ delta

        # slope of the function along delta, at step t: nondecreasing in t,
        # negative at 0. The full step if it is still downhill there,
        # otherwise the step where the slope crosses zero.
        a0 = alpha0 * float(lam @ delta) - float(dw @ delta)
        a1 = alpha0 * float(delta @ delta)
        pg = p * gtd

        def slope(t: float) -> float:
            return a0 + a1 * t + float(np.clip(z + t * pg, 0.0, chi_max) @ gtd)

        step = 1.0
        if slope(1.0) > 0.0:
            lo, hi = 0.0, 1.0
            for _bisect in range(50):
                mid = 0.5 * (lo + hi)
                if slope(mid) > 0.0:
                    hi = mid
                else:
                    lo = mid
            step = 0.5 * (lo + hi)
        lam = lam + step * delta
        gtl = gtl + step * gtd
        z = m_ref + p * gtl
        m = np.clip(z, 0.0, chi_max)
        if step == 1.0 and np.array_equal(free, (z >= 0.0) & (z <= chi_max)):
            converged = True    # the free set held through a full step: that step solved the equations exactly
            break
    return m, converged


def _select_regularization_strength(
    Gw: np.ndarray, dw: np.ndarray, p0: np.ndarray, alpha_scale: float, chi_max: float, target_rms_nt: float, m_ref: np.ndarray
) -> float:
    """Discrepancy-principle search for the regularization_strength
    multiplier that brings the misfit under weighting p0 to a target RMS
    level derived from the survey's assumed noise floor - the standard,
    principled way to pick "how much" Tikhonov regularization to apply
    (UBC-GIF/SimPEG "beta search"), instead of leaving it to trial and
    error. p0 should already fold in a representative compact-weight
    snapshot (see invert(), which builds one before calling this) -
    searching against the plain depth-weighted misfit undershoots,
    because the IRLS focusing that follows concentrates the model and
    raises the final misfit past what the flat first pass predicts.

    The misfit of the constrained solution never decreases as the
    regularization grows (true of Tikhonov regularization over any convex
    set), so there is one crossing to find. (The search used to scan a
    grid for a minimum first: with the old clipped solve the misfit rose
    again at weak regularization, an artefact of the clip - see
    _bounded_solve.)

    It is looked for from the strong side down, half a decade at a time,
    each solve starting from the last: strongly regularized solves are
    the cheap ones (see _bounded_solve), and the walk usually ends long
    before the expensive ones. It ends

      * where the misfit first comes under the target - the crossing is
        then pinned down by bisection inside that half decade; or
      * where another half decade of weakening buys less than
        _SEARCH_MIN_GAIN of misfit, or the bottom of the range
        (_SEARCH_WEAKEST) is reached, or a solve does not converge. The
        target is then out of reach - a noise floor set too tight, or
        data an induced-only model cannot explain (remanence) - and
        weakening further would trade a rougher model for no better fit.

    Every probe is a bounded solve with the same p0, so they share one
    free-cell product cache."""
    gram = _FreeGram(Gw, p0)
    state: dict = {"m": None, "n": 0}

    def misfit_at(reg_strength: float) -> tuple[float, bool]:
        progress.report("정규화 강도 탐색", min(0.5, 0.5 * state["n"] / 20.0))
        state["n"] += 1
        m, converged = _bounded_solve(Gw, dw, p0, alpha_scale * reg_strength, chi_max, m_ref, gram=gram, m_start=state["m"])
        if converged:
            state["m"] = m
        return float(np.sqrt(np.mean((Gw @ m - dw) ** 2))), converged

    target = float(target_rms_nt)
    hi = 1e3
    hi_misfit, _ = misfit_at(hi)
    for _ in range(8):
        if hi_misfit >= target:
            break
        hi *= 10.0
        hi_misfit, _ = misfit_at(hi)
    else:
        return float(hi)        # fits within the target however hard it is regularized

    step = 10.0 ** 0.5
    while True:
        lo = hi / step
        lo_misfit, converged = misfit_at(lo)
        if not converged:
            return float(hi)
        if lo_misfit <= target:
            break
        if lo_misfit > (1.0 - _SEARCH_MIN_GAIN) * hi_misfit or lo <= _SEARCH_WEAKEST:
            return float(lo)
        hi, hi_misfit = lo, lo_misfit

    for _ in range(7):        # pins the strength down to within 1 %
        mid = float(np.sqrt(lo * hi))
        mid_misfit, converged = misfit_at(mid)
        if not converged or mid_misfit <= target:
            lo = mid
        else:
            hi = mid
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
    m_ref: np.ndarray | None = None,
) -> InversionResult:
    """Depth-weighted Tikhonov inversion with IRLS compact/focusing
    reweighting (Li & Oldenburg 1996 + Portniaguine & Zhdanov 2002).

    Each iteration solves the regularized least-squares problem
        min ||G m - d||^2 + alpha * sum(depth_weight^2 * compact_weight * (m - m_ref)^2)
    with every cell held between 0 and chi_max (a real bound-constrained
    solve, see _ridge_solve - not a clip of the unconstrained answer),
    then recomputes the compact ("minimum support") weight from the new
    model before the next pass. A fixed
    IRLS iteration count is used (keeps run time predictable at this mesh
    scale) - but the regularization strength itself can now be chosen
    automatically: if assumed_noise_nt is given, regularization_strength
    is *overridden* by a discrepancy-principle search (see
    _select_regularization_strength) targeting that noise level instead
    of using the passed-in value directly.

    m_ref: optional reference model (n_active,), one susceptibility value
    per active cell in the same (rows, cols, layers) order the caller
    used to build G - when given, every regularization term below pulls
    the solution toward m_ref instead of toward zero, and the IRLS
    compact/focusing weight concentrates *deviations from m_ref* into
    blocky anomalies rather than concentrating the raw susceptibility
    itself (Li & Oldenburg's own reference-model mechanism: given
    external geological information - e.g. a user-digitized geology map,
    see store.py's geology_units - reduces potential-field inversion's
    inherent non-uniqueness by starting the answer from a geologically
    plausible background instead of "quiet earth"). None (default)
    reproduces the original toward-zero behavior exactly.

    The mesh routinely has far more voxels than there are observations
    (n_active >> n_obs), so the normal equations are solved in "data
    space" via the standard Woodbury/dual reformulation instead of
    forming and factorizing the dense n_active x n_active system
    directly: for P = diag(1/(depth_weight^2 * compact_weight)),
        (G P Gt + alpha*I) lambda = d - G@m_ref,   m = m_ref + P Gt lambda
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
    if m_ref is None:
        m_ref_arr = np.zeros(n_active, dtype=float)
    else:
        m_ref_arr = np.asarray(m_ref, dtype=float)
        if m_ref_arr.shape[0] != n_active:
            raise InversionError("reference model 크기가 활성 셀 수와 일치하지 않습니다.")

    ground_at_cell = mesh.ground_elevation[rows, cols]
    z_center = mesh.z_centers[layers]
    depth_below_surface = np.clip(ground_at_cell - z_center, 0.0, None)
    # Reference offset in the standard Li & Oldenburg depth-weighting
    # formula 1/(z+z0)^1.5 - conventionally the mesh's own near-surface
    # cell dimension, to avoid a singularity right at zero depth. Uses
    # the shallowest layer's thickness (not the horizontal cell size)
    # since a graded mesh's finest cells - the ones this offset is meant
    # to represent - now sit in z, not x/y (see build_mesh's growth_factor).
    z0 = 0.5 * float(np.min(mesh.layer_thickness_m))
    depth_weight = 1.0 / np.power(depth_below_surface + z0, 1.5)
    depth_weight = depth_weight / np.max(depth_weight)

    # trace(Gt G) == trace(G Gt) == sum of squared entries - avoids ever
    # forming either dense n_active x n_active or n_obs x n_obs product
    # just for this scalar. alpha scales with the data's own units (via
    # sum(G^2)) so regularization_strength alone is a unit-free knob.
    alpha_scale = float(np.einsum("ij,ij->", Gw, Gw)) / n_active

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
        p_flat = 1.0 / (depth_weight**2)
        m_snapshot = _ridge_solve(Gw, dw, p_flat, alpha_scale, chi_max, m_ref_arr)
        # Compact/focusing weighting concentrates *deviations from the
        # reference model* (m_ref_arr, zeros when there's no reference)
        # into blocky anomalies, not the raw susceptibility itself - a
        # known-magnetite unit's own background susceptibility shouldn't
        # count against it as "not compact".
        deviation_snapshot = m_snapshot - m_ref_arr
        dev_nonzero_snapshot = deviation_snapshot[deviation_snapshot != 0]
        rms_dev_snapshot = float(np.sqrt(np.mean(dev_nonzero_snapshot**2))) if dev_nonzero_snapshot.size else 1e-6
        eps_snapshot = max(1e-6, 0.05 * rms_dev_snapshot)
        compact_weight_snapshot = 1.0 / (deviation_snapshot**2 + eps_snapshot**2)
        compact_weight_snapshot = compact_weight_snapshot / np.max(compact_weight_snapshot)
        compact_weight_snapshot = np.clip(compact_weight_snapshot, 0.02, 1.0)
        p_snapshot = 1.0 / (depth_weight**2 * compact_weight_snapshot)

        regularization_strength = _select_regularization_strength(
            Gw, dw, p_snapshot, alpha_scale, chi_max, float(assumed_noise_nt), m_ref_arr
        )

    alpha0 = alpha_scale * float(regularization_strength)

    m = m_ref_arr.copy()
    compact_weight = np.ones(n_active, dtype=float)
    eps = None
    n_iter = max(1, int(n_irls_iterations))
    for it in range(n_iter):
        progress.report("반복 역산 (IRLS)", 0.5 + 0.5 * it / n_iter)
        wm2 = (depth_weight ** 2) * compact_weight
        p = 1.0 / wm2
        m = _ridge_solve(Gw, dw, p, alpha0, chi_max, m_ref_arr, m_start=m if it else None)

        # The stabilizing epsilon is fixed from the first (plain
        # depth-weighted L2) solve and held constant afterwards - if it
        # were recomputed from each iteration's own model, cells that grew
        # large would keep shrinking their own regularization, creating an
        # unstable positive-feedback loop (verified empirically: rms
        # misfit diverged and chi blew past physical bounds within a few
        # iterations without this fix). Deviation from m_ref (zeros when
        # there's no reference model) is what "compact" means here - see
        # the snapshot solve above.
        deviation = m - m_ref_arr
        if eps is None:
            dev_nonzero = deviation[deviation != 0]
            rms_dev = float(np.sqrt(np.mean(dev_nonzero ** 2))) if dev_nonzero.size else 1e-6
            eps = max(1e-6, 0.05 * rms_dev)

        compact_weight = 1.0 / (deviation ** 2 + eps ** 2)
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


def interpretable_cells(mesh: InversionMesh) -> np.ndarray:
    """(ny, nx, nz) bool - the cells whose susceptibility is a result:
    below ground and under a column with data above it. The padding
    around the survey is in the model (the solve needs somewhere to put
    what lies outside) and is not one of them: with no data above, what
    the solve puts there is whatever is cheapest, not what is there."""
    if mesh.data_support is None:
        return mesh.active
    return mesh.active & mesh.data_support[:, :, None]


def interpretable_susceptibility(result: InversionResult, fill: float = np.nan) -> np.ndarray:
    """The model as it is shown and summarised: susceptibility where
    interpretable_cells, `fill` everywhere else (NaN for the slices and
    sections, which draw it transparent; 0 for the iso-surface)."""
    return np.where(interpretable_cells(result.mesh), result.susceptibility.astype(float), fill)


def padding_share(result: InversionResult) -> float | None:
    """The fraction of the model (susceptibility x volume) that sits in
    the padding, outside the data - how much of the fit rests on sources
    the survey did not fly over. None when the mesh records no support."""
    mesh = result.mesh
    if mesh.data_support is None:
        return None
    weight = result.susceptibility * mesh.layer_thickness_m[None, None, :]
    total = float(weight[mesh.active].sum())
    if total <= 0:
        return 0.0
    return float(weight[mesh.active & ~mesh.data_support[:, :, None]].sum() / total)


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
    Interpolates by real physical position (scipy.interpolate.
    RegularGridInterpolator over the mesh's actual x/y/z center
    coordinates) rather than by array index (the previous
    scipy.ndimage.zoom implementation) - with a depth-graded mesh
    (build_mesh's growth_factor) z_centers is no longer evenly spaced, so
    interpolating by index would silently misplace values relative to
    their true depth. Returns (x_centers, y_centers, z_centers, fine_chi)
    on a finer grid, uniformly spaced in x/y/z for a clean render.
    """
    ny, nx, nz = chi.shape
    total_cells = max(ny * nx * nz, 1)
    factor = float(np.clip((target_points / total_cells) ** (1.0 / 3.0), 1.0, max_factor))

    if factor <= 1.0:
        return mesh.x_centers, mesh.y_centers, mesh.z_centers, chi

    fine_nx = max(nx, int(round(nx * factor)))
    fine_ny = max(ny, int(round(ny * factor)))
    fine_nz = max(nz, int(round(nz * factor)))
    x_centers = np.linspace(mesh.x_centers[0], mesh.x_centers[-1], fine_nx)
    y_centers = np.linspace(mesh.y_centers[0], mesh.y_centers[-1], fine_ny)
    z_centers = np.linspace(mesh.z_centers[0], mesh.z_centers[-1], fine_nz)

    # mesh.z_centers runs descending (index 0 = shallowest/highest
    # elevation - see InversionMesh); RegularGridInterpolator requires
    # each axis strictly ascending, so flip both the axis and the data
    # to match before building the interpolator.
    z_axis_asc = mesh.z_centers[::-1]
    chi_asc = chi[:, :, ::-1]
    interpolator = RegularGridInterpolator(
        (mesh.y_centers, mesh.x_centers, z_axis_asc), chi_asc,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    Yq, Xq, Zq = np.meshgrid(y_centers, x_centers, z_centers, indexing="ij")
    query = np.stack([Yq.ravel(), Xq.ravel(), Zq.ravel()], axis=-1)
    fine_chi = interpolator(query).reshape(fine_ny, fine_nx, fine_nz)
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
    slice_2d = interpretable_susceptibility(result)[:, :, layer_index]
    slice_2d = _apply_range(slice_2d, threshold, threshold_max)
    return slice_2d


def _sample_along_path(mesh: InversionMesh, path_x: np.ndarray, path_y: np.ndarray):
    """Nearest mesh column (row_idx, col_idx) for each point of an
    arbitrary-direction path, in local UTM meters - the shared lookup
    behind vertical_section (2D distance-vs-depth PNG) and path_slice_3d
    (interactive 3D plane at the path's true x/y position)."""
    path_x = np.asarray(path_x, dtype=float)
    path_y = np.asarray(path_y, dtype=float)
    if len(path_x) < 2:
        raise InversionError("수직 단면을 위해서는 2개 이상의 경로 점이 필요합니다.")
    col_idx = np.argmin(np.abs(mesh.x_centers[None, :] - path_x[:, None]), axis=1)
    row_idx = np.argmin(np.abs(mesh.y_centers[None, :] - path_y[:, None]), axis=1)
    return row_idx, col_idx


def azimuth_line(mesh: InversionMesh, azimuth_deg: float, sample_spacing_m: float = 10.0):
    """A densified (path_x, path_y) straight line crossing the full mesh
    through its own center at an arbitrary compass bearing (0 = a
    north-south trending line, 90 = an east-west trending line) - lets a
    user sweep an oblique section plane with just a slider instead of
    having to hand-draw a path on the map, for internal_slice's ew/ns
    orientations' natural third option. Feed the result into
    path_slice_3d (interactive 3D plane) or vertical_section (2D
    distance-vs-depth PNG) exactly like a hand-drawn custom path."""
    az = np.radians(azimuth_deg)
    dx, dy = float(np.sin(az)), float(np.cos(az))
    cx = float(mesh.x_centers.mean())
    cy = float(mesh.y_centers.mean())
    # Half-length spans the mesh's own diagonal, so the line always
    # crosses the full mesh regardless of bearing.
    half_len = 0.5 * float(np.hypot(mesh.x_centers.max() - mesh.x_centers.min(), mesh.y_centers.max() - mesh.y_centers.min()))
    n_samples = max(2, int(2 * half_len / sample_spacing_m) + 1)
    t = np.linspace(-half_len, half_len, n_samples)
    return cx + t * dx, cy + t * dy


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
    row_idx, col_idx = _sample_along_path(mesh, path_x, path_y)

    seg_len = np.hypot(np.diff(path_x), np.diff(path_y))
    distance = np.concatenate([[0.0], np.cumsum(seg_len)])

    section = interpretable_susceptibility(result)[row_idx, col_idx, :].T  # (n_layers, n_samples)
    section = _apply_range(section, threshold, threshold_max)
    ground_elevation = mesh.ground_elevation[row_idx, col_idx]

    return section, distance, ground_elevation


def path_slice_3d(
    result: InversionResult,
    path_x: np.ndarray,
    path_y: np.ndarray,
    threshold: float | None = None,
    threshold_max: float | None = None,
) -> dict:
    """Like vertical_section, but returns the plane in real 3D (x, y, z)
    coordinates instead of collapsing it to along-path distance - so an
    arbitrary-direction section can be rendered as its own Plotly surface
    trace positioned correctly inside the same 3D scene as the isosurface
    "blob" and the orthogonal ew/ns slices from internal_slice(), the way
    mining-industry 3D modeling packages combine a thresholded volume with
    several simultaneous cross-cutting section planes in one view."""
    mesh = result.mesh
    path_x = np.asarray(path_x, dtype=float)
    path_y = np.asarray(path_y, dtype=float)
    row_idx, col_idx = _sample_along_path(mesh, path_x, path_y)

    section = interpretable_susceptibility(result)[row_idx, col_idx, :].T  # (n_layers, n_samples)
    section = _apply_range(section, threshold, threshold_max)

    n_layers, n_samples = section.shape
    x_grid = np.tile(path_x, (n_layers, 1))
    y_grid = np.tile(path_y, (n_layers, 1))
    z_grid = np.tile(mesh.z_centers[:, np.newaxis], (1, n_samples))
    return {
        "x": x_grid.tolist(),
        "y": y_grid.tolist(),
        "z": z_grid.tolist(),
        "value": section.tolist(),
    }


def _wall_along_x(result: InversionResult, row_idx: int) -> dict:
    """Vertical plane at a fixed north-south position (row_idx): distance
    runs east-west, so the coordinate grid is (x, z). Shared by box_faces
    (called at the row 0 / ny-1 mesh boundary) and internal_slice
    (called at an arbitrary interior row for an "ew"-oriented cut)."""
    mesh = result.mesh
    x, z = mesh.x_centers, mesh.z_centers
    x_grid_w, z_grid_w = np.meshgrid(x, z)  # (nz, nx)
    val = interpretable_susceptibility(result)[row_idx, :, :].T
    return {
        "x": x_grid_w.tolist(),
        "y": np.full_like(x_grid_w, float(mesh.y_centers[row_idx])).tolist(),
        "z": z_grid_w.tolist(),
        "value": val.tolist(),
    }


def _wall_along_y(result: InversionResult, col_idx: int) -> dict:
    """Vertical plane at a fixed east-west position (col_idx): distance
    runs north-south, so the coordinate grid is (y, z). Shared by
    box_faces (mesh boundary) and internal_slice (arbitrary interior
    column for an "ns"-oriented cut)."""
    mesh = result.mesh
    y, z = mesh.y_centers, mesh.z_centers
    y_grid_w, z_grid_w = np.meshgrid(y, z)  # (nz, ny)
    val = interpretable_susceptibility(result)[:, col_idx, :].T
    return {
        "x": np.full_like(y_grid_w, float(mesh.x_centers[col_idx])).tolist(),
        "y": y_grid_w.tolist(),
        "z": z_grid_w.tolist(),
        "value": val.tolist(),
    }


def internal_slice(
    result: InversionResult,
    orientation: str,
    position_frac: float,
    threshold: float | None = None,
    threshold_max: float | None = None,
) -> dict:
    """A single interior vertical cutting plane through the mesh, at an
    arbitrary position instead of only the mesh boundary (see box_faces'
    south/north/west/east walls, which are this function's endpoints):
    "ew" cuts across east-west at some north-south position_frac (0=south
    edge, 1=north edge), "ns" cuts across north-south at some east-west
    position_frac (0=west edge, 1=east edge). continuous SI-colored
    (masked NaN for inactive/air cells), for combining with the
    isosurface "blob" and/or a path_slice_3d into one multi-panel 3D
    scene showing the volume and several simultaneous section planes at
    once (the standard presentation in mining-industry 3D modeling
    packages such as GOCAD/Leapfrog)."""
    mesh = result.mesh
    ny, nx, _ = result.susceptibility.shape
    position_frac = float(np.clip(position_frac, 0.0, 1.0))
    if orientation == "ew":
        row_idx = int(round(position_frac * (ny - 1)))
        face = _wall_along_x(result, row_idx)
        face["position_m"] = float(mesh.y_centers[row_idx])
    elif orientation == "ns":
        col_idx = int(round(position_frac * (nx - 1)))
        face = _wall_along_y(result, col_idx)
        face["position_m"] = float(mesh.x_centers[col_idx])
    else:
        raise InversionError(f"알 수 없는 단면 방향입니다: {orientation}")

    value = _apply_range(np.asarray(face["value"], dtype=float), threshold, threshold_max)
    face["value"] = value.tolist()
    face["orientation"] = orientation
    face["position_frac"] = position_frac
    return face


def box_faces(result: InversionResult, top_layer_index: int = 0) -> dict:
    """Assemble a "fence diagram" style box for 3D display: the top
    surface plus susceptibility on the 4 vertical boundary walls of the
    inversion mesh (south/north/west/east), all continuous (no threshold
    gating) and sharing one SI colorscale - mirrors the standard
    published-figure style of a colored box with a distinct top surface
    and 4 colored side faces, as an alternative to the single-threshold
    isosurface "blob" view.

    The top surface follows the real terrain at every column - each
    column starts from its own shallowest ACTIVE layer (immediately
    below that column's local ground elevation) and steps
    top_layer_index layers deeper from there, so the slider that used to
    pick one fixed elevation shared by the whole mesh instead explores
    progressively deeper "sheets" that still hug the true terrain shape.
    A single shared elevation only reproduces the actual ground surface
    where the terrain happens to sit exactly at that height - everywhere
    else it was a flat plane cutting through empty air or through rock,
    not the surface a user asking to "see the 3D terrain shape" wants."""
    mesh = result.mesh
    chi = result.susceptibility
    active = interpretable_cells(mesh)
    x, y, z = mesh.x_centers, mesh.y_centers, mesh.z_centers
    ny, nx, nz = chi.shape
    top_layer_index = int(np.clip(top_layer_index, 0, nz - 1))

    x_grid, y_grid = np.meshgrid(x, y)  # (ny, nx)

    has_active = active.any(axis=2)
    # First True along the depth axis (index 0 = shallowest) - the layer
    # immediately below this column's real ground elevation.
    first_active = np.argmax(active, axis=2)
    col_layer = np.clip(first_active + top_layer_index, 0, nz - 1)
    top_chi = np.take_along_axis(chi, col_layer[:, :, None], axis=2)[:, :, 0]
    top_chi = np.where(has_active, top_chi, np.nan)
    top_z = np.where(has_active, z[col_layer], mesh.ground_elevation)

    top = {
        "x": x_grid.tolist(),
        "y": y_grid.tolist(),
        "z": top_z.tolist(),
        "value": top_chi.tolist(),
    }

    active_chi = chi[active & (chi > 0)]
    return {
        "top": top,
        "south": _wall_along_x(result, 0),
        "north": _wall_along_x(result, ny - 1),
        "west": _wall_along_y(result, 0),
        "east": _wall_along_y(result, nx - 1),
        "vmin": 0.0,
        "vmax": float(active_chi.max()) if active_chi.size else 1.0,
        "top_layer_index": top_layer_index,
        "top_elevation_m": float(np.nanmean(top_z)) if has_active.any() else float(z[top_layer_index]),
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

    # matplotlib.rc_context scopes the font override to this call (restored
    # on exit) instead of writing straight into matplotlib.rcParams, which
    # would leave every other renderer in the process permanently switched
    # to NanumGothic/no-unicode-minus after the first section PNG - a
    # standing side effect no caller of this function asked for, and (since
    # FastAPI's sync endpoints run in a real OS threadpool) a source of
    # cross-request races on that same global dict.
    with matplotlib.rc_context({"font.family": "NanumGothic", "axes.unicode_minus": False}):
        fig = Figure(figsize=(11.0, 6.5), dpi=110)
        FigureCanvasAgg(fig)
        fig.set_facecolor("white")
        ax = fig.add_axes([0.09, 0.13, 0.78, 0.75])

        cmap = matplotlib.colormaps[cmap_name]
        im = ax.imshow(
            section, extent=[0.0, dist_max, z_lo, z_hi], origin="upper", aspect="auto",
            cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        # Above-ground cells are already NaN (transparent) in `section`, so
        # the axes' own white background already reads as "air" - the fill
        # just makes that reading unambiguous even when the terrain relief
        # is a small fraction of the plotted depth range (a thin line alone
        # can look like a flat cutoff near the top of a tall, mostly-empty
        # plot).
        ax.fill_between(distance_m, ground_elevation_m, z_hi, color="white", zorder=2)
        ax.plot(distance_m, ground_elevation_m, color="black", linewidth=2.2, zorder=3, label="지표면")
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

        # Locator inset: full mesh (project inversion) extent as a
        # light-gray box, this profile's path drawn on top, so the user can
        # see where the cross-section sits within the overall surveyed
        # area.
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
