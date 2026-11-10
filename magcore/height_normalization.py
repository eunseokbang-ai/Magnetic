"""Flight-height normalisation: the readings, taken along a terrain-
following flight at altitudes that differ by tens of metres, restated at
one reference altitude.

Everything downstream - the grids and their FFT derivatives, Euler
deconvolution, depth estimates, the 3D inversion - takes the readings to
lie on one horizontal plane. A drone at 50 m AGL over hills does not: a
shallow source read from 10 m higher is already a third weaker (a dipole
falls with the cube of the distance), and where the altitude changes with
the line, the change shows as line striping.

The remedy is Dampney's (1969) equivalent source in the scattered-point
form of Cordell (1992): a layer of dipoles below the survey is fitted to
the readings at their true 3D positions, and the same layer is then
evaluated where the reading would have been at the reference altitude.

The model
---------
Induced dipoles (moment along the IGRF direction, one scalar unknown each)
on regular horizontal grids. Remanence is ignored - the layer only has to
reproduce the observed field, and a fixed direction keeps the solution
stable. A constant is fitted alongside; it needs no height correction.

Three layers, fitted in sequence, each to what the layers before it left
(the "boosting" of Soler & Uieda, 2021): a deep coarse one for the
regional, a middle one, and a shallow fine one for the sources the
flight height actually matters for. One shallow layer alone reproduces
the readings to a fraction of a nanotesla yet gets the height correction
mostly wrong, because it explains the regional with sources near its own
edges whose vertical gradient is not the regional's - see the table.

What replaces the reading is not the layer's prediction but the reading
plus the layers' *difference* between the two altitudes:

    corrected = observed + [ layer(x, y, z_ref) - layer(x, y, z_obs) ]

What the layers do not fit - along-line detail shorter than the line
spacing, noise - stays in the data instead of being thrown away, and a
flat flight is left alone whatever the fit quality. Replacing the reading
by the prediction was measured too and recovered the field at z_ref
slightly worse while losing that detail.

Solving
-------
The readings are block-averaged to `cell_size_m` cells first (the Haenam
block's 965k readings become about 70k cells) and each layer's damped
least-squares problem is solved by CGLS - conjugate gradients on the
normal equations without forming them - with the kernel evaluated on the
fly in numba, because the dense matrix would be several GB. Columns are
scaled to unit norm and the damping is a fraction of the largest singular
value (power iteration), the same relative form as source_removal.py.
At Haenam scale one matrix product takes half a second on a 32-core
machine; the whole normalisation is a minute or two.

Measured on the synthetic of tests/synthetic_hill_survey.py - two shallow
induced dipoles 40 m and 60 m under the ground, a regional geology of
1.4 km wavelength, 50 m lines (L) at 50 m AGL over +-30 m of relief,
0.3 nT noise. "before/after" are the rms difference from the true field
at z_ref (the median altitude) over the survey; 1 km x 1 km survey unless
said otherwise:

    layers (depth, spacing, margin) in L            before   after   ratio
    shallow only (1.25, 0.5, 2) + plane              7.07    4.81    0.68
    shallow only, layer 2.0 deep                     7.07    2.64    0.37
    deep only (5, 2, 5)                              7.07    0.85    0.12
    deep (5,2,5) -> shallow (1.25,0.5,2)             7.07    0.64    0.09
    deep (8,3,8) -> mid (3,1,3) -> shallow           7.07    0.43    0.06   <- default
    the same on a 2 km x 2 km survey                 6.74    0.37    0.05

    noise (nT)      0.3    1.0    2.0    5.0        ratio  0.04  0.05  0.07  0.14
    shallow sources only (no regional)              2.90    0.77    0.26  (peak error 53 -> 13 nT)
    flat flight, +-1 m                              median |correction| 0.11 nT, max 0.68

30 CGLS iterations per layer reach the noise floor (fit 0.30 nT at
0.3 nT noise); 60 improve the result by less than 1%. Damping between
0.1% and 1% of the largest singular value gives the same answer; 10%
under-fits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import progress
from numba import njit, prange
from scipy.spatial import cKDTree

DEFAULT_CELL_SIZE_M = 10.0
# (depth below the lowest reading, grid spacing, margin past the readings),
# all in line spacings, fitted in this order - each layer to the residual
# of the ones before it.
DEFAULT_LAYERS = ((8.0, 3.0, 8.0), (3.0, 1.0, 3.0), (1.25, 0.5, 2.0))
# Damping as a fraction of the largest singular value of the scaled system.
DEFAULT_DAMPING = 0.01
DEFAULT_MAX_ITER = 30
# A fit worse than this fraction of the data's own variation means the
# layers are not describing the field, and the correction they imply is
# not to be trusted - reported as a warning.
MISFIT_WARNING_RATIO = 0.3
_MIN_POINTS = 50
# Readings more than this many line spacings from z_ref are flagged when
# they are more than _FAR_WARNING_PCT of the total.
_FAR_LINE_SPACINGS = 2.0
_FAR_WARNING_PCT = 0.5
_POWER_ITERATIONS = 5
_CG_TOL = 1e-4


class HeightNormalizationError(ValueError):
    pass


# ------------------------------------------------------------- kernels
# Total-field anomaly, in nT, of an induced dipole of unit moment (1 A m^2
# along the field direction f) at (sx, sy, sz), read at (ox, oy, oz):
#   100 * (3 (f.r)^2 / r^2 - 1) / r^3        (mu0/4pi = 1e-7, x 1e9 for nT)
# The same expression as source_removal._kernel projected onto f, with the
# observation height carried per point instead of fixed at z = 0.

@njit(parallel=True, fastmath=True)
def _forward(ox, oy, oz, sx, sy, sz, m, fx, fy, fz):
    n_obs = ox.shape[0]
    n_src = sx.shape[0]
    out = np.empty(n_obs)
    for i in prange(n_obs):
        acc = 0.0
        xi = ox[i]
        yi = oy[i]
        zi = oz[i]
        for j in range(n_src):
            rx = xi - sx[j]
            ry = yi - sy[j]
            rz = zi - sz[j]
            r2 = rx * rx + ry * ry + rz * rz
            fr = fx * rx + fy * ry + fz * rz
            inv_r = 1.0 / np.sqrt(r2)
            acc += m[j] * (3.0 * fr * fr / r2 - 1.0) * inv_r * inv_r * inv_r
        out[i] = 100.0 * acc
    return out


@njit(parallel=True, fastmath=True)
def _adjoint(ox, oy, oz, sx, sy, sz, u, fx, fy, fz):
    n_obs = ox.shape[0]
    n_src = sx.shape[0]
    out = np.empty(n_src)
    for j in prange(n_src):
        acc = 0.0
        xj = sx[j]
        yj = sy[j]
        zj = sz[j]
        for i in range(n_obs):
            rx = ox[i] - xj
            ry = oy[i] - yj
            rz = oz[i] - zj
            r2 = rx * rx + ry * ry + rz * rz
            fr = fx * rx + fy * ry + fz * rz
            inv_r = 1.0 / np.sqrt(r2)
            acc += u[i] * (3.0 * fr * fr / r2 - 1.0) * inv_r * inv_r * inv_r
        out[j] = 100.0 * acc
    return out


@njit(parallel=True, fastmath=True)
def _column_norms(ox, oy, oz, sx, sy, sz, fx, fy, fz):
    n_obs = ox.shape[0]
    n_src = sx.shape[0]
    out = np.empty(n_src)
    for j in prange(n_src):
        acc = 0.0
        xj = sx[j]
        yj = sy[j]
        zj = sz[j]
        for i in range(n_obs):
            rx = ox[i] - xj
            ry = oy[i] - yj
            rz = oz[i] - zj
            r2 = rx * rx + ry * ry + rz * rz
            fr = fx * rx + fy * ry + fz * rz
            inv_r = 1.0 / np.sqrt(r2)
            k = (3.0 * fr * fr / r2 - 1.0) * inv_r * inv_r * inv_r
            acc += k * k
        out[j] = 100.0 * np.sqrt(acc)
    return out


def field_direction(inclination_deg: float, declination_deg: float) -> np.ndarray:
    inc, dec = np.radians(inclination_deg), np.radians(declination_deg)
    return np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)])


def _c(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def dipole_layer_field(x, y, z, sources_xyz, moments, inclination_deg, declination_deg) -> np.ndarray:
    """Total-field anomaly (nT) of induced dipoles `sources_xyz` with
    moments `moments` (A m^2), read at the 3D points (x, y, z). Public so
    the tests can build their truth with the same kernel."""
    f = field_direction(inclination_deg, declination_deg)
    s = np.asarray(sources_xyz, dtype=np.float64)
    if len(s) == 0:
        return np.zeros(len(x))
    return _forward(
        _c(x), _c(y), _c(z), _c(s[:, 0]), _c(s[:, 1]), _c(s[:, 2]),
        _c(moments), float(f[0]), float(f[1]), float(f[2]),
    )


# --------------------------------------------------------------- result
@dataclass
class HeightNormalization:
    sources_xyz: np.ndarray            # (n_src, 3); z is each source's altitude
    moments: np.ndarray                # (n_src,), A m^2 along the field
    inclination_deg: float
    declination_deg: float
    z_ref_m: float
    line_spacing_m: float
    cell_size_m: float
    layers: list                       # per layer: depth, spacing, n_sources, fit_rms after it, iterations
    n_points_fitted: int
    n_blocks: int
    fit_rms_nt: float                  # rms of what all the layers together leave, over the blocks
    data_std_nt: float                 # std of the block values about their mean
    altitude: dict                     # of the fitted points: min/median/max/std
    warnings: list = field(default_factory=list)
    # Filled by whoever applies the model to a set of points.
    correction_stats: dict | None = None

    # ---- prediction
    def predict(self, x, y, z) -> np.ndarray:
        """The layers' field (no constant) at the 3D points."""
        return dipole_layer_field(x, y, z, self.sources_xyz, self.moments, self.inclination_deg, self.declination_deg)

    def correction(self, x, y, z) -> np.ndarray:
        """What to add to a reading taken at (x, y, z) to state it at
        z_ref: layer(z_ref) - layer(z). Zero where z is unknown, because
        a correction cannot be measured there."""
        x, y, z = _c(x), _c(y), _c(z)
        out = np.zeros(len(x))
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not ok.any():
            return out
        xo, yo, zo = x[ok], y[ok], z[ok]
        at_ref = self.predict(xo, yo, np.full(len(xo), self.z_ref_m))
        at_obs = self.predict(xo, yo, zo)
        out[ok] = at_ref - at_obs
        return out

    @property
    def misfit_ratio(self) -> float:
        return float(self.fit_rms_nt / self.data_std_nt) if self.data_std_nt > 0 else float("nan")

    @property
    def n_iter(self) -> int:
        return int(sum(layer["n_iter"] for layer in self.layers))

    # ---- persistence
    def to_dict(self) -> dict:
        return {
            "sources_xyz": self.sources_xyz.tolist(),
            "moments": self.moments.tolist(),
            "inclination_deg": self.inclination_deg,
            "declination_deg": self.declination_deg,
            "z_ref_m": self.z_ref_m,
            "line_spacing_m": self.line_spacing_m,
            "cell_size_m": self.cell_size_m,
            "layers": self.layers,
            "n_points_fitted": self.n_points_fitted,
            "n_blocks": self.n_blocks,
            "fit_rms_nt": self.fit_rms_nt,
            "data_std_nt": self.data_std_nt,
            "altitude": self.altitude,
            "warnings": list(self.warnings),
            "correction_stats": self.correction_stats,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HeightNormalization":
        return cls(
            sources_xyz=np.asarray(d["sources_xyz"], dtype=float).reshape(-1, 3),
            moments=np.asarray(d["moments"], dtype=float),
            inclination_deg=float(d["inclination_deg"]),
            declination_deg=float(d["declination_deg"]),
            z_ref_m=float(d["z_ref_m"]),
            line_spacing_m=float(d["line_spacing_m"]),
            cell_size_m=float(d["cell_size_m"]),
            layers=list(d.get("layers") or []),
            n_points_fitted=int(d["n_points_fitted"]),
            n_blocks=int(d["n_blocks"]),
            fit_rms_nt=float(d["fit_rms_nt"]),
            data_std_nt=float(d["data_std_nt"]),
            altitude=dict(d.get("altitude") or {}),
            warnings=list(d.get("warnings") or []),
            correction_stats=d.get("correction_stats"),
        )

    def summary(self) -> dict:
        return {
            "z_ref_m": round(self.z_ref_m, 2),
            "altitude": self.altitude,
            "n_sources": int(len(self.sources_xyz)),
            "layers": self.layers,
            "cell_size_m": self.cell_size_m,
            "n_points_fitted": self.n_points_fitted,
            "n_blocks": self.n_blocks,
            "fit_rms_nt": round(self.fit_rms_nt, 3),
            "data_std_nt": round(self.data_std_nt, 3),
            "misfit_ratio": round(self.misfit_ratio, 3) if np.isfinite(self.misfit_ratio) else None,
            "n_iter": self.n_iter,
            "correction_stats": self.correction_stats,
            "warnings": list(self.warnings),
        }


# ------------------------------------------------------------------ fit
def block_average(x, y, z, values, cell_size_m: float):
    """Mean position, altitude and value per `cell_size_m` square cell."""
    ix = np.floor((x - x.min()) / cell_size_m).astype(np.int64)
    iy = np.floor((y - y.min()) / cell_size_m).astype(np.int64)
    key = ix * (iy.max() + 1) + iy
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    n = len(counts)

    def mean(a):
        return np.bincount(inverse, weights=a, minlength=n) / counts

    return mean(x), mean(y), mean(z), mean(values)


def _source_grid(bx, by, spacing: float, margin: float, z: float) -> np.ndarray:
    x0, x1 = bx.min() - margin, bx.max() + margin
    y0, y1 = by.min() - margin, by.max() + margin
    gx = np.arange(x0, x1 + spacing, spacing)
    gy = np.arange(y0, y1 + spacing, spacing)
    gx += (x0 + x1 - gx[0] - gx[-1]) / 2.0
    gy += (y0 + y1 - gy[0] - gy[-1]) / 2.0
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    # Only where there is data within reach: an L-shaped or ragged survey
    # would otherwise get a bounding box of unconstrained sources.
    dist, _ = cKDTree(np.column_stack([bx, by])).query(pts, k=1)
    pts = pts[dist <= margin]
    if len(pts) == 0:
        pts = np.array([[float(bx.mean()), float(by.mean())]])
    return np.column_stack([pts, np.full(len(pts), z)])


def altitude_stats(z) -> dict:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return {"n": 0, "min_m": None, "median_m": None, "max_m": None, "std_m": None, "range_m": None}
    return {
        "n": int(len(z)),
        "min_m": round(float(z.min()), 2),
        "median_m": round(float(np.median(z)), 2),
        "max_m": round(float(z.max()), 2),
        "std_m": round(float(z.std()), 2),
        "range_m": round(float(z.max() - z.min()), 2),
    }


def correction_stats(corr) -> dict:
    c = np.asarray(corr, dtype=float)
    c = c[np.isfinite(c)]
    if len(c) == 0:
        return {"n": 0, "median_abs_nt": None, "p95_abs_nt": None, "max_abs_nt": None, "rms_nt": None}
    a = np.abs(c)
    return {
        "n": int(len(c)),
        "median_abs_nt": round(float(np.median(a)), 3),
        "p95_abs_nt": round(float(np.percentile(a, 95)), 3),
        "max_abs_nt": round(float(a.max()), 3),
        "rms_nt": round(float(np.sqrt(np.mean(c * c))), 3),
    }


def fit_height_normalization(
    x,
    y,
    z,
    values,
    inclination_deg: float,
    declination_deg: float,
    line_spacing_m: float | None,
    *,
    z_ref_m: float | None = None,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    layers=DEFAULT_LAYERS,
    damping: float = DEFAULT_DAMPING,
    max_iter: int = DEFAULT_MAX_ITER,
) -> HeightNormalization:
    """Fit the equivalent layers to readings `values` at the 3D points
    (x, y, z) - z the ellipsoidal altitude of the sensor - and return
    them with the reference altitude (default: the median altitude)
    attached. `layers` are (depth below the lowest reading, spacing,
    margin) triples in line spacings, deepest first.
    """
    x, y, z, values = _c(x), _c(y), _c(z), _c(values)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(values)
    if int(ok.sum()) < _MIN_POINTS:
        raise HeightNormalizationError("고도와 자력값이 모두 있는 측점이 너무 적습니다.")
    x, y, z, values = x[ok], y[ok], z[ok], values[ok]
    n_points = int(len(x))
    alt = altitude_stats(z)
    if z_ref_m is None:
        z_ref_m = float(np.median(z))

    spacing_L = float(line_spacing_m) if line_spacing_m and line_spacing_m > 0 else 50.0

    bx, by, bz, bv = block_average(x, y, z, values, float(cell_size_m))
    bx, by, bz = _c(bx), _c(by), _c(bz)
    n_blocks = int(len(bx))
    lowest = float(bz.min())
    f = field_direction(inclination_deg, declination_deg)

    residual = bv - float(bv.mean())
    data_std = float(residual.std())
    all_sources, all_moments, layer_reports = [], [], []
    for depth_factor, spacing_factor, margin_factor in layers:
        sources = _source_grid(bx, by, spacing_factor * spacing_L, margin_factor * spacing_L,
                               lowest - depth_factor * spacing_L)
        moments, n_iter, converged, residual = _fit_layer(bx, by, bz, residual, sources, f, damping, max_iter)
        all_sources.append(sources)
        all_moments.append(moments)
        layer_reports.append({
            "depth_below_lowest_m": round(depth_factor * spacing_L, 1),
            "spacing_m": round(spacing_factor * spacing_L, 1),
            "n_sources": int(len(sources)),
            "fit_rms_nt": round(float(np.sqrt(np.mean(residual * residual))), 3),
            "n_iter": int(n_iter),
            "converged": bool(converged),
        })
    fit_rms = float(np.sqrt(np.mean(residual * residual)))

    warnings: list[str] = []
    ratio = fit_rms / data_std if data_std > 0 else float("nan")
    if np.isfinite(ratio) and ratio > MISFIT_WARNING_RATIO:
        warnings.append(
            f"등가층의 맞춤 오차({fit_rms:.2f} nT rms)가 자료 변동({data_std:.2f} nT)의 "
            f"{100 * ratio:.0f}%입니다. 층이 자료를 충분히 설명하지 못하므로 보정량을 신뢰하기 어렵습니다."
        )
    if alt["range_m"] is not None and alt["range_m"] < 2.0:
        warnings.append("측점 고도 변화가 2 m 미만이어서 정규화 보정량이 잡음 수준입니다.")
    # Readings far from z_ref get the layers' answer at a height the
    # layers learnt from other readings - a continuation, not a
    # measurement. On the Haenam block 2.5% of the readings sit more than
    # 50 m above the median altitude (a hill flown high).
    far = np.abs(z - z_ref_m) > _FAR_LINE_SPACINGS * spacing_L
    far_pct = 100.0 * float(far.mean())
    if far_pct > _FAR_WARNING_PCT:
        warnings.append(
            f"측점의 {far_pct:.1f}%가 기준 고도에서 측선 간격의 {_FAR_LINE_SPACINGS:.0f}배"
            f"({_FAR_LINE_SPACINGS * spacing_L:.0f} m) 이상 떨어져 있습니다. 그 측점의 보정량은 주변 측점에서 "
            f"배운 층을 다른 고도로 옮긴 값이라 가까운 고도의 측점보다 불확실합니다."
        )

    return HeightNormalization(
        sources_xyz=np.vstack(all_sources),
        moments=np.concatenate(all_moments),
        inclination_deg=float(inclination_deg),
        declination_deg=float(declination_deg),
        z_ref_m=float(z_ref_m),
        line_spacing_m=spacing_L,
        cell_size_m=float(cell_size_m),
        layers=layer_reports,
        n_points_fitted=n_points,
        n_blocks=n_blocks,
        fit_rms_nt=fit_rms,
        data_std_nt=data_std,
        altitude=alt,
        warnings=warnings,
    )


def _fit_layer(bx, by, bz, d, sources, f, damping, max_iter):
    """One layer, damped least squares against `d` (already zero-mean),
    with a constant fitted alongside. Returns the physical moments and
    what the layer leaves behind (zero-mean again)."""
    fx, fy, fz = (float(v) for v in f)
    sx, sy, sz = (_c(sources[:, k]) for k in range(3))
    n_src = len(sources)
    n_blocks = len(bx)

    col_norm = _column_norms(bx, by, bz, sx, sy, sz, fx, fy, fz)
    col_norm[col_norm == 0] = 1.0
    const_norm = float(np.sqrt(n_blocks))

    def a_mv(v):
        return _forward(bx, by, bz, sx, sy, sz, _c(v[:n_src] / col_norm), fx, fy, fz) + v[n_src] / const_norm

    def a_rmv(u):
        u = _c(u)
        return np.r_[_adjoint(bx, by, bz, sx, sy, sz, u, fx, fy, fz) / col_norm, u.sum() / const_norm]

    # Largest singular value of the scaled system, for the damping.
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n_src + 1)
    v /= np.linalg.norm(v)
    sigma = 1.0
    for _ in range(_POWER_ITERATIONS):
        w = a_rmv(a_mv(v))
        norm_w = float(np.linalg.norm(w))
        if norm_w == 0.0:
            break
        sigma = float(np.sqrt(norm_w))
        v = w / norm_w

    m, n_iter, converged = _cgls(a_mv, a_rmv, d, n_src + 1, damping * sigma, max_iter)
    residual = d - a_mv(m)
    residual -= residual.mean()
    return m[:n_src] / col_norm, n_iter, converged, residual


def _cgls(a_mv, a_rmv, d, n_unknowns: int, lam: float, max_iter: int):
    """min |A m - d|^2 + lam^2 |m|^2 by CGLS: conjugate gradients on the
    normal equations, with only products by A and A^T. Stops when the
    gradient has shrunk by _CG_TOL or the data misfit stops moving."""
    m = np.zeros(n_unknowns)
    r = d.copy()
    s = a_rmv(r)
    p = s.copy()
    gamma = float(s @ s)
    gamma0 = gamma
    prev_misfit = float(np.sqrt(np.mean(r * r)))
    n_iter = 0
    converged = False
    lam2 = lam * lam
    if gamma0 == 0.0:
        return m, 0, True
    for it in range(1, max_iter + 1):
        progress.check()                     # stop here if the user cancelled
        q = a_mv(p)
        denom = float(q @ q) + lam2 * float(p @ p)
        if denom <= 0.0:
            break
        alpha = gamma / denom
        m += alpha * p
        r -= alpha * q
        n_iter = it
        s = a_rmv(r) - lam2 * m
        gamma_new = float(s @ s)
        misfit = float(np.sqrt(np.mean(r * r)))
        if np.sqrt(gamma_new) <= _CG_TOL * np.sqrt(gamma0) or abs(prev_misfit - misfit) <= 1e-3 * prev_misfit:
            converged = True
            break
        prev_misfit = misfit
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new
    return m, n_iter, converged
