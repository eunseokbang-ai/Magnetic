"""Removing a ground structure's anomaly by modelling it and subtracting
the model, instead of cutting a hole and filling it.

Cutting and filling (processing/manual_smooth.py) has two failures that no
choice of region fixes, both seen on the 2026-09 HaeNam survey around a
984 nT structure anomaly:

* The anomaly does not stop at the region's edge. A compact source's
  field falls to a tenth of its peak only at about twice its depth, and
  its negative lobe sits a depth-and-a-bit away - here partly beyond the
  survey edge. Whatever the region leaves outside stays in the data, and
  the fill has to meet it: the removed patch came out as a flat trough
  40-50 nT below its surroundings with steep walls, which the analytic
  signal (a gradient magnitude) draws as a bright ring round a hollow
  centre. RTP, being a level rather than a gradient, hides it.

* A fill interpolated along each flight line separately knows nothing of
  the neighbouring lines. Each replaced line lands at its own level, so
  the patch is striped parallel to the lines - walls one line spacing
  apart that every derivative amplifies. And a line whose replaced stretch
  runs into the survey edge has only one side to anchor to, so it is held
  flat at whatever that side reads - which, next to a dipole, is the lobe.

Modelling avoids both. The structure's field is fitted as a patch of
point dipoles confined to the region the operator marked, together with a
smooth background for the geology around it, and only the dipoles' field
is subtracted - everywhere it reaches, tails and lobes included, outside
the marked region as much as inside. Nothing is replaced, so there is no
edge to meet and no line-by-line level to get wrong, and what the
geology under the structure was doing is left in the data rather than
overwritten by an interpolation.

Measured on the case above (a 220 m region): the dipole patch fits the
window to 6.3 nT rms against a ~1000 nT peak. Mean analytic signal
inside the region went from 5.65 to 0.07, and in the 150 m ring around it
to 0.22, against 0.29 for ordinary ground further out - no ring, stripe
or hollow. The line-by-line fill of the same region left 0.31 inside with
stripes along the lines, and a two-dimensional robust fill 0.36 with the
lobe beyond the region still in place.

The limit is the one any removal has. Sources are confined to the marked
region and kept below the survey plane, and the background absorbs what
is broad; but geology that is itself compact and inside the marked region
is indistinguishable from the structure and goes with it. That is why the
region is the operator's call, and why the fit reports how much it
removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath

# Relative depths tried, as multiples of the depth hint (or of the
# region's equivalent radius when there is none). The one that fits the
# window best is kept.
#
# Choosing instead by how well held-out lines are predicted was tried and
# is worse for this job: predicting a line 50 m from its neighbours past a
# 1000 nT source favours deep, heavily damped patches that under-fit the
# peak. On the HaeNam case it left the analytic signal inside the region
# at 0.37 against 0.29 for ordinary ground nearby, where choosing by fit
# brought it to 0.07. The model is only ever evaluated at the samples it
# is subtracted from, so what it does between lines does not reach the
# data. (Held-out prediction does not work as a check either: it is
# dominated by whichever line happens to cross the peak, and flagged every
# real case.)
_DEPTH_FACTORS = (0.3, 0.45, 0.6, 0.8, 1.0, 1.3)
# Damping, relative to the kernel's largest singular value.
_LAMBDA = 1e-2
# Most points used for the fit. The window can hold tens of thousands of
# 10 Hz samples, far more than a few hundred unknowns need.
_MAX_FIT_POINTS = 6000
# The model is subtracted wherever it still exceeds this.
_REACH_NT = 0.1
_CHUNK = 4000


@dataclass
class SourceRemoval:
    polygon_xy: list[tuple[float, float]]
    sources_xyz: np.ndarray            # (n, 3) east, north, up (up negative = below the survey plane)
    moments: np.ndarray                # (n, 3) free vector moments, so remanence is allowed
    inclination_deg: float
    declination_deg: float
    depth_m: float
    fit_rms_nt: float
    data_rms_nt: float                 # spread of the window before removal, for scale
    peak_model_nt: float
    reach_m: float
    n_points_fitted: int
    warnings: list[str] = field(default_factory=list)

    def field_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Total-field anomaly of the fitted sources at survey-plane points."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        out = np.zeros(len(x))
        cx, cy = self.centre
        near = np.hypot(x - cx, y - cy) <= self.reach_m
        idx = np.flatnonzero(near)
        f = _field_direction(self.inclination_deg, self.declination_deg)
        flat_m = self.moments.reshape(-1)
        for i0 in range(0, len(idx), _CHUNK):
            part = idx[i0:i0 + _CHUNK]
            g = _kernel(x[part], y[part], self.sources_xyz, f)
            out[part] = g @ flat_m
        return out

    @property
    def centre(self) -> tuple[float, float]:
        """The polygon's vertex mean - the same point reach_m was measured from."""
        p = np.asarray(self.polygon_xy, dtype=float)
        return float(p[:, 0].mean()), float(p[:, 1].mean())

    def to_dict(self) -> dict:
        return {
            "polygon_xy": [list(map(float, p)) for p in self.polygon_xy],
            "sources_xyz": self.sources_xyz.tolist(),
            "moments": self.moments.tolist(),
            "inclination_deg": self.inclination_deg,
            "declination_deg": self.declination_deg,
            "depth_m": self.depth_m,
            "fit_rms_nt": self.fit_rms_nt,
            "data_rms_nt": self.data_rms_nt,
            "peak_model_nt": self.peak_model_nt,
            "reach_m": self.reach_m,
            "n_points_fitted": self.n_points_fitted,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SourceRemoval":
        return cls(
            polygon_xy=[tuple(p) for p in d["polygon_xy"]],
            sources_xyz=np.asarray(d["sources_xyz"], dtype=float),
            moments=np.asarray(d["moments"], dtype=float),
            inclination_deg=float(d["inclination_deg"]),
            declination_deg=float(d["declination_deg"]),
            depth_m=float(d["depth_m"]),
            fit_rms_nt=float(d["fit_rms_nt"]),
            data_rms_nt=float(d["data_rms_nt"]),
            peak_model_nt=float(d["peak_model_nt"]),
            reach_m=float(d["reach_m"]),
            n_points_fitted=int(d["n_points_fitted"]),
            warnings=list(d.get("warnings", [])),
        )

    def summary(self) -> dict:
        return {
            "depth_m": self.depth_m,
            "n_sources": int(len(self.sources_xyz)),
            "fit_rms_nt": self.fit_rms_nt,
            "data_rms_nt": self.data_rms_nt,
            "peak_model_nt": self.peak_model_nt,
            "reach_m": self.reach_m,
            "warnings": list(self.warnings),
        }


def _field_direction(inclination_deg: float, declination_deg: float) -> np.ndarray:
    inc, dec = np.radians(inclination_deg), np.radians(declination_deg)
    return np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)])


def _kernel(x: np.ndarray, y: np.ndarray, sources: np.ndarray, f: np.ndarray) -> np.ndarray:
    """(n_obs, 3 * n_src): total-field anomaly in nT per A*m^2 of each
    moment component, observed on the plane z = 0."""
    rx = x[:, None] - sources[None, :, 0]
    ry = y[:, None] - sources[None, :, 1]
    rz = 0.0 - sources[None, :, 2]
    r2 = rx * rx + ry * ry + rz * rz
    r5 = r2 * r2 * np.sqrt(r2)
    fr = f[0] * rx + f[1] * ry + f[2] * rz
    g = np.empty(rx.shape + (3,))
    for k, rk in enumerate((rx, ry, rz)):
        g[..., k] = (3.0 * fr * rk - f[k] * r2) / r5
    return (g * 100.0).reshape(len(x), -1)       # mu0/4pi = 1e-7 T*m/A -> x 1e9 nT


def _background_columns(x: np.ndarray, y: np.ndarray, cx: float, cy: float, scale: float) -> np.ndarray:
    """A plane for the geology across the window.

    A quadratic was tried and trades off against the dipoles: its
    curvature takes part of the structure's own tail, which then stays in
    the data. On a synthetic with a 40 m-deep source and geology varying
    over 1.4 km, the field left in the 90-200 m ring round the region was
    1.96 nT with a plane against 4.18 nT with a quadratic (18.5 nT before
    removal). Widening the window did not help either - 2.95 nT at 4x the
    deepest depth, 5.34 at 6x - because a plane then fits the geology less
    well and what it misses leaks into the dipoles instead.
    """
    u, v = (x - cx) / scale, (y - cy) / scale
    return np.column_stack([np.ones_like(u), u, v])


def _source_grid(polygon: MplPath, depth: float, spacing: float) -> np.ndarray:
    xs, ys = polygon.vertices[:, 0], polygon.vertices[:, 1]
    gx = np.arange(xs.min(), xs.max() + spacing, spacing)
    gy = np.arange(ys.min(), ys.max() + spacing, spacing)
    gx = gx + (xs.min() + xs.max() - gx[0] - gx[-1]) / 2.0
    gy = gy + (ys.min() + ys.max() - gy[0] - gy[-1]) / 2.0
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inside = polygon.contains_points(pts)
    if not inside.any():
        c = pts.mean(axis=0)
        pts, inside = c[None, :], np.array([True])
    pts = pts[inside]
    return np.column_stack([pts, np.full(len(pts), -depth)])


def _solve(A: np.ndarray, b: np.ndarray, n_src_cols: int, lam: float) -> np.ndarray:
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1.0
    As = A / scale
    ref = np.linalg.norm(As[:, :n_src_cols], 2) if n_src_cols else 1.0
    reg = np.zeros((n_src_cols, As.shape[1]))
    reg[:, :n_src_cols] = lam * ref * np.eye(n_src_cols)
    sol, *_ = np.linalg.lstsq(np.vstack([As, reg]), np.r_[b, np.zeros(n_src_cols)], rcond=None)
    return sol / scale


def fit_source_removal(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    line_id: np.ndarray,
    polygon_xy: list[tuple[float, float]],
    inclination_deg: float,
    declination_deg: float,
    line_spacing_m: float | None,
    depth_hint_m: float | None = None,
) -> SourceRemoval:
    """Fit the anomaly inside `polygon_xy` as a patch of dipoles plus a
    smooth background, over a window reaching well past the polygon.

    Depth is chosen by fit - see _DEPTH_FACTORS for why not by held-out
    lines.
    """
    poly_xy = np.asarray(polygon_xy, dtype=float)
    if len(poly_xy) < 3:
        raise ValueError("polygon needs at least three vertices")
    path = MplPath(poly_xy)
    area = abs(np.sum(poly_xy[:, 0] * np.roll(poly_xy[:, 1], -1) - np.roll(poly_xy[:, 0], -1) * poly_xy[:, 1])) / 2
    eq_radius = float(np.sqrt(max(area, 1.0) / np.pi))
    cx, cy = float(poly_xy[:, 0].mean()), float(poly_xy[:, 1].mean())
    spacing_floor = (line_spacing_m or 20.0) / 2.0

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    line_id = np.asarray(line_id)
    ok = np.isfinite(values) & (line_id >= 0)

    hint = depth_hint_m if depth_hint_m and depth_hint_m > 0 else eq_radius
    depths = sorted({max(5.0, round(hint * k, 1)) for k in _DEPTH_FACTORS})
    # The window has to reach past where the deepest candidate's field has
    # faded, or the background will absorb the tails instead of the
    # dipoles - but not much further (see _background_columns).
    window_r = eq_radius + 2.5 * max(depths) + (line_spacing_m or 50.0)
    in_win = ok & (np.hypot(x - cx, y - cy) <= window_r)
    if in_win.sum() < 50:
        raise ValueError("too few survey points around the region to fit it")

    wx, wy, wv, wl = x[in_win], y[in_win], values[in_win], line_id[in_win]
    stride = max(1, int(np.ceil(len(wx) / _MAX_FIT_POINTS)))
    wx, wy, wv, wl = wx[::stride], wy[::stride], wv[::stride], wl[::stride]

    f = _field_direction(inclination_deg, declination_deg)
    bg = _background_columns(wx, wy, cx, cy, window_r)
    best = None
    for depth in depths:
        spacing = max(0.6 * depth, spacing_floor)
        sources = _source_grid(path, depth, spacing)
        g = _kernel(wx, wy, sources, f)
        A = np.hstack([g, bg])
        sol = _solve(A, wv, g.shape[1], _LAMBDA)
        fit_rms = float(np.sqrt(np.mean((wv - A @ sol) ** 2)))
        if best is None or fit_rms < best[0]:
            best = (fit_rms, depth, sources, g, A, sol)

    fit_rms, depth, sources, g, A, sol = best
    moments = sol[: g.shape[1]].reshape(-1, 3)
    model_win = g @ sol[: g.shape[1]]

    removal = SourceRemoval(
        polygon_xy=[tuple(p) for p in poly_xy],
        sources_xyz=sources,
        moments=moments,
        inclination_deg=float(inclination_deg),
        declination_deg=float(declination_deg),
        depth_m=float(depth),
        fit_rms_nt=fit_rms,
        data_rms_nt=float(np.std(wv)),
        peak_model_nt=float(np.max(np.abs(model_win))) if len(model_win) else 0.0,
        reach_m=window_r,
        n_points_fitted=int(len(wv)),
    )
    removal.reach_m = _reach(removal, cx, cy, window_r)

    if removal.depth_m in (min(depths), max(depths)) and len(depths) > 1:
        removal.warnings.append(
            "맞춘 깊이가 시험 범위의 끝값입니다 - 영역이 이상체를 제대로 감싸지 못했을 수 있습니다."
        )
    if removal.peak_model_nt < 0.5 * removal.data_rms_nt:
        removal.warnings.append(
            "영역 안에서 뺄 만한 국지 이상을 찾지 못했습니다 - 영역이 구조물 이상을 벗어났을 수 있습니다."
        )
    if removal.fit_rms_nt > 0.3 * removal.data_rms_nt:
        removal.warnings.append(
            "모델이 이 이상을 잘 설명하지 못합니다 (맞춤 오차가 자료 변동의 30% 이상) - "
            "영역을 이상체에 맞게 다시 그리거나 보간 방식을 쓰세요."
        )
    return removal


def _reach(removal: SourceRemoval, cx: float, cy: float, start: float) -> float:
    """How far from the centre the model still exceeds _REACH_NT, found by
    walking outward along a few directions."""
    f = _field_direction(removal.inclination_deg, removal.declination_deg)
    flat_m = removal.moments.reshape(-1)
    r = start
    for _ in range(12):
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        px, py = cx + r * np.cos(ang), cy + r * np.sin(ang)
        if np.max(np.abs(_kernel(px, py, removal.sources_xyz, f) @ flat_m)) < _REACH_NT:
            return float(r)
        r *= 1.5
    return float(r)


def apply_source_removals(
    x: np.ndarray, y: np.ndarray, values: np.ndarray, removals: list[SourceRemoval]
) -> np.ndarray:
    """values with every removal's modelled field subtracted."""
    out = np.asarray(values, dtype=float).copy()
    for removal in removals:
        out = out - removal.field_at(x, y)
    return out
