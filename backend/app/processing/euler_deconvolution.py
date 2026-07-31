"""Euler deconvolution: automatic estimate of causative-source location and
depth from a gridded magnetic anomaly, using Euler's homogeneity equation
(Thompson 1982; Reid et al. 1990) - the standard "quick look" source-depth
technique in airborne/UAV magnetics, complementary to but far cheaper than
the full 3D inversion.

    (x - x0) dT/dx + (y - y0) dT/dy + (z - z0) dT/dz = N (B - T)

is solved for the unknown source location (x0, y0, z0) and background
level B in a least-squares sense over each of many small sliding windows,
for a structural index N assumed by the user (0 = contact/fault edge,
1 = thin dyke/sill/sheet edge, 2 = pipe/vertical cylinder, 3 = sphere/
point dipole - the four textbook source-geometry classes). x, y are local
UTM meters and z is defined positive DOWNWARD from the (assumed flat)
observation surface at z=0, matching the sign convention already used by
processing/transforms.py:vertical_derivative (a wavenumber-domain |k|
filter) - so a solved z0 is directly usable as "depth below the flight
surface" with no sign flip.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

from .transforms import vertical_derivative


@dataclass
class EulerSolution:
    lat: float
    lon: float
    depth_m: float
    base_level_nt: float
    uncertainty_m: float


def _fill_nan(grid: np.ndarray) -> np.ndarray:
    finite = grid[np.isfinite(grid)]
    fill = float(np.mean(finite)) if finite.size else 0.0
    return np.where(np.isfinite(grid), grid, fill)


def run_euler_deconvolution(
    grid_values: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    cell_size_m: float,
    utm_epsg: int,
    structural_index: float = 1.0,
    window_size_m: float = 100.0,
    max_depth_uncertainty_pct: float = 30.0,
    max_depth_m: float | None = None,
) -> list[EulerSolution]:
    finite_mask = np.isfinite(grid_values)
    if finite_mask.sum() < 20:
        return []

    filled = _fill_nan(grid_values)
    dTdy = np.gradient(filled, cell_size_m, axis=0)
    dTdx = np.gradient(filled, cell_size_m, axis=1)
    dTdz = vertical_derivative(grid_values, cell_size_m, order=1)  # NaN-aware, matches z-positive-down convention

    dTdx = np.where(finite_mask, dTdx, np.nan)
    dTdy = np.where(finite_mask, dTdy, np.nan)
    # dTdz already NaN outside coverage (transforms.py re-masks after the FFT round-trip)

    w = max(5, int(round(window_size_m / cell_size_m)))
    stride = max(1, w // 2)
    n_north, n_east = grid_values.shape
    if max_depth_m is None:
        max_depth_m = 20.0 * window_size_m  # generous sanity ceiling, not a tuned geophysical limit

    x2d, y2d = np.meshgrid(easting, northing)

    solutions_xyz: list[tuple[float, float, float, float, float]] = []
    for r0 in range(0, max(1, n_north - w + 1), stride):
        for c0 in range(0, max(1, n_east - w + 1), stride):
            sl = (slice(r0, r0 + w), slice(c0, c0 + w))
            t = grid_values[sl]
            gx, gy, gz = dTdx[sl], dTdy[sl], dTdz[sl]
            x, y = x2d[sl], y2d[sl]

            valid = np.isfinite(t) & np.isfinite(gx) & np.isfinite(gy) & np.isfinite(gz)
            n_valid = int(valid.sum())
            if n_valid < max(10, w):
                continue

            t_v, gx_v, gy_v, gz_v = t[valid], gx[valid], gy[valid], gz[valid]
            x_v, y_v = x[valid], y[valid]

            A = np.column_stack([gx_v, gy_v, gz_v, np.full(n_valid, structural_index)])
            b = x_v * gx_v + y_v * gy_v + structural_index * t_v

            try:
                sol, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if rank < 4:
                continue

            x0, y0, z0, base_level = sol
            resid_vec = A @ sol - b
            dof = n_valid - 4
            if dof <= 0:
                continue
            rms = float(np.sqrt(np.sum(resid_vec**2) / dof))
            try:
                cov = np.linalg.inv(A.T @ A) * rms**2
                sigma_z0 = float(np.sqrt(max(cov[2, 2], 0.0)))
            except np.linalg.LinAlgError:
                continue

            if z0 <= 0 or z0 > max_depth_m:
                continue
            if sigma_z0 > (max_depth_uncertainty_pct / 100.0) * z0:
                continue
            # solved (x0, y0) should fall roughly within the window, not run off to infinity
            if not (x_v.min() - w * cell_size_m <= x0 <= x_v.max() + w * cell_size_m):
                continue
            if not (y_v.min() - w * cell_size_m <= y0 <= y_v.max() + w * cell_size_m):
                continue

            solutions_xyz.append((float(x0), float(y0), float(z0), float(base_level), sigma_z0))

    if not solutions_xyz:
        return []

    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    xs = [s[0] for s in solutions_xyz]
    ys = [s[1] for s in solutions_xyz]
    lons, lats = transformer.transform(xs, ys)

    return [
        EulerSolution(lat=float(lat), lon=float(lon), depth_m=z0, base_level_nt=base_level, uncertainty_m=sigma_z0)
        for lat, lon, (_, _, z0, base_level, sigma_z0) in zip(lats, lons, solutions_xyz)
    ]
