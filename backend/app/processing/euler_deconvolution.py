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
UTM meters and z is positive DOWNWARD.

The equation holds at any observation elevation z, not just z=0 - by
default (altitude_grid=None) every observation is assumed to sit on one
flat plane at z=0, matching the sign convention already used by
processing/transforms.py:vertical_derivative (a wavenumber-domain |k|
filter), so the solved z0 is directly usable as "depth below the flight
surface". For a draped/terrain-following survey this flat-plane
assumption is only approximate: the drone's actual recorded altitude
varies with the terrain, and the standard "non-level Euler deconvolution"
refinement (Reid et al. 1990; FitzGerald et al. 2004) is to use each
window's real observation elevation instead of 0 - pass the drone's own
gridded altitude as altitude_grid to do that (the vertical-derivative
term dTdz itself is still computed assuming a single flat plane, since
correcting that too would need the full sensitivity-matrix machinery the
3D inversion already provides - see processing/inversion.py; this is the
standard first-order draped-survey correction, not a complete one).
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
    # What depth_m is measured from - "flat_datum" (legacy behaviour, a
    # single assumed-flat z=0 observation plane), "flight_altitude" (below
    # this window's actual local flight elevation, altitude_grid given but
    # no AGL to reference it to ground), or "ground_surface" (below the
    # estimated ground elevation = flight altitude - flight_agl_m).
    depth_reference: str = "flat_datum"


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
    altitude_grid: np.ndarray | None = None,
    flight_agl_m: float | None = None,
) -> list[EulerSolution]:
    """altitude_grid (same shape as grid_values, meters, positive up - the
    drone's own recorded elevation gridded the same way as grid_values)
    switches each window from the flat-z=0 assumption to that window's
    real observation elevations, referenced to the window's own mean
    altitude (so the numbers stay small/well-conditioned regardless of
    the survey's absolute elevation). flight_agl_m (the flight's constant
    height above ground, for a terrain-following/draped survey) then just
    shifts the reported depth from "below the local flight altitude" to
    "below the local ground surface" - see module docstring."""
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

    if altitude_grid is None:
        depth_reference = "flat_datum"
    elif flight_agl_m is not None:
        depth_reference = "ground_surface"
    else:
        depth_reference = "flight_altitude"
    agl_offset = flight_agl_m if flight_agl_m is not None else 0.0

    solutions_xyz: list[tuple[float, float, float, float, float]] = []
    for r0 in range(0, max(1, n_north - w + 1), stride):
        for c0 in range(0, max(1, n_east - w + 1), stride):
            sl = (slice(r0, r0 + w), slice(c0, c0 + w))
            t = grid_values[sl]
            gx, gy, gz = dTdx[sl], dTdy[sl], dTdz[sl]
            x, y = x2d[sl], y2d[sl]
            alt = altitude_grid[sl] if altitude_grid is not None else None

            valid = np.isfinite(t) & np.isfinite(gx) & np.isfinite(gy) & np.isfinite(gz)
            if alt is not None:
                valid &= np.isfinite(alt)
            n_valid = int(valid.sum())
            if n_valid < max(10, w):
                continue

            t_v, gx_v, gy_v, gz_v = t[valid], gx[valid], gy[valid], gz[valid]
            x_v, y_v = x[valid], y[valid]
            if alt is not None:
                alt_v = alt[valid]
                # Positive-down, referenced to this window's own mean
                # flight altitude - keeps z small/well-conditioned
                # regardless of the survey's absolute elevation, and
                # means the solved z0 comes out directly as "depth below
                # this window's average flight height".
                z_v = float(np.mean(alt_v)) - alt_v
            else:
                z_v = np.zeros(n_valid)

            A = np.column_stack([gx_v, gy_v, gz_v, np.full(n_valid, structural_index)])
            b = x_v * gx_v + y_v * gy_v + z_v * gz_v + structural_index * t_v

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

            # z0 here is depth below this window's mean flight altitude;
            # the ground sits flight_agl_m BELOW that (ground = flight
            # altitude - AGL), so depth below ground is that much LESS
            # than depth below the flight altitude - subtract, not add.
            depth_m = z0 - agl_offset
            if depth_m <= 0 or depth_m > max_depth_m:
                continue
            if sigma_z0 > (max_depth_uncertainty_pct / 100.0) * depth_m:
                continue
            # solved (x0, y0) should fall roughly within the window, not run off to infinity
            if not (x_v.min() - w * cell_size_m <= x0 <= x_v.max() + w * cell_size_m):
                continue
            if not (y_v.min() - w * cell_size_m <= y0 <= y_v.max() + w * cell_size_m):
                continue

            solutions_xyz.append((float(x0), float(y0), float(depth_m), float(base_level), sigma_z0))

    if not solutions_xyz:
        return []

    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    xs = [s[0] for s in solutions_xyz]
    ys = [s[1] for s in solutions_xyz]
    lons, lats = transformer.transform(xs, ys)

    return [
        EulerSolution(
            lat=float(lat), lon=float(lon), depth_m=depth_m, base_level_nt=base_level,
            uncertainty_m=sigma_z0, depth_reference=depth_reference,
        )
        for lat, lon, (_, _, depth_m, base_level, sigma_z0) in zip(lats, lons, solutions_xyz)
    ]
