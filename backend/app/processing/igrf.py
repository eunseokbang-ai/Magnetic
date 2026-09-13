"""IGRF correction: compute the modeled main-field total intensity at each
survey point and derive the magnetic anomaly (TMI - IGRF).

Uses ppigrf, which bundles IGRF coefficient files and needs no network
access at runtime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ppigrf
from scipy.interpolate import RegularGridInterpolator

# ppigrf.igrf's own cost is dominated by fixed per-call overhead, not by how
# many points are in the call - empirically flat at ~0.02-0.03s for a batch
# of anywhere from 1 to ~1000 points, then growing much faster than linearly
# beyond that (a 90,000-point call took ~6s in testing, ~200x the per-call
# floor for "only" 90x the points). A drone survey easily has hundreds of
# thousands of points per day, so calling it once per point (or even once
# per day on the full per-day point array) is by far the dominant cost of
# the whole processing pipeline - profiling a real 544k-point/6-day survey
# showed IGRF alone at 32s of a 37s total run.
#
# The field itself has no fine spatial structure to lose by not evaluating
# it at every point directly: IGRF is a low-degree spherical harmonic model,
# smooth over tens to hundreds of km, so its value across a drone survey's
# few-km footprint is captured accurately by a coarse regular grid with
# ordinary trilinear interpolation - the interpolation error this
# introduces is far below both the survey's own noise floor and the
# altitude/positioning uncertainty already present in the input data.
_GRID_LAT_NODES = 10
_GRID_LON_NODES = 10
_GRID_ALT_NODES = 5
_MIN_SPAN_DEG = 0.01  # ~1km - avoids a degenerate (zero-width) interpolation axis
_MIN_SPAN_KM = 0.05


def _padded_axis(values: np.ndarray, n_nodes: int, min_span: float) -> np.ndarray:
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if hi - lo < min_span:
        mid = (hi + lo) / 2.0
        lo, hi = mid - min_span / 2.0, mid + min_span / 2.0
    return np.linspace(lo, hi, n_nodes)


def _igrf_grid_interpolated(lon: np.ndarray, lat: np.ndarray, h_km: np.ndarray, date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """be/bn/bu at every (lon, lat, h_km) point, via ppigrf evaluated once
    on a coarse regular grid spanning the input's bounding box (plus a
    little padding) and trilinearly interpolated - see module docstring."""
    lat_axis = _padded_axis(lat, _GRID_LAT_NODES, _MIN_SPAN_DEG)
    lon_axis = _padded_axis(lon, _GRID_LON_NODES, _MIN_SPAN_DEG)
    alt_axis = _padded_axis(h_km, _GRID_ALT_NODES, _MIN_SPAN_KM)

    grid_lat, grid_lon, grid_alt = np.meshgrid(lat_axis, lon_axis, alt_axis, indexing="ij")
    shape = grid_lat.shape
    be_g, bn_g, bu_g = ppigrf.igrf(grid_lon.ravel(), grid_lat.ravel(), grid_alt.ravel(), date)
    be_g = np.ravel(be_g).reshape(shape)
    bn_g = np.ravel(bn_g).reshape(shape)
    bu_g = np.ravel(bu_g).reshape(shape)

    points = np.column_stack([lat, lon, h_km])
    be = RegularGridInterpolator((lat_axis, lon_axis, alt_axis), be_g)(points)
    bn = RegularGridInterpolator((lat_axis, lon_axis, alt_axis), bn_g)(points)
    bu = RegularGridInterpolator((lat_axis, lon_axis, alt_axis), bu_g)(points)
    return be, bn, bu


def compute_igrf_total_field(lat: np.ndarray, lon: np.ndarray, altitude_ellipsoidal_m: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Return the IGRF total field intensity (nT) at each point.

    Points with a non-finite lat/lon/altitude are left as NaN in the
    output rather than fed into the interpolation grid - matching what a
    direct per-point ppigrf.igrf() call would have done (NaN in, NaN
    out), and keeping a handful of NaN inputs (e.g. an altitude field
    that a given loader format never populates) from turning the coarse
    interpolation grid's own axis bounds into NaN and failing every
    point for that day, not just the bad ones."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    h_km = np.asarray(altitude_ellipsoidal_m, dtype=float) / 1000.0
    date_values = pd.DatetimeIndex(dates).to_pydatetime()

    be = np.full(len(lat), np.nan)
    bn = np.full(len(lat), np.nan)
    bu = np.full(len(lat), np.nan)
    finite = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(h_km)

    # Grouped by calendar day since IGRF varies negligibly within a single
    # survey day (secular variation is an annual-scale effect) - each
    # day's points share one coarse interpolation grid.
    dates_day = pd.Series(date_values).dt.floor("D")
    for day, idx in dates_day.groupby(dates_day).groups.items():
        idx = np.asarray(idx)[finite[np.asarray(idx)]]
        if len(idx) == 0:
            continue
        b_e, b_n, b_u = _igrf_grid_interpolated(lon[idx], lat[idx], h_km[idx], day.to_pydatetime())
        be[idx] = b_e
        bn[idx] = b_n
        bu[idx] = b_u

    return np.sqrt(be**2 + bn**2 + bu**2)


def mean_inclination_declination(lat: np.ndarray, lon: np.ndarray, altitude_ellipsoidal_m: np.ndarray, dates: pd.Series) -> tuple[float, float]:
    """Representative (mean-location) IGRF inclination/declination in
    degrees, used as the ambient-field direction for RTP/RTE filters."""
    lat_c = float(np.nanmean(lat))
    lon_c = float(np.nanmean(lon))
    h_km = float(np.nanmean(altitude_ellipsoidal_m)) / 1000.0
    dates_dt = pd.to_datetime(pd.Series(dates))
    date_c = pd.Timestamp(int(dates_dt.astype("int64").mean())).round("s").to_pydatetime()

    be, bn, bu = ppigrf.igrf(lon_c, lat_c, h_km, date_c)
    be, bn, bu = float(np.ravel(be)[0]), float(np.ravel(bn)[0]), float(np.ravel(bu)[0])

    # Geomagnetic convention: inclination measured from horizontal with
    # positive = downward, i.e. vertical component Z = -Bu (ppigrf's Bu is
    # positive upward).
    h_horiz = np.hypot(be, bn)
    inclination = float(np.degrees(np.arctan2(-bu, h_horiz)))
    declination = float(np.degrees(np.arctan2(be, bn)))
    return inclination, declination


def mean_field_intensity_nt(lat: np.ndarray, lon: np.ndarray, altitude_ellipsoidal_m: np.ndarray, dates: pd.Series) -> float:
    """Representative (mean-location) IGRF total field intensity in nT,
    used to convert susceptibility to induced magnetization for the 3D
    inversion forward model."""
    lat_c = float(np.nanmean(lat))
    lon_c = float(np.nanmean(lon))
    h_km = float(np.nanmean(altitude_ellipsoidal_m)) / 1000.0
    dates_dt = pd.to_datetime(pd.Series(dates))
    date_c = pd.Timestamp(int(dates_dt.astype("int64").mean())).round("s").to_pydatetime()

    be, bn, bu = ppigrf.igrf(lon_c, lat_c, h_km, date_c)
    be, bn, bu = float(np.ravel(be)[0]), float(np.ravel(bn)[0]), float(np.ravel(bu)[0])
    return float(np.sqrt(be**2 + bn**2 + bu**2))
