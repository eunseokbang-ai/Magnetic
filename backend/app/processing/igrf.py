"""IGRF correction: compute the modeled main-field total intensity at each
survey point and derive the magnetic anomaly (TMI - IGRF).

Uses ppigrf, which bundles IGRF coefficient files and needs no network
access at runtime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ppigrf


def compute_igrf_total_field(lat: np.ndarray, lon: np.ndarray, altitude_ellipsoidal_m: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Return the IGRF total field intensity (nT) at each point."""
    h_km = np.asarray(altitude_ellipsoidal_m, dtype=float) / 1000.0
    date_values = pd.DatetimeIndex(dates).to_pydatetime()

    be = np.empty(len(lat), dtype=float)
    bn = np.empty(len(lat), dtype=float)
    bu = np.empty(len(lat), dtype=float)

    # ppigrf broadcasts over identical-length coordinate arrays but expects
    # a single date (or one date shared by the whole batch) far more
    # efficiently than one call per unique timestamp; group by calendar day
    # since IGRF varies negligibly within a single survey day.
    dates_day = pd.Series(date_values).dt.floor("D")
    for day, idx in dates_day.groupby(dates_day).groups.items():
        idx = np.asarray(idx)
        b_e, b_n, b_u = ppigrf.igrf(
            np.asarray(lon)[idx], np.asarray(lat)[idx], h_km[idx], day.to_pydatetime()
        )
        be[idx] = np.ravel(b_e)
        bn[idx] = np.ravel(b_n)
        bu[idx] = np.ravel(b_u)

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
