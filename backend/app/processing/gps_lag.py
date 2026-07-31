"""Corrects a constant time lag between the magnetometer and GPS position
streams within a single merged sample log. Even though every row shares
one timestamp column, the magnetometer's own internal filtering/telemetry
path can introduce a small delay relative to the GPS fix, so the position
stored on a row is not exactly where that magnetic reading was taken -
most visible as an anomaly from a small, sharp source appearing shifted a
few meters forward or backward along the flight direction, in opposite
directions on forward vs. reverse-flown lines. Correcting for a known
constant lag means resampling the position track at a time offset instead
of shifting the mag values themselves, since the mag reading's timestamp
is the one thing already known to be correct."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer

from .lines import project_to_local_xy


def apply_gps_mag_lag(df: pd.DataFrame, lag_seconds: float) -> pd.DataFrame:
    """Returns a copy of df with lat/lon (and altitude, if present)
    replaced by the position interpolated lag_seconds later along the
    original GPS track (negative lag_seconds looks earlier instead).
    mag_raw and every other column stay attached to their original
    timestamp - only the position assigned to each sample changes. Rows
    whose shifted time falls outside the original track are dropped, since
    they no longer have a defined position (analogous to takeoff/landing
    trimming elsewhere in the pipeline)."""
    if lag_seconds == 0.0:
        return df

    out = df.reset_index(drop=True).copy()
    t = (out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    x, y, epsg = project_to_local_xy(out["lat"].to_numpy(), out["lon"].to_numpy())
    t_shifted = t + lag_seconds

    x_shifted = np.interp(t_shifted, t, x, left=np.nan, right=np.nan)
    y_shifted = np.interp(t_shifted, t, y, left=np.nan, right=np.nan)

    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon_shifted, lat_shifted = transformer.transform(x_shifted, y_shifted)
    out["lat"] = lat_shifted
    out["lon"] = lon_shifted

    if "altitude_ellipsoidal_m" in out.columns:
        out["altitude_ellipsoidal_m"] = np.interp(
            t_shifted, t, out["altitude_ellipsoidal_m"].to_numpy(), left=np.nan, right=np.nan
        )

    valid = np.isfinite(out["lat"].to_numpy()) & np.isfinite(out["lon"].to_numpy())
    return out.loc[valid].reset_index(drop=True)
