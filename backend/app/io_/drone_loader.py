"""Drone magnetic survey CSV loader.

Handles the common logger export shape: one row per sample with
Date/Time/Latitude/Longitude/Mag plus a mix of GNSS/IMU fields where some
columns (Altitude, Hdop, ...) are only populated on GGA fix rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Date", "Time", "Latitude", "Longitude", "Mag"]
SPARSE_NUMERIC_COLUMNS = ["Altitude", "HeightOverEllipsoid", "Hdop", "SpeedOverGround"]


class DroneLoadError(ValueError):
    pass


def load_drone_csv(path_or_buffer) -> pd.DataFrame:
    """Parse a drone magnetometer CSV into a normalized DataFrame.

    Returns columns: point_id, timestamp, lat, lon, mag_raw, altitude_msl_m,
    geoid_separation_m, altitude_ellipsoidal_m, speed_over_ground (may be
    NaN if not present in source).

    Note: despite its name, the source `HeightOverEllipsoid` column matches
    the GGA "geoid separation" field position/magnitude (near-constant,
    tens of meters), not a true ellipsoidal height, so it is treated as
    geoid separation here: ellipsoidal height = MSL altitude + geoid
    separation.
    """
    # GgaSentence/RmcSentence fields embed commas inside single-quoted
    # strings, and every data row carries one trailing empty field from a
    # final comma; quotechar+index_col=False keep the C parser from
    # mis-aligning columns or inferring a spurious index column.
    df = pd.read_csv(
        path_or_buffer,
        skipinitialspace=True,
        quotechar="'",
        index_col=False,
        low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DroneLoadError(f"드론 파일에 필수 컬럼이 없습니다: {missing}")

    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        errors="coerce",
    )
    for col in ["Latitude", "Longitude", "Mag"] + SPARSE_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "MagValid" in df.columns:
        df["MagValid"] = pd.to_numeric(df["MagValid"], errors="coerce")
        df = df[(df["MagValid"].isna()) | (df["MagValid"] != 0)]

    df = df.dropna(subset=["timestamp", "Latitude", "Longitude", "Mag"])
    if df.empty:
        raise DroneLoadError("유효한 위치/자력 데이터가 없습니다.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # GGA-only fields (Altitude, Hdop, ...) are sparse; interpolate over time
    # so every 10Hz sample has a usable value.
    for col in SPARSE_NUMERIC_COLUMNS:
        if col in df.columns and df[col].notna().any():
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

    altitude_msl = df["Altitude"].astype(float).values if "Altitude" in df.columns else np.full(len(df), np.nan)
    geoid_sep = (
        df["HeightOverEllipsoid"].astype(float).values
        if "HeightOverEllipsoid" in df.columns
        else np.full(len(df), np.nan)
    )
    altitude_ellipsoidal = np.where(
        ~np.isnan(altitude_msl) & ~np.isnan(geoid_sep),
        altitude_msl + geoid_sep,
        altitude_msl,
    )

    out = pd.DataFrame(
        {
            "point_id": np.arange(len(df), dtype=np.int64),
            "timestamp": df["timestamp"].values,
            "lat": df["Latitude"].astype(float).values,
            "lon": df["Longitude"].astype(float).values,
            "mag_raw": df["Mag"].astype(float).values,
            "altitude_msl_m": altitude_msl,
            "geoid_separation_m": geoid_sep,
            "altitude_ellipsoidal_m": altitude_ellipsoidal,
            "speed_over_ground": (
                df["SpeedOverGround"].astype(float).values
                if "SpeedOverGround" in df.columns
                else np.nan
            ),
        }
    )
    return out


def load_drone_csvs(buffers: list) -> pd.DataFrame:
    """Load and concatenate multiple drone CSVs (e.g. several flights).

    Each file is parsed (and its sparse GGA-only fields interpolated)
    independently before concatenation, since interpolating across a time
    gap between two separate flights would be meaningless. The combined
    result is re-sorted by timestamp and point_id is reassigned 0..N-1.
    """
    if not buffers:
        raise DroneLoadError("드론 파일이 없습니다.")
    parts = [load_drone_csv(buf) for buf in buffers]
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined["point_id"] = np.arange(len(combined), dtype=np.int64)
    return combined
