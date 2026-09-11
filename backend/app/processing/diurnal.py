"""Diurnal correction: remove the time-varying background field measured by
a stationary base station from the drone's magnetic readings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DiurnalResult:
    corrected: np.ndarray
    coverage_pct: float
    base_reference_value: float
    base_time_range: tuple
    drone_time_range: tuple
    has_overlap: bool
    # True for each drone sample whose timestamp falls outside the base
    # station's own [t0, t1] coverage - np.interp has no extrapolation
    # mode, so those samples silently get base_interp clamped to the
    # nearest boundary base reading instead of tracking the (still
    # varying) real external field, a real mechanism for a day-specific
    # offset in exactly the stretch coverage_pct already warns about
    # (e.g. the base logger powered on after takeoff or off before
    # landing). Same length as drone_mag/corrected.
    extrapolated_mask: np.ndarray


def apply_diurnal_correction(
    drone_timestamps: pd.Series,
    drone_mag: np.ndarray,
    base_df: pd.DataFrame,
    time_offset_seconds: float = 0.0,
    reference: str = "mean",
) -> DiurnalResult:
    """Subtract (interpolated base reading - reference level) from drone_mag.

    time_offset_seconds shifts the base station clock to compensate for an
    unsynced logger clock (common when the base unit isn't GPS-time-synced);
    positive shifts the base timestamps later.
    reference: "mean" (mean of base samples inside the drone's flight
    window) or a fixed numeric string/float base level.
    """
    base = base_df.sort_values("timestamp").copy()
    base["timestamp"] = base["timestamp"] + pd.Timedelta(seconds=time_offset_seconds)

    base_t = base["timestamp"].astype("int64").to_numpy()
    drone_t = pd.Series(drone_timestamps).astype("int64").to_numpy()

    base_interp = np.interp(drone_t, base_t, base["mag"].to_numpy())

    t0, t1 = base_t.min(), base_t.max()
    within = (drone_t >= t0) & (drone_t <= t1)
    coverage_pct = float(100.0 * within.sum() / len(within)) if len(within) else 0.0

    if reference == "mean":
        window = base[
            (base["timestamp"] >= pd.Timestamp(drone_timestamps.min()))
            & (base["timestamp"] <= pd.Timestamp(drone_timestamps.max()))
        ]
        ref_value = float(window["mag"].mean()) if len(window) else float(base["mag"].mean())
    elif reference == "first":
        ref_value = float(base["mag"].iloc[0])
    else:
        ref_value = float(reference)

    correction = base_interp - ref_value
    corrected = np.asarray(drone_mag, dtype=float) - correction

    return DiurnalResult(
        corrected=corrected,
        coverage_pct=coverage_pct,
        base_reference_value=ref_value,
        base_time_range=(base["timestamp"].min(), base["timestamp"].max()),
        drone_time_range=(drone_timestamps.min(), drone_timestamps.max()),
        has_overlap=bool(coverage_pct > 0),
        extrapolated_mask=~within,
    )
