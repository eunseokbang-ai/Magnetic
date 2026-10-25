"""Detects samples taken while the (typically pole/boom-suspended) drone
magnetometer was swinging or rotating rather than hanging steady.

Motivates this over relying solely on the along-line low-pass smoothing in
processing/gridding.py: field comparisons of ground- vs UAV-based
single-sensor magnetometer surveys report that sensor swaying and the
resulting heading error - not motor noise - is the dominant source of
along-line streaking/corrugation in the raw signal (SPH Engineering's
UAV-based magnetometer comparison series). Along-line smoothing treats the
symptom after gridding; this instead flags the specific raw samples where
the IMU shows the sensor was actually moving abnormally, so they can be
excluded before gridding the same way turn/takeoff-landing samples already
are.

Only usable when the source CSV carries gyroscope/accelerometer columns
(currently just the generic/Geometrics MagArrow schema - see
io_/drone_loader.py); every other format's gyro_mag/accel_horiz_g columns
are all-NaN, in which case detect_sway is a no-op (available=False).
"""
from __future__ import annotations

import numpy as np

# Converts MAD to a consistent estimator of standard deviation under a
# normal distribution - same robust-statistics scale factor used by
# processing/despike.py, for consistency across the codebase's outlier
# detectors.
_MAD_TO_STD = 1.4826


def _robust_z(values: np.ndarray) -> np.ndarray:
    """Robust (median/MAD-based) z-score, NaN where the input is NaN."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    robust_std = mad * _MAD_TO_STD
    if not robust_std or not np.isfinite(robust_std):
        return np.zeros_like(values, dtype=float)
    return (values - med) / robust_std


def detect_sway(
    gyro_mag: np.ndarray,
    accel_horiz_g: np.ndarray,
    threshold_k: float = 4.0,
) -> tuple[np.ndarray, dict]:
    """Flags samples whose gyroscope rotation-rate magnitude and/or
    horizontal accelerometer magnitude is a robust-z-score outlier relative
    to the rest of the flight, i.e. an abnormal swing/rotation event rather
    than the sensor's normal (steady, low-amplitude) in-flight motion.

    Uses whichever of the two signals is actually available in the source
    file (gyro is checked first since angular rate is the more direct
    measure of "swinging"); combines both via elementwise max when both are
    present, so a sample only needs to be an outlier on one signal to be
    flagged.

    Returns (mask, info) where mask is a boolean array the same length as
    the inputs (all-False when neither signal is available) and info
    reports whether IMU data was usable, which signal(s) contributed, and
    the flagged fraction.
    """
    gyro_mag = np.asarray(gyro_mag, dtype=float)
    accel_horiz_g = np.asarray(accel_horiz_g, dtype=float)
    n = len(gyro_mag)

    has_gyro = np.isfinite(gyro_mag).sum() >= 10
    has_accel = np.isfinite(accel_horiz_g).sum() >= 10

    if not has_gyro and not has_accel:
        return np.zeros(n, dtype=bool), {
            "enabled": True,
            "available": False,
            "signal_used": None,
            "n_points_flagged": 0,
            "pct_points_flagged": 0.0,
            "threshold_k": threshold_k,
        }

    z_stack = []
    if has_gyro:
        z_stack.append(_robust_z(gyro_mag))
    if has_accel:
        z_stack.append(_robust_z(accel_horiz_g))
    combined_z = np.nanmax(np.vstack(z_stack), axis=0)

    mask = np.nan_to_num(combined_z, nan=0.0) > threshold_k
    signal_used = "gyro+accel" if (has_gyro and has_accel) else ("gyro" if has_gyro else "accel")

    return mask, {
        "enabled": True,
        "available": True,
        "signal_used": signal_used,
        "n_points_flagged": int(mask.sum()),
        "pct_points_flagged": float(mask.mean() * 100.0),
        "threshold_k": threshold_k,
    }
