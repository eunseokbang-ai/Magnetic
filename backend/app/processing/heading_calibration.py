"""Heading-effect calibration and compensation for total-field
magnetometers, implementing the method described in Zhang, R., F.
Hanselmann, R. Fochs, M. Lawrence, K. Hurley, and M. Prouty, 2022,
"Motion noise reduction in drone magnetometry using a dead-zone-free
total-field magnetometer system," The Leading Edge, 41(7), 306-310
(https://doi.org/10.1190/tle41070306.1) - the Geometrics MagArrow's own
manufacturer paper.

Rather than discarding samples taken while the sensor's orientation was
changing (see processing/sway.py, which excludes them), this recovers
them: the magnetometer's own reading offset is a function of its
orientation relative to the earth's field alone - not of position - so a
short calibration flight that sweeps through many orientations over a
magnetically quiet patch of ground lets that offset function be measured
once and then subtracted from every survey sample at its own orientation.

Uses the built-in 3-axis compass (a vector magnetometer riding alongside
the total-field sensor) to compute each sample's heading as (theta, phi):
theta is the polar angle from the sensor's local +Z axis, phi is the
azimuth in its local XY plane - matching the paper's Figure 1 convention.
The heading effect only depends on this relative orientation, not on
absolute compass accuracy, which is why the method works despite compass
calibration typically being coarse.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata

# Calibration points closer together than this (in degrees, on the
# theta/phi grid) don't meaningfully add coverage - just a sanity floor so
# a degenerate near-duplicate calibration flight doesn't silently look
# "successful" with almost no real angular spread.
_MIN_CALIBRATION_POINTS = 30


class HeadingCalibrationError(ValueError):
    pass


def magnetic_heading(cx: np.ndarray, cy: np.ndarray, cz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polar angle theta (0-180 deg, from +Z) and azimuth phi (0-360 deg,
    in the XY plane) of the earth's field in the magnetometer's own
    compass-defined Cartesian frame - see Figure 1 of Zhang et al. (2022).
    NaN in, NaN out (propagates missing compass samples)."""
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    cz = np.asarray(cz, dtype=float)
    magnitude = np.sqrt(cx**2 + cy**2 + cz**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.degrees(np.arccos(np.clip(cz / magnitude, -1.0, 1.0)))
    phi = np.degrees(np.arctan2(cy, cx)) % 360.0
    return theta, phi


@dataclass
class HeadingEffectMap:
    """A fitted heading-effect deviation surface DeltaB(theta, phi), plus
    enough of the source calibration scatter to query it."""

    theta_cal: np.ndarray
    phi_cal: np.ndarray
    deviation_cal: np.ndarray

    def query(self, theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated DeltaB(theta, phi) at the given headings, and a
        boolean mask of which of those points fell outside the calibration
        flight's convex hull (nearest-neighbor extrapolated rather than
        linearly interpolated - still far better than leaving the sample
        uncorrected, but reported separately since it's a weaker estimate).
        theta/phi may contain NaN (e.g. missing compass data) - those
        positions return NaN with extrapolated=False."""
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        valid = np.isfinite(theta) & np.isfinite(phi)

        deviation = np.full(theta.shape, np.nan, dtype=float)
        extrapolated = np.zeros(theta.shape, dtype=bool)
        if not valid.any():
            return deviation, extrapolated

        # phi is periodic (0 deg == 360 deg); pad the calibration scatter
        # with copies shifted by a full turn on either side so linear
        # interpolation wraps correctly for query points near the 0/360
        # boundary instead of treating it as a hard edge.
        theta_pad = np.tile(self.theta_cal, 3)
        phi_pad = np.concatenate([self.phi_cal - 360.0, self.phi_cal, self.phi_cal + 360.0])
        dev_pad = np.tile(self.deviation_cal, 3)
        points = np.column_stack([theta_pad, phi_pad])
        query_points = np.column_stack([theta[valid], phi[valid]])

        linear = griddata(points, dev_pad, query_points, method="linear")
        needs_fallback = np.isnan(linear)
        if needs_fallback.any():
            nearest = griddata(points, dev_pad, query_points[needs_fallback], method="nearest")
            linear[needs_fallback] = nearest

        deviation[valid] = linear
        ext = np.zeros(valid.sum(), dtype=bool)
        ext[needs_fallback] = True
        extrapolated[valid] = ext
        return deviation, extrapolated


def fit_heading_effect_map(theta_cal: np.ndarray, phi_cal: np.ndarray, mag_diurnal_corrected_cal: np.ndarray) -> HeadingEffectMap:
    """Builds a HeadingEffectMap from a calibration flight's diurnal-
    corrected total-field readings and their corresponding (theta, phi)
    headings. Following the paper: the calibration flight is expected to
    stay within a small, low magnetic-gradient patch of ground, so any
    remaining variation in the (already diurnal-corrected) readings is
    attributed entirely to the heading effect, isolated by simply
    subtracting the calibration data's own mean (equation of ΔB(θ,φ) in
    the Methods section - "the reading is centered around zero")."""
    theta_cal = np.asarray(theta_cal, dtype=float)
    phi_cal = np.asarray(phi_cal, dtype=float)
    values = np.asarray(mag_diurnal_corrected_cal, dtype=float)

    valid = np.isfinite(theta_cal) & np.isfinite(phi_cal) & np.isfinite(values)
    if valid.sum() < _MIN_CALIBRATION_POINTS:
        raise HeadingCalibrationError(
            f"헤딩효과 캘리브레이션 자료가 부족합니다 (유효 포인트 {int(valid.sum())}개, "
            f"최소 {_MIN_CALIBRATION_POINTS}개 필요 - 나침반(Compass) 컬럼이 있는 지원 포맷인지 확인하세요)."
        )

    deviation_cal = values[valid] - float(np.mean(values[valid]))
    return HeadingEffectMap(theta_cal=theta_cal[valid], phi_cal=phi_cal[valid], deviation_cal=deviation_cal)


def calibration_angular_coverage_deg(theta_cal: np.ndarray, phi_cal: np.ndarray) -> dict:
    """Rough diagnostic of how much of the (theta, phi) sphere a
    calibration flight actually swept, for surfacing to the user - a
    narrow spread means the compensation will only be reliable for survey
    samples flown at similar attitudes."""
    return {
        "theta_range_deg": [float(np.min(theta_cal)), float(np.max(theta_cal))],
        "phi_range_deg": [float(np.min(phi_cal)), float(np.max(phi_cal))],
    }
