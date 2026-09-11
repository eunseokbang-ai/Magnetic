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

Two ways to build the calibration data are supported, matching Geometrics'
own MagArrow documentation (Calibration Flight Instructions; Survey Data
Processing Programs Guideline):
- fit_heading_effect_map: from a dedicated short calibration flight held
  over one low-gradient patch (a separate upload).
- build_turn_based_calibration: auto-extracted from the survey's own turn
  segments when no dedicated calibration flight exists - per the
  manufacturer's own guidance, "the best calibration data is usually
  located where the drone makes a turn."
cross_validate_heading_effect_map provides a held-out quality check for
either source, mirroring the Pass/Fail calibration-quality check in
Geometrics' own processing software.
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


def fit_heading_effect_map_from_deviation(theta_cal: np.ndarray, phi_cal: np.ndarray, deviation_cal: np.ndarray) -> HeadingEffectMap:
    """Core builder: wraps already-isolated ΔB(θ,φ) calibration points
    (see fit_heading_effect_map and build_turn_based_calibration for the
    two ways deviation_cal gets produced) into a queryable HeadingEffectMap,
    after dropping any non-finite rows and checking there's enough spread
    to interpolate from."""
    theta_cal = np.asarray(theta_cal, dtype=float)
    phi_cal = np.asarray(phi_cal, dtype=float)
    deviation_cal = np.asarray(deviation_cal, dtype=float)

    valid = np.isfinite(theta_cal) & np.isfinite(phi_cal) & np.isfinite(deviation_cal)
    if valid.sum() < _MIN_CALIBRATION_POINTS:
        raise HeadingCalibrationError(
            f"헤딩효과 캘리브레이션 자료가 부족합니다 (유효 포인트 {int(valid.sum())}개, "
            f"최소 {_MIN_CALIBRATION_POINTS}개 필요 - 나침반(Compass) 컬럼이 있는 지원 포맷인지 확인하세요)."
        )
    return HeadingEffectMap(theta_cal=theta_cal[valid], phi_cal=phi_cal[valid], deviation_cal=deviation_cal[valid])


def fit_heading_effect_map(theta_cal: np.ndarray, phi_cal: np.ndarray, mag_diurnal_corrected_cal: np.ndarray) -> HeadingEffectMap:
    """Builds a HeadingEffectMap from a dedicated calibration flight's
    diurnal-corrected total-field readings and their corresponding
    (theta, phi) headings. Following the paper: the calibration flight is
    expected to stay within a small, low magnetic-gradient patch of
    ground, so any remaining variation in the (already diurnal-corrected)
    readings is attributed entirely to the heading effect, isolated by
    simply subtracting the calibration data's own mean (equation of
    ΔB(θ,φ) in the Methods section - "the reading is centered around
    zero"). For calibration data auto-extracted from a survey's own turn
    segments (each potentially at a different location/background field),
    see build_turn_based_calibration instead, which demeans per segment
    rather than globally."""
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
    return fit_heading_effect_map_from_deviation(theta_cal[valid], phi_cal[valid], deviation_cal)


# A turn segment shorter than this contributes too few points to trust its
# own local mean as a stand-in for "the background field at this turn" -
# too easily thrown off by a single anomaly sample.
_MIN_POINTS_PER_TURN = 15


def build_turn_based_calibration(
    theta: np.ndarray,
    phi: np.ndarray,
    mag_diurnal_corrected: np.ndarray,
    turn_group_id: np.ndarray,
    min_points_per_turn: int = _MIN_POINTS_PER_TURN,
) -> HeadingEffectMap:
    """Builds a HeadingEffectMap from a survey's own turn segments instead
    of a dedicated calibration flight (see processing/lines.py's
    group_turn_segments) - the manufacturer's own calibration guide notes
    turns are usually the best calibration data available, since the drone
    sweeps a wide range of headings there. Unlike a single dedicated
    calibration flight held at one location, different turns sit at
    different points around the survey (and so at different real
    background field levels), so each turn's deviation is isolated by
    subtracting that turn's *own* mean rather than one global mean -
    keeping the per-turn assumption ("this turn's own footprint is small
    and low-gradient") close to what a real calibration flight relies on,
    without requiring the whole survey area to be low-gradient."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    values = np.asarray(mag_diurnal_corrected, dtype=float)
    turn_group_id = np.asarray(turn_group_id)

    finite = np.isfinite(theta) & np.isfinite(phi) & np.isfinite(values)
    theta_out, phi_out, deviation_out = [], [], []
    for group in np.unique(turn_group_id[turn_group_id >= 0]):
        idx = np.flatnonzero((turn_group_id == group) & finite)
        if idx.size < min_points_per_turn:
            continue
        theta_out.append(theta[idx])
        phi_out.append(phi[idx])
        deviation_out.append(values[idx] - float(np.mean(values[idx])))

    if not theta_out:
        raise HeadingCalibrationError(
            "측선 자료에서 캘리브레이션으로 쓸 만한 턴(turn) 구간을 찾지 못했습니다 "
            f"(턴마다 최소 {min_points_per_turn}개 포인트 필요) - 별도 캘리브레이션 비행 자료를 업로드하세요."
        )

    return fit_heading_effect_map_from_deviation(
        np.concatenate(theta_out), np.concatenate(phi_out), np.concatenate(deviation_out)
    )


def cross_validate_heading_effect_map(
    theta_cal: np.ndarray, phi_cal: np.ndarray, deviation_cal: np.ndarray, n_folds: int = 5, seed: int = 0
) -> dict:
    """Leave-out-fold quality check for a calibration dataset: repeatedly
    fits the map on most of the data and predicts the held-out remainder,
    then reports how far those out-of-fold predictions miss. A calibration
    flight that really does isolate pure heading effect (constant
    background field, low gradient) should let nearby points in
    (theta, phi) space predict each other well; a large residual instead
    signals contamination - e.g. real spatial gradient, drone motor noise,
    or (for build_turn_based_calibration) a turn that happened to pass
    over a real anomaly. Note this can't just re-query the already-fitted
    map at its own input points: scattered linear interpolation reproduces
    its inputs almost exactly there, which would trivially "pass" no
    matter how noisy the data actually is - genuine held-out points are
    required to measure anything meaningful."""
    theta_cal = np.asarray(theta_cal, dtype=float)
    phi_cal = np.asarray(phi_cal, dtype=float)
    deviation_cal = np.asarray(deviation_cal, dtype=float)
    n = len(theta_cal)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, min(n_folds, max(2, n // _MIN_CALIBRATION_POINTS)))

    residuals = []
    for i, held_out in enumerate(folds):
        train = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        try:
            fold_map = fit_heading_effect_map_from_deviation(theta_cal[train], phi_cal[train], deviation_cal[train])
        except HeadingCalibrationError:
            continue
        predicted, _extrapolated = fold_map.query(theta_cal[held_out], phi_cal[held_out])
        residuals.append(deviation_cal[held_out] - predicted)

    if not residuals:
        return {"available": False}

    residual = np.concatenate(residuals)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return {"available": False}

    return {
        "available": True,
        "n_folds": len(folds),
        "residual_std_nt": float(np.std(residual)),
        "residual_p2p_nt": float(np.ptp(residual)),
    }


def calibration_angular_coverage_deg(theta_cal: np.ndarray, phi_cal: np.ndarray) -> dict:
    """Rough diagnostic of how much of the (theta, phi) sphere a
    calibration flight actually swept, for surfacing to the user - a
    narrow spread means the compensation will only be reliable for survey
    samples flown at similar attitudes."""
    return {
        "theta_range_deg": [float(np.min(theta_cal)), float(np.max(theta_cal))],
        "phi_range_deg": [float(np.min(phi_cal)), float(np.max(phi_cal))],
    }
