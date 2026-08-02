"""Tests for the Zhang et al. (2022, The Leading Edge) heading-effect
calibration-and-compensation method - processing/heading_calibration.py
and its wiring into Project.run_pipeline in store.py.

Unlike processing/sway.py (which excludes samples taken while the sensor
was swinging), this recovers them: a short calibration flight measures the
magnetometer's reading offset as a function of its 3-axis-compass-derived
orientation (theta, phi), and that measured offset is then subtracted from
the survey data at each sample's own orientation.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.models import ProcessParams
from app.processing.heading_calibration import (
    HeadingCalibrationError,
    build_turn_based_calibration,
    cross_validate_heading_effect_map,
    fit_heading_effect_map,
    magnetic_heading,
)
from app.processing.lines import group_turn_segments
from app.store import Project

RNG = np.random.default_rng(0)


def _to_compass(theta_deg, phi_deg):
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    return np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)


def test_magnetic_heading_known_vectors():
    # field along local +Z -> polar angle 0, azimuth undefined but harmless
    theta, phi = magnetic_heading(np.array([0.0]), np.array([0.0]), np.array([1.0]))
    assert np.isclose(theta[0], 0.0)

    # field along local +X, in the XY plane -> polar angle 90, azimuth 0
    theta, phi = magnetic_heading(np.array([1.0]), np.array([0.0]), np.array([0.0]))
    assert np.isclose(theta[0], 90.0)
    assert np.isclose(phi[0], 0.0)

    # field along local +Y -> polar angle 90, azimuth 90
    theta, phi = magnetic_heading(np.array([0.0]), np.array([1.0]), np.array([0.0]))
    assert np.isclose(theta[0], 90.0)
    assert np.isclose(phi[0], 90.0)

    # NaN compass sample propagates as NaN, not a crash
    theta, phi = magnetic_heading(np.array([np.nan]), np.array([0.0]), np.array([1.0]))
    assert np.isnan(theta[0])


def test_fit_heading_effect_map_recovers_synthetic_bias():
    """A calibration flight that sweeps many (theta, phi) orientations
    over a synthetic heading-dependent bias should let the fitted map
    predict that same bias (up to interpolation error) at held-out
    orientations within its coverage."""

    def true_bias(theta_deg, phi_deg):
        return 3.0 * np.sin(np.radians(theta_deg)) * np.cos(np.radians(phi_deg))

    n = 800
    theta_cal = RNG.uniform(20, 160, n)
    phi_cal = RNG.uniform(0, 360, n)
    noise = RNG.normal(0, 0.05, n)
    readings = 50000.0 + true_bias(theta_cal, phi_cal) + noise  # constant background + bias + tiny noise

    heading_map = fit_heading_effect_map(theta_cal, phi_cal, readings)

    theta_q = RNG.uniform(30, 150, 200)
    phi_q = RNG.uniform(5, 355, 200)  # stay away from the 0/360 seam for this check
    deviation, extrapolated = heading_map.query(theta_q, phi_q)

    # heading_map's deviation is centered on the calibration mean of true_bias
    expected = true_bias(theta_q, phi_q) - np.mean(true_bias(theta_cal, phi_cal))
    assert np.nanmean(np.abs(deviation - expected)) < 0.3
    assert not extrapolated.all()


def test_fit_heading_effect_map_wraps_azimuth_seam():
    """A query near phi=0/360 should interpolate using calibration points
    on both sides of the seam, not treat it as a hard edge."""

    def true_bias(theta_deg, phi_deg):
        return 2.0 * np.cos(np.radians(phi_deg))  # smooth, periodic in phi

    n = 600
    theta_cal = RNG.uniform(60, 120, n)
    phi_cal = RNG.uniform(0, 360, n)
    readings = 50000.0 + true_bias(theta_cal, phi_cal)
    heading_map = fit_heading_effect_map(theta_cal, phi_cal, readings)

    theta_q = np.full(20, 90.0)
    phi_q = np.linspace(355.0, 365.0 % 360.0, 20)  # straddles the seam
    deviation, extrapolated = heading_map.query(theta_q, phi_q)
    expected = true_bias(theta_q, phi_q) - np.mean(true_bias(theta_cal, phi_cal))
    assert np.nanmean(np.abs(deviation - expected)) < 0.3


def test_fit_heading_effect_map_rejects_too_few_points():
    theta_cal = np.linspace(0, 180, 5)
    phi_cal = np.linspace(0, 360, 5)
    readings = np.full(5, 50000.0)
    try:
        fit_heading_effect_map(theta_cal, phi_cal, readings)
        assert False, "expected HeadingCalibrationError"
    except HeadingCalibrationError:
        pass


def test_group_turn_segments_isolates_contiguous_turn_events():
    n = 20
    df = pd.DataFrame(
        {
            "exclusion_reason": (
                ["off_azimuth_turn"] * 5 + [None] * 8 + ["turn_buffer"] * 3 + [None] * 2 + ["off_azimuth_turn"] * 2
            ),
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="100ms"),
        }
    )
    group_id = group_turn_segments(df)
    assert (group_id[:5] == group_id[0]).all() and group_id[0] >= 0
    assert (group_id[5:13] == -1).all()
    assert (group_id[13:16] == group_id[13]).all() and group_id[13] >= 0
    assert group_id[13] != group_id[0]  # a separate turn event, not merged with the first
    assert (group_id[16:18] == -1).all()
    assert (group_id[18:20] == group_id[18]).all() and group_id[18] not in (group_id[0], group_id[13])


def test_group_turn_segments_splits_on_time_gap():
    df = pd.DataFrame(
        {
            "exclusion_reason": ["off_azimuth_turn"] * 10,
            "timestamp": list(pd.date_range("2026-01-01", periods=5, freq="100ms"))
            + list(pd.date_range("2026-01-01T00:10:00", periods=5, freq="100ms")),
        }
    )
    group_id = group_turn_segments(df, max_gap_seconds=1.0)
    assert group_id[0] != group_id[5], "a big time gap should split what looks like one contiguous turn run into two"


def test_build_turn_based_calibration_recovers_bias_across_different_locations():
    """Three synthetic 'turns' at three different background field levels
    (simulating turns at different points around a survey, not one fixed
    calibration location) - per-group demeaning should let the pooled fit
    still recover the shared heading bias function despite that."""

    def true_bias(theta_deg, phi_deg):
        return 3.0 * np.sin(np.radians(theta_deg)) * np.cos(np.radians(phi_deg))

    theta_all, phi_all, values_all, group_all = [], [], [], []
    backgrounds = [50000.0, 50120.0, 49870.0]  # deliberately different per "turn"
    for g, bg in enumerate(backgrounds):
        n = 300
        theta = RNG.uniform(30, 150, n)
        phi = RNG.uniform(0, 360, n)
        theta_all.append(theta)
        phi_all.append(phi)
        values_all.append(bg + true_bias(theta, phi))
        group_all.append(np.full(n, g))

    theta_all, phi_all, values_all, group_all = (np.concatenate(a) for a in (theta_all, phi_all, values_all, group_all))
    heading_map = build_turn_based_calibration(theta_all, phi_all, values_all, group_all)

    theta_q = RNG.uniform(40, 140, 100)
    phi_q = RNG.uniform(10, 350, 100)
    deviation, _extrapolated = heading_map.query(theta_q, phi_q)
    # each group is demeaned around its own bias mean, so the recovered
    # surface is centered on the *average* of the per-group means, unlike
    # fit_heading_effect_map's single global center
    expected = true_bias(theta_q, phi_q) - np.mean([true_bias(theta_all, phi_all)[group_all == g].mean() for g in range(3)])
    assert np.nanmean(np.abs(deviation - expected)) < 0.5


def test_build_turn_based_calibration_raises_when_no_turn_has_enough_points():
    theta = RNG.uniform(0, 180, 10)
    phi = RNG.uniform(0, 360, 10)
    values = np.full(10, 50000.0)
    group_id = np.arange(10)  # every point its own "turn" - none reach min_points_per_turn
    try:
        build_turn_based_calibration(theta, phi, values, group_id)
        assert False, "expected HeadingCalibrationError"
    except HeadingCalibrationError:
        pass


def test_cross_validate_flags_contaminated_calibration():
    def true_bias(theta_deg, phi_deg):
        return 3.0 * np.sin(np.radians(theta_deg)) * np.cos(np.radians(phi_deg))

    n = 600
    theta = RNG.uniform(20, 160, n)
    phi = RNG.uniform(0, 360, n)
    clean_deviation = true_bias(theta, phi) - np.mean(true_bias(theta, phi))
    contaminated_deviation = clean_deviation + RNG.normal(0, 8.0, n)  # real, unexplainable noise

    clean_quality = cross_validate_heading_effect_map(theta, phi, clean_deviation)
    contaminated_quality = cross_validate_heading_effect_map(theta, phi, contaminated_deviation)

    assert clean_quality["available"] and contaminated_quality["available"]
    assert clean_quality["residual_p2p_nt"] < 2.0
    assert contaminated_quality["residual_p2p_nt"] > clean_quality["residual_p2p_nt"] * 3


def _make_synthetic_project_with_heading_bias(with_calibration: bool, enabled: bool = True):
    """4 parallel N-S survey lines, each sample carrying a compass reading
    whose (theta, phi) varies smoothly with along-line position (mimicking
    real in-flight sensor sway) and a matching injected heading-effect
    bias in mag_raw - the same physical relationship the paper's
    compensation method is meant to undo. A short calibration "flight"
    (a tight cluster of points sweeping many orientations) is optionally
    attached."""
    lat0, lon0 = 37.5, 127.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))

    def true_bias(theta_deg, phi_deg):
        return 4.0 * np.sin(np.radians(theta_deg)) * np.cos(np.radians(phi_deg))

    rows = []
    t = pd.Timestamp("2026-01-01T00:00:00")
    turn_gap = pd.Timedelta(seconds=30)
    for line_idx, x0 in enumerate([0.0, 50.0, 100.0, 150.0]):
        t += turn_gap
        ys = np.arange(0.0, 300.0, 2.0)
        for j, y in enumerate(ys):
            theta = 90.0 + 20.0 * np.sin(j / 8.0)  # oscillating attitude, as if swaying gently
            phi = (180.0 * line_idx + 10.0 * np.cos(j / 5.0)) % 360.0
            cx = np.sin(np.radians(theta)) * np.cos(np.radians(phi))
            cy = np.sin(np.radians(theta)) * np.sin(np.radians(phi))
            cz = np.cos(np.radians(theta))
            true_field = 50000.0 + 0.01 * x0 + 0.01 * y  # smooth "real" background
            rows.append(
                {
                    "x": x0,
                    "y": y,
                    "mag_raw": true_field + true_bias(theta, phi),
                    "timestamp": t,
                    "compass_x": cx,
                    "compass_y": cy,
                    "compass_z": cz,
                }
            )
            t += pd.Timedelta(milliseconds=100)

    df = pd.DataFrame(rows)
    df["lat"] = lat0 + df["y"] / m_per_deg_lat
    df["lon"] = lon0 + df["x"] / m_per_deg_lon
    df["point_id"] = np.arange(len(df), dtype=np.int64)
    df["altitude_msl_m"] = 100.0
    df["geoid_separation_m"] = 0.0
    df["altitude_ellipsoidal_m"] = 100.0
    df["speed_over_ground"] = np.nan
    df["gyro_mag"] = np.nan
    df["accel_horiz_g"] = np.nan

    base_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(df["timestamp"].min() - pd.Timedelta(minutes=1), df["timestamp"].max() + pd.Timedelta(minutes=1), periods=50),
            "mag": 50000.0,
        }
    )

    project = Project(id="test-heading-cal")
    project.drone_raw = df
    project.base_raw = base_df

    if with_calibration:
        n_cal = 1000
        theta_cal = RNG.uniform(50, 130, n_cal)
        phi_cal = RNG.uniform(0, 360, n_cal)
        cx = np.sin(np.radians(theta_cal)) * np.cos(np.radians(phi_cal))
        cy = np.sin(np.radians(theta_cal)) * np.sin(np.radians(phi_cal))
        cz = np.cos(np.radians(theta_cal))
        cal_rows = pd.DataFrame(
            {
                "timestamp": pd.date_range(base_df["timestamp"].iloc[0], periods=n_cal, freq="100ms"),
                "lat": lat0,
                "lon": lon0,
                "mag_raw": 50000.0 + true_bias(theta_cal, phi_cal),
                "altitude_msl_m": 100.0,
                "geoid_separation_m": 0.0,
                "altitude_ellipsoidal_m": 100.0,
                "speed_over_ground": np.nan,
                "gyro_mag": np.nan,
                "accel_horiz_g": np.nan,
                "compass_x": cx,
                "compass_y": cy,
                "compass_z": cz,
            }
        )
        project.calibration_raw = cal_rows

    params = ProcessParams()
    params.line_params.min_speed_mps = 0.1
    params.line_params.min_line_length_m = 50.0
    params.line_params.turn_buffer_m = 2.0
    params.line_params.max_gap_seconds = 5.0
    params.sway_detection.enabled = False  # isolate the calibration compensation effect
    params.heading_effect_calibration.enabled = enabled
    project.run_pipeline(params)
    return project, true_bias


def test_pipeline_without_calibration_upload_is_noop():
    project, _true_bias = _make_synthetic_project_with_heading_bias(with_calibration=False)
    assert project.heading_calibration_info["available"] is False
    assert project.heading_calibration_info["applied"] is False


def test_pipeline_heading_calibration_reduces_bias_correlation_with_orientation():
    """The key end-to-end check: without compensation, the processed
    anomaly should still correlate strongly with the injected
    orientation-dependent bias (reconstructed from the survey's own
    compass columns); with compensation, that correlation should collapse
    - the residual should look like real (smooth, position-only) signal,
    not motion noise."""
    project_off, true_bias = _make_synthetic_project_with_heading_bias(with_calibration=True, enabled=False)
    project_on, _ = _make_synthetic_project_with_heading_bias(with_calibration=True, enabled=True)

    assert project_off.heading_calibration_info["applied"] is False
    info_on = project_on.heading_calibration_info
    assert info_on["available"] is True
    assert info_on["applied"] is True
    assert info_on["n_survey_points_corrected"] > 500

    df_off = project_off.processed
    df_on = project_on.processed

    theta, phi = np.zeros(len(df_off)), np.zeros(len(df_off))
    from app.processing.heading_calibration import magnetic_heading

    theta, phi = magnetic_heading(df_off["compass_x"].to_numpy(), df_off["compass_y"].to_numpy(), df_off["compass_z"].to_numpy())
    injected_bias = true_bias(theta, phi)

    # remove the smooth (x, y) trend both share, so we isolate the
    # heading-correlated component specifically
    def detrended(values):
        return values - (0.01 * df_off["x"].to_numpy() + 0.01 * df_off["y"].to_numpy())

    corr_off = np.corrcoef(detrended(df_off["tmi"].to_numpy()), injected_bias)[0, 1]
    corr_on = np.corrcoef(detrended(df_on["tmi"].to_numpy()), injected_bias)[0, 1]

    assert corr_off > 0.5, f"sanity check: uncompensated data should correlate with injected bias, got {corr_off}"
    assert corr_on < corr_off * 0.3, f"compensation should sharply reduce heading-correlated noise: off={corr_off}, on={corr_on}"


def _make_turn_injected_project(contaminate_turns: bool = False, enabled: bool = True):
    """Same 4-line synthetic survey as _make_synthetic_project_with_heading_bias,
    but with no dedicated calibration_raw upload - instead, a short
    "turn" block (wide heading sweep, small/near-stationary spatial
    footprint) is spliced in after each line, exactly the kind of segment
    detect_lines already tags off_azimuth_turn and group_turn_segments
    picks up. When contaminate_turns=True, the turn points additionally
    carry random noise unrelated to heading (simulating a turn that
    happened to pass over real spatial gradient/anomaly/drone noise), to
    verify the calibration quality check catches it. Uses its own local,
    fixed-seed RNG (not the shared module-level RNG) so the resulting
    calibration quality is deterministic regardless of what other tests
    ran first in the same process."""
    rng = np.random.default_rng(42)
    lat0, lon0 = 37.5, 127.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))

    def true_bias(theta_deg, phi_deg):
        return 4.0 * np.sin(np.radians(theta_deg)) * np.cos(np.radians(phi_deg))

    rows = []
    t = pd.Timestamp("2026-01-01T00:00:00")
    for line_idx, x0 in enumerate([0.0, 50.0, 100.0, 150.0]):
        ys = np.arange(0.0, 300.0, 2.0)
        for j, y in enumerate(ys):
            theta = 90.0 + 20.0 * np.sin(j / 8.0)
            phi = (180.0 * line_idx + 10.0 * np.cos(j / 5.0)) % 360.0
            cx, cy, cz = _to_compass(theta, phi)
            true_field = 50000.0 + 0.01 * x0 + 0.01 * y
            rows.append(
                {"x": x0, "y": y, "mag_raw": true_field + true_bias(theta, phi), "timestamp": t, "compass_x": cx, "compass_y": cy, "compass_z": cz}
            )
            t += pd.Timedelta(milliseconds=100)

        # a turn: a small circular loop (constant speed, so it never dips
        # into the low-speed takeoff/landing exclusion) whose heading
        # continuously rotates through all directions, so most of it reads
        # as off-azimuth relative to the fixed N-S survey direction - just
        # like a real U-turn - while staying within a small (2*R) spatial
        # footprint like a real calibration flight's low-gradient patch.
        # The orientation (theta, phi) sweeps a Lissajous-style path
        # (non-commensurate theta/phi frequencies) rather than a simple
        # ramp, so it fills a genuine 2D patch of (theta, phi) space
        # instead of tracing a single thin curve through it - otherwise
        # scattered-interpolation queries land in gaps between turns and
        # systematically underestimate the true bias despite still
        # correlating with it (a real geometric limitation of turn-swept
        # calibration data, not something a test should paper over, but
        # also not what this test is meant to exercise).
        turn_y = ys[-1] + 2.0
        turn_true_field = 50000.0 + 0.01 * x0 + 0.01 * turn_y
        n_turn, period, radius = 80, 40.0, 6.0
        theta_phase = rng.uniform(0, 2 * np.pi)
        phi_phase = rng.uniform(0, 2 * np.pi)
        for k in range(n_turn):
            angle = 2 * np.pi * k / period
            x = x0 + radius * np.sin(angle)
            y = turn_y + radius * (1 - np.cos(angle))
            theta_t = 90.0 + 40.0 * np.sin(2 * np.pi * k / n_turn * 3.7 + theta_phase)
            phi_t = np.degrees(2 * np.pi * k / n_turn * 5.3 + phi_phase) % 360.0
            cx, cy, cz = _to_compass(theta_t, phi_t)
            noise = rng.normal(0, 8.0) if contaminate_turns else 0.0
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "mag_raw": turn_true_field + true_bias(theta_t, phi_t) + noise,
                    "timestamp": t,
                    "compass_x": cx,
                    "compass_y": cy,
                    "compass_z": cz,
                }
            )
            t += pd.Timedelta(milliseconds=100)
        t += pd.Timedelta(seconds=2)

    df = pd.DataFrame(rows)
    df["lat"] = lat0 + df["y"] / m_per_deg_lat
    df["lon"] = lon0 + df["x"] / m_per_deg_lon
    df["point_id"] = np.arange(len(df), dtype=np.int64)
    df["altitude_msl_m"] = 100.0
    df["geoid_separation_m"] = 0.0
    df["altitude_ellipsoidal_m"] = 100.0
    df["speed_over_ground"] = np.nan
    df["gyro_mag"] = np.nan
    df["accel_horiz_g"] = np.nan

    base_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(df["timestamp"].min() - pd.Timedelta(minutes=1), df["timestamp"].max() + pd.Timedelta(minutes=1), periods=50),
            "mag": 50000.0,
        }
    )

    project = Project(id="test-turn-cal")
    project.drone_raw = df
    project.base_raw = base_df

    params = ProcessParams()
    params.line_params.min_speed_mps = 0.1
    params.line_params.min_line_length_m = 50.0
    params.line_params.turn_buffer_m = 2.0
    params.line_params.max_gap_seconds = 5.0
    params.sway_detection.enabled = False
    # heading_correction (default on) estimates and removes its own
    # heading-group-dependent offset from reverse-flown line pairs - an
    # independent correction that would otherwise confound this test's
    # controlled comparison of the heading-effect-calibration feature
    # specifically, since it too responds to (differently-derived)
    # heading-correlated variation.
    params.heading_correction.enabled = False
    params.heading_effect_calibration.enabled = enabled
    params.heading_effect_calibration.auto_calibrate_from_turns = True
    project.run_pipeline(params)
    return project, true_bias


def test_pipeline_auto_calibrates_from_survey_turns_when_no_upload():
    project, true_bias = _make_turn_injected_project(contaminate_turns=False)
    info = project.heading_calibration_info
    assert info["available"] is True
    assert info["applied"] is True
    assert info["calibration_source"] == "auto_turns"
    assert info["quality_check"]["available"] is True
    assert info["quality_pass"] is True, info["quality_check"]

    # same bias-correlation check as the dedicated-file test, but comparing
    # against a run with the feature disabled entirely
    project_off, _ = _make_turn_injected_project(contaminate_turns=False, enabled=False)
    assert project_off.heading_calibration_info["applied"] is False

    # restrict to the kept survey line points (line_id >= 0) - the turn
    # points themselves are intentionally excluded from the final survey
    # output either way (same as any turn/off-azimuth segment), and their
    # true_field was generated from the turn's fixed anchor position while
    # their logged x/y trace a small loop, so detrending by x/y wouldn't
    # cleanly isolate the heading-only component for them the way it does
    # for real survey samples.
    df_on = project.processed
    df_on = df_on[df_on["line_id"] >= 0]
    df_off = project_off.processed
    df_off = df_off[df_off["line_id"] >= 0]
    theta, phi = magnetic_heading(df_off["compass_x"].to_numpy(), df_off["compass_y"].to_numpy(), df_off["compass_z"].to_numpy())
    injected_bias = true_bias(theta, phi)

    def detrended(df):
        return df["tmi"].to_numpy() - (0.01 * df["x"].to_numpy() + 0.01 * df["y"].to_numpy())

    d_off = detrended(df_off)
    d_on = detrended(df_on)
    # Variance reduction, not correlation, is the right metric here: an
    # imperfect (but still substantially correct) compensation leaves a
    # residual that is itself a *scaled-down copy* of the original bias,
    # e.g. residual = bias * (1 - recovery_fraction) + small orthogonal
    # error - which by construction stays highly correlated with the
    # original bias (correlation is scale-invariant) even after the
    # dominant share of its variance has genuinely been removed. Turn-swept
    # calibration data is sparser than a dedicated calibration flight (see
    # test_pipeline_heading_calibration_reduces_bias_correlation_with_orientation,
    # which uses 1000 calibration points vs ~200 here and *does* show a
    # sharp correlation collapse), so this test checks the metric that
    # actually reflects "how much of the noise did this remove."
    assert np.corrcoef(d_off, injected_bias)[0, 1] > 0.9, "sanity check: uncompensated data should track the injected bias almost exactly"
    assert d_on.std() < d_off.std() * 0.5, f"expected a large variance reduction: off_std={d_off.std():.3f}, on_std={d_on.std():.3f}"


def test_pipeline_turn_calibration_quality_check_fails_when_contaminated():
    project, _true_bias = _make_turn_injected_project(contaminate_turns=True)
    info = project.heading_calibration_info
    assert info["available"] is True
    assert info["calibration_source"] == "auto_turns"
    # A calibration surface that fails its own held-out quality check is
    # fit to noise/real gradient rather than a clean heading effect -
    # applying it anyway would inject that noise into the survey data
    # (confirmed on real survey data to roughly double the point-to-point
    # anomaly jump and show up as along-track corrugation in the gridded
    # output), so quality_pass gates whether the correction is applied at
    # all, mirroring Geometrics' own Pass/Fail calibration-file gate.
    assert info["applied"] is False
    assert info["quality_pass"] is False, info["quality_check"]
    assert "reason" in info


if __name__ == "__main__":
    test_magnetic_heading_known_vectors()
    test_fit_heading_effect_map_recovers_synthetic_bias()
    test_fit_heading_effect_map_wraps_azimuth_seam()
    test_fit_heading_effect_map_rejects_too_few_points()
    test_group_turn_segments_isolates_contiguous_turn_events()
    test_group_turn_segments_splits_on_time_gap()
    test_build_turn_based_calibration_recovers_bias_across_different_locations()
    test_build_turn_based_calibration_raises_when_no_turn_has_enough_points()
    test_cross_validate_flags_contaminated_calibration()
    test_pipeline_without_calibration_upload_is_noop()
    test_pipeline_heading_calibration_reduces_bias_correlation_with_orientation()
    test_pipeline_auto_calibrates_from_survey_turns_when_no_upload()
    test_pipeline_turn_calibration_quality_check_fails_when_contaminated()
    print("ALL CHECKS PASSED")
