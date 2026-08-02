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
    fit_heading_effect_map,
    magnetic_heading,
)
from app.store import Project

RNG = np.random.default_rng(0)


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


if __name__ == "__main__":
    test_magnetic_heading_known_vectors()
    test_fit_heading_effect_map_recovers_synthetic_bias()
    test_fit_heading_effect_map_wraps_azimuth_seam()
    test_fit_heading_effect_map_rejects_too_few_points()
    test_pipeline_without_calibration_upload_is_noop()
    test_pipeline_heading_calibration_reduces_bias_correlation_with_orientation()
    print("ALL CHECKS PASSED")
