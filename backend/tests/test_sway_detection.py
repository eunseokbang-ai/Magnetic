"""Tests for IMU-based sway (sensor swing/rotation) detection -
processing/sway.py and its wiring into Project.run_pipeline in store.py.

Motivation: field comparisons of ground- vs UAV-based single-sensor
magnetometer surveys report sensor swaying/heading error - not motor
noise - as the dominant source of along-line data corruption, and note
that suspended-sensor rigs already carry gyroscope/accelerometer data that
could be used to flag those samples directly (see processing/sway.py's
module docstring). These tests check the detector in isolation and its
end-to-end effect on the processing pipeline.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.models import ProcessParams
from app.processing.sway import detect_sway
from app.store import Project

RNG = np.random.default_rng(0)


def test_detect_sway_no_imu_data_is_unavailable():
    n = 200
    gyro = np.full(n, np.nan)
    accel = np.full(n, np.nan)
    mask, info = detect_sway(gyro, accel)
    assert not mask.any()
    assert info["available"] is False
    assert info["n_points_flagged"] == 0


def test_detect_sway_flags_injected_gyro_burst():
    n = 500
    gyro = RNG.normal(0.2, 0.05, n)  # steady in-flight baseline
    accel = np.full(n, np.nan)  # accel not available - gyro alone should suffice
    burst = slice(240, 260)
    gyro[burst] = 6.0  # abnormal swing event
    mask, info = detect_sway(gyro, accel, threshold_k=4.0)

    assert info["available"] is True
    assert info["signal_used"] == "gyro"
    # every injected burst sample should be flagged, and only a small
    # fraction of the steady baseline should be (false positives from
    # normal statistical scatter, not many)
    assert mask[burst].all()
    assert mask.sum() < 40  # well under 2x the 20-sample burst


def test_detect_sway_combines_gyro_and_accel():
    n = 400
    gyro = RNG.normal(0.2, 0.05, n)
    accel = RNG.normal(1.0, 0.02, n)
    # two disjoint bursts, one visible only on each signal
    gyro[50:60] = 5.0
    accel[300:310] = 1.5
    mask, info = detect_sway(gyro, accel, threshold_k=4.0)

    assert info["signal_used"] == "gyro+accel"
    assert mask[50:60].all()
    assert mask[300:310].all()


def test_detect_sway_threshold_k_is_monotonic():
    n = 500
    gyro = RNG.normal(0.2, 0.05, n)
    gyro[np.arange(0, n, 7)] += RNG.normal(1.0, 0.5, len(np.arange(0, n, 7)))
    accel = np.full(n, np.nan)

    _, info_strict = detect_sway(gyro, accel, threshold_k=6.0)
    _, info_loose = detect_sway(gyro, accel, threshold_k=2.0)
    assert info_loose["n_points_flagged"] >= info_strict["n_points_flagged"]


def _make_synthetic_project(sway_enabled: bool, inject_gyro_burst: bool) -> tuple[Project, dict]:
    """4 parallel N-S survey lines with a smooth mag field, matching the
    equirectangular-approximation pattern used in test_crossover_leveling.py
    so detect_lines' UTM projection round-trips cleanly. One line carries a
    contiguous run of abnormally high gyro readings partway through, as if
    the suspended sensor swung during that stretch."""
    lat0, lon0 = 37.5, 127.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))

    rows = []
    t = pd.Timestamp("2026-01-01T00:00:00")
    turn_gap = pd.Timedelta(seconds=30)
    burst_row_indices = []
    for line_idx, x0 in enumerate([0.0, 50.0, 100.0, 150.0]):
        t += turn_gap
        ys = np.arange(0.0, 300.0, 2.0)  # 2 m/s at 1 Hz -> well past min_speed/min_line_length
        for j, y in enumerate(ys):
            is_burst = inject_gyro_burst and line_idx == 1 and 60 <= j < 80
            gyro = 5.0 if is_burst else float(RNG.normal(0.2, 0.05))
            if is_burst:
                burst_row_indices.append(len(rows))
            rows.append(
                {
                    "x": x0,
                    "y": y,
                    "mag_raw": 50000.0 + 0.01 * x0 + 0.01 * y,
                    "timestamp": t,
                    "gyro_mag": gyro,
                }
            )
            t += pd.Timedelta(milliseconds=100)  # 10 Hz, matching a real drone logger's sample rate

    df = pd.DataFrame(rows)
    df["lat"] = lat0 + df["y"] / m_per_deg_lat
    df["lon"] = lon0 + df["x"] / m_per_deg_lon
    df["point_id"] = np.arange(len(df), dtype=np.int64)
    df["altitude_msl_m"] = 100.0
    df["geoid_separation_m"] = 0.0
    df["altitude_ellipsoidal_m"] = 100.0
    df["speed_over_ground"] = np.nan
    df["accel_horiz_g"] = np.nan
    df["compass_x"] = np.nan
    df["compass_y"] = np.nan
    df["compass_z"] = np.nan

    base_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(df["timestamp"].min() - pd.Timedelta(minutes=1), df["timestamp"].max() + pd.Timedelta(minutes=1), periods=50),
            "mag": 50000.0,
        }
    )

    project = Project(id="test-sway")
    project.drone_raw = df
    project.base_raw = base_df

    params = ProcessParams()
    params.line_params.min_speed_mps = 0.1
    params.line_params.min_line_length_m = 50.0
    params.line_params.turn_buffer_m = 2.0
    params.line_params.max_gap_seconds = 5.0
    params.sway_detection.enabled = sway_enabled
    params.sway_detection.threshold_k = 4.0
    project.run_pipeline(params)
    return project, {"burst_row_indices": burst_row_indices}


def test_pipeline_excludes_high_sway_points():
    project, ctx = _make_synthetic_project(sway_enabled=True, inject_gyro_burst=True)
    assert project.sway_info["available"] is True
    assert project.sway_info["n_points_excluded"] >= 15  # the 20-sample burst, minus a couple lost to turn-buffer trimming

    df = project.processed
    burst_lines = df.loc[ctx["burst_row_indices"], "line_id"]
    assert (burst_lines < 0).all(), "every burst sample should have been excluded from its line"
    exclusion_reasons = df.loc[ctx["burst_row_indices"], "exclusion_reason"]
    assert (exclusion_reasons == "high_sway").all()


def test_pipeline_without_gyro_burst_excludes_nothing():
    project, _ctx = _make_synthetic_project(sway_enabled=True, inject_gyro_burst=False)
    assert project.sway_info["available"] is True
    assert project.sway_info["n_points_excluded"] == 0


def test_pipeline_sway_detection_disabled_keeps_burst_points_active():
    project, ctx = _make_synthetic_project(sway_enabled=False, inject_gyro_burst=True)
    assert project.sway_info["enabled"] is False
    df = project.processed
    burst_lines = df.loc[ctx["burst_row_indices"], "line_id"]
    # with the feature off, the burst samples are judged purely on
    # position/heading like any other sample, so most stay on their line
    assert (burst_lines >= 0).sum() > 0


if __name__ == "__main__":
    test_detect_sway_no_imu_data_is_unavailable()
    test_detect_sway_flags_injected_gyro_burst()
    test_detect_sway_combines_gyro_and_accel()
    test_detect_sway_threshold_k_is_monotonic()
    test_pipeline_excludes_high_sway_points()
    test_pipeline_without_gyro_burst_excludes_nothing()
    test_pipeline_sway_detection_disabled_keeps_burst_points_active()
    print("ALL CHECKS PASSED")
