"""Store-level wiring tests for the UAV magnetics guideline-inspired
additions: file-level (tile) DC offset detection, the grid cell-size
guideline warning, notch-filter application inside run_pipeline, and the
multiscale-edges/power-spectrum/repeatability Project methods. Algorithm
correctness for each underlying function is covered by its own dedicated
test file (test_noise_qc.py, test_repeatability.py, test_multiscale_edges.py,
test_spectrum_notch.py) - these tests only check the store.py plumbing."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.models import (
    GridRequest,
    MultiscaleEdgeRequest,
    PowerSpectrumRequest,
    ProcessParams,
)
from app.store import Project

RNG = np.random.default_rng(0)

LAT0, LON0 = 37.5, 127.0
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(LAT0))


def _make_project(n_lines=4, level_offset_on_last_file=0.0, two_files=False, tone_hz=None, tone_amp=0.0, sample_hz=10.0):
    rows = []
    t = pd.Timestamp("2026-01-01T00:00:00")
    turn_gap = pd.Timedelta(seconds=30)
    dt = pd.Timedelta(seconds=1.0 / sample_hz)
    sample_i = 0
    for line_idx, x0 in enumerate(np.arange(n_lines) * 50.0):
        t += turn_gap
        ys = np.arange(0.0, 300.0, 2.0)
        for y in ys:
            true_value = 50000.0 + 0.01 * x0 + 0.01 * y
            value = true_value
            if tone_hz is not None:
                value += tone_amp * np.sin(2 * np.pi * tone_hz * sample_i / sample_hz)
            # last line's file gets a DC offset to simulate a moved base station
            source_file_index = 1 if (two_files and line_idx == n_lines - 1) else 0
            if source_file_index == 1:
                value += level_offset_on_last_file
            rows.append(
                {
                    "x": x0, "y": y, "mag_raw": value, "true_value": true_value,
                    "timestamp": t, "source_file_index": source_file_index,
                }
            )
            t += dt
            sample_i += 1

    df = pd.DataFrame(rows)
    df["lat"] = LAT0 + df["y"] / M_PER_DEG_LAT
    df["lon"] = LON0 + df["x"] / M_PER_DEG_LON
    df["point_id"] = np.arange(len(df), dtype=np.int64)
    df["altitude_msl_m"] = 100.0
    df["geoid_separation_m"] = 0.0
    df["altitude_ellipsoidal_m"] = 100.0
    df["speed_over_ground"] = np.nan
    df["gyro_mag"] = np.nan
    df["accel_horiz_g"] = np.nan
    df["compass_x"] = np.nan
    df["compass_y"] = np.nan
    df["compass_z"] = np.nan

    base_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                df["timestamp"].min() - pd.Timedelta(minutes=1), df["timestamp"].max() + pd.Timedelta(minutes=1), periods=50
            ),
            "mag": 50000.0,
        }
    )

    project = Project(id="test-guideline")
    project.drone_raw = df
    project.base_raw = base_df
    return project


def _base_params():
    params = ProcessParams()
    params.line_params.min_speed_mps = 0.1
    params.line_params.min_line_length_m = 50.0
    params.line_params.turn_buffer_m = 2.0
    params.line_params.max_gap_seconds = 5.0
    params.sway_detection.enabled = False
    params.heading_effect_calibration.enabled = False
    return params


def test_file_level_check_flags_injected_offset():
    project = _make_project(n_lines=4, two_files=True, level_offset_on_last_file=20.0)
    project.run_pipeline(_base_params())
    info = project.file_level_info
    assert info["available"] is True
    assert info["n_files"] == 2
    assert info["flagged_any"] is True
    flagged = [f for f in info["files"] if f["flagged"]]
    assert len(flagged) == 1
    assert flagged[0]["source_file_index"] == 1


def test_file_level_check_no_flag_without_offset():
    project = _make_project(n_lines=4, two_files=True, level_offset_on_last_file=0.0)
    project.run_pipeline(_base_params())
    info = project.file_level_info
    assert info["available"] is True
    assert info["flagged_any"] is False


def test_file_level_check_unavailable_with_single_file():
    project = _make_project(n_lines=3, two_files=False)
    project.run_pipeline(_base_params())
    assert project.file_level_info["available"] is False


def test_cell_size_guideline_warning_flags_too_coarse():
    project = _make_project(n_lines=4)
    project.run_pipeline(_base_params())
    assert project.line_spacing_m is not None
    too_coarse = project.line_spacing_m  # ratio = 1, well below the 3x floor
    overlay = project.get_grid_overlay(GridRequest(value="anomaly", cell_size_m=too_coarse, method="nearest"))
    assert overlay["cell_size_guideline_warning"] is not None


def test_cell_size_guideline_warning_absent_for_reasonable_size():
    project = _make_project(n_lines=4)
    project.run_pipeline(_base_params())
    good_cell = project.line_spacing_m / 5.0
    overlay = project.get_grid_overlay(GridRequest(value="anomaly", cell_size_m=good_cell, method="nearest"))
    assert overlay["cell_size_guideline_warning"] is None


def test_noise_qc_available_after_pipeline():
    project = _make_project(n_lines=4)
    project.run_pipeline(_base_params())
    assert project.noise_qc_info["available"] is True
    assert project.noise_qc_info["overall_rms_4th_diff_nt"] is not None


def test_notch_filter_reduces_injected_tone_noise():
    # Well below the default low-pass cutoff (1.0 Hz) so the tone survives
    # the main filter untouched - isolating the notch filter's own effect
    # rather than measuring what the low-pass would have removed anyway.
    tone_hz = 0.3
    params_off = _base_params()
    params_on = _base_params()
    # Isolate the notch filter's own effect - despike's adaptive Hampel
    # filter would otherwise also partially clip the injected sine tone's
    # peaks (it looks like a run of outliers locally), confounding the
    # comparison between "off" and "on".
    params_off.despike_params.enabled = False
    params_on.despike_params.enabled = False
    # A high Q (narrow notch) has a settling time of roughly Q/(pi*f), which
    # for the default Q=30 exceeds this short synthetic line's ~15s
    # duration - a lower Q here isolates "does the wiring apply the notch
    # at all" from "how narrow is too narrow for a 15s test signal"
    # (narrow-Q accuracy on a long, realistic signal is covered by
    # test_spectrum_notch.py).
    params_on.notch_filter.frequencies_hz = [tone_hz]
    params_on.notch_filter.quality_factor = 5.0

    project_off = _make_project(n_lines=1, tone_hz=tone_hz, tone_amp=5.0, sample_hz=10.0)
    project_off.run_pipeline(params_off)
    project_on = _make_project(n_lines=1, tone_hz=tone_hz, tone_amp=5.0, sample_hz=10.0)
    project_on.run_pipeline(params_on)

    # Compare against the known true (tone-free) signal rather than raw
    # std, so the deterministic geological trend doesn't dilute the
    # tone-removal signal.
    true_off = project_off.processed["x"] * 0.01 + 50000.0 + project_off.processed["y"] * 0.01
    resid_off = (project_off.processed["mag_filtered"] - true_off).to_numpy()
    true_on = project_on.processed["x"] * 0.01 + 50000.0 + project_on.processed["y"] * 0.01
    resid_on = (project_on.processed["mag_filtered"] - true_on).to_numpy()

    std_off = float(np.std(resid_off))
    std_on = float(np.std(resid_on))
    print(f"residual std vs true signal: notch off={std_off:.3f} on={std_on:.3f}")
    assert std_on < std_off * 0.5


def test_run_multiscale_edges_returns_points_without_crashing():
    project = _make_project(n_lines=4)
    project.run_pipeline(_base_params())
    result = project.run_multiscale_edges(
        MultiscaleEdgeRequest(value="anomaly", cell_size_m=10.0, heights_m=[0.0, 25.0], percentile=70.0)
    )
    assert "n_points" in result
    assert "points" in result


def test_get_power_spectrum_available_for_valid_line():
    project = _make_project(n_lines=4)
    project.run_pipeline(_base_params())
    line_id = int(project.processed.loc[project.processed["line_id"] >= 0, "line_id"].iloc[0])
    result = project.get_power_spectrum(PowerSpectrumRequest(line_id=line_id, value="mag_raw"))
    assert result["available"] is True


def test_repeatability_upload_and_analyze_end_to_end():
    project = _make_project(n_lines=4)
    project.run_pipeline(_base_params())

    # a tiny standalone repeat-flight dataset: same track flown twice
    rows = []
    t = pd.Timestamp("2026-02-01T00:00:00")
    for rep in range(2):
        for y in np.arange(0.0, 200.0, 2.0):
            rows.append({"x": 500.0, "y": y, "mag_raw": 50000.0 + 0.01 * y + RNG.normal(0, 0.3), "timestamp": t})
            t += pd.Timedelta(seconds=1)
        t += pd.Timedelta(seconds=60)
    rep_df = pd.DataFrame(rows)
    rep_df["lat"] = LAT0 + rep_df["y"] / M_PER_DEG_LAT
    rep_df["lon"] = LON0 + rep_df["x"] / M_PER_DEG_LON
    project.repeatability_raw = rep_df

    result = project.run_repeatability_analysis()
    assert result["available"] is True
    assert result["n_groups"] >= 1


if __name__ == "__main__":
    test_file_level_check_flags_injected_offset()
    test_file_level_check_no_flag_without_offset()
    test_file_level_check_unavailable_with_single_file()
    test_cell_size_guideline_warning_flags_too_coarse()
    test_cell_size_guideline_warning_absent_for_reasonable_size()
    test_noise_qc_available_after_pipeline()
    test_notch_filter_reduces_injected_tone_noise()
    test_run_multiscale_edges_returns_points_without_crashing()
    test_get_power_spectrum_available_for_valid_line()
    test_repeatability_upload_and_analyze_end_to_end()
    print("ALL CHECKS PASSED")
