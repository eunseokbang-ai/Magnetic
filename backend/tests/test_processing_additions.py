"""Unit tests (synthetic data, no network) for the processing-module
additions: THDR/upward-continuation/detrend/micro-leveling transforms,
Savitzky-Golay/moving-average filters, and loader QC (duplicate
timestamps, invalid coordinates).
"""
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.io_.drone_loader import load_drone_csv, load_drone_csvs
from app.io_.base_loader import load_base_csvs
from app.processing.filters import moving_average_filter, savgol_filter_1d
from app.processing.microlevel import apply_microleveling
from app.processing.transforms import total_horizontal_derivative, upward_continuation
from app.processing.trend import remove_regional_trend

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _synthetic_grid():
    easting = np.linspace(0, 500, 60)
    northing = np.linspace(0, 500, 60)
    E, N = np.meshgrid(easting, northing)
    return E, N, easting, northing


def test_total_horizontal_derivative_peaks_at_step_edge():
    E, N, easting, northing = _synthetic_grid()
    # a smooth step (tanh) along easting - THDR should peak right at the step
    step_x = 250.0
    grid = 100.0 * np.tanh((E - step_x) / 15.0)
    thdr = total_horizontal_derivative(grid, cell_size_m=float(easting[1] - easting[0]))
    peak_col = np.argmax(np.abs(thdr[30, :]))  # middle row
    peak_x = easting[peak_col]
    assert abs(peak_x - step_x) < 20.0, f"THDR peak at x={peak_x}, expected near {step_x}"
    print("THDR EDGE-LOCALIZATION CHECK PASSED")


def test_upward_continuation_smooths_and_attenuates():
    E, N, easting, northing = _synthetic_grid()
    cell = float(easting[1] - easting[0])
    # short-wavelength ripple + a broad Gaussian bump
    short = 10.0 * np.sin(2 * np.pi * E / 20.0)
    broad = 50.0 * np.exp(-((E - 250) ** 2 + (N - 250) ** 2) / (2 * 100.0**2))
    grid = short + broad

    continued = upward_continuation(grid, cell, height_m=15.0)
    # short-wavelength ripple should be attenuated far more than the broad bump
    assert np.std(continued) < np.std(grid)
    ripple_after = np.std(continued - broad)  # residual ripple energy after continuation
    assert ripple_after < np.std(short) * 0.5

    try:
        upward_continuation(grid, cell, height_m=0.0)
        assert False, "should reject non-positive height"
    except ValueError:
        pass
    print("UPWARD CONTINUATION CHECKS PASSED")


def test_detrend_removes_known_plane():
    E, N, easting, northing = _synthetic_grid()
    a, b, c = 0.3, -0.15, 100.0
    plane = a * E + b * N + c
    bump = 20.0 * np.exp(-((E - 250) ** 2 + (N - 250) ** 2) / (2 * 30.0**2))
    grid = plane + bump

    residual, trend = remove_regional_trend(grid, easting, northing, order=1)
    # the fitted trend should closely match the true plane (bump is a small
    # localized perturbation relative to the whole grid, so a 1st-order fit
    # is dominated by the plane)
    assert np.nanmax(np.abs(trend - plane)) < 5.0
    # residual should be close to the bump alone
    assert np.nanmax(np.abs(residual - bump)) < 5.0
    print("DETREND PLANE-RECOVERY CHECK PASSED")


def test_microleveling_suppresses_corrugation_preserves_trend():
    E, N, easting, northing = _synthetic_grid()
    cell = float(easting[1] - easting[0])
    line_azimuth_deg = 90.0  # lines run north-south
    line_spacing_m = 40.0

    trend = 0.05 * E + 30.0 * np.sin(2 * np.pi * E / 600.0)
    corrugation = 15.0 * np.sin(2 * np.pi * E / line_spacing_m)  # stripes across the lines
    grid = trend + corrugation

    leveled = apply_microleveling(
        grid, cell, line_azimuth_deg, line_spacing_m, strength=0.9, angle_tolerance_deg=15.0, wavelength_bandwidth_factor=1.5
    )
    corrugation_before = np.std(grid - trend)
    corrugation_after = np.std(leveled - trend)
    assert corrugation_after < corrugation_before * 0.4, (corrugation_before, corrugation_after)

    try:
        apply_microleveling(grid, cell, line_azimuth_deg, line_spacing_m=None, strength=0.9)
        assert False, "should reject missing line spacing"
    except (ValueError, TypeError):
        pass
    print("MICROLEVELING CORRUGATION-SUPPRESSION CHECK PASSED")


def _make_time_series(n=500, fs=10.0, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-01-01")
    timestamps = pd.Series([t0 + pd.Timedelta(seconds=i / fs) for i in range(n)])
    true_signal = 50.0 * np.sin(2 * np.pi * 0.05 * np.arange(n) / fs)
    noisy = true_signal + rng.normal(0, 5.0, size=n)
    return timestamps, true_signal, noisy


def test_moving_average_and_savgol_reduce_noise():
    timestamps, true_signal, noisy = _make_time_series()
    ma = moving_average_filter(noisy, timestamps, window_seconds=1.0)
    sg = savgol_filter_1d(noisy, timestamps, window_seconds=1.0, polyorder=3)

    err_noisy = np.std(noisy - true_signal)
    err_ma = np.std(ma - true_signal)
    err_sg = np.std(sg - true_signal)
    assert err_ma < err_noisy
    assert err_sg < err_noisy
    print(f"noise std={err_noisy:.2f} moving_avg={err_ma:.2f} savgol={err_sg:.2f}")
    print("FILTER NOISE-REDUCTION CHECKS PASSED")


def _drone_csv_bytes(rows: list[dict]) -> io.BytesIO:
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def test_drone_loader_removes_duplicate_timestamps_and_bad_coords():
    base_lat, base_lon = 46.5, 106.27
    rows = []
    for i in range(20):
        rows.append(
            {
                "Date": "2026-07-24",
                "Time": f"06:00:{i:02d}.0",
                "Latitude": base_lat + i * 0.0001,
                "Longitude": base_lon + i * 0.0001,
                "Mag": 59000.0 + i,
            }
        )
    # duplicate an existing timestamp exactly (within the same file - this
    # exercises row-level identical-timestamp collision handling before
    # any cross-file concat happens)
    rows.append(dict(rows[5]))
    # invalid coordinates: out-of-range latitude, and the (0,0) sentinel
    rows.append({"Date": "2026-07-24", "Time": "06:00:30.0", "Latitude": 999.0, "Longitude": base_lon, "Mag": 59000.0})
    rows.append({"Date": "2026-07-24", "Time": "06:00:31.0", "Latitude": 0.0, "Longitude": 0.0, "Mag": 59000.0})

    df = load_drone_csv(_drone_csv_bytes(rows))
    assert df.attrs["n_invalid_coords_removed"] == 2, df.attrs
    # the exact in-file duplicate timestamp row survives load_drone_csv
    # (dedup happens at the load_drone_csvs concat stage, see below) but
    # invalid coordinate rows must already be gone.
    assert len(df) == 21, len(df)
    assert ((df["lat"] == 999.0) | ((df["lat"] == 0.0) & (df["lon"] == 0.0))).sum() == 0

    combined = load_drone_csvs([_drone_csv_bytes(rows)])
    assert combined.attrs["n_duplicate_timestamps_removed"] == 1, combined.attrs
    assert len(combined) == 20, len(combined)
    print("DRONE LOADER QC CHECKS PASSED")


def test_base_loader_removes_duplicate_timestamps():
    lines = []
    for i in range(10):
        lines.append(f"0,{59000.0+i},0,오전 6:00:{i:02d},07/24/26,0")
    lines.append(lines[3])  # exact duplicate row
    csv_bytes = ("\n".join(lines) + "\n").encode("utf-8-sig")
    combined = load_base_csvs([io.BytesIO(csv_bytes)])
    assert combined.attrs["n_duplicate_timestamps_removed"] == 1, combined.attrs
    assert len(combined) == 10, len(combined)
    print("BASE LOADER QC CHECKS PASSED")


if __name__ == "__main__":
    test_total_horizontal_derivative_peaks_at_step_edge()
    test_upward_continuation_smooths_and_attenuates()
    test_detrend_removes_known_plane()
    test_microleveling_suppresses_corrugation_preserves_trend()
    test_moving_average_and_savgol_reduce_noise()
    test_drone_loader_removes_duplicate_timestamps_and_bad_coords()
    test_base_loader_removes_duplicate_timestamps()
    print("\nALL CHECKS PASSED")
