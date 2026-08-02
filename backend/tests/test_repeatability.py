"""Synthetic validation of the repeatability-test analysis: a small N-S
track flown 3 times (2 forward, 1 reverse) with known injected per-pass
noise and a known heading-direction-dependent bias, confirming the
grouping recovers all 3 passes as one group, the pooled 1-sigma noise is
close to the injected noise level, and the recovered heading error is
close to the injected bias."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.processing.lines import LineDetectionParams
from app.processing.repeatability import analyze_repeatability

RNG = np.random.default_rng(0)

LAT0, LON0 = 37.5, 127.0
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(LAT0))


def _xy_to_latlon(x, y):
    return LAT0 + y / M_PER_DEG_LAT, LON0 + x / M_PER_DEG_LON


def _geology(y):
    return 50000.0 + 0.01 * y  # smooth, slow background


def _make_pass(x0, y_start, y_end, t0, noise_std, bias=0.0, n=200):
    """A single straight pass along x=x0, y from y_start to y_end (forward
    if y_end > y_start, reverse otherwise), sampled at 1 point/second."""
    y = np.linspace(y_start, y_end, n)
    t = pd.date_range(t0, periods=n, freq="1s")
    value = _geology(y) + bias + RNG.normal(0, noise_std, n)
    lat, lon = _xy_to_latlon(np.full(n, x0), y)
    return pd.DataFrame({"timestamp": t, "lat": lat, "lon": lon, "mag_raw": value})


def _make_base_raw(t_start, t_end):
    t = pd.date_range(t_start - pd.Timedelta(minutes=1), t_end + pd.Timedelta(minutes=1), periods=200)
    return pd.DataFrame({"timestamp": t, "mag": 50000.0})


def _line_params():
    return LineDetectionParams(
        heading_lag_seconds=1.0, heading_tolerance_deg=20.0, min_speed_mps=0.1,
        min_line_length_m=50.0, turn_buffer_m=2.0, max_gap_seconds=5.0,
    )


def test_groups_repeated_passes_and_recovers_noise_and_heading_error():
    injected_sigma = 0.5
    injected_bias = 3.0  # heading-error-like offset for reverse passes

    t = pd.Timestamp("2026-01-01T00:00:00")
    gap = pd.Timedelta(seconds=60)

    p1 = _make_pass(100.0, 0.0, 400.0, t, injected_sigma)  # forward
    t += gap + pd.Timedelta(seconds=200)
    p2 = _make_pass(100.0, 400.0, 0.0, t, injected_sigma, bias=injected_bias)  # reverse
    t += gap + pd.Timedelta(seconds=200)
    p3 = _make_pass(100.0, 0.0, 400.0, t, injected_sigma)  # forward again
    t_end = t + pd.Timedelta(seconds=200)

    df = pd.concat([p1, p2, p3], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    base_raw = _make_base_raw(df["timestamp"].min(), t_end)

    result = analyze_repeatability(df, base_raw, line_params=_line_params())
    assert result["available"], result
    assert result["n_groups"] == 1, result
    assert result["n_passes_total"] == 3, result

    group = result["groups"][0]
    assert group["n_forward"] == 2
    assert group["n_reverse"] == 1

    print(f"noise_1sigma={group['noise_1sigma_nt']:.3f} (injected {injected_sigma}), "
          f"heading_error={group['heading_error_nt']:.3f} (injected {injected_bias})")
    assert abs(group["noise_1sigma_nt"] - injected_sigma) < 0.5 * injected_sigma + 0.1
    assert group["heading_error_nt"] is not None
    assert abs(group["heading_error_nt"] - injected_bias) < 0.5 * injected_bias + 0.3


def test_unavailable_with_single_pass():
    t = pd.Timestamp("2026-01-01T00:00:00")
    p1 = _make_pass(100.0, 0.0, 400.0, t, 0.5)
    base_raw = _make_base_raw(p1["timestamp"].min(), p1["timestamp"].max())
    result = analyze_repeatability(p1, base_raw, line_params=_line_params())
    assert result["available"] is False
    assert "reason" in result


def test_unavailable_with_two_far_apart_lines():
    """Two straight lines that never come close to each other (a normal
    2-line survey, not a repeat) should not be treated as repeats."""
    t = pd.Timestamp("2026-01-01T00:00:00")
    gap = pd.Timedelta(seconds=60)
    p1 = _make_pass(0.0, 0.0, 400.0, t, 0.5)
    t += gap + pd.Timedelta(seconds=200)
    p2 = _make_pass(500.0, 0.0, 400.0, t, 0.5)  # 500m away - not the same track
    t_end = t + pd.Timedelta(seconds=200)

    df = pd.concat([p1, p2], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    base_raw = _make_base_raw(df["timestamp"].min(), t_end)
    result = analyze_repeatability(df, base_raw, line_params=_line_params())
    assert result["available"] is False


if __name__ == "__main__":
    test_groups_repeated_passes_and_recovers_noise_and_heading_error()
    test_unavailable_with_single_pass()
    test_unavailable_with_two_far_apart_lines()
    print("ALL CHECKS PASSED")
