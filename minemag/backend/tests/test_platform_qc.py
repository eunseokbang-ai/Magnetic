"""Platform-noise QC on a synthetic carrier and on the real fixture.

The synthetic reproduces the robot case: two probes, 200 Hz, 1.1 m/s,
white noise plus a 2 Hz gait swing that the probes see only partly in
common. The QC has to find the frequency, report the right spatial
period, and say that the gradient will not cancel it; with the same
swing fully common to both probes it has to say the opposite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.processing.platform_qc import GAIT_BAND_HZ, platform_qc
from magcore.magdrone_r3 import load_magdrone_r3_files
from conftest import R3_FIXTURE


def _survey(common_fraction: float, gait_nt: float = 80.0, noise_nt: float = 3.0, seed: int = 0, fs: float = 200.0,
            speed: float = 1.1, seconds: float = 120.0, target_nt: float = 0.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t_ms = np.arange(n) * (1000.0 / fs)
    t = t_ms / 1000.0
    y = speed * t
    gait_common = gait_nt * np.sin(2 * np.pi * 2.0 * t)
    gait_own = [gait_nt * np.sin(2 * np.pi * 2.0 * t + ph) for ph in (0.0, 2.0)]
    rows = []
    for s, x_off in ((1, 0.0), (2, 1.0)):
        v = common_fraction * gait_common + (1 - common_fraction) * gait_own[s - 1] + noise_nt * rng.standard_normal(n)
        if target_nt:
            v = v + target_nt * np.exp(-((y - 60.0) ** 2) / (2 * 0.4 ** 2))
        rows.append(pd.DataFrame({"t_ms": t_ms, "sensor": s, "x": x_off, "y": y, "tfa": v, "track": 1}))
    df = pd.concat(rows, ignore_index=True)
    df["file_index"] = 0
    df["timestamp"] = pd.Timestamp("2026-04-25") + pd.to_timedelta(df["t_ms"], unit="ms")
    df.attrs["files"] = [{"filename": "synthetic.asc"}]
    return df


def test_gait_noise_is_found_with_its_spatial_period_and_the_gradient_does_not_cancel_it():
    qc = platform_qc(_survey(common_fraction=0.5))
    f = qc["files"][0]
    assert f["sample_rate_hz"] == 200.0
    assert abs(f["speed_m_s"] - 1.1) < 0.05
    peak = f["spectrum"]["peaks"][0]
    assert abs(peak["hz"] - 2.0) < 0.15
    assert peak["in_gait_band"] is True
    assert abs(peak["spatial_period_m"] - 0.55) < 0.05
    assert f["gradient"]["cancels"] is False
    assert f["gradient"]["probe_correlation_above_1hz"] < 0.9
    assert f["per_probe"]["1"]["quiet_rms_nt"] > 40
    assert any("상쇄되지 않" in line for line in f["verdict"])
    assert any("주기성 잡음" in line for line in f["verdict"])


def test_common_mode_noise_is_cancelled_by_the_gradient():
    qc = platform_qc(_survey(common_fraction=1.0))
    g = qc["files"][0]["gradient"]
    assert g["cancels"] is True
    assert g["probe_correlation_above_1hz"] > 0.95
    assert g["cancellation_pct"] > 80
    assert any("줄입니다" in line for line in qc["files"][0]["verdict"])


def test_quiet_platform_reads_as_quiet_and_targets_are_excluded_from_the_noise():
    qc = platform_qc(_survey(common_fraction=0.5, gait_nt=0.0, noise_nt=3.0, target_nt=800.0))
    f = qc["files"][0]
    assert f["per_probe"]["1"]["quiet_rms_nt"] < 5          # the 800 nT target is not counted as noise
    assert f["quiet_fraction"] < 1.0
    assert "조용한 편" in f["verdict"][0]
    assert f["spectrum"] is None or not f["spectrum"]["peaks"] or not f["spectrum"]["peaks"][0]["in_gait_band"]


def test_the_real_fixture_runs_end_to_end():
    df = load_magdrone_r3_files([R3_FIXTURE.read_bytes()], [R3_FIXTURE.name])
    df["x"] = (df["lon"] - df["lon"].mean()) * 111320 * np.cos(np.radians(36.38))
    df["y"] = (df["lat"] - df["lat"].mean()) * 110540
    qc = platform_qc(df)
    f = qc["files"][0]
    assert f["sample_rate_hz"] == 20.0                      # every 10th 5 ms sample
    assert f["sensors"] == [1, 2]
    assert f["per_probe"]["1"]["quiet_rms_nt"] > 10        # the robot's gait, even thinned
    assert f["gradient"] is not None
    assert len(f["verdict"]) >= 2
    assert list(GAIT_BAND_HZ) == qc["gait_band_hz"]
