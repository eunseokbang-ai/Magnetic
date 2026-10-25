"""The paper's pre-processing on a synthetic quadruped run: resampling,
adaptive Hampel, the 2 Hz zero-phase Gaussian low-pass, turn trimming
and the heading-correction measurement."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.processing.preprocess import PreprocessParams, gaussian_lowpass, preprocess

FS = 200.0


def _run(seed=0, gait_nt=80.0, spikes=True, anomaly_nt=300.0):
    """Two probes, 1 m/s, 6 N-S tracks 12 m long with turns, a 1 m-wide
    anomaly on track 3, 4.2 Hz gait noise and a few spikes."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = 0.0
    for tr in range(1, 7):
        n = int(12.0 * FS)                       # 12 s of walking at 1 m/s
        t = t0 + np.arange(n) / FS
        y = np.arange(n) / FS if tr % 2 else 12.0 - np.arange(n) / FS
        x_line = tr * 0.5
        # a turn: 1.5 s moving sideways at the end of the track
        nt = int(1.5 * FS)
        tt = t[-1] + np.arange(1, nt + 1) / FS
        yt = np.full(nt, y[-1])
        xt = x_line + np.arange(1, nt + 1) / FS * 0.33
        tt_all = np.r_[t, tt]
        for s, off in ((1, -0.5), (2, 0.5)):
            xs = np.r_[np.full(n, x_line), xt] + off
            ys = np.r_[y, yt]
            v = gait_nt * np.sin(2 * np.pi * 4.2 * tt_all + s) + 3.0 * rng.standard_normal(len(tt_all)) + 20.0 * s
            if tr == 3:
                v = v + anomaly_nt * np.exp(-((ys - 6.0) ** 2) / (2 * 0.3 ** 2))
            if spikes:
                # 0.3 s bursts: a single 5 ms spike mostly falls between
                # the resampled points (the paper resamples before the
                # Hampel filter), a burst does not
                for k in (300, 800, 1600, 2000, 2200):              # apart, away from the ends and from the y = 6 m anomaly
                    v[k:k + 60] += 400.0
            rows.append(pd.DataFrame({"t_ms": tt_all * 1000.0, "track": tr, "sensor": s, "x": xs, "y": ys, "lat": 36.0, "lon": 127.0,
                                      "alt": 50.0, "tfa": v, "bx": 0.0, "by": 0.0, "bz": 0.0}))
        t0 = tt[-1] + 2.0                          # a 2 s pause between tracks
    df = pd.concat(rows, ignore_index=True)
    df["file_index"] = 0
    df["timestamp"] = pd.Timestamp("2026-04-25") + pd.to_timedelta(df["t_ms"], unit="ms")
    df.attrs["files"] = [{"filename": "robot.asc"}]
    return df


def test_gaussian_lowpass_cutoff_is_minus_3db():
    fs, fc = 200.0, 2.0
    t = np.arange(0, 60, 1 / fs)
    for f, expect in ((0.5, 0.95), (2.0, 0.707), (4.2, 0.25)):   # exp(-2 pi^2 sigma^2 f^2): 0.22 at 4.2 Hz
        y = gaussian_lowpass(np.sin(2 * np.pi * f * t), fs, fc)
        gain = np.sqrt(2) * np.std(y[2000:-2000])
        if f == 2.0:
            assert abs(gain - expect) < 0.03
        elif f < fc:
            assert gain > expect
        else:
            assert gain < expect


def test_the_gait_noise_goes_and_the_anomaly_stays():
    df = _run()
    proc, rep = preprocess(df, PreprocessParams(lpf_hz=2.0))
    f = rep["files"][0]
    assert f["resample_hz"] == 10.0                    # 10 samples/m at 1 m/s
    assert f["lpf_hz"] == 2.0
    assert f["n_spikes"] >= 5
    # the 2 Hz Gaussian passes 4.2 Hz at 0.22: the fourth-difference noise drops about 4x
    assert f["noise_before_nt"] / f["noise_after_nt"] > 3.5
    # the bursts are gone from the filtered trace off the anomaly track
    off = proc[(proc["track"] != 3) & (proc["sensor"] == 2) & (proc["line_id"] >= 0)]
    resid = (off["tfa_proc"] - off["tfa_proc"].median()).abs()
    raw = (off["tfa_raw"] - off["tfa_raw"].median()).abs()
    # the Hampel removes the bursts (measured 67 of 75 samples); what
    # survives is a burst's edge sample, partly interpolated and under
    # the threshold, at about half the burst's height
    assert (resid > 120).sum() <= 0.2 * (raw > 120).sum()
    assert resid.max() < 260
    assert f["n_lines"] == 6 and 0.05 < f["turn_fraction"] < 0.3
    assert (proc["line_id"] >= 0).sum() > 0.7 * len(proc)
    # the anomaly on track 3, probe 1: peak within 10% of 300 nT above its own baseline
    p = proc[(proc["track"] == 3) & (proc["sensor"] == 1)].sort_values("y")
    base = np.median(p["tfa_proc"].to_numpy()[:20])
    peak = p["tfa_proc"].max() - base
    assert abs(peak - 300.0) < 30.0
    # and it did not move: the maximum sits at y = 6 m
    assert abs(float(p.loc[p["tfa_proc"].idxmax(), "y"]) - 6.0) < 0.15
    assert "tfa" in proc and np.allclose(proc["tfa"], proc["tfa_proc"])


def test_turn_samples_are_excluded_and_lines_keep_their_track_ids():
    df = _run(spikes=False)
    proc, _ = preprocess(df, PreprocessParams(lpf_hz=2.0, heading_correction=False))
    turns = proc[proc["line_id"] < 0]
    lines = proc[proc["line_id"] >= 0]
    assert sorted(lines["line_id"].unique()) == [1, 2, 3, 4, 5, 6]
    # the sideways turn segments move in x; the lines do not
    assert turns["x"].std() > 0.1
    for tr, g in lines.groupby("line_id"):
        assert g["x"].std() < 0.6                      # two probes 1 m apart, each on its own line


def test_per_file_cutoff_and_explicit_resampling():
    df = _run()
    proc, rep = preprocess(df, PreprocessParams(resample_hz=20.0, lpf_hz=2.0, lpf_hz_by_file={0: 5.0}))
    assert rep["files"][0]["resample_hz"] == 20.0
    assert rep["files"][0]["lpf_hz"] == 5.0
    # at 5 Hz the 4.2 Hz gait noise is barely touched
    assert rep["files"][0]["noise_before_nt"] / rep["files"][0]["noise_after_nt"] < 2.5


def test_heading_correction_is_measured_and_reported():
    df = _run(spikes=False, gait_nt=0.0)
    # a forward/backward step: odd tracks read 15 nT high
    odd = df["track"] % 2 == 1
    df.loc[odd, "tfa"] += 15.0
    proc, rep = preprocess(df, PreprocessParams(lpf_hz=2.0))
    h = rep["files"][0]["heading_correction"]
    assert h["enabled"] is True
    assert h["offset_nt"] is not None
    assert abs(abs(h["offset_nt"]) - 15.0) < 5.0
