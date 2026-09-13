"""Synthetic validation of base-station QC: trimming the noisy
installation/pickup transient at the start/end of a log, and despiking
mid-record outliers, before the series is used for diurnal correction."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.processing.base_qc import process_base_station, trim_base_transients

RNG = np.random.default_rng(0)


def _make_base_df(n_settled=600, n_transient=40, dt_seconds=1.0, quiet_noise=0.05, transient_noise=8.0):
    """A smooth settled diurnal drift, with high-variance handling noise
    tacked on at the very start and very end (installation/pickup)."""
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    n_total = n_transient + n_settled + n_transient
    t = pd.date_range(t0, periods=n_total, freq=pd.Timedelta(seconds=dt_seconds))

    settled_drift = 50000.0 + 3.0 * np.sin(np.linspace(0, 2 * np.pi, n_settled))
    quiet = settled_drift + RNG.normal(0, quiet_noise, n_settled)

    start_transient = settled_drift[0] + RNG.normal(0, transient_noise, n_transient)
    end_transient = settled_drift[-1] + RNG.normal(0, transient_noise, n_transient)

    mag = np.concatenate([start_transient, quiet, end_transient])
    return pd.DataFrame({"timestamp": t, "mag": mag}), n_transient, n_settled


def test_trims_installation_and_pickup_transients():
    df, n_transient, n_settled = _make_base_df()
    result = trim_base_transients(df, window_seconds=15.0, threshold_k=5.0, confirm_seconds=20.0, max_trim_fraction=0.3)

    # should trim close to (not necessarily exactly) the injected transient
    # length on both ends, and leave the settled interior intact.
    assert 0 < result["n_trimmed_start"] <= int(len(df) * 0.3)
    assert 0 < result["n_trimmed_end"] <= int(len(df) * 0.3)
    assert result["n_trimmed_start"] >= n_transient * 0.5
    assert result["n_trimmed_end"] >= n_transient * 0.5
    # rolling-window smoothing can bleed a few samples across the
    # transient/settled boundary, so allow a small margin here.
    assert len(result["trimmed"]) >= n_settled - 10


def test_trim_disabled_leaves_full_record():
    df, _, _ = _make_base_df()
    result = trim_base_transients(df, threshold_k=1e9)  # threshold impossibly high -> nothing exceeds it
    assert result["n_trimmed_start"] == 0
    assert result["n_trimmed_end"] == 0
    assert len(result["trimmed"]) == len(df)


def test_process_base_station_despikes_interior_outliers():
    df, n_transient, n_settled = _make_base_df()
    # inject a handful of sharp mid-record spikes into the settled section
    spike_idx = n_transient + n_settled // 2
    df = df.copy()
    df.loc[spike_idx, "mag"] += 200.0
    df.loc[spike_idx + 50, "mag"] -= 150.0

    result = process_base_station(
        df,
        trim_enabled=True, trim_window_seconds=15.0, trim_threshold_k=5.0, trim_confirm_seconds=20.0, trim_max_fraction=0.3,
        despike_enabled=True, despike_window_size=11, despike_threshold_k=4.0,
    )
    assert result.n_spikes_removed >= 2
    assert result.corrected["mag"].max() < df["mag"].max()
    # the corrected series should be much closer to the smooth settled
    # drift than the raw one (no giant spikes left, no transient extremes)
    assert result.corrected["mag"].std() < df["mag"].std()


def test_process_base_station_disabled_returns_raw_unchanged():
    df, _, _ = _make_base_df()
    result = process_base_station(df, trim_enabled=False, despike_enabled=False)
    assert len(result.corrected) == len(df)
    assert np.allclose(result.corrected["mag"].to_numpy(), df.sort_values("timestamp")["mag"].to_numpy())
    assert result.n_trimmed_start == 0
    assert result.n_trimmed_end == 0
    assert result.n_spikes_removed == 0


if __name__ == "__main__":
    test_trims_installation_and_pickup_transients()
    test_trim_disabled_leaves_full_record()
    test_process_base_station_despikes_interior_outliers()
    test_process_base_station_disabled_returns_raw_unchanged()
    print("ALL CHECKS PASSED")
