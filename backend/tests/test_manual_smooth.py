"""Validates manual distortion-removal smoothing: a user-marked run of
points along a line gets bridged (linearly interpolated) between the
last good value before it and the first good value after it, per line,
without disturbing unrelated lines or unflagged points."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.processing.manual_smooth import apply_manual_smoothing, _interpolate_flagged_runs


def test_interpolate_flagged_runs_bridges_interior_run():
    values = np.array([0.0, 0.0, 0.0, 50.0, 60.0, 55.0, 0.0, 0.0, 0.0])
    flagged = np.array([False, False, False, True, True, True, False, False, False])
    out = _interpolate_flagged_runs(values, flagged)
    assert np.allclose(out[3:6], [0.0, 0.0, 0.0])
    assert np.allclose(out[:3], values[:3])
    assert np.allclose(out[6:], values[6:])


def test_interpolate_flagged_runs_edge_run_holds_flat():
    values = np.array([99.0, 99.0, 0.0, 1.0, 2.0])
    flagged = np.array([True, True, False, False, False])
    out = _interpolate_flagged_runs(values, flagged, background_window=1)
    assert np.allclose(out[:2], [0.0, 0.0])


def test_interpolate_flagged_runs_median_window_ignores_single_noisy_boundary():
    # The sample right next to the flagged run is itself slightly elevated
    # (e.g. the dipole's tail bleeding just outside what the user marked) -
    # a single-anchor bridge would anchor to that noisy value, but the
    # window-median should see past it to the genuinely flat background.
    values = np.array([10.0, 10.0, 10.0, 10.0, 25.0, 40.0, 100.0, 100.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    flagged = np.array([False, False, False, False, False, True, True, True, False, False, False, False, False])

    out_single_anchor = _interpolate_flagged_runs(values, flagged, background_window=1)
    # old single-anchor behaviour: bridges from the noisy 25.0, so the
    # smoothed run isn't flat at the true 10.0 background.
    assert not np.allclose(out_single_anchor[5:8], 10.0, atol=1.0)

    out_median_window = _interpolate_flagged_runs(values, flagged, background_window=5)
    assert np.allclose(out_median_window[5:8], 10.0, atol=1e-6)


def _make_df():
    n = 20
    t = pd.date_range("2026-01-01", periods=n, freq="1s")
    line_id = np.array([0] * 10 + [1] * 10)
    anomaly = np.concatenate([np.full(10, 10.0), np.full(10, 20.0)])
    tmi = anomaly + 50000.0
    # inject a "house" distortion into the middle of line 0
    anomaly[4:7] += 30.0
    tmi[4:7] += 30.0
    return pd.DataFrame(
        {
            "point_id": np.arange(n),
            "timestamp": t,
            "line_id": line_id,
            "anomaly": anomaly,
            "tmi": tmi,
        }
    )


def test_apply_manual_smoothing_only_touches_flagged_line():
    df = _make_df()
    distorted_ids = set(df.loc[4:6, "point_id"])
    out = apply_manual_smoothing(df, distorted_ids)

    # distortion removed on line 0
    assert np.allclose(out.loc[4:6, "anomaly"], 10.0, atol=1e-6)
    # line 1 (untouched) unchanged
    assert np.allclose(out.loc[10:19, "anomaly"], df.loc[10:19, "anomaly"])
    # non-flagged points on line 0 unchanged
    assert np.allclose(out.loc[[0, 1, 2, 3, 7, 8, 9], "anomaly"], 10.0)


def test_apply_manual_smoothing_empty_point_ids_returns_independent_copy():
    # Must be an independent copy, not the same object, even when there's
    # nothing to smooth - store.py assigns the result to self.processed
    # while df itself is often self.processed_base, a pristine snapshot
    # that must never alias (and so accidentally mutate alongside) it.
    df = _make_df()
    out = apply_manual_smoothing(df, set())
    assert out is not df
    pd.testing.assert_frame_equal(out, df)
    out.loc[0, "anomaly"] = 999.0
    assert df.loc[0, "anomaly"] != 999.0


if __name__ == "__main__":
    test_interpolate_flagged_runs_bridges_interior_run()
    test_interpolate_flagged_runs_edge_run_holds_flat()
    test_apply_manual_smoothing_only_touches_flagged_line()
    test_apply_manual_smoothing_empty_point_ids_returns_independent_copy()
    print("ALL CHECKS PASSED")
