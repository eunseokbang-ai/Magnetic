"""Levelling a base log that was recorded at more than one location.

The failure these guard against is quiet: the corrected anomaly comes out
looking perfectly reasonable, just with one flight's worth of it sitting a
constant offset away from the rest, before any levelling step runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.processing.base_segments import find_base_segments, level_base_segments
from app.processing.diurnal import apply_diurnal_correction


def _base_day(date: str, level: float, amplitude: float, start="09:00", end="11:00"):
    t = pd.date_range(f"{date} {start}", f"{date} {end}", freq="1s")
    diurnal = amplitude * np.sin(np.linspace(0, np.pi, len(t)))
    return pd.DataFrame({"timestamp": t, "mag": level + diurnal}), t, diurnal


def _drone_over(date, t_base, diurnal, start="09:20", end="10:40", true_field=50000.0):
    dt = pd.date_range(f"{date} {start}", f"{date} {end}", freq="1s")
    d = np.interp(dt.astype("int64"), t_base.astype("int64"), diurnal)
    return pd.DataFrame({"timestamp": dt, "mag": true_field + d})


def test_a_stationary_base_is_left_alone():
    base, _, _ = _base_day("2026-09-02", 50500.0, 30.0)
    result = level_base_segments(base)

    assert result.n_segments == 1
    assert result.max_offset_nt == 0.0
    assert not result.warnings
    np.testing.assert_allclose(result.base["mag"].to_numpy(), base["mag"].to_numpy())


def test_a_move_marked_by_a_logging_gap_is_found_and_levelled():
    b1, _, _ = _base_day("2026-09-02", 50500.0, 30.0)
    b2, _, _ = _base_day("2026-09-03", 50700.0, 25.0)  # 200 nT away
    base = pd.concat([b1, b2], ignore_index=True)

    result = level_base_segments(base)

    assert result.n_segments == 2
    assert result.segments[1].split_reason == "gap"
    # The two deployments end up on one level; what is left is the
    # difference in their diurnal shapes, not the 200 nT move.
    levelled = result.base
    day = levelled["timestamp"].dt.date.astype(str)
    gap = abs(levelled.loc[day == "2026-09-02", "mag"].median()
              - levelled.loc[day == "2026-09-03", "mag"].median())
    assert gap < 10.0
    assert any("옮긴 것으로 보입니다" in w for w in result.warnings)


def test_a_move_with_no_gap_in_logging_is_caught_as_a_step():
    t = pd.date_range("2026-09-02 09:00", "2026-09-02 13:00", freq="1s")
    diurnal = 20.0 * np.sin(np.linspace(0, np.pi, len(t)))
    mag = 50500.0 + diurnal
    mag[len(t) // 2:] += 150.0  # carried to a new spot, logger never stopped
    base = pd.DataFrame({"timestamp": t, "mag": mag})

    result = level_base_segments(base)

    assert result.n_segments == 2
    assert result.segments[1].split_reason == "step"
    assert abs(result.segments[1].level_nt - result.segments[0].level_nt) > 100.0


def test_a_magnetic_storm_is_not_mistaken_for_a_move():
    """Fast is not the same as discontinuous - a storm ramps, and the
    correction is exactly what should carry it through to the data."""
    t = pd.date_range("2026-09-02 09:00", "2026-09-02 13:00", freq="1s")
    ramp = np.linspace(0.0, 180.0, len(t))  # 180 nT over four hours
    base = pd.DataFrame({"timestamp": t, "mag": 50500.0 + ramp})

    result = level_base_segments(base)

    assert result.n_segments == 1
    assert result.max_offset_nt == 0.0


def test_a_handful_of_stray_samples_does_not_become_its_own_deployment():
    b1, _, _ = _base_day("2026-09-02", 50500.0, 10.0, start="09:00", end="11:00")
    stray = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-02 11:30", periods=5, freq="1s"),
        "mag": [50900.0] * 5,
    })
    result = level_base_segments(pd.concat([b1, stray], ignore_index=True))

    assert result.n_segments == 1


def test_levelling_removes_the_whole_day_offset_the_move_would_have_caused():
    """The end-to-end point: same geology on both days, base moved between
    them, and the corrected data should not know the difference."""
    b1, t1, d1 = _base_day("2026-09-02", 50500.0, 30.0)
    b2, t2, d2 = _base_day("2026-09-03", 50700.0, 30.0)
    base = pd.concat([b1, b2], ignore_index=True)
    drone = pd.concat([_drone_over("2026-09-02", t1, d1),
                       _drone_over("2026-09-03", t2, d2)], ignore_index=True)
    day = drone["timestamp"].dt.date.astype(str).to_numpy()

    def day_step(base_used):
        out = apply_diurnal_correction(drone["timestamp"], drone["mag"].to_numpy(), base_used)
        return abs(out.corrected[day == "2026-09-03"].mean()
                   - out.corrected[day == "2026-09-02"].mean())

    before = day_step(base)
    after = day_step(level_base_segments(base).base)

    assert before > 150.0        # the move lands in the data nearly in full
    assert after < 1.0           # and is gone once the base is levelled


def test_segments_are_reported_even_when_there_is_only_one():
    base, _, _ = _base_day("2026-09-02", 50500.0, 30.0)
    summary = level_base_segments(base).summary()

    assert summary["n_segments"] == 1
    assert len(summary["segments"]) == 1
    assert summary["segments"][0]["split_reason"] is None


def test_find_segments_returns_one_id_per_row():
    b1, _, _ = _base_day("2026-09-02", 50500.0, 30.0)
    b2, _, _ = _base_day("2026-09-03", 50700.0, 30.0)
    base = pd.concat([b1, b2], ignore_index=True)

    seg, reasons = find_base_segments(base)

    assert len(seg) == len(base)
    assert set(np.unique(seg)) == {0, 1}
    assert reasons == ["gap"]
