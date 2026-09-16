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


def test_a_gap_is_reported_but_not_levelled_by_default():
    """The level difference across an overnight gap is normally the real
    day-to-day field change - the thing the base was deployed to capture -
    so it is flagged for the operator, not quietly removed."""
    b1, _, _ = _base_day("2026-09-02", 50500.0, 30.0)
    b2, _, _ = _base_day("2026-09-03", 50700.0, 25.0)
    base = pd.concat([b1, b2], ignore_index=True)

    result = level_base_segments(base)   # mode="steps"

    assert result.n_segments == 2
    assert result.segments[1].split_reason == "gap"
    assert result.max_offset_nt == 0.0
    np.testing.assert_allclose(result.base["mag"].to_numpy(), base["mag"].to_numpy())
    assert any("옮겼을 가능성" in w for w in result.warnings)


def test_a_small_day_to_day_difference_is_not_even_flagged():
    """A fixed station moves 5-20 nT between days; that must not look like
    a relocation."""
    b1, _, _ = _base_day("2026-09-02", 50500.0, 30.0)
    b2, _, _ = _base_day("2026-09-03", 50508.0, 25.0)
    result = level_base_segments(pd.concat([b1, b2], ignore_index=True))

    assert result.max_offset_nt == 0.0
    assert not any("옮겼을 가능성" in w for w in result.warnings)


def test_a_move_across_a_gap_is_levelled_when_the_operator_says_so():
    b1, _, _ = _base_day("2026-09-02", 50500.0, 30.0)
    b2, _, _ = _base_day("2026-09-03", 50700.0, 25.0)  # 200 nT away
    base = pd.concat([b1, b2], ignore_index=True)

    result = level_base_segments(base, mode="all")

    assert result.n_segments == 2
    assert result.segments[1].split_reason == "gap"
    # The two deployments end up on one level; what is left is the
    # difference in their diurnal shapes, not the 200 nT move.
    levelled = result.base
    day = levelled["timestamp"].dt.date.astype(str)
    gap = abs(levelled.loc[day == "2026-09-02", "mag"].median()
              - levelled.loc[day == "2026-09-03", "mag"].median())
    assert gap < 10.0


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
    after = day_step(level_base_segments(base, mode="all").base)

    assert before > 150.0        # the move lands in the data nearly in full
    assert after < 1.0           # and is gone once the base is levelled


def test_a_step_inside_continuous_logging_is_levelled_by_default():
    """Nothing in the field jumps between consecutive samples, so this one
    needs no permission."""
    t = pd.date_range("2026-09-02 09:00", "2026-09-02 13:00", freq="1s")
    mag = 50500.0 + 20.0 * np.sin(np.linspace(0, np.pi, len(t)))
    mag[len(t) // 2:] += 150.0
    base = pd.DataFrame({"timestamp": t, "mag": mag})

    result = level_base_segments(base)   # default mode

    assert result.n_segments == 2
    assert result.segments[1].levelled
    levelled = result.base["mag"].to_numpy()
    assert abs(np.median(levelled[len(t) // 2:]) - np.median(levelled[:len(t) // 2])) < 25.0


def test_mode_off_measures_without_changing_anything():
    t = pd.date_range("2026-09-02 09:00", "2026-09-02 13:00", freq="1s")
    mag = 50500.0 + np.zeros(len(t)); mag[len(t) // 2:] += 150.0
    base = pd.DataFrame({"timestamp": t, "mag": mag})

    result = level_base_segments(base, mode="off")

    assert result.n_segments == 2
    assert result.max_offset_nt == 0.0
    np.testing.assert_allclose(result.base["mag"].to_numpy(), mag)


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


def _logging_base(start, hours, level, amp=20.0, seed=0):
    """Continuous 1-minute base logging with a real diurnal curve on it."""
    t = pd.date_range(start, periods=int(hours * 60), freq="1min")
    h = t.hour + t.minute / 60.0
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"timestamp": t,
                         "mag": level + amp * np.sin(2 * np.pi * h / 24.0)
                                + rng.normal(0, 0.1, len(t))})


def test_the_correction_applied_is_the_step_that_was_detected():
    """The offset used to come from the segments' own medians, which is a
    different quantity from the step. Two segments covering different
    stretches of the diurnal curve have different medians whatever the
    step is - so a 10 nT jump could be "corrected" by any number at all.

    The break is at 00:00 UT because that is where this actually happens:
    observatory data is published one UT day at a time, so a record
    assembled from it carries its seams there.
    """
    before = _logging_base("2026-09-08 15:00", hours=9, level=50130.0, seed=1)
    after = _logging_base("2026-09-09 00:00", hours=3, level=50130.0, seed=2)
    after["mag"] += 10.0                       # the step, and nothing else
    base = pd.concat([before, after], ignore_index=True)

    result = level_base_segments(base, mode="steps")

    levelled = [s for s in result.segments if s.levelled]
    assert len(levelled) == 1
    assert abs(abs(levelled[0].offset_applied_nt) - 10.0) < 1.5, levelled[0].offset_applied_nt


def test_a_levelled_record_comes_out_continuous_across_the_break():
    before = _logging_base("2026-09-08 15:00", hours=9, level=50130.0, seed=3)
    after = _logging_base("2026-09-09 00:00", hours=3, level=50130.0, seed=4)
    after["mag"] -= 19.0
    base = pd.concat([before, after], ignore_index=True)

    out = level_base_segments(base, mode="steps").base.sort_values("timestamp")

    jumps = out["mag"].diff().abs()
    assert jumps.max() < 2.0, f"still steps {jumps.max():.2f} nT"


def test_levelling_a_step_does_not_move_the_record_before_it():
    """Everything ahead of the first correction keeps the level it was
    recorded at - only what follows a break is shifted onto it."""
    before = _logging_base("2026-09-08 15:00", hours=9, level=50130.0, seed=5)
    after = _logging_base("2026-09-09 00:00", hours=3, level=50130.0, seed=6)
    after["mag"] += 12.0
    base = pd.concat([before, after], ignore_index=True)

    out = level_base_segments(base, mode="steps").base.sort_values("timestamp")

    head = out[out["timestamp"] < "2026-09-09 00:00"]["mag"].to_numpy()
    assert np.allclose(head, before["mag"].to_numpy(), atol=1e-9)
