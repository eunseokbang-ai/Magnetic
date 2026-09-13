"""Choosing between two recordings of the same stretch of one line.

A flight interrupted mid-line backs up a little before resuming, so the
rejoined line has a stretch flown twice. Both recordings of it come from
the worst moment on the line - the end of one is an abrupt stop, the start
of the other is a restart - so the choice matters and is not arbitrary.

Taken from a real case: ACQU55 on the 2026-09 HaeNam block stopped for
9.8 s, backed up 10.5 m, and the resumed recording was six times noisier
than the one it was overlapping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.processing.line_overlap import resolve_line_overlaps


def _line_with_overlap(noise_first: float, noise_second: float, overlap_m: float = 20.0,
                       stop_seconds: float = 10.0, seed: int = 0):
    """One line, stopped mid-way, resumed `overlap_m` back down the line."""
    rng = np.random.default_rng(seed)
    step = 0.5

    y1 = np.arange(0.0, 200.0, step)
    t1 = pd.Timestamp("2026-09-09 00:15:00") + pd.to_timedelta(np.arange(len(y1)) * 0.1, unit="s")
    v1 = 50000.0 + rng.normal(0.0, noise_first, len(y1))

    y2 = np.arange(200.0 - overlap_m, 500.0, step)
    t2 = t1[-1] + pd.Timedelta(seconds=stop_seconds) + pd.to_timedelta(
        np.arange(len(y2)) * 0.1, unit="s")
    v2 = 50000.0 + rng.normal(0.0, noise_second, len(y2))

    df = pd.DataFrame({
        "timestamp": pd.DatetimeIndex(np.concatenate([t1.to_numpy(), t2.to_numpy()])),
        "x": np.zeros(len(y1) + len(y2)),
        "y": np.concatenate([y1, y2]),
        "mag_filtered": np.concatenate([v1, v2]),
    })
    line_id = np.zeros(len(df), dtype=int)
    reason = np.full(len(df), None, dtype=object)
    return df, line_id, reason


def _resolve(df, line_id, reason, **kw):
    # azimuth 90 = along +y, which is how the fixture is laid out
    return resolve_line_overlaps(df, "mag_filtered", 90.0, line_id, reason, **kw)


def test_the_quieter_recording_of_the_shared_stretch_is_kept():
    df, lid, reason = _line_with_overlap(noise_first=0.01, noise_second=0.08)

    result = _resolve(df, lid, reason)

    assert len(result.decisions) == 1
    d = result.decisions[0]
    assert d.kept == "first"
    assert d.reason == "noise"
    assert d.noise_second_nt > d.noise_first_nt
    assert d.n_points_excluded > 0


def test_it_keeps_the_second_recording_when_that_is_the_quiet_one():
    """The stop is not always the bad half - sometimes the drone was
    already being blown about before it gave up."""
    df, lid, reason = _line_with_overlap(noise_first=0.08, noise_second=0.01)

    d = _resolve(df, lid, reason).decisions[0]

    assert d.kept == "second"


def test_only_the_overlapping_samples_are_dropped():
    df, lid, reason = _line_with_overlap(noise_first=0.01, noise_second=0.08, overlap_m=20.0)

    result = _resolve(df, lid, reason)
    dropped = result.line_id < 0

    assert dropped.sum() == result.decisions[0].n_points_excluded
    # 20 m of overlap at 0.5 m spacing is about 40 samples; nothing like
    # the whole second half (600 samples) should go.
    assert 20 <= dropped.sum() <= 60
    assert (result.exclusion_reason[dropped] == "overlap_duplicate").all()


def test_an_operator_override_beats_the_noise_comparison():
    df, lid, reason = _line_with_overlap(noise_first=0.01, noise_second=0.08)

    d = _resolve(df, lid, reason, overrides={0: "second"}).decisions[0]

    assert d.kept == "second"
    assert d.reason == "override"


def test_a_line_flown_without_interruption_is_left_alone():
    rng = np.random.default_rng(1)
    y = np.arange(0.0, 500.0, 0.5)
    df = pd.DataFrame({
        "timestamp": pd.Timestamp("2026-09-09 00:15:00") + pd.to_timedelta(
            np.arange(len(y)) * 0.1, unit="s"),
        "x": np.zeros(len(y)),
        "y": y,
        "mag_filtered": 50000.0 + rng.normal(0.0, 0.02, len(y)),
    })
    result = _resolve(df, np.zeros(len(df), dtype=int), np.full(len(df), None, dtype=object))

    assert result.decisions == []
    assert (result.line_id >= 0).all()


def test_a_stop_with_no_backing_up_leaves_nothing_to_resolve():
    df, lid, reason = _line_with_overlap(noise_first=0.01, noise_second=0.08, overlap_m=0.0)

    result = _resolve(df, lid, reason)

    assert result.decisions == []
    assert (result.line_id >= 0).all()


def test_the_decision_is_reported_for_the_operator_to_check():
    df, lid, reason = _line_with_overlap(noise_first=0.01, noise_second=0.08)

    summary = _resolve(df, lid, reason).summary()[0]

    assert summary["line_id"] == 0
    assert summary["kept"] == "first"
    assert summary["overlap_m"] > 0
    assert summary["noise_first_nt"] is not None and summary["noise_second_nt"] is not None
    assert summary["first_time"] < summary["second_time"]
