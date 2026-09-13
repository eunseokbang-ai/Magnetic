"""One physical line should come back as one line id.

A survey line gets broken up for reasons that have nothing to do with the
survey: the drone pauses, gets pushed off heading and recovers, or the
flight is interrupted and resumed as a separate file. Each fragment
otherwise gets its own level correction, its own row in the line list, and
has to be edited separately.

What must NOT be merged is the other thing that looks similar: a line
deliberately flown twice. Those are two passes the operator chooses
between.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.processing.lines import LineDetectionParams, detect_lines, merge_continued_lines

LAT0, LON0 = 34.55, 126.40
M_PER_DEG_LAT = 111_320.0


def _leg(n, x0, y0, dx, dy, t0, dt=0.1):
    """A straight run of n samples starting at local metres (x0, y0)."""
    x = x0 + dx * np.arange(n)
    y = y0 + dy * np.arange(n)
    t = pd.Timestamp(t0) + pd.to_timedelta(np.arange(n) * dt, unit="s")
    return x, y, t


def _frame(parts, source_index=None):
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    t = pd.DatetimeIndex(np.concatenate([p[2].to_numpy() for p in parts]))
    lat = LAT0 + y / M_PER_DEG_LAT
    lon = LON0 + x / (M_PER_DEG_LAT * np.cos(np.radians(LAT0)))
    df = pd.DataFrame({"timestamp": t, "lat": lat, "lon": lon,
                       "mag_raw": np.zeros(len(x))})
    if source_index is not None:
        df["source_file_index"] = np.concatenate(
            [np.full(len(p[0]), s) for p, s in zip(parts, source_index)])
    return df


def _n_lines(df):
    return len({int(v) for v in df["line_id"].unique() if v >= 0})


def test_a_pause_mid_line_does_not_start_a_new_line():
    """The drone stops, hovers for a few seconds, then carries on - a
    descent down a cliff face does this, and it is still one line."""
    fly1 = _leg(400, 0.0, 0.0, 0.0, 0.5, "2026-09-02 09:00:00")
    hover = _leg(100, 0.0, 200.0, 0.02, 0.02, "2026-09-02 09:00:40")
    fly2 = _leg(400, 0.0, 202.0, 0.0, 0.5, "2026-09-02 09:00:50")
    out = detect_lines(_frame([fly1, hover, fly2]))

    assert _n_lines(out) == 1


def test_a_turn_onto_the_next_line_still_separates_them():
    down = _leg(400, 0.0, 0.0, 0.0, 0.5, "2026-09-02 09:00:00")
    turn = _leg(120, 0.0, 200.0, 0.42, 0.0, "2026-09-02 09:00:40")
    up = _leg(400, 50.0, 200.0, 0.0, -0.5, "2026-09-02 09:00:52")
    out = detect_lines(_frame([down, turn, up]))

    assert _n_lines(out) == 2


def test_a_flight_resumed_in_a_separate_file_is_one_line():
    """The break the bridging cannot see: the second half arrives as its
    own file, so nothing in that file knows the first half exists."""
    first = _leg(300, 0.0, 0.0, 0.0, 0.5, "2026-09-02 09:00:00")
    second = _leg(300, 0.0, 160.0, 0.0, 0.5, "2026-09-02 09:20:00")
    df = _frame([first, second], source_index=[0, 1])
    out = detect_lines(df)

    assert _n_lines(out) == 1


def test_a_small_overlap_where_the_flight_picked_up_again_is_still_one_line():
    first = _leg(300, 0.0, 0.0, 0.0, 0.5, "2026-09-02 09:00:00")
    second = _leg(300, 0.0, 130.0, 0.0, 0.5, "2026-09-02 09:20:00")  # doubles back 20 m
    out = detect_lines(_frame([first, second], source_index=[0, 1]))

    assert _n_lines(out) == 1


def test_a_deliberately_re_flown_line_stays_two_lines():
    """Same ground twice over is two passes to choose between, not one
    line - this is the case the merge must not swallow."""
    first = _leg(300, 0.0, 0.0, 0.0, 0.5, "2026-09-02 09:00:00")
    repeat = _leg(300, 0.0, 0.0, 0.0, 0.5, "2026-09-02 09:30:00")
    out = detect_lines(_frame([first, repeat], source_index=[0, 1]))

    assert _n_lines(out) == 2


def test_the_next_line_over_is_never_merged_however_it_lines_up():
    a = _leg(300, 0.0, 0.0, 0.0, 0.5, "2026-09-02 09:00:00")
    b = _leg(300, 50.0, 160.0, 0.0, 0.5, "2026-09-02 09:20:00")
    out = detect_lines(_frame([a, b], source_index=[0, 1]))

    assert _n_lines(out) == 2


def test_merging_renumbers_in_the_order_the_lines_were_flown():
    # Three separate lines, one line spacing apart, flown out of id order.
    df = pd.DataFrame({
        "x": [0.0, 0.0, 50.0, 50.0, 100.0, 100.0],
        "y": [0.0, 10.0, 0.0, 10.0, 0.0, 10.0],
        "timestamp": pd.to_datetime([
            "2026-09-02 09:00:00", "2026-09-02 09:00:01",   # flown 2nd
            "2026-09-02 09:10:00", "2026-09-02 09:10:01",   # flown 3rd
            "2026-09-02 08:50:00", "2026-09-02 08:50:01",   # flown 1st
        ]),
    })
    merged = merge_continued_lines(df, np.array([0, 0, 1, 1, 2, 2]), 90.0, max_offset_m=15.0)

    assert set(merged) == {0, 1, 2}      # nothing merged - they are distinct lines
    assert merged[4] == 0                # the one flown first is now line 0
    assert merged[0] == 1
    assert merged[2] == 2


def test_merging_can_be_turned_off():
    first = _leg(300, 0.0, 0.0, 0.0, 0.5, "2026-09-02 09:00:00")
    second = _leg(300, 0.0, 160.0, 0.0, 0.5, "2026-09-02 09:20:00")
    df = _frame([first, second], source_index=[0, 1])

    assert _n_lines(detect_lines(df, LineDetectionParams(merge_continued_lines=False))) == 2
