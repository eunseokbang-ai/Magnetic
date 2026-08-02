"""Synthetic validation of two flight-line detection reliability additions:
(1) bridging a brief mid-line wobble (wind, a momentary heading/GPS blip)
back into the same line instead of leaving a data gap or splitting the
line in two, and (2) tagging the takeoff->first-line and
last-line->landing transit points separately (takeoff_ramp/landing_ramp)
from ordinary in-survey turns, so the line editor can hide/protect them
independently."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.processing.lines import LineDetectionParams, _bridge_line_gaps, detect_lines

LAT0, LON0 = 37.5, 127.0
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(LAT0))


def _to_latlon(x, y):
    lat = LAT0 + np.asarray(y) / M_PER_DEG_LAT
    lon = LON0 + np.asarray(x) / M_PER_DEG_LON
    return lat, lon


def _make_df(x, y, t0="2026-01-01T00:00:00", dt_seconds=1.0):
    n = len(x)
    lat, lon = _to_latlon(x, y)
    timestamp = pd.date_range(t0, periods=n, freq=pd.Timedelta(seconds=dt_seconds))
    return pd.DataFrame({"timestamp": timestamp, "lat": lat, "lon": lon})


# ---------------------------------------------------------------------------
# _bridge_line_gaps unit tests: exercise the merge logic directly against
# hand-built line_id/reason arrays, since precisely controlling the along-
# track gap length and cross-track offset that a real detect_lines() run
# produces (through the heading-tolerance classifier) is fiddly - direct
# calls make each guard rail unambiguous to test in isolation.
# ---------------------------------------------------------------------------

def _straight_out(n=11, source_file_index=0):
    x = np.arange(n, dtype=float) * 10.0
    y = np.zeros(n)
    out = pd.DataFrame({"source_file_index": np.full(n, source_file_index)})
    return out, x, y


def test_bridge_merges_short_same_offset_gap():
    out, x, y = _straight_out(11)
    line_id = np.array([0, 0, 0, 0, -1, -1, -1, 1, 1, 1, 1])
    reason = np.array([None] * 4 + ["off_azimuth_turn"] * 3 + [None] * 4, dtype=object)
    params = LineDetectionParams(bridge_max_gap_m=100.0, bridge_max_offset_m=15.0)

    merged_id, merged_reason = _bridge_line_gaps(out, x, y, line_id, reason, dominant_azimuth_deg=0.0, params=params)

    assert (merged_id >= 0).all(), "expected the whole span to be bridged into one line"
    assert len(np.unique(merged_id)) == 1
    assert all(r is None for r in merged_reason)


def test_bridge_skips_gap_longer_than_threshold():
    out, x, y = _straight_out(11)
    line_id = np.array([0, 0, 0, 0, -1, -1, -1, 1, 1, 1, 1])
    reason = np.array([None] * 4 + ["off_azimuth_turn"] * 3 + [None] * 4, dtype=object)
    # the gap here spans 3 * 10m = 30m; with a threshold well under that,
    # it must NOT be bridged.
    params = LineDetectionParams(bridge_max_gap_m=10.0, bridge_max_offset_m=15.0)

    merged_id, merged_reason = _bridge_line_gaps(out, x, y, line_id, reason, dominant_azimuth_deg=0.0, params=params)

    assert list(merged_id) == list(line_id), "a too-long gap should be left untouched"
    assert list(merged_reason) == list(reason)


def test_bridge_skips_gap_with_large_cross_track_offset():
    out, x, y = _straight_out(11)
    y = y.copy()
    y[7:] = 50.0  # the second run sits 50m off to the side - a different line, not a wobble
    line_id = np.array([0, 0, 0, 0, -1, -1, -1, 1, 1, 1, 1])
    reason = np.array([None] * 4 + ["off_azimuth_turn"] * 3 + [None] * 4, dtype=object)
    params = LineDetectionParams(bridge_max_gap_m=100.0, bridge_max_offset_m=15.0)

    merged_id, merged_reason = _bridge_line_gaps(out, x, y, line_id, reason, dominant_azimuth_deg=0.0, params=params)

    assert list(merged_id) == list(line_id), "a real turn onto an adjacent line should not be bridged"
    assert list(merged_reason) == list(reason)


def test_bridge_chains_multiple_wobbles_into_one_line():
    # three runs, two short bridgeable gaps in between - all three should
    # end up merged into a single line id.
    n = 17
    x = np.arange(n, dtype=float) * 10.0
    y = np.zeros(n)
    out = pd.DataFrame({"source_file_index": np.zeros(n, dtype=int)})
    line_id = np.array([0] * 4 + [-1] * 2 + [1] * 4 + [-1] * 2 + [2] * 5)
    reason = np.array(
        [None] * 4 + ["off_azimuth_turn"] * 2 + [None] * 4 + ["off_azimuth_turn"] * 2 + [None] * 5, dtype=object
    )
    params = LineDetectionParams(bridge_max_gap_m=100.0, bridge_max_offset_m=15.0)

    merged_id, _ = _bridge_line_gaps(out, x, y, line_id, reason, dominant_azimuth_deg=0.0, params=params)

    assert (merged_id >= 0).all()
    assert len(np.unique(merged_id)) == 1


def test_bridge_does_not_cross_source_file_boundary():
    # two separate uploaded files placed back-to-back in time/space should
    # never be bridged into one line, even if geometrically adjacent.
    n = 8
    x = np.arange(n, dtype=float) * 10.0
    y = np.zeros(n)
    out = pd.DataFrame({"source_file_index": [0, 0, 0, 0, 1, 1, 1, 1]})
    line_id = np.array([0, 0, 0, -1, -1, 1, 1, 1])
    reason = np.array([None, None, None, "off_azimuth_turn", "off_azimuth_turn", None, None, None], dtype=object)
    params = LineDetectionParams(bridge_max_gap_m=100.0, bridge_max_offset_m=15.0)

    merged_id, _ = _bridge_line_gaps(out, x, y, line_id, reason, dominant_azimuth_deg=0.0, params=params)

    assert list(merged_id) == list(line_id)


# ---------------------------------------------------------------------------
# End-to-end check through detect_lines(): a real, brief sideways wobble
# (wind gust) confirmed to get excluded by the heading-tolerance
# classifier still ends up as a single bridged line with no data gap.
# ---------------------------------------------------------------------------

def test_detect_lines_bridges_brief_wobble_end_to_end():
    x_a = np.arange(0.0, 255.0, 5.0)
    y_a = np.zeros_like(x_a)
    x_wobble = np.arange(255.0, 300.0, 5.0)
    y_wobble = np.array([4.0, 8.0, 12.0, 16.0, 12.0, 8.0, 4.0, 2.0, 0.0])[: len(x_wobble)]
    x_b = np.arange(300.0, 550.0, 5.0)
    y_b = np.zeros_like(x_b)
    x = np.concatenate([x_a, x_wobble, x_b])
    y = np.concatenate([y_a, y_wobble, y_b])
    df = _make_df(x, y)

    params = LineDetectionParams(min_line_length_m=50.0, turn_buffer_m=10.0)
    detected = detect_lines(df, params)

    kept = detected[detected["line_id"] >= 0]
    assert kept["line_id"].nunique() == 1, (
        f"expected the wobble to be bridged into a single line, got "
        f"{kept['line_id'].nunique()} lines: {sorted(detected['line_id'].unique())}"
    )
    kept_idx = kept.index.to_numpy()
    interior = detected.loc[kept_idx.min():kept_idx.max()]
    assert (interior["line_id"] >= 0).all(), "expected no interior data gap after bridging"


def test_detect_lines_bridge_disabled_leaves_a_gap():
    x_a = np.arange(0.0, 255.0, 5.0)
    y_a = np.zeros_like(x_a)
    x_wobble = np.arange(255.0, 300.0, 5.0)
    y_wobble = np.array([4.0, 8.0, 12.0, 16.0, 12.0, 8.0, 4.0, 2.0, 0.0])[: len(x_wobble)]
    x_b = np.arange(300.0, 550.0, 5.0)
    y_b = np.zeros_like(x_b)
    x = np.concatenate([x_a, x_wobble, x_b])
    y = np.concatenate([y_a, y_wobble, y_b])
    df = _make_df(x, y)

    params = LineDetectionParams(min_line_length_m=50.0, turn_buffer_m=10.0, bridge_gaps=False)
    detected = detect_lines(df, params)
    kept = detected[detected["line_id"] >= 0]
    assert kept["line_id"].nunique() == 2, "expected the wobble to split the line when bridging is disabled"


# ---------------------------------------------------------------------------
# takeoff_ramp / landing_ramp tagging
# ---------------------------------------------------------------------------

def test_tags_takeoff_and_landing_ramps_separately_from_turns():
    # Ground/climb-out (slow, below min_speed) -> transit to the first
    # line -> line 1 -> a real turn -> line 2 -> transit -> landing.
    rng_pre = np.linspace(0.0, 20.0, 10)  # slow climb-out, low speed
    x_pre, y_pre = rng_pre, np.zeros_like(rng_pre)

    x_l1 = np.arange(20.0, 270.0, 5.0)
    y_l1 = np.zeros_like(x_l1)

    # a real turn: swings far off-azimuth and moves to an adjacent line
    x_turn = np.arange(270.0, 320.0, 5.0)
    y_turn = np.linspace(0.0, 50.0, len(x_turn))

    x_l2 = np.arange(320.0, 570.0, 5.0)
    y_l2 = np.full_like(x_l2, 50.0)

    rng_post = np.linspace(570.0, 590.0, 10)  # slow landing approach
    x_post, y_post = rng_post, np.full_like(rng_post, 50.0)

    x = np.concatenate([x_pre, x_l1, x_turn, x_l2, x_post])
    y = np.concatenate([y_pre, y_l1, y_turn, y_l2, y_post])
    df = _make_df(x, y, dt_seconds=1.0)

    params = LineDetectionParams(min_line_length_m=50.0, turn_buffer_m=10.0, min_speed_mps=1.5)
    detected = detect_lines(df, params)

    kept = detected[detected["line_id"] >= 0]
    assert kept["line_id"].nunique() == 2, "expected two distinct lines separated by a real turn"

    first_kept, last_kept = kept.index.min(), kept.index.max()
    pre = detected.loc[: first_kept - 1]
    post = detected.loc[last_kept + 1:]
    interior_excluded = detected.loc[first_kept:last_kept]
    interior_excluded = interior_excluded[interior_excluded["line_id"] < 0]

    assert (pre["exclusion_reason"] == "takeoff_ramp").all(), pre["exclusion_reason"].tolist()
    assert (post["exclusion_reason"] == "landing_ramp").all(), post["exclusion_reason"].tolist()
    # the real interior turn must NOT be mislabeled as a takeoff/landing ramp
    assert not interior_excluded["exclusion_reason"].isin(["takeoff_ramp", "landing_ramp"]).any(), (
        interior_excluded["exclusion_reason"].tolist()
    )
    assert interior_excluded["exclusion_reason"].notna().all()


if __name__ == "__main__":
    test_bridge_merges_short_same_offset_gap()
    test_bridge_skips_gap_longer_than_threshold()
    test_bridge_skips_gap_with_large_cross_track_offset()
    test_bridge_chains_multiple_wobbles_into_one_line()
    test_bridge_does_not_cross_source_file_boundary()
    test_detect_lines_bridges_brief_wobble_end_to_end()
    test_detect_lines_bridge_disabled_leaves_a_gap()
    test_tags_takeoff_and_landing_ramps_separately_from_turns()
    print("ALL CHECKS PASSED")
