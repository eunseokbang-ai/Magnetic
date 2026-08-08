"""Tests for the locally-adaptive gap-fill fix in grid_points: a single
project-wide average line spacing understates the true gap wherever lines
are bowed/skewed or a stretch was flown with locally wider spacing than
the survey average, leaving real gaps in the middle of an otherwise
well-covered block even at a fine cell size. _local_line_gap_m computes
the true local gap (distance to nearest point on each of the two closest
distinct flight lines) at every grid cell instead.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from app.processing.gridding import _local_line_gap_m, grid_points


def _three_line_survey():
    # Lines at y=0, y=10 (10m gap, "normal") and y=40 (30m gap from the
    # middle line, "locally widened" - e.g. a bowed/skewed pass or a day
    # flown farther apart than usual).
    xs = np.arange(0, 50, 1.0)
    x, y, line_id = [], [], []
    for lid, ypos in enumerate([0.0, 10.0, 40.0]):
        x.append(xs)
        y.append(np.full_like(xs, ypos))
        line_id.append(np.full(len(xs), lid))
    x = np.concatenate(x)
    y = np.concatenate(y)
    line_id = np.concatenate(line_id)
    values = np.full(len(x), 100.0)  # flat field - only coverage/masking is under test
    return x, y, values, line_id


def test_local_line_gap_reflects_true_local_spacing_not_a_single_average():
    x, y, values, line_id = _three_line_survey()
    grid_x, grid_y = np.meshgrid(np.arange(0, 50, 1.0), np.array([5.0, 25.0]))
    gap = _local_line_gap_m(grid_x, grid_y, x, y, line_id)
    # y=5 sits between the two closely-spaced lines (0 and 10) -> local gap ~10m
    assert np.allclose(gap[0], 10.0, atol=0.6)
    # y=25 sits between the widely-spaced lines (10 and 40) -> local gap ~30m,
    # not some single blended "average spacing" figure
    assert np.allclose(gap[1], 30.0, atol=0.6)


def test_auto_max_distance_fills_the_widened_gap_a_flat_global_threshold_would_mask():
    x, y, values, line_id = _three_line_survey()
    cell_size = 1.0

    result_auto = grid_points(x, y, values, cell_size, method="nearest", max_distance_m=None, line_id=line_id)
    # A naive single global threshold based on the *average* of the two
    # gaps (10 and 40) would be 0.6 * 20 = 12m - too small to reach the
    # midpoint (15m from each line) of the widened gap.
    naive_global_threshold = 0.6 * 20.0
    result_flat = grid_points(x, y, values, cell_size, method="nearest", max_distance_m=naive_global_threshold, line_id=line_id)

    def value_at(result, target_x, target_y):
        col = int(np.argmin(np.abs(result.easting - target_x)))
        row = int(np.argmin(np.abs(result.northing - target_y)))
        return result.values[row, col]

    # Midpoint of the widened gap (10m from each flanking line at y=10/y=40)
    mid_wide = value_at(result_flat, 25.0, 25.0)
    assert np.isnan(mid_wide), "sanity check: the naive flat global threshold should leave this real gap masked"

    mid_wide_auto = value_at(result_auto, 25.0, 25.0)
    assert not np.isnan(mid_wide_auto), "locally-adaptive auto masking should fill the same point the flat threshold missed"

    # Midpoint of the normal (10m) gap must still be filled in both cases -
    # the fix must not start over-filling regions that were already fine.
    mid_normal_auto = value_at(result_auto, 25.0, 5.0)
    mid_normal_flat = value_at(result_flat, 25.0, 5.0)
    assert not np.isnan(mid_normal_auto)
    assert not np.isnan(mid_normal_flat)


def test_local_line_gap_returns_none_for_single_line():
    xs = np.arange(0, 50, 1.0)
    line_id = np.zeros(len(xs))
    grid_x, grid_y = np.meshgrid(xs, np.array([0.0]))
    gap = _local_line_gap_m(grid_x, grid_y, xs, np.zeros(len(xs)), line_id)
    assert gap is None


def test_grid_points_single_line_auto_mode_falls_back_without_crashing():
    xs = np.arange(0, 50, 1.0)
    ys = np.zeros(len(xs))
    values = np.full(len(xs), 50.0)
    line_id = np.zeros(len(xs))
    result = grid_points(xs, ys, values, 1.0, method="nearest", max_distance_m=None, line_id=line_id)
    # falls back to the flat 2*cell_size_m behavior - should still produce
    # a narrow filled band along the single line, not crash or mask everything
    assert np.isfinite(result.values).any()
