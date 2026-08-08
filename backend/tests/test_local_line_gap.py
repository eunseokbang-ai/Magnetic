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

from app.processing.gridding import _local_line_gap_m, _local_line_gap_m_exact, grid_points


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


def _two_line_asymmetric_survey():
    # Line A: full length along y=0, x in [0, 50]. Line B: much shorter,
    # along y=50, x in [0, 20] only - so the *rectangular bounding box*
    # (x:[0,50], y:[0,50]) has an empty corner near (50, 50) that lies well
    # outside the *actual* convex hull of the survey
    # ((0,0)-(50,0)-(20,50)-(0,50)). This is exactly the shape that
    # triggered the unbounded triangular-fan extrapolation bug: near that
    # corner, the two nearest distinct lines are both far away and roughly
    # comparably distant, so local_gap grows with distance from the survey
    # instead of staying bounded.
    xa = np.arange(0, 51, 1.0)
    xb = np.arange(0, 21, 1.0)
    x = np.concatenate([xa, xb])
    y = np.concatenate([np.zeros_like(xa), np.full_like(xb, 50.0)])
    line_id = np.concatenate([np.zeros(len(xa)), np.ones(len(xb))])
    values = np.full(len(x), 100.0)
    return x, y, values, line_id


def test_hull_cap_prevents_unbounded_fan_extrapolation_past_line_endpoints():
    x, y, values, line_id = _two_line_asymmetric_survey()
    result = grid_points(x, y, values, 1.0, method="nearest", max_distance_m=None, line_id=line_id)

    def value_at(target_x, target_y):
        col = int(np.argmin(np.abs(result.easting - target_x)))
        row = int(np.argmin(np.abs(result.northing - target_y)))
        return result.values[row, col]

    # The empty corner near (50, 50) sits inside the rectangular bounding
    # box but well outside the true survey footprint - must stay NaN no
    # matter what the local-gap heuristic alone computes there.
    assert np.isnan(value_at(48.0, 48.0))

    # A point that legitimately sits inside the surveyed footprint (between
    # the two lines, over x where both lines exist) must still be filled -
    # the hull cap must not regress ordinary interior gap-filling.
    assert not np.isnan(value_at(10.0, 25.0))


def test_local_line_gap_downsampled_path_tracks_the_exact_computation():
    x, y, values, line_id = _three_line_survey()
    xs = np.linspace(0, 49, 90)  # 90x90 = 8100 cells, above the downsample threshold
    grid_x, grid_y = np.meshgrid(xs, xs)

    exact = _local_line_gap_m_exact(grid_x, grid_y, x, y, line_id, np.unique(line_id))
    approx = _local_line_gap_m(grid_x, grid_y, x, y, line_id)

    assert approx is not None
    assert approx.shape == exact.shape
    # local_gap is a smooth, spacing-scale quantity - the coarse-grid +
    # upsample approximation should track it closely, not exactly.
    assert np.nanmean(np.abs(approx - exact)) < 2.0


def _irregular_multi_line_survey(seed=2, n_lines=10, line_spacing=50.0, line_length=1200.0, along_spacing=3.0):
    """A more realistic multi-line survey than _three_line_survey above:
    each line's start/end trimmed by a random amount (as real lines rarely
    all start/end at exactly the same along-track position) and given a
    gentle bow/heading-jitter, rather than perfectly straight and equal-
    length. Used to reproduce a real "그리드 간격을 15m 했을 때는 nodata가
    없는데 10m 하면 중간중간 생긴다" report: the interior-fill threshold
    only needs to reach a bit past the ideal straight-line half-gap to
    cover this kind of everyday irregularity, and a fine grid samples the
    threshold at far more discrete points across the same interior area
    than a coarse one does - so a margin that's fine on paper for the
    idealized case can still leave a few isolated interior cells just
    outside reach only once the grid is fine enough to happen to land a
    cell there.
    """
    rng = np.random.default_rng(seed)
    xs, ys, vals, lids = [], [], [], []
    for i in range(n_lines):
        y0 = i * line_spacing
        start = rng.uniform(0, 60)
        end = line_length - rng.uniform(0, 60)
        n_pts = int((end - start) / along_spacing)
        x = np.linspace(start, end, n_pts)
        y = y0 + 3.0 * np.sin(x / 250.0 + i * 0.7) + rng.normal(0, 0.3, n_pts)
        v = 100 + 5 * np.sin(x / 100.0) + rng.normal(0, 2, n_pts)
        xs.append(x)
        ys.append(y)
        vals.append(v)
        lids.append(np.full(n_pts, i))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(vals), np.concatenate(lids)


def test_fine_cell_size_has_no_more_nodata_than_a_coarse_one():
    """Regression test for a real user report: gridding a 50m-line-spacing
    survey at 15m cell size had no nodata gaps, but reducing the cell size
    to 10m (still coarser than the recommended 1/4-1/5 of line spacing)
    introduced a handful of scattered nodata cells "중간중간" (here and
    there). The interior-fill reach must not get *effectively* stingier as
    the cell size shrinks - a finer grid should only add resolution, never
    punch new holes in coverage a coarser grid of the same data didn't
    have. On a somewhat messy synthetic survey (bowed lines, staggered
    start/end points - not perfectly straight/equal-length), the previous
    0.6 coefficient left a handful of cells unfilled at some cell sizes but
    not others; checked all the way down to line_spacing/10 (5m here), the
    finest ratio explicitly requested for detailed work."""
    x, y, v, line_id = _irregular_multi_line_survey()
    line_spacing = 50.0

    nan_counts = {}
    for cell in (15.0, 10.0, 5.0):
        result = grid_points(
            x, y, v, cell, method="nearest", max_distance_m=None, line_id=line_id, typical_line_spacing_m=line_spacing
        )
        nan_counts[cell] = int(np.isnan(result.values).sum())

    assert all(n == 0 for n in nan_counts.values()), f"nodata cells remain: {nan_counts}"


def test_typical_line_spacing_widens_hull_buffer_but_stays_bounded():
    """typical_line_spacing_m (store.py's self.line_spacing_m - a single
    robust, project-wide statistic) widens the hull cap's buffer beyond
    the tiny flat _HULL_BUFFER_CELLS default, so a real multi-line survey
    with any bowing/curvature in its flight path (a slightly concave
    footprint) doesn't get real interior gap-fill area near the bends
    clipped back out - see grid_points' docstring. It must still stay
    *bounded*: a modest, realistic spacing value must not resurrect the
    unbounded far-outside-hull extrapolation bug _two_line_asymmetric_
    survey regression-tests below."""
    x, y, values, line_id = _two_line_asymmetric_survey()

    def value_at(result, target_x, target_y):
        col = int(np.argmin(np.abs(result.easting - target_x)))
        row = int(np.argmin(np.abs(result.northing - target_y)))
        return result.values[row, col]

    result_default = grid_points(x, y, values, 1.0, method="nearest", max_distance_m=None, line_id=line_id)
    result_spaced = grid_points(
        x, y, values, 1.0, method="nearest", max_distance_m=None, line_id=line_id, typical_line_spacing_m=15.0
    )
    # a realistic line-spacing value still keeps the pathological far
    # corner excluded...
    assert np.isnan(value_at(result_default, 48.0, 48.0))
    assert np.isnan(value_at(result_spaced, 48.0, 48.0))
    # ...while genuinely widening the fill area closer to the hull edge
    # compared to the tiny flat default (a point just past the strict
    # hull - too far for the flat 2-cell/2m default but within 1.2*15m=18m).
    assert np.isnan(value_at(result_default, 25.0, 47.0))
    assert not np.isnan(value_at(result_spaced, 25.0, 47.0))
