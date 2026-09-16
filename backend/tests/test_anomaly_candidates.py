"""Finding the anomalies worth looking at before deciding what to cut.

The fixtures here are induced dipoles at the HaeNam block's inclination
(50.94 deg), which is the shape a compact ground structure actually makes:
a positive peak with a negative lobe of about -31% of it, one source-depth
to the north. Most of what this module has to get right follows from that
shape - not splitting one object into two candidates, and cutting a region
wide enough that the negative lobe goes with the peak.
"""
from __future__ import annotations

import numpy as np

from app.processing.anomaly_candidates import find_anomaly_candidates
from app.processing.transforms import analytic_signal

INCLINATION_DEG = 50.94
DECLINATION_DEG = -8.10


def _dipole_field(easting, northing, sources, depth_m):
    """Total-field anomaly of induced dipoles at `sources` [(x, y, amplitude)]."""
    inc, dec = np.radians(INCLINATION_DEG), np.radians(DECLINATION_DEG)
    f = np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)])
    E, N = np.meshgrid(easting, northing)
    total = np.zeros_like(E, dtype=float)
    for sx, sy, amplitude in sources:
        dx, dy = E - sx, N - sy
        r = np.sqrt(dx**2 + dy**2 + depth_m**2)
        rhat = np.stack([dx / r, dy / r, -depth_m / r])
        fdotr = f[0] * rhat[0] + f[1] * rhat[1] + f[2] * rhat[2]
        t = (3 * fdotr**2 - 1) / r**3
        total += t / np.abs(t).max() * amplitude
    return total


def _grid(sources, depth_m=10.0, cell=2.0, span=400.0, noise_nt=0.0, seed=0):
    axis = np.arange(-span / 2, span / 2 + cell, cell)
    anomaly = _dipole_field(axis, axis, sources, depth_m)
    if noise_nt:
        anomaly = anomaly + np.random.default_rng(seed).normal(0.0, noise_nt, anomaly.shape)
    return anomaly, axis, axis, cell


def _candidates(anomaly, easting, northing, cell, **kw):
    return find_anomaly_candidates(
        analytic_signal(anomaly, cell), anomaly, easting, northing, cell, **kw
    )


def test_one_object_produces_one_candidate_not_two():
    """The whole reason candidates are ranked on the analytic signal: on
    the anomaly itself this source has a +100 nT peak AND a -31 nT lobe,
    and ranking on that would spend two of the ten slots on one object."""
    anomaly, e, n, cell = _grid([(0.0, 0.0, 100.0)])

    found = _candidates(anomaly, e, n, cell, n_candidates=10)

    assert len(found) == 1
    assert abs(found[0].x) <= 3 * cell and abs(found[0].y) <= 3 * cell


def test_the_suggested_region_takes_the_negative_lobe_with_the_peak():
    """Cutting the peak alone would leave a crater at -31% of it - which,
    on the surveys this feature is for, is still bigger than the geology."""
    depth = 10.0
    anomaly, e, n, cell = _grid([(0.0, 0.0, 100.0)], depth_m=depth)
    c = _candidates(anomaly, e, n, cell)[0]

    E, N = np.meshgrid(e, n)
    from matplotlib.path import Path as MplPath

    inside = MplPath(c.polygon_xy).contains_points(
        np.column_stack([E.ravel(), N.ravel()])
    ).reshape(anomaly.shape)

    # everything the source still moves by more than a tenth of its peak
    significant = np.abs(anomaly) > 0.10 * np.abs(anomaly).max()
    assert significant.sum() > 0
    assert inside[significant].all()
    # the lobe is the part that would otherwise be left behind
    assert inside[np.unravel_index(np.argmin(anomaly), anomaly.shape)]


def test_the_half_width_recovers_the_source_depth():
    """Measured ratio: |T| falls to half its peak at exactly one source
    depth, independent of the depth itself."""
    for depth in (5.0, 10.0, 20.0):
        anomaly, e, n, cell = _grid([(0.0, 0.0, 100.0)], depth_m=depth, cell=depth / 10.0)
        c = _candidates(anomaly, e, n, cell)[0]
        assert abs(c.depth_m - depth) < 0.25 * depth, f"depth {depth}: got {c.depth_m}"


def test_candidates_come_back_strongest_first():
    anomaly, e, n, cell = _grid(
        [(-120.0, -120.0, 40.0), (0.0, 0.0, 100.0), (120.0, 120.0, 70.0)]
    )

    found = _candidates(anomaly, e, n, cell, n_candidates=10)

    assert len(found) == 3
    peaks = [abs(c.peak_anomaly_nt) for c in found]
    assert peaks == sorted(peaks, reverse=True)
    assert [c.rank for c in found] == [1, 2, 3]


def test_ten_scattered_objects_come_back_as_ten_separate_candidates():
    rng = np.random.default_rng(3)
    positions = [(x, y) for x in (-150, -50, 50, 150) for y in (-150, -50, 50, 150)]
    sources = [(x, y, float(rng.uniform(40, 120))) for x, y in positions[:10]]
    anomaly, e, n, cell = _grid(sources, span=500.0)

    found = _candidates(anomaly, e, n, cell, n_candidates=10)

    assert len(found) == 10
    centres = np.array([(c.x, c.y) for c in found])
    separations = np.hypot(
        centres[:, None, 0] - centres[None, :, 0], centres[:, None, 1] - centres[None, :, 1]
    )
    np.fill_diagonal(separations, np.inf)
    assert separations.min() > 50.0


def test_a_block_with_no_compact_source_offers_nothing_to_cut():
    """Ordinary data must not be handed to the operator as ten
    "candidates" to punch holes in.

    The background here is broad, smooth wander - a regional field, not
    white noise. That is what a gridded survey with no cultural sources in
    it actually looks like: the app low-passes along the lines and caps
    the grid at its across-line resolution before any of this runs, so
    structure finer than the line spacing cannot survive into the input.
    """
    from scipy import ndimage as ndi

    rng = np.random.default_rng(7)
    axis = np.arange(-200.0, 202.0, 2.0)
    background = ndi.gaussian_filter(rng.normal(0.0, 1.0, (len(axis), len(axis))), 30.0)
    background *= 20.0 / background.std()   # 20 nT of regional swell, no compact source

    found = _candidates(background, axis, axis, 2.0, n_candidates=10)

    # Whatever survives here is the analytic signal's own edge effect
    # where the transform had to invent the field beyond the block, and
    # comes back flagged as such rather than as an object to cut.
    assert [c for c in found if not c.at_coverage_edge] == []


def test_it_reports_how_many_lines_actually_crossed_the_anomaly():
    """A feature only one line saw cannot be told from a levelling stripe
    on the grid, so the count is reported rather than the candidate being
    silently offered."""
    anomaly, e, n, cell = _grid([(0.0, 0.0, 100.0)])
    # flight lines 50 m apart, running north, sampled every 0.6 m
    line_x = np.arange(-200.0, 201.0, 50.0)
    along = np.arange(-200.0, 200.0, 0.6)
    px = np.repeat(line_x, len(along))
    py = np.tile(along, len(line_x))
    lid = np.repeat(np.arange(len(line_x)), len(along))

    c = _candidates(anomaly, e, n, cell, point_x=px, point_y=py, point_line_id=lid)[0]

    assert c.n_points > 0
    assert c.n_lines >= 1
    assert c.single_line == (c.n_lines <= 1)


def test_a_candidate_against_the_edge_of_coverage_is_flagged():
    """Grid edges are where interpolation is least supported, so a peak
    there is worth a second look before it is cut."""
    anomaly, e, n, cell = _grid([(0.0, 0.0, 100.0)], span=200.0)
    anomaly[:, : len(e) // 2] = np.nan     # half the block has no coverage

    found = _candidates(anomaly, e, n, cell)

    assert found and found[0].at_coverage_edge


def test_the_region_can_be_capped_so_a_broad_high_is_not_swallowed():
    anomaly, e, n, cell = _grid([(0.0, 0.0, 100.0)], depth_m=40.0, span=600.0, cell=4.0)

    tight = _candidates(anomaly, e, n, cell, max_radius_m=20.0)[0]
    loose = _candidates(anomaly, e, n, cell, max_radius_m=200.0)[0]

    assert tight.radius_m < loose.radius_m


def test_the_depth_is_the_same_whichever_field_the_ranking_used():
    """Ranked on the residual anomaly the top peak is a lobe, a source-depth
    away from the source; ranked on the analytic signal it is over the
    source. The depth must not change with that choice - it is measured at
    the analytic signal's peak either way."""
    anomaly, e, n, cell = _grid([(0.0, 0.0, 100.0)], depth_m=15.0, cell=1.5)

    on_signal = find_anomaly_candidates(
        analytic_signal(anomaly, cell), anomaly, e, n, cell
    )[0]
    on_anomaly = find_anomaly_candidates(anomaly, anomaly, e, n, cell)[0]

    assert abs(on_signal.depth_m - on_anomaly.depth_m) < 0.15 * on_signal.depth_m
    assert abs(on_anomaly.depth_m - 15.0) < 0.25 * 15.0


def test_an_anomaly_too_wide_for_the_cap_says_so():
    """The cut then stops short of the 10% contour, and a feature that
    broad is worth asking about before removing it at all."""
    anomaly, e, n, cell = _grid([(0.0, 0.0, 100.0)], depth_m=40.0, span=600.0, cell=4.0)

    assert _candidates(anomaly, e, n, cell, max_radius_m=30.0)[0].region_capped
    assert not _candidates(anomaly, e, n, cell, max_radius_m=200.0)[0].region_capped


def test_a_source_the_grid_cannot_resolve_is_flagged_not_guessed():
    """A source 3 m below a grid with 8 m cells is finer than the grid can
    show, so the depth reported is the grid's own floor - an upper bound on
    how shallow the source is, not a measurement. Which matters here: a
    depth near the flight clearance is the main evidence that a candidate
    is an object on the ground rather than geology."""
    anomaly, e, n, cell = _grid([(0.0, 0.0, 100.0)], depth_m=3.0, cell=8.0, span=480.0)

    found = _candidates(anomaly, e, n, cell)

    assert found and not found[0].depth_resolved
    assert found[0].depth_m > 3.0        # the floor, and it overstates the depth


def test_the_reported_radius_is_what_the_cut_region_actually_reaches():
    """It used to report the uncapped figure: 155 m for a region clipped
    to 60 m."""
    anomaly, e, n, cell = _grid([(0.0, 0.0, 100.0)], depth_m=40.0, span=600.0, cell=4.0)

    c = _candidates(anomaly, e, n, cell, max_radius_m=30.0)[0]

    reach = max(np.hypot(px - c.x, py - c.y) for px, py in c.polygon_xy)
    assert c.radius_m <= 30.0 + 1e-6
    assert reach <= 30.0 + 2 * cell


def test_a_candidate_offers_a_source_region_not_capped_like_the_cut():
    """Removing by model needs room for the whole source; the cut's cap is
    about how far a hole may reach, and would squeeze a broad source."""
    anomaly, e, n, cell = _grid([(0.0, 0.0, 100.0)], depth_m=40.0, span=600.0, cell=4.0)

    c = _candidates(anomaly, e, n, cell, max_radius_m=30.0)[0]

    cut = max(np.hypot(px - c.x, py - c.y) for px, py in c.polygon_xy)
    src = max(np.hypot(px - c.x, py - c.y) for px, py in c.source_polygon_xy)
    assert src > cut
    assert src <= 300.0 + 2 * cell
