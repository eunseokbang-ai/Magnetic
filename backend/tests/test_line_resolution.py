"""Does the line spacing resolve the field? - processing/line_resolution.py

The diagnostic exists because six processing-side explanations for
line-parallel striping were measured on real data and ruled out, leaving
the survey's own across-line sampling. So the tests build fields whose
resolvability is known by construction - a deep smooth source that any
spacing can follow, and a shallow rough one that none of these can - and
check the diagnostic tells them apart rather than reporting a number that
merely sounds plausible.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from app.processing.line_resolution import assess_line_resolution

SPACING = 60.0
LINE_LENGTH = 2400.0
STEP = 2.0


def _survey(source_depth_m, n_lines=7, spacing=SPACING, seed=5, n_sources=90):
    """Points along parallel north-south lines over dipoles at one depth.

    Depth is the whole experiment: it sets how fast the field can change
    sideways, and therefore whether lines this far apart can follow it.
    """
    rng = np.random.default_rng(seed)
    sx = rng.uniform(-200, n_lines * spacing + 200, n_sources)
    sy = rng.uniform(-200, LINE_LENGTH + 200, n_sources)
    moment = rng.normal(0, 1.0, n_sources) * source_depth_m ** 3

    xs, ys, vs, ids = [], [], [], []
    for line in range(n_lines):
        x = np.full(int(LINE_LENGTH / STEP), line * spacing)
        y = np.arange(0, LINE_LENGTH, STEP)
        r2 = (x[:, None] - sx[None, :]) ** 2 + (y[:, None] - sy[None, :]) ** 2 + source_depth_m ** 2
        v = (moment[None, :] * (2 * source_depth_m ** 2 - (r2 - source_depth_m ** 2)) / r2 ** 2.5).sum(axis=1)
        xs.append(x); ys.append(y); vs.append(v); ids.append(np.full(len(y), line))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(vs), np.concatenate(ids))


def _assess(depth, **kw):
    x, y, v, lid = _survey(depth, **kw)
    return assess_line_resolution(x, y, v, lid, azimuth_deg=0.0)


# ---------------------------------------------------------------------------
# the measurement itself
# ---------------------------------------------------------------------------


def test_a_deep_smooth_field_is_reported_as_resolved():
    """Sources far below the sensor spread their field over a scale much
    wider than the line spacing, so neighbours predict each other well."""
    result = _assess(400.0)

    assert result["available"] is True
    assert result["unresolved_pct"] < 25, result["verdict"]


def test_a_shallow_rough_field_is_reported_as_undersampled():
    """Sources close to the sensor change over a scale near the spacing,
    so each line sees something its neighbours never do."""
    result = _assess(25.0)

    assert result["unresolved_pct"] > 50, result["verdict"]
    assert "측정하지 않은" in result["verdict"]


def test_the_unresolved_share_grows_as_the_sources_get_shallower():
    shares = [_assess(d)["unresolved_pct"] for d in (400.0, 120.0, 40.0)]

    assert shares == sorted(shares), shares


def test_widening_the_spacing_on_the_same_ground_makes_it_worse():
    """The same field, sampled more coarsely, must score worse - this is
    the comparison a survey planner actually wants."""
    tight = assess_line_resolution(*_survey(60.0, n_lines=9, spacing=30.0), azimuth_deg=0.0)
    loose = assess_line_resolution(*_survey(60.0, n_lines=9, spacing=120.0), azimuth_deg=0.0)

    assert tight["unresolved_pct"] < loose["unresolved_pct"]


def test_it_reports_the_spacing_it_measured():
    result = _assess(60.0, spacing=80.0)

    assert result["line_spacing_m"] == pytest.approx(80.0, abs=2.0)


# ---------------------------------------------------------------------------
# the advice
# ---------------------------------------------------------------------------


def test_the_source_distance_is_recovered_from_the_spectrum():
    """Spector-Grant on the along-line profile, so no DEM is needed. It is
    an estimate, not a measurement - a factor-of-two band is the honest
    claim to pin."""
    result = _assess(80.0)

    assert 40.0 <= result["source_distance_m"] <= 160.0


def test_the_recommended_spacing_is_half_the_source_distance():
    result = _assess(40.0)

    assert result["recommended_spacing_m"] == pytest.approx(result["source_distance_m"] / 2.0, abs=1.0)


def test_a_comfortable_survey_is_not_told_to_fly_tighter_lines():
    result = _assess(400.0)

    assert "따라잡고 있습니다" in result["verdict"]
    assert "필요합니다" not in result["verdict"]


# ---------------------------------------------------------------------------
# declining rather than guessing
# ---------------------------------------------------------------------------


def test_two_lines_cannot_be_compared_with_a_neighbour_average():
    x, y, v, lid = _survey(60.0, n_lines=2)

    result = assess_line_resolution(x, y, v, lid, azimuth_deg=0.0)

    assert result["available"] is False and "3개 이상" in result["reason"]


def test_an_unknown_azimuth_declines():
    x, y, v, lid = _survey(60.0)

    assert assess_line_resolution(x, y, v, lid, azimuth_deg=None)["available"] is False


def test_lines_that_barely_overlap_decline_instead_of_reporting_noise():
    """Two lines sharing 100 m of ground say nothing about a 600 m
    wavelength, and a number computed from that would be worse than none."""
    x, y, v, lid = _survey(60.0)
    # stagger the lines so consecutive ones hardly share any northing
    y = y + lid * (LINE_LENGTH - 100.0)

    result = assess_line_resolution(x, y, v, lid, azimuth_deg=0.0)

    assert result["available"] is False and "짧아" in result["reason"]


def test_excluded_points_are_ignored():
    """line_id < 0 marks points the pipeline dropped; including them would
    let excluded ground drive the verdict."""
    x, y, v, lid = _survey(400.0)
    lid = lid.copy()
    lid[lid == 3] = -1

    result = assess_line_resolution(x, y, v, lid, azimuth_deg=0.0)

    assert result["available"] is True
    assert result["line_spacing_m"] == pytest.approx(SPACING, abs=SPACING * 0.6)


def test_a_diagnostic_failure_never_breaks_a_run():
    """Nonsense in, a reason out - this is reported next to a finished map
    and must not be able to take the run down with it."""
    result = assess_line_resolution(
        np.array([0.0]), np.array([0.0]), np.array([np.nan]), np.array([0]), azimuth_deg=0.0
    )

    assert result["available"] is False
