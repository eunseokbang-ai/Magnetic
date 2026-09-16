"""Removing a ground structure by modelling it, rather than cutting a hole.

The failure this replaces, seen on the 2026-09 HaeNam survey around a
984 nT structure: a region filled by interpolating along each flight line
came out as a trough with walls parallel to the lines, and the analytic
signal drew it as a bright ring round a hollow centre - the lobe the
region did not reach was still there, and each line had been filled at
its own level.

The fixture is that situation in miniature: north-south lines 50 m apart,
a compact source with its own (remanent) magnetisation direction, and
broad geology underneath that must come through untouched.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from app.processing.source_removal import SourceRemoval, apply_source_removals, fit_source_removal

INC, DEC = 50.94, -8.10
SRC = (0.0, 0.0, -40.0)            # 40 m below the survey plane


def _survey(spacing=50.0, half=600.0, step=2.0):
    xs = np.arange(-half, half + 1, spacing)
    ys = np.arange(-half, half + 1, step)
    x = np.repeat(xs, len(ys))
    y = np.tile(ys, len(xs))
    line = np.repeat(np.arange(len(xs)), len(ys))
    return x, y, line


def _dipole(x, y, m=(3e5, -2e5, -6e5)):
    inc, dec = np.radians(INC), np.radians(DEC)
    f = np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)])
    rx, ry, rz = x - SRC[0], y - SRC[1], 0.0 - SRC[2]
    r2 = rx**2 + ry**2 + rz**2
    r = np.sqrt(r2)
    mdotr = m[0] * rx + m[1] * ry + m[2] * rz
    b = [(3 * mdotr * c / r2 - mc) / r**3 for c, mc in zip((rx, ry, rz), m)]
    return 100.0 * (f[0] * b[0] + f[1] * b[1] + f[2] * b[2])


def _geology(x, y):
    return 0.03 * x - 0.02 * y + 25.0 * np.sin(2 * np.pi * x / 1400.0) * np.cos(2 * np.pi * y / 1700.0)


def _circle(r, n=32):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return list(zip(r * np.cos(t), r * np.sin(t)))


@pytest.fixture(scope="module")
def case():
    x, y, line = _survey()
    structure = _dipole(x, y)
    geology = _geology(x, y)
    rng = np.random.default_rng(0)
    data = geology + structure + rng.normal(0, 0.3, len(x))
    removal = fit_source_removal(x, y, data, line, _circle(90.0), INC, DEC, 50.0, depth_hint_m=40.0)
    cleaned = apply_source_removals(x, y, data, [removal])
    return dict(x=x, y=y, line=line, structure=structure, geology=geology, data=data,
                removal=removal, cleaned=cleaned)


def test_the_fixture_is_the_hard_case(case):
    """A strong structure whose field reaches well past the marked region."""
    r = np.hypot(case["x"], case["y"])
    assert np.abs(case["structure"]).max() > 500
    assert np.abs(case["structure"][r > 90]).max() > 50


def test_the_structure_is_removed_inside_the_region(case):
    r = np.hypot(case["x"], case["y"])
    inside = r <= 90
    residual = case["cleaned"][inside] - case["geology"][inside]
    assert np.abs(case["structure"][inside]).max() > 500
    assert np.sqrt(np.mean(residual**2)) < 0.05 * np.sqrt(np.mean(case["structure"][inside] ** 2))


def test_the_tails_and_lobe_outside_the_region_go_too(case):
    """What a cut-and-fill cannot do: the field beyond the region's edge,
    which a cut leaves exactly as it was. Measured: 18.5 nT rms in this
    ring before, 1.96 nT after."""
    r = np.hypot(case["x"], case["y"])
    ring = (r > 90) & (r <= 200)
    before = case["data"][ring] - case["geology"][ring]
    after = case["cleaned"][ring] - case["geology"][ring]
    assert np.sqrt(np.mean(after**2)) < 0.12 * np.sqrt(np.mean(before**2))


def test_the_geology_under_and_around_it_is_left_alone(case):
    """Far out, what is subtracted is the structure's own faint tail and
    nothing else - the cleaned data sits on the geology."""
    r = np.hypot(case["x"], case["y"])
    far = r > 350
    assert np.sqrt(np.mean((case["cleaned"][far] - case["geology"][far]) ** 2)) < 0.5
    # and nothing is flattened: the geology's own gradient still reads
    # through the region
    inside = r <= 90
    got = np.polyfit(case["x"][inside], case["cleaned"][inside], 1)[0]
    want = np.polyfit(case["x"][inside], case["geology"][inside], 1)[0]
    assert abs(got - want) < 0.3 * abs(want)


def test_no_wall_between_lines_where_the_region_ends(case):
    """The line-by-line fill left neighbouring lines at different levels.
    Here the level change from one line to the next across the region's
    edge must look like the geology's, not like a step."""
    x, y, cleaned, geology = case["x"], case["y"], case["cleaned"], case["geology"]
    band = np.abs(y) < 20
    jumps = []
    for xa, xb in ((50.0, 100.0), (100.0, 150.0), (-100.0, -50.0), (-150.0, -100.0)):
        a = band & (x == xa)
        b = band & (x == xb)
        jumps.append(abs((cleaned[b] - geology[b]).mean() - (cleaned[a] - geology[a]).mean()))
    assert max(jumps) < 5.0


def test_the_fit_reports_what_it_did(case):
    s = case["removal"].summary()
    assert s["n_sources"] > 0
    assert s["fit_rms_nt"] < 0.05 * s["data_rms_nt"]
    assert s["peak_model_nt"] > 500
    assert s["reach_m"] > 90
    assert s["warnings"] == []


def test_a_region_that_misses_the_source_says_so():
    x, y, line = _survey()
    data = _geology(x, y) + _dipole(x, y)
    off = [(px + 400.0, py + 400.0) for px, py in _circle(60.0)]
    removal = fit_source_removal(x, y, data, line, off, INC, DEC, 50.0)
    assert removal.warnings


def test_a_removal_survives_a_round_trip_exactly(case):
    again = SourceRemoval.from_dict(case["removal"].to_dict())
    xs, ys = case["x"][::97], case["y"][::97]
    assert np.allclose(again.field_at(xs, ys), case["removal"].field_at(xs, ys))


def test_too_few_points_is_an_error_not_a_guess():
    x = np.array([0.0, 10.0, 20.0])
    with pytest.raises(ValueError):
        fit_source_removal(x, x, x, np.zeros(3, int), _circle(50.0), INC, DEC, 50.0)
