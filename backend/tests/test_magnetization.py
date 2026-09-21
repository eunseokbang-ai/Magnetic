"""Telling a steel object's own magnetization from the induced kind.

The fixture is the survey this is for: north-south lines 50 m apart,
readings every 2 m along them, geology underneath, noise on top, and one
compact source 50 m below the sensor - the depth a structure on the
ground sits at when the drone flies 50 m above it.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from app.processing.magnetization import estimate_magnetization
from app.processing.source_removal import _field_direction, _kernel

INC, DEC = 50.94, -8.10
DEPTH = 50.0


def _survey(spacing=50.0, half=400.0, step=2.0):
    xs = np.arange(-half, half + 1, spacing)
    ys = np.arange(-half, half + 1, step)
    return np.repeat(xs, len(ys)), np.tile(ys, len(xs))


def _rotate(f, inc_deg, dec_deg):
    return _field_direction(inc_deg, dec_deg)


def _case(moment_dir, strength=3e5, noise_nt=1.0, depth=DEPTH, seed=0):
    x, y = _survey()
    f = _field_direction(INC, DEC)
    source = np.array([[0.0, 0.0, -depth]])
    field = _kernel(x, y, source, f) @ (strength * moment_dir)
    geology = 0.02 * x - 0.015 * y + 20.0 * np.sin(2 * np.pi * x / 1500.0)
    rng = np.random.default_rng(seed)
    values = field + geology + rng.normal(0, noise_nt, len(x))
    return x, y, values, field


def _estimate(case, depth=DEPTH):
    x, y, values, _ = case
    return estimate_magnetization(x, y, values, 0.0, 0.0, depth, INC, DEC, line_spacing_m=50.0)


def test_an_induced_source_reads_as_induced():
    est = _estimate(_case(_field_direction(INC, DEC)))

    assert est is not None
    assert est.angle_from_induced_deg < 15.0
    assert abs(est.inclination_deg - INC) < 15.0
    assert est.verdict == "induced"


def test_a_source_magnetized_sideways_reads_as_remanent():
    """A piece of steel magnetized while lying in some other orientation:
    109 degrees from the present field here."""
    direction = _rotate(None, -10.0, 100.0)
    truth = np.degrees(np.arccos(np.dot(direction, _field_direction(INC, DEC))))

    est = _estimate(_case(direction))

    assert est is not None
    assert abs(est.angle_from_induced_deg - truth) < 15.0
    assert abs(est.inclination_deg - (-10.0)) < 15.0
    assert est.verdict == "remanent"
    assert est.improvement > 0.25


def test_a_reversed_source_is_not_called_induced():
    """The case no fit-improvement test can catch: a moment along the
    field but pointing the other way fits exactly as well as an induced
    one with a negative amplitude - which would be a negative
    susceptibility, so it is remanence."""
    est = _estimate(_case(-_field_direction(INC, DEC)))

    assert est.angle_from_induced_deg > 150.0
    assert est.improvement < 0.25          # freeing the direction buys nothing here
    assert est.verdict == "remanent"


def test_freedom_buys_nothing_on_an_induced_source():
    """The number that makes the verdict trustworthy: allowing any
    direction must not improve the fit when the source really is
    induced."""
    est = _estimate(_case(_field_direction(INC, DEC)))

    assert est.improvement < 0.25


def test_an_anomaly_lost_in_noise_is_unclear_not_remanent():
    est = _estimate(_case(_rotate(None, -10.0, 100.0), strength=2e3, noise_nt=3.0))

    assert est is not None
    assert est.verdict in ("unclear", "induced")
    assert est.verdict != "remanent"


def test_it_declines_when_there_is_nothing_to_fit():
    x = np.array([0.0, 10.0, 20.0])
    assert estimate_magnetization(x, x, x, 0.0, 0.0, 50.0, INC, DEC, 50.0) is None


def test_the_estimate_survives_a_wrong_depth():
    """The depth handed in comes from the analytic signal's width and can
    be out by a third; the direction must not flip because of that."""
    case = _case(_rotate(None, -10.0, 100.0))
    near = _estimate(case, depth=DEPTH)
    shallow = _estimate(case, depth=DEPTH * 0.7)
    deep = _estimate(case, depth=DEPTH * 1.4)

    for est in (shallow, deep):
        assert abs(est.angle_from_induced_deg - near.angle_from_induced_deg) < 20.0
        assert est.verdict == "remanent"


def test_it_reports_what_it_fitted():
    est = _estimate(_case(_field_direction(INC, DEC)))
    d = est.to_dict()

    assert d["n_points"] > 40
    assert d["label"] == "유도자화에 가까움"
    assert d["free_fit_rms_nt"] <= d["induced_fit_rms_nt"] + 1e-9
