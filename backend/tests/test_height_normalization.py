"""Flight-height normalisation against a known field.

The synthetic (synthetic_hill_survey.py) is the situation the feature
exists for: a terrain-following flight over +-30 m of relief, so the
sensor altitude varies by 60 m across the survey, over shallow sources
and a regional. Everything is dipoles, so the true field at the reference
altitude is exact and the normalised readings can be held against it.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from app.processing.height_normalization import (
    HeightNormalization,
    HeightNormalizationError,
    block_average,
    dipole_layer_field,
    fit_height_normalization,
)
from tests.synthetic_hill_survey import DEC, INC, LINE_SPACING, hill_survey, true_field, true_sources


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


@pytest.fixture(scope="module")
def hill():
    x, y, z, line_id, anomaly = hill_survey(noise_nt=0.3)
    z_ref = float(np.median(z))
    truth_ref = true_field(x, y, np.full(len(x), z_ref))
    model = fit_height_normalization(x, y, z, anomaly, INC, DEC, LINE_SPACING)
    return dict(x=x, y=y, z=z, anomaly=anomaly, z_ref=z_ref, truth_ref=truth_ref, model=model)


def test_normalised_readings_match_the_field_at_the_reference_altitude(hill):
    """The handoff's acceptance test: rms error against the true field at
    z_ref falls to well under a third of what it was. Measured 7.07 ->
    0.43 nT (ratio 0.06) - see the module docstring's table."""
    m = hill["model"]
    corr = m.correction(hill["x"], hill["y"], hill["z"])
    before = _rms(hill["anomaly"] - hill["truth_ref"])
    after = _rms(hill["anomaly"] + corr - hill["truth_ref"])
    assert before > 5.0                      # the height effect is real on this synthetic
    assert after < before / 3.0
    assert after < 1.0
    assert abs(m.z_ref_m - hill["z_ref"]) < 1e-9   # default z_ref is the median altitude


def test_the_layers_fit_the_readings_to_the_noise_floor_within_the_iteration_budget(hill):
    m = hill["model"]
    assert m.fit_rms_nt < 0.6                # 0.3 nT noise was added
    assert m.misfit_ratio < 0.02
    assert len(m.layers) == 3
    assert all(layer["n_iter"] <= 30 for layer in m.layers)
    assert not m.warnings


def test_shallow_sources_alone_are_corrected_too():
    """Without a regional, the whole height effect is the shallow
    sources'. Measured 2.90 -> 0.77 nT rms; peak error 53 -> 13 nT."""
    s, moments = true_sources()
    x, y, z, _, _ = hill_survey()
    rng = np.random.default_rng(1)
    strong = moments[:2] * 5
    obs = dipole_layer_field(x, y, z, s[:2], strong, INC, DEC) + 0.3 * rng.standard_normal(len(x))
    z_ref = float(np.median(z))
    ref = dipole_layer_field(x, y, np.full(len(x), z_ref), s[:2], strong, INC, DEC)

    m = fit_height_normalization(x, y, z, obs, INC, DEC, LINE_SPACING)
    corr = m.correction(x, y, z)

    before, after = _rms(obs - ref), _rms(obs + corr - ref)
    assert after < before / 2.0
    assert np.abs(obs + corr - ref).max() < np.abs(obs - ref).max() / 3.0


def test_a_flat_flight_is_left_alone():
    """+-1 m of altitude: the correction is at the noise level (median
    0.11 nT measured against 0.3 nT noise), and the model says so."""
    x, y, z, _, anomaly = hill_survey(relief=1.0)
    m = fit_height_normalization(x, y, z, anomaly, INC, DEC, LINE_SPACING)
    corr = m.correction(x, y, z)
    assert np.median(np.abs(corr)) < 0.3
    assert np.abs(corr).max() < 1.5
    assert any("2 m 미만" in w for w in m.warnings)


def test_a_user_reference_altitude_is_honoured(hill):
    x, y, z, anomaly = hill["x"], hill["y"], hill["z"], hill["anomaly"]
    z_ref = float(z.max()) + 20.0
    m = fit_height_normalization(x, y, z, anomaly, INC, DEC, LINE_SPACING, z_ref_m=z_ref)
    truth = true_field(x, y, np.full(len(x), z_ref))
    corr = m.correction(x, y, z)
    assert _rms(anomaly + corr - truth) < _rms(anomaly - truth) / 3.0


def test_no_correction_where_the_altitude_is_unknown(hill):
    m = hill["model"]
    z = hill["z"].copy()
    z[:10] = np.nan
    corr = m.correction(hill["x"], hill["y"], z)
    assert np.all(corr[:10] == 0.0)
    assert np.all(np.isfinite(corr))
    assert np.abs(corr[10:]).max() > 1.0


def test_the_model_round_trips_through_its_dict(hill):
    m = hill["model"]
    m.correction_stats = {"n": 3, "median_abs_nt": 1.0}
    back = HeightNormalization.from_dict(m.to_dict())
    x, y, z = hill["x"][:500], hill["y"][:500], hill["z"][:500]
    assert np.allclose(back.correction(x, y, z), m.correction(x, y, z), atol=1e-3)
    assert back.z_ref_m == m.z_ref_m
    assert back.layers == m.layers
    assert back.correction_stats == m.correction_stats
    assert back.summary()["n_sources"] == len(m.sources_xyz)


def test_too_few_usable_points_is_refused():
    x = np.linspace(0, 100, 30)
    with pytest.raises(HeightNormalizationError):
        fit_height_normalization(x, x, np.full(30, 50.0), np.zeros(30), INC, DEC, LINE_SPACING)


def test_block_average_means_each_cell():
    x = np.array([0.0, 1.0, 25.0, 26.0])
    y = np.array([0.0, 2.0, 0.0, 1.0])
    z = np.array([10.0, 20.0, 30.0, 50.0])
    v = np.array([1.0, 3.0, 5.0, 7.0])
    bx, by, bz, bv = block_average(x, y, z, v, 10.0)
    assert len(bx) == 2
    order = np.argsort(bx)
    assert np.allclose(bx[order], [0.5, 25.5])
    assert np.allclose(bz[order], [15.0, 40.0])
    assert np.allclose(bv[order], [2.0, 6.0])


def test_the_kernel_matches_the_source_removal_kernel_on_the_plane():
    """Same dipole physics as processing/source_removal.py, which the
    structure models rely on, evaluated there on z = 0."""
    from app.processing.source_removal import _field_direction, _kernel

    rng = np.random.default_rng(3)
    x, y = rng.uniform(-100, 100, 40), rng.uniform(-100, 100, 40)
    sources = np.column_stack([rng.uniform(-50, 50, 5), rng.uniform(-50, 50, 5), rng.uniform(-80, -30, 5)])
    moments = rng.uniform(1e3, 1e4, 5)
    f = _field_direction(INC, DEC)
    g = _kernel(x, y, sources, f)                       # (n, 3 * n_src), per moment component
    induced = (g.reshape(len(x), len(sources), 3) @ f) @ moments
    ours = dipole_layer_field(x, y, np.zeros(len(x)), sources, moments, INC, DEC)
    assert np.allclose(ours, induced, rtol=1e-9, atol=1e-9)
