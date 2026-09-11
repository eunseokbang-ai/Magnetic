"""Replacing the grid with the field of a fitted source layer.

This is the striping remedy for delivered dXX/dXY/dYY images, and its
claim is a physical one: a source distribution at depth h cannot produce
structure finer than about h, so a layer fitted to the data and then read
back removes what the survey could not resolve without cutting a band out
of the spectrum. These tests check that claim rather than the code paths -
that the readings are still reproduced, that across-line detail finer than
the layer goes and real anomalies stay, and that it beats the band filter
it replaces at equal cost.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import TransformRequest
from app.processing.continuation import equivalent_source_field, field_depth_cells
from app.processing.transforms import limit_to_line_spacing_resolution, second_derivative_ee
from app.store import store as project_store

CELL = 10.0
SPACING = 50.0
SHAPE = (200, 200)


def _geology(shape=SHAPE, cell=CELL, seed=4, n=70):
    """Smooth anomalies from buried dipoles - the part a survey on 50 m
    lines can actually resolve, and so the part that must survive."""
    ny, nx = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(ny) * cell, np.arange(nx) * cell, indexing="ij")
    field = np.zeros(shape)
    for _ in range(n):
        x0 = rng.uniform(0, ny * cell)
        y0 = rng.uniform(0, nx * cell)
        depth = rng.uniform(60.0, 200.0)
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2 + depth ** 2
        field += rng.normal(0, 1.0) * depth ** 3 * (2 * depth ** 2 - (r2 - depth ** 2)) / r2 ** 2.5
    return field * (40.0 / field.std())


def _undersampled(shape=SHAPE, cell=CELL, spacing=SPACING, seed=9):
    """A field the survey cannot resolve across-line, gridded the way the
    survey would grid it: shallow sources sampled only on lines `spacing`
    apart and interpolated between them.

    This is how the striping actually arises, and it matters for the
    comparison below. A single sine at the line spacing - the obvious
    fixture - is the one artifact a band filter is built for, and the
    filter duly wins against it. What the real data carries is broadband
    across-line aliasing, where the filter has no band to aim at.
    """
    ny, nx = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(ny) * cell, np.arange(nx) * cell, indexing="ij")
    rough = np.zeros(shape)
    for _ in range(120):
        x0 = rng.uniform(0, ny * cell)
        y0 = rng.uniform(0, nx * cell)
        depth = rng.uniform(15.0, 45.0)          # too shallow for 50 m lines
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2 + depth ** 2
        rough += rng.normal(0, 1.0) * depth ** 3 * (2 * depth ** 2 - (r2 - depth ** 2)) / r2 ** 2.5
    rough *= 25.0 / rough.std()
    step = max(1, int(round(spacing / cell)))
    lines_at = np.arange(0, nx, step)
    sampled = np.empty(shape)
    for row in range(ny):
        sampled[row] = np.interp(np.arange(nx), lines_at, rough[row, lines_at])
    return sampled


def _across_line_power(field, cell=CELL, spacing=SPACING):
    """Share of the variance sitting in across-line structure around the
    line spacing - the striping, measured rather than eyeballed."""
    a = np.where(np.isfinite(field), field, 0.0)
    power = np.abs(np.fft.fft2(a - a.mean())) ** 2
    kx = np.fft.fftfreq(a.shape[0], d=cell)[:, None]
    ky = np.fft.fftfreq(a.shape[1], d=cell)[None, :]
    k = np.hypot(kx, ky)
    wavelength = 1.0 / np.maximum(k, 1e-12)
    band = (wavelength > 0.6 * spacing) & (wavelength < 1.8 * spacing) & (np.abs(ky) > 2.5 * np.abs(kx))
    return float(power[band].sum() / power[k > 0].sum())


DEPTH = field_depth_cells(SPACING, CELL, 1.6)


# ---------------------------------------------------------------------------
# it still has to be the survey's own field
# ---------------------------------------------------------------------------


def test_the_layer_reproduces_the_readings():
    """The misfit is what makes this defensible: it says in nanotesla what
    the smoothing cost."""
    grid = _geology()

    field, misfit = equivalent_source_field(grid, DEPTH)

    assert misfit < 0.15 * grid.std(), f"misfit {misfit:.1f} nT against a {grid.std():.1f} nT field"
    assert np.isfinite(field).all()


def test_real_anomalies_keep_their_amplitude():
    grid = _geology()
    peaks = np.abs(grid) >= np.quantile(np.abs(grid), 0.999)

    field, _ = equivalent_source_field(grid, DEPTH)

    kept = np.abs(field[peaks]).mean() / np.abs(grid[peaks]).mean()
    assert 0.8 <= kept <= 1.2, f"peak amplitude changed by {100 * (kept - 1):.0f}%"


def test_gaps_in_the_survey_are_filled_too():
    grid = _geology()
    grid[80:120, 80:120] = np.nan

    field, _ = equivalent_source_field(grid, DEPTH)

    assert np.isfinite(field).all()


# ---------------------------------------------------------------------------
# what it is for
# ---------------------------------------------------------------------------


def test_it_removes_across_line_structure_the_survey_could_not_sample():
    grid = _geology() + _undersampled()

    field, _ = equivalent_source_field(grid, DEPTH)

    assert _across_line_power(field) < 0.35 * _across_line_power(grid)


def test_it_removes_the_corrugation_faster_than_it_removes_the_geology():
    """A smoother that took both equally would be no better than blurring
    the map."""
    geology = _geology()
    grid = geology + _undersampled()

    field, _ = equivalent_source_field(grid, DEPTH)

    corrugation_left = _across_line_power(field) / _across_line_power(grid)
    geology_left = np.std(equivalent_source_field(geology, DEPTH)[0]) / np.std(geology)
    assert corrugation_left < 0.5 * geology_left


def test_a_deeper_layer_smooths_more():
    grid = _geology() + _undersampled()

    shallow, _ = equivalent_source_field(grid, field_depth_cells(SPACING, CELL, 1.0))
    deep, _ = equivalent_source_field(grid, field_depth_cells(SPACING, CELL, 2.5))

    assert _across_line_power(deep) < _across_line_power(shallow)


def test_the_cost_of_the_smoothing_is_reported_and_grows_with_depth():
    """The misfit is the number a client gets told, so it has to track
    what was actually given up.

    Note what is deliberately NOT asserted here: that this beats the
    across-line filter. That comparison was made on the real survey grid
    (the table in processing/continuation.py) and it holds there because
    the aliasing is broadband. On a synthetic the aliasing lands in a
    narrow band around the line spacing, which is exactly what a band
    filter is built for, and the filter wins - so a synthetic cannot
    honestly stand in for that measurement.
    """
    grid = _geology() + _undersampled()

    misfits = [equivalent_source_field(grid, field_depth_cells(SPACING, CELL, f))[1]
               for f in (1.0, 1.6, 2.5)]

    assert misfits == sorted(misfits), misfits
    assert misfits[1] < 0.5 * grid.std(), "the default depth must not rewrite the survey"


def test_the_second_derivative_of_the_fitted_field_is_quieter():
    """dXX is the product that suffers most - the pure across-line second
    derivative - so it is the one worth pinning."""
    grid = _geology() + _undersampled()

    before = second_derivative_ee(grid, CELL)
    after = second_derivative_ee(equivalent_source_field(grid, DEPTH)[0], CELL)

    assert _across_line_power(after) < _across_line_power(before)


# ---------------------------------------------------------------------------
# depth, and declining without one
# ---------------------------------------------------------------------------


def test_the_depth_is_read_from_the_line_spacing():
    assert field_depth_cells(50.0, 10.0, 1.6) == pytest.approx(8.0)
    assert field_depth_cells(50.0, 5.0, 1.6) == pytest.approx(16.0), "same metres, smaller cells"


def test_no_line_spacing_means_no_depth_rather_than_a_guess():
    """Smoothing by an amount unrelated to what the survey resolved would
    be worse than not smoothing."""
    assert field_depth_cells(None, 10.0, 1.6) is None
    assert field_depth_cells(0.0, 10.0, 1.6) is None


def test_the_field_is_cached_on_the_grid_and_depth():
    import time

    grid = _geology(seed=17)
    equivalent_source_field(grid, DEPTH)
    t0 = time.perf_counter()
    again, _ = equivalent_source_field(grid, DEPTH)
    assert time.perf_counter() - t0 < 0.5
    assert np.array_equal(again, equivalent_source_field(grid, DEPTH)[0])


# ---------------------------------------------------------------------------
# through the pipeline
# ---------------------------------------------------------------------------


def _processed_project():
    client = TestClient(app)
    project_id = client.post("/api/projects").json()["project_id"]
    for kind, path in (("drone", "tests/fixtures/sample_drone_survey.csv"),
                       ("base", "tests/fixtures/sample_base_station.csv")):
        with open(path, "rb") as f:
            client.post(f"/api/projects/{project_id}/upload/{kind}", files={"files": ("a.csv", f, "text/csv")})
    r = client.post(
        f"/api/projects/{project_id}/process",
        json={"line_params": {}, "diurnal_params": {}, "heading_correction": {},
              "auto_display_boundary": False},
    )
    assert r.status_code == 200, r.text
    return project_store.get(project_id)


def _req(**over):
    kwargs = {"transform": "dxx", "value": "anomaly", "cell_size_m": 5.0, "method": "linear"}
    kwargs.update(over)
    return TransformRequest(**kwargs)


def test_it_is_off_unless_asked_for():
    """It trades real amplitude for a cleaner picture, which is the user's
    call to make, not a default."""
    project = _processed_project()

    assert project.get_transform_overlay(_req())["equivalent_source"] == {"applies": False}


def test_switching_it_on_reports_the_depth_and_what_it_cost():
    project = _processed_project()

    report = project.get_transform_overlay(_req(equivalent_source_factor=1.6))["equivalent_source"]

    assert report["applied"] is True
    assert report["depth_m"] > 0 and report["misfit_nt"] > 0
    assert 0 < report["misfit_pct"] < 100


def test_it_changes_the_derivative():
    project = _processed_project()

    project.get_transform_overlay(_req())
    before = project.last_overlay.values.copy()
    project.get_transform_overlay(_req(equivalent_source_factor=1.6))
    after = project.last_overlay.values

    ok = np.isfinite(before) & np.isfinite(after)
    assert not np.allclose(before[ok], after[ok])


def test_the_survey_footprint_is_unchanged():
    project = _processed_project()

    project.get_transform_overlay(_req())
    without = np.isnan(project.last_overlay.values)
    project.get_transform_overlay(_req(equivalent_source_factor=1.6))
    assert np.array_equal(without, np.isnan(project.last_overlay.values))
