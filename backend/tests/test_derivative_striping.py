"""Cell-scale striping in derivative grids (AS, dXY, 1VD, ...).

Two independent causes, both measured here against a field whose exact
answer is known, so "better" means closer to the truth rather than just
smoother-looking:

1. NaN gaps were filled with the grid mean before the FFT. On a real
   survey that is a step of hundreds of nT at the edge of every masked
   area, and differentiating a step gives a spike - Gibbs ringing spread
   along the ragged, cell-scale edges a distance mask leaves.
2. The default "raw cell" gridding is piecewise constant: each cell takes
   its nearest sounding's value, so across the flight lines whole runs of
   cells are exactly equal with steps between them. Differentiating that
   measures the interpolator, not the ground.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.processing.gridding import grid_points
from app.processing.transforms import (
    _fill_gaps,
    analytic_signal,
    limit_to_line_spacing_resolution,
    second_derivative_en,
)

CELL = 10.0
SPACING = 50.0


def _field(x, y):
    """A smooth field with structure well above the line spacing, so a
    correct derivative of it is fully determined by the survey."""
    return (
        300 * np.sin(2 * np.pi * x / 600)
        + 200 * np.cos(2 * np.pi * y / 800)
        + 60 * np.sin(2 * np.pi * (x + y) / 400)
    )


def _relative_error(actual: np.ndarray, truth: np.ndarray) -> float:
    """Mean |difference| as a fraction of the truth's own scale, ignoring
    the FFT taper zone at the edges."""
    m = np.isfinite(actual) & np.isfinite(truth)
    m[:8] = m[-8:] = False
    m[:, :8] = m[:, -8:] = False
    return float(np.mean(np.abs(actual[m] - truth[m])) / np.mean(np.abs(truth[m])))


def _cell_scale_striping(a: np.ndarray) -> float:
    """Mean |second difference| between adjacent rows, as a fraction of the
    grid's own scale - the alternation a stripe is made of."""
    v = np.where(np.isfinite(a), a, np.nan)
    return float(np.nanmean(np.abs(v[2:, :] - 2 * v[1:-1, :] + v[:-2, :])) / np.nanmean(np.abs(v)))


# ---------------------------------------------------------------------------
# cause 1: NaN gaps
# ---------------------------------------------------------------------------


def _grid_with_a_ragged_hole():
    yy, xx = np.mgrid[0:200, 0:200] * CELL
    clean = _field(xx, yy)
    holed = clean.copy()
    # A masked area with a cell-scale ragged edge, which is the shape a
    # max-distance mask leaves around flown lines.
    mask = np.zeros(clean.shape, bool)
    mask[60:140, 60:140] = True
    mask &= np.random.default_rng(0).random(clean.shape) > 0.15
    holed[mask] = np.nan
    return clean, holed


def test_masked_gaps_no_longer_ring_into_the_analytic_signal():
    clean, holed = _grid_with_a_ragged_hole()
    truth = analytic_signal(clean, CELL)

    out = analytic_signal(holed, CELL)

    valid = np.isfinite(out)
    # The mean fill used to leave 21% striping and a mean error of 1.43
    # against this same truth.
    assert _cell_scale_striping(out) < 0.03
    assert np.mean(np.abs(out[valid] - truth[valid])) < 0.05


def test_the_gap_fill_leaves_real_data_untouched():
    """It may only invent values where there were none."""
    _clean, holed = _grid_with_a_ragged_hole()
    mask = np.isnan(holed)

    filled = _fill_gaps(holed, mask)

    assert np.array_equal(filled[~mask], holed[~mask])
    assert np.isfinite(filled).all(), "every gap must end up filled"


def test_the_gap_fill_does_not_invent_values_outside_the_data_range():
    """A harmonic fill has no extrema of its own - anything outside the
    surrounding range would be a new anomaly the survey never measured."""
    _clean, holed = _grid_with_a_ragged_hole()
    mask = np.isnan(holed)

    filled = _fill_gaps(holed, mask)

    assert filled[mask].min() >= np.nanmin(holed) - 1e-9
    assert filled[mask].max() <= np.nanmax(holed) + 1e-9


def test_a_grid_with_no_gaps_is_passed_through_unchanged():
    clean, _holed = _grid_with_a_ragged_hole()
    assert _fill_gaps(clean, np.isnan(clean)) is clean


# ---------------------------------------------------------------------------
# cause 2: differentiating a piecewise-constant grid
# ---------------------------------------------------------------------------


def _line_survey():
    xs, ys, vals, lids = [], [], [], []
    for i in range(30):
        x = i * SPACING
        y = np.arange(0, 1200, 1.0)
        xs.append(np.full_like(y, x))
        ys.append(y)
        vals.append(_field(np.full_like(y, x), y))
        lids.append(np.full(len(y), i))
    return (np.concatenate(a) for a in (xs, ys, vals, lids))


def _gridded(method="nearest", cell=CELL):
    x, y, v, lid = _line_survey()
    g = grid_points(x, y, v, cell_size_m=cell, method=method, line_id=lid, typical_line_spacing_m=SPACING)
    east, north = np.meshgrid(g.easting, g.northing)
    return g, _field(east, north)


def test_raw_cell_gridding_really_is_piecewise_constant():
    """The premise of the fix: with nearest gridding most neighbouring
    cells across the lines hold the identical value, so the grid is flat
    blocks with steps between them - and a derivative sees the steps."""
    g, _truth = _gridded("nearest")
    d = np.diff(g.values, axis=1)
    finite = np.isfinite(d)

    assert np.mean(d[finite] == 0) > 0.5


def test_presmoothing_makes_the_analytic_signal_track_the_real_field():
    g, truth_field = _gridded("nearest")
    truth = analytic_signal(truth_field, CELL)

    raw = _relative_error(analytic_signal(g.values, CELL), truth)
    smoothed = _relative_error(
        analytic_signal(limit_to_line_spacing_resolution(g.values, CELL, SPACING), CELL), truth
    )

    assert raw > 0.5, "fixture no longer reproduces the problem this guards"
    assert smoothed < 0.15
    assert smoothed < raw / 5


def test_presmoothing_helps_the_second_derivative_too():
    g, truth_field = _gridded("nearest")
    truth = second_derivative_en(truth_field, CELL)

    raw = _relative_error(second_derivative_en(g.values, CELL), truth)
    smoothed = _relative_error(
        second_derivative_en(limit_to_line_spacing_resolution(g.values, CELL, SPACING), CELL), truth
    )

    assert smoothed < raw / 5


def test_presmoothing_beats_paying_for_a_smoother_interpolation():
    """Worth knowing, because re-gridding is the obvious alternative fix
    and it is both slower and less accurate here."""
    g_near, truth_field = _gridded("nearest")
    g_lin, _ = _gridded("linear")
    truth = analytic_signal(truth_field, CELL)

    smoothed = _relative_error(
        analytic_signal(limit_to_line_spacing_resolution(g_near.values, CELL, SPACING), CELL), truth
    )
    linear = _relative_error(analytic_signal(g_lin.values, CELL), truth)

    assert smoothed < linear


@pytest.mark.parametrize("cell", [10.0, 20.0])
def test_the_smoothing_width_follows_the_cell_size(cell):
    """The width is set in metres (a fraction of the line spacing), so it
    has to help at any cell size, not just the one it was tuned at."""
    g, truth_field = _gridded("nearest", cell=cell)
    truth = analytic_signal(truth_field, cell)

    raw = _relative_error(analytic_signal(g.values, cell), truth)
    smoothed = _relative_error(
        analytic_signal(limit_to_line_spacing_resolution(g.values, cell, SPACING), cell), truth
    )

    assert smoothed < raw / 3


def test_smoothing_is_skipped_when_it_would_do_nothing_or_cannot_be_sized():
    g, _truth = _gridded("nearest")

    # No line spacing known - nothing to size the filter from.
    assert limit_to_line_spacing_resolution(g.values, CELL, None) is g.values
    # Cells already as coarse as the resolution limit.
    assert limit_to_line_spacing_resolution(g.values, 100.0, SPACING) is g.values


def test_smoothing_keeps_the_masked_area_masked():
    """It must not bleed values into cells the survey never covered."""
    g, _truth = _gridded("nearest")
    values = g.values.copy()
    values[40:60, 40:60] = np.nan

    out = limit_to_line_spacing_resolution(values, CELL, SPACING)

    assert np.isnan(out[40:60, 40:60]).all()
    assert np.isfinite(out[np.isfinite(values)]).all()


# ---------------------------------------------------------------------------
# through the API
# ---------------------------------------------------------------------------


def test_derived_grids_are_presmoothed_by_default_and_it_can_be_turned_off():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    for endpoint, path in (
        ("drone", "tests/fixtures/sample_drone_survey.csv"),
        ("base", "tests/fixtures/sample_base_station.csv"),
    ):
        with open(path, "rb") as f:
            client.post(f"/api/projects/{pid}/upload/{endpoint}", files={"files": ("f.csv", f, "text/csv")})
    client.post(f"/api/projects/{pid}/process", json={"line_params": {}, "diurnal_params": {}, "heading_correction": {}})

    from app.store import store as project_store

    project = project_store.get(pid)
    body = {"transform": "as", "value": "anomaly", "cell_size_m": 20.0, "method": "nearest"}

    assert client.post(f"/api/projects/{pid}/transform", json=body).status_code == 200
    on = _cell_scale_striping(project.last_overlay.values)
    assert client.post(f"/api/projects/{pid}/transform", json={**body, "derivative_presmooth": False}).status_code == 200
    off = _cell_scale_striping(project.last_overlay.values)

    assert on < off, "the default must be the smoother one"
