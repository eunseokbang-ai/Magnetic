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


def test_the_filter_removes_the_unresolved_across_line_detail():
    """What the filter is actually for: across the lines a survey resolves
    nothing below the line spacing, and differentiation multiplies that
    invented detail by the wavenumber. Measured on a real survey grid the
    analytic signal carried 2.75% of local variance at 20-60 m; the filter
    takes it to 0.18%."""
    yy, xx = np.mgrid[0:200, 0:200] * CELL
    base = 300 * np.sin(2 * np.pi * xx / 1500) + 200 * np.cos(2 * np.pi * yy / 1800)
    # Unresolved across-line detail: shorter than the line spacing, running
    # parallel to the lines.
    unresolved = base + 15 * np.sin(2 * np.pi * xx / (0.4 * SPACING))

    filtered = limit_to_line_spacing_resolution(unresolved, CELL, SPACING)

    inner = (slice(40, -40), slice(40, -40))
    left = float(np.std((filtered - limit_to_line_spacing_resolution(base, CELL, SPACING))[inner]))
    assert left < 0.25 * 15.0, f"{left:.2f} nT of the 15 nT artifact survived"


def test_the_filter_removes_stripes_running_either_way():
    """Surveys are flown in blocks at right angles, so one grid can carry
    corrugation running both ways - which is exactly what the real grid
    showed (83% of neighbouring tiles agreed on a direction, but both
    directions were present). The filter has to catch both without being
    told which."""
    yy, xx = np.mgrid[0:200, 0:200] * CELL
    base = 300 * np.sin(2 * np.pi * xx / 1500) + 200 * np.cos(2 * np.pi * yy / 1800)
    clean = limit_to_line_spacing_resolution(base, CELL, SPACING)
    inner = (slice(40, -40), slice(40, -40))

    for axis, coord in (("vertical stripes", xx), ("horizontal stripes", yy)):
        striped = base + 15 * np.sin(2 * np.pi * coord / (0.4 * SPACING))
        left = float(np.std((limit_to_line_spacing_resolution(striped, CELL, SPACING) - clean)[inner]))
        assert left < 0.25 * 15.0, f"{axis}: {left:.2f} nT of 15 survived"


def test_the_filter_keeps_a_compact_target_an_isotropic_one_would_erase():
    """Why the filter is directional rather than a plain smooth. A buried
    object is short-wavelength in every direction, so an isotropic filter
    wide enough to kill corrugation takes the target with it - on the real
    survey grid it cost 45% of a 544 nT dipole's amplitude. Corrugation
    only points across the lines, so attenuating just that direction
    separates the two: 96% of the dipole kept."""
    from scipy import ndimage

    yy, xx = np.mgrid[0:200, 0:200] * CELL
    r2 = (xx - 1000.0) ** 2 + (yy - 1000.0) ** 2
    dipole = 400.0 * (yy - 1000.0) / 60.0 * np.exp(-r2 / (2 * 60.0**2))
    grid = 300 * np.sin(2 * np.pi * xx / 1500) + dipole
    target = (slice(70, 130), slice(70, 130))

    def peak_to_peak(a):
        return float(np.max(a[target]) - np.min(a[target]))

    before = peak_to_peak(grid)
    directional = peak_to_peak(limit_to_line_spacing_resolution(grid, CELL, SPACING))
    isotropic = peak_to_peak(ndimage.gaussian_filter(grid, 0.4 * SPACING / CELL, mode="nearest"))

    assert directional > 0.85 * before, f"directional kept only {directional / before:.0%}"
    assert directional > isotropic, "the directional filter must beat the isotropic one it replaced"


def test_it_does_not_pretend_to_rescue_a_raw_cell_grid():
    """An honest limitation, pinned so it cannot be quietly forgotten. A
    nearest-gridded surface steps at exactly the line spacing; clearing
    that needs a cutoff near twice the spacing, which costs a third of
    every compact target. So the filter helps but does not fix it - the
    fix is to grid smoothly, and the API warns about it (see
    store._raw_cell_derivative_warning)."""
    g, truth_field = _gridded("nearest")
    truth = analytic_signal(truth_field, CELL)

    raw = _relative_error(analytic_signal(g.values, CELL), truth)
    filtered = _relative_error(
        analytic_signal(limit_to_line_spacing_resolution(g.values, CELL, SPACING), CELL), truth
    )
    smooth_gridded = _relative_error(analytic_signal(_gridded("linear")[0].values, CELL), truth)

    assert filtered < raw / 2, "it should still help substantially"
    assert filtered > smooth_gridded, "but gridding smoothly is what actually fixes this"


def test_smoothing_is_skipped_when_it_would_do_nothing_or_cannot_be_sized():
    g, _truth = _gridded("nearest")

    # No line spacing known - nothing to size the filter from.
    assert limit_to_line_spacing_resolution(g.values, CELL, None) is g.values
    # Cells already as coarse as the cutoff.
    assert limit_to_line_spacing_resolution(g.values, 100.0, SPACING) is g.values


def test_the_filter_keeps_the_masked_area_masked():
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
