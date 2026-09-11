"""The boundary-margin option: marking (or hiding) the band along the edge
of the data where a derived grid is partly measuring its own boundary.

The band exists because the FFT transforms have to invent values outside
the survey; an anomaly cut by the coverage boundary then streaks along the
grid axes. processing/edge_margin.py records the measurements that ruled
out fixing the fill instead. What these tests pin down is that the option
marks the right band and, above all, leaves the interior alone - an option
that quietly shaved good data would be worse than the streak it hides.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import TransformRequest
from app.processing.edge_margin import apply_margin, edge_distance_m, margin_mask, margin_outline
from app.store import store as project_store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"

CELL = 10.0


def _full_grid(n=40):
    return np.ones((n, n), dtype=float)


def _grid_with_hole(n=40):
    g = _full_grid(n)
    g[18:22, 18:22] = np.nan
    return g


# ---------------------------------------------------------------------------
# the distance field
# ---------------------------------------------------------------------------


def test_the_array_border_counts_as_a_boundary():
    """A grid full to its own edge is still one cell from the outside
    world - the FFT wraps there, so the outermost row is margin."""
    dist = edge_distance_m(_full_grid(), CELL)

    assert dist[0, 0] == pytest.approx(CELL)
    assert dist[0, 20] == pytest.approx(CELL)
    assert dist[20, 20] == pytest.approx(20 * CELL), "nearest outside cell is the row past the far edge"


def test_distance_is_measured_to_the_nearest_gap_not_only_the_outline():
    """An interior hole (a no-fly zone, a dropped line) is a boundary too -
    the fill invents values around it exactly as it does at the edge."""
    dist = edge_distance_m(_grid_with_hole(), CELL)

    assert dist[17, 20] == pytest.approx(CELL)
    assert dist[20, 20] == 0.0, "a cell that is itself a gap has no distance to one"


def test_distance_scales_with_the_cell_size():
    assert edge_distance_m(_full_grid(), 5.0)[20, 20] == pytest.approx(20 * 5.0)


def test_a_nonpositive_cell_size_is_rejected():
    with pytest.raises(ValueError):
        edge_distance_m(_full_grid(), 0.0)


# ---------------------------------------------------------------------------
# the mask
# ---------------------------------------------------------------------------


def test_the_mask_covers_the_edge_band_and_nothing_deeper():
    mask = margin_mask(_full_grid(), CELL, margin_m=30.0)

    assert mask[0, 0] and mask[2, 20], "three cells in is within a 30 m margin"
    assert not mask[3, 20] and not mask[20, 20]


def test_cells_that_are_already_gaps_are_not_counted_as_margin():
    """Otherwise the reported margin area would grow with the size of the
    bounding box rather than with the survey."""
    mask = margin_mask(_grid_with_hole(), CELL, margin_m=30.0)

    assert not mask[20, 20]
    assert mask[17, 20], "but the data ring around the hole is margin"


def test_a_zero_margin_marks_nothing():
    assert not margin_mask(_full_grid(), CELL, margin_m=0.0).any()


def test_applying_the_mask_leaves_the_input_array_untouched():
    """Callers hand in cached grid/transform arrays that the next request
    still has to be able to use."""
    values = np.arange(1600, dtype=float).reshape(40, 40)
    before = values.copy()

    out = apply_margin(values, margin_mask(_full_grid(), CELL, 30.0))

    assert np.array_equal(values, before)
    assert np.isnan(out[0, 0]) and out[20, 20] == before[20, 20]


def test_masking_removes_exactly_the_marked_cells():
    values = np.arange(1600, dtype=float).reshape(40, 40)
    mask = margin_mask(_full_grid(), CELL, 30.0)

    out = apply_margin(values, mask)

    assert np.array_equal(np.isnan(out), mask)


# ---------------------------------------------------------------------------
# the outline (the non-destructive half of the option)
# ---------------------------------------------------------------------------


def test_the_outline_traces_the_line_where_the_margin_ends():
    easting = np.arange(40) * CELL
    northing = np.arange(40) * CELL
    margin = 50.0

    paths = margin_outline(_full_grid(), CELL, margin, easting, northing)

    assert paths, "a square survey has an inner region to draw around"
    dist = edge_distance_m(_full_grid(), CELL)
    for path in paths:
        # Every vertex sits on the margin_m level of the distance field,
        # to within the bilinear interpolation of one cell.
        cols = path[:, 0] / CELL
        rows = path[:, 1] / CELL
        sampled = dist[np.rint(rows).astype(int), np.rint(cols).astype(int)]
        assert np.abs(sampled - margin).max() <= CELL


def test_no_outline_when_the_margin_swallows_the_survey():
    """Nothing to draw a line around; the caller says so in words instead
    of showing a line that does not mean what it looks like."""
    easting = np.arange(40) * CELL
    northing = np.arange(40) * CELL

    assert margin_outline(_full_grid(), CELL, 10_000.0, easting, northing) == []


# ---------------------------------------------------------------------------
# through the pipeline
# ---------------------------------------------------------------------------


def _processed_project():
    client = TestClient(app)
    project_id = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(
        f"/api/projects/{project_id}/process",
        json={"line_params": {}, "diurnal_params": {}, "heading_correction": {},
              "auto_display_boundary": False},
    )
    assert r.status_code == 200, r.text
    return project_store.get(project_id)


def _transform(**over):
    kwargs = {"transform": "as", "value": "anomaly", "cell_size_m": 5.0, "method": "linear"}
    kwargs.update(over)
    return TransformRequest(**kwargs)


def test_off_by_default_nothing_is_hidden():
    """A target sitting on the survey edge is real data; hiding it without
    being asked would be worse than the streak."""
    project = _processed_project()

    overlay = project.get_transform_overlay(_transform())

    assert overlay["boundary_margin"] == {"mode": "off", "applied": False}


def test_masking_blanks_the_edge_band_and_keeps_the_interior_identical():
    project = _processed_project()
    plain = project.get_transform_overlay(_transform())
    before = project.last_overlay.values.copy()

    overlay = project.get_transform_overlay(_transform(boundary_margin_mode="mask"))
    after = project.last_overlay.values

    hidden = np.isfinite(before) & ~np.isfinite(after)
    assert hidden.any(), "the option has to actually hide something"
    kept = np.isfinite(after)
    assert np.allclose(after[kept], before[kept], equal_nan=True), "interior values must be untouched"
    assert overlay["boundary_margin"]["applied"] is True
    assert overlay["boundary_margin"]["n_cells_in_margin"] == int(hidden.sum())
    assert 0.0 < overlay["boundary_margin"]["pct_of_grid"] < 100.0
    assert plain["boundary_margin"]["applied"] is False


def test_the_hidden_cells_are_the_ones_near_the_data_boundary():
    project = _processed_project()
    project.get_transform_overlay(_transform())
    before = project.last_overlay.values.copy()

    overlay = project.get_transform_overlay(_transform(boundary_margin_mode="mask", boundary_margin_m=30.0))
    after = project.last_overlay.values

    hidden = np.isfinite(before) & ~np.isfinite(after)
    dist = edge_distance_m(before, 5.0)
    assert dist[hidden].max() <= 30.0 + 1e-6
    assert dist[np.isfinite(after)].min() > 30.0


def test_outline_mode_hides_nothing():
    """"표시" and "마스킹" are separate choices - outline is the one you can
    turn on without losing a single cell."""
    project = _processed_project()
    project.get_transform_overlay(_transform())
    before = project.last_overlay.values.copy()

    overlay = project.get_transform_overlay(_transform(boundary_margin_mode="outline"))
    after = project.last_overlay.values

    assert np.array_equal(np.isnan(before), np.isnan(after))
    margin = overlay["boundary_margin"]
    assert margin["applied"] is True and margin["outline"]
    # The outline comes back in lat/lon so the map can draw it directly:
    # every vertex has to land inside the overlay's own footprint.
    (south, west), (north, east) = overlay["bounds"]
    for lat, lon in margin["outline"][0]:
        assert south <= lat <= north and west <= lon <= east


def test_the_default_width_is_the_gridding_extrapolation_radius():
    """The band worth marking is the one whose values were extrapolated
    outward from the lines rather than interpolated between them."""
    project = _processed_project()
    req = _transform(boundary_margin_mode="outline")

    margin = project.get_transform_overlay(req)["boundary_margin"]

    assert margin["from_extrapolation_radius"] is True
    assert margin["margin_m"] == pytest.approx(round(project._resolve_max_distance(5.0, None), 1))


def test_a_margin_wider_than_the_survey_declines_instead_of_blanking_the_map():
    project = _processed_project()

    overlay = project.get_transform_overlay(
        _transform(boundary_margin_mode="mask", boundary_margin_m=100_000.0)
    )

    margin = overlay["boundary_margin"]
    assert margin["applied"] is False and "줄이세요" in margin["reason"]
    assert np.isfinite(project.last_overlay.values).any(), "the map must survive"


def test_exports_carry_the_same_margin_as_the_screen():
    """A GeoTIFF that puts back what the map is hiding is the one that ends
    up in the report."""
    project = _processed_project()
    plain = project.export_transform_xyz(_transform())
    masked = project.export_transform_xyz(_transform(boundary_margin_mode="mask"))

    assert len(masked.splitlines()) < len(plain.splitlines())


def test_the_report_says_the_margin_is_a_guide_not_a_guarantee():
    """The streak decays slowly; "outside the margin" is not a clean bill
    of health, and the readout has to admit that."""
    project = _processed_project()

    margin = project.get_transform_overlay(_transform(boundary_margin_mode="mask"))["boundary_margin"]

    assert "여백 바깥이라고 해서" in margin["note"]
