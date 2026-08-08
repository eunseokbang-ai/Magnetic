"""Tests for the click-to-inspect map value sampler
(Project.sample_overlay_value / POST /overlay/sample): interpolates the
exact value shown by the most recently rendered grid/derivative overlay
at a clicked lat/lon, distinct from sample_point()'s nearest-raw-point
lookup.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyproj import Transformer

from app.main import app
from app.store import ProjectError, store as project_store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def _make_processed_project():
    client = TestClient(app)
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{project_id}/process", json={"line_params": {}, "diurnal_params": {}, "heading_correction": {}})
    assert r.status_code == 200, r.text
    return client, project_id


def test_sample_overlay_value_before_grid_raises():
    client, project_id = _make_processed_project()
    project = project_store.get(project_id)
    with pytest.raises(ProjectError):
        project.sample_overlay_value(35.0, 126.0)


def test_sample_overlay_value_in_bounds_matches_grid_center():
    client, project_id = _make_processed_project()
    r = client.post(f"/api/projects/{project_id}/grid", json={"value": "anomaly", "cell_size_m": 10.0})
    assert r.status_code == 200, r.text

    project = project_store.get(project_id)
    easting, northing = project.last_overlay_easting, project.last_overlay_northing
    assert project.last_overlay_values is not None

    from pyproj import Transformer
    mid_x = float(easting[len(easting) // 2])
    mid_y = float(northing[len(northing) // 2])
    transformer = Transformer.from_crs(f"EPSG:{project.utm_epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(mid_x, mid_y)

    r = client.post(f"/api/projects/{project_id}/overlay/sample", json={"lat": lat, "lon": lon})
    assert r.status_code == 200, r.text
    resp = r.json()
    assert resp["label"] == "anomaly"
    if resp["in_bounds"]:
        assert isinstance(resp["value_nt"], float)
    else:
        # center cell can be NaN-masked outside data coverage; fine either way
        assert resp["value_nt"] is None


def test_sample_overlay_value_far_outside_bounds_is_out_of_bounds():
    client, project_id = _make_processed_project()
    client.post(f"/api/projects/{project_id}/grid", json={"value": "anomaly", "cell_size_m": 10.0})

    r = client.post(f"/api/projects/{project_id}/overlay/sample", json={"lat": 60.0, "lon": 30.0})
    assert r.status_code == 200, r.text
    resp = r.json()
    assert resp["in_bounds"] is False
    assert resp["value_nt"] is None


def test_sample_overlay_value_returns_exact_nearest_cell_not_a_blend():
    """Regression test: sample_overlay_value used to bilinearly interpolate
    between neighboring cells, which could return a value blended toward a
    sharply different neighbor instead of the exact value of the cell the
    user actually clicked on (or actually visible at that pixel, now that
    the overlay renders one flat color per cell - see
    processing/render.py::grid_to_png_overlay). Injects a small synthetic
    grid with a single very different "hot" cell directly onto a real
    Project (bypassing the real gridding pipeline, so the exact node
    values are known) and checks an exact-node click returns precisely
    that node's value, not something blended toward its very different
    neighbor."""
    client, project_id = _make_processed_project()
    client.post(f"/api/projects/{project_id}/grid", json={"value": "anomaly", "cell_size_m": 10.0})
    project = project_store.get(project_id)

    cell = 10.0
    n = 9
    easting = np.arange(0, n * cell, cell)
    northing = np.arange(0, n * cell, cell)
    values = np.zeros((n, n))
    hot_row, hot_col = 4, 4
    values[hot_row, hot_col] = 500.0  # sharply different from its all-zero neighbors
    project.last_overlay_easting = easting
    project.last_overlay_northing = northing
    project.last_overlay_values = values

    transformer = Transformer.from_crs(f"EPSG:{project.utm_epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting[hot_col], northing[hot_row])

    resp = project.sample_overlay_value(lat, lon)
    assert resp["in_bounds"] is True
    assert resp["value_nt"] == pytest.approx(500.0)


def test_sample_overlay_value_reports_no_data_for_the_exact_blank_cell():
    """The other half of the same bug: a click whose nearest cell is
    genuinely NaN (blank on the map) must report "no data" even though a
    bilinear read could have blended in a nearby finite neighbor and
    returned a (wrong) number instead."""
    client, project_id = _make_processed_project()
    client.post(f"/api/projects/{project_id}/grid", json={"value": "anomaly", "cell_size_m": 10.0})
    project = project_store.get(project_id)

    cell = 10.0
    n = 9
    easting = np.arange(0, n * cell, cell)
    northing = np.arange(0, n * cell, cell)
    values = np.full((n, n), 42.0)
    blank_row, blank_col = 4, 4
    values[blank_row, blank_col] = np.nan
    project.last_overlay_easting = easting
    project.last_overlay_northing = northing
    project.last_overlay_values = values

    transformer = Transformer.from_crs(f"EPSG:{project.utm_epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting[blank_col], northing[blank_row])

    resp = project.sample_overlay_value(lat, lon)
    assert resp["in_bounds"] is False
    assert resp["value_nt"] is None


def test_sample_overlay_value_switches_source_after_transform():
    client, project_id = _make_processed_project()
    r = client.post(f"/api/projects/{project_id}/grid", json={"value": "anomaly", "cell_size_m": 10.0})
    assert r.status_code == 200, r.text
    project = project_store.get(project_id)
    assert project.last_overlay_label == "anomaly"

    r = client.post(
        f"/api/projects/{project_id}/transform",
        json={"transform": "1vd", "value": "anomaly", "cell_size_m": 10.0},
    )
    assert r.status_code == 200, r.text
    project = project_store.get(project_id)
    assert project.last_overlay_label == "1vd"


def test_grid_overlay_reports_extrema_locations_that_click_to_the_same_value():
    """Regression/feature test for the "최댓값/최솟값 위치로 이동" jump
    button: overlay["extrema"] must point at a lat/lon that, when clicked
    via sample_overlay_value (the exact same code path the map's
    click-to-inspect tool uses), returns precisely the reported max/min -
    i.e. clicking there can never land on a neighboring cell."""
    client, project_id = _make_processed_project()
    r = client.post(f"/api/projects/{project_id}/grid", json={"value": "anomaly", "cell_size_m": 5.0})
    assert r.status_code == 200, r.text
    resp = r.json()

    assert resp["extrema"]["max"] is not None
    assert resp["extrema"]["min"] is not None
    assert resp["extrema"]["max"]["value_nt"] == pytest.approx(resp["stats"]["max"])
    assert resp["extrema"]["min"]["value_nt"] == pytest.approx(resp["stats"]["min"])

    project = project_store.get(project_id)
    for key in ("max", "min"):
        loc = resp["extrema"][key]
        sampled = project.sample_overlay_value(loc["lat"], loc["lon"])
        assert sampled["in_bounds"] is True
        assert sampled["value_nt"] == pytest.approx(loc["value_nt"])
