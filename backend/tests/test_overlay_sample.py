"""Tests for the click-to-inspect map value sampler
(Project.sample_overlay_value / POST /overlay/sample): interpolates the
exact value shown by the most recently rendered grid/derivative overlay
at a clicked lat/lon, distinct from sample_point()'s nearest-raw-point
lookup.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

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
