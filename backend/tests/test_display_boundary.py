"""Tests for the optional user-drawn display-boundary polygon
(store.py::set_display_boundary / _apply_display_boundary) - lets the user
clip grid overlay/export output to an explicit outline, on top of the
automatic convex-hull extrapolation cap in processing/gridding.py. Mainly
useful for a concave (e.g. L-shaped) survey footprint where the hull cap
alone still leaves the concave notch filled in.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import DisplayBoundaryRequest, GridRequest
from app.store import ProjectError
from app.store import store as project_store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def _make_processed_project(auto_boundary: bool = False):
    """auto_boundary defaults to False here even though processing itself
    defaults it to True (ProcessParams.auto_display_boundary): these tests
    are about the *user-drawn* boundary, and an auto boundary already
    applied on top would mean "no boundary set" never actually means an
    unclipped grid. See test_auto_boundary.py for the automatic path."""
    client = TestClient(app)
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(
        f"/api/projects/{project_id}/process",
        json={
            "line_params": {},
            "diurnal_params": {},
            "heading_correction": {},
            "auto_display_boundary": auto_boundary,
        },
    )
    assert r.status_code == 200, r.text
    return project_store.get(project_id)


def test_display_boundary_clips_grid_to_polygon_and_clears():
    project = _make_processed_project()
    req = GridRequest(value="anomaly", cell_size_m=5.0)

    project.get_grid_overlay(req)
    full_finite = int(np.isfinite(project.last_overlay.values).sum())
    assert full_finite > 0

    df = project.processed
    lat_min, lat_max = float(df["lat"].min()), float(df["lat"].max())
    lon_min, lon_max = float(df["lon"].min()), float(df["lon"].max())
    lon_mid = (lon_min + lon_max) / 2.0
    # only the western half of the survey's bounding box
    boundary = [[lat_min, lon_min], [lat_min, lon_mid], [lat_max, lon_mid], [lat_max, lon_min]]

    project.set_display_boundary(DisplayBoundaryRequest(polygon=boundary))
    project.get_grid_overlay(req)
    half_finite = int(np.isfinite(project.last_overlay.values).sum())
    assert 0 < half_finite < full_finite

    # clearing (polygon=None) restores the unclipped extent
    project.set_display_boundary(DisplayBoundaryRequest(polygon=None))
    project.get_grid_overlay(req)
    restored_finite = int(np.isfinite(project.last_overlay.values).sum())
    assert restored_finite == full_finite


def test_display_boundary_rejects_short_polygon():
    project = _make_processed_project()
    with pytest.raises(ProjectError):
        project.set_display_boundary(DisplayBoundaryRequest(polygon=[[35.0, 127.0], [35.1, 127.1]]))


def test_display_boundary_does_not_invalidate_grid_cache():
    project = _make_processed_project()
    req = GridRequest(value="anomaly", cell_size_m=5.0)
    project.get_grid_overlay(req)
    assert project.grid_cache  # populated by the call above

    df = project.processed
    lat_min, lat_max = float(df["lat"].min()), float(df["lat"].max())
    lon_min, lon_max = float(df["lon"].min()), float(df["lon"].max())
    boundary = [[lat_min, lon_min], [lat_min, lon_max], [lat_max, lon_max], [lat_max, lon_min]]

    cache_before = dict(project.grid_cache)
    project.set_display_boundary(DisplayBoundaryRequest(polygon=boundary))
    # setting the boundary is a cheap render-time mask, not a regrid - the
    # existing cached grid entries must survive untouched.
    assert project.grid_cache == cache_before


def test_display_boundary_via_api_endpoint():
    client = TestClient(app)
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    client.post(f"/api/projects/{project_id}/process", json={"line_params": {}, "diurnal_params": {}, "heading_correction": {}})

    boundary = [[35.0, 127.0], [35.1, 127.0], [35.1, 127.1], [35.0, 127.1]]
    r = client.post(f"/api/projects/{project_id}/display-boundary", json={"polygon": boundary})
    assert r.status_code == 200, r.text
    assert r.json()["display_boundary_polygon"] == boundary

    r = client.post(f"/api/projects/{project_id}/display-boundary", json={"polygon": None})
    assert r.status_code == 200, r.text
    assert r.json()["display_boundary_polygon"] is None
