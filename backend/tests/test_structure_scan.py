"""Integration tests for the structure-distortion auto-scan feature
(store.py::scan_structure_distortion, POST /structure-scan) and the new
"polygons" batch mode on the existing manual-smoothing endpoint that lets
its results be applied in one call instead of one polygon at a time.
Overpass is mocked throughout - see test_osm_structures.py for why.
"""
import pathlib
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.store import store as project_store

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


def _mock_overpass_response(building_center):
    lat0, lon0 = building_center
    d = 0.0002
    return MagicMock(
        status_code=200,
        json=lambda: {
            "elements": [
                {
                    "type": "way",
                    "tags": {"building": "yes"},
                    "geometry": [
                        {"lat": lat0 - d, "lon": lon0 - d},
                        {"lat": lat0 - d, "lon": lon0 + d},
                        {"lat": lat0 + d, "lon": lon0 + d},
                        {"lat": lat0 + d, "lon": lon0 - d},
                        {"lat": lat0 - d, "lon": lon0 - d},
                    ],
                }
            ]
        },
    )


def test_structure_scan_endpoint_returns_buildings_and_anomalies():
    client, project_id = _make_processed_project()
    project = project_store.get(project_id)
    df = project.processed
    center = (float(df["lat"].median()), float(df["lon"].median()))

    with patch("app.processing.osm_structures.requests.post", return_value=_mock_overpass_response(center)):
        r = client.post(f"/api/projects/{project_id}/structure-scan", json={})
    assert r.status_code == 200, r.text
    resp = r.json()
    assert resp["n_buildings"] == 1
    assert resp["n_structure_polygons"] == 1
    assert "anomalies" in resp and "n_anomalies" in resp
    assert resp["n_anomalies"] == len(resp["anomalies"])
    assert resp["n_matched"] <= resp["n_anomalies"]


def test_structure_scan_endpoint_requires_processed_data():
    client = TestClient(app)
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    r = client.post(f"/api/projects/{project_id}/structure-scan", json={})
    assert r.status_code != 200


def test_structure_scan_propagates_network_failure_as_clear_error():
    client, project_id = _make_processed_project()
    import requests

    with patch("app.processing.osm_structures.requests.post", side_effect=requests.ConnectionError("blocked")):
        r = client.post(f"/api/projects/{project_id}/structure-scan", json={})
    assert r.status_code != 200
    assert "네트워크" in r.json()["detail"] or "OpenStreetMap" in r.json()["detail"]


def test_smoothing_polygons_mode_unions_multiple_regions_in_one_call():
    client, project_id = _make_processed_project()
    project = project_store.get(project_id)
    df = project.processed_base
    active = project._active_mask()
    pts = df.loc[active].sort_values("lat").reset_index(drop=True)
    assert len(pts) > 20

    def _box_around(row, eps=0.00005):
        lat, lon = float(row["lat"]), float(row["lon"])
        return [[lat - eps, lon - eps], [lat - eps, lon + eps], [lat + eps, lon + eps], [lat + eps, lon - eps]]

    poly_a = _box_around(pts.iloc[5])
    poly_b = _box_around(pts.iloc[-5])

    r = client.post(f"/api/projects/{project_id}/smooth", json={"mode": "polygons", "polygons": [poly_a, poly_b]})
    assert r.status_code == 200, r.text
    resp = r.json()
    smoothed_ids = set(resp["manual_smooth_point_ids"])
    assert len(smoothed_ids) >= 2  # at least the two boxed points themselves

    # equivalent to applying each polygon separately and taking the union
    r2 = client.post(f"/api/projects/{project_id}/smooth", json={"mode": "reset"})
    assert r2.status_code == 200
    r3 = client.post(f"/api/projects/{project_id}/smooth", json={"mode": "polygon", "polygon": poly_a})
    r4 = client.post(f"/api/projects/{project_id}/smooth", json={"mode": "polygon", "polygon": poly_b})
    separate_ids = set(r4.json()["manual_smooth_point_ids"])
    assert smoothed_ids == separate_ids


def test_smoothing_polygons_mode_requires_polygons_field():
    client, project_id = _make_processed_project()
    r = client.post(f"/api/projects/{project_id}/smooth", json={"mode": "polygons"})
    assert r.status_code != 200
