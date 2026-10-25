"""The area of interest: readings outside the polygon are left out of
everything downstream, and clearing it brings them back."""
from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from conftest import R3_FIXTURE


def test_area_clips_qc_preprocessing_grid_and_targets():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(R3_FIXTURE, "rb") as f:
        s = client.post(f"/api/projects/{pid}/upload", files={"files": (R3_FIXTURE.name, f, "text/plain")}).json()
    pts = client.get(f"/api/projects/{pid}/points", params={"max_points": 5000}).json()
    lat, lon = np.array(pts["lat"]), np.array(pts["lon"])
    assert pts["in_area"] is None
    # the southern half of the two tracks
    mid = float(np.median(lat))
    poly = [[lat.min() - 1e-5, lon.min() - 1e-4], [lat.min() - 1e-5, lon.max() + 1e-4], [mid, lon.max() + 1e-4], [mid, lon.min() - 1e-4]]
    whole = client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.2}).json()["stats"]["n_points"]

    a = client.put(f"/api/projects/{pid}/area", json={"polygon": poly}).json()
    assert a["n_total"] == s["n_rows"] and 0.3 < a["n_inside"] / a["n_total"] < 0.7
    assert client.get(f"/api/projects/{pid}/area").json()["polygon"] == poly
    pts2 = client.get(f"/api/projects/{pid}/points", params={"max_points": 5000}).json()
    assert pts2["in_area"] is not None and 0.3 < np.mean(pts2["in_area"]) < 0.7

    clipped = client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.2}).json()["stats"]["n_points"]
    assert clipped < whole and abs(clipped - a["n_inside"]) <= 2          # on-line rows only; the fixture has no turns
    qc = client.post(f"/api/projects/{pid}/qc", json={}).json()
    assert qc["files"][0]["n_paired_samples"] < s["n_rows"] / 2
    pre = client.post(f"/api/projects/{pid}/preprocess", json={"lpf_hz": 2.0}).json()
    assert pre["files"][0]["n_in"] == a["n_inside"]
    t = client.post(f"/api/projects/{pid}/targets", json={"sensor_height_m": 0.3}).json()
    assert all(lat.min() <= tt["lat"] <= mid + 1e-4 for tt in t["targets"])

    assert client.delete(f"/api/projects/{pid}/area").json()["polygon"] is None
    assert client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.2}).json()["stats"]["n_points"] == whole


def test_an_empty_area_is_refused():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(R3_FIXTURE, "rb") as f:
        client.post(f"/api/projects/{pid}/upload", files={"files": (R3_FIXTURE.name, f, "text/plain")})
    r = client.put(f"/api/projects/{pid}/area", json={"polygon": [[0, 0], [0, 1], [1, 1]]})
    assert r.status_code == 400 and "영역" in r.json()["detail"]
    assert client.get(f"/api/projects/{pid}/area").json()["polygon"] is None       # the old (no) area is kept
    assert client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.2}).status_code == 200
    assert client.put(f"/api/projects/{pid}/area", json={"polygon": [[0, 0]]}).status_code == 422
