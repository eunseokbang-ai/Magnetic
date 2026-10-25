"""MineMag's API skeleton: create a project, upload R3 exports, read the
summary and the thinned points for the map."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from conftest import R3_FIXTURE


def _client_and_project():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    return client, pid


def test_upload_returns_the_summary_and_the_points_follow():
    client, pid = _client_and_project()
    with open(R3_FIXTURE, "rb") as f:
        r = client.post(f"/api/projects/{pid}/upload", files={"files": (R3_FIXTURE.name, f, "text/plain")})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["sensors"] == [1, 2] and s["n_tracks"] == 2 and s["utm_epsg"] == 32652
    assert s["files"][0]["start_date"] == "2026-04-25"

    assert client.get(f"/api/projects/{pid}/summary").json() == s

    pts = client.get(f"/api/projects/{pid}/points", params={"sensor": 1, "max_points": 500}).json()
    assert pts["n_total"] == s["per_sensor"]["1"]["n"]
    assert len(pts["lat"]) <= 500 and set(pts["sensor"]) == {1}
    assert pts["stride"] >= 2


def test_version_and_errors():
    client, pid = _client_and_project()
    assert client.get("/api/version").json()["product"] == "MineMag"
    assert client.get(f"/api/projects/{pid}/summary").status_code == 400          # nothing uploaded yet
    assert client.get("/api/projects/nope/summary").status_code == 404
    r = client.post(f"/api/projects/{pid}/upload", files={"files": ("bad.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400
    assert "헤더" in r.json()["detail"]
