"""Click-to-inspect, the legend colour bar, the line profile, and the
magnetization judgement on the targets."""
from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.processing.targets import pick_targets
from conftest import R3_FIXTURE
from test_targets import DEC, INC, PROBE_H, _survey


def _loaded():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(R3_FIXTURE, "rb") as f:
        client.post(f"/api/projects/{pid}/upload", files={"files": (R3_FIXTURE.name, f, "text/plain")})
    return client, pid


def test_click_reports_the_shown_grid_cell_and_the_nearest_reading():
    client, pid = _loaded()
    g = client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.2, "kind": "thd"}).json()
    assert g["overlay"]["colorbar_data_url"].startswith("data:image/png;base64,")
    assert len(g["overlay"]["legend_ticks"]) >= 2
    (south, west), (north, east) = g["overlay"]["bounds"]
    r = client.post(f"/api/projects/{pid}/sample", json={"lat": (south + north) / 2, "lon": (west + east) / 2})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["grid_kind"] == "thd" and s["in_grid"] is True
    assert s["nearest"]["distance_m"] < 5 and s["nearest"]["sensor"] in (1, 2)
    outside = client.post(f"/api/projects/{pid}/sample", json={"lat": south - 0.01, "lon": west}).json()
    assert outside["in_grid"] is False and outside["grid_value"] is None


def test_line_profile_before_and_after_preprocessing():
    client, pid = _loaded()
    p = client.get(f"/api/projects/{pid}/line-profile", params={"file_index": 0, "line_id": 10}).json()
    assert p["tracks"] == [10, 11] and {s["sensor"] for s in p["series"]} == {1, 2}
    s1 = next(s for s in p["series"] if s["sensor"] == 1)
    assert s1["raw_value"] is None and len(s1["value"]) == len(s1["distance_m"])
    assert s1["distance_m"][-1] > 20                           # a 27 m track, thinned
    client.post(f"/api/projects/{pid}/preprocess", json={"lpf_hz": 2.0})
    p2 = client.get(f"/api/projects/{pid}/line-profile", params={"file_index": 0, "line_id": 10, "sensor": 1}).json()
    assert p2["processed"] is True and len(p2["series"]) == 1
    assert p2["series"][0]["raw_value"] is not None and p2["series"][0]["excluded"] is not None
    missing = client.get(f"/api/projects/{pid}/line-profile", params={"file_index": 0, "line_id": 99})
    assert missing.status_code == 400


def test_induced_synthetic_dipoles_are_judged_induced():
    df = _survey(1)
    r = pick_targets(df, INC, DEC, 32652, noise_by_file={0: 3.0}, sensor_height_m=PROBE_H)
    judged = [t["magnetization"] for t in r["targets"] if t["magnetization"]]
    assert judged
    assert any(m["verdict"] == "induced" for m in judged)
    assert all(m["angle_from_induced_deg"] < 45 for m in judged if m["verdict"] == "induced")
    off = pick_targets(df, INC, DEC, 32652, noise_by_file={0: 3.0}, sensor_height_m=PROBE_H, magnetization=False)
    assert all(t["magnetization"] is None for t in off["targets"])
