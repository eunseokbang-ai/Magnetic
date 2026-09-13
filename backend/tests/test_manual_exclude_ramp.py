"""Verifies that takeoff/landing ramp points (exclusion_reason
takeoff_ramp/landing_ramp - see processing/lines.py) are protected from
manual exclude/include drawing by default, and only affected when the
request explicitly opts in via include_ramp=True (the line editor's
"show ramp points" toggle) - see store.py::set_manual_exclude."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _make_processed_project():
    client = TestClient(app)
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]

    drone_path = FIXTURES / "sample_drone_survey.csv"
    base_path = FIXTURES / "sample_base_station.csv"
    client.post(f"/api/projects/{project_id}/upload/drone", files=[("files", (drone_path.name, open(drone_path, "rb"), "text/csv"))])
    client.post(f"/api/projects/{project_id}/upload/base", files=[("files", (base_path.name, open(base_path, "rb"), "text/csv"))])
    r = client.post(f"/api/projects/{project_id}/process", json={"line_params": {}})
    assert r.status_code == 200, r.text
    return client, project_id


def test_ramp_points_are_protected_from_manual_include_by_default():
    client, project_id = _make_processed_project()

    r = client.get(f"/api/projects/{project_id}/points", params={"value": "anomaly"})
    assert r.status_code == 200, r.text
    points = r.json()

    ramp_excluded = [p for p in points if p["is_ramp"] and p["excluded"]]
    assert ramp_excluded, "expected the real sample survey to have takeoff/landing ramp points"
    target = ramp_excluded[0]["point_id"]

    # default (include_ramp not set -> False): the ramp point must stay excluded
    r = client.post(
        f"/api/projects/{project_id}/manual-exclude",
        json={"mode": "point_ids", "action": "include", "point_ids": [target]},
    )
    assert r.status_code == 200, r.text
    exclusion = r.json()["exclusion"]
    idx = exclusion["point_id"].index(target)
    assert exclusion["excluded"][idx] is True, "a ramp point should not be includable without include_ramp=True"

    # with include_ramp=True, the same request should actually flip it
    r = client.post(
        f"/api/projects/{project_id}/manual-exclude",
        json={"mode": "point_ids", "action": "include", "point_ids": [target], "include_ramp": True},
    )
    assert r.status_code == 200, r.text
    exclusion = r.json()["exclusion"]
    idx = exclusion["point_id"].index(target)
    assert exclusion["excluded"][idx] is False, "include_ramp=True should allow restoring a ramp point"


def test_ramp_points_are_protected_from_polygon_exclude_by_default():
    client, project_id = _make_processed_project()

    r = client.get(f"/api/projects/{project_id}/points", params={"value": "anomaly"})
    points = r.json()
    ramp_kept_candidates = [p for p in points if p["is_ramp"]]
    assert ramp_kept_candidates

    # a polygon covering the whole survey extent (so it definitely covers
    # every ramp point too) - only the *not-ramp* excluded state should be
    # affected by an "exclude everything" draw when include_ramp is False.
    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]
    pad = 0.01
    polygon = [
        [min(lats) - pad, min(lons) - pad],
        [min(lats) - pad, max(lons) + pad],
        [max(lats) + pad, max(lons) + pad],
        [max(lats) + pad, min(lons) - pad],
    ]

    r = client.post(
        f"/api/projects/{project_id}/manual-exclude",
        json={"mode": "polygon", "action": "include", "polygon": polygon},
    )
    assert r.status_code == 200, r.text
    exclusion = r.json()["exclusion"]
    excluded_by_id = dict(zip(exclusion["point_id"], exclusion["excluded"]))

    ramp_ids = {p["point_id"] for p in ramp_kept_candidates}
    assert all(excluded_by_id[pid] for pid in ramp_ids), "ramp points must remain excluded despite an include-all draw"


if __name__ == "__main__":
    test_ramp_points_are_protected_from_manual_include_by_default()
    test_ramp_points_are_protected_from_polygon_exclude_by_default()
    print("ALL CHECKS PASSED")
