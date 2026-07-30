"""End-to-end smoke test of the full API against the real sample survey.

Not a pytest suite (no pytest in requirements) - run directly:
    python3 tests/test_pipeline_e2e.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def main():
    client = TestClient(app)

    r = client.post("/api/projects")
    assert r.status_code == 200, r.text
    project_id = r.json()["project_id"]
    print("project_id:", project_id)

    with open(FIXTURES / "sample_drone_survey.csv", "rb") as f:
        r = client.post(f"/api/projects/{project_id}/upload/drone", files={"file": ("drone.csv", f, "text/csv")})
    assert r.status_code == 200, r.text
    print("drone summary:", r.json())

    with open(FIXTURES / "sample_base_station.csv", "rb") as f:
        r = client.post(f"/api/projects/{project_id}/upload/base", files={"file": ("base.csv", f, "text/csv")})
    assert r.status_code == 200, r.text
    print("base summary:", r.json())

    r = client.post(
        f"/api/projects/{project_id}/process",
        json={
            "filter_cutoff_hz": 1.0,
            "line_params": {},
            "diurnal_params": {"time_offset_seconds": 0.0, "reference": "mean"},
        },
    )
    assert r.status_code == 200, r.text
    summary = r.json()
    print("process summary:", {k: v for k, v in summary.items() if k != "lines"})
    print("n lines:", len(summary["lines"]))
    assert summary["n_lines"] >= 1
    assert summary["diurnal"]["has_overlap"] is False  # known clock mismatch in sample data

    r = client.get(f"/api/projects/{project_id}/points", params={"value": "anomaly"})
    assert r.status_code == 200, r.text
    pts = r.json()
    print("n points returned:", len(pts))
    assert len(pts) == summary["n_points"]
    n_excluded = sum(1 for p in pts if p["excluded"])
    assert n_excluded == summary["n_points"] - summary["n_kept"]

    # manual exclude by line id
    first_line = summary["lines"][0]["line_id"]
    r = client.post(
        f"/api/projects/{project_id}/manual-exclude",
        json={"mode": "lines", "action": "exclude", "line_ids": [first_line]},
    )
    assert r.status_code == 200, r.text
    summary2 = r.json()
    print("after manual exclude, n_kept:", summary2["n_kept"], "n_excluded_manual:", summary2["n_excluded_manual"])
    assert summary2["n_kept"] < summary["n_kept"]

    # manual polygon exclude (small box around the survey centroid)
    lats = [p["lat"] for p in pts]
    lons = [p["lon"] for p in pts]
    clat, clon = sum(lats) / len(lats), sum(lons) / len(lons)
    d = 0.001
    polygon = [[clat - d, clon - d], [clat - d, clon + d], [clat + d, clon + d], [clat + d, clon - d]]
    r = client.post(
        f"/api/projects/{project_id}/manual-exclude",
        json={"mode": "polygon", "action": "exclude", "polygon": polygon},
    )
    assert r.status_code == 200, r.text
    print("after polygon exclude, n_kept:", r.json()["n_kept"])

    # undo the line exclusion to leave enough points for gridding
    r = client.post(
        f"/api/projects/{project_id}/manual-exclude",
        json={"mode": "lines", "action": "include", "line_ids": [first_line]},
    )
    assert r.status_code == 200, r.text

    r = client.post(f"/api/projects/{project_id}/grid", json={"value": "anomaly", "cell_size_m": 10.0})
    assert r.status_code == 200, r.text
    grid_resp = r.json()
    print("grid overlay bounds:", grid_resp["bounds"], "stats:", grid_resp["stats"])
    assert grid_resp["image_data_url"].startswith("data:image/png;base64,")

    for transform_name in ["rtp", "rte", "1vd", "as"]:
        r = client.post(
            f"/api/projects/{project_id}/transform",
            json={"transform": transform_name, "value": "anomaly", "cell_size_m": 10.0},
        )
        assert r.status_code == 200, r.text
        tr = r.json()
        print(f"{transform_name} stats:", tr["stats"])
        assert tr["image_data_url"].startswith("data:image/png;base64,")

    # error path: unknown project id
    r = client.get("/api/projects/does-not-exist/summary")
    assert r.status_code == 400, r.text

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
