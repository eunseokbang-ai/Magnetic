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


def _multi_files(field: str, *paths: pathlib.Path):
    return [(field, (p.name, open(p, "rb"), "text/csv")) for p in paths]


def main():
    client = TestClient(app)

    r = client.post("/api/projects")
    assert r.status_code == 200, r.text
    project_id = r.json()["project_id"]
    print("project_id:", project_id)

    drone_path = FIXTURES / "sample_drone_survey.csv"
    base_path = FIXTURES / "sample_base_station.csv"

    # multi-file upload: same fixture supplied twice, should double the row count
    r = client.post(f"/api/projects/{project_id}/upload/drone", files=_multi_files("files", drone_path, drone_path))
    assert r.status_code == 200, r.text
    drone_summary_multi = r.json()
    print("drone summary (2 files):", drone_summary_multi)
    assert drone_summary_multi["n_points"] == 2 * 22025, drone_summary_multi["n_points"]

    r = client.post(f"/api/projects/{project_id}/upload/base", files=_multi_files("files", base_path, base_path))
    assert r.status_code == 200, r.text
    base_summary_multi = r.json()
    print("base summary (2 files):", base_summary_multi)
    assert base_summary_multi["n_points"] == 2 * 20314, base_summary_multi["n_points"]

    # re-upload single files for the rest of the pipeline (multi-file loading
    # already proven above; a single real flight keeps line detection sane)
    r = client.post(f"/api/projects/{project_id}/upload/drone", files=_multi_files("files", drone_path))
    assert r.status_code == 200, r.text
    print("drone summary (1 file):", r.json())

    r = client.post(f"/api/projects/{project_id}/upload/base", files=_multi_files("files", base_path))
    assert r.status_code == 200, r.text
    print("base summary (1 file):", r.json())

    r = client.post(
        f"/api/projects/{project_id}/process",
        json={
            "filter_cutoff_hz": 1.0,
            "line_params": {},
            "diurnal_params": {"time_offset_seconds": 0.0, "reference": "mean"},
            "heading_correction": {"enabled": True, "quiet_percentile": 40.0},
        },
    )
    assert r.status_code == 200, r.text
    summary = r.json()
    print("process summary:", {k: v for k, v in summary.items() if k != "lines"})
    print("lines:", summary["lines"])
    assert summary["n_lines"] >= 1
    assert summary["diurnal"]["has_overlap"] is False  # known clock mismatch in sample data
    assert summary["line_spacing_m"] is not None and summary["line_spacing_m"] > 0
    assert summary["heading_correction"]["applied"] is True, summary["heading_correction"]
    assert summary["heading_correction"]["offset_nt"] is not None
    assert all(l["heading_group"] in ("A", "B") for l in summary["lines"])
    assert all(l["heading_shift_nt"] is not None for l in summary["lines"])

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
    print("after manual exclude, n_kept:", summary2["n_kept"], "n_manual_excluded:", summary2["n_manual_excluded"])
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

    # force-include a batch of auto-excluded points (e.g. a turbulence
    # segment the auto line-detector correctly dropped but the user wants
    # back) - pick a polygon around some points with line_id == -1
    excluded_pts = [p for p in pts if p["excluded"] and p["line_id"] < 0]
    assert excluded_pts, "expected some auto-excluded points to restore"
    target = excluded_pts[len(excluded_pts) // 2]
    d2 = 0.0005
    restore_polygon = [
        [target["lat"] - d2, target["lon"] - d2],
        [target["lat"] - d2, target["lon"] + d2],
        [target["lat"] + d2, target["lon"] + d2],
        [target["lat"] + d2, target["lon"] - d2],
    ]
    before_kept = r.json()["n_kept"]
    r = client.post(
        f"/api/projects/{project_id}/manual-exclude",
        json={"mode": "polygon", "action": "include", "polygon": restore_polygon},
    )
    assert r.status_code == 200, r.text
    summary_restored = r.json()
    print("after force-include, n_kept:", summary_restored["n_kept"], "n_manual_included:", summary_restored["n_manual_included"])
    assert summary_restored["n_kept"] > before_kept, "force-including auto-excluded points should raise n_kept"
    assert summary_restored["n_manual_included"] > 0

    # full reset should return to the original automatic n_kept
    r = client.post(f"/api/projects/{project_id}/manual-exclude", json={"mode": "reset"})
    assert r.status_code == 200, r.text
    summary_reset = r.json()
    print("after reset, n_kept:", summary_reset["n_kept"])
    assert summary_reset["n_kept"] == summary["n_kept"]
    assert summary_reset["n_manual_included"] == 0 and summary_reset["n_manual_excluded"] == 0

    for method in ["nearest", "spline", "linear", "cubic"]:
        r = client.post(f"/api/projects/{project_id}/grid", json={"value": "anomaly", "cell_size_m": 10.0, "method": method})
        assert r.status_code == 200, r.text
        grid_resp = r.json()
        print(f"grid[{method}] bounds:", grid_resp["bounds"], "stats:", grid_resp["stats"])
        assert grid_resp["image_data_url"].startswith("data:image/png;base64,")

    # manual vmin/vmax should be honored as-is (not overridden by the
    # symmetric-around-zero auto-expansion for anomaly grids)
    r = client.post(
        f"/api/projects/{project_id}/grid",
        json={"value": "anomaly", "cell_size_m": 10.0, "method": "nearest", "vmin": 1000.0, "vmax": 2000.0},
    )
    assert r.status_code == 200, r.text
    manual_range_resp = r.json()
    print("manual vmin/vmax grid response range:", manual_range_resp["vmin"], manual_range_resp["vmax"])
    assert manual_range_resp["vmin"] == 1000.0 and manual_range_resp["vmax"] == 2000.0

    # colormap selection should be honored
    r = client.post(
        f"/api/projects/{project_id}/grid",
        json={"value": "anomaly", "cell_size_m": 10.0, "method": "nearest", "cmap": "turbo"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["cmap"] == "turbo"

    # confirm the auto max_distance (line-spacing based) fills far more of
    # the block than the old fixed 2*cell_size default would have
    from app.store import store as project_store

    project = project_store.get(project_id)
    auto_grid = project._grid_for("anomaly", 10.0, "spline", None)
    tight_grid = project._grid_for("anomaly", 10.0, "spline", 20.0)
    import numpy as np

    auto_nan_pct = float(np.isnan(auto_grid.values).mean())
    tight_nan_pct = float(np.isnan(tight_grid.values).mean())
    print(f"grid NaN%% auto={auto_nan_pct:.3f} vs old-style tight={tight_nan_pct:.3f}")
    assert auto_nan_pct < tight_nan_pct - 0.2, "line-spacing-based mask should fill noticeably more of the block"

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
