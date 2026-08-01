"""Tests for near-surface compact-target detection (mines/ordnance/hidden
vehicles): the dipole-fit physics/candidate-picker module in isolation
(synthetic ground truth, no network), and the Project/API/report/chat
wiring against the real sample survey data.
"""
import io
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi.testclient import TestClient

from app.chat import _execute_tool
from app.main import app
from app.models import TargetDetectionRequest
from app.processing.dipole_fit import MU0, detect_targets, field_direction_enu
from app.processing.gridding import grid_points
from app.store import store as project_store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def _synthetic_dipole_grid(targets_true, inclination_deg=55.0, declination_deg=-8.0, seed=1):
    rng = np.random.default_rng(seed)
    lines_y = np.arange(0, 40, 2.0)
    along = np.arange(0, 40, 0.3)
    xs, ys = [], []
    for ly in lines_y:
        xs.append(along)
        ys.append(np.full_like(along, ly))
    x = np.concatenate(xs)
    y = np.concatenate(ys)

    f_hat = field_direction_enu(inclination_deg, declination_deg)
    values = np.zeros_like(x)
    for x0, y0, depth, m in targets_true:
        rx, ry = x - x0, y - y0
        rz = np.full_like(rx, depth)
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        dot = (f_hat[0] * rx + f_hat[1] * ry + f_hat[2] * rz) / r
        values = values + (MU0 * m / (4 * np.pi * r**3)) * (3 * dot**2 - 1) * 1e9

    regional_trend = 0.05 * x + 0.02 * y  # broad background - must NOT be picked up as a target
    noise = rng.normal(0, 0.5, size=x.shape)
    values = values + regional_trend + noise + 5.0
    return x, y, values, inclination_deg, declination_deg


def test_dipole_fit_recovers_known_synthetic_targets():
    targets_true = [(10.0, 15.0, 0.8, 2.0), (25.0, 20.0, 2.0, 80.0)]
    x, y, values, incl, decl = _synthetic_dipole_grid(targets_true)
    grid = grid_points(x, y, values, cell_size_m=0.5, method="nearest", max_distance_m=1.5)

    found = detect_targets(
        x, y, values, grid.values, grid.easting, grid.northing, grid.cell_size_m,
        incl, decl,
        threshold_nt=3.0, min_footprint_m=0.3, max_footprint_m=15.0,
        fit_window_m=5.0, max_depth_m=5.0, min_fit_quality=0.2,
    )
    assert len(found) == 2

    found_by_proximity = sorted(found, key=lambda t: min(np.hypot(t.x - tx, t.y - ty) for tx, ty, _, _ in targets_true))
    for t in found_by_proximity:
        best = min(targets_true, key=lambda tt: np.hypot(t.x - tt[0], t.y - tt[1]))
        x0, y0, depth, m = best
        assert abs(t.x - x0) < 0.5
        assert abs(t.y - y0) < 0.5
        assert abs(t.depth_m - depth) < 0.3
        assert abs(t.moment_am2 - m) / m < 0.1
        assert t.fit_quality > 0.9
    print("SYNTHETIC TARGET RECOVERY CHECKS PASSED")


def test_dipole_fit_rejects_regional_trend_and_pure_noise():
    x, y, values, incl, decl = _synthetic_dipole_grid(targets_true=[], seed=2)
    grid = grid_points(x, y, values, cell_size_m=0.5, method="nearest", max_distance_m=1.5)
    found = detect_targets(
        x, y, values, grid.values, grid.easting, grid.northing, grid.cell_size_m,
        incl, decl,
        threshold_nt=3.0, min_footprint_m=0.3, max_footprint_m=15.0,
        fit_window_m=5.0, max_depth_m=5.0, min_fit_quality=0.2,
    )
    assert found == []
    print("NO-FALSE-POSITIVE CHECKS PASSED")


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


def test_run_target_detection_via_api_on_real_sample_data():
    client, project_id = _make_processed_project()
    # Real sample survey has ~100m line spacing (regional mineral-exploration
    # scale, not a dedicated fine mine-detection survey), so this is a smoke
    # test of the wiring/shape, not an expectation of finding real targets.
    r = client.post(
        f"/api/projects/{project_id}/target-detection",
        json=TargetDetectionRequest(cell_size_m=5.0, fit_window_m=20.0, max_depth_m=10.0).model_dump(),
    )
    assert r.status_code == 200, r.text
    resp = r.json()
    assert "n_targets" in resp and "targets" in resp and "amplitude_threshold_nt" in resp
    assert resp["n_targets"] == len(resp["targets"])

    project = project_store.get(project_id)
    assert project.target_summary_cache == resp

    report = project.generate_report()
    assert "표적탐지" in report
    print("REAL-DATA TARGET DETECTION API CHECKS PASSED")


def test_target_detection_requires_processed_data():
    client = TestClient(app)
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    r = client.post(f"/api/projects/{project_id}/target-detection", json=TargetDetectionRequest().model_dump())
    assert r.status_code == 400, r.text
    print("UNPROCESSED-PROJECT ERROR PATH OK")


def test_target_detection_rejects_absurdly_fine_cell_size_on_large_survey():
    client, project_id = _make_processed_project()
    # The sample survey spans several km; a 0.01m cell size there would
    # blow up into a many-hundred-million-cell grid, so this should be
    # rejected up front instead of hanging/exhausting memory.
    r = client.post(
        f"/api/projects/{project_id}/target-detection",
        json=TargetDetectionRequest(cell_size_m=0.01).model_dump(),
    )
    assert r.status_code == 400, r.text
    assert "촘촘" in r.json()["detail"]
    print("GRID-SIZE-CAP ERROR PATH OK")


def test_chat_tool_get_target_detection_summary():
    client, project_id = _make_processed_project()
    project = project_store.get(project_id)

    out = json.loads(_execute_tool(project, "get_target_detection_summary", {}))
    assert "error" in out

    project.run_target_detection(TargetDetectionRequest(cell_size_m=5.0, fit_window_m=20.0, max_depth_m=10.0))
    out = json.loads(_execute_tool(project, "get_target_detection_summary", {}))
    assert "error" not in out
    assert "n_targets" in out and "targets" in out
    print("CHAT TARGET DETECTION TOOL CHECKS PASSED")


if __name__ == "__main__":
    test_dipole_fit_recovers_known_synthetic_targets()
    test_dipole_fit_rejects_regional_trend_and_pure_noise()
    test_run_target_detection_via_api_on_real_sample_data()
    test_target_detection_requires_processed_data()
    test_target_detection_rejects_absurdly_fine_cell_size_on_large_survey()
    test_chat_tool_get_target_detection_summary()
    print("\nALL CHECKS PASSED")
