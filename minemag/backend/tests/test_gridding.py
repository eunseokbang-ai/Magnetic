"""Grids and the comparison of repeated runs, on the real fixture and
through the API."""
from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.processing.gridding import grid_survey, repeat_comparison
from app.store import Project
from conftest import R3_FIXTURE


def _project() -> Project:
    p = Project(id="t")
    p.load_files([R3_FIXTURE.read_bytes()], [R3_FIXTURE.name])
    return p


def test_grid_covers_the_tracks_and_leaves_the_rest_empty():
    p = _project()
    grid, stats = grid_survey(p.raw, cell_size_m=0.1)
    assert grid.cell_size_m == 0.1
    # two tracks 0.5 m apart with probes 1 m apart fill their own 1.5 m
    # strip at the default reach; a tight reach leaves the gaps between
    # the four probe lines empty
    assert stats["coverage_pct"] == 100.0
    _, tight = grid_survey(p.raw, cell_size_m=0.1, max_distance_m=0.12)
    assert 0 < tight["coverage_pct"] < 90
    assert stats["n_points"] == len(p.raw)
    assert stats["rms_nt"] > 10                       # the targets near track 10/11
    # the probe offset was levelled out: both probes alone centre near zero
    _, s1 = grid_survey(p.raw, sensors=[1])
    _, s2 = grid_survey(p.raw, sensors=[2])
    assert s1["sensors"] == [1] and s2["sensors"] == [2]
    assert abs(np.nanmedian(grid.values)) < 20


def test_repeat_comparison_of_a_file_with_itself_is_perfect_and_a_single_file_is_refused():
    p = _project()
    assert repeat_comparison(p.raw)["pairs"] == []
    two = Project(id="two")
    data = R3_FIXTURE.read_bytes()
    two.load_files([data, data], ["20260425_001316_MD-R3_#0115_a.asc", "20260426_001316_MD-R3_#0115_b.asc"])
    r = repeat_comparison(two.raw, compare_cell_m=0.2)
    assert len(r["pairs"]) == 1
    pair = r["pairs"][0]
    assert pair["correlation"] > 0.999
    assert pair["diff_rms_nt"] < 0.01
    assert "거의 같습니다" in r["note"]


def test_the_api_returns_the_overlay_statistics_qc_and_repeat():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    data = R3_FIXTURE.read_bytes()
    r = client.post(f"/api/projects/{pid}/upload", files=[
        ("files", ("20260425_001316_MD-R3_#0115_a.asc", data, "text/plain")),
        ("files", ("20260426_001316_MD-R3_#0115_b.asc", data, "text/plain")),
    ])
    assert r.status_code == 200, r.text

    g = client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.2}).json()
    assert g["overlay"]["image_data_url"].startswith("data:image/png;base64,")
    assert len(g["overlay"]["topleft"]) == 2 and g["stats"]["coverage_pct"] > 0

    qc = client.post(f"/api/projects/{pid}/qc", json={}).json()
    assert len(qc["files"]) == 2 and qc["files"][0]["per_probe"]["1"]["quiet_rms_nt"] is not None

    rep = client.post(f"/api/projects/{pid}/repeat", json={"compare_cell_m": 0.2}).json()
    assert rep["pairs"][0]["correlation"] > 0.999

    bad = client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.2, "file_index": 7})
    assert bad.status_code == 400
