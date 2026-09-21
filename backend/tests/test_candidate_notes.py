"""The log of what the operator decided about each anomaly candidate.

The point of keeping it: removing a structure is a judgement call made
against an orthophoto, and three weeks later - or in front of a client -
"why is there no anomaly here?" needs an answer better than memory. What
has to hold is that a decision survives a re-scan that renumbers the
candidates, survives reprocessing, comes back with a saved project, and
comes out as a table.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.store import Project
from app.store import store as project_store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


@pytest.fixture
def project():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    assert client.post(f"/api/projects/{pid}/process", json={"line_params": {}, "diurnal_params": {}}).status_code == 200
    return client, pid, project_store.get(pid)


def _a_candidate(client, pid):
    r = client.post(f"/api/projects/{pid}/anomaly-candidates", json={"n_candidates": 3})
    assert r.status_code == 200, r.text
    candidates = r.json()["candidates"]
    assert candidates
    return candidates[0]


def _decide(client, pid, candidate, verdict="structure", note="정사영상에서 창고 확인"):
    return client.post(
        f"/api/projects/{pid}/candidate-notes",
        json={
            "lat": candidate["lat"], "lon": candidate["lon"], "verdict": verdict, "note": note,
            "peak_anomaly_nt": candidate["peak_anomaly_nt"], "depth_m": candidate["depth_m"],
            "radius_m": candidate["radius_m"], "removed_method": "model",
        },
    )


def test_a_decision_comes_back_on_the_next_scan(project):
    client, pid, _ = project
    candidate = _a_candidate(client, pid)

    assert _decide(client, pid, candidate).status_code == 200

    again = _a_candidate(client, pid)
    assert again["note"]["verdict"] == "structure"
    assert again["note"]["note"] == "정사영상에서 창고 확인"
    assert again["note"]["removed_method"] == "model"


def test_it_is_kept_by_position_so_renumbering_does_not_lose_it(project):
    """Ranks belong to a scan. Re-scanning on a different grid renumbers
    everything, and the decision has to follow the ground."""
    client, pid, _ = project
    candidate = _a_candidate(client, pid)
    _decide(client, pid, candidate)

    coarser = client.post(f"/api/projects/{pid}/anomaly-candidates",
                          json={"n_candidates": 5, "cell_size_m": 4.0}).json()["candidates"]
    matched = [c for c in coarser if c["note"]]

    assert matched, "the decision did not survive a re-scan"
    assert matched[0]["note"]["verdict"] == "structure"


def test_deciding_again_replaces_rather_than_piles_up(project):
    client, pid, _ = project
    candidate = _a_candidate(client, pid)
    _decide(client, pid, candidate, verdict="structure")

    r = _decide(client, pid, candidate, verdict="geology", note="주변 노두와 연결됨")

    assert len(r.json()["notes"]) == 1
    assert r.json()["note"]["verdict"] == "geology"
    assert r.json()["note"]["note"] == "주변 노두와 연결됨"


def test_a_decision_can_be_cleared(project):
    client, pid, _ = project
    candidate = _a_candidate(client, pid)
    _decide(client, pid, candidate)

    r = client.post(f"/api/projects/{pid}/candidate-notes",
                    json={"lat": candidate["lat"], "lon": candidate["lon"], "verdict": "clear"})

    assert r.json()["notes"] == []
    assert _a_candidate(client, pid)["note"] is None


def test_decisions_survive_reprocessing(project):
    """They are about what is on the ground, not about the settings this
    run happened to use."""
    client, pid, _ = project
    _decide(client, pid, _a_candidate(client, pid))

    client.post(f"/api/projects/{pid}/process", json={"line_params": {}, "diurnal_params": {}})

    assert len(client.get(f"/api/projects/{pid}/candidate-notes").json()["notes"]) == 1


def test_they_are_saved_with_the_project(project):
    client, pid, proj = project
    _decide(client, pid, _a_candidate(client, pid))

    other = Project(id="reloaded")
    other.load_project_bundle(proj.save_project_bundle())

    assert [n["verdict"] for n in other.candidate_notes] == ["structure"]


def test_the_log_exports_as_a_table(project):
    client, pid, _ = project
    _decide(client, pid, _a_candidate(client, pid))

    r = client.get(f"/api/projects/{pid}/candidate-notes/csv")

    assert r.status_code == 200
    text = r.content.decode("utf-8-sig")
    header, row = text.splitlines()[:2]
    assert header.split(",")[:5] == ["latitude", "longitude", "easting", "northing", "verdict"]
    assert "지상구조물" in row and "정사영상에서 창고 확인" in row
