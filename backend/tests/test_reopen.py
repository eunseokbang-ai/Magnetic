"""Reopening a project by its id - from memory after a browser refresh,
from the autosaved bundle after a server restart - and the write interval
that follows what a write costs."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.autosave import AutosaveManager
from app.main import app
from app.store import store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"
KML = """<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">
<Placemark><name>BH-1</name><Point><coordinates>106.27,46.50,0</coordinates></Point></Placemark></kml>"""


@pytest.fixture()
def session(tmp_path, monkeypatch):
    manager = AutosaveManager(store, directory=tmp_path)
    monkeypatch.setattr(main_module, "autosave", manager)
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files=[("files", ("d.csv", f.read(), "text/csv"))])
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files=[("files", ("b.csv", f.read(), "text/csv"))])
    assert client.post(f"/api/projects/{pid}/process", json={"filter_cutoff_hz": 0.8}).status_code == 200
    assert client.post(f"/api/projects/{pid}/reference-layers", files={"file": ("drill.kml", KML.encode(), "text/xml")}).status_code == 200
    assert client.post(f"/api/projects/{pid}/inversion", json={"regularization_strength": 1.0}).status_code == 200
    return client, pid, manager


def test_a_loaded_project_reopens_from_memory_without_being_touched(session):
    client, pid, _manager = session
    project = store.get(pid)
    frame, changed = project.processed, project.changed_at

    r = client.post(f"/api/projects/{pid}/open")
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["project_id"], body["source"], body["processed"]) == (pid, "memory", True)
    # the same things a bundle load hands the screen
    assert body["params"]["filter_cutoff_hz"] == 0.8
    assert body["process_summary"]["n_lines"] == 6
    assert body["drone_summary"]["n_points"] == 22025 and body["base_summary"]["n_points"] > 0
    assert body["inversion_restored"] is True and body["inversion_summary"]["n_active_cells"] > 0
    assert [(l["type"], l["name"]) for l in body["reference_layers"]] == [("vector", "drill")]
    # and nothing was recomputed or marked as changed by looking
    assert project.processed is frame and project.changed_at == changed


def test_after_a_restart_the_same_address_comes_back_from_disk(session):
    client, pid, manager = session
    assert manager.save_project(store.get(pid)) is not None
    with store._lock:                                   # the server went away and came back empty
        del store._projects[pid]

    r = client.post(f"/api/projects/{pid}/open")
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["project_id"], body["source"], body["processed"]) == (pid, "disk", True)
    assert body["params"]["filter_cutoff_hz"] == 0.8 and body["process_summary"]["n_lines"] == 6
    assert [(l["type"], l["name"]) for l in body["reference_layers"]] == [("vector", "drill")]
    # the inversion came back with it, and the assistant's lookup can see it again
    assert body["inversion_restored"] is True
    assert store.get(pid).inversion_summary_cache["n_active_cells"] == body["inversion_summary"]["n_active_cells"]
    # a second open is now a memory hit
    assert client.post(f"/api/projects/{pid}/open").json()["source"] == "memory"


def test_an_empty_project_left_in_memory_does_not_hide_its_saved_copy(session):
    client, pid, manager = session
    manager.save_project(store.get(pid))
    with store._lock:
        del store._projects[pid]
    store.get_or_create(pid)                             # e.g. a stray request recreated the id with nothing in it
    assert client.post(f"/api/projects/{pid}/open").json()["source"] == "disk"


def test_an_address_that_leads_nowhere_is_a_404(session):
    client, _pid, _manager = session
    r = client.post("/api/projects/000000000000/open")
    assert r.status_code == 404 and "찾을 수 없습니다" in r.json()["detail"]


def test_the_wait_between_writes_follows_what_a_write_costs(session, tmp_path):
    _client, pid, _ = session
    manager = AutosaveManager(store, directory=tmp_path, interval_seconds=120.0, quiet_seconds=8.0)
    assert manager.interval_for(pid) == 120.0            # nothing known yet (and the first write is not held back anyway)
    for cost, expected in ((0.36, 15.0), (2.2, 66.0), (10.0, 120.0)):   # Andong, HaeNam, the old CSV bundle
        manager._last_cost[pid] = cost
        assert manager.interval_for(pid) == pytest.approx(expected)

    project = store.get(pid)
    project.mark_changed()
    now = project.changed_at
    assert pid in manager.save_due(now=now + 10)      # first write: settled, nothing to wait for
    assert manager._last_cost[pid] < 0.5                 # a small project: the 15 s minimum applies
    project.changed_at = now + 11
    assert pid not in manager.save_due(now=now + 20)      # 10 s after the last write
    assert pid in manager.save_due(now=now + 26)      # 16 s after it - not the two minutes it used to be
