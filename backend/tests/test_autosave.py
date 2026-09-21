"""Surviving a restart.

What has to hold: the work is written without the request waiting for it,
one burst of edits costs one write, an interrupted write never replaces a
good bundle with a broken one, and what comes back is the project that
went in - under its own id.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.autosave import AutosaveManager
from app.main import app
from app.store import ProjectStore

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """A processed project, its own store, and a manager writing to
    tmp_path - never the user's real autosave directory."""
    monkeypatch.setenv("MAGNETIC_AUTOSAVE_DIR", str(tmp_path))
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{pid}/process", json={"line_params": {}, "diurnal_params": {}})
    assert r.status_code == 200, r.text
    from app.store import store as real_store

    return client, pid, real_store.get(pid), AutosaveManager(real_store, directory=tmp_path,
                                                            interval_seconds=60.0, quiet_seconds=5.0)


def test_it_saves_and_restores_under_the_same_id(saved, tmp_path):
    _client, pid, project, manager = saved
    n_points = len(project.processed)

    assert manager.save_project(project) is not None

    # a fresh server: nothing in memory, one bundle on disk
    other = ProjectStore()
    fresh = AutosaveManager(other, directory=tmp_path)
    listed = fresh.list_saved()
    assert [s.project_id for s in listed] == [pid]
    assert listed[0].to_dict()["n_points"] > 0

    restored, summary = fresh.restore(pid)
    assert restored.id == pid
    assert len(restored.processed) == n_points
    assert summary


def test_edits_after_a_save_come_back(saved, tmp_path):
    client, pid, project, manager = saved
    manager.save_project(project)
    df = project.processed
    line = df[df["line_id"] >= 0]["line_id"].value_counts().index[0]
    stretch = df[df["line_id"] == line].sort_values("timestamp")
    ids = [int(i) for i in stretch["point_id"].iloc[10:70]]
    client.post(f"/api/projects/{pid}/smooth", json={"mode": "point_ids", "point_ids": ids})

    manager.save_project(project)
    restored, _ = AutosaveManager(ProjectStore(), directory=tmp_path).restore(pid)

    assert set(restored.manual_smooth_point_ids) == set(ids)


def test_a_burst_of_edits_costs_one_write(saved):
    _client, pid, project, manager = saved
    now = 1000.0

    project.changed_at = now
    assert manager.save_due(now=now + 1) == []          # still being edited
    assert manager.save_due(now=now + 10) == [pid]      # settled, written
    project.changed_at = now + 11
    assert manager.save_due(now=now + 20) == []         # too soon after the last write
    project.changed_at = now + 100
    assert manager.save_due(now=now + 200) == [pid]     # interval elapsed


def test_nothing_is_written_for_a_project_with_no_readings(tmp_path, monkeypatch):
    monkeypatch.setenv("MAGNETIC_AUTOSAVE_DIR", str(tmp_path))
    empty = ProjectStore()
    project = empty.create()
    project.changed_at = 1.0
    manager = AutosaveManager(empty, directory=tmp_path, quiet_seconds=0.0, interval_seconds=0.0)

    assert manager.save_project(project) is None
    assert manager.save_due(now=100.0) == []
    assert manager.list_saved() == []


def test_an_interrupted_write_leaves_the_previous_bundle_alone(saved, tmp_path, monkeypatch):
    _client, pid, project, manager = saved
    good = manager.save_project(project).read_bytes()

    def explode(self):
        raise RuntimeError("disk full")

    monkeypatch.setattr(type(project), "save_project_bundle", explode)
    assert manager.save_project(project) is None
    assert (tmp_path / f"{pid}.magproj").read_bytes() == good
    assert not list(tmp_path.glob(".*tmp"))


def test_only_the_newest_bundles_are_kept(saved, tmp_path):
    from app.autosave import MAX_SAVED

    _client, pid, project, manager = saved
    manager.save_project(project)
    for n in range(MAX_SAVED + 2):
        project.id = f"copy{n:02d}"
        manager.save_project(project)

    kept = manager.list_saved()
    assert len(kept) == MAX_SAVED
    assert pid not in [s.project_id for s in kept]      # the oldest went first


def test_the_api_lists_restores_and_deletes(saved, tmp_path, monkeypatch):
    client, pid, project, manager = saved
    import app.main as main

    monkeypatch.setattr(main, "autosave", manager)
    manager.save_project(project)

    listed = client.get("/api/projects/autosaves").json()["autosaves"]
    assert [a["project_id"] for a in listed] == [pid]

    r = client.post(f"/api/projects/autosaves/{pid}/restore")
    assert r.status_code == 200 and r.json()["project_id"] == pid

    assert client.delete(f"/api/projects/autosaves/{pid}").json()["autosaves"] == []
    assert client.post(f"/api/projects/autosaves/{pid}/restore").status_code == 404
