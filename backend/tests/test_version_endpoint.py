"""GET /api/version: reports which commit the running backend process was
started from, so the UI can show it directly (see main.py::
_detect_running_version) - "already-fixed" bug reports have repeatedly
turned out to be a stale run.bat build rather than a real regression, and
this gives users a concrete way to check for themselves.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app


def test_version_endpoint_reports_a_commit_hash_in_this_git_checkout():
    client = TestClient(app)
    r = client.get("/api/version")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"commit", "commit_date", "branch"}
    # This test itself only runs inside a git checkout (the whole app does),
    # so a working detection must find a real short hash, not the None
    # fallback reserved for a non-git deployment.
    assert body["commit"] and len(body["commit"]) >= 6
    assert body["commit_date"]
    assert body["branch"]
