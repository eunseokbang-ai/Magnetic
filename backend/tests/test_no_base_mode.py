"""Processing without any base (diurnal) station data - the user assumes
the Earth's field was steady over the survey window (see
DiurnalParams.mode == "assume_constant"), so the drone's own filtered
reading is used directly instead of being corrected against a base log."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def client():
    return TestClient(app)


def _new_project_with_drone(client):
    r = client.post("/api/projects")
    assert r.status_code == 200, r.text
    project_id = r.json()["project_id"]
    drone_path = FIXTURES / "sample_drone_survey.csv"
    with open(drone_path, "rb") as f:
        r = client.post(f"/api/projects/{project_id}/upload/drone", files={"files": (drone_path.name, f, "text/csv")})
    assert r.status_code == 200, r.text
    return project_id


def test_process_without_base_requires_assume_constant_mode(client):
    project_id = _new_project_with_drone(client)
    r = client.post(f"/api/projects/{project_id}/process", json={})
    assert r.status_code != 200
    assert "베이스" in r.json()["detail"]


def test_process_without_base_succeeds_in_assume_constant_mode(client):
    project_id = _new_project_with_drone(client)
    r = client.post(
        f"/api/projects/{project_id}/process",
        json={"diurnal_params": {"mode": "assume_constant"}},
    )
    assert r.status_code == 200, r.text
    summary = r.json()
    assert summary["n_lines"] > 0
    assert summary["diurnal"]["mode"] == "assume_constant"
    assert summary["diurnal"]["has_overlap"] is False
    assert summary["base_qc"] is None

    r = client.get(f"/api/projects/{project_id}/points", params={"value": "tmi"})
    assert r.status_code == 200, r.text
    points = r.json()
    assert len(points) > 0


def test_assume_constant_mode_matches_zero_diurnal_correction(client):
    """With a real base file whose value is forced constant, diurnal
    correction subtracts (constant - mean) == 0 everywhere - the same
    result assume_constant mode gives without any base file at all."""
    project_id = _new_project_with_drone(client)
    r = client.post(f"/api/projects/{project_id}/process", json={"diurnal_params": {"mode": "assume_constant"}})
    assert r.status_code == 200, r.text
    no_base_summary = r.json()

    r = client.get(f"/api/projects/{project_id}/points", params={"value": "anomaly"})
    no_base_points = {p["point_id"]: p["value"] for p in r.json()}

    import io
    import pandas as pd

    base_df = pd.read_csv(FIXTURES / "sample_base_station.csv", header=None, encoding="utf-8-sig")
    base_df[1] = 50000.0  # force a perfectly constant reading
    buf = io.BytesIO()
    base_df.to_csv(buf, index=False, header=False)
    buf.seek(0)

    r = client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("const_base.csv", buf, "text/csv")})
    assert r.status_code == 200, r.text

    r = client.post(
        f"/api/projects/{project_id}/process",
        json={"diurnal_params": {"mode": "base_station"}, "base_qc_params": {"trim_enabled": False, "despike_enabled": False}},
    )
    assert r.status_code == 200, r.text

    r = client.get(f"/api/projects/{project_id}/points", params={"value": "anomaly"})
    const_base_points = {p["point_id"]: p["value"] for p in r.json()}

    common_ids = set(no_base_points) & set(const_base_points)
    assert len(common_ids) > 100
    for pid in list(common_ids)[:200]:
        assert no_base_points[pid] == pytest.approx(const_base_points[pid], abs=1e-6)


if __name__ == "__main__":
    c = TestClient(app)
    test_process_without_base_requires_assume_constant_mode(c)
    test_process_without_base_succeeds_in_assume_constant_mode(c)
    test_assume_constant_mode_matches_zero_diurnal_correction(c)
    print("ALL CHECKS PASSED")
