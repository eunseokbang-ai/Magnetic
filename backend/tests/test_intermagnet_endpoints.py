"""API-level test of the IAGA-2002 manual-upload substitute-base-station
flow: upload -> preview (not yet applied) -> apply -> usable by run_pipeline.
Uses the same hand-written format-accurate synthetic IAGA-2002 text as
test_intermagnet.py (not real observatory data)."""
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"

# Spans the sample drone survey's flight window (2026-07-24 06:21-06:58)
# with 1-minute samples, station placed near the Mongolia sample survey's
# coordinates (46.48N, 106.27E) so the preview's distance calc is small.
_IAGA_TEXT = """\
 Format                 IAGA-2002                                    |
 Source of Data         Test Institute                                |
 Station Name           Irkutsk-like Test Station                     |
 IAGA CODE              IRT                                           |
 Geodetic Latitude      52.270                                        |
 Geodetic Longitude     104.450                                       |
 Elevation              500                                           |
 Reported                XYZF                                        |
DATE       TIME         DOY     IRTX      IRTY      IRTZ      IRTF   |
2026-07-24 06:00:00.000 205     20000.00  0.00      45000.00  49244.29
2026-07-24 06:30:00.000 205     20001.00  0.10      45001.00  49245.31
2026-07-24 07:00:00.000 205     20002.00  0.20      45002.00  49246.33
2026-07-24 07:30:00.000 205     20003.00  0.30      45003.00  49247.34
"""


@pytest.fixture
def client():
    return TestClient(app)


def _new_project(client):
    r = client.post("/api/projects")
    assert r.status_code == 200, r.text
    return r.json()["project_id"]


def _upload_drone(client, project_id):
    drone_path = FIXTURES / "sample_drone_survey.csv"
    with open(drone_path, "rb") as f:
        r = client.post(f"/api/projects/{project_id}/upload/drone", files={"files": (drone_path.name, f, "text/csv")})
    assert r.status_code == 200, r.text


def test_upload_previews_without_committing_as_base(client):
    project_id = _new_project(client)
    _upload_drone(client, project_id)

    r = client.post(
        f"/api/projects/{project_id}/base/iaga2002/upload",
        files={"file": ("irt.txt", io.BytesIO(_IAGA_TEXT.encode()), "text/plain")},
    )
    assert r.status_code == 200, r.text
    preview = r.json()
    assert preview["iaga_code"] == "IRT"
    assert preview["n_points"] == 4
    assert preview["distance_from_survey_km"] is not None
    assert preview["distance_from_survey_km"] < 1000  # Irkutsk-like coords are a few hundred km from the sample survey

    # not yet applied - a plain process() call still requires a real base
    r = client.post(f"/api/projects/{project_id}/process", json={})
    assert r.status_code != 200


def test_apply_commits_preview_as_base_and_unblocks_processing(client):
    project_id = _new_project(client)
    _upload_drone(client, project_id)

    r = client.post(
        f"/api/projects/{project_id}/base/iaga2002/upload",
        files={"file": ("irt.txt", io.BytesIO(_IAGA_TEXT.encode()), "text/plain")},
    )
    assert r.status_code == 200, r.text

    r = client.post(f"/api/projects/{project_id}/base/intermagnet/apply")
    assert r.status_code == 200, r.text
    base_summary = r.json()
    assert base_summary["n_points"] == 4
    assert base_summary["source"]["type"] == "intermagnet"
    assert base_summary["source"]["iaga_code"] == "IRT"

    r = client.post(f"/api/projects/{project_id}/process", json={})
    assert r.status_code == 200, r.text
    assert r.json()["diurnal"]["mode"] == "base_station"


def test_apply_without_preview_fails_clearly(client):
    project_id = _new_project(client)
    r = client.post(f"/api/projects/{project_id}/base/intermagnet/apply")
    assert r.status_code != 200
    assert "미리보기" in r.json()["detail"]


def test_upload_rejects_non_iaga_text(client):
    project_id = _new_project(client)
    r = client.post(
        f"/api/projects/{project_id}/base/iaga2002/upload",
        files={"file": ("bad.txt", io.BytesIO(b"not an iaga file"), "text/plain")},
    )
    assert r.status_code != 200


def test_fetch_rejects_malformed_start_date_with_clear_korean_message(client):
    project_id = _new_project(client)
    r = client.post(
        f"/api/projects/{project_id}/base/intermagnet/fetch",
        json={"iaga_code": "IRT", "start_date": "not-a-date", "end_date": "2026-07-24"},
    )
    assert r.status_code == 400, r.text
    assert "시작일" in r.json()["detail"]


def test_fetch_rejects_end_date_before_start_date(client):
    project_id = _new_project(client)
    r = client.post(
        f"/api/projects/{project_id}/base/intermagnet/fetch",
        json={"iaga_code": "IRT", "start_date": "2026-07-24", "end_date": "2026-07-20"},
    )
    assert r.status_code == 400, r.text
    assert "종료일" in r.json()["detail"]


if __name__ == "__main__":
    c = TestClient(app)
    test_upload_previews_without_committing_as_base(c)
    test_apply_commits_preview_as_base_and_unblocks_processing(c)
    test_apply_without_preview_fails_clearly(c)
    test_upload_rejects_non_iaga_text(c)
    test_fetch_rejects_malformed_start_date_with_clear_korean_message(c)
    test_fetch_rejects_end_date_before_start_date(c)
    print("ALL CHECKS PASSED")
