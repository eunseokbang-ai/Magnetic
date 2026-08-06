"""API-level test of the auto-selected nearest-INTERMAGNET-observatories
substitute-base-station flow: fetch (preview, not yet applied) -> apply ->
usable by run_pipeline, plus the CSV export endpoint. requests.get is
mocked (see test_intermagnet_nearest.py) - only synthetic, format-accurate
IAGA-2002 text is used, never real observatory data."""
import pathlib
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"

# The sample drone survey's average position is ~(46.48N, 106.27E) - place
# mock stations spread around it so the nearest-selection has real work to
# do, one of them (IRT) essentially on top of the target.
_MOCK_STATIONS = {
    "IRT": (46.50, 106.30, 49245.0),  # ~on target
    "BOX": (58.00, 38.00, 55000.0),  # far NW
    "KHB": (48.50, 135.00, 47000.0),  # east
    "MZL": (30.00, 106.00, 44000.0),  # south
}


def _make_iaga_text(code: str, lat: float, lon: float, base_f: float, n: int = 4) -> str:
    lines = [
        " Format                 IAGA-2002                                    |",
        f" Station Name           {code} Test Station                          |",
        f" IAGA CODE              {code}                                           |",
        f" Geodetic Latitude      {lat:.3f}                                        |",
        f" Geodetic Longitude     {lon:.3f}                                        |",
        " Elevation              500                                           |",
        " Reported                XYZF                                        |",
        f"DATE       TIME         DOY     {code}X      {code}Y      {code}Z      {code}F   |",
    ]
    for i in range(n):
        lines.append(f"2026-07-24 06:{i*10:02d}:00.000 205     20000.00  0.00      45000.00  {base_f + i:.2f}")
    return "\n".join(lines) + "\n"


def _fake_requests_get(url, params=None, timeout=None):
    code = (params or {}).get("observatoryIagaCode")
    resp = MagicMock()
    if code in _MOCK_STATIONS:
        lat, lon, base_f = _MOCK_STATIONS[code]
        resp.status_code = 200
        resp.text = _make_iaga_text(code, lat, lon, base_f)
    else:
        resp.status_code = 404
        resp.text = "not found"
    return resp


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


def test_fetch_previews_nearest_stations_without_committing(client):
    project_id = _new_project(client)
    _upload_drone(client, project_id)

    with patch("app.processing.intermagnet.requests.get", side_effect=_fake_requests_get):
        r = client.post(
            f"/api/projects/{project_id}/base/intermagnet/nearest/fetch",
            json={"start_date": "2026-07-24", "end_date": "2026-07-24", "n_stations": 3},
        )
    assert r.status_code == 200, r.text
    preview = r.json()
    assert 1 <= len(preview["stations"]) <= 3
    codes = {s["iaga_code"] for s in preview["stations"]}
    assert codes.issubset(set(_MOCK_STATIONS))
    assert preview["n_points"] > 0

    # not yet applied - a plain process() call still requires a real base
    r = client.post(f"/api/projects/{project_id}/process", json={})
    assert r.status_code != 200


def test_fetch_auto_detects_dates_from_uploaded_drone_survey(client):
    # Omitting start_date/end_date entirely should auto-target the
    # project's own survey flight date(s) - the sample fixture flies on
    # 2026-07-24 only, which is also the date the mock station data is
    # dated, so this must succeed identically to explicitly passing that
    # date.
    project_id = _new_project(client)
    _upload_drone(client, project_id)

    with patch("app.processing.intermagnet.requests.get", side_effect=_fake_requests_get):
        r = client.post(
            f"/api/projects/{project_id}/base/intermagnet/nearest/fetch",
            json={"n_stations": 3},
        )
    assert r.status_code == 200, r.text
    preview = r.json()
    assert 1 <= len(preview["stations"]) <= 3


def test_fetch_auto_mode_without_drone_data_fails_clearly(client):
    project_id = _new_project(client)
    with patch("app.processing.intermagnet.requests.get", side_effect=_fake_requests_get):
        r = client.post(
            f"/api/projects/{project_id}/base/intermagnet/nearest/fetch",
            json={"n_stations": 3, "target_lat": 46.48, "target_lon": 106.27},
        )
    assert r.status_code == 400, r.text
    assert "드론" in r.json()["detail"]


def test_fetch_rejects_only_one_of_start_end_date(client):
    project_id = _new_project(client)
    _upload_drone(client, project_id)
    r = client.post(
        f"/api/projects/{project_id}/base/intermagnet/nearest/fetch",
        json={"start_date": "2026-07-24", "n_stations": 3},
    )
    assert r.status_code == 400, r.text
    assert "모두" in r.json()["detail"]


def test_apply_commits_nearest_preview_as_base_and_unblocks_processing(client):
    project_id = _new_project(client)
    _upload_drone(client, project_id)

    with patch("app.processing.intermagnet.requests.get", side_effect=_fake_requests_get):
        r = client.post(
            f"/api/projects/{project_id}/base/intermagnet/nearest/fetch",
            json={"start_date": "2026-07-24", "end_date": "2026-07-24", "n_stations": 3},
        )
    assert r.status_code == 200, r.text

    r = client.post(f"/api/projects/{project_id}/base/intermagnet/nearest/apply")
    assert r.status_code == 200, r.text
    base_summary = r.json()
    assert base_summary["n_points"] > 0
    assert base_summary["source"]["type"] == "intermagnet_nearest"

    r = client.post(f"/api/projects/{project_id}/process", json={})
    assert r.status_code == 200, r.text
    assert r.json()["diurnal"]["mode"] == "base_station"


def test_comparison_available_after_fetch_before_apply(client):
    project_id = _new_project(client)
    _upload_drone(client, project_id)

    with patch("app.processing.intermagnet.requests.get", side_effect=_fake_requests_get):
        r = client.post(
            f"/api/projects/{project_id}/base/intermagnet/nearest/fetch",
            json={"start_date": "2026-07-24", "end_date": "2026-07-24", "n_stations": 3},
        )
    assert r.status_code == 200, r.text

    r = client.get(f"/api/projects/{project_id}/base/intermagnet/nearest/comparison")
    assert r.status_code == 200, r.text
    comparison = r.json()
    assert len(comparison["combined"]["timestamp"]) > 0
    assert len(comparison["combined"]["timestamp"]) == len(comparison["combined"]["mag"])
    assert 1 <= len(comparison["stations"]) <= 3
    for s in comparison["stations"]:
        assert s["iaga_code"] in _MOCK_STATIONS
        assert len(s["timestamp"]) == len(s["mag"]) > 0
        assert s["distance_km"] >= 0


def test_comparison_still_available_after_apply(client):
    project_id = _new_project(client)
    _upload_drone(client, project_id)

    with patch("app.processing.intermagnet.requests.get", side_effect=_fake_requests_get):
        r = client.post(
            f"/api/projects/{project_id}/base/intermagnet/nearest/fetch",
            json={"start_date": "2026-07-24", "end_date": "2026-07-24", "n_stations": 3},
        )
    assert r.status_code == 200, r.text

    r = client.post(f"/api/projects/{project_id}/base/intermagnet/nearest/apply")
    assert r.status_code == 200, r.text

    # the preview is cleared by apply, but the comparison data should
    # survive - the user typically asks for the comparison after already
    # applying the estimate, not only during the pending-preview window.
    r = client.get(f"/api/projects/{project_id}/base/intermagnet/nearest/comparison")
    assert r.status_code == 200, r.text
    comparison = r.json()
    assert len(comparison["stations"]) >= 1


def test_comparison_without_any_fetch_fails_clearly(client):
    project_id = _new_project(client)
    r = client.get(f"/api/projects/{project_id}/base/intermagnet/nearest/comparison")
    assert r.status_code != 200
    assert "조회" in r.json()["detail"]


def test_apply_without_preview_fails_clearly(client):
    project_id = _new_project(client)
    r = client.post(f"/api/projects/{project_id}/base/intermagnet/nearest/apply")
    assert r.status_code != 200
    assert "미리보기" in r.json()["detail"]


def test_csv_export_without_preview_fails_clearly(client):
    project_id = _new_project(client)
    r = client.get(f"/api/projects/{project_id}/base/intermagnet/nearest/csv")
    assert r.status_code != 200


def test_csv_export_returns_combined_series(client):
    project_id = _new_project(client)
    _upload_drone(client, project_id)

    with patch("app.processing.intermagnet.requests.get", side_effect=_fake_requests_get):
        r = client.post(
            f"/api/projects/{project_id}/base/intermagnet/nearest/fetch",
            json={"start_date": "2026-07-24", "end_date": "2026-07-24", "n_stations": 3},
        )
    assert r.status_code == 200, r.text

    r = client.get(f"/api/projects/{project_id}/base/intermagnet/nearest/csv")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    body = r.text
    assert "timestamp" in body.splitlines()[0]
    assert "mag_nT" in body.splitlines()[0]
    assert len(body.splitlines()) > 1


def test_fetch_rejects_malformed_start_date_with_clear_korean_message(client):
    project_id = _new_project(client)
    r = client.post(
        f"/api/projects/{project_id}/base/intermagnet/nearest/fetch",
        json={"start_date": "not-a-date", "end_date": "2026-07-24", "n_stations": 3},
    )
    assert r.status_code == 400, r.text
    assert "시작일" in r.json()["detail"]


def test_fetch_rejects_end_date_before_start_date(client):
    project_id = _new_project(client)
    r = client.post(
        f"/api/projects/{project_id}/base/intermagnet/nearest/fetch",
        json={"start_date": "2026-07-24", "end_date": "2026-07-20", "n_stations": 3},
    )
    assert r.status_code == 400, r.text
    assert "종료일" in r.json()["detail"]


def test_fetch_without_drone_data_or_explicit_target_fails_clearly(client):
    project_id = _new_project(client)
    with patch("app.processing.intermagnet.requests.get", side_effect=_fake_requests_get):
        r = client.post(
            f"/api/projects/{project_id}/base/intermagnet/nearest/fetch",
            json={"start_date": "2026-07-24", "end_date": "2026-07-24", "n_stations": 3},
        )
    assert r.status_code != 200


def test_fetch_accepts_explicit_target_override_without_drone_data(client):
    project_id = _new_project(client)
    with patch("app.processing.intermagnet.requests.get", side_effect=_fake_requests_get):
        r = client.post(
            f"/api/projects/{project_id}/base/intermagnet/nearest/fetch",
            json={"start_date": "2026-07-24", "end_date": "2026-07-24", "n_stations": 3, "target_lat": 46.48, "target_lon": 106.27},
        )
    assert r.status_code == 200, r.text


if __name__ == "__main__":
    c = TestClient(app)
    test_fetch_previews_nearest_stations_without_committing(c)
    test_fetch_auto_detects_dates_from_uploaded_drone_survey(c)
    test_fetch_auto_mode_without_drone_data_fails_clearly(c)
    test_fetch_rejects_only_one_of_start_end_date(c)
    test_apply_commits_nearest_preview_as_base_and_unblocks_processing(c)
    test_comparison_available_after_fetch_before_apply(c)
    test_comparison_still_available_after_apply(c)
    test_comparison_without_any_fetch_fails_clearly(c)
    test_apply_without_preview_fails_clearly(c)
    test_csv_export_without_preview_fails_clearly(c)
    test_csv_export_returns_combined_series(c)
    test_fetch_rejects_malformed_start_date_with_clear_korean_message(c)
    test_fetch_rejects_end_date_before_start_date(c)
    test_fetch_without_drone_data_or_explicit_target_fails_clearly(c)
    test_fetch_accepts_explicit_target_override_without_drone_data(c)
    print("ALL CHECKS PASSED")
