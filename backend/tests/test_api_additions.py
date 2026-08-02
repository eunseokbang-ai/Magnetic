"""API-level integration tests for the group 1+2 additions: new transform
endpoints (THDR, upward continuation, detrend, micro-leveling), line
profile, grid/transform XYZ export, points CSV export, filter method
selection, and UTM EPSG override - run against the real sample survey
via the FastAPI TestClient (no live network access needed).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def _make_processed_project(process_overrides=None):
    client = TestClient(app)
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    payload = {"line_params": {}, "diurnal_params": {}, "heading_correction": {}}
    if process_overrides:
        payload.update(process_overrides)
    r = client.post(f"/api/projects/{project_id}/process", json=payload)
    assert r.status_code == 200, r.text
    return client, project_id, r.json()


def test_new_transforms_via_api():
    client, project_id, _summary = _make_processed_project()

    r = client.post(f"/api/projects/{project_id}/transform", json={"transform": "thdr", "cell_size_m": 15.0})
    assert r.status_code == 200, r.text
    assert r.json()["stats"]["min"] >= 0  # THDR is a non-negative amplitude

    r = client.post(
        f"/api/projects/{project_id}/transform",
        json={"transform": "upward_continuation", "cell_size_m": 15.0, "continuation_height_m": 30.0},
    )
    assert r.status_code == 200, r.text

    r = client.post(f"/api/projects/{project_id}/transform", json={"transform": "detrend", "cell_size_m": 15.0, "trend_order": 2})
    assert r.status_code == 200, r.text

    r = client.post(f"/api/projects/{project_id}/transform", json={"transform": "microlevel", "cell_size_m": 15.0})
    assert r.status_code == 200, r.text

    r = client.post(
        f"/api/projects/{project_id}/transform",
        json={"transform": "upward_continuation", "cell_size_m": 15.0, "continuation_height_m": -1.0},
    )
    assert r.status_code in (400, 422), r.text  # rejected either by pydantic (gt=0) or the processing function
    print("NEW TRANSFORM ENDPOINT CHECKS PASSED")


def test_microlevel_pre_apply_affects_other_transforms_and_base_grid():
    """microlevel_pre_apply should change the result of any transform
    other than "microlevel" itself, and of the plain grid endpoint too -
    it's meant to pre-clean corrugation before RTP/derivative transforms
    (or the base grid view) see it, not just be reachable as its own
    mutually-exclusive transform choice."""
    client, project_id, _summary = _make_processed_project()

    r_off = client.post(f"/api/projects/{project_id}/transform", json={"transform": "rtp", "cell_size_m": 15.0})
    assert r_off.status_code == 200, r_off.text
    r_on = client.post(
        f"/api/projects/{project_id}/transform",
        json={"transform": "rtp", "cell_size_m": 15.0, "microlevel_pre_apply": True},
    )
    assert r_on.status_code == 200, r_on.text
    assert r_off.json()["stats"] != r_on.json()["stats"]

    # applying it to "microlevel" itself must not double-apply / error
    r_ml_off = client.post(f"/api/projects/{project_id}/transform", json={"transform": "microlevel", "cell_size_m": 15.0})
    r_ml_on = client.post(
        f"/api/projects/{project_id}/transform",
        json={"transform": "microlevel", "cell_size_m": 15.0, "microlevel_pre_apply": True},
    )
    assert r_ml_off.status_code == 200 and r_ml_on.status_code == 200
    assert r_ml_off.json()["stats"] == r_ml_on.json()["stats"]

    # the plain grid endpoint (grid(원본) in the UI) must respect it too
    g_off = client.post(f"/api/projects/{project_id}/grid", json={"cell_size_m": 15.0})
    g_on = client.post(f"/api/projects/{project_id}/grid", json={"cell_size_m": 15.0, "microlevel_pre_apply": True})
    assert g_off.status_code == 200 and g_on.status_code == 200
    assert g_off.json()["stats"] != g_on.json()["stats"]
    print("MICROLEVEL PRE-APPLY CHECKS PASSED")


def test_line_profile_endpoint():
    client, project_id, summary = _make_processed_project()
    line_id = summary["lines"][0]["line_id"]

    r = client.get(f"/api/projects/{project_id}/line-profile", params={"line_id": line_id, "value": "anomaly"})
    assert r.status_code == 200, r.text
    profile = r.json()
    assert profile["line_id"] == line_id
    n = len(profile["distance_m"])
    assert n > 0
    assert len(profile["value"]) == n and len(profile["lat"]) == n and len(profile["excluded"]) == n
    assert profile["distance_m"][0] == 0.0
    assert profile["distance_m"] == sorted(profile["distance_m"])  # monotonically non-decreasing along track

    r = client.get(f"/api/projects/{project_id}/line-profile", params={"line_id": 999999, "value": "anomaly"})
    assert r.status_code == 400, r.text
    print("LINE PROFILE ENDPOINT CHECKS PASSED")


def test_grid_and_transform_xyz_export():
    client, project_id, _summary = _make_processed_project()

    r = client.post(f"/api/projects/{project_id}/grid/xyz", json={"cell_size_m": 20.0})
    assert r.status_code == 200, r.text
    text = r.content.decode("utf-8")
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) > 10
    parts = lines[0].split()
    assert len(parts) == 3
    lon, lat, value = (float(p) for p in parts)
    assert 100 < lon < 115  # sanity: Mongolia-area sample data
    assert 40 < lat < 55

    r = client.post(f"/api/projects/{project_id}/transform/xyz", json={"transform": "thdr", "cell_size_m": 20.0})
    assert r.status_code == 200, r.text
    assert len(r.content) > 0
    print("GRID/TRANSFORM XYZ EXPORT CHECKS PASSED")


def test_points_csv_export():
    client, project_id, _summary = _make_processed_project()
    r = client.get(f"/api/projects/{project_id}/points/csv")
    assert r.status_code == 200, r.text
    text = r.content.decode("utf-8")
    header = text.splitlines()[0]
    for col in ["point_id", "lat", "lon", "anomaly_nt", "tmi_nt", "line_id", "excluded"]:
        assert col in header, header
    assert len(text.splitlines()) > 100
    print("POINTS CSV EXPORT CHECKS PASSED")


def test_filter_method_selection():
    for method, extra in [
        ("butterworth", {"filter_cutoff_hz": 1.0}),
        ("moving_average", {"filter_window_seconds": 0.5}),
        ("savgol", {"filter_window_seconds": 0.5, "filter_polyorder": 3}),
    ]:
        client, project_id, summary = _make_processed_project({"filter_method": method, **extra})
        assert summary["anomaly_stats"]["min"] is not None, (method, summary)
    print("FILTER METHOD SELECTION CHECKS PASSED")


def test_utm_epsg_override():
    client, project_id, summary = _make_processed_project({"utm_epsg_override": 32649})
    assert summary["utm_epsg"] == 32649, summary["utm_epsg"]
    print("UTM EPSG OVERRIDE CHECK PASSED")


if __name__ == "__main__":
    test_new_transforms_via_api()
    test_line_profile_endpoint()
    test_grid_and_transform_xyz_export()
    test_points_csv_export()
    test_filter_method_selection()
    test_utm_epsg_override()
    print("\nALL CHECKS PASSED")
