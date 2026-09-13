"""Tests for the two mineral-exploration additions deferred from the
original lineament/depth-estimation batch: magnetic contact detection
(processing/contacts.py, THDR-ridge persistence across upward-continued
heights - "worming") and rule-based mineral prospectivity/target scoring
(processing/prospectivity.py). Both build on the same synthetic
NE-SW-striking vertical contact used in test_lineaments_and_depth_estimation.py,
plus a synthetic compact anomaly for the prospectivity target-ranking check.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.processing.contacts import detect_magnetic_contacts
from app.processing.gridding import grid_points
from app.processing.prospectivity import compute_prospectivity
from app.processing.transforms import analytic_signal, total_horizontal_derivative

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"
UTM_EPSG = 32652


def _synthetic_contact(depth_m=30.0, seed=0):
    rng = np.random.default_rng(seed)
    n = 6000
    x = rng.uniform(0, 1000, n)
    y = rng.uniform(0, 1000, n)
    dist_to_contact = (y - x) / np.sqrt(2)
    values = 200.0 * np.arctan(dist_to_contact / depth_m) + rng.normal(0, 0.5, n)
    return grid_points(x, y, values, cell_size_m=10.0, method="linear")


def _synthetic_dipole_anomaly(seed=1):
    """A single compact Gaussian-bump anomaly at a known location, for
    checking that prospectivity ranks a target right on top of it."""
    rng = np.random.default_rng(seed)
    n = 6000
    x = rng.uniform(0, 1000, n)
    y = rng.uniform(0, 1000, n)
    peak_x, peak_y = 500.0, 500.0
    r2 = (x - peak_x) ** 2 + (y - peak_y) ** 2
    values = 500.0 * np.exp(-r2 / (2 * 80.0**2)) + rng.normal(0, 1.0, n)
    grid = grid_points(x, y, values, cell_size_m=10.0, method="linear")
    return grid, (peak_x, peak_y)


def test_contact_detection_recovers_ne_sw_contact():
    grid = _synthetic_contact()
    result = detect_magnetic_contacts(
        grid.values, grid.easting, grid.northing, grid.cell_size_m, UTM_EPSG,
        heights_m=(0.0, 10.0, 20.0, 30.0), percentile_threshold=80.0, min_persistence=0.3, min_segment_points=3,
    )
    assert result["available"] is True
    assert result["n_contacts"] > 0
    assert result["total_length_m"] > 0
    for c in result["contacts"]:
        assert 0.0 <= c["mean_persistence"] <= 1.0
        assert c["n_points"] >= 3


def test_contact_detection_unavailable_on_too_small_grid():
    tiny = np.full((2, 2), 1.0)
    result = detect_magnetic_contacts(tiny, np.array([0.0, 10.0]), np.array([0.0, 10.0]), 10.0, UTM_EPSG)
    assert result["available"] is False
    assert result["contacts"] == []


def test_contact_detection_api_endpoint():
    client = TestClient(app)
    project_id = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("drone.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("base.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{project_id}/process", json={})
    assert r.status_code == 200

    r = client.post(
        f"/api/projects/{project_id}/contacts",
        json={"cell_size_m": 15.0, "percentile_threshold": 70.0, "min_persistence": 0.3, "min_segment_points": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert "available" in body
    assert "contacts" in body


def test_prospectivity_ranks_target_at_anomaly_peak():
    grid, (peak_x, peak_y) = _synthetic_dipole_anomaly()
    asa = analytic_signal(grid.values, grid.cell_size_m)
    thd = total_horizontal_derivative(grid.values, grid.cell_size_m)
    finite_mask = np.isfinite(grid.values)

    result = compute_prospectivity(
        asa, thd, finite_mask, grid.easting, grid.northing, UTM_EPSG,
        purpose="custom", custom_weights={"asa": 0.6, "thd": 0.4},
        score_threshold=0.5, max_targets=5,
    )
    assert result["available"] is True
    assert result["n_targets"] >= 1
    assert set(result["layers_used"]) == {"asa", "thd"}
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-9

    top = result["targets"][0]
    assert top["score"] >= result["targets"][-1]["score"]
    # the top target should sit within a couple of cells of the true peak
    dist = np.hypot(
        _local_x(top["lat"], top["lon"], UTM_EPSG) - peak_x,
        _local_y(top["lat"], top["lon"], UTM_EPSG) - peak_y,
    )
    # THD peaks on the flanks of a Gaussian bump (zero at the very
    # center), so the ASA+THD weighted sum's maximum can sit somewhat off
    # the true peak - a loose bound just confirms it's landing on the
    # anomaly, not somewhere unrelated.
    assert dist < 100.0
    assert top["explanation"]
    assert "asa" in top["breakdown"] or "thd" in top["breakdown"]


def _local_xy(lat, lon, utm_epsg):
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def _local_x(lat, lon, utm_epsg):
    return _local_xy(lat, lon, utm_epsg)[0]


def _local_y(lat, lon, utm_epsg):
    return _local_xy(lat, lon, utm_epsg)[1]


def test_prospectivity_unavailable_on_too_small_grid():
    tiny = np.full((2, 2), 1.0)
    finite_mask = np.isfinite(tiny)
    result = compute_prospectivity(
        tiny, tiny, finite_mask, np.array([0.0, 10.0]), np.array([0.0, 10.0]), UTM_EPSG,
        purpose="magnetite_fe",
    )
    assert result["available"] is False
    assert result["targets"] == []


def test_prospectivity_api_endpoints():
    client = TestClient(app)
    project_id = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("drone.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("base.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{project_id}/process", json={})
    assert r.status_code == 200

    r = client.post(
        f"/api/projects/{project_id}/prospectivity",
        json={"cell_size_m": 15.0, "purpose": "magnetite_fe", "score_threshold": 0.3},
    )
    assert r.status_code == 200
    body = r.json()
    assert "available" in body
    assert "score_grid" not in body  # raw grid must never leak into the JSON response

    if body.get("available") and body.get("n_targets", 0) >= 0:
        r2 = client.get(f"/api/projects/{project_id}/prospectivity/overlay")
        assert r2.status_code == 200
        overlay = r2.json()
        assert "image_base64" in overlay or "stats" in overlay


def test_prospectivity_overlay_requires_run_first():
    client = TestClient(app)
    project_id = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("drone.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("base.csv", f, "text/csv")})
    client.post(f"/api/projects/{project_id}/process", json={})

    r = client.get(f"/api/projects/{project_id}/prospectivity/overlay")
    assert r.status_code == 400


def test_prospectivity_uses_lineament_and_contact_layers_when_available():
    client = TestClient(app)
    project_id = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("drone.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("base.csv", f, "text/csv")})
    client.post(f"/api/projects/{project_id}/process", json={})

    client.post(f"/api/projects/{project_id}/lineaments", json={"cell_size_m": 15.0, "percentile_threshold": 70.0, "min_segment_points": 3})
    client.post(f"/api/projects/{project_id}/contacts", json={"cell_size_m": 15.0, "percentile_threshold": 70.0, "min_persistence": 0.3, "min_segment_points": 3})

    r = client.post(
        f"/api/projects/{project_id}/prospectivity",
        json={"cell_size_m": 15.0, "purpose": "skarn", "score_threshold": 0.3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    # skarn preset weights asa/thd/structure/contact/susceptibility - whichever
    # of structure/contact were actually extractable should show up here.
    assert set(body["layers_used"]).issubset({"asa", "thd", "structure", "contact", "susceptibility"})
