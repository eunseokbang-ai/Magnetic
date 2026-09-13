"""Tests for the two mineral-exploration interpretation additions:
magnetic lineament extraction + rose-diagram statistics
(processing/lineaments.py), and the quick tilt-depth/analytic-signal-
depth/spectral-depth estimators (processing/depth_estimation.py). Both
are exercised on a synthetic NE-SW-striking vertical contact (a
standard textbook arctan step-response anomaly), which gives a known
ground truth for strike direction and lets the depth estimators be
checked against the model's actual depth.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.processing.depth_estimation import (
    analytic_signal_depth_estimates,
    spectral_depth_diagnostic,
    tilt_depth_estimates,
)
from app.processing.gridding import grid_points
from app.processing.lineaments import extract_lineaments
from app.processing.transforms import total_horizontal_derivative

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"
UTM_EPSG = 32652


def _synthetic_contact(depth_m=30.0, seed=0):
    """A single vertical contact striking NE-SW (45 deg azimuth), whose
    TMI response is the standard 2D arctan step model - amplitude and
    depth are both known ground truth."""
    rng = np.random.default_rng(seed)
    n = 6000
    x = rng.uniform(0, 1000, n)
    y = rng.uniform(0, 1000, n)
    dist_to_contact = (y - x) / np.sqrt(2)  # signed perpendicular distance to the y=x line
    values = 200.0 * np.arctan(dist_to_contact / depth_m) + rng.normal(0, 0.5, n)
    grid = grid_points(x, y, values, cell_size_m=10.0, method="linear")
    return grid


def test_lineament_extraction_recovers_ne_sw_strike():
    grid = _synthetic_contact()
    thd = total_horizontal_derivative(grid.values, grid.cell_size_m)
    result = extract_lineaments(thd, grid.easting, grid.northing, grid.cell_size_m, UTM_EPSG, percentile_threshold=90, min_segment_points=3)
    assert result["available"] is True
    assert result["n_lineaments"] > 0

    # length-weighted mean strike should land near the true 45 deg (NE-SW)
    total_len = sum(l["length_m"] for l in result["lineaments"])
    mean_strike = sum(l["strike_deg"] * l["length_m"] for l in result["lineaments"]) / total_len
    assert abs(mean_strike - 45.0) < 20.0

    ne_sw = next(q for q in result["quadrant_summary"] if q["name"] == "NE-SW")
    assert ne_sw["pct_of_total_length"] > 40.0, "NE-SW should dominate the rose-diagram summary"

    for l in result["lineaments"]:
        for lat, lon in l["points_latlon"]:
            assert np.isfinite(lat) and np.isfinite(lon)


def test_lineament_extraction_unavailable_on_too_small_grid():
    tiny = np.array([[1.0, 2.0], [3.0, 4.0]])
    easting = np.array([0.0, 10.0])
    northing = np.array([0.0, 10.0])
    result = extract_lineaments(tiny, easting, northing, 10.0, UTM_EPSG)
    assert result["available"] is False
    assert "reason" in result


def test_lineament_extraction_endpoint_via_api():
    client = TestClient(app)
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{project_id}/process", json={"line_params": {}, "diurnal_params": {}, "heading_correction": {}})
    assert r.status_code == 200, r.text

    r = client.post(
        f"/api/projects/{project_id}/lineaments",
        json={"value": "thd", "cell_size_m": 10.0, "method": "nearest", "percentile_threshold": 85.0, "min_segment_points": 3},
    )
    assert r.status_code == 200, r.text
    resp = r.json()
    assert "available" in resp and "rose_bins" in resp


def test_tilt_depth_recovers_known_contact_depth():
    grid = _synthetic_contact(depth_m=30.0)
    result = tilt_depth_estimates(grid.values, grid.easting, grid.northing, grid.cell_size_m, UTM_EPSG)
    assert result["available"] is True
    assert result["n_points"] > 0
    # tilt-depth is exact for this idealized 2D contact model - median
    # pick should land close to the true 30m depth.
    assert abs(result["depth_stats"]["median"] - 30.0) < 15.0


def test_analytic_signal_depth_recovers_known_contact_depth():
    grid = _synthetic_contact(depth_m=30.0)
    result = analytic_signal_depth_estimates(grid.values, grid.easting, grid.northing, grid.cell_size_m, UTM_EPSG)
    assert result["available"] is True
    assert result["n_points"] > 0
    assert abs(result["depth_stats"]["median"] - 30.0) < 20.0


def test_spectral_depth_diagnostic_returns_positive_depth_and_spectrum_points():
    grid = _synthetic_contact(depth_m=30.0)
    result = spectral_depth_diagnostic(grid.values, grid.cell_size_m)
    assert result["available"] is True
    assert result["depth_m"] > 0
    assert len(result["wavenumbers_rad_per_m"]) == len(result["ln_power"]) > 0


def test_depth_estimation_unavailable_on_tiny_grid():
    tiny = np.array([[1.0, 2.0], [3.0, 4.0]])
    easting = np.array([0.0, 10.0])
    northing = np.array([0.0, 10.0])
    assert tilt_depth_estimates(tiny, easting, northing, 10.0, UTM_EPSG)["available"] is False
    assert analytic_signal_depth_estimates(tiny, easting, northing, 10.0, UTM_EPSG)["available"] is False
    assert spectral_depth_diagnostic(tiny, 10.0)["available"] is False


def test_depth_estimation_endpoints_via_api():
    client = TestClient(app)
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{project_id}/process", json={"line_params": {}, "diurnal_params": {}, "heading_correction": {}})
    assert r.status_code == 200, r.text

    body = {"value": "anomaly", "cell_size_m": 20.0, "method": "nearest"}
    r = client.post(f"/api/projects/{project_id}/depth-estimation/tilt", json=body)
    assert r.status_code == 200, r.text
    assert "available" in r.json()

    r = client.post(f"/api/projects/{project_id}/depth-estimation/analytic-signal", json=body)
    assert r.status_code == 200, r.text
    assert "available" in r.json()

    r = client.post(f"/api/projects/{project_id}/depth-estimation/spectral", json=body)
    assert r.status_code == 200, r.text
    assert "available" in r.json()
