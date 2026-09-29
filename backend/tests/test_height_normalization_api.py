"""Flight-height normalisation through the project and the API.

What has to hold beyond the fit itself: the correction goes in after the
flight levelling and before the structure models, so those compose and
undo independently; reset restores the readings exactly; a saved project
comes back normalised; and a survey without altitudes is refused with a
reason rather than normalised to nothing.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi.testclient import TestClient
from pyproj import Transformer

from app.main import app
from app.store import Project
from app.store import store as project_store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def _processed():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{pid}/process", json={"line_params": {}, "diurnal_params": {}})
    assert r.status_code == 200, r.text
    return client, pid, project_store.get(pid)


def test_status_reports_the_altitude_spread_before_anything_is_applied():
    client, pid, project = _processed()
    r = client.get(f"/api/projects/{pid}/height-normalization")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is True
    assert body["applied"] is False
    alt = body["altitude"]
    # the fixture survey really was flown over relief
    assert alt["range_m"] > 20
    assert alt["min_m"] <= body["z_ref_default_m"] <= alt["max_m"]
    assert body["agl"] is None                     # no DEM loaded
    assert project.process_summary()["height_normalization"] == {"applied": False}


def test_apply_changes_the_readings_and_reset_restores_them_exactly():
    client, pid, project = _processed()
    before = project.processed["anomaly"].to_numpy().copy()
    tmi_before = project.processed["tmi"].to_numpy().copy()

    r = client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applied"] is True
    hn = body["height_normalization"]
    assert hn["applied"] is True
    assert hn["n_sources"] > 0
    assert hn["correction_stats"]["max_abs_nt"] > 0
    assert len(hn["layers"]) == 3

    after = project.processed["anomaly"].to_numpy()
    correction = after - before
    assert np.abs(correction).max() > 0.01
    # the same shift goes on tmi, so anomaly and tmi stay consistent
    assert np.allclose(project.processed["tmi"].to_numpy() - tmi_before, correction)
    # the reference altitude defaults to the median altitude of the points in use
    status = client.get(f"/api/projects/{pid}/height-normalization").json()
    assert status["applied"] is True
    assert abs(status["z_ref_m"] - status["altitude"]["median_m"]) < 0.05

    r = client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "reset"})
    assert r.json()["applied"] is False
    assert np.array_equal(project.processed["anomaly"].to_numpy(), before)
    assert project.process_summary()["height_normalization"] == {"applied": False}


def test_a_chosen_reference_altitude_moves_the_readings_further():
    client, pid, project = _processed()
    base = project.processed["anomaly"].to_numpy().copy()
    z = project.processed["altitude_ellipsoidal_m"].to_numpy()

    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    at_median = project.processed["anomaly"].to_numpy() - base
    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply", "z_ref_m": float(z.max()) + 30})
    higher = project.processed["anomaly"].to_numpy() - base

    assert project.height_normalization.z_ref_m == float(z.max()) + 30
    assert np.abs(higher).mean() > np.abs(at_median).mean()


def _add_structure(project, strength=4e4):
    """Plant a compact source in the processed data and return a polygon
    round it in lat/lon (as tests/test_source_removal_api.py does)."""
    df = project.processed_base
    cx, cy = float(df["x"].median()), float(df["y"].median())
    inc, dec = np.radians(project.inclination_deg), np.radians(project.declination_deg)
    f = np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)])
    rx, ry, rz = df["x"] - cx, df["y"] - cy, 15.0
    r2 = rx**2 + ry**2 + rz**2
    m = strength * f
    mdotr = m[0] * rx + m[1] * ry + m[2] * rz
    field = 100.0 * sum(fk * (3 * mdotr * c / r2 - mk) / r2**1.5 for fk, c, mk in zip(f, (rx, ry, rz), m))
    for col in ("anomaly", "tmi"):
        df[col] = df[col] + field.to_numpy()
    project.processed = df.copy()
    project.grid_cache = {}
    to_ll = Transformer.from_crs(f"EPSG:{project.utm_epsg}", "EPSG:4326", always_xy=True)
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    lon, lat = to_ll.transform(cx + 30 * np.cos(t), cy + 30 * np.sin(t))
    return [[float(a), float(o)] for a, o in zip(lat, lon)]


def test_structure_removal_and_smoothing_compose_with_it_and_undo_independently():
    client, pid, project = _processed()
    polygon = _add_structure(project)

    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    normalised = project.processed["anomaly"].to_numpy().copy()

    # a structure model on top, then undone: back to the normalised data
    r = client.post(f"/api/projects/{pid}/source-removal", json={"mode": "add", "polygons": [polygon]})
    assert r.status_code == 200, r.text
    assert not np.allclose(project.processed["anomaly"].to_numpy(), normalised)
    client.post(f"/api/projects/{pid}/source-removal", json={"mode": "reset"})
    assert np.allclose(project.processed["anomaly"].to_numpy(), normalised)

    # point smoothing on top, then the normalisation undone under it: the
    # smoothing stays, applied to the un-normalised data
    ids = [int(i) for i in project.processed["point_id"].iloc[100:160]]
    client.post(f"/api/projects/{pid}/smooth", json={"mode": "point_ids", "point_ids": ids})
    smoothed_normalised = project.processed["anomaly"].to_numpy().copy()
    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "reset"})
    assert project.manual_smooth_point_ids
    assert not np.allclose(project.processed["anomaly"].to_numpy(), smoothed_normalised)
    client.post(f"/api/projects/{pid}/smooth", json={"mode": "reset"})
    project.height_normalization = None
    # and applying again reproduces the same normalised data
    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    assert np.allclose(project.processed["anomaly"].to_numpy(), normalised)


def test_the_normalisation_survives_a_save_and_reload():
    client, pid, project = _processed()
    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    normalised = project.processed["anomaly"].to_numpy().copy()
    z_ref = project.height_normalization.z_ref_m

    other = Project(id="reloaded")
    other.load_project_bundle(project.save_project_bundle())

    assert other.height_normalization is not None
    assert other.height_normalization.z_ref_m == z_ref
    assert other.process_summary()["height_normalization"]["applied"] is True
    assert np.allclose(other.processed["anomaly"].to_numpy(), normalised, atol=1e-4)


def test_reprocessing_clears_it():
    client, pid, project = _processed()
    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    assert project.height_normalization is not None
    client.post(f"/api/projects/{pid}/process", json={"line_params": {}, "diurnal_params": {}})
    assert project.height_normalization is None
    assert project.process_summary()["height_normalization"] == {"applied": False}


def test_a_survey_without_altitudes_is_refused_with_a_reason():
    client, pid, project = _processed()
    project.processed_base["altitude_ellipsoidal_m"] = np.nan
    project.processed["altitude_ellipsoidal_m"] = np.nan

    status = client.get(f"/api/projects/{pid}/height-normalization").json()
    assert status["available"] is False
    assert "고도" in status["reason"]

    r = client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    assert r.status_code == 400
    assert "고도" in r.json()["detail"]
    assert project.height_normalization is None


def test_the_report_mentions_it():
    client, pid, project = _processed()
    assert "비행 고도 정규화: 미적용" in project.generate_report()
    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    report = project.generate_report()
    assert "**비행 고도 정규화**" in report
    assert "기준 고도" in report
