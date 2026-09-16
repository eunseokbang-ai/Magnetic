"""Structure removal by modelling, through the project and the API.

What has to hold beyond the fit itself: a removal can be undone or cleared
without touching the point smoothing, the two compose, and a saved project
comes back with exactly the removal it was saved with.
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


def _processed_project():
    client = TestClient(app)
    project_id = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{project_id}/process", json={"line_params": {}, "diurnal_params": {}})
    assert r.status_code == 200, r.text
    return client, project_id, project_store.get(project_id)


def _add_structure(project, strength=4e4):
    """Plant a compact source in the processed data, so there is something
    real to remove, and return a polygon round it in lat/lon."""
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
    return [[float(a), float(o)] for a, o in zip(lat, lon)], field.to_numpy()


def test_add_undo_reset():
    client, pid, project = _processed_project()
    polygon, field = _add_structure(project)
    before = project.processed["anomaly"].to_numpy().copy()

    r = client.post(f"/api/projects/{pid}/source-removal", json={"mode": "add", "polygons": [polygon]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_source_removals"] == 1
    assert body["added"][0]["n_sources"] > 0
    assert len(body["source_removals"]) == 1
    removed = before - project.processed["anomaly"].to_numpy()
    assert np.abs(removed).max() > 0.5 * np.abs(field).max()

    r = client.post(f"/api/projects/{pid}/source-removal", json={"mode": "undo"})
    assert r.json()["n_source_removals"] == 0
    assert np.allclose(project.processed["anomaly"].to_numpy(), before)

    client.post(f"/api/projects/{pid}/source-removal", json={"mode": "add", "polygons": [polygon]})
    client.post(f"/api/projects/{pid}/source-removal", json={"mode": "add", "polygons": [polygon]})
    r = client.post(f"/api/projects/{pid}/source-removal", json={"mode": "reset"})
    assert r.json()["n_source_removals"] == 0
    assert np.allclose(project.processed["anomaly"].to_numpy(), before)


def test_point_smoothing_applies_on_top_and_each_undoes_separately():
    client, pid, project = _processed_project()
    polygon, _ = _add_structure(project)
    before = project.processed["anomaly"].to_numpy().copy()
    client.post(f"/api/projects/{pid}/source-removal", json={"mode": "add", "polygons": [polygon]})
    after_model = project.processed["anomaly"].to_numpy().copy()

    # a stretch of one real line through the middle of the survey
    df = project.processed
    on_line = df[df["line_id"] >= 0]
    line = on_line["line_id"].value_counts().index[0]
    stretch = on_line[on_line["line_id"] == line].sort_values("timestamp")
    ids = [int(i) for i in stretch["point_id"].iloc[len(stretch) // 3: len(stretch) // 3 + 60]]
    client.post(f"/api/projects/{pid}/smooth", json={"mode": "point_ids", "point_ids": ids})
    both = project.processed["anomaly"].to_numpy().copy()
    assert not np.allclose(both, after_model)

    # undoing the model keeps the smoothing
    client.post(f"/api/projects/{pid}/source-removal", json={"mode": "undo"})
    assert project.manual_smooth_point_ids
    untouched = ~project.processed["point_id"].isin(ids).to_numpy()
    assert np.allclose(project.processed["anomaly"].to_numpy()[untouched], before[untouched])

    # and resetting the smoothing keeps nothing but the pristine data
    client.post(f"/api/projects/{pid}/smooth", json={"mode": "reset"})
    assert np.allclose(project.processed["anomaly"].to_numpy(), before)


def test_a_saved_project_comes_back_with_the_same_removal():
    client, pid, project = _processed_project()
    polygon, _ = _add_structure(project)
    client.post(f"/api/projects/{pid}/source-removal", json={"mode": "add", "polygons": [polygon]})
    model = project.source_removals[0]
    xs = project.processed["x"].to_numpy()[::50]
    ys = project.processed["y"].to_numpy()[::50]

    bundle = project.save_project_bundle()
    other = Project(id="reloaded")
    other.load_project_bundle(bundle)

    assert len(other.source_removals) == 1
    assert np.allclose(other.source_removals[0].field_at(xs, ys), model.field_at(xs, ys))


def test_reprocessing_starts_clean():
    client, pid, project = _processed_project()
    polygon, _ = _add_structure(project)
    client.post(f"/api/projects/{pid}/source-removal", json={"mode": "add", "polygons": [polygon]})
    assert project.source_removals

    r = client.post(f"/api/projects/{pid}/process", json={"line_params": {}, "diurnal_params": {}})
    assert r.status_code == 200
    assert project.source_removals == []
    assert r.json().get("source_removals") == []


def test_add_without_polygons_is_a_clear_error():
    client, pid, _ = _processed_project()
    r = client.post(f"/api/projects/{pid}/source-removal", json={"mode": "add"})
    assert r.status_code == 400
