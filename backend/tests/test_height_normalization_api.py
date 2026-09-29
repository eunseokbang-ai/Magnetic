"""Height normalization through the project and the API.

What has to hold beyond the fit itself: apply then reset gives the data
back exactly, a saved project comes back normalized, and the structure
models sit on top of it - either can be undone without the other.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_origin

from app.main import app
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


def _make_hilly(project, relief=25.0):
    """The fixture was flown level. Give it terrain-following heights and
    a field that depends on them, so there is something to normalize."""
    df = project.processed_base
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    cx, cy = float(x.mean()), float(y.mean())
    hill = relief * np.sin(2 * np.pi * (x - cx) / 400.0) * np.cos(2 * np.pi * (y - cy) / 300.0)
    df["altitude_ellipsoidal_m"] = df["altitude_ellipsoidal_m"].to_numpy() + hill
    inc, dec = np.radians(project.inclination_deg), np.radians(project.declination_deg)
    f = np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)])
    z0 = float(np.nanmedian(df["altitude_ellipsoidal_m"]))
    rx, ry, rz = x - cx, y - cy, df["altitude_ellipsoidal_m"].to_numpy() - (z0 - 120.0)
    r2 = rx**2 + ry**2 + rz**2
    fr = f[0] * rx + f[1] * ry + f[2] * rz
    field = 100.0 * 3e6 * (3 * fr * fr - r2) / (r2**2 * np.sqrt(r2))
    for col in ("anomaly", "tmi"):
        df[col] = df[col].to_numpy() + field
    project.processed_base = df
    project._rebuild_processed()


def test_analyze_reports_the_flight_and_apply_reset_is_exact():
    client, pid, project = _processed_project()
    _make_hilly(project)
    before = project.processed[["anomaly", "tmi"]].to_numpy().copy()

    r = client.post(f"/api/projects/{pid}/height-normalization")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] and body["applied"] is False
    assert body["altitude"]["std"] > 5.0
    assert body["suggested_z_ref_m"] == body["altitude"]["median"]
    assert body["agl"] is None                             # no DEM loaded

    r = client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    assert r.status_code == 200, r.text
    summary = r.json()["height_normalization"]
    assert summary["applied"] and summary["n_sources"] > 0 and summary["converged"]
    after = project.processed[["anomaly", "tmi"]].to_numpy()
    shift = after - before
    assert np.abs(shift[:, 0]).max() > 0.5
    assert np.allclose(shift[:, 0], shift[:, 1])           # the same correction on anomaly and tmi
    assert client.post(f"/api/projects/{pid}/height-normalization").json()["applied"] is True

    r = client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "reset"})
    assert r.json()["height_normalization"] == {"applied": False}
    assert np.array_equal(project.processed[["anomaly", "tmi"]].to_numpy(), before)


def test_a_chosen_reference_height_is_used():
    client, pid, project = _processed_project()
    _make_hilly(project)
    z_top = float(np.nanmax(project.processed_base["altitude_ellipsoidal_m"]))

    r = client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply", "z_ref_m": z_top})

    assert r.json()["height_normalization"]["z_ref_m"] == round(z_top, 2)


def test_a_saved_project_comes_back_normalized():
    client, pid, project = _processed_project()
    _make_hilly(project)
    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    normalized = project.processed["anomaly"].to_numpy().copy()
    summary = project.height_normalization.summary()

    bundle = project.save_project_bundle()
    other = project_store.create()
    other.load_project_bundle(bundle)

    assert other.height_normalization is not None
    restored = other.height_normalization.summary()
    assert restored["z_ref_m"] == summary["z_ref_m"] and restored["n_sources"] == summary["n_sources"]
    assert restored["fit_rms_nt"] == summary["fit_rms_nt"] and restored["warnings"] == summary["warnings"]
    for key in ("median_abs_nt", "max_abs_nt", "rms_nt"):   # stored as float32
        assert abs(restored["correction"][key] - summary["correction"][key]) < 1e-2
    # The fixture's pipeline replay gives the same base, and the stored
    # correction is applied on top of it - not refitted.
    assert np.allclose(
        other.processed["anomaly"].to_numpy() - other.processed_base["anomaly"].to_numpy(),
        project.height_normalization.correction_nt, atol=1e-3,
    )
    assert other.process_summary()["height_normalization"]["applied"] is True
    del normalized


def test_structure_removal_sits_on_top_and_each_undoes_alone():
    client, pid, project = _processed_project()
    _make_hilly(project)
    base = project.processed["anomaly"].to_numpy().copy()

    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    normalized = project.processed["anomaly"].to_numpy().copy()
    correction = project.height_normalization.correction_nt.copy()

    # A structure fitted on the normalized readings
    df = project.processed_base
    cx, cy = float(df["x"].median()), float(df["y"].median())
    to_ll = Transformer.from_crs(f"EPSG:{project.utm_epsg}", "EPSG:4326", always_xy=True)
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    lon, lat = to_ll.transform(cx + 30 * np.cos(t), cy + 30 * np.sin(t))
    polygon = [[float(a), float(o)] for a, o in zip(lat, lon)]
    r = client.post(f"/api/projects/{pid}/source-removal", json={"mode": "add", "polygons": [polygon]})
    assert r.status_code == 200, r.text
    removed = normalized - project.processed["anomaly"].to_numpy()
    assert np.abs(removed).max() > 0.0

    # Undo the normalization: the structure model stays subtracted
    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "reset"})
    assert project.height_normalization is None
    assert len(project.source_removals) == 1
    assert np.allclose(project.processed["anomaly"].to_numpy(), base - removed, atol=1e-6)

    # Put it back, then undo the structure model: the normalization stays
    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    client.post(f"/api/projects/{pid}/source-removal", json={"mode": "reset"})
    assert project.height_normalization is not None
    assert np.allclose(project.processed["anomaly"].to_numpy(), base + correction, atol=1e-6)


def test_the_report_and_summary_carry_it():
    client, pid, project = _processed_project()
    _make_hilly(project)
    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})

    report = project.generate_report()

    assert "비행 고도 정규화" in report
    assert f"기준 고도 {project.height_normalization.z_ref_m:.2f} m" in report


def test_rerunning_the_pipeline_clears_it():
    client, pid, project = _processed_project()
    _make_hilly(project)
    client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})

    r = client.post(f"/api/projects/{pid}/process", json={"line_params": {}, "diurnal_params": {}})

    assert r.json()["height_normalization"] == {"applied": False}
    assert project.height_normalization is None


def test_without_heights_it_says_so_instead_of_guessing():
    client, pid, project = _processed_project()
    project.processed_base["altitude_ellipsoidal_m"] = np.nan
    project._rebuild_processed()

    body = client.post(f"/api/projects/{pid}/height-normalization").json()
    assert body["available"] is False and "고도" in body["reason"]

    r = client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    assert r.status_code == 400
    assert "고도" in r.json()["detail"]


def test_with_a_dem_the_height_above_ground_is_reported(tmp_path):
    client, pid, project = _processed_project()
    df = project.processed
    xmin, xmax = df["x"].min() - 500, df["x"].max() + 500
    ymin, ymax = df["y"].min() - 500, df["y"].max() + 500
    dem_path = str(tmp_path / "dem.tif")
    w, h = 40, 40
    transform = from_origin(xmin, ymax, (xmax - xmin) / w, (ymax - ymin) / h)
    ground = float(np.nanmedian(df["altitude_msl_m"])) - 60.0
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs=CRS.from_epsg(project.utm_epsg), transform=transform,
    ) as dst:
        dst.write(np.full((h, w), ground, dtype="float32"), 1)
    with open(dem_path, "rb") as f:
        r = client.post(f"/api/projects/{pid}/upload/dem", files={"file": ("dem.tif", f, "image/tiff")})
    assert r.status_code == 200, r.text

    body = client.post(f"/api/projects/{pid}/height-normalization").json()

    agl = body["agl"]
    assert agl["datum"] == "msl"
    assert agl["n"] > 1000
    assert abs(agl["median"] - 60.0) < 5.0            # the sample of active readings vs. the median of all of them
