"""Tests InversionParams.dem_geoid_offset_m: reconciles a DEM's usual
orthometric height with the drone GPS's ellipsoidal altitude (the two
otherwise sit in different vertical datums, silently offsetting the mesh's
ground surface relative to the flight track - see models.py's docstring
for the offset field)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.crs import CRS
from rasterio.transform import from_origin

from app.main import app
from app.store import store as project_store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def _make_processed_project_with_dem(tmp_path):
    client = TestClient(app)
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{project_id}/process", json={"line_params": {}, "diurnal_params": {}, "heading_correction": {}})
    assert r.status_code == 200, r.text

    project = project_store.get(project_id)
    df = project.processed
    xmin, xmax = df["x"].min() - 500, df["x"].max() + 500
    ymin, ymax = df["y"].min() - 500, df["y"].max() + 500
    dem_path = str(tmp_path / "dem.tif")
    w, h = 40, 40
    transform = from_origin(xmin, ymax, (xmax - xmin) / w, (ymax - ymin) / h)
    crs = CRS.from_epsg(project.utm_epsg)
    elev = np.full((h, w), 50.0, dtype="float32")
    with rasterio.open(dem_path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=crs, transform=transform) as dst:
        dst.write(elev, 1)

    with open(dem_path, "rb") as f:
        r = client.post(f"/api/projects/{project_id}/upload/dem", files={"file": ("dem.tif", f, "image/tiff")})
    assert r.status_code == 200, r.text
    return client, project_id


def test_dem_geoid_offset_shifts_mesh_ground_elevation(tmp_path):
    client, project_id = _make_processed_project_with_dem(tmp_path)

    r = client.post(
        f"/api/projects/{project_id}/inversion",
        json={"obs_cell_size_m": 40.0, "depth_extent_m": 100.0, "n_layers": 4, "dem_geoid_offset_m": 0.0},
    )
    assert r.status_code == 200, r.text
    ground_no_offset = project_store.get(project_id).inversion_result.mesh.ground_elevation.copy()

    r = client.post(
        f"/api/projects/{project_id}/inversion",
        json={"obs_cell_size_m": 40.0, "depth_extent_m": 100.0, "n_layers": 4, "dem_geoid_offset_m": 25.0},
    )
    assert r.status_code == 200, r.text
    ground_with_offset = project_store.get(project_id).inversion_result.mesh.ground_elevation.copy()

    finite_both = np.isfinite(ground_no_offset) & np.isfinite(ground_with_offset)
    assert finite_both.any()
    diff = ground_with_offset[finite_both] - ground_no_offset[finite_both]
    assert np.allclose(diff, 25.0, atol=1e-6)


def test_dem_geoid_offset_default_is_zero_unchanged():
    """Omitting dem_geoid_offset_m must reproduce the original (uncorrected)
    behavior exactly - this is a backward-compatibility guarantee, not
    just a convenience default."""
    from app.models import InversionParams

    assert InversionParams().dem_geoid_offset_m == 0.0


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_dem_geoid_offset_shifts_mesh_ground_elevation(pathlib.Path(d))
    test_dem_geoid_offset_default_is_zero_unchanged()
    print("ALL CHECKS PASSED")
