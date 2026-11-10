"""Exports in a chosen coordinate system, and the Geosoft XYZ and
GeoPackage formats (magcore/export_formats.py)."""
import io
import pathlib
import sqlite3
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from pyproj import Transformer

from app.main import app
from app.store import store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"
GRID = {"value": "anomaly", "cell_size_m": 10.0}


@pytest.fixture(scope="module")
def project():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files=[("files", ("d.csv", f.read(), "text/csv"))])
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files=[("files", ("b.csv", f.read(), "text/csv"))])
    assert client.post(f"/api/projects/{pid}/process", json={}).status_code == 200
    return client, pid, store.get(pid)


def _raster(content):
    with rasterio.open(io.BytesIO(content)) as ds:
        a = ds.read(1, masked=True).astype(float).filled(np.nan)
        return ds.crs.to_epsg(), ds.bounds, a, ds.res


def test_geotiff_leaves_in_the_requested_crs_with_the_same_field(project):
    client, pid, p = project
    native = client.post(f"/api/projects/{pid}/grid/geotiff", json=GRID)
    epsg0, bounds0, a0, _ = _raster(native.content)
    assert epsg0 == p.utm_epsg
    for epsg in (4326, 32649, 5186):                        # geographic, the neighbouring UTM zone, a Korean TM belt
        r = client.post(f"/api/projects/{pid}/grid/geotiff", json={**GRID, "export_epsg": epsg})
        assert r.status_code == 200, r.text
        got, bounds, a, res = _raster(r.content)
        assert got == epsg
        # the same ground: the native corners, transformed, lie on the reprojected extent
        t = Transformer.from_crs(f"EPSG:{epsg0}", f"EPSG:{epsg}", always_xy=True)
        xs, ys = t.transform([bounds0.left, bounds0.right, bounds0.left, bounds0.right], [bounds0.bottom, bounds0.bottom, bounds0.top, bounds0.top])
        span = max(bounds.right - bounds.left, bounds.top - bounds.bottom)
        assert bounds.left == pytest.approx(min(xs), abs=0.02 * span) and bounds.top == pytest.approx(max(ys), abs=0.02 * span)
        # the same field: resampled, so cell for cell it differs, but its statistics hold
        assert np.nanmedian(a) == pytest.approx(np.nanmedian(a0), abs=0.02 * np.nanstd(a0))
        assert np.nanstd(a) == pytest.approx(np.nanstd(a0), rel=0.03)
        if epsg != 4326:
            # the same ground covered: cells x cell area (a degree grid has other cells, so it is left out here)
            assert np.isfinite(a).sum() * res[0] * res[1] == pytest.approx(np.isfinite(a0).sum() * 100.0, rel=0.08)
    # asking for the project's own CRS changes nothing at all
    same = client.post(f"/api/projects/{pid}/grid/geotiff", json={**GRID, "export_epsg": p.utm_epsg})
    assert same.content == native.content
    r = client.post(f"/api/projects/{pid}/grid/geotiff", json={**GRID, "export_epsg": 999999})
    assert r.status_code in (400, 422)


def test_derived_and_coloured_grids_follow_too(project):
    client, pid, _ = project
    r = client.post(f"/api/projects/{pid}/transform/geotiff", json={**GRID, "transform": "rtp", "export_epsg": 4326})
    assert r.status_code == 200 and _raster(r.content)[0] == 4326
    r = client.post(f"/api/projects/{pid}/grid/geotiff", json={**GRID, "colored": True, "export_epsg": 4326})
    assert r.status_code == 200, r.text
    with rasterio.open(io.BytesIO(r.content)) as ds:
        assert ds.crs.to_epsg() == 4326 and ds.count >= 3 and ds.dtypes[0] == "uint8"


def test_xyz_keeps_every_cell_and_moves_only_the_coordinates(project):
    client, pid, p = project
    lonlat = client.post(f"/api/projects/{pid}/grid/xyz", json=GRID).content.decode().splitlines()
    assert not lonlat[0].startswith("#")                           # without a CRS it is what it always was: lon lat value
    out = client.post(f"/api/projects/{pid}/grid/xyz", json={**GRID, "export_epsg": 5186}).content.decode().splitlines()
    assert out[0].startswith("# EPSG:5186") and out[1] == "# x y value" and len(out) - 2 == len(lonlat)
    x, y, v = (float(t) for t in out[2].split())
    lon, lat, v0 = (float(t) for t in lonlat[0].split())
    ex, ey = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True).transform(lon, lat)
    assert (x, y) == pytest.approx((ex, ey), abs=0.05) and v == v0   # same cell, same value, new coordinates
    geo = client.post(f"/api/projects/{pid}/grid/xyz", json={**GRID, "export_epsg": 4326}).content.decode().splitlines()
    assert geo[1] == "# lon lat value" and float(geo[2].split()[0]) == pytest.approx(lon, abs=1e-6)


def test_surfer_grid_in_another_crs(project):
    client, pid, p = project
    native = client.post(f"/api/projects/{pid}/grid/grd", json=GRID).content
    warped = client.post(f"/api/projects/{pid}/grid/grd", json={**GRID, "export_epsg": 4326}).content
    assert native[:4] == warped[:4] == b"DSBB"
    xmin, xmax, ymin, ymax = struct.unpack("<4d", warped[8:40])
    assert 100.0 < xmin < xmax < 115.0 and 40.0 < ymin < ymax < 55.0      # degrees now, over the sample survey
    assert struct.unpack("<4d", native[8:40])[0] > 1000.0                 # metres before


def test_geopackage_grid_opens_as_a_raster(project, tmp_path):
    client, pid, _ = project
    r = client.post(f"/api/projects/{pid}/grid/gpkg", json={**GRID, "export_epsg": 4326})
    assert r.status_code == 200, r.text
    path = tmp_path / "grid.gpkg"
    path.write_bytes(r.content)
    assert r.content[:15] == b"SQLite format 3"
    with rasterio.open(path) as ds:
        a = ds.read(1, masked=True).astype(float).filled(np.nan)
        assert ds.crs.to_epsg() == 4326 and ds.driver == "GPKG" and np.isfinite(a).sum() > 100
    ref = _raster(client.post(f"/api/projects/{pid}/grid/geotiff", json={**GRID, "export_epsg": 4326}).content)[2]
    assert np.nanmedian(a) == pytest.approx(np.nanmedian(ref), abs=0.5)
    r = client.post(f"/api/projects/{pid}/transform/gpkg", json={**GRID, "transform": "1vd"})
    assert r.status_code == 200 and r.content[:15] == b"SQLite format 3"


def test_geosoft_xyz_is_line_blocks_of_readings(project):
    client, pid, p = project
    text = client.get(f"/api/projects/{pid}/points/geosoft-xyz?epsg=5186").content.decode()
    lines = text.splitlines()
    assert lines[0].startswith("/ DroneMag Studio - EPSG:5186") and lines[1].startswith("/") and "ANOMALY_NT" in lines[1]
    headers = [l for l in lines if l.startswith("Line ")]
    assert len(headers) == 6 and headers[0] == "Line 1"                 # the sample survey's six lines
    rows = [l.split() for l in lines if l and l[0] not in "/LT"]
    assert all(len(r) == 8 for r in rows)
    active = p.processed.loc[p._active_mask()]
    assert len(rows) == len(active)                                     # every reading on a line, and only those
    first = active[active["line_id"] == active["line_id"].min()].iloc[0]
    ex, ey = Transformer.from_crs(f"EPSG:{p.utm_epsg}", "EPSG:5186", always_xy=True).transform(first["x"], first["y"])
    assert (float(rows[0][0]), float(rows[0][1])) == pytest.approx((ex, ey), abs=0.02)
    assert float(rows[0][4]) == pytest.approx(first["anomaly"], abs=0.001)


def test_geopackage_points_is_a_valid_feature_layer(project, tmp_path):
    client, pid, p = project
    r = client.get(f"/api/projects/{pid}/points/gpkg?epsg=4326")
    assert r.status_code == 200, r.text
    path = tmp_path / "points.gpkg"
    path.write_bytes(r.content)
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA application_id").fetchone()[0] == 1196444487          # "GPKG"
    assert con.execute("SELECT table_name, data_type, srs_id FROM gpkg_contents").fetchall() == [("points", "features", 4326)]
    assert con.execute("SELECT column_name, geometry_type_name, srs_id FROM gpkg_geometry_columns").fetchall() == [("geom", "POINT", 4326)]
    assert con.execute("SELECT COUNT(*) FROM points").fetchone()[0] == len(p.processed)
    geom, lat, lon, anomaly, excluded = con.execute("SELECT geom, lat, lon, anomaly_nt, excluded FROM points ORDER BY fid LIMIT 1").fetchone()
    assert geom[:2] == b"GP" and struct.unpack("<i", geom[4:8])[0] == 4326
    byte_order, wkb_type, x, y = struct.unpack("<BIdd", geom[8:])
    assert (byte_order, wkb_type) == (1, 1) and (x, y) == pytest.approx((lon, lat), abs=1e-9)
    assert con.execute("SELECT COUNT(*) FROM points WHERE excluded").fetchone()[0] == int((~p._active_mask()).sum())
    con.close()
    # the project's own CRS when none is asked for
    path.write_bytes(client.get(f"/api/projects/{pid}/points/gpkg").content)
    con = sqlite3.connect(path)
    assert con.execute("SELECT srs_id FROM gpkg_contents").fetchone()[0] == p.utm_epsg
    con.close()
