"""A GeoTIFF on disk served as XYZ tiles (magcore.local_raster) and its
endpoints: register, tile, unregister."""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_origin

from app.main import app
from magcore import local_raster


def _tile_xy(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def _geotiff(path, crs="EPSG:4326", n=400, lon0=128.49, lat0=36.53, px=0.00001, bands=3):
    data = np.zeros((bands, n, n), dtype="uint8")
    data[0] = 200
    data[1] = 90
    data[2] = 40
    if bands == 4:
        data[3] = 255
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=bands, dtype="uint8", crs=crs,
                       transform=from_origin(lon0, lat0, px, px), tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(data)
    return path


@pytest.fixture(autouse=True)
def _cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(local_raster.paths, "data_dir", lambda: tmp_path / "data")


def test_register_and_tiles_inside_and_outside(tmp_path):
    tif = _geotiff(tmp_path / "ortho.tif")
    layer = local_raster.register_local_raster(str(tif), "ortho")
    assert layer.crs == "EPSG:4326" and layer.bands == 3 and layer.has_overviews is False
    south, west, north, east = layer.bounds
    assert abs(west - 128.49) < 1e-6 and abs(north - 36.53) < 1e-6 and south < north and east > west
    assert 17 <= layer.max_zoom <= 21 and 0.5 < layer.res_m < 1.2        # 1e-5 deg ~ 0.9 m
    z = layer.max_zoom - 2
    x, y = _tile_xy(36.528, 128.492, z)                                 # inside
    png = local_raster.get_tile(layer.id, z, x, y)
    assert png and png[:8] == b"\x89PNG\r\n\x1a\n"
    im = np.asarray(Image.open(__import__("io").BytesIO(png)))
    assert im.shape == (256, 256, 4) and im[..., 3].max() == 255
    assert abs(int(im[im[..., 3] > 0][:, 0].mean()) - 200) < 3            # the red band came through
    assert local_raster.get_tile(layer.id, z, x + 50, y + 50) is None   # far outside: empty
    # cached on disk after the first render
    cached = list((tmp_path / "data" / "raster_tiles").rglob("*.png"))
    assert len(cached) == 1
    local_raster.unregister_local_raster(layer.id)
    assert local_raster.get_tile(layer.id, z, x, y) is None


def test_projected_crs_and_rgba(tmp_path):
    # Korea 2000 / East Belt around the Andong block, 4 bands with alpha
    tif = _geotiff(tmp_path / "ortho5187.tif", crs="EPSG:5187", n=300, lon0=152600, lat0=437400, px=0.5, bands=4)
    layer = local_raster.register_local_raster(str(tif))
    south, west, north, east = layer.bounds
    assert 36.5 < south < north < 36.6 and 128.4 < west < east < 128.6
    assert abs(layer.res_m - 0.5) < 1e-6 and layer.label == "ortho5187"
    z = layer.max_zoom
    x, y = _tile_xy((south + north) / 2, (west + east) / 2, z)
    assert local_raster.get_tile(layer.id, z, x, y) is not None
    local_raster.unregister_local_raster(layer.id)


def test_api_register_tile_and_errors(tmp_path):
    client = TestClient(app)
    tif = _geotiff(tmp_path / "api.tif")
    r = client.post("/api/local-rasters/register", json={"path": str(tif), "label": "시험 정사영상"})
    assert r.status_code == 200, r.text
    layer = r.json()
    assert layer["label"] == "시험 정사영상" and layer["bounds"][0][0] < layer["bounds"][1][0]
    z = layer["max_zoom"] - 2
    x, y = _tile_xy(36.528, 128.492, z)
    t = client.get(f"/api/local-rasters/{layer['id']}/{z}/{x}/{y}")
    assert t.status_code == 200 and t.headers["content-type"] == "image/png"
    assert client.get(f"/api/local-rasters/{layer['id']}/{z}/{x + 50}/{y + 50}").status_code == 204
    assert client.get(f"/api/local-rasters/nope/{z}/{x}/{y}").status_code == 404
    assert client.delete(f"/api/local-rasters/{layer['id']}").status_code == 200
    bad = client.post("/api/local-rasters/register", json={"path": str(tmp_path / "missing.tif")})
    assert bad.status_code == 400 and "찾을 수 없습니다" in bad.json()["detail"]
