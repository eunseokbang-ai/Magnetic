"""Basemaps (GeoTIFF overlay, local tile folders), the derived grid kinds,
RTP as a picking surface, and the top-N cut."""
from __future__ import annotations

import io

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from app.main import app
from app.processing.peaks import KIND_LABELS, derivative_grid
from app.processing.targets import pick_targets
from conftest import R3_FIXTURE
from test_targets import DEC, INC, PROBE_H, TRUTH, _survey


def _geotiff_bytes(lon0=127.3609, lat0=36.3792, n=40) -> bytes:
    """A small RGB GeoTIFF in WGS84 over the fixture's site."""
    data = (np.random.default_rng(0).integers(0, 255, (3, n, n))).astype("uint8")
    buf = io.BytesIO()
    with rasterio.open(buf, "w", driver="GTiff", height=n, width=n, count=3, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(lon0, lat0, 0.00001, 0.00001)) as dst:
        dst.write(data)
    return buf.getvalue()


def test_geotiff_basemap_round_trip_through_the_api():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    r = client.post(f"/api/projects/{pid}/basemaps", files={"file": ("ortho.tif", _geotiff_bytes(), "image/tiff")})
    assert r.status_code == 200, r.text
    assert [b["name"] for b in r.json()["basemaps"]] == ["ortho.tif"]
    assert [b["name"] for b in client.get(f"/api/projects/{pid}/basemaps").json()["basemaps"]] == ["ortho.tif"]
    full = client.get(f"/api/projects/{pid}/basemaps/ortho.tif").json()
    assert full["image_data_url"].startswith("data:image/png;base64,")
    (south, west), (north, east) = full["bounds"]
    assert 36.37 < south < north < 36.38 and 127.36 < west < east < 127.37
    assert client.delete(f"/api/projects/{pid}/basemaps/ortho.tif").json()["basemaps"] == []
    bad = client.post(f"/api/projects/{pid}/basemaps", files={"file": ("x.tif", b"not a tiff", "image/tiff")})
    assert bad.status_code == 400


def test_local_tile_folder_is_served_from_disk(tmp_path):
    folder = tmp_path / "tiles"
    (folder / "18" / "223900").mkdir(parents=True)
    png = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (256, 256), (10, 200, 10)).save(png, format="PNG")
    (folder / "18" / "223900" / "102400.png").write_bytes(png.getvalue())
    client = TestClient(app)
    r = client.post("/api/local-tiles/register", json={"path": str(folder), "label": "ortho"})
    assert r.status_code == 200, r.text
    layer = r.json()
    assert layer["min_zoom"] == 18 and layer["max_zoom"] == 18
    tile = client.get(f"/api/local-tiles/{layer['id']}/18/223900/102400")
    assert tile.status_code == 200 and tile.headers["content-type"].startswith("image/png")
    assert client.get(f"/api/local-tiles/{layer['id']}/18/1/1").status_code == 404
    assert client.delete(f"/api/local-tiles/{layer['id']}").status_code == 200
    assert client.post("/api/local-tiles/register", json={"path": str(tmp_path / "nope")}).status_code == 400


def test_every_kind_renders_and_rtp_moves_the_maximum_over_the_source():
    from test_peaks import _grid
    g, e, n = _grid(noise=0.5)
    for kind in KIND_LABELS:
        d = derivative_grid(g, 0.1, kind, INC, DEC)
        assert d.shape == g.shape and np.isfinite(d).any()
    rtp = derivative_grid(g, 0.1, "rtp", INC, DEC)
    r, c = np.unravel_index(np.nanargmax(rtp), rtp.shape)
    assert np.hypot(e[c] - 3.0, n[r] - 3.0) < 0.25          # over the big dipole
    with pytest.raises(ValueError):
        derivative_grid(g, 0.1, "rtp")                         # needs the field direction


def test_rtp_peaks_pick_the_dipoles_and_the_top_n_cut_keeps_the_strongest():
    df = _survey(1)
    r = pick_targets(df, INC, DEC, 32652, noise_by_file={0: 3.0}, sensor_height_m=PROBE_H, derivative="rtp")
    assert r["n_targets"] >= 3
    for sx, sy, _d, _m in TRUTH:
        assert min(np.hypot(t["x"] - sx, t["y"] - sy) for t in r["targets"]) < 0.3
    two = pick_targets(df, INC, DEC, 32652, noise_by_file={0: 3.0}, sensor_height_m=PROBE_H, max_targets=2)
    assert two["n_targets"] == 2 and two["n_before_cut"] >= 3 and two["max_targets"] == 2
    ks = [t["k_needed"] for t in two["targets"]]
    assert ks == sorted(ks, reverse=True)


def test_the_api_accepts_the_new_kinds_and_the_cut():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(R3_FIXTURE, "rb") as f:
        client.post(f"/api/projects/{pid}/upload", files={"files": (R3_FIXTURE.name, f, "text/plain")})
    for kind in ("rtp", "tilt", "vd1"):
        g = client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.2, "kind": kind})
        assert g.status_code == 200, g.text
        assert g.json()["stats"]["kind"] == kind
    t = client.post(f"/api/projects/{pid}/targets", json={"derivative": "rtp", "max_targets": 1, "sensor_height_m": 0.3})
    assert t.status_code == 200, t.text
    assert t.json()["n_targets"] <= 1
