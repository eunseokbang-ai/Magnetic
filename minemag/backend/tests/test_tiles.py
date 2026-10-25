"""Basemap tiles through MineMag's own cache: estimate, bulk download,
status, serving, and the cache living in MineMag's data dir (not
DroneMag Studio's). requests.get is mocked."""
from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import paths
from app.main import app
from magcore import tile_cache


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tile_cache, "CACHE_ROOT", tmp_path / "tile_cache")


_BBOX = {"source": "esri", "min_lat": 36.378, "min_lon": 127.360, "max_lat": 36.380, "max_lon": 127.362, "min_zoom": 16, "max_zoom": 17}


def _ok(content=b"\x89PNG tile"):
    r = MagicMock()
    r.status_code = 200
    r.content = content
    return r


def test_cache_root_is_minemags_own_data_dir():
    """configure() in app/__init__.py runs before magcore.tile_cache fixes
    its root, so the cache is MineMag's, not DroneMag Studio's."""
    import magcore.paths as mp
    assert mp.data_dir() == paths.data_dir()
    assert paths.data_dir() == pathlib.Path(__file__).resolve().parents[1] / ".data"


def test_estimate_download_status_and_serve():
    client = TestClient(app)
    est = client.post("/api/tiles/estimate", json=_BBOX).json()
    assert 2 <= est["n_tiles"] <= 20 and est["exceeds_max"] is False
    with patch.object(tile_cache.requests, "get", return_value=_ok()) as get:
        d = client.post("/api/tiles/download", json=_BBOX).json()
        assert d["n_downloaded"] == est["n_tiles"] and d["n_failed"] == 0 and get.call_count == est["n_tiles"]
        again = client.post("/api/tiles/download", json=_BBOX).json()
        assert again["n_already_cached"] == est["n_tiles"] and again["n_downloaded"] == 0
    st = client.get("/api/tiles/status").json()
    assert set(st) == {"osm", "esri"} and st["esri"]["n_tiles"] == est["n_tiles"] and st["esri"]["zoom_levels"] == [16, 17]
    z, (x, y) = 16, tile_cache.tiles_for_bbox(36.378, 127.360, 36.380, 127.362, 16)[0]
    with patch.object(tile_cache.requests, "get", side_effect=AssertionError("must come from the cache")):
        r = client.get(f"/api/tiles/esri/{z}/{x}/{y}")
    assert r.status_code == 200 and r.content == b"\x89PNG tile" and r.headers["content-type"] == "image/jpeg"
    with patch.object(tile_cache.requests, "get", return_value=_ok(b"live")):
        r = client.get("/api/tiles/osm/16/1/1")              # not cached: fetched live and stored
    assert r.status_code == 200 and r.content == b"live"
    with patch.object(tile_cache.requests, "get", side_effect=tile_cache.requests.RequestException("offline")):
        assert client.get("/api/tiles/osm/16/1/1").status_code == 200       # now from disk
        assert client.get("/api/tiles/osm/16/2/2").status_code == 404       # offline and not cached
    assert client.get("/api/tiles/google/16/1/1").status_code == 404
    assert client.post("/api/tiles/estimate", json={**_BBOX, "source": "google"}).status_code == 422
    assert client.post("/api/tiles/download", json={**_BBOX, "min_zoom": 0, "max_zoom": 19, "min_lat": 30, "max_lat": 40, "min_lon": 120, "max_lon": 130}).status_code == 400
