"""API-level test of the offline tile cache endpoints: estimate, bulk
download, status, and the tile-serving/opportunistic-cache endpoint.
requests.get is mocked (this sandbox's own network policy blocks the
real tile servers - see tile_cache.py's module docstring); the FastAPI
routing/wiring itself is what's verified here."""
import pathlib
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.processing import tile_cache


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tile_cache, "CACHE_ROOT", tmp_path / "tile_cache")
    yield


_BBOX = {
    "source": "osm",
    "min_lat": 46.40,
    "min_lon": 106.20,
    "max_lat": 46.41,
    "max_lon": 106.21,
    "min_zoom": 14,
    "max_zoom": 14,
}


def test_estimate_returns_tile_count_and_size(client):
    r = client.post("/api/tiles/estimate", json=_BBOX)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["n_tiles"] > 0
    assert data["estimated_mb"] >= 0
    assert data["exceeds_max"] is False

    # a larger area/zoom range should report proportionally more tiles and MB
    r2 = client.post("/api/tiles/estimate", json={**_BBOX, "max_lat": 46.60, "max_lon": 106.40, "max_zoom": 16})
    data2 = r2.json()
    assert data2["n_tiles"] > data["n_tiles"]
    assert data2["estimated_mb"] > 0


def test_estimate_rejects_unknown_source(client):
    r = client.post("/api/tiles/estimate", json={**_BBOX, "source": "google"})
    assert r.status_code == 422, r.text  # pydantic Literal validation


def test_download_then_status_then_serve_tile(client):
    fake_resp = MagicMock(status_code=200, content=b"fake-tile-bytes")
    with patch("app.processing.tile_cache.requests.get", return_value=fake_resp):
        r = client.post("/api/tiles/download", json=_BBOX)
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["n_downloaded"] == result["n_total"]
    assert result["n_failed"] == 0

    r = client.get("/api/tiles/status")
    assert r.status_code == 200, r.text
    status = r.json()
    assert status["osm"]["n_tiles"] == result["n_total"]

    tiles = tile_cache.tiles_for_bbox(_BBOX["min_lat"], _BBOX["min_lon"], _BBOX["max_lat"], _BBOX["max_lon"], 14)
    x, y = tiles[0]
    r = client.get(f"/api/tiles/osm/14/{x}/{y}")
    assert r.status_code == 200, r.text
    assert r.content == b"fake-tile-bytes"
    assert r.headers["content-type"] == "image/png"


def test_get_tile_falls_back_to_live_fetch_when_not_cached(client):
    fake_resp = MagicMock(status_code=200, content=b"live-fetched")
    with patch("app.processing.tile_cache.requests.get", return_value=fake_resp):
        r = client.get("/api/tiles/osm/9/300/150")
    assert r.status_code == 200, r.text
    assert r.content == b"live-fetched"
    # now cached, so a second request should not need the network at all
    with patch("app.processing.tile_cache.requests.get", side_effect=AssertionError("should not be called")):
        r2 = client.get("/api/tiles/osm/9/300/150")
    assert r2.status_code == 200, r2.text
    assert r2.content == b"live-fetched"


def test_get_tile_404s_when_uncached_and_unreachable(client):
    import requests as requests_module

    with patch("app.processing.tile_cache.requests.get", side_effect=requests_module.ConnectionError("offline")):
        r = client.get("/api/tiles/osm/9/300/150")
    assert r.status_code == 404


def test_download_rejects_bbox_exceeding_tile_cap(client):
    with patch.object(tile_cache, "MAX_TILES_PER_REQUEST", 4):
        r = client.post(
            "/api/tiles/download",
            json={**_BBOX, "min_lat": 40.0, "min_lon": 100.0, "max_lat": 50.0, "max_lon": 115.0, "min_zoom": 8, "max_zoom": 14},
        )
    assert r.status_code == 400, r.text
    assert "최대치" in r.json()["detail"]


if __name__ == "__main__":
    c = TestClient(app)
    print("Run via pytest for isolated tmp_path fixture support.")
