"""Serving an OGC WMS (KIGAM's geology sheets) as ordinary map tiles.

The point of doing it this way rather than pointing the browser straight
at the service: the tile pipeline this app already has caches to disk,
pre-downloads a survey area for field use, and serves everything from
localhost. A geology layer that goes through it inherits all three - and
the service refuses requests that do not look like they came from a
browser, which is easier to satisfy from the backend than from a Leaflet
layer.

The arithmetic here is the part that can silently go wrong: a bounding
box that is off by a tile, or given in the wrong axis order, still
returns a perfectly valid picture of the wrong place.
"""
from __future__ import annotations

import math
import pathlib
import sys
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from app.processing import tile_cache

MERCATOR_ORIGIN = 20037508.342789244


def _query(url: str) -> dict:
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def test_the_whole_world_is_one_tile_at_zoom_zero():
    assert tile_cache.tile_bbox_3857(0, 0, 0) == pytest.approx(
        (-MERCATOR_ORIGIN, -MERCATOR_ORIGIN, MERCATOR_ORIGIN, MERCATOR_ORIGIN)
    )


def test_tiles_tile_the_plane_without_gaps_or_overlaps():
    """Neighbouring tiles must share an edge exactly: a rounding error
    here shows up as a seam of missing or doubled geology."""
    left = tile_cache.tile_bbox_3857(5, 26, 12)
    right = tile_cache.tile_bbox_3857(5, 27, 12)
    below = tile_cache.tile_bbox_3857(5, 26, 13)

    assert left[2] == pytest.approx(right[0])          # left's east edge = right's west
    assert left[1] == pytest.approx(below[3])          # left's south edge = below's north
    assert left[3] - left[1] == pytest.approx(left[2] - left[0])   # square


def test_the_bbox_lands_on_the_ground_the_tile_covers():
    """Checked against the survey this is for: the HaeNam block sits near
    126.55E, 34.35N, and the tile containing it must ask for a box
    containing it."""
    lat, lon, zoom = 34.35, 126.55, 13
    x, y = tile_cache._deg_to_tile(lat, lon, zoom)

    min_x, min_y, max_x, max_y = tile_cache.tile_bbox_3857(zoom, x, y)

    want_x = MERCATOR_ORIGIN * lon / 180.0
    want_y = MERCATOR_ORIGIN * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) / math.pi
    assert min_x < want_x < max_x
    assert min_y < want_y < max_y


def test_the_request_is_a_getmap_for_exactly_that_box():
    url = tile_cache.tile_url("kigam_50k", 12, 3500, 1620)
    q = _query(url)

    assert url.startswith("https://data.kigam.re.kr/geoserver/wms?")
    assert q["SERVICE"] == "WMS" and q["REQUEST"] == "GetMap" and q["VERSION"] == "1.3.0"
    assert q["LAYERS"] == "Geology_map:L_50K_Geology_Map_Latest"
    # EPSG:3857 is what the map itself uses, so no reprojection is needed;
    # a projected CRS keeps easting first even in WMS 1.3.0.
    assert q["CRS"] == "EPSG:3857"
    assert [float(v) for v in q["BBOX"].split(",")] == pytest.approx(
        list(tile_cache.tile_bbox_3857(12, 3500, 1620))
    )
    assert q["WIDTH"] == q["HEIGHT"] == str(tile_cache.TILE_PIXELS)
    # Without this the sheet hides the basemap underneath it.
    assert q["TRANSPARENT"] == "true"


def test_an_ordinary_xyz_source_is_untouched():
    assert tile_cache.tile_url("osm", 7, 109, 50) == "https://tile.openstreetmap.org/7/109/50.png"


def test_the_geology_service_is_asked_as_a_browser(monkeypatch, tmp_path):
    """Measured against the live service: the same URL answers 400
    ("Request Blocked") to this app's own User-Agent and 200 with a PNG
    to a browser's. The polite agent is still used for the public
    basemaps, whose tile policies ask for an identifying one."""
    monkeypatch.setattr(tile_cache, "CACHE_ROOT", tmp_path)
    seen = {}

    class Response:
        status_code = 200
        content = b"PNG"

    def fake_get(url, timeout, headers):
        seen[url.split("?")[0]] = headers["User-Agent"]
        return Response()

    monkeypatch.setattr(tile_cache.requests, "get", fake_get)

    tile_cache.fetch_and_cache_tile("kigam_50k", 12, 3500, 1620)
    tile_cache.fetch_and_cache_tile("osm", 12, 3500, 1620)

    assert "Mozilla" in seen["https://data.kigam.re.kr/geoserver/wms"]
    assert "Magnetic-Survey-App" in seen["https://tile.openstreetmap.org/12/3500/1620.png"]


def test_geology_tiles_cache_like_any_other(monkeypatch, tmp_path):
    monkeypatch.setattr(tile_cache, "CACHE_ROOT", tmp_path)
    calls = []

    class Response:
        status_code = 200
        content = b"PNG-DATA"

    monkeypatch.setattr(tile_cache.requests, "get",
                        lambda url, timeout, headers: (calls.append(url), Response())[1])

    assert tile_cache.get_cached_tile("kigam_50k", 12, 3500, 1620) is None
    tile_cache.fetch_and_cache_tile("kigam_50k", 12, 3500, 1620)

    assert tile_cache.get_cached_tile("kigam_50k", 12, 3500, 1620) == b"PNG-DATA"
    assert (tmp_path / "kigam_50k" / "12" / "3500" / "1620.png").exists()
    assert len(calls) == 1
    assert "kigam_50k" in tile_cache.cache_status()


def test_an_unknown_source_is_refused():
    with pytest.raises(tile_cache.TileCacheError):
        tile_cache.tile_url("kigam_500k", 10, 1, 1)
