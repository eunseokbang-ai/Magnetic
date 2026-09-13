"""Validates the offline basemap tile cache: standard slippy-map (Web
Mercator XYZ) tile index math, bbox->tile-list coverage, the per-request
tile-count cap, and disk caching. HTTP calls are mocked (requests.get) -
this sandbox's own network policy blocks the real tile servers, so the
math/caching logic is what's verified here; live download needs a real
run against the internet, which the user should do on their machine."""
import pathlib
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from app.processing import tile_cache


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tile_cache, "CACHE_ROOT", tmp_path / "tile_cache")
    yield


def test_deg_to_tile_zoom_0_covers_whole_world():
    # At zoom 0 there is exactly one tile (0, 0) covering the entire
    # world, regardless of where on Earth you are.
    assert tile_cache._deg_to_tile(0.0, 0.0, 0) == (0, 0)
    assert tile_cache._deg_to_tile(46.5, 106.27, 0) == (0, 0)
    assert tile_cache._deg_to_tile(-33.0, -70.0, 0) == (0, 0)


def test_deg_to_tile_origin_is_center_tile_at_any_zoom():
    # (lat=0, lon=0) sits exactly on the antimeridian-free center of the
    # Web Mercator projection, so it always maps to the middle tile
    # (2^(z-1), 2^(z-1)) - the standard formula's defining invariant.
    for z in (1, 5, 10, 15):
        x, y = tile_cache._deg_to_tile(0.0, 0.0, z)
        assert (x, y) == (2 ** (z - 1), 2 ** (z - 1))


def test_deg_to_tile_x_increases_east_y_increases_south():
    z = 12
    x_west, _ = tile_cache._deg_to_tile(46.5, 105.0, z)
    x_east, _ = tile_cache._deg_to_tile(46.5, 108.0, z)
    assert x_east > x_west

    _, y_north = tile_cache._deg_to_tile(50.0, 106.0, z)
    _, y_south = tile_cache._deg_to_tile(40.0, 106.0, z)
    assert y_south > y_north


def test_tiles_for_bbox_covers_expected_grid():
    tiles = tile_cache.tiles_for_bbox(46.40, 106.20, 46.60, 106.40, zoom=12)
    xs = {x for x, y in tiles}
    ys = {y for x, y in tiles}
    # a small bbox at zoom 12 should span a handful of tiles, not one and
    # not thousands
    assert 1 <= len(xs) <= 6
    assert 1 <= len(ys) <= 6
    assert len(tiles) == len(xs) * len(ys)


def test_count_tiles_matches_sum_across_zoom_levels():
    n = tile_cache.count_tiles("osm", 46.40, 106.20, 46.60, 106.40, min_zoom=10, max_zoom=12)
    expected = sum(len(tile_cache.tiles_for_bbox(46.40, 106.20, 46.60, 106.40, z)) for z in (10, 11, 12))
    assert n == expected


def test_rejects_unknown_source():
    with pytest.raises(tile_cache.TileCacheError):
        tile_cache.count_tiles("google", 46.4, 106.2, 46.6, 106.4, 10, 12)


def test_rejects_invalid_bbox():
    with pytest.raises(tile_cache.TileCacheError):
        tile_cache.count_tiles("osm", 46.6, 106.2, 46.4, 106.4, 10, 12)  # min_lat > max_lat


def test_rejects_invalid_zoom_range():
    with pytest.raises(tile_cache.TileCacheError):
        tile_cache.count_tiles("osm", 46.4, 106.2, 46.6, 106.4, min_zoom=14, max_zoom=10)


def test_download_tiles_rejects_over_cap_request():
    with patch.object(tile_cache, "MAX_TILES_PER_REQUEST", 4):
        with pytest.raises(tile_cache.TileCacheError):
            tile_cache.download_tiles("osm", 46.0, 106.0, 47.0, 108.0, min_zoom=10, max_zoom=14)


def test_fetch_and_cache_tile_writes_to_disk_and_returns_content():
    fake_resp = MagicMock(status_code=200, content=b"fake-png-bytes")
    with patch("app.processing.tile_cache.requests.get", return_value=fake_resp) as mock_get:
        content = tile_cache.fetch_and_cache_tile("osm", 12, 100, 200)
    assert content == b"fake-png-bytes"
    assert tile_cache.get_cached_tile("osm", 12, 100, 200) == b"fake-png-bytes"
    # a real-looking, identifying User-Agent should be sent, not a bare requests default
    assert "User-Agent" in mock_get.call_args.kwargs["headers"]


def test_fetch_and_cache_tile_returns_none_on_network_failure():
    import requests as requests_module

    with patch("app.processing.tile_cache.requests.get", side_effect=requests_module.RequestException("boom")):
        content = tile_cache.fetch_and_cache_tile("osm", 12, 100, 200)
    assert content is None
    assert tile_cache.get_cached_tile("osm", 12, 100, 200) is None


def test_download_tiles_skips_already_cached_tiles():
    fake_resp = MagicMock(status_code=200, content=b"x")
    with patch("app.processing.tile_cache.requests.get", return_value=fake_resp):
        result1 = tile_cache.download_tiles("osm", 46.40, 106.20, 46.41, 106.21, min_zoom=14, max_zoom=14)
        assert result1.n_downloaded == result1.n_total
        assert result1.n_already_cached == 0

        result2 = tile_cache.download_tiles("osm", 46.40, 106.20, 46.41, 106.21, min_zoom=14, max_zoom=14)
        assert result2.n_downloaded == 0
        assert result2.n_already_cached == result2.n_total


def test_download_tiles_counts_failures_separately():
    import requests as requests_module

    with patch("app.processing.tile_cache.requests.get", side_effect=requests_module.ConnectionError("network down")):
        result = tile_cache.download_tiles("osm", 46.40, 106.20, 46.41, 106.21, min_zoom=14, max_zoom=14)
    assert result.n_failed == result.n_total
    assert result.n_downloaded == 0


def test_cache_status_reports_tile_count_and_size():
    fake_resp = MagicMock(status_code=200, content=b"0123456789")  # 10 bytes
    with patch("app.processing.tile_cache.requests.get", return_value=fake_resp):
        tile_cache.download_tiles("osm", 46.40, 106.20, 46.41, 106.21, min_zoom=14, max_zoom=14)
    status = tile_cache.cache_status()
    assert status["osm"]["n_tiles"] > 0
    assert status["osm"]["zoom_levels"] == [14]
    assert status["esri"]["n_tiles"] == 0


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call(["pytest", __file__, "-q"]))
