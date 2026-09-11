"""Offline basemap tile cache: pre-downloads slippy-map (XYZ/Web
Mercator) tiles for a bounding box so the map background still works
with no internet connection, once a survey area is known in advance
(e.g. downloaded once at the office before going out to the field).
Tiles are cached to disk under CACHE_ROOT and served back by this app
itself (see main.py's /api/tiles/* endpoints), so once cached the
browser never needs to reach the public tile servers again for that
area/zoom range.

Only two sources are offered for bulk pre-download: OpenStreetMap's
standard tile server and Esri World Imagery. Both are downloaded
conservatively (a hard cap on tile count per request, a small delay
between requests, and an identifying User-Agent) out of respect for
their tile usage policies, which are aimed at normal interactive
browsing rather than bulk downloading - see
https://operations.osmfoundation.org/policies/tiles/. This is meant for
a small, specific survey area at a handful of zoom levels, not for
downloading whole regions/countries. The unofficial Google tile source
used elsewhere in this app for live browsing is deliberately NOT
offered here, since bulk-downloading it is a clearer ToS problem than
either of the above.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import requests

CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / ".data" / "tile_cache"

TILE_SOURCES = {
    "osm": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "ext": "png",
        "media_type": "image/png",
        "label": "OpenStreetMap",
    },
    "esri": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "ext": "jpg",
        "media_type": "image/jpeg",
        "label": "Esri World Imagery (위성)",
    },
}

# A single request can trigger up to this many tile fetches - keeps a
# careless "whole country at max zoom" request from turning into tens of
# thousands of requests to a free public tile service. Callers should
# scope requests to just the survey area (see the module docstring).
MAX_TILES_PER_REQUEST = 5000
_REQUEST_DELAY_SECONDS = 0.05
_USER_AGENT = "Magnetic-Survey-App/1.0 (offline field-use tile cache; personal, non-redistributed)"


class TileCacheError(ValueError):
    pass


def _deg_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Standard Web Mercator slippy-map tile indices for a lat/lon at a
    given zoom level (the same scheme every OSM/XYZ-style tile server
    uses)."""
    lat_rad = math.radians(max(-85.0511, min(85.0511, lat)))
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def tiles_for_bbox(min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int) -> list[tuple[int, int]]:
    x0, y1 = _deg_to_tile(min_lat, min_lon, zoom)  # SW corner (smaller lat -> larger y)
    x1, y0 = _deg_to_tile(max_lat, max_lon, zoom)  # NE corner (larger lat -> smaller y)
    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)
    return [(x, y) for x in xs for y in ys]


def _validate_source(source: str) -> dict:
    if source not in TILE_SOURCES:
        raise TileCacheError(f"지원하지 않는 지도 소스입니다: {source!r} (사용 가능: {', '.join(TILE_SOURCES)})")
    return TILE_SOURCES[source]


def _validate_bbox(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> None:
    if not (-90 <= min_lat < max_lat <= 90):
        raise TileCacheError("위도 범위가 올바르지 않습니다 (min_lat < max_lat, -90~90).")
    if not (-180 <= min_lon < max_lon <= 180):
        raise TileCacheError("경도 범위가 올바르지 않습니다 (min_lon < max_lon, -180~180).")


def _validate_zoom(min_zoom: int, max_zoom: int) -> None:
    if not (0 <= min_zoom <= max_zoom <= 19):
        raise TileCacheError("확대 단계(zoom) 범위가 올바르지 않습니다 (0~19, min_zoom <= max_zoom).")


def count_tiles(source: str, min_lat: float, min_lon: float, max_lat: float, max_lon: float, min_zoom: int, max_zoom: int) -> int:
    _validate_source(source)
    _validate_bbox(min_lat, min_lon, max_lat, max_lon)
    _validate_zoom(min_zoom, max_zoom)
    return sum(len(tiles_for_bbox(min_lat, min_lon, max_lat, max_lon, z)) for z in range(min_zoom, max_zoom + 1))


def _tile_path(source: str, z: int, x: int, y: int) -> Path:
    ext = TILE_SOURCES[source]["ext"]
    return CACHE_ROOT / source / str(z) / str(x) / f"{y}.{ext}"


def get_cached_tile(source: str, z: int, x: int, y: int) -> bytes | None:
    path = _tile_path(source, z, x, y)
    return path.read_bytes() if path.exists() else None


def fetch_and_cache_tile(source: str, z: int, x: int, y: int, timeout: float = 10.0) -> bytes | None:
    """Fetch one tile live and cache it to disk. Used both by
    download_tiles() for bulk pre-fetching and opportunistically by the
    tile-serving endpoint when a requested tile isn't cached yet but the
    internet happens to be reachable - so simply browsing the map while
    online gradually builds up the offline cache for wherever the user
    looked, even without an explicit bulk download."""
    info = _validate_source(source)
    url = info["url"].format(z=z, x=x, y=y)
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
    except requests.RequestException:
        return None
    if resp.status_code != 200 or not resp.content:
        return None
    path = _tile_path(source, z, x, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return resp.content


@dataclass
class DownloadResult:
    n_total: int
    n_already_cached: int
    n_downloaded: int
    n_failed: int


def download_tiles(
    source: str,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    min_zoom: int,
    max_zoom: int,
) -> DownloadResult:
    _validate_source(source)
    _validate_bbox(min_lat, min_lon, max_lat, max_lon)
    _validate_zoom(min_zoom, max_zoom)

    total = count_tiles(source, min_lat, min_lon, max_lat, max_lon, min_zoom, max_zoom)
    if total > MAX_TILES_PER_REQUEST:
        raise TileCacheError(
            f"요청한 영역/확대범위의 타일 수({total}개)가 한 번에 받을 수 있는 최대치"
            f"({MAX_TILES_PER_REQUEST}개)를 넘습니다. 영역을 좁히거나 확대 단계 범위를 줄이세요."
        )

    n_already = n_downloaded = n_failed = 0
    for z in range(min_zoom, max_zoom + 1):
        for x, y in tiles_for_bbox(min_lat, min_lon, max_lat, max_lon, z):
            if _tile_path(source, z, x, y).exists():
                n_already += 1
                continue
            content = fetch_and_cache_tile(source, z, x, y)
            if content is None:
                n_failed += 1
            else:
                n_downloaded += 1
            time.sleep(_REQUEST_DELAY_SECONDS)

    return DownloadResult(n_total=total, n_already_cached=n_already, n_downloaded=n_downloaded, n_failed=n_failed)


def cache_status() -> dict:
    status = {}
    for source, info in TILE_SOURCES.items():
        source_dir = CACHE_ROOT / source
        if not source_dir.is_dir():
            status[source] = {"label": info["label"], "n_tiles": 0, "size_mb": 0.0, "zoom_levels": []}
            continue
        files = list(source_dir.rglob(f"*.{info['ext']}"))
        total_bytes = sum(f.stat().st_size for f in files)
        zoom_levels = sorted({int(f.relative_to(source_dir).parts[0]) for f in files})
        status[source] = {
            "label": info["label"],
            "n_tiles": len(files),
            "size_mb": round(total_bytes / (1024 * 1024), 2),
            "zoom_levels": zoom_levels,
        }
    return status
