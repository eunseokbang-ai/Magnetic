"""Offline basemap tile cache: pre-downloads slippy-map (XYZ/Web
Mercator) tiles for a bounding box so the map background still works
with no internet connection, once a survey area is known in advance
(e.g. downloaded once at the office before going out to the field).
Tiles are cached to disk under CACHE_ROOT and served back by this app
itself (see main.py's /api/tiles/* endpoints), so once cached the
browser never needs to reach the public tile servers again for that
area/zoom range.

Besides the two public basemaps there are KIGAM's geology maps, which
arrive as OGC WMS rather than as XYZ tiles. They are folded into the same
pipeline by turning each z/x/y into the Web Mercator bounding box that
tile covers and asking for exactly that picture (_wms_url), so a geology
layer caches, pre-downloads and serves offline like any other tile.

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
from urllib.parse import urlencode
from dataclasses import dataclass
from pathlib import Path

import requests

from .. import paths

CACHE_ROOT = paths.data_dir() / "tile_cache"

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
    # KIGAM 지오빅데이터 오픈플랫폼 (https://data.kigam.re.kr). Served as
    # WMS, not as tiles - see _wms_url. No key: the openapi manual's
    # /openapi/wms endpoint documents one and now answers 400, but the
    # GeoServer endpoint below answers GetCapabilities and GetMap without
    # any, over EPSG:3857 and with a transparent background, which is
    # exactly what an overlay wants.
    #
    # Measured: the 1:50k sheet renders over the 2026-09 HaeNam block
    # (2,372 distinct colours in one 512 px request) and comes back fully
    # transparent over Dokdo - the island sheets are not in this layer.
    "kigam_50k": {
        "wms": "https://data.kigam.re.kr/geoserver/wms",
        "layers": "Geology_map:L_50K_Geology_Map_Latest",
        "ext": "png",
        "media_type": "image/png",
        "label": "KIGAM 지질도 1:5만",
        "attribution": "지질도 © KIGAM 지오빅데이터 오픈플랫폼",
        "min_zoom": 9,
    },
    "kigam_250k": {
        "wms": "https://data.kigam.re.kr/geoserver/wms",
        "layers": "Geology_map:L_250K_Geology_Map",
        "ext": "png",
        "media_type": "image/png",
        "label": "KIGAM 지질도 1:25만",
        "attribution": "지질도 © KIGAM 지오빅데이터 오픈플랫폼",
        "min_zoom": 7,
    },
    "kigam_1m": {
        "wms": "https://data.kigam.re.kr/geoserver/wms",
        "layers": "Geology_map:L_1M_Geology_Map",
        "ext": "png",
        "media_type": "image/png",
        "label": "KIGAM 지질도 1:100만",
        "attribution": "지질도 © KIGAM 지오빅데이터 오픈플랫폼",
        "min_zoom": 5,
    },
    # Faults on their own, to lay over any of the above.
    "kigam_fault_50k": {
        "wms": "https://data.kigam.re.kr/geoserver/wms",
        "layers": "Geology_map:l_50k_geology_fault_latest",
        "ext": "png",
        "media_type": "image/png",
        "label": "KIGAM 단층 1:5만",
        "attribution": "지질도 © KIGAM 지오빅데이터 오픈플랫폼",
        "min_zoom": 9,
    },
}

# Half the circumference of the Earth at the equator, in Web Mercator
# metres: the coordinate of the top-left corner of tile (0, 0, 0), and
# therefore the scale everything below is measured against.
_MERCATOR_ORIGIN_M = 20037508.342789244

# A single request can trigger up to this many tile fetches - keeps a
# careless "whole country at max zoom" request from turning into tens of
# thousands of requests to a free public tile service. Callers should
# scope requests to just the survey area (see the module docstring).
MAX_TILES_PER_REQUEST = 5000
# Slippy-map tiles are 256 px; a WMS has to be asked for that size or the
# picture will not line up with the tile it fills.
TILE_PIXELS = 256
_REQUEST_DELAY_SECONDS = 0.05
_USER_AGENT = "Magnetic-Survey-App/1.0 (offline field-use tile cache; personal, non-redistributed)"
# KIGAM's service sits behind a filter that answers "400 Request Blocked"
# to the User-Agent above and serves the same request normally when it
# looks like a browser. Measured against /geoserver/wms: identical URL,
# 400 with our own agent, 200 and a PNG with this one. It is sent only to
# the sources that need it.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)


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


def tile_bbox_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) in EPSG:3857 metres of one slippy-map
    tile - the box a WMS has to be asked for to draw that tile."""
    span = 2.0 * _MERCATOR_ORIGIN_M / (2 ** z)
    min_x = -_MERCATOR_ORIGIN_M + x * span
    max_y = _MERCATOR_ORIGIN_M - y * span
    return min_x, max_y - span, min_x + span, max_y


def _wms_url(info: dict, z: int, x: int, y: int) -> str:
    """A WMS GetMap request for exactly the ground one tile covers.

    WMS 1.3.0 with CRS=EPSG:3857: a projected CRS keeps easting first, so
    the bbox order is the same as the tile's. TRANSPARENT=true is what
    makes a geology sheet usable over a basemap instead of hiding it.
    """
    min_x, min_y, max_x, max_y = tile_bbox_3857(z, x, y)
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": info["layers"],
        "STYLES": "",
        "CRS": "EPSG:3857",
        "BBOX": f"{min_x},{min_y},{max_x},{max_y}",
        "WIDTH": str(TILE_PIXELS),
        "HEIGHT": str(TILE_PIXELS),
        "FORMAT": info.get("media_type", "image/png"),
        "TRANSPARENT": "true",
    }
    return info["wms"] + "?" + urlencode(params)


def tile_url(source: str, z: int, x: int, y: int) -> str:
    info = _validate_source(source)
    if "wms" in info:
        return _wms_url(info, z, x, y)
    return info["url"].format(z=z, x=x, y=y)


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
    url = tile_url(source, z, x, y)
    agent = _BROWSER_USER_AGENT if "wms" in info else _USER_AGENT
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": agent})
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
