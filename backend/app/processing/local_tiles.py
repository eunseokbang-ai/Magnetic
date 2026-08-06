"""Serves an already-tiled XYZ/TMS raster pyramid straight off local disk
as a map layer, without this app ever reading or re-rendering the source
raster itself.

This exists for very large orthophotos (tens of GB) where the normal path
- upload the whole GeoTIFF, have the backend decode+resample it into one
PNG (see processing/overlay_image.py) - is impractical: the file is too
big to upload over HTTP in one shot, and even a decimated read of a raster
that large (especially one without pre-built internal overviews) can take
a very long time.

The intended workflow: the user tiles the orthophoto once with an
established GIS tool (QGIS's "Raster > 타일 생성 (XYZ)"/"Generate XYZ
tiles" tool, or GDAL's gdal2tiles.py) into a standard {z}/{x}/{y}.<ext>
folder pyramid, then points this app at that folder's path. Since this app
runs locally alongside the user's own files (see run.bat), the backend can
read that folder directly from disk - no upload, no re-encoding, and each
tile request is just a single file read, so this is fast at any raster
size. Re-tiling itself is intentionally left to those existing, optimized
tools rather than reimplemented here.

Two on-disk tile row conventions exist and are auto-detected best-effort:
- "xyz" (row 0 = north, the scheme this app's own /api/tiles/* offline
  cache and virtually all web map libraries use natively)
- "tms" (row 0 = south, the OGC Tile Map Service convention - this is
  gdal2tiles.py's default output unless it's run with --xyz; its presence
  is inferred from a tilemapresource.xml file, which gdal2tiles.py only
  writes for its default TMS output)
Tiles are always served back out in "xyz" row order regardless of the
on-disk convention, so the frontend never needs to know which one a given
folder used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

_TILE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
_EXT_MEDIA_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}


class LocalTileError(ValueError):
    pass


@dataclass
class LocalTileLayer:
    id: str
    path: Path
    label: str
    min_zoom: int
    max_zoom: int
    scheme: str  # "xyz" or "tms"
    ext: str
    bounds: tuple[float, float, float, float]  # (south, west, north, east)


_registry: dict[str, LocalTileLayer] = {}
_next_id = 0


def _tile_to_deg(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Inverse of the standard Web Mercator tile index formula: the
    lat/lon of the NW corner of tile (x, y) at the given zoom (XYZ row
    convention, row 0 = north)."""
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def _int_subdirs(directory: Path) -> list[int]:
    return sorted(int(p.name) for p in directory.iterdir() if p.is_dir() and p.name.isdigit())


def _scan_ext_and_bounds(path: Path, zoom: int, scheme: str) -> tuple[str, tuple[float, float, float, float]] | tuple[None, None]:
    zoom_dir = path / str(zoom)
    if not zoom_dir.is_dir():
        return None, None
    x_dirs = _int_subdirs(zoom_dir)
    if not x_dirs:
        return None, None

    ext: str | None = None
    min_y = max_y = None
    for xd in x_dirs:
        for f in (zoom_dir / str(xd)).iterdir():
            if not f.is_file():
                continue
            stem, dot, suffix = f.name.partition(".")
            suffix = suffix.lower()
            if not dot or not stem.isdigit() or suffix not in _TILE_EXTENSIONS:
                continue
            ext = ext or suffix
            y = int(stem)
            min_y = y if min_y is None else min(min_y, y)
            max_y = y if max_y is None else max(max_y, y)

    if ext is None or min_y is None:
        return None, None

    min_x, max_x = x_dirs[0], x_dirs[-1]
    n = 2**zoom
    if scheme == "tms":
        # TMS row 0 is the southernmost row; flip to XYZ row order for the tile-index math below.
        xyz_min_y, xyz_max_y = n - 1 - max_y, n - 1 - min_y
    else:
        xyz_min_y, xyz_max_y = min_y, max_y

    north, west = _tile_to_deg(min_x, xyz_min_y, zoom)
    south, east = _tile_to_deg(max_x + 1, xyz_max_y + 1, zoom)
    return ext, (south, west, north, east)


def register_local_tile_folder(path_str: str, label: str | None = None, scheme_override: str | None = None) -> LocalTileLayer:
    global _next_id

    path = Path(path_str).expanduser()
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise LocalTileError(f"폴더를 찾을 수 없습니다: {path_str!r} ({exc})") from exc
    if not path.is_dir():
        raise LocalTileError(f"폴더가 아닙니다: {path_str!r}")

    zoom_levels = _int_subdirs(path)
    if not zoom_levels:
        raise LocalTileError(
            "타일 폴더 형식이 아닙니다 (0, 1, 2 같은 확대단계 숫자 이름의 하위 폴더가 없습니다). "
            "QGIS의 '래스터 > 타일 생성(XYZ)' 또는 GDAL의 gdal2tiles.py로 만든 타일 폴더를 지정하세요."
        )
    min_zoom, max_zoom = zoom_levels[0], zoom_levels[-1]

    scheme = scheme_override or ("tms" if (path / "tilemapresource.xml").exists() else "xyz")
    if scheme not in ("xyz", "tms"):
        raise LocalTileError(f"알 수 없는 타일 스킴입니다: {scheme!r} (xyz 또는 tms)")

    ext, bounds = _scan_ext_and_bounds(path, max_zoom, scheme)
    if ext is None:
        # Fall back to the lowest zoom level in case the highest one is sparse/incomplete.
        ext, bounds = _scan_ext_and_bounds(path, min_zoom, scheme)
    if ext is None:
        raise LocalTileError("타일 폴더에서 실제 타일 이미지 파일(.png/.jpg/.jpeg/.webp)을 찾지 못했습니다.")

    _next_id += 1
    layer_id = f"local{_next_id}"
    layer = LocalTileLayer(
        id=layer_id,
        path=path,
        label=label or path.name,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        scheme=scheme,
        ext=ext,
        bounds=bounds,
    )
    _registry[layer_id] = layer
    return layer


def unregister_local_tile_folder(layer_id: str) -> None:
    _registry.pop(layer_id, None)


def get_tile(layer_id: str, z: int, x: int, y: int) -> tuple[bytes | None, str]:
    layer = _registry.get(layer_id)
    if layer is None or not (layer.min_zoom <= z <= layer.max_zoom):
        return None, "application/octet-stream"

    file_y = (2**z - 1 - y) if layer.scheme == "tms" else y
    tile_path = layer.path / str(z) / str(x) / f"{file_y}.{layer.ext}"
    if not tile_path.is_file():
        return None, "application/octet-stream"
    return tile_path.read_bytes(), _EXT_MEDIA_TYPES[layer.ext]
