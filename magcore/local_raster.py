"""Serves a large GeoTIFF on local disk as Web Mercator map tiles, read
on demand - the orthophoto never gets uploaded or re-tiled.

Why this exists next to local_tiles.py: a photogrammetry orthomosaic is
exported as one GeoTIFF of tens of GB (Andong 2026-10-28: 187,062 x
123,655 px, 4-band, 114 GB), and tiling it with gdal2tiles takes hours and
a second copy on disk. But such a file already carries what a tile
server needs - 256 px internal blocks and a pyramid of internal overviews
(Metashape writes them; `gdaladdo` adds them to any other) - so a tile is
one windowed read from the overview nearest the requested zoom, warped to
EPSG:3857 by rasterio's WarpedVRT. Measured on the Andong file: a cold
read is 50-400 ms per tile across zoom 13-21, a cached one a few ms.
Tiles are also cached as PNG under the data dir, so a second look at the
same place is a file read.

A file without overviews still works but every low-zoom tile then reads
the full-resolution raster and is slow; register_local_raster() reports
`has_overviews` so the UI can say "run gdaladdo first".

The Metashape project's own `*.files/.../orthomosaic` folder is NOT a
tile pyramid any map can read (orthomosaic.zip index + tile-row-col.tif +
zipped level pyramids in Metashape's internal layout); export the
orthomosaic to GeoTIFF (or to "Google Map Tiles", which local_tiles.py
reads) and point this at the result.
"""
from __future__ import annotations

import hashlib
import io
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from . import paths

TILE_PX = 256
_ORIGIN = 20037508.342789244
_MAX_SIZE_PX = 2_000_000          # per side; nothing sensible is bigger
_MAX_ZOOM = 24


class LocalRasterError(ValueError):
    pass


@dataclass
class LocalRasterLayer:
    id: str
    path: Path
    label: str
    bounds: tuple[float, float, float, float]      # (south, west, north, east), WGS84
    min_zoom: int
    max_zoom: int
    width: int
    height: int
    bands: int
    crs: str
    res_m: float
    has_overviews: bool
    size_gb: float


_registry: dict[str, LocalRasterLayer] = {}
_open: dict[str, tuple] = {}                     # id -> (dataset, vrt, lock)
_next_id = 0


def _cache_dir() -> Path:
    return paths.data_dir() / "raster_tiles"


def tile_bounds_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    span = 2.0 * _ORIGIN / 2 ** z
    return (-_ORIGIN + x * span, _ORIGIN - (y + 1) * span, -_ORIGIN + (x + 1) * span, _ORIGIN - y * span)


def _native_zoom(res_m: float, lat_deg: float) -> int:
    """The zoom whose tile pixel is about as fine as the raster's pixel."""
    ground = 156543.03392 * math.cos(math.radians(lat_deg))     # m/px at zoom 0
    return int(min(_MAX_ZOOM, max(0, math.ceil(math.log2(ground / max(res_m, 1e-6))))))


def register_local_raster(path_str: str, label: str | None = None) -> LocalRasterLayer:
    global _next_id
    path = Path(path_str).expanduser()
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise LocalRasterError(f"파일을 찾을 수 없습니다: {path_str!r} ({exc})") from exc
    if not path.is_file():
        raise LocalRasterError(f"파일이 아닙니다: {path_str!r}")
    try:
        src = rasterio.open(path)
    except Exception as exc:                       # noqa: BLE001 - rasterio raises several types
        raise LocalRasterError(f"GeoTIFF로 열 수 없습니다: {path.name} ({exc})") from exc
    if src.crs is None:
        src.close()
        raise LocalRasterError(f"{path.name}에 좌표계가 없습니다. 좌표계가 내장된 GeoTIFF를 지정하세요.")
    if src.width > _MAX_SIZE_PX or src.height > _MAX_SIZE_PX:
        src.close()
        raise LocalRasterError(f"래스터가 너무 큽니다: {src.width} x {src.height} px")
    west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    # ground resolution in metres (geographic rasters give degrees)
    if src.crs.is_geographic:
        res_m = float(src.res[0]) * 111320.0 * math.cos(math.radians((south + north) / 2))
    else:
        res_m = float(src.res[0])
    max_zoom = _native_zoom(res_m, (south + north) / 2)
    has_ovr = bool(src.overviews(1))
    _next_id += 1
    layer_id = f"raster{_next_id}"
    layer = LocalRasterLayer(
        id=layer_id, path=path, label=label or path.stem, bounds=(south, west, north, east),
        min_zoom=max(0, max_zoom - 12), max_zoom=max_zoom, width=src.width, height=src.height, bands=src.count,
        crs=str(src.crs), res_m=round(res_m, 4), has_overviews=has_ovr, size_gb=round(path.stat().st_size / 1e9, 2),
    )
    vrt = WarpedVRT(src, crs="EPSG:3857", resampling=Resampling.bilinear)
    _registry[layer_id] = layer
    _open[layer_id] = (src, vrt, threading.Lock())
    return layer


def unregister_local_raster(layer_id: str) -> None:
    _registry.pop(layer_id, None)
    opened = _open.pop(layer_id, None)
    if opened:
        src, vrt, _lock = opened
        try:
            vrt.close()
            src.close()
        except Exception:                            # noqa: BLE001
            pass


def get_layer(layer_id: str) -> LocalRasterLayer | None:
    return _registry.get(layer_id)


def _render(layer_id: str, z: int, x: int, y: int) -> bytes | None:
    layer = _registry.get(layer_id)
    if layer is None or not (0 <= z <= _MAX_ZOOM):
        return None
    src, vrt, lock = _open[layer_id]
    bounds = tile_bounds_3857(z, x, y)
    # skip tiles entirely outside the raster (cheap, before any read)
    tw, ts, te, tn = transform_bounds("EPSG:3857", "EPSG:4326", *bounds)
    s, w, n, e = layer.bounds
    if te < w or tw > e or tn < s or ts > n:
        return None
    with lock:
        # WarpedVRT refuses boundless reads: clip the tile window to the
        # raster and paste the piece at its place in the 256 px tile
        win = from_bounds(*bounds, transform=vrt.transform)
        full = Window(0, 0, vrt.width, vrt.height)
        try:
            inter = win.intersection(full)
        except Exception:                            # noqa: BLE001 - rasterio raises WindowError on no overlap
            return None
        if inter.width <= 0 or inter.height <= 0:
            return None
        sx, sy = TILE_PX / win.width, TILE_PX / win.height
        col0 = int(round((inter.col_off - win.col_off) * sx))
        row0 = int(round((inter.row_off - win.row_off) * sy))
        out_w = max(1, min(TILE_PX - col0, int(round(inter.width * sx))))
        out_h = max(1, min(TILE_PX - row0, int(round(inter.height * sy))))
        piece = vrt.read(window=inter, out_shape=(src.count, out_h, out_w), resampling=Resampling.bilinear)
        arr = np.zeros((src.count, TILE_PX, TILE_PX), dtype=piece.dtype)
        arr[:, row0:row0 + out_h, col0:col0 + out_w] = piece
        if src.count >= 4:
            alpha = arr[3]
        else:
            mpiece = vrt.read_masks(window=inter, out_shape=(src.count, out_h, out_w))
            alpha = np.zeros((TILE_PX, TILE_PX), dtype=np.uint8)
            alpha[row0:row0 + out_h, col0:col0 + out_w] = mpiece.min(axis=0)
    if src.count == 1:
        rgb = np.repeat(arr[:1], 3, axis=0)
    elif src.count == 2:
        rgb = np.concatenate([arr[:1], arr[:1], arr[:1]], axis=0)
    else:
        rgb = arr[:3]
    if rgb.dtype != np.uint8:
        lo, hi = np.nanpercentile(rgb[:, alpha > 0], [1, 99]) if (alpha > 0).any() else (0, 1)
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    if not alpha.any():
        return None
    rgba = np.concatenate([rgb, alpha[None].astype(np.uint8)], axis=0)
    img = Image.fromarray(np.moveaxis(rgba, 0, -1), "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=3)
    return buf.getvalue()


def get_tile(layer_id: str, z: int, x: int, y: int) -> bytes | None:
    """PNG bytes of one XYZ tile, from the on-disk cache or rendered now;
    None when the tile is empty or the layer unknown."""
    layer = _registry.get(layer_id)
    if layer is None:
        return None
    # a stable key (hashlib, not hash(): str hashes change per process) so the
    # cache survives restarts and is dropped when the file is replaced
    key = hashlib.md5(f"{layer.path}|{layer.path.stat().st_mtime_ns}|{layer.path.stat().st_size}".encode("utf-8")).hexdigest()[:16]
    cached = _cache_dir() / key / str(z) / str(x) / f"{y}.png"
    if cached.is_file():
        return cached.read_bytes()
    png = _render(layer_id, z, x, y)
    if png is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(png)
    return png


def pick_file_dialog(title: str = "GeoTIFF 선택") -> str:
    """Native file picker on the machine running the backend (see
    local_tiles.pick_folder_dialog for why that is the right machine)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise LocalRasterError("이 서버 환경에서는 파일 선택 창을 지원하지 않습니다 (tkinter 미설치) - 경로를 직접 입력하세요.") from exc
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            path = filedialog.askopenfilename(title=title, filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")])
        finally:
            root.destroy()
    except Exception as exc:                         # noqa: BLE001 - tk errors vary by platform
        raise LocalRasterError(f"파일 선택 창을 열지 못했습니다: {exc}") from exc
    return path or ""
