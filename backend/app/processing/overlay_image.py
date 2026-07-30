"""Georeferenced raster (GeoTIFF) overlay loader for user-supplied reference
layers such as geology maps, rendered the same way as our own grid overlays
(PNG data URL + lat/lon bounding box for Leaflet's ImageOverlay).

The source raster keeps its own pixel grid; only the four corners are
reprojected to WGS84 to get an axis-aligned bounding box. That matches how
processing/render.py handles our own grids and is a good approximation as
long as the source CRS isn't heavily rotated relative to north.
"""
from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

MAX_DIMENSION_PX = 2000


class OverlayImageError(ValueError):
    pass


def load_geotiff_overlay(buffer, name: str = "overlay") -> dict:
    try:
        with rasterio.open(buffer) as src:
            return _render(src, name)
    except rasterio.errors.RasterioIOError as exc:
        raise OverlayImageError(f"GeoTIFF 파일을 열 수 없습니다: {exc}") from exc


def _render(src, name: str) -> dict:
    if src.crs is None:
        raise OverlayImageError("GeoTIFF에 좌표계(CRS) 정보가 없습니다 (좌표화되지 않은 이미지).")

    height, width = src.height, src.width
    scale = min(1.0, MAX_DIMENSION_PX / max(height, width))
    out_height = max(1, round(height * scale))
    out_width = max(1, round(width * scale))

    rgba = _read_rgba(src, out_height, out_width)

    b = src.bounds
    west, south, east, north = transform_bounds(src.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)

    image = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    image.save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "name": name,
        "image_data_url": f"data:image/png;base64,{png_b64}",
        "bounds": [[float(south), float(west)], [float(north), float(east)]],
        "width": out_width,
        "height": out_height,
    }


def _read_rgba(src, out_height: int, out_width: int) -> np.ndarray:
    count = src.count

    if count == 1:
        band = src.read(1, out_shape=(out_height, out_width), resampling=Resampling.nearest)
        colormap = _try_read_colormap(src)
        nodata = src.nodata

        if colormap is not None:
            lut = colormap
            band_clipped = np.clip(band.astype(np.int64), 0, lut.shape[0] - 1)
            rgba = lut[band_clipped]
            if nodata is not None:
                rgba = rgba.copy()
                rgba[..., 3] = np.where(band == nodata, 0, rgba[..., 3])
            return rgba.astype(np.uint8)

        valid_mask = np.ones(band.shape, dtype=bool)
        if nodata is not None:
            valid_mask &= band != nodata
        if np.issubdtype(band.dtype, np.floating):
            valid_mask &= np.isfinite(band)
        valid = band[valid_mask]
        if valid.size == 0:
            raise OverlayImageError("유효한 픽셀 값이 없습니다.")
        vmin, vmax = np.percentile(valid, [2, 98])
        if vmin == vmax:
            vmin, vmax = vmin - 1, vmax + 1
        normalized = np.clip((band.astype(float) - vmin) / (vmax - vmin), 0, 1)
        gray = (normalized * 255).astype(np.uint8)
        alpha = np.where(valid_mask, 255, 0).astype(np.uint8)
        return np.stack([gray, gray, gray, alpha], axis=-1)

    n_read = min(count, 4)
    bands = src.read(list(range(1, n_read + 1)), out_shape=(n_read, out_height, out_width), resampling=Resampling.bilinear)
    bands_u8 = _to_uint8(bands)

    nodata = src.nodata
    if bands_u8.shape[0] >= 4:
        rgb = bands_u8[:3]
        alpha = bands_u8[3]
    else:
        rgb = bands_u8[:3] if bands_u8.shape[0] >= 3 else np.repeat(bands_u8[:1], 3, axis=0)
        alpha = np.full((out_height, out_width), 255, dtype=np.uint8)
        if nodata is not None:
            mask = np.all(bands == nodata, axis=0)
            alpha = np.where(mask, 0, 255).astype(np.uint8)

    rgba = np.moveaxis(rgb, 0, -1)
    return np.concatenate([rgba, alpha[..., None]], axis=-1).astype(np.uint8)


def _try_read_colormap(src) -> np.ndarray | None:
    try:
        cmap = src.colormap(1)
    except ValueError:
        return None
    if not cmap:
        return None
    max_idx = max(cmap.keys())
    lut = np.zeros((max_idx + 1, 4), dtype=np.uint8)
    for idx, rgba in cmap.items():
        rgba = tuple(rgba)
        lut[idx] = rgba[:4] if len(rgba) >= 4 else (*rgba, 255)
    return lut


def _to_uint8(bands: np.ndarray) -> np.ndarray:
    if bands.dtype == np.uint8:
        return bands
    out = np.empty(bands.shape, dtype=np.uint8)
    for i in range(bands.shape[0]):
        b = bands[i].astype(float)
        vmin, vmax = np.percentile(b, [1, 99])
        if vmin == vmax:
            vmin, vmax = vmin - 1, vmax + 1
        out[i] = np.clip((b - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    return out
