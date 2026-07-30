"""Render a 2D grid (with NaN outside coverage) into a colored PNG plus the
lat/lon bounding box Leaflet needs for an ImageOverlay."""
from __future__ import annotations

import base64
from io import BytesIO

import matplotlib
import numpy as np
from PIL import Image
from pyproj import Transformer


def grid_to_png_overlay(
    grid_values: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    utm_epsg: int,
    cmap_name: str = "viridis",
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict:
    finite = grid_values[np.isfinite(grid_values)]
    if finite.size == 0:
        raise ValueError("표시할 유효한 그리드 값이 없습니다.")

    explicit_range = vmin is not None and vmax is not None
    if vmin is None:
        vmin = float(np.nanpercentile(finite, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(finite, 98))
    if symmetric and not explicit_range:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]
    rgba = (cmap(norm(grid_values)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(grid_values), 255, 0).astype(np.uint8)

    # array row 0 = southmost northing; image row 0 must be the top (north).
    img_array = np.flipud(rgba)
    image = Image.fromarray(img_array, mode="RGBA")
    buf = BytesIO()
    image.save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    corners_e = [easting.min(), easting.max(), easting.min(), easting.max()]
    corners_n = [northing.min(), northing.min(), northing.max(), northing.max()]
    lon_c, lat_c = transformer.transform(corners_e, corners_n)

    bounds = [[float(min(lat_c)), float(min(lon_c))], [float(max(lat_c)), float(max(lon_c))]]

    return {
        "image_data_url": f"data:image/png;base64,{png_b64}",
        "bounds": bounds,
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cmap_name,
    }
