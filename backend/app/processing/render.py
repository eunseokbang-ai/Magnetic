"""Render a 2D grid (with NaN outside coverage) into a colored PNG plus the
lat/lon bounding box Leaflet needs for an ImageOverlay."""
from __future__ import annotations

import base64
from io import BytesIO

import matplotlib
import numpy as np
import rasterio
from matplotlib.colors import LightSource
from PIL import Image
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_origin


def grid_to_png_overlay(
    grid_values: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    utm_epsg: int,
    cmap_name: str = "viridis",
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    hillshade: bool = False,
    hillshade_azimuth_deg: float = 315.0,
    hillshade_altitude_deg: float = 45.0,
    hillshade_exaggeration: float = 3.0,
    cell_size_m: float = 1.0,
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

    if hillshade:
        # Geosoft Oasis Montaj-style "color-shaded relief": treat the
        # grid's own values as a pseudo-terrain and illuminate it, so
        # subtle gradients/edges in the anomaly pattern show up as
        # raised/shadowed texture instead of disappearing into flat
        # color bands - a standard display mode for airborne geophysics.
        # LightSource can't handle NaN when computing slopes, so gaps are
        # filled with the mean before shading and then re-masked to
        # transparent afterward exactly as in the non-hillshade path.
        filled = np.where(np.isfinite(grid_values), grid_values, float(np.mean(finite)))
        light = LightSource(azdeg=hillshade_azimuth_deg, altdeg=hillshade_altitude_deg)
        shaded = light.shade(
            filled,
            cmap=cmap,
            norm=norm,
            blend_mode="overlay",
            vert_exag=hillshade_exaggeration,
            dx=cell_size_m,
            dy=cell_size_m,
        )
        rgba = (np.clip(shaded, 0, 1) * 255).astype(np.uint8)
    else:
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


def grid_to_geotiff_bytes(grid_values: np.ndarray, easting: np.ndarray, northing: np.ndarray, utm_epsg: int) -> bytes:
    """Single-band float32 GeoTIFF with the raw (uncolored) grid values -
    unlike the PNG overlay above (which bakes in a colormap for display
    in this app), this preserves the actual numbers so the grid can be
    reopened and re-styled in Oasis Montaj, QGIS, ArcGIS, etc."""
    cell_size = float(easting[1] - easting[0])
    # array row 0 = southmost northing; GeoTIFF row 0 must be the top (north).
    data = np.flipud(grid_values).astype("float32")
    transform = from_origin(easting[0] - cell_size / 2, northing[-1] + cell_size / 2, cell_size, cell_size)
    crs = CRS.from_epsg(utm_epsg)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=np.nan,
            compress="deflate",
        ) as dst:
            dst.write(data, 1)
        return bytes(memfile.read())
