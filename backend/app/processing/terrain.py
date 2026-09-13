"""Ground elevation for the 3D inversion mesh: either resampled from an
uploaded DEM GeoTIFF, or estimated from the drone's own GPS track under a
constant-AGL (terrain-following flight) assumption when no DEM is supplied.
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject


class TerrainError(ValueError):
    pass


def estimate_ground_elevation(altitude_ellipsoidal_m: np.ndarray, assumed_agl_m: float) -> np.ndarray:
    """Ground elevation under a constant above-ground-level flight
    assumption: ground = drone ellipsoidal altitude - AGL.

    Only the relative shape of the terrain is recovered correctly this way;
    the absolute elevation depends entirely on the assumed AGL constant."""
    return np.asarray(altitude_ellipsoidal_m, dtype=float) - float(assumed_agl_m)


def load_dem_geotiff(buffer, x_centers: np.ndarray, y_centers: np.ndarray, target_epsg: int) -> np.ndarray:
    """Resample an uploaded DEM onto the inversion mesh's (x_centers,
    y_centers) cell-center grid (local UTM meters, EPSG:target_epsg).

    Returns a 2D array of shape (len(y_centers), len(x_centers)) of
    elevation values, matching the (northing, easting) row/column
    convention used elsewhere in this app (processing/gridding.py,
    processing/render.py).

    Returns the DEM's own raw values/datum as-is (typically orthometric
    height) - callers comparing this against the drone's ellipsoidal GPS
    altitude must reconcile the two themselves; see
    InversionParams.dem_geoid_offset_m in models.py.
    """
    x_centers = np.asarray(x_centers, dtype=float)
    y_centers = np.asarray(y_centers, dtype=float)
    if len(x_centers) < 2 or len(y_centers) < 2:
        raise TerrainError("DEM 리샘플링을 위한 격자 크기가 너무 작습니다.")

    try:
        with rasterio.open(buffer) as src:
            if src.crs is None:
                raise TerrainError("DEM GeoTIFF에 좌표계(CRS) 정보가 없습니다.")

            dx = x_centers[1] - x_centers[0]
            dy = y_centers[1] - y_centers[0]
            dst_transform = Affine(dx, 0.0, x_centers[0] - dx / 2.0, 0.0, dy, y_centers[0] - dy / 2.0)
            dst_shape = (len(y_centers), len(x_centers))
            dst_crs = f"EPSG:{target_epsg}"

            destination = np.full(dst_shape, np.nan, dtype=np.float64)
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    except rasterio.errors.RasterioIOError as exc:
        raise TerrainError(f"DEM GeoTIFF 파일을 열 수 없습니다: {exc}") from exc

    if not np.isfinite(destination).any():
        raise TerrainError("DEM이 역산 대상 영역과 겹치지 않습니다.")

    return destination
