"""Sample a single pixel value (and, if present, an indexed colormap
entry) from a project-scoped reference-layer GeoTIFF at a given lat/lon -
used by the chat assistant to ground answers about background geology in
the actual uploaded map rather than guessing from a screenshot."""
from __future__ import annotations

import io

import rasterio
from rasterio.warp import transform as warp_transform


class GeologySampleError(ValueError):
    pass


def sample_geotiff_at_point(data: bytes, lat: float, lon: float) -> dict:
    try:
        with rasterio.open(io.BytesIO(data)) as src:
            if src.crs is None:
                raise GeologySampleError("참조 레이어에 좌표계(CRS) 정보가 없습니다.")
            xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
            x, y = xs[0], ys[0]
            row, col = src.index(x, y)
            if not (0 <= row < src.height and 0 <= col < src.width):
                return {"in_bounds": False}

            band1 = src.read(1, window=((row, row + 1), (col, col + 1)))
            value = float(band1[0, 0])
            nodata = src.nodata
            if nodata is not None and value == nodata:
                return {"in_bounds": True, "value": None}

            result: dict = {"in_bounds": True, "value": value}
            try:
                colormap = src.colormap(1)
            except ValueError:
                colormap = None
            if colormap:
                idx = int(value)
                if idx in colormap:
                    result["color_rgba"] = list(colormap[idx])
            return result
    except rasterio.errors.RasterioIOError as exc:
        raise GeologySampleError(f"참조 레이어를 열 수 없습니다: {exc}") from exc
