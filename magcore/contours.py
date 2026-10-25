"""Contour line extraction from a gridded surface, returned as lat/lon
polylines for a vector overlay on the Leaflet map - kept separate from the
raster PNG so lines stay crisp at any zoom and can be toggled without a
re-render."""
from __future__ import annotations

import contourpy
import numpy as np
from pyproj import Transformer


def compute_contours(
    grid_values: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    utm_epsg: int,
    interval: float | None = None,
    n_levels: int = 10,
) -> dict:
    """Returns {"levels": [nT, ...], "features": [{"level": nT, "path": [[lat, lon], ...]}, ...]}."""
    finite = grid_values[np.isfinite(grid_values)]
    if finite.size == 0:
        return {"levels": [], "features": []}

    vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    if vmin >= vmax:
        return {"levels": [], "features": []}

    if interval and interval > 0:
        start = np.ceil(vmin / interval) * interval
        levels = np.arange(start, vmax, interval)
    else:
        # skip the exact min/max - contours right at the data extremes are
        # usually degenerate (a single point) or empty
        levels = np.linspace(vmin, vmax, n_levels + 2)[1:-1]

    if len(levels) == 0:
        return {"levels": [], "features": []}

    east2d, north2d = np.meshgrid(easting, northing)
    masked = np.ma.masked_invalid(grid_values)
    cg = contourpy.contour_generator(x=east2d, y=north2d, z=masked, line_type=contourpy.LineType.Separate)

    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

    features = []
    for level in levels:
        for line in cg.lines(float(level)):
            if len(line) < 2:
                continue
            lon, lat = transformer.transform(line[:, 0], line[:, 1])
            features.append({"level": float(level), "path": [[float(a), float(b)] for a, b in zip(lat, lon)]})

    return {"levels": [float(l) for l in levels], "features": features}
