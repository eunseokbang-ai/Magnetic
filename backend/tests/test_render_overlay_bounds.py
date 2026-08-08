"""Regression test for a real coordinate-registration bug in
processing/render.py::grid_to_png_overlay: the PNG's `bounds` (used by
Leaflet's ImageOverlay to stretch the image across the map) were computed
from the outermost grid *node* coordinates directly, with no allowance for
each pixel representing a whole cell's worth of ground around that node
("pixel is area", the same convention grid_to_geotiff_bytes already used
correctly via rasterio's from_origin(easting[0] - cell_size/2, ...)).

Without the half-cell extension, stretching an N-pixel-wide image across a
bound that is only (N-1)*cell_size wide (instead of N*cell_size) makes
every pixel's true rendered position drift away from its real node
coordinate - zero error exactly at the grid's own center, growing to a
full half-cell at the edges. This is what let a user click on what looked
like a hot/anomalous cell and get back the neighboring cell's very
different value (or "no data" over a visually-colored cell, and vice
versa) - see store.py::sample_overlay_value, which relies on the same
half-cell convention to sample the exact cell whose color the map shows.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import base64
import io

import numpy as np
import pytest
from PIL import Image
from pyproj import Transformer

from app.processing.render import grid_to_png_overlay

UTM_EPSG = 32652


def _decode(overlay):
    b64 = overlay["image_data_url"].split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_off_center_cell_renders_at_its_true_geographic_position():
    cell = 10.0
    n = 9
    easting = np.arange(0, n * cell, cell)
    northing = np.arange(0, n * cell, cell)
    values = np.zeros((n, n))
    hot_row, hot_col = 1, 7  # deliberately off-center - the bug is invisible at the exact center
    values[hot_row, hot_col] = 1.0

    overlay = grid_to_png_overlay(values, easting, northing, UTM_EPSG, cmap_name="viridis", vmin=0, vmax=1, cell_size_m=cell)
    img = _decode(overlay)
    W, H = img.size
    south, west = overlay["bounds"][0]
    north, east = overlay["bounds"][1]

    transformer = Transformer.from_crs(f"EPSG:{UTM_EPSG}", "EPSG:4326", always_xy=True)
    lon_hot, lat_hot = transformer.transform(easting[hot_col], northing[hot_row])

    # fractional pixel the hot node's true coordinate maps to under the
    # returned bounds - must land exactly at the pixel *center* (col+0.5,
    # row+0.5, row counted from the image top since PNG row 0 = north)
    px = (lon_hot - west) / (east - west) * W
    py = (north - lat_hot) / (north - south) * H
    assert px == pytest.approx(hot_col + 0.5, abs=1e-6)
    assert py == pytest.approx((n - 1 - hot_row) + 0.5, abs=1e-6)
