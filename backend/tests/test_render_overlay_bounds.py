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
UTM_EPSG_106E = 32648  # zone 48N, central meridian 105E


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


def test_axis_aligned_bounds_alone_misplace_a_tall_grid_far_from_the_central_meridian():
    """Regression test for a second, independent coordinate bug: `bounds`
    is only the axis-aligned N/S/E/W envelope of the image's 4 corners, not
    its true footprint. A UTM grid's rows/columns are exactly north-south/
    east-west only along its own zone's central meridian; anywhere else,
    "grid north" is rotated away from true north (map convergence), so a
    tall/wide grid's real shape on a lat/lon map is a sheared parallelogram,
    not an axis-aligned rectangle. Leaflet's plain <ImageOverlay bounds=..>
    can only stretch the PNG into that axis-aligned box, silently discarding
    the shear - every pixel except the two corners that happen to be both
    the N/S and E/W extremes drifts from its true position. This is what
    let a user see the *color* at a map location visibly disagree with the
    *number* "지점값 확인" reports for that same location (the number is
    always exact - see sample_overlay_value, which reprojects each click
    independently rather than reading off this image), even after the
    half-cell bug above was fixed. The frontend now positions the image via
    the 3 corners below (leaflet-imageoverlay-rotated) instead of `bounds`."""
    # A grid running well north-south (tall, narrow - a common survey
    # shape) sitting ~1.3 degrees of longitude east of UTM zone 48N's
    # central meridian (105E) at a mid-latitude, close to this app's own
    # real sample-fixture location/shape (see test_overlay_sample.py) -
    # exactly where map convergence is large enough to matter.
    cell = 10.0
    n_e, n_n = 40, 400
    easting = 597500.0 + np.arange(n_e) * cell
    northing = 5148400.0 + np.arange(n_n) * cell
    values = np.random.default_rng(0).normal(size=(n_n, n_e))

    overlay = grid_to_png_overlay(values, easting, northing, UTM_EPSG_106E, cmap_name="viridis", vmin=-3, vmax=3, cell_size_m=cell)

    assert overlay["topleft"] and overlay["topright"] and overlay["bottomleft"]
    south, west = overlay["bounds"][0]
    north, east = overlay["bounds"][1]
    tl_lat, tl_lon = overlay["topleft"]
    tr_lat, tr_lon = overlay["topright"]
    bl_lat, bl_lon = overlay["bottomleft"]

    # bounds must be a valid envelope of the 3 true corners (used only as a
    # fallback/fit-to-view helper, not for placing the image).
    assert south <= min(tl_lat, tr_lat, bl_lat) + 1e-9
    assert north >= max(tl_lat, tr_lat, bl_lat) - 1e-9
    assert west <= min(tl_lon, tr_lon, bl_lon) + 1e-9
    assert east >= max(tl_lon, tr_lon, bl_lon) - 1e-9

    # A plain axis-aligned <ImageOverlay bounds=..> linearly stretches pixel
    # fraction (col_frac, row_frac) to lat = south + row_frac*(north-south),
    # lon = west + col_frac*(east-west) - reproduce that placement for the
    # true east-edge, south-edge (bottom-right) node and confirm it lands a
    # real, visually-noticeable distance from that node's actual reprojected
    # position (the true position sample_overlay_value/the rotated overlay
    # both use), proving this grid's real shear is large enough to matter.
    transformer = Transformer.from_crs(f"EPSG:{UTM_EPSG_106E}", "EPSG:4326", always_xy=True)
    true_br_lon, true_br_lat = transformer.transform(easting[-1] + cell / 2.0, northing[0] - cell / 2.0)
    naive_br_lat = south  # row_frac=0 -> south edge
    naive_br_lon = east  # col_frac=1 -> east edge

    import math

    dlat_m = (naive_br_lat - true_br_lat) * 111320
    dlon_m = (naive_br_lon - true_br_lon) * 111320 * math.cos(math.radians(true_br_lat))
    shear_error_m = math.hypot(dlat_m, dlon_m)
    assert shear_error_m > 5.0  # a real, visually-noticeable amount
