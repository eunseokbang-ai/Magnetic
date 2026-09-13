"""grid_points must return an easting/northing axis whose actual node
spacing exactly matches the requested cell_size_m on BOTH axes - every
FFT-based derivative transform (processing/transforms.py) and this
module's own auto-mask math treat cell_size_m as if it were the true
spacing. verde's default grid_coordinates(adjust="spacing") instead keeps
the exact input region and adjusts the spacing to fit it, which can differ
from cell_size_m (and differently on each axis, since region width/height
aren't generally equal) - grid_points uses adjust="region" instead to
avoid that mismatch."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from app.processing.gridding import grid_points


def test_grid_axis_spacing_exactly_matches_cell_size_on_both_axes():
    rng = np.random.default_rng(0)
    # a region whose width/height are NOT exact multiples of cell_size_m,
    # and not equal to each other - exactly the case that used to produce
    # two different, cell_size_m-disagreeing spacings.
    x = rng.uniform(0.0, 437.0, 300)
    y = rng.uniform(0.0, 291.0, 300)
    values = np.sin(x / 50.0) + np.cos(y / 40.0)
    cell_size_m = 10.0

    grid = grid_points(x, y, values, cell_size_m, method="nearest")

    d_east = np.diff(grid.easting)
    d_north = np.diff(grid.northing)
    assert np.allclose(d_east, cell_size_m, atol=1e-9)
    assert np.allclose(d_north, cell_size_m, atol=1e-9)


if __name__ == "__main__":
    test_grid_axis_spacing_exactly_matches_cell_size_on_both_axes()
    print("ALL CHECKS PASSED")
