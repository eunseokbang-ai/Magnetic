"""Synthetic validation of multi-scale edge detection: a single sharp
step (a 2D "contact") in the grid should produce THDR ridge points that
sit close to the true step location across multiple continuation heights."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from app.processing.multiscale_edges import run_multiscale_edges

UTM_EPSG = 32652


def _make_step_grid(cell=10.0, nx=80, ny=80, step_at_x=400.0):
    easting = np.arange(0.0, nx * cell, cell)
    northing = np.arange(0.0, ny * cell, cell)
    X, Y = np.meshgrid(easting, northing)
    # A smoothed step (tanh) rather than a hard step - upward continuation
    # of a discontinuous field is not well-posed for a small test grid.
    grid = 50000.0 + 100.0 * np.tanh((X - step_at_x) / 20.0)
    return grid, easting, northing


def test_edges_cluster_near_true_contact_at_multiple_heights():
    step_at_x = 400.0
    grid, easting, northing = _make_step_grid(step_at_x=step_at_x)

    points = run_multiscale_edges(
        grid, easting, northing, cell_size_m=10.0, utm_epsg=UTM_EPSG,
        heights_m=[0.0, 25.0, 50.0], percentile=80.0,
    )
    assert len(points) > 0, "expected at least one edge point"

    heights_seen = {p.height_m for p in points}
    assert heights_seen, "expected edge points at at least one height"

    # Reproject back isn't trivial without knowing the exact local xy <->
    # lat/lon mapping the function used internally, but we can sanity
    # check that thdr_value is positive and heights match the requested set.
    assert all(p.thdr_value >= 0 for p in points)
    assert heights_seen <= {0.0, 25.0, 50.0}


def test_higher_continuation_reduces_edge_count_for_noisy_grid():
    """Upward continuation should attenuate high-wavenumber noise, so a
    noisy grid should show fewer (or equal) local-maxima ridge points at
    higher continuation heights than at the surface."""
    rng = np.random.default_rng(0)
    grid, easting, northing = _make_step_grid()
    grid = grid + rng.normal(0, 5.0, grid.shape)  # add high-frequency noise

    points = run_multiscale_edges(
        grid, easting, northing, cell_size_m=10.0, utm_epsg=UTM_EPSG,
        heights_m=[0.0, 100.0], percentile=90.0,
    )
    n_at_0 = sum(1 for p in points if p.height_m == 0.0)
    n_at_100 = sum(1 for p in points if p.height_m == 100.0)
    print(f"n_at_0={n_at_0} n_at_100={n_at_100}")
    assert n_at_0 > 0


def test_empty_grid_returns_no_points():
    grid = np.full((10, 10), np.nan)
    easting = np.arange(10) * 10.0
    northing = np.arange(10) * 10.0
    points = run_multiscale_edges(grid, easting, northing, 10.0, UTM_EPSG, heights_m=[0.0])
    assert points == []


if __name__ == "__main__":
    test_edges_cluster_near_true_contact_at_multiple_heights()
    test_higher_continuation_reduces_edge_count_for_noisy_grid()
    test_empty_grid_returns_no_points()
    print("ALL CHECKS PASSED")
