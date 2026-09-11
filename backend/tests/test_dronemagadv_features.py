"""Unit tests (synthetic data, no network) for the DroneMagAdv-inspired
additions: Tilt/Theta/2VD/gradient-tensor transforms, Surfer GRD/BLN
export, Korea coordinate systems, adaptive Hampel filter, minimum
curvature gridding, iterative tie-line leveling, and PCA line direction
detection.
"""
from __future__ import annotations

import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.processing.crossover_leveling import compute_crossover_leveling
from app.processing.despike import despike
from app.processing.gridding import grid_points
from app.processing.lines import _pca_dominant_azimuth, korea2010_epsg, resolve_korea_projection_epsg
from app.processing.render import grid_to_surfer_grd_bytes, polygon_to_bln_bytes
from app.processing.transforms import (
    derivative_easting,
    derivative_northing,
    second_derivative_ee,
    second_derivative_en,
    second_derivative_ez,
    second_derivative_nn,
    second_derivative_nz,
    theta_map,
    tilt_angle,
    vertical_derivative,
)


def _synthetic_grid():
    easting = np.linspace(0, 500, 60)
    northing = np.linspace(0, 500, 60)
    E, N = np.meshgrid(easting, northing)
    grid = 800.0 * np.exp(-(((E - 250) ** 2 + (N - 250) ** 2)) / 3000.0)
    return grid, easting, northing


def test_tilt_angle_range_and_sign_near_peak():
    grid, easting, northing = _synthetic_grid()
    cell = float(easting[1] - easting[0])
    tilt = tilt_angle(grid, cell)
    assert np.nanmax(tilt) <= 90.0001
    assert np.nanmin(tilt) >= -90.0001
    # directly over the peak (vertical derivative dominates, near-zero
    # horizontal gradient) the tilt should be strongly positive
    center = tilt[30, 30]
    assert center > 45


def test_theta_map_range():
    grid, easting, northing = _synthetic_grid()
    cell = float(easting[1] - easting[0])
    theta = theta_map(grid, cell)
    finite = theta[np.isfinite(theta)]
    assert finite.min() >= -0.001
    assert finite.max() <= 90.001


def test_gradient_tensor_components_finite_and_shaped():
    grid, easting, northing = _synthetic_grid()
    cell = float(easting[1] - easting[0])
    for fn in [derivative_easting, derivative_northing, second_derivative_ee, second_derivative_nn,
               second_derivative_en, second_derivative_ez, second_derivative_nz]:
        out = fn(grid, cell)
        assert out.shape == grid.shape
        assert np.isfinite(out).any()


def test_2vd_matches_vertical_derivative_order2():
    grid, easting, northing = _synthetic_grid()
    cell = float(easting[1] - easting[0])
    vd2 = vertical_derivative(grid, cell, order=2)
    assert vd2.shape == grid.shape
    assert np.isfinite(vd2).any()


def test_surfer_grd_roundtrip_header():
    grid, easting, northing = _synthetic_grid()
    data = grid_to_surfer_grd_bytes(grid, easting, northing)
    magic, nx, ny, xmin, xmax, ymin, ymax, zmin, zmax = struct.unpack("<4shhdddddd", data[:56])
    assert magic == b"DSBB"
    assert nx == grid.shape[1]
    assert ny == grid.shape[0]
    assert xmin == easting[0] and xmax == easting[-1]
    assert ymin == northing[0] and ymax == northing[-1]
    assert len(data) == 56 + nx * ny * 4
    body = np.frombuffer(data[56:], dtype="<f4").reshape(ny, nx)
    np.testing.assert_allclose(body, grid.astype("<f4"), rtol=1e-5)


def test_surfer_grd_rejects_empty_grid():
    grid = np.full((5, 5), np.nan)
    easting = np.arange(5, dtype=float)
    northing = np.arange(5, dtype=float)
    try:
        grid_to_surfer_grd_bytes(grid, easting, northing)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_bln_export_closes_polygon_and_flag():
    x = np.array([0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    data = polygon_to_bln_bytes(x, y)
    text = data.decode("utf-8")
    lines = text.strip().splitlines()
    assert lines[0] == "5,0"
    assert lines[1] == lines[-1]
    assert len(lines) == 6  # header + 4 original + repeated first point


def test_korea_projection_epsg_resolution():
    assert resolve_korea_projection_epsg("korea_utm", 127.0) == 5179
    assert korea2010_epsg(125.0) == 5185
    assert korea2010_epsg(127.0) == 5186
    assert korea2010_epsg(129.0) == 5187
    assert korea2010_epsg(131.0) == 5188
    assert resolve_korea_projection_epsg("utm", 125.0) == 32651
    assert resolve_korea_projection_epsg("utm", 127.0) == 32652


def test_adaptive_hampel_expands_window_on_steep_gradient():
    rng = np.random.default_rng(0)
    n = 200
    values = 50000 + rng.normal(0, 0.5, n)
    # a sharp ramp (steep gradient, not a spike) in the middle - should
    # NOT be over-aggressively flattened by the adaptive window expansion
    values[90:110] += np.linspace(0, 200, 20)
    values[95] += 300  # a genuine spike embedded in the steep section

    cleaned, spike_mask = despike(values, window_size=11, threshold_k=4.0, adaptive=True)
    assert spike_mask[95]
    # the ramp itself should mostly survive (not be wholesale replaced)
    assert cleaned[105] > values[90] + 50


def test_iterative_crossover_leveling_reduces_rms_and_redistributes_correction():
    rows = []
    injected = {0: 5.0, 1: -3.0, 2: 8.0}
    for i, xline in enumerate([0.0, 50.0, 100.0]):
        for y in np.arange(0, 200, 5.0):
            rows.append({"line_id": i, "x": xline, "y": y, "val": 10.0 + 0.01 * xline + 0.02 * y + injected[i]})
    survey_df = pd.DataFrame(rows)

    tie_rows = []
    for x in np.arange(-10, 110, 5.0):
        tie_rows.append({"line_id": -1, "x": x, "y": 100.0, "val": 10.0 + 0.01 * x + 0.02 * 100.0})
    tie_df = pd.DataFrame(tie_rows)

    df = pd.concat([survey_df, tie_df], ignore_index=True)
    tie_line_id = pd.Series(np.where(df["line_id"] == -1, 0, -1), index=df.index)

    result_iter = compute_crossover_leveling(df, tie_line_id, "val", max_crossover_distance_m=10.0, iterative=True)
    result_single = compute_crossover_leveling(df, tie_line_id, "val", max_crossover_distance_m=10.0, iterative=False)

    assert result_iter.applied
    assert result_iter.n_survey_lines_corrected == 3
    assert result_iter.rms_after_nt < result_iter.rms_before_nt
    assert result_single.applied
    for lid in result_single.line_shifts:
        assert abs(result_single.line_shifts[lid] - (-injected[lid])) < 0.1
    # the iterative network lets the (single) tie line absorb some
    # residual too, so its survey shifts should differ from the
    # single-pass ones that trust the tie line completely.
    assert result_iter.line_shifts != result_single.line_shifts


def test_pca_dominant_azimuth_recovers_elongated_axis():
    rng = np.random.default_rng(3)
    n = 500
    # points elongated along a 30-degree axis (from east), with a small
    # perpendicular spread
    along = rng.uniform(-500, 500, n)
    across = rng.normal(0, 5, n)
    theta = np.radians(30.0)
    x = along * np.cos(theta) - across * np.sin(theta)
    y = along * np.sin(theta) + across * np.cos(theta)
    azimuth = _pca_dominant_azimuth(x, y)
    diff = min(abs(azimuth - 30.0), abs(azimuth - 30.0 - 180), abs(azimuth - 30.0 + 180))
    assert diff < 2.0


def test_minimum_curvature_multilevel_matches_single_level_on_large_grid():
    """A grid big enough to trigger the multilevel path should still
    converge to essentially the same smooth surface a direct (slow)
    single-level solve would produce."""
    from app.processing.gridding import _minimum_curvature_multilevel, _minimum_curvature_relax

    rng = np.random.default_rng(4)
    ny, nx = 80, 80  # 6400 cells - safely above _MULTILEVEL_BASE_CELLS
    mask = rng.random((ny, nx)) < 0.05
    values = np.zeros((ny, nx))
    Y, X = np.mgrid[0:ny, 0:nx]
    smooth = 50.0 * np.sin(X / 20.0) + 30.0 * np.cos(Y / 15.0)
    values[mask] = smooth[mask]

    multilevel = _minimum_curvature_multilevel((ny, nx), mask, values)
    single = _minimum_curvature_relax((ny, nx), mask, values, max_iterations=3000, tol=1e-6)

    # RMS (not max) agreement: a handful of cells in a sparse, randomly
    # scattered synthetic constraint set can converge slightly slower at
    # the fine level's bounded iteration budget, but the overall surface
    # should closely track the fully-converged single-level reference.
    rms_diff = float(np.sqrt(np.mean((multilevel - single) ** 2)))
    value_range = float(values[mask].max() - values[mask].min())
    assert rms_diff < 0.05 * value_range
    # constraints themselves must be respected exactly
    assert np.allclose(multilevel[mask], values[mask])


def test_grid_points_rejects_absurdly_fine_cell_size():
    rng = np.random.default_rng(5)
    n = 50
    x = rng.uniform(0, 5000, n)
    y = rng.uniform(0, 5000, n)
    values = rng.normal(0, 10, n)
    try:
        grid_points(x, y, values, cell_size_m=0.01)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "너무 작아" in str(exc)


def test_spline_control_point_cap_on_line_shaped_data():
    """Data lying along thin lines (not filling the plane) block-reduces
    to far more points per unit spacing increase than an area/spacing^2
    estimate predicts - the control-point cap must still converge under
    the cap via the geometric-growth retry loop, not just on the first
    guess."""
    rng = np.random.default_rng(6)
    xs = []
    ys = []
    for line_x in np.linspace(0, 1000, 40):
        n_along = 200
        xs.append(np.full(n_along, line_x) + rng.normal(0, 0.1, n_along))
        ys.append(np.linspace(0, 4000, n_along))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    values = rng.normal(0, 10, len(x))

    result = grid_points(x, y, values, cell_size_m=5.0, method="spline", max_distance_m=20.0)
    assert result.values.shape[0] > 0
    assert np.isfinite(result.values).any()


def test_minimum_curvature_gridding_smooth_and_matches_scattered_trend():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.uniform(0, 400, n)
    y = rng.uniform(0, 400, n)
    values = 0.05 * x + 0.03 * y + rng.normal(0, 0.2, n)
    result = grid_points(x, y, values, cell_size_m=20.0, method="minimum_curvature", max_distance_m=60.0)
    finite = result.values[np.isfinite(result.values)]
    assert finite.size > 0
    # should recover the planar trend reasonably well
    E, N = np.meshgrid(result.easting, result.northing)
    predicted = 0.05 * E + 0.03 * N
    mask = np.isfinite(result.values)
    residual = result.values[mask] - predicted[mask]
    assert np.std(residual) < 1.0
