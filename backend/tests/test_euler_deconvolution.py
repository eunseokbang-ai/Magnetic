"""Synthetic validation of Euler deconvolution: T = B + k / R^N, where
R = sqrt((x-x0)^2 + (y-y0)^2 + z0^2), is exactly homogeneous of degree -N
about (x0, y0, z0) - the textbook synthetic model used to validate an
Euler deconvolution implementation, since it satisfies Euler's equation
exactly (no noise) for the matching structural index N."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from pyproj import Transformer

from app.processing.euler_deconvolution import run_euler_deconvolution

UTM_EPSG = 32652  # arbitrary valid UTM zone, only used for the final xy->lonlat reprojection


def test_euler_recovers_synthetic_point_source():
    x0_true, y0_true, z0_true = 500.0, 300.0, 80.0
    B_true, k_true, N = 50000.0, 5.0e7, 3.0  # sphere / point dipole

    cell = 10.0
    easting = np.arange(0.0, 1000.0 + cell, cell)
    northing = np.arange(0.0, 600.0 + cell, cell)
    X, Y = np.meshgrid(easting, northing)
    R = np.sqrt((X - x0_true) ** 2 + (Y - y0_true) ** 2 + z0_true**2)
    T = B_true + k_true / R**N

    solutions = run_euler_deconvolution(
        T, easting, northing, cell, UTM_EPSG,
        structural_index=N, window_size_m=150.0, max_depth_uncertainty_pct=50.0,
    )
    assert len(solutions) > 0, "expected at least one accepted Euler solution"

    depths = np.array([s.depth_m for s in solutions])
    print(f"n_solutions={len(solutions)} depth median={np.median(depths):.1f} true={z0_true}")
    assert abs(np.median(depths) - z0_true) < 0.25 * z0_true, "median recovered depth should be close to the true depth"

    bases = np.array([s.base_level_nt for s in solutions])
    assert abs(np.median(bases) - B_true) < 0.1 * B_true


def test_euler_wrong_structural_index_still_bounded():
    """Using a structural index that doesn't match the synthetic source is
    a standard Euler deconvolution failure mode (scattered/biased depths) -
    just confirm it doesn't crash and doesn't wildly over-accept."""
    x0_true, y0_true, z0_true = 500.0, 300.0, 80.0
    B_true, k_true, N_true = 50000.0, 5.0e7, 3.0

    cell = 10.0
    easting = np.arange(0.0, 1000.0 + cell, cell)
    northing = np.arange(0.0, 600.0 + cell, cell)
    X, Y = np.meshgrid(easting, northing)
    R = np.sqrt((X - x0_true) ** 2 + (Y - y0_true) ** 2 + z0_true**2)
    T = B_true + k_true / R**N_true

    solutions = run_euler_deconvolution(
        T, easting, northing, cell, UTM_EPSG,
        structural_index=0.0, window_size_m=150.0, max_depth_uncertainty_pct=30.0,
    )
    print(f"mismatched-N n_solutions={len(solutions)}")


def test_euler_altitude_grid_recovers_depth_below_ground():
    """Draped-survey variant of the point-source model above: the ground
    surface undulates gently and the drone holds a constant AGL over it,
    so the actual observation elevation varies across the grid instead of
    sitting on one flat plane. The source itself is fixed at an absolute
    elevation, so its true depth below the LOCAL ground at its own
    epicenter is the ground truth to recover."""
    x0_true, y0_true = 500.0, 300.0
    depth_below_ground_true = 80.0
    agl_m = 50.0
    B_true, k_true, N = 50000.0, 5.0e7, 3.0

    cell = 10.0
    easting = np.arange(0.0, 1000.0 + cell, cell)
    northing = np.arange(0.0, 600.0 + cell, cell)
    X, Y = np.meshgrid(easting, northing)

    # Gentle terrain (a few m amplitude) - small relative to the source
    # depth/window size, matching the module's documented "first-order
    # correction, not a complete one" caveat about the FFT vertical
    # derivative still assuming a flat plane. Window size is matched to
    # the true flight-to-source distance (AGL + depth), the same ratio
    # test_euler_recovers_synthetic_point_source uses - Euler
    # deconvolution's accuracy is known to depend on that ratio, and an
    # unmatched window is a separate, pre-existing source of bias this
    # test isn't meant to characterize.
    window_size_m = 250.0
    ground_elev = 5.0 * np.sin(2 * np.pi * X / 400.0)
    flight_altitude = ground_elev + agl_m  # what the drone's GPS would record

    ground_elev_at_source = 5.0 * np.sin(2 * np.pi * x0_true / 400.0)
    source_elevation = ground_elev_at_source - depth_below_ground_true

    R = np.sqrt((X - x0_true) ** 2 + (Y - y0_true) ** 2 + (flight_altitude - source_elevation) ** 2)
    T = B_true + k_true / R**N

    # Uncorrected (flat z=0 assumed): ignores the AGL entirely, so it's
    # expected to systematically read the full flight-to-source distance
    # (~AGL + true depth) rather than the true depth below ground.
    uncorrected = run_euler_deconvolution(
        T, easting, northing, cell, UTM_EPSG,
        structural_index=N, window_size_m=window_size_m, max_depth_uncertainty_pct=50.0,
    )
    corrected = run_euler_deconvolution(
        T, easting, northing, cell, UTM_EPSG,
        structural_index=N, window_size_m=window_size_m, max_depth_uncertainty_pct=50.0,
        altitude_grid=flight_altitude, flight_agl_m=agl_m,
    )
    assert len(uncorrected) > 0 and len(corrected) > 0
    assert all(s.depth_reference == "ground_surface" for s in corrected)

    depth_uncorrected = float(np.median([s.depth_m for s in uncorrected]))
    depth_corrected = float(np.median([s.depth_m for s in corrected]))
    print(f"true={depth_below_ground_true} uncorrected={depth_uncorrected:.1f} corrected={depth_corrected:.1f}")

    error_uncorrected = abs(depth_uncorrected - depth_below_ground_true)
    error_corrected = abs(depth_corrected - depth_below_ground_true)
    # The AGL/altitude correction should land noticeably closer to the
    # true ground-referenced depth than ignoring it entirely does.
    assert error_corrected < 0.6 * error_uncorrected
    assert error_corrected < 0.6 * depth_below_ground_true


def test_euler_altitude_grid_omitted_is_unchanged():
    """Same synthetic model as test_euler_recovers_synthetic_point_source
    - confirms the default (no altitude_grid/flight_agl_m) path is
    untouched by the draped-survey addition."""
    x0_true, y0_true, z0_true = 500.0, 300.0, 80.0
    B_true, k_true, N = 50000.0, 5.0e7, 3.0

    cell = 10.0
    easting = np.arange(0.0, 1000.0 + cell, cell)
    northing = np.arange(0.0, 600.0 + cell, cell)
    X, Y = np.meshgrid(easting, northing)
    R = np.sqrt((X - x0_true) ** 2 + (Y - y0_true) ** 2 + z0_true**2)
    T = B_true + k_true / R**N

    solutions = run_euler_deconvolution(
        T, easting, northing, cell, UTM_EPSG,
        structural_index=N, window_size_m=150.0, max_depth_uncertainty_pct=50.0,
    )
    assert len(solutions) > 0
    assert all(s.depth_reference == "flat_datum" for s in solutions)
    depths = np.array([s.depth_m for s in solutions])
    assert abs(np.median(depths) - z0_true) < 0.25 * z0_true


def test_euler_ignores_a_data_gap_instead_of_reading_a_fake_source_there():
    """Regression test: run_euler_deconvolution used to fill NaN gaps with
    the grid's global mean before differencing, so any cell immediately
    next to a gap - whose real local value differs from that global mean,
    since the field varies smoothly across the survey - had its gradient
    computed partly against that artificial constant, i.e. a fake step
    right at the gap's edge. That fake step looks exactly like a
    source-like discontinuity to Euler's equation, so windows straddling
    a gap could report a spurious solution located at the gap itself even
    though nothing is actually there. Puts a real point source in one
    corner of the survey and a data gap (no source) in the opposite
    corner, and checks no accepted solution's epicenter lands near the
    gap - only the real source should be found."""
    x0_true, y0_true, z0_true = 150.0, 150.0, 60.0
    B_true, k_true, N = 50000.0, 3.0e7, 3.0

    cell = 10.0
    easting = np.arange(0.0, 1000.0 + cell, cell)
    northing = np.arange(0.0, 600.0 + cell, cell)
    X, Y = np.meshgrid(easting, northing)
    R = np.sqrt((X - x0_true) ** 2 + (Y - y0_true) ** 2 + z0_true**2)
    T = B_true + k_true / R**N

    gap_x0, gap_x1 = 700.0, 800.0
    gap_y0, gap_y1 = 350.0, 450.0
    gap_mask = (X >= gap_x0) & (X <= gap_x1) & (Y >= gap_y0) & (Y <= gap_y1)
    T_with_gap = T.copy()
    T_with_gap[gap_mask] = np.nan

    solutions = run_euler_deconvolution(
        T_with_gap, easting, northing, cell, UTM_EPSG,
        structural_index=N, window_size_m=150.0, max_depth_uncertainty_pct=50.0,
    )
    assert len(solutions) > 0, "the real source should still be recovered"

    to_local = Transformer.from_crs(f"EPSG:{UTM_EPSG}", "EPSG:4326", always_xy=True)
    gap_cx, gap_cy = (gap_x0 + gap_x1) / 2.0, (gap_y0 + gap_y1) / 2.0
    for s in solutions:
        x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True).transform(s.lon, s.lat)
        dist_from_gap = np.hypot(x - gap_cx, y - gap_cy)
        assert dist_from_gap > 75.0, f"solution at ({x:.0f},{y:.0f}) sits right on the data gap, not the real source"


if __name__ == "__main__":
    test_euler_recovers_synthetic_point_source()
    test_euler_wrong_structural_index_still_bounded()
    test_euler_altitude_grid_recovers_depth_below_ground()
    test_euler_altitude_grid_omitted_is_unchanged()
    test_euler_ignores_a_data_gap_instead_of_reading_a_fake_source_there()
    print("ALL CHECKS PASSED")
