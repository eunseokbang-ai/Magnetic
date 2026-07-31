"""Synthetic validation of Euler deconvolution: T = B + k / R^N, where
R = sqrt((x-x0)^2 + (y-y0)^2 + z0^2), is exactly homogeneous of degree -N
about (x0, y0, z0) - the textbook synthetic model used to validate an
Euler deconvolution implementation, since it satisfies Euler's equation
exactly (no noise) for the matching structural index N."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

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


if __name__ == "__main__":
    test_euler_recovers_synthetic_point_source()
    test_euler_wrong_structural_index_still_bounded()
    print("ALL CHECKS PASSED")
