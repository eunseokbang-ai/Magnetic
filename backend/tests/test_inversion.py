"""Synthetic validation of the 3D magnetic susceptibility inversion:
forward-model a single buried susceptible block through the real
sensitivity-matrix builder, invert the resulting (noise-free) TMI
response, and confirm the recovered model peaks near the true block -
plus targeted tests for the two new reliability additions (automatic
discrepancy-principle regularization, and the per-layer resolution
diagnostic)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from app.processing.inversion import (
    InversionResult,
    box_faces,
    build_mesh,
    build_sensitivity_matrix,
    invert,
    resolution_diagnostics,
)

RNG = np.random.default_rng(0)


def _make_mesh():
    x_centers = np.arange(-100.0, 101.0, 20.0)  # 11 columns
    y_centers = np.arange(-100.0, 101.0, 20.0)  # 11 rows
    ground_elevation = np.full((len(y_centers), len(x_centers)), 50.0)
    return build_mesh(x_centers, y_centers, ground_elevation, cell_size_m=20.0, depth_extent_m=100.0, n_layers=5)


def _make_observations(mesh):
    obs_x, obs_y = np.meshgrid(np.arange(-90.0, 91.0, 15.0), np.arange(-90.0, 91.0, 15.0))
    obs_x, obs_y = obs_x.ravel(), obs_y.ravel()
    obs_z = np.full(obs_x.shape, 60.0)  # 10m above the flat 50m ground
    return obs_x, obs_y, obs_z


def test_inversion_recovers_synthetic_block_location():
    mesh = _make_mesh()
    obs_x, obs_y, obs_z = _make_observations(mesh)

    inclination_deg, declination_deg, field_intensity_nt = 55.0, -8.0, 50000.0
    G, rows, cols, layers = build_sensitivity_matrix(obs_x, obs_y, obs_z, mesh, inclination_deg, declination_deg, field_intensity_nt)

    # Inject a single "true" susceptibility block at the mesh center,
    # second-shallowest layer (index 1), and forward-model the noise-free
    # data through the same G used for the inversion.
    true_m = np.zeros(G.shape[1])
    center_row = np.argmin(np.abs(mesh.y_centers - 0.0))
    center_col = np.argmin(np.abs(mesh.x_centers - 0.0))
    target_layer = 1
    target_idx = np.flatnonzero((rows == center_row) & (cols == center_col) & (layers == target_layer))
    assert target_idx.size == 1, "expected exactly one active cell at the target location"
    true_m[target_idx[0]] = 0.05  # SI, a plausible moderately-magnetic body
    data_nt = G @ true_m

    result = invert(G, data_nt, mesh, rows, cols, layers, regularization_strength=0.3, n_irls_iterations=6)

    assert result.rms_misfit_nt < 1.0, f"expected a good fit to noise-free synthetic data, got {result.rms_misfit_nt}"

    peak_flat_idx = int(np.argmax(result.susceptibility))
    peak_row, peak_col, peak_layer = np.unravel_index(peak_flat_idx, result.susceptibility.shape)
    print(f"true=({center_row},{center_col},{target_layer}) recovered_peak=({peak_row},{peak_col},{peak_layer})")
    assert abs(peak_row - center_row) <= 1
    assert abs(peak_col - center_col) <= 1
    assert abs(peak_layer - target_layer) <= 1


def test_auto_regularization_targets_assumed_noise():
    mesh = _make_mesh()
    obs_x, obs_y, obs_z = _make_observations(mesh)
    G, rows, cols, layers = build_sensitivity_matrix(obs_x, obs_y, obs_z, mesh, 55.0, -8.0, 50000.0)

    true_m = np.zeros(G.shape[1])
    true_m[G.shape[1] // 2] = 0.03
    data_nt = G @ true_m + RNG.normal(0, 0.3, G.shape[0])

    target_noise = 0.5
    result = invert(G, data_nt, mesh, rows, cols, layers, assumed_noise_nt=target_noise, n_irls_iterations=5)

    print(f"target={target_noise} regularization_strength_used={result.regularization_strength_used:.4g} "
          f"final_rms={result.rms_misfit_nt:.3f}")
    # The discrepancy-principle search targets the *first-pass* misfit
    # exactly; the full IRLS run afterwards can drift from that somewhat,
    # so this checks it lands in the right ballpark, not an exact match.
    assert result.rms_misfit_nt < target_noise * 3.0
    assert result.regularization_strength_used > 0


def test_stronger_target_noise_yields_larger_regularization():
    """A looser (larger) target misfit should be reachable with *more*
    regularization than a tighter one - misfit increases monotonically
    with regularization strength, so this is a basic sanity check that
    the discrepancy-principle search moves in the right direction."""
    mesh = _make_mesh()
    obs_x, obs_y, obs_z = _make_observations(mesh)
    G, rows, cols, layers = build_sensitivity_matrix(obs_x, obs_y, obs_z, mesh, 55.0, -8.0, 50000.0)
    true_m = np.zeros(G.shape[1])
    true_m[G.shape[1] // 3] = 0.04
    data_nt = G @ true_m + RNG.normal(0, 0.2, G.shape[0])

    tight = invert(G, data_nt, mesh, rows, cols, layers, assumed_noise_nt=0.3, n_irls_iterations=3)
    loose = invert(G, data_nt, mesh, rows, cols, layers, assumed_noise_nt=3.0, n_irls_iterations=3)
    print(f"tight reg={tight.regularization_strength_used:.4g}, loose reg={loose.regularization_strength_used:.4g}")
    assert loose.regularization_strength_used > tight.regularization_strength_used


def test_resolution_diagnostics_decreases_with_depth():
    mesh = _make_mesh()
    obs_x, obs_y, obs_z = _make_observations(mesh)
    G, rows, cols, layers = build_sensitivity_matrix(obs_x, obs_y, obs_z, mesh, 55.0, -8.0, 50000.0)
    n_layers = mesh.active.shape[2]

    diag = resolution_diagnostics(G, layers, n_layers)
    relative = diag["per_layer_relative_sensitivity"]
    print("per-layer relative sensitivity:", relative)
    # Layer 0 is shallowest (closest to the sensors) - physically it must
    # have the highest sensitivity, and deeper layers should fall off
    # monotonically (or very close to it) since every cell has the same
    # susceptibility-response character, only depth changes.
    assert relative[0] == max(relative)
    assert all(relative[i] >= relative[i + 1] - 1e-6 for i in range(len(relative) - 1))
    assert diag["n_cells_per_layer"][0] > 0


def test_resolution_diagnostics_flags_poorly_resolved_layers():
    # Hand-built G: 3 layers, first has strong sensitivity, last is ~1000x weaker.
    layers = np.array([0, 0, 1, 1, 2, 2])
    G = np.zeros((4, 6))
    G[:, 0:2] = 1.0     # layer 0: strong
    G[:, 2:4] = 0.1      # layer 1: moderate
    G[:, 4:6] = 0.0001   # layer 2: negligible
    diag = resolution_diagnostics(G, layers, n_layers=3)
    assert diag["poorly_resolved_layers"] == [2]


def test_build_mesh_growth_factor_one_is_uniform():
    """growth_factor=1.0 (the default) must reproduce the original
    uniform-thickness mesh exactly - no behavior change for existing
    callers that never pass growth_factor."""
    x_centers = np.arange(-100.0, 101.0, 20.0)
    y_centers = np.arange(-100.0, 101.0, 20.0)
    ground_elevation = np.full((len(y_centers), len(x_centers)), 50.0)
    mesh = build_mesh(x_centers, y_centers, ground_elevation, cell_size_m=20.0, depth_extent_m=100.0, n_layers=5)
    assert np.allclose(mesh.layer_thickness_m, 20.0)
    assert np.allclose(np.diff(mesh.z_centers), -20.0)


def test_build_mesh_graded_thickness_increases_with_depth():
    """growth_factor > 1.0 should give a strictly increasing per-layer
    thickness schedule (index 0 = shallowest/thinnest), summing back to
    the requested depth_extent_m - the UBC-GIF/SimPEG "core mesh" grading
    this feature implements."""
    x_centers = np.arange(-100.0, 101.0, 20.0)
    y_centers = np.arange(-100.0, 101.0, 20.0)
    ground_elevation = np.full((len(y_centers), len(x_centers)), 50.0)
    depth_extent_m = 200.0
    mesh = build_mesh(
        x_centers, y_centers, ground_elevation, cell_size_m=20.0,
        depth_extent_m=depth_extent_m, n_layers=8, growth_factor=1.15,
    )
    thickness = mesh.layer_thickness_m
    print("graded layer thickness:", thickness)
    assert len(thickness) == 8
    assert all(thickness[i + 1] > thickness[i] for i in range(len(thickness) - 1))
    assert np.isclose(float(np.sum(thickness)), depth_extent_m)
    # shallowest layer should hug the surface much finer than the old
    # uniform 200/8=25m slabs would have.
    assert thickness[0] < 25.0


def test_box_faces_top_follows_sloped_terrain():
    """box_faces()'s top surface must vary with each column's real ground
    elevation - a single flat top (the bug this session fixed) would mean
    every finite top_z value is identical even though the input terrain
    genuinely slopes."""
    x_centers = np.arange(-100.0, 101.0, 20.0)  # 11 columns
    y_centers = np.arange(-100.0, 101.0, 20.0)  # 11 rows
    # A real east-west slope: 40m of relief across the mesh.
    ground_elevation = 50.0 + 2.0 * x_centers[np.newaxis, :] / 10.0
    ground_elevation = np.broadcast_to(ground_elevation, (len(y_centers), len(x_centers))).copy()
    mesh = build_mesh(
        x_centers, y_centers, ground_elevation, cell_size_m=20.0,
        depth_extent_m=120.0, n_layers=10, growth_factor=1.15,
    )
    susceptibility = np.zeros(mesh.active.shape, dtype=float)
    result = InversionResult(
        mesh=mesh, susceptibility=susceptibility,
        predicted_nt=np.zeros(1), observed_nt=np.zeros(1),
        rms_misfit_nt=0.0, n_active_cells=int(mesh.active.sum()), n_obs=1, iterations=1,
    )
    faces = box_faces(result, top_layer_index=0)
    top_z = np.array(faces["top"]["z"])
    finite = top_z[np.isfinite(top_z)]
    print("top_z unique finite values:", np.unique(finite))
    assert len(np.unique(np.round(finite, 3))) > 1, "top surface should vary across columns for sloped terrain"
    # the west edge (lowest x) should sit lower than the east edge (highest x)
    assert np.nanmean(top_z[:, 0]) < np.nanmean(top_z[:, -1])


if __name__ == "__main__":
    test_inversion_recovers_synthetic_block_location()
    test_auto_regularization_targets_assumed_noise()
    test_stronger_target_noise_yields_larger_regularization()
    test_resolution_diagnostics_decreases_with_depth()
    test_resolution_diagnostics_flags_poorly_resolved_layers()
    test_build_mesh_growth_factor_one_is_uniform()
    test_build_mesh_graded_thickness_increases_with_depth()
    test_box_faces_top_follows_sloped_terrain()
    print("ALL CHECKS PASSED")
