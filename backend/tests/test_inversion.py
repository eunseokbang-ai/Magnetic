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
    azimuth_line,
    box_faces,
    build_mesh,
    build_sensitivity_matrix,
    internal_slice,
    invert,
    path_slice_3d,
    resolution_diagnostics,
    vertical_section,
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


def _make_result_with_distinct_values():
    """A zero-forward-model InversionResult whose susceptibility encodes
    its own (row, col, layer) index, so a slice function's output can be
    checked against the exact cells it's supposed to have picked out -
    not just "some plausible-looking numbers"."""
    mesh = _make_mesh()
    susceptibility = np.zeros(mesh.active.shape, dtype=float)
    ny, nx, nz = mesh.active.shape
    for r in range(ny):
        for c in range(nx):
            for k in range(nz):
                if mesh.active[r, c, k]:
                    susceptibility[r, c, k] = 0.001 * (r * 1000 + c * 10 + k)
    return InversionResult(
        mesh=mesh, susceptibility=susceptibility,
        predicted_nt=np.zeros(1), observed_nt=np.zeros(1),
        rms_misfit_nt=0.0, n_active_cells=int(mesh.active.sum()), n_obs=1, iterations=1,
    )


def test_internal_slice_boundary_matches_box_faces_walls():
    """internal_slice at position_frac 0.0/1.0 is the same plane as
    box_faces' boundary walls - the interior-slice feature is meant to
    be a strict generalization of the boundary-only walls, not a
    different code path that could silently disagree at the edges."""
    result = _make_result_with_distinct_values()
    faces = box_faces(result, top_layer_index=0)

    south = internal_slice(result, "ew", 0.0)
    north = internal_slice(result, "ew", 1.0)
    west = internal_slice(result, "ns", 0.0)
    east = internal_slice(result, "ns", 1.0)

    assert np.allclose(np.nan_to_num(np.array(south["value"])), np.nan_to_num(np.array(faces["south"]["value"])))
    assert np.allclose(np.nan_to_num(np.array(north["value"])), np.nan_to_num(np.array(faces["north"]["value"])))
    assert np.allclose(np.nan_to_num(np.array(west["value"])), np.nan_to_num(np.array(faces["west"]["value"])))
    assert np.allclose(np.nan_to_num(np.array(east["value"])), np.nan_to_num(np.array(faces["east"]["value"])))


def test_internal_slice_interior_position_matches_expected_row():
    """A mid-mesh position_frac should extract exactly the mesh row/col
    at that fractional position - the whole point of internal_slice over
    box_faces' boundary-only walls."""
    result = _make_result_with_distinct_values()
    mesh = result.mesh
    ny, nx, nz = result.susceptibility.shape

    face = internal_slice(result, "ew", 0.5)
    row_idx = int(round(0.5 * (ny - 1)))
    expected = result.susceptibility[row_idx, :, :].T
    expected = np.where(mesh.active[row_idx, :, :].T, expected, np.nan)
    assert np.allclose(np.nan_to_num(np.array(face["value"])), np.nan_to_num(expected))
    assert np.isclose(face["position_m"], mesh.y_centers[row_idx])

    face_ns = internal_slice(result, "ns", 0.25)
    col_idx = int(round(0.25 * (nx - 1)))
    expected_ns = result.susceptibility[:, col_idx, :].T
    expected_ns = np.where(mesh.active[:, col_idx, :].T, expected_ns, np.nan)
    assert np.allclose(np.nan_to_num(np.array(face_ns["value"])), np.nan_to_num(expected_ns))
    assert np.isclose(face_ns["position_m"], mesh.x_centers[col_idx])


def test_path_slice_3d_matches_vertical_section_values():
    """path_slice_3d (the interactive 3D-embedded arbitrary-direction
    plane) must carry the exact same susceptibility values as
    vertical_section (the distance-vs-depth PNG version) for the same
    path - they share the same underlying nearest-cell sampling, just
    packaged differently (real x/y/z vs. along-path distance)."""
    result = _make_result_with_distinct_values()
    mesh = result.mesh
    path_x = np.linspace(mesh.x_centers.min(), mesh.x_centers.max(), 9)
    path_y = np.linspace(mesh.y_centers.min(), mesh.y_centers.max(), 9)

    section, _distance, _ground_elev = vertical_section(result, path_x, path_y)
    face = path_slice_3d(result, path_x, path_y)

    got_value = np.array(face["value"])
    assert got_value.shape == section.shape
    assert np.allclose(np.nan_to_num(got_value), np.nan_to_num(section))

    x_grid = np.array(face["x"])
    y_grid = np.array(face["y"])
    z_grid = np.array(face["z"])
    assert np.allclose(x_grid[0], path_x)
    assert np.allclose(y_grid[0], path_y)
    assert np.allclose(z_grid[:, 0], mesh.z_centers)


def test_azimuth_line_matches_ew_ns_at_mesh_center():
    """azimuth_deg=90 (east-west trending line) through the mesh center
    should sample only values that internal_slice's "ew" orientation at
    position_frac=0.5 (the mesh's own center row) also sees; azimuth_deg=0
    (north-south trending line) should agree with "ns" at position_frac=0.5
    the same way - two different code paths computing the same physical
    plane must agree, even though azimuth_line's own sample spacing
    doesn't line up exactly with the mesh's discrete columns."""
    result = _make_result_with_distinct_values()
    mesh = result.mesh

    def finite_value_set(face):
        return set(np.round([v for v in np.array(face["value"]).ravel() if np.isfinite(v)], 6))

    path_x, path_y = azimuth_line(mesh, 90.0, sample_spacing_m=20.0)
    az_values = finite_value_set(path_slice_3d(result, path_x, path_y))
    internal_values = finite_value_set(internal_slice(result, "ew", 0.5))
    assert az_values, "azimuth=90 line produced no finite samples"
    assert az_values.issubset(internal_values)

    path_x2, path_y2 = azimuth_line(mesh, 0.0, sample_spacing_m=20.0)
    az_values2 = finite_value_set(path_slice_3d(result, path_x2, path_y2))
    internal_values2 = finite_value_set(internal_slice(result, "ns", 0.5))
    assert az_values2, "azimuth=0 line produced no finite samples"
    assert az_values2.issubset(internal_values2)


def test_azimuth_line_crosses_full_mesh_diagonal():
    """The line's endpoints should reach at least the mesh's own half-
    diagonal distance from the center regardless of bearing - otherwise
    a slice at some odd angle could fall short of the mesh edge."""
    mesh = _make_mesh()
    half_diag = 0.5 * np.hypot(mesh.x_centers.max() - mesh.x_centers.min(), mesh.y_centers.max() - mesh.y_centers.min())
    for az in (0.0, 45.0, 90.0, 135.0):
        path_x, path_y = azimuth_line(mesh, az, sample_spacing_m=10.0)
        cx, cy = float(mesh.x_centers.mean()), float(mesh.y_centers.mean())
        dist_from_center = np.hypot(path_x - cx, path_y - cy)
        assert dist_from_center.max() >= half_diag - 1e-6


if __name__ == "__main__":
    test_inversion_recovers_synthetic_block_location()
    test_auto_regularization_targets_assumed_noise()
    test_stronger_target_noise_yields_larger_regularization()
    test_resolution_diagnostics_decreases_with_depth()
    test_resolution_diagnostics_flags_poorly_resolved_layers()
    test_build_mesh_growth_factor_one_is_uniform()
    test_build_mesh_graded_thickness_increases_with_depth()
    test_box_faces_top_follows_sloped_terrain()
    test_internal_slice_boundary_matches_box_faces_walls()
    test_internal_slice_interior_position_matches_expected_row()
    test_path_slice_3d_matches_vertical_section_values()
    test_azimuth_line_matches_ew_ns_at_mesh_center()
    test_azimuth_line_crosses_full_mesh_diagonal()
    print("ALL CHECKS PASSED")
