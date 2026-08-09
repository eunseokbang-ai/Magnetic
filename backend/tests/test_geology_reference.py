"""Tests for the 3D inversion's geology reference model: user-digitized
polygon blocks (see GeologyUnitInput) regularize the inversion toward a
susceptibility value assigned per block instead of toward zero (Li &
Oldenburg's own reference-model mechanism) - both the invert() math
itself and the geology-unit CRUD + run_inversion API wiring."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.processing.inversion import build_mesh, build_sensitivity_matrix, invert

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def _make_mesh_and_sensitivity():
    x_centers = np.arange(-100.0, 101.0, 20.0)
    y_centers = np.arange(-100.0, 101.0, 20.0)
    ground_elevation = np.full((len(y_centers), len(x_centers)), 50.0)
    mesh = build_mesh(x_centers, y_centers, ground_elevation, cell_size_m=20.0, depth_extent_m=100.0, n_layers=5)

    obs_x, obs_y = np.meshgrid(np.arange(-90.0, 91.0, 15.0), np.arange(-90.0, 91.0, 15.0))
    obs_x, obs_y = obs_x.ravel(), obs_y.ravel()
    obs_z = np.full(obs_x.shape, 60.0)

    G, rows, cols, layers = build_sensitivity_matrix(obs_x, obs_y, obs_z, mesh, 55.0, -8.0, 50000.0)
    return mesh, G, rows, cols, layers


def test_invert_m_ref_none_matches_explicit_zeros_array():
    """m_ref=None (the default, used by every pre-existing caller) must
    reproduce exactly the same result as an explicit all-zeros reference
    model - the "no prior information" case shouldn't secretly behave
    differently depending on which of the two equivalent spellings is used."""
    mesh, G, rows, cols, layers = _make_mesh_and_sensitivity()
    true_m = np.zeros(G.shape[1])
    true_m[G.shape[1] // 2] = 0.04
    data_nt = G @ true_m

    result_none = invert(G, data_nt, mesh, rows, cols, layers, regularization_strength=0.5, n_irls_iterations=3)
    result_zeros = invert(
        G, data_nt, mesh, rows, cols, layers, regularization_strength=0.5, n_irls_iterations=3, m_ref=np.zeros(G.shape[1])
    )
    assert np.allclose(result_none.susceptibility, result_zeros.susceptibility)


def test_invert_heavy_regularization_approaches_reference_model():
    """As regularization_strength grows, the alpha term in the normal
    equations dominates over the (fixed) data-misfit term, so the
    solution should converge toward m_ref itself - the basic asymptotic
    check that supplying a reference model actually pulls results toward
    it (Li & Oldenburg's mechanism for injecting known geology), not just
    a change that happens to leave results unchanged."""
    mesh, G, rows, cols, layers = _make_mesh_and_sensitivity()
    n_active = G.shape[1]
    data_nt = np.zeros(G.shape[0])
    m_ref = np.full(n_active, 0.02)

    result_loose = invert(G, data_nt, mesh, rows, cols, layers, regularization_strength=0.01, n_irls_iterations=1, m_ref=m_ref)
    result_tight = invert(G, data_nt, mesh, rows, cols, layers, regularization_strength=1e4, n_irls_iterations=1, m_ref=m_ref)

    m_loose = result_loose.susceptibility[rows, cols, layers]
    m_tight = result_tight.susceptibility[rows, cols, layers]

    dist_loose = float(np.mean(np.abs(m_loose - m_ref)))
    dist_tight = float(np.mean(np.abs(m_tight - m_ref)))
    print(f"mean|m-m_ref| loose={dist_loose:.5f} tight={dist_tight:.5f}")
    assert dist_tight < dist_loose
    assert dist_tight < 0.005


def _make_processed_project():
    client = TestClient(app)
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    payload = {"line_params": {}, "diurnal_params": {}, "heading_correction": {}}
    r = client.post(f"/api/projects/{project_id}/process", json=payload)
    assert r.status_code == 200, r.text
    return client, project_id


def test_geology_unit_crud_via_api():
    client, project_id = _make_processed_project()

    r = client.get(f"/api/projects/{project_id}/geology/units")
    assert r.status_code == 200, r.text
    assert r.json()["units"] == []

    r = client.post(f"/api/projects/{project_id}/grid", json={"value": "anomaly", "cell_size_m": 20.0})
    assert r.status_code == 200, r.text
    (south, west), (north, east) = r.json()["bounds"]
    mid_lat, mid_lon = (south + north) / 2, (west + east) / 2
    path = [[south, west], [south, mid_lon], [mid_lat, mid_lon], [mid_lat, west]]

    r = client.post(
        f"/api/projects/{project_id}/geology/units",
        json={"name": "화강암", "susceptibility_si": 0.02, "path": path},
    )
    assert r.status_code == 200, r.text
    unit_id = r.json()["unit"]["id"]
    assert len(r.json()["units"]) == 1

    r = client.patch(f"/api/projects/{project_id}/geology/units/{unit_id}", json={"susceptibility_si": 0.05})
    assert r.status_code == 200, r.text
    assert r.json()["unit"]["susceptibility_si"] == 0.05
    assert r.json()["unit"]["name"] == "화강암"

    r = client.delete(f"/api/projects/{project_id}/geology/units/{unit_id}")
    assert r.status_code == 200, r.text
    assert r.json()["units"] == []

    r = client.delete(f"/api/projects/{project_id}/geology/units/{unit_id}")
    assert r.status_code == 400


def test_run_inversion_uses_geology_reference():
    client, project_id = _make_processed_project()

    r = client.post(f"/api/projects/{project_id}/grid", json={"value": "anomaly", "cell_size_m": 20.0})
    assert r.status_code == 200, r.text
    (south, west), (north, east) = r.json()["bounds"]
    mid_lon = (west + east) / 2
    path = [[south, west], [south, mid_lon], [north, mid_lon], [north, west]]

    r = client.post(
        f"/api/projects/{project_id}/geology/units",
        json={"name": "현무암", "susceptibility_si": 0.03, "path": path},
    )
    assert r.status_code == 200, r.text

    r = client.post(f"/api/projects/{project_id}/inversion", json={"assumed_agl_m": 50.0})
    assert r.status_code == 200, r.text
    summary = r.json()
    assert summary["geology_reference_used"] is True
    assert summary["n_geology_units"] == 1
    assert 0.0 < summary["geology_reference_coverage"] <= 1.0

    r2 = client.post(f"/api/projects/{project_id}/inversion", json={"assumed_agl_m": 50.0, "use_geology_reference": False})
    assert r2.status_code == 200, r2.text
    summary2 = r2.json()
    assert summary2["geology_reference_used"] is False
    assert summary2["n_geology_units"] == 1


if __name__ == "__main__":
    test_invert_m_ref_none_matches_explicit_zeros_array()
    test_invert_heavy_regularization_approaches_reference_model()
    test_geology_unit_crud_via_api()
    test_run_inversion_uses_geology_reference()
    print("ALL CHECKS PASSED")
