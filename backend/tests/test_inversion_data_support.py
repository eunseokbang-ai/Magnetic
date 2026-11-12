"""The inversion model outside the data.

The mesh is the survey's bounding rectangle and a survey seldom fills it.
The columns with no data above them used to be solved for and shown like
the rest: on GaSaDo (2026-11, 61% of the columns under data) a quarter of
the model above 0.05 SI stood out there, drawn as bodies along the
rectangle's edges where nothing was flown. Now a band of
INVERSION_PADDING_CELLS around the data stays in the solve as padding -
dropping it altogether rebuilds outside sources inside the survey's rim -
and only the cells under data are shown, counted and summarised.
"""
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.project_inversion import INVERSION_PADDING_CELLS, _data_support_summary
from app.store import store
from magcore.inversion import (
    InversionError,
    InversionResult,
    box_faces,
    build_mesh,
    build_sensitivity_matrix,
    horizontal_slice,
    internal_slice,
    interpretable_cells,
    interpretable_susceptibility,
    invert,
    padding_share,
    vertical_section,
)

GASADO = pathlib.Path(__file__).resolve().parents[2] / "GaSaDo"
CELL = 20.0


def survey_mask(ny=15, nx=15):
    """Data over the western two thirds of the rectangle only."""
    support = np.zeros((ny, nx), dtype=bool)
    support[:, :9] = True
    return support


def make_mesh(support, padding_cells):
    ny, nx = support.shape
    x, y = np.arange(nx) * CELL, np.arange(ny) * CELL
    ground = np.full((ny, nx), 50.0)
    return build_mesh(x, y, ground, cell_size_m=CELL, depth_extent_m=100.0, n_layers=5, data_support=support, padding_cells=padding_cells)


def result_with(mesh, susceptibility):
    return InversionResult(mesh=mesh, susceptibility=susceptibility, predicted_nt=np.zeros(3), observed_nt=np.zeros(3),
                           rms_misfit_nt=0.0, n_active_cells=int(mesh.active.sum()), n_obs=3, iterations=1)


def test_mesh_reaches_the_padding_and_no_farther():
    support = survey_mask()
    mesh = make_mesh(support, 2)
    in_model = mesh.active.any(axis=2)
    assert in_model[:, :11].all()                       # under data, and the two columns of padding
    assert not in_model[:, 11:].any()                   # beyond reach: not solved for
    assert np.array_equal(mesh.data_support, support)
    shown = interpretable_cells(mesh)
    assert shown[:, :9].all() and not shown[:, 9:].any()

    # no padding given: every column is active, as before, and the support is still recorded
    whole = make_mesh(support, None)
    assert whole.active.all() and np.array_equal(whole.data_support, support)
    # no support given: nothing is hidden
    x, y = np.arange(15) * CELL, np.arange(15) * CELL
    plain = build_mesh(x, y, np.full((15, 15), 50.0), cell_size_m=CELL, depth_extent_m=100.0, n_layers=5)
    assert plain.data_support is None and interpretable_cells(plain) is plain.active


def test_mesh_rejects_a_support_that_does_not_fit():
    x, y = np.arange(15) * CELL, np.arange(15) * CELL
    ground = np.full((15, 15), 50.0)
    with pytest.raises(InversionError):
        build_mesh(x, y, ground, cell_size_m=CELL, depth_extent_m=100.0, n_layers=5, data_support=np.ones((3, 3), bool))
    with pytest.raises(InversionError):
        build_mesh(x, y, ground, cell_size_m=CELL, depth_extent_m=100.0, n_layers=5, data_support=np.zeros((15, 15), bool))


def test_views_show_only_what_is_under_data():
    support = survey_mask()
    mesh = make_mesh(support, 2)
    chi = np.where(mesh.active, 0.01, 0.0)
    chi[7, 10, :] = 0.9                                 # a strong body in the padding
    chi[7, 4, 1] = 0.3                                  # and a real one under data
    result = result_with(mesh, chi)

    shown = interpretable_susceptibility(result)
    assert np.isnan(shown[:, 9:, :]).all() and np.nanmax(shown) == pytest.approx(0.3)
    assert interpretable_susceptibility(result, fill=0.0)[7, 10, :].max() == 0.0

    for layer in range(5):
        s = horizontal_slice(result, layer)
        assert np.isnan(s[:, 9:]).all() and np.isfinite(s[:, :9]).all()
    assert horizontal_slice(result, 1)[7, 4] == pytest.approx(0.3)

    # an east-west section through both bodies: the one in the padding is not on it
    section, distance, _ = vertical_section(result, mesh.x_centers, np.full(15, mesh.y_centers[7]))
    assert np.isnan(section[:, 9:]).all() and np.nanmax(section) == pytest.approx(0.3)
    face = internal_slice(result, "ew", 0.5)
    assert np.nanmax(np.asarray(face["value"], dtype=float)) == pytest.approx(0.3)

    faces = box_faces(result)
    assert faces["vmax"] == pytest.approx(0.3)          # the colour range is not stretched by the padding
    top = np.asarray(faces["top"]["value"], dtype=float)
    assert np.isnan(top[:, 9:]).all() and np.isfinite(top[:, :9]).all()
    assert np.isnan(np.asarray(faces["east"]["value"], dtype=float)).all()     # the east wall is all outside the data

    share = padding_share(result)
    assert 0.3 < share < 0.6                            # the body in the padding is a good part of this model
    summary = _data_support_summary(result, 2)
    assert summary["n_columns_with_data"] == int(support.sum()) and summary["n_padding_columns"] == 15 * 2
    assert summary["padding_model_share"] == pytest.approx(share) and summary["note"]


def test_padding_takes_the_outside_source_and_no_padding_puts_it_inside():
    """A block outside the data, close to its edge. With padding the solve
    puts it out there and the model under data stays quiet; with none it
    is rebuilt inside the survey's rim - the reason the padding is solved
    for at all."""
    support = survey_mask()
    full = make_mesh(support, None)
    obs_x, obs_y = np.meshgrid(full.x_centers[:9], full.y_centers)
    obs_x, obs_y = obs_x.ravel(), obs_y.ravel()
    obs_z = np.full(obs_x.shape, 65.0)
    G, rows, cols, layers = build_sensitivity_matrix(obs_x, obs_y, obs_z, full, 55.0, -8.0, 50000.0)
    truth = np.where((np.abs(rows - 7) <= 1) & (cols == 10) & (layers >= 1) & (layers <= 3), 0.08, 0.0)
    data = G @ truth
    assert np.abs(data).max() > 20                      # it is plainly in the data at the edge

    worst = {}
    for padding in (0, 2):
        mesh = make_mesh(support, padding)
        keep = mesh.active[rows, cols, layers]
        res = invert(np.ascontiguousarray(G[:, keep]), data, mesh, rows[keep], cols[keep], layers[keep], regularization_strength=0.3)
        worst[padding] = (float(np.nanmax(interpretable_susceptibility(res))), res.rms_misfit_nt, padding_share(res))
    assert worst[2][0] < 0.5 * worst[0][0]              # the false body under data shrinks
    assert worst[2][1] < worst[0][1]                    # and the data are fitted better
    assert worst[2][2] > 0.5 and worst[0][2] == 0.0     # because the source went where it is


def _project_with(result):
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    proj = store.get(pid)
    proj.inversion_result = result
    proj.utm_epsg = 32652
    proj.inclination_deg, proj.declination_deg = 51.0, -8.0
    return client, pid, proj


def test_volume_and_exports_keep_the_padding_out_of_the_result():
    support = survey_mask()
    mesh = make_mesh(support, 2)
    chi = np.where(mesh.active, 0.01, 0.0)
    chi[7, 10, :] = 0.9
    chi[7, 4, 1] = 0.3
    client, pid, proj = _project_with(result_with(mesh, chi))

    volume = proj.get_inversion_volume()
    assert max(volume["value"]) == pytest.approx(0.3, abs=0.02)        # the iso-surface cannot find the padding body
    assert volume["stats"]["max"] == pytest.approx(0.3)

    csv = pd.read_csv(io.BytesIO(proj.export_inversion_csv()))
    assert set(csv["under_data"]) == {0, 1}
    assert csv.loc[csv["under_data"] == 0, "susceptibility_si"].max() == pytest.approx(0.9)    # still in the file, marked
    assert csv.loc[csv["under_data"] == 1, "susceptibility_si"].max() == pytest.approx(0.3)

    # a saved model keeps its support; one saved before it was recorded takes it from its observation grid
    saved = proj.export_inversion_npz()
    summary = proj.import_inversion_npz(saved)
    assert np.array_equal(proj.inversion_result.mesh.data_support, support)
    assert summary["data_support"]["n_columns_with_data"] == int(support.sum()) and summary["susceptibility_stats"]["max"] == pytest.approx(0.3)

    with np.load(io.BytesIO(saved)) as npz:
        old = {k: npz[k] for k in npz.files if k != "data_support"}
    old.update(obs_easting=mesh.x_centers, obs_northing=mesh.y_centers, obs_values=np.where(support, 1.0, np.nan),
               obs_cell_size_m=np.array(CELL), obs_region=np.array([0.0, 280.0, 0.0, 280.0]))
    buf = io.BytesIO(); np.savez_compressed(buf, **old)
    proj.import_inversion_npz(buf.getvalue())
    assert np.array_equal(proj.inversion_result.mesh.data_support, support)
    assert np.nanmax(horizontal_slice(proj.inversion_result, 0)) < 0.5

    # and one with neither is shown whole, as it always was
    bare = {k: v for k, v in old.items() if not k.startswith("obs_")}
    buf = io.BytesIO(); np.savez_compressed(buf, **bare)
    summary = proj.import_inversion_npz(buf.getvalue())
    assert proj.inversion_result.mesh.data_support is None and summary["data_support"] is None
    assert np.nanmax(horizontal_slice(proj.inversion_result, 0)) == pytest.approx(0.9)


@pytest.mark.skipif(not (GASADO / "compensated").is_dir(), reason="GaSaDo survey not present")
def test_gasado_shows_nothing_where_nothing_was_flown():
    files = sorted((GASADO / "compensated").glob("*.csv"), key=lambda f: int(f.name.split("ACQU")[1].split("-")[0]))[:24]
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    assert client.post(f"/api/projects/{pid}/upload/drone", files=[("files", (f.name, f.read_bytes(), "text/csv")) for f in files]).status_code == 200
    assert client.post(f"/api/projects/{pid}/process", json={"diurnal_params": {"mode": "assume_constant"}}).status_code == 200
    r = client.post(f"/api/projects/{pid}/inversion", json={})
    assert r.status_code == 200, r.text[:300]
    summary = r.json()
    proj = store.get(pid)
    result = proj.inversion_result
    mesh = result.mesh
    support = mesh.data_support
    ds = summary["data_support"]
    assert ds["padding_cells"] == INVERSION_PADDING_CELLS
    assert 0.3 < ds["coverage_frac"] < 0.95             # the survey does not fill its rectangle
    assert ds["n_padding_columns"] > 0 and 0.0 < ds["padding_model_share"] < 0.6
    assert np.array_equal(support, np.isfinite(proj.inversion_obs_grid.values))

    # the model does hold susceptibility out there, and none of it is shown
    outside = result.susceptibility[~support]
    assert outside.max() > 0.05
    for layer in range(mesh.active.shape[2]):
        assert np.isnan(horizontal_slice(result, layer)[~support]).all()
    volume = proj.get_inversion_volume(threshold=0.05)
    value = np.asarray(volume["value"]).reshape(volume["shape"])
    x0, y0 = mesh.x_centers.mean(), mesh.y_centers.mean()
    fine_y = np.linspace(mesh.y_centers[0], mesh.y_centers[-1], value.shape[0])
    fine_x = np.linspace(mesh.x_centers[0], mesh.x_centers[-1], value.shape[1])
    # every drawn voxel lies within a cell of a column that has data (the
    # smoothing of the iso-surface carries a strong cell's value that far)
    from scipy.ndimage import distance_transform_edt, map_coordinates
    cells_from_data = distance_transform_edt(~support)
    rr = (fine_y - mesh.y_centers[0]) / mesh.cell_size_m
    cc = (fine_x - mesh.x_centers[0]) / mesh.cell_size_m
    R, C = np.meshgrid(rr, cc, indexing="ij")
    drawn = (value >= 0.05).any(axis=2)
    assert drawn.any() and map_coordinates(cells_from_data, [R, C], order=1)[drawn].max() < 1.0
    # beyond the padding nothing is solved for
    far = cells_from_data > INVERSION_PADDING_CELLS + 1e-9
    assert far.any() and not mesh.active[far].any()
