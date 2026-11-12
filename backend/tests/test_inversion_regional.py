"""The background taken out of the data before a 3D inversion.

The inversion's model is susceptibility between 0 and a ceiling; it has
no unknown for the level the data sit at, and explains one with
susceptibility that is not there (GaSaDo 2026-11: anomaly at -100 nT; left
in, the model averages 0.16 SI with 5.4 % of its cells at the ceiling,
taken out 0.09 SI and 1.7 %, at much the same misfit).
magcore.trend.separate_regional removes a level or a robust low-order
surface first; InversionParams.regional_removal chooses which.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.store import store
from magcore.trend import remove_regional_trend, separate_regional

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def field(background, seed=0):
    """A background surface with compact anomalies standing on it, most of
    them positive, and no data in one corner."""
    rng = np.random.default_rng(seed)
    e, n = np.arange(0.0, 2000.0, 25.0), np.arange(0.0, 1500.0, 25.0)
    x, y = np.meshgrid(e, n)
    regional = background(x, y)
    anomaly = np.zeros_like(x)
    for cx, cy, amp, width in ((500, 400, 600, 90), (1400, 1000, 450, 120), (1500, 300, 300, 70), (700, 1100, -200, 80)):
        anomaly += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width ** 2))
    values = regional + anomaly + rng.normal(0, 2.0, x.shape)
    values[(x > 1700) & (y > 1200)] = np.nan
    return values, e, n, regional, anomaly


def test_none_changes_nothing_and_level_is_the_median():
    values, e, n, *_ = field(lambda x, y: -90.0 + 0 * x)
    residual, regional, info = separate_regional(values, e, n, "none")
    finite = np.isfinite(values)
    assert np.array_equal(residual[finite], values[finite]) and np.isnan(residual[~finite]).all()
    assert info["method"] == "none" and info["level_nt"] == 0.0
    assert info["data_median_after_nt"] == pytest.approx(np.median(values[finite]))

    residual, regional, info = separate_regional(values, e, n, "level")
    assert info["level_nt"] == pytest.approx(np.median(values[finite]))
    assert info["level_nt"] == pytest.approx(-90.0, abs=3.0)          # the anomalies do not move a median far
    assert info["data_median_after_nt"] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(residual[~finite]).all() and np.isnan(regional[~finite]).all()


@pytest.mark.parametrize("method, background", [
    ("plane", lambda x, y: -60.0 + 0.05 * x - 0.03 * y),
    ("quadratic", lambda x, y: -60.0 + 0.05 * x - 0.03 * y + 4e-5 * (x - 1000) ** 2 - 3e-5 * (y - 700) ** 2),
])
def test_a_surface_follows_the_background_not_the_anomalies_on_it(method, background):
    values, e, n, true_regional, anomaly = field(background)
    finite = np.isfinite(values)
    residual, regional, info = separate_regional(values, e, n, method)
    error = float(np.sqrt(np.mean((regional[finite] - true_regional[finite]) ** 2)))
    assert error < 2.0, error
    # what is left is the anomalies, at full height
    assert residual[finite].max() == pytest.approx(anomaly[finite].max(), abs=10.0)
    assert info["min_nt"] < info["level_nt"] < info["max_nt"]

    # a plain least-squares surface of the same order is pulled toward them
    order = {"plane": 1, "quadratic": 2}[method]
    _res, lsq = remove_regional_trend(values, e, n, order=order)
    lsq_error = float(np.sqrt(np.mean((lsq[finite] - true_regional[finite]) ** 2)))
    assert lsq_error > 4 * error


def test_unknown_method_is_refused():
    values, e, n, *_ = field(lambda x, y: 0 * x)
    with pytest.raises(ValueError):
        separate_regional(values, e, n, "cubic")


# ---------------------------------------------------------------------------
# Through the inversion.
# ---------------------------------------------------------------------------
PARAMS = {"obs_cell_size_m": 40.0, "depth_extent_m": 120.0, "n_layers": 5, "n_irls_iterations": 3}


@pytest.fixture(scope="module")
def project():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{pid}/process", json={"line_params": {}, "diurnal_params": {}, "heading_correction": {}})
    assert r.status_code == 200, r.text
    return client, pid, store.get(pid)


def run(client, pid, **params):
    r = client.post(f"/api/projects/{pid}/inversion", json={**PARAMS, **params})
    assert r.status_code == 200, r.text
    result = store.get(pid).inversion_result
    return r.json(), result.susceptibility.copy(), result.observed_nt.copy()


def test_the_level_is_removed_by_default_and_reported(project):
    client, pid, proj = project
    summary, _chi, observed = run(client, pid)
    info = summary["regional_removal"]
    assert info["method"] == "level"
    assert info["data_median_after_nt"] == pytest.approx(0.0, abs=1e-6)
    assert info["level_nt"] == pytest.approx(info["data_median_before_nt"])
    assert np.median(observed) == pytest.approx(0.0, abs=1e-6)        # what was inverted is the residual
    assert summary["regional_warning"] is None
    # ... and it is the residual that is kept as the inversion's input map
    grid = proj.inversion_obs_grid.values
    assert np.median(grid[np.isfinite(grid)]) == pytest.approx(0.0, abs=1e-6)


def test_a_shift_of_the_whole_survey_no_longer_changes_the_model(project):
    client, pid, proj = project
    original = proj.processed["anomaly"].to_numpy().copy()
    try:
        summary_a, chi_a, _o = run(client, pid, regional_removal="level")
        shift = -2.0 * summary_a["regional_removal"]["data_std_nt"]         # well off zero, by the data's own measure
        proj.processed["anomaly"] = original + shift
        summary_b, chi_b, _o = run(client, pid, regional_removal="level")
        assert summary_b["regional_removal"]["level_nt"] == pytest.approx(summary_a["regional_removal"]["level_nt"] + shift, abs=1e-6)
        assert np.allclose(chi_a, chi_b, atol=1e-9)

        # left in, the same shift is explained with susceptibility - and said so
        summary_c, chi_c, _o = run(client, pid, regional_removal="none")
        assert summary_c["regional_removal"]["method"] == "none"
        assert summary_c["regional_removal"]["data_median_after_nt"] == pytest.approx(summary_b["regional_removal"]["level_nt"], abs=1e-6)
        assert summary_c["regional_warning"] and f"{summary_b['regional_removal']['level_nt']:+.0f} nT" in summary_c["regional_warning"]
        assert not np.allclose(chi_a, chi_c, atol=1e-4)

        # and nothing is said when the data sit near zero anyway
        proj.processed["anomaly"] = original - summary_a["regional_removal"]["level_nt"]
        summary_d, _chi, _o = run(client, pid, regional_removal="none")
        assert summary_d["regional_warning"] is None
    finally:
        proj.processed["anomaly"] = original


def test_a_surface_can_be_chosen_and_an_unknown_one_cannot(project):
    client, pid, _proj = project
    summary, _chi, _observed = run(client, pid, regional_removal="plane")
    info = summary["regional_removal"]
    assert info["method"] == "plane" and info["min_nt"] <= info["level_nt"] <= info["max_nt"]
    r = client.post(f"/api/projects/{pid}/inversion", json={**PARAMS, "regional_removal": "cubic"})
    assert r.status_code == 422
