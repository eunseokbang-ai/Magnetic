"""Confidence carried from the grid into what is computed from it
(magcore/uncertainty.py): derived grids, Euler solutions, and the map of
how well repeated ground repeats."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from magcore.transforms import derivative_easting, reduction_to_pole, vertical_derivative
from magcore.uncertainty import derived_confidence, footprint, propagate, window_mean

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


# ---------------------------------------------------------------- the rule
def _radius(kernel, share=0.9):
    """Distance from the centre within which `share` of the footprint lies, in cells."""
    ny, nx = kernel.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - ny // 2, xx - nx // 2).ravel()
    order = np.argsort(r)
    return float(r[order][np.searchsorted(np.cumsum(kernel.ravel()[order]), share)])


@pytest.mark.parametrize("shape", [(64, 80), (65, 81), (64, 81)])
def test_a_transform_that_reads_only_its_own_cell_changes_nothing(shape):
    rng = np.random.default_rng(0)
    confidence = rng.uniform(0, 1, shape)
    kernel = footprint(lambda v: 3.0 * v, shape)                 # a scaling: footprint is the cell itself
    assert kernel.sum() == pytest.approx(1.0) and kernel[shape[0] // 2, shape[1] // 2] == pytest.approx(1.0)
    np.testing.assert_allclose(propagate(confidence, kernel), confidence, atol=1e-9)   # and nothing shifts, odd or even


def test_footprints_are_as_wide_as_the_operators_are():
    shape, cell = (128, 128), 10.0
    derivative = footprint(lambda v: derivative_easting(v, cell), shape)
    vertical = footprint(lambda v: vertical_derivative(v, cell, order=1), shape)
    pole = footprint(lambda v: reduction_to_pole(v, cell, 53.0, -8.8), shape)
    # measured: 90 % of the squared response lies within 5 cells (horizontal derivative), 1 cell (vertical
    # derivative) and 1.4 cells (reduction to the pole) - all local, none reading the whole grid
    assert _radius(derivative) == pytest.approx(5.0, abs=1.5)
    assert _radius(vertical) <= 1.5 and _radius(pole) <= 2.5
    # the horizontal derivative reads along its own axis only; the reduction to the pole has the longer tail of the two compact ones
    rows = derivative.sum(axis=1)
    assert rows[64] == pytest.approx(1.0, abs=1e-6)
    assert _radius(pole, 0.99) > _radius(vertical, 0.99)


def test_a_gap_stays_a_gap_and_spreads_as_far_as_the_operator_reads():
    shape, cell = (120, 120), 10.0
    confidence = np.ones(shape)
    confidence[:, 55:65] = 0.1                                    # a north-south strip between two lines, poorly supported
    confidence[:5, :] = np.nan                                    # outside the survey
    across = derived_confidence(confidence, [lambda v: derivative_easting(v, cell)])        # reads across the strip
    local = derived_confidence(confidence, [lambda v: vertical_derivative(v, cell, order=1)])
    for c in (across, local):
        assert np.isnan(c[:5, :]).all() and np.nanmin(c) >= 0.0 and np.nanmax(c) <= 1.0
        assert np.nanmean(c[20:100, 58:62]) < 0.5                 # the gap is still low
        assert np.nanmean(c[20:100, 5:20]) > 0.97                 # far from it nothing changed
    # the supported cells next to the gap lose more under the operator that reads further across it
    beside = (slice(20, 100), slice(65, 68))
    assert np.nanmean(across[beside]) < np.nanmean(local[beside]) - 0.02
    # and the survey's own edge costs support: the operator's footprint hangs over cells that hold nothing
    # (3 % for the first covered row here: the vertical derivative reads little beyond its own cell)
    assert np.nanmean(local[5:6, 5:50]) < np.nanmean(local[40:60, 5:50]) - 0.01


def test_a_product_of_several_derivatives_is_no_better_than_its_worst():
    shape, cell = (96, 96), 10.0
    rng = np.random.default_rng(1)
    confidence = np.clip(rng.uniform(0.2, 1.0, shape), 0, 1)
    dx = [lambda v: derivative_easting(v, cell)]
    dz = [lambda v: vertical_derivative(v, cell, order=1)]
    both = derived_confidence(confidence, dx + dz)
    assert np.all(both <= derived_confidence(confidence, dx) + 1e-12) and np.all(both <= derived_confidence(confidence, dz) + 1e-12)
    np.testing.assert_allclose(derived_confidence(confidence, []), confidence)          # nothing read across cells


def test_window_mean_is_the_mean_over_the_window():
    confidence = np.zeros((40, 40))
    confidence[20, 20] = 1.0
    out = window_mean(confidence, 5)
    assert out[20, 20] == pytest.approx(1 / 25) and out[18, 22] == pytest.approx(1 / 25) and out[20, 23] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------ through the API
@pytest.fixture(scope="module")
def processed():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files=[("files", ("d.csv", f.read(), "text/csv"))])
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files=[("files", ("b.csv", f.read(), "text/csv"))])
    assert client.post(f"/api/projects/{pid}/process", json={}).status_code == 200
    return client, pid


def test_a_derived_grid_has_its_own_confidence_layer(processed):
    client, pid = processed
    base = client.post(f"/api/projects/{pid}/grid/confidence", json={"cell_size_m": 10.0, "method": "spline"}).json()
    out = {}
    for transform in ("1vd", "rtp", "as", "detrend"):
        r = client.post(f"/api/projects/{pid}/transform/confidence", json={"transform": transform, "cell_size_m": 10.0})
        assert r.status_code == 200, r.text
        out[transform] = body = r.json()
        assert body["image_data_url"].startswith("data:image/png;base64,") and body["bounds"] == base["bounds"]
        assert 0.0 <= body["stats"]["min"] <= body["stats"]["max"] <= 1.0 and body["transform"] == transform
        assert body["base_low_fraction"] == pytest.approx(base["low_fraction"], abs=1e-4)
    # a product that does not read across cells keeps the base layer
    assert out["detrend"]["stats"]["mean"] == pytest.approx(base["stats"]["mean"], abs=1e-6)
    # a derivative product loses support at the survey's edge, where its footprint reaches past the data
    assert out["1vd"]["stats"]["mean"] < base["stats"]["mean"]
    # three derivatives are no better than one of them
    assert out["as"]["stats"]["mean"] <= out["1vd"]["stats"]["mean"] + 1e-6


def test_euler_solutions_carry_the_support_of_the_window_they_came_from(processed):
    client, pid = processed
    r = client.post(f"/api/projects/{pid}/euler-deconvolution", json={"cell_size_m": 10.0, "structural_index": 1.0, "window_size_m": 100.0})
    assert r.status_code == 200, r.text
    body = r.json()
    scores = [s["confidence"] for s in body["solutions"]]
    assert body["n_solutions"] == len(scores) > 0 and all(c is not None and 0.0 <= c <= 1.0 for c in scores)
    assert body["low_confidence_threshold"] == 0.5 and body["n_low_confidence"] == sum(1 for c in scores if c < 0.5)
    assert max(scores) > min(scores)                               # it tells solutions apart


LAT0, LON0 = 36.5, 127.8


def _repeat_csv(noise):
    """Six north-south lines 50 m apart, the first one flown a second time."""
    rng = np.random.default_rng(5)
    rows = ["Counter,Date,Time,Latitude,Longitude,Mag, MagValid,Altitude"]
    t, counter = 3 * 3600.0, 0
    for k, east in enumerate([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 0.0]):
        north = np.arange(0.0, 800.0, 0.5)
        if k % 2:
            north = north[::-1]
        field = 50000.0 + 40.0 * np.exp(-(((east - 120.0) / 60.0) ** 2 + ((north - 400.0) / 80.0) ** 2))
        for y, b in zip(north, field + rng.normal(0, noise, len(north))):
            lat = LAT0 + y / 110950.0
            lon = LON0 + east / (111320.0 * np.cos(np.radians(LAT0)))
            hh, mm, ss = int(t // 3600), int(t % 3600 // 60), t % 60
            rows.append(f"{counter},2026-10-28,{hh:02d}:{mm:02d}:{ss:06.3f},{lat:.8f},{lon:.8f},{b:.3f},1,120.0")
            t += 0.1
            counter += 1
        t += 20.0
    return ("\n".join(rows) + "\n").encode()


def _repeat_map(noise):
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    assert client.post(f"/api/projects/{pid}/upload/drone", files=[("files", ("r.csv", _repeat_csv(noise), "text/csv"))]).status_code == 200
    r = client.post(f"/api/projects/{pid}/process", json={"diurnal_params": {"mode": "assume_constant"}, "statistical_leveling": {"enabled": False},
                                                          "despike_params": {"enabled": False}})
    assert r.status_code == 200, r.text
    return client.post(f"/api/projects/{pid}/repeat-passes/map", json={})


def test_the_repeat_map_measures_the_noise_that_was_put_in():
    quiet, loud = _repeat_map(1.0), _repeat_map(4.0)
    assert quiet.status_code == 200, quiet.text
    q, l = quiet.json(), loud.json()
    assert q["image_data_url"].startswith("data:image/png;base64,") and q["n_pairs"] == 1 and q["total_overlap_m"] > 600.0
    # four times the noise, four times the figure
    assert l["repeatability_nt"] == pytest.approx(4.0 * q["repeatability_nt"], rel=0.25)
    # the figure is the noise of what is compared - 5 m bin medians of ten 1 nT readings - not of a single reading
    assert 0.2 < q["repeatability_nt"] < 1.0
    assert q["difference_rms_nt"] == pytest.approx(q["repeatability_nt"] * np.sqrt(2.0), abs=0.002)


def test_no_repeated_ground_no_map(processed):
    client, pid = processed
    r = client.post(f"/api/projects/{pid}/repeat-passes/map", json={})
    if r.status_code == 200:                                       # the sample survey may hold a repeat; then the map must be sane
        assert r.json()["n_pairs"] >= 1
    else:
        assert r.status_code == 400 and "두 번 비행한 자료가 없어" in r.json()["detail"]
