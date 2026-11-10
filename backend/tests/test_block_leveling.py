"""Block levelling: one block of files brought onto another's level from
their shared ground (magcore/block_leveling.py), through the project and
the API."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.store import store
from magcore.block_leveling import BlockLevelingError, estimate_block_shift


# ------------------------------------------------------------ the estimator
def _field(x, y):
    """A regional slope plus three anomalies - enough structure that two
    blocks sampling different spots of a cell really do read differently."""
    return (0.02 * x + 0.01 * y
            + 120.0 * np.exp(-(((x - 900.0) / 150.0) ** 2 + ((y - 500.0) / 150.0) ** 2))
            - 80.0 * np.exp(-(((x - 1100.0) / 100.0) ** 2 + ((y - 900.0) / 120.0) ** 2))
            + 60.0 * np.exp(-(((x - 1000.0) / 80.0) ** 2 + ((y - 200.0) / 80.0) ** 2)))


def _block(x0, x1, spacing, rng, phase=0.0, noise=0.5, y_max=1200.0):
    """North-south lines every `spacing` m between x0 and x1, a reading every 2 m."""
    xs, ys = [], []
    for lx in np.arange(x0 + phase, x1, spacing):
        y = np.arange(0.0, y_max, 2.0)
        xs.append(np.full_like(y, lx) + rng.normal(0, 1.0, len(y)))
        ys.append(y)
    x, y = np.concatenate(xs), np.concatenate(ys)
    return x, y, _field(x, y) + rng.normal(0, noise, len(x))


def test_a_constant_shift_is_recovered_within_its_own_uncertainty():
    rng = np.random.default_rng(1)
    ax, ay, av = _block(0, 1200, 50, rng)
    bx, by, bv = _block(800, 2000, 50, rng, phase=23.0)          # overlaps 800..1200, lines not coincident
    shift = estimate_block_shift(ax, ay, av, bx, by, bv + 17.0, cell_size_m=50.0)
    assert shift.model == "constant"                              # "auto": no significant gradient, so no plane
    assert 0.0 < shift.constant_se_nt < 3.0
    assert abs(shift.constant_nt - 17.0) < max(3 * shift.constant_se_nt, 0.5)
    assert shift.n_cells >= 100 and shift.overlap_area_m2 == shift.n_cells * 2500.0
    assert shift.gradient_significant is False and shift.warnings == []
    np.testing.assert_allclose(shift.value_at([900.0, 1500.0], [0.0, 700.0]), shift.constant_nt)


def test_a_drifting_difference_is_found_as_a_plane_and_auto_picks_it():
    rng = np.random.default_rng(2)
    ax, ay, av = _block(0, 1200, 50, rng)
    bx, by, bv = _block(600, 2000, 50, rng, phase=23.0)
    drift = 10.0 + 30.0 * (by - 600.0) / 1000.0                   # +30 nT per km northward
    shift = estimate_block_shift(ax, ay, av, bx, by, bv + drift, cell_size_m=50.0)
    assert shift.model == "plane" and shift.gradient_significant is True
    assert abs(shift.gradient_y_nt_per_km - 30.0) < max(3 * shift.gradient_y_se, 2.0)
    assert abs(shift.gradient_x_nt_per_km) < max(3 * shift.gradient_x_se, 3.0)
    # subtracting the model levels the target over the shared ground...
    left = (bv + drift) - shift.value_at(bx, by) - bv
    overlap = bx < 1200.0
    assert abs(np.median(left[overlap])) < 1.5
    # ...and further out it is as good as the model says it is: the stated uncertainty grows with the
    # distance the plane is carried, and what is actually left stays inside three of it
    sigma = shift.uncertainty_at(bx, by)
    assert sigma[bx > 1900.0].mean() > 2 * sigma[overlap].mean()
    assert np.all(np.abs(left) < 3 * sigma + 1.5)
    # forcing a constant on the same data says the difference is not one
    constant = estimate_block_shift(ax, ay, av, bx, by, bv + drift, cell_size_m=50.0, model="constant")
    assert constant.model == "constant" and any("평면" in w for w in constant.warnings)


def test_uncertainty_grows_when_the_overlap_shrinks():
    rng = np.random.default_rng(3)
    ax, ay, av = _block(0, 1200, 50, rng)
    wide = estimate_block_shift(ax, ay, av, *_shifted(_block(600, 2000, 50, rng, phase=23.0), 9.0), cell_size_m=50.0, model="constant")
    narrow = estimate_block_shift(ax, ay, av, *_shifted(_block(1080, 2000, 50, rng, phase=23.0, y_max=500.0), 9.0), cell_size_m=50.0, model="constant")
    assert narrow.n_cells < wide.n_cells / 5
    assert narrow.constant_se_nt > wide.constant_se_nt


def _shifted(block, offset):
    x, y, v = block
    return x, y, v + offset


def test_what_cannot_be_measured_is_refused_or_flagged():
    rng = np.random.default_rng(4)
    ax, ay, av = _block(0, 1000, 50, rng)
    bx, by, bv = _block(1500, 2500, 50, rng)
    with pytest.raises(BlockLevelingError, match="함께 지나간 구역이 너무 작습니다"):
        estimate_block_shift(ax, ay, av, bx, by, bv, cell_size_m=50.0)
    # a strip one cell wide cannot hold a plane
    sx, sy, sv = _block(960, 1040, 50, rng, phase=10.0)
    with pytest.raises(BlockLevelingError, match="평면을 정할 수 없습니다"):
        estimate_block_shift(ax, ay, av, sx, sy, sv, cell_size_m=50.0, model="plane")
    # blocks flown at different heights: the difference is not a level shift, and the result says so
    cx, cy, cv = _block(600, 1600, 50, rng, phase=23.0)
    shift = estimate_block_shift(ax, ay, av, cx, cy, cv, cell_size_m=50.0, ref_alt=np.full(len(ax), 80.0), tgt_alt=np.full(len(cx), 95.0))
    assert shift.altitude_difference_m == pytest.approx(15.0) and any("고도" in w for w in shift.warnings)


# ------------------------------------------------------------ through the API
LAT0, LON0 = 36.5, 127.8


def _csv(x0, x1, offset, t0, phase=0.0):
    """One block as a MagArrow-style CSV: north-south lines 50 m apart, 10 Hz at 5 m/s."""
    rng = np.random.default_rng(int(x0))
    rows = ["Counter,Date,Time,Latitude,Longitude,Mag, MagValid,Altitude"]
    t, counter = float(t0), 0
    for k, lx in enumerate(np.arange(x0 + phase, x1, 50.0)):
        ys = np.arange(0.0, 1000.0, 0.5)
        if k % 2:
            ys = ys[::-1]
        for y in ys:
            lat = LAT0 + y / 110540.0
            lon = LON0 + lx / (111320.0 * np.cos(np.radians(LAT0)))
            mag = 50000.0 + _field(lx, y) + offset + rng.normal(0, 0.3)
            hh, mm, ss = int(t // 3600), int(t % 3600 // 60), t % 60
            rows.append(f"{counter},2026-10-28,{hh:02d}:{mm:02d}:{ss:06.3f},{lat:.8f},{lon:.8f},{mag:.3f},1,120.0")
            t += 0.1
            counter += 1
        t += 20.0
    return ("\n".join(rows) + "\n").encode()


@pytest.fixture(scope="module")
def two_blocks():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    files = [("files", ("block_west.csv", _csv(0, 800, 0.0, 3 * 3600), "text/csv")),
             ("files", ("block_east.csv", _csv(500, 1400, 22.0, 5 * 3600, phase=23.0), "text/csv"))]
    assert client.post(f"/api/projects/{pid}/upload/drone", files=files).status_code == 200
    r = client.post(f"/api/projects/{pid}/process", json={"diurnal_params": {"mode": "assume_constant"}, "statistical_leveling": {"enabled": False}})
    assert r.status_code == 200, r.text
    return client, pid


def test_files_are_listed_by_name_with_their_extent(two_blocks):
    client, pid = two_blocks
    body = client.get(f"/api/projects/{pid}/block-leveling/files").json()
    assert [(f["index"], f["name"]) for f in body["files"]] == [(0, "block_west.csv"), (1, "block_east.csv")]
    assert all(f["n_on_lines"] > 1000 and f["shift_index"] is None for f in body["files"]) and body["shifts"] == []
    assert body["files"][0]["center"][1] < body["files"][1]["center"][1]            # west is west


def test_estimate_apply_and_reset_through_the_api(two_blocks):
    client, pid = two_blocks
    project = store.get(pid)
    east = project.processed["source_file_index"].to_numpy() == 1
    before = project.processed["anomaly"].to_numpy().copy()

    r = client.post(f"/api/projects/{pid}/block-leveling/estimate", json={"reference_files": [0], "target_files": [1]})
    assert r.status_code == 200, r.text
    shift = r.json()["shift"]
    assert shift["model"] == "constant" and abs(shift["constant_nt"] - 22.0) < max(3 * shift["constant_se_nt"], 1.0)
    assert shift["cell_size_m"] == pytest.approx(50.0, abs=2.0)                     # the line spacing, by default
    np.testing.assert_array_equal(project.processed["anomaly"].to_numpy(), before)   # measuring changes nothing

    r = client.post(f"/api/projects/{pid}/block-leveling/apply", json={"reference_files": [0], "target_files": [1]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["process_summary"]["block_leveling"]["applied"] is True
    (stored,) = body["shifts"]
    assert stored["files"] == [1] and stored["source"] == "overlap" and stored["reference_files"] == [0]
    after = project.processed["anomaly"].to_numpy()
    np.testing.assert_allclose(after[east], before[east] - stored["constant_nt"], atol=1e-6)   # the target block moved...
    np.testing.assert_array_equal(after[~east], before[~east])                               # ...and only it
    assert client.get(f"/api/projects/{pid}/block-leveling/files").json()["files"][1]["shift_index"] == 0

    # estimating again re-measures the same shift (the stored one is set aside), it does not report ~0 and stack
    again = client.post(f"/api/projects/{pid}/block-leveling/estimate", json={"reference_files": [0], "target_files": [1]}).json()["shift"]
    assert again["constant_nt"] == pytest.approx(stored["constant_nt"], abs=1e-6)

    # the shift travels with the project
    data = project.save_project_bundle()
    r = client.post(f"/api/projects/{pid}/load", files={"file": ("p.magproj", data, "application/zip")})
    assert r.status_code == 200, r.text
    assert r.json()["process_summary"]["block_leveling"]["shifts"][0]["constant_nt"] == stored["constant_nt"]
    np.testing.assert_allclose(project.processed["anomaly"].to_numpy()[east], before[east] - stored["constant_nt"], atol=1e-6)

    r = client.post(f"/api/projects/{pid}/block-leveling/apply", json={"mode": "reset", "target_files": [1]})
    assert r.json()["shifts"] == []
    np.testing.assert_allclose(project.processed["anomaly"].to_numpy(), before, atol=1e-9)


def test_a_known_sensor_offset_is_applied_directly(two_blocks):
    client, pid = two_blocks
    project = store.get(pid)
    east = project.processed["source_file_index"].to_numpy() == 1
    before = project.processed["anomaly"].to_numpy().copy()
    r = client.post(f"/api/projects/{pid}/block-leveling/apply", json={"mode": "manual", "target_files": [1], "manual_constant_nt": 21.4})
    assert r.status_code == 200, r.text
    (stored,) = r.json()["shifts"]
    assert (stored["source"], stored["model"], stored["constant_nt"], stored["constant_se_nt"]) == ("manual", "constant", 21.4, 0.0)
    np.testing.assert_allclose(project.processed["anomaly"].to_numpy()[east], before[east] - 21.4, atol=1e-9)
    assert client.post(f"/api/projects/{pid}/block-leveling/apply", json={"mode": "reset"}).json()["shifts"] == []


def test_requests_that_make_no_sense_are_refused_with_a_reason(two_blocks):
    client, pid = two_blocks
    for body, expected in (
        ({"reference_files": [0], "target_files": []}, "각각 하나 이상"),
        ({"reference_files": [0, 1], "target_files": [1]}, "함께 들어"),
        ({"mode": "manual", "target_files": [1]}, "오프셋"),
    ):
        endpoint = "apply" if "mode" in body else "estimate"
        r = client.post(f"/api/projects/{pid}/block-leveling/{endpoint}", json=body)
        assert r.status_code == 400 and expected in r.json()["detail"], (body, r.text)
