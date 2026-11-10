"""Products that assume a level observation plane say so when the survey
is not on one and height normalisation has not been applied
(project_grids.py::_altitude_warning)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.store import store

LAT0, LON0 = 36.5, 127.8


def _csv(relief_m):
    """Six north-south lines 50 m apart over a field with an anomaly; the
    sensor altitude follows a ridge `relief_m` high across the block."""
    rng = np.random.default_rng(7)
    rows = ["Counter,Date,Time,Latitude,Longitude,Mag, MagValid,Altitude"]
    t, counter = 3 * 3600.0, 0
    for k, east in enumerate(np.arange(0.0, 300.0, 50.0)):
        north = np.arange(0.0, 800.0, 1.0)
        if k % 2:
            north = north[::-1]
        for y in north:
            lat = LAT0 + y / 110950.0
            lon = LON0 + east / (111320.0 * np.cos(np.radians(LAT0)))
            b = 50000.0 + 60.0 * np.exp(-(((east - 120.0) / 70.0) ** 2 + ((y - 400.0) / 90.0) ** 2)) + rng.normal(0, 0.3)
            altitude = 150.0 + relief_m * np.exp(-((y - 400.0) / 200.0) ** 2) + rng.normal(0, 0.3)
            hh, mm, ss = int(t // 3600), int(t % 3600 // 60), t % 60
            rows.append(f"{counter},2026-10-28,{hh:02d}:{mm:02d}:{ss:06.3f},{lat:.8f},{lon:.8f},{b:.3f},1,{altitude:.2f}")
            t += 0.2
            counter += 1
        t += 20.0
    return ("\n".join(rows) + "\n").encode()


def _project(relief_m):
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    assert client.post(f"/api/projects/{pid}/upload/drone", files=[("files", ("s.csv", _csv(relief_m), "text/csv"))]).status_code == 200
    r = client.post(f"/api/projects/{pid}/process", json={"diurnal_params": {"mode": "assume_constant"}})
    assert r.status_code == 200, r.text
    return client, pid


@pytest.fixture(scope="module")
def draped():
    return _project(relief_m=40.0)


def test_a_draped_survey_is_told_before_its_derived_grids_are_trusted(draped):
    client, pid = draped
    for transform in ("rtp", "1vd", "as", "upward_continuation"):
        body = client.post(f"/api/projects/{pid}/transform", json={"transform": transform, "cell_size_m": 10.0}).json()
        assert "비행 고도 정규화를 적용하지 않았습니다" in body["altitude_warning"], transform
        assert "수평면" in body["altitude_warning"]
    # the spread it quotes is the survey's own: a 40 m ridge, 5th to 95th percentile
    z = store.get(pid).processed["altitude_ellipsoidal_m"].to_numpy()
    spread = np.percentile(z[np.isfinite(z)], 95) - np.percentile(z[np.isfinite(z)], 5)
    assert f"{spread:.0f} m" in body["altitude_warning"] and 25.0 < spread < 45.0
    # a product that does not read across cells as a plane operator carries no such warning
    assert client.post(f"/api/projects/{pid}/transform", json={"transform": "detrend", "cell_size_m": 10.0}).json()["altitude_warning"] is None


def test_euler_says_it_too_and_the_inversion_does_not_need_to(draped):
    client, pid = draped
    euler = client.post(f"/api/projects/{pid}/euler-deconvolution", json={"cell_size_m": 10.0, "structural_index": 1.0, "window_size_m": 100.0}).json()
    assert "비행 고도 정규화" in euler["altitude_warning"]
    # the inversion places each observation at its own altitude: nothing to warn about there
    inversion = client.post(f"/api/projects/{pid}/inversion", json={"regularization_strength": 1.0}).json()
    assert "altitude_warning" not in inversion


def test_applying_the_normalisation_silences_it_and_undoing_brings_it_back(draped):
    client, pid = draped
    r = client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "apply"})
    assert r.status_code == 200, r.text
    assert client.post(f"/api/projects/{pid}/transform", json={"transform": "rtp", "cell_size_m": 10.0}).json()["altitude_warning"] is None
    assert client.post(f"/api/projects/{pid}/height-normalization/apply", json={"mode": "reset"}).status_code == 200
    assert client.post(f"/api/projects/{pid}/transform", json={"transform": "rtp", "cell_size_m": 10.0}).json()["altitude_warning"] is not None


def test_a_level_flight_is_left_alone():
    client, pid = _project(relief_m=0.0)                       # 0.3 m of GPS scatter on a level plane
    assert client.post(f"/api/projects/{pid}/transform", json={"transform": "rtp", "cell_size_m": 10.0}).json()["altitude_warning"] is None
    euler = client.post(f"/api/projects/{pid}/euler-deconvolution", json={"cell_size_m": 10.0, "structural_index": 1.0, "window_size_m": 100.0}).json()
    assert euler["altitude_warning"] is None
