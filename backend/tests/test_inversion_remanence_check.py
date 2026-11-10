"""The 3D inversion assumes induced magnetization. Its summary says where
the survey's strongest anomalies contradict that (project_inversion.py::
_remanence_check, built on magcore/magnetization.py)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pyproj import Geod

from app.main import app
from magcore.igrf import mean_inclination_declination
from magcore.source_removal import _field_direction, _kernel

GEOD = Geod(ellps="WGS84")
LAT0, LON0, ALT = 36.5, 127.8, 150.0
DEPTH = 50.0                      # source below the sensor


def _field_dir():
    """The direction the project will work out for this place and day."""
    inc, dec = mean_inclination_declination(np.array([LAT0]), np.array([LON0]), np.array([ALT]), pd.Series([pd.Timestamp("2026-10-28 03:00:00")]))
    return inc, dec, _field_direction(inc, dec)


def _survey_csv(sources):
    """North-south lines 50 m apart over 900 x 900 m, a reading every 2 m.
    `sources` = [(east_m, north_m, moment_vector)], each DEPTH below the sensor."""
    rng = np.random.default_rng(0)
    _, _, f = _field_dir()
    rows = ["Counter,Date,Time,Latitude,Longitude,Mag, MagValid,Altitude"]
    t, counter = 3 * 3600.0, 0
    for k, east in enumerate(np.arange(0.0, 901.0, 50.0)):
        north = np.arange(0.0, 901.0, 2.0)
        if k % 2:
            north = north[::-1]
        x = np.full(len(north), east)
        anomaly = sum(_kernel(x, north, np.array([[sx, sy, -DEPTH]]), f) @ m for sx, sy, m in sources)
        anomaly = anomaly + 0.01 * x - 0.008 * north + rng.normal(0, 0.5, len(north))
        n = len(north)
        lon, lat, _ = GEOD.fwd(np.full(n, LON0), np.full(n, LAT0), np.full(n, 90.0), x)
        lon, lat, _ = GEOD.fwd(lon, lat, np.zeros(n), north)
        for la, lo, a in zip(lat, lon, anomaly):
            hh, mm, ss = int(t // 3600), int(t % 3600 // 60), t % 60
            rows.append(f"{counter},2026-10-28,{hh:02d}:{mm:02d}:{ss:06.3f},{la:.8f},{lo:.8f},{50000.0 + a:.3f},1,{ALT}")
            t += 0.4
            counter += 1
        t += 20.0
    return ("\n".join(rows) + "\n").encode()


def _invert(sources):
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    assert client.post(f"/api/projects/{pid}/upload/drone", files=[("files", ("s.csv", _survey_csv(sources), "text/csv"))]).status_code == 200
    r = client.post(f"/api/projects/{pid}/process", json={"diurnal_params": {"mode": "assume_constant"}, "statistical_leveling": {"enabled": False}})
    assert r.status_code == 200, r.text
    r = client.post(f"/api/projects/{pid}/inversion", json={"regularization_strength": 1.0})
    assert r.status_code == 200, r.text
    return client, pid, r.json()


def _where(east, north):
    lon, lat, _ = GEOD.fwd(LON0, LAT0, 90.0, east)
    lon, lat, _ = GEOD.fwd(lon, lat, 0.0, north)
    return lat, lon


def test_a_remanent_source_in_the_block_is_named_in_the_inversion_summary():
    _, _, f = _field_dir()
    sideways = np.cross(f, [0.0, 0.0, 1.0])
    sideways /= np.linalg.norm(sideways)                      # 90 degrees from the field: no induced source points there
    client, pid, summary = _invert([(250.0, 300.0, 4e5 * f), (650.0, 600.0, 4e5 * sideways)])

    check = summary["remanence_check"]
    assert check["available"] is True and check["n_checked"] >= 2 and check["n_remanent"] == 1
    assert "잔류자화" in check["warning"] and "유도자화만 가정" in check["warning"]
    (remanent,) = [s for s in check["sources"] if s["verdict"] == "remanent"]
    lat, lon = _where(650.0, 600.0)
    assert GEOD.inv(lon, lat, remanent["lon"], remanent["lat"])[2] < 60.0          # it is the planted one
    assert remanent["angle_from_induced_deg"] == pytest.approx(90.0, abs=20.0) and remanent["improvement"] > 0.5
    # the induced source is among those checked and is not accused
    lat, lon = _where(250.0, 300.0)
    induced = min(check["sources"], key=lambda s: GEOD.inv(lon, lat, s["lon"], s["lat"])[2])
    assert GEOD.inv(lon, lat, induced["lon"], induced["lat"])[2] < 60.0 and induced["verdict"] == "induced"

    # the warning travels: the report carries it, and so does a saved project
    report = client.get(f"/api/projects/{pid}/report").content.decode("utf-8")
    assert check["warning"] in report
    reopened = client.post(f"/api/projects/{pid}/open").json()
    assert reopened["inversion_summary"]["remanence_check"]["n_remanent"] == 1


def test_a_block_of_induced_sources_gets_no_warning():
    _, _, f = _field_dir()
    _client, _pid, summary = _invert([(250.0, 300.0, 4e5 * f), (650.0, 600.0, 2.5e5 * f)])
    check = summary["remanence_check"]
    assert check["available"] is True and check["n_remanent"] == 0 and check["warning"] is None
    assert sum(1 for s in check["sources"] if s["verdict"] == "induced") >= 2
