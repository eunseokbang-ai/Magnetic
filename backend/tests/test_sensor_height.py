"""Sensor height correction: the GPS height of a towed sensor, anchored to
the ground it stood on before takeoff and after landing.

magcore/sensor_height.py finds those moments, measures the GPS error at
each against a terrain model, fits how the error wanders in time and
predicts it for every sample (simple kriging), with the uncertainty that
is left. Checked here on flights built with a known wandering error, where
the truth is known at every sample; through the API on the sample survey;
and on the survey it was measured on (GaSaDo, skipped where the field
data is absent).
"""
import io
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from app.main import app
from app.store import store
from magcore.sensor_height import (
    DEFAULT_SILL_M2,
    DEFAULT_TAU_MIN,
    ErrorModel,
    GroundCheck,
    SensorHeightCorrection,
    build_correction,
    find_ground_checks,
    fit_error_model,
    leave_out_validation,
    predict_error,
)

DT = 0.1


def wandering_error(rng, times_s, mean, sd, tau_min):
    """An Ornstein-Uhlenbeck path sampled at times_s (seconds, increasing)."""
    out = np.empty(len(times_s))
    out[0] = rng.normal(0, sd)
    for k in range(1, len(times_s)):
        a = np.exp(-(times_s[k] - times_s[k - 1]) / (tau_min * 60.0))
        out[k] = a * out[k - 1] + rng.normal(0, sd * np.sqrt(1 - a * a))
    return mean + out


def survey(n_flights=24, mean=8.0, sd=6.0, tau_min=25.0, seed=0, flight_min=20.0, turnaround_min=2.0, pad_elevation=12.0):
    """Flights from one pad: 6 s on the ground, a climb to 60 m above it,
    a level cruise, a descent, 6 s on the ground. The GPS height is the
    true one plus one slowly wandering error that runs on through the
    turnarounds, as a receiver that stays switched on does."""
    rng = np.random.default_rng(seed)
    t_all, true_all, flight_all, speed_all, ground_all = [], [], [], [], []
    t0 = 0.0
    for f in range(n_flights):
        n = int(flight_min * 60 / DT)
        t = t0 + np.arange(n) * DT
        rel = np.arange(n) * DT
        h = np.clip(np.minimum(rel - 6.0, rel[-1] - 6.0 - rel) * 2.0, 0.0, 60.0)       # 2 m/s up and down
        speed = np.where(h > 1.0, 5.0, 0.0)
        t_all.append(t); true_all.append(pad_elevation + h); flight_all.append(np.full(n, f)); speed_all.append(speed)
        ground_all.append(np.where(h < 1.0, pad_elevation, pad_elevation - 3.0))       # the pad, and lower ground away from it
        t0 = t[-1] + turnaround_min * 60.0
    t = np.concatenate(t_all)
    coarse = np.arange(t[0], t[-1] + 10.0, 10.0)
    error = np.interp(t, coarse, wandering_error(rng, coarse, mean, sd, tau_min))
    true = np.concatenate(true_all)
    return dict(t=t, true=true, gps=true + error, error=error, flight=np.concatenate(flight_all), speed=np.concatenate(speed_all), ground=np.concatenate(ground_all))


def test_the_checks_are_the_error_at_takeoff_and_landing():
    s = survey(n_flights=6)
    checks = find_ground_checks(s["t"], s["gps"], s["speed"], s["flight"], s["ground"])
    assert [(c.flight, c.kind) for c in checks] == [(f, k) for f in range(6) for k in ("takeoff", "landing")]
    for c in checks:
        truth = s["error"][np.argmin(np.abs(s["t"] - c.time_s))]
        assert c.used and c.error_m == pytest.approx(truth, abs=0.3)
        assert c.ground_m == 12.0 and 4.0 < c.seconds < 8.0


def test_a_sensor_resting_above_the_ground_is_allowed_for():
    s = survey(n_flights=2)
    plain = find_ground_checks(s["t"], s["gps"], s["speed"], s["flight"], s["ground"])
    raised = find_ground_checks(s["t"], s["gps"], s["speed"], s["flight"], s["ground"], resting_offset_m=0.4)
    assert [c.error_m - r.error_m for c, r in zip(plain, raised)] == pytest.approx([0.4] * 4)


def test_checks_that_cannot_be_trusted_are_listed_but_not_used():
    s = survey(n_flights=8)
    ground = s["ground"].copy()
    ground[(s["flight"] == 1)] = np.nan                       # the terrain model does not reach this pad
    ground[(s["flight"] == 2) & (ground == 12.0)] = 0.0       # ... and calls this one the sea
    gps = s["gps"].copy()
    last = np.flatnonzero(s["flight"] == 3)
    cut = last[: int(len(last) * 0.5)]                        # flight 3's record stops in mid-air, hovering
    keep = np.ones(len(gps), dtype=bool); keep[last[int(len(last) * 0.5):]] = False
    speed = s["speed"].copy(); speed[cut[-80:]] = 0.0
    args = [a[keep] for a in (s["t"], gps, speed, s["flight"], ground)]
    checks = find_ground_checks(*args)
    by = {(c.flight, c.kind): c for c in checks}
    assert not by[(1, "takeoff")].used and by[(1, "takeoff")].excluded_reason == "no_terrain"
    assert not by[(2, "landing")].used and by[(2, "landing")].excluded_reason == "sea_level"
    assert not by[(3, "landing")].used and by[(3, "landing")].excluded_reason == "outlier"
    assert by[(3, "takeoff")].used and by[(0, "landing")].used
    # the sea-level rule can be turned off (a pad that really is at 0 m)
    kept = {(c.flight, c.kind): c for c in find_ground_checks(*args, exclude_sea_level=False)}
    assert kept[(2, "landing")].excluded_reason in (None, "outlier")


def test_a_record_that_starts_in_motion_gives_no_check():
    s = survey(n_flights=1)
    moving = s["speed"] > 0
    checks = find_ground_checks(s["t"][moving], s["gps"][moving], s["speed"][moving], s["flight"][moving], s["ground"][moving])
    assert checks == []


def test_prediction_honours_the_checks_and_forgets_them_with_distance():
    model = ErrorModel(mean_m=9.0, sill_m2=81.0, tau_min=30.0, fitted=True, n_checks=2)
    checks = [GroundCheck(0, "takeoff", 0.0, 0, 0, 20.0, 5.0), GroundCheck(0, "landing", 1200.0, 0, 0, 2.0, 5.0)]
    t = np.array([0.0, 600.0, 1200.0, 1200.0 + 3600.0 * 8])
    mean, sd = predict_error(t, checks, model)
    assert mean[0] == pytest.approx(20.0, abs=0.2) and mean[2] == pytest.approx(2.0, abs=0.2)
    assert sd[0] < 1.0 and sd[2] < 1.0
    assert 2.0 < mean[1] < 20.0 and 3.0 < sd[1] < 9.0          # between them: in between, and less sure
    assert mean[3] == pytest.approx(9.0, abs=0.05) and sd[3] == pytest.approx(9.0, abs=0.05)   # hours later: the mean, and no better
    # no checks at all: the mean and the full spread
    mean, sd = predict_error(t, [], model)
    assert np.allclose(mean, 9.0) and np.allclose(sd, 9.0)


def test_the_error_model_is_recovered_from_enough_checks_and_defaulted_from_few():
    s = survey(n_flights=60, mean=8.0, sd=6.0, tau_min=25.0, seed=3)
    model = fit_error_model(find_ground_checks(s["t"], s["gps"], s["speed"], s["flight"], s["ground"]))
    assert model.fitted and model.n_checks == 120
    assert model.mean_m == pytest.approx(8.0, abs=2.0)
    assert np.sqrt(model.sill_m2) == pytest.approx(6.0, rel=0.35)
    assert 12.0 < model.tau_min < 50.0

    few = survey(n_flights=4, seed=4)
    checks = find_ground_checks(few["t"], few["gps"], few["speed"], few["flight"], few["ground"])
    model = fit_error_model(checks)
    assert not model.fitted and (model.sill_m2, model.tau_min) == (DEFAULT_SILL_M2, DEFAULT_TAU_MIN)
    assert model.mean_m == pytest.approx(np.median([c.error_m for c in checks]))
    assert fit_error_model([]).mean_m == 0.0


def test_the_correction_brings_the_height_back_and_says_how_far():
    s = survey(n_flights=40, mean=9.0, sd=7.0, tau_min=25.0, seed=5)
    corr = build_correction(s["t"], s["gps"], s["speed"], s["flight"], s["ground"])
    predicted, sigma = corr.predict(s["t"])
    residual = (s["gps"] - predicted) - s["true"]
    before = float(np.sqrt(np.mean((s["gps"] - s["true"]) ** 2)))
    after = float(np.sqrt(np.mean(residual ** 2)))
    assert before > 10.0 and after < 0.5 * before
    # nearly exact at the ends of a flight, worst in its middle - and the stated uncertainty says so
    rel = np.concatenate([np.linspace(0, 1, int((s["flight"] == f).sum())) for f in range(40)])
    ends, middle = (rel < 0.05) | (rel > 0.95), (rel > 0.4) & (rel < 0.6)
    assert np.sqrt(np.mean(residual[ends] ** 2)) < 2.0 < np.sqrt(np.mean(residual[middle] ** 2))
    assert np.median(sigma[ends]) < 2.0 < np.median(sigma[middle])
    # ... honestly: the residual is about as large as the stated sigma, in the middle and overall
    assert 0.6 < np.sqrt(np.mean((residual[middle] / sigma[middle]) ** 2)) < 1.6
    assert 0.6 < np.sqrt(np.mean((residual / sigma) ** 2)) < 1.6

    report = leave_out_validation(corr.checks, corr.model, 5.0)
    assert report["rms_kriged_m"] < report["rms_mean_only_m"] < report["rms_uncorrected_m"]


def test_a_saved_correction_predicts_what_it_predicted():
    s = survey(n_flights=5, seed=6)
    corr = build_correction(s["t"], s["gps"], s["speed"], s["flight"], s["ground"])
    again = SensorHeightCorrection.from_dict(corr.to_dict())
    a, b = corr.predict(s["t"][::50]), again.predict(s["t"][::50])
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    import json
    json.dumps(corr.to_dict())                                # goes into the project file


# ---------------------------------------------------------------------------
# Through the project, on the sample survey given a flight profile that
# starts and ends on the ground.
# ---------------------------------------------------------------------------
DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"
PAD_M = 40.0


def flat_terrain(lon, lat, elevation=PAD_M) -> bytes:
    west, east, south, north = lon.min() - 0.01, lon.max() + 0.01, lat.min() - 0.01, lat.max() + 0.01
    data = np.full((50, 50), elevation, dtype="float32")
    buffer = io.BytesIO()
    with rasterio.open(buffer, "w", driver="GTiff", height=50, width=50, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(west, north, (east - west) / 50, (north - south) / 50), nodata=-9999.0) as dst:
        dst.write(data, 1)
    return buffer.getvalue()


@pytest.fixture()
def project():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{pid}/process", json={"line_params": {}, "diurnal_params": {}, "heading_correction": {}})
    assert r.status_code == 200, r.text
    proj = store.get(pid)
    # one flight: on the ground for the first and last 5 s, 50 m up in between, the GPS reading 12 m high then 4 m high
    base = proj.processed_base
    n = len(base)
    t = (base["timestamp"] - base["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    standing = (t < 5.0) | (t > t[-1] - 5.0)
    gps_error = np.interp(t, [0, t[-1]], [12.0, 4.0])
    base["altitude_msl_m"] = PAD_M + np.where(standing, 0.0, 50.0) + gps_error
    base["altitude_ellipsoidal_m"] = base["altitude_msl_m"] + 25.0
    base["geoid_separation_m"] = 25.0
    base["speed_mps"] = np.where(standing, 0.0, base["speed_mps"].fillna(3.0).clip(lower=1.0))
    proj._rebuild_processed()
    return client, pid, proj, flat_terrain(base["lon"].to_numpy(), base["lat"].to_numpy()), gps_error


def test_applied_through_the_api_it_corrects_the_heights_and_nothing_else(project):
    client, pid, proj, terrain, gps_error = project
    before = proj.processed.copy()
    status = client.get(f"/api/projects/{pid}/sensor-height").json()
    assert status["available"] and not status["applied"] and status["geoid_separation_m"] == 25.0 and status["checks"] == []

    r = client.post(f"/api/projects/{pid}/sensor-height/apply", data={"mode": "apply"})
    assert r.status_code == 400 and "지형 자료" in r.json()["detail"]          # no terrain model anywhere

    r = client.post(f"/api/projects/{pid}/sensor-height/apply", data={"mode": "apply"}, files={"file": ("pad.tif", terrain, "image/tiff")})
    assert r.status_code == 200, r.text
    body = r.json()
    summary = body["sensor_height"]
    assert summary["applied"] and summary["n_checks_used"] == 2 and summary["terrain_name"] == "pad.tif"
    assert summary["model"]["fitted"] is False                                   # two checks: default spread and correlation time
    assert summary["check_error_m"] == {"median": 8.0, "min": pytest.approx(4.0, abs=0.1), "max": pytest.approx(12.0, abs=0.1)}
    checks = body["sensor_height_status"]["checks"]
    assert [(c["kind"], round(c["error_m"])) for c in checks] == [("takeoff", 12), ("landing", 4)]

    after = proj.processed
    assert np.allclose(after["altitude_gps_msl_m"], before["altitude_msl_m"])
    correction = (after["altitude_gps_msl_m"] - after["altitude_msl_m"]).to_numpy()
    assert correction[0] == pytest.approx(12.0, abs=0.3) and correction[-1] == pytest.approx(4.0, abs=0.3)
    assert np.allclose(after["altitude_gps_ellipsoidal_m"] - after["altitude_ellipsoidal_m"], correction)
    assert after["altitude_sigma_m"].iloc[0] < 1.5 < after["altitude_sigma_m"].iloc[len(after) // 2]
    for column in ("anomaly", "tmi", "lat", "lon", "line_id"):
        assert before[column].equals(after[column]), column
    assert proj._inversion_height_note(after)["corrected"] is True

    # saved with the project and back again, without the terrain model
    bundle = client.get(f"/api/projects/{pid}/save")
    assert bundle.status_code == 200
    pid2 = client.post("/api/projects").json()["project_id"]
    r = client.post(f"/api/projects/{pid2}/load", files={"file": ("p.magproj", bundle.content, "application/octet-stream")})
    assert r.status_code == 200, r.text
    restored = store.get(pid2)
    assert restored.sensor_height is not None and len(restored.sensor_height.checks) == 2
    assert "altitude_sigma_m" in restored.processed.columns and restored.process_summary()["sensor_height"]["applied"]

    # and off again: the receiver's own heights, exactly
    r = client.post(f"/api/projects/{pid}/sensor-height/apply", data={"mode": "reset"})
    assert r.status_code == 200 and r.json()["sensor_height"] == {"applied": False}
    back = proj.processed
    assert np.allclose(back["altitude_msl_m"], before["altitude_msl_m"]) and "altitude_sigma_m" not in back.columns
    assert "altitude_gps_msl_m" not in back.columns


def test_with_no_stretch_on_the_ground_it_says_so(project):
    client, pid, proj, terrain, _ = project
    proj.processed_base["speed_mps"] = 4.0                                       # never standing
    r = client.post(f"/api/projects/{pid}/sensor-height/apply", data={"mode": "apply"}, files={"file": ("pad.tif", terrain, "image/tiff")})
    assert r.status_code == 400 and "지상 검사가 없습니다" in r.json()["detail"]
    assert proj.sensor_height is None
    r = client.post(f"/api/projects/{pid}/sensor-height/apply", data={"mode": "sideways"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GaSaDo: 41 flights, the 0.5 m surface model.
# ---------------------------------------------------------------------------
GASADO = pathlib.Path(os.environ.get("MAGNETIC_GASADO_DIR") or pathlib.Path(__file__).resolve().parents[2] / "GaSaDo")
COMPENSATED, DSM = GASADO / "compensated", GASADO / "dsm" / "dsm(0.5m).tif"


@pytest.mark.skipif(len(list(COMPENSATED.glob("SRVY0-ACQU*-comp.csv"))) < 41 or not DSM.exists(), reason=f"GaSaDo flights or surface model not present ({GASADO})")
def test_gasado_heights_against_the_surface_model():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    number = lambda f: int(f.name.split("ACQU")[1].split("-")[0])  # noqa: E731
    files = sorted((f for f in COMPENSATED.glob("SRVY0-ACQU*-comp.csv") if 85 <= number(f) <= 125), key=number)
    r = client.post(f"/api/projects/{pid}/upload/drone", files=[("files", (f.name, f.read_bytes(), "text/csv")) for f in files])
    assert r.status_code == 200, r.text
    params = {"diurnal_params": {"mode": "assume_constant"}}
    assert client.post(f"/api/projects/{pid}/process", json=params).status_code == 200
    proj = store.get(pid)
    recorded = proj.processed["altitude_msl_m"].to_numpy().copy()

    r = client.post(f"/api/projects/{pid}/sensor-height/apply", data={"mode": "apply"}, files={"file": (DSM.name, DSM.read_bytes(), "image/tiff")})
    assert r.status_code == 200, r.text
    s = r.json()["sensor_height"]
    assert (s["n_checks"], s["n_checks_used"], s["n_checks_excluded"]) == (75, 74, {"sea_level": 1})
    assert s["flights_without_checks"] == []
    assert s["model"]["fitted"] and s["model"]["mean_m"] == pytest.approx(9.64, abs=0.3)
    assert np.sqrt(s["model"]["sill_m2"]) == pytest.approx(8.9, abs=1.0) and 18 < s["model"]["tau_min"] < 40
    assert s["check_error_m"]["min"] < -8 and s["check_error_m"]["max"] > 24
    assert s["correction_on_lines_m"]["median"] == pytest.approx(9.1, abs=0.5)
    assert s["sigma_on_lines_m"]["median"] == pytest.approx(5.1, abs=0.6)
    near, far = s["validation"]
    assert near["rms_uncorrected_m"] > 11 and near["rms_kriged_m"] < 4 and far["rms_kriged_m"] < far["rms_mean_only_m"]
    # the mission was flown with the sensor about 50 m above the ground
    assert s["height_above_terrain_m"]["before_median"] > 53 and 42 < s["height_above_terrain_m"]["after_median"] < 52

    # reprocessing the same flights keeps it; the receiver's own heights stay beside it
    assert client.post(f"/api/projects/{pid}/process", json=params).json()["sensor_height"]["applied"]
    table = store.get(pid).processed
    assert np.allclose(table["altitude_gps_msl_m"], recorded, equal_nan=True)
    assert np.nanmedian(recorded - table["altitude_msl_m"]) == pytest.approx(9.3, abs=1.0)
    # loading other flights drops it
    r = client.post(f"/api/projects/{pid}/upload/drone", files=[("files", (files[0].name, files[0].read_bytes(), "text/csv"))])
    assert r.status_code == 200 and store.get(pid).sensor_height is None
