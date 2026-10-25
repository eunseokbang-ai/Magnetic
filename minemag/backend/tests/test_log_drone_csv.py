"""The Log_Drone optically-pumped CSV (Gongju ADD site, 2026-10-19):
loader, path-based tracks, timestamps from TOW + file name, IGRF anomaly,
format detection, and the whole pipeline through the API."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.io_.loaders import detect_format, load_survey_files
from app.io_.log_drone_csv import load_log_drone_csv, parse_filename, segment_tracks
from app.main import app
from conftest import FIXTURES, R3_FIXTURE

LD_FIXTURE = FIXTURES / "Log_Drone_20261019_134937_rows6000-8600.csv"


def test_loader_schema_units_time_and_tracks():
    df = load_log_drone_csv(LD_FIXTURE.read_bytes(), LD_FIXTURE.name)
    assert sorted(df["sensor"].unique()) == [1, 2]
    assert len(df) == 2 * 1300
    for c in ("timestamp", "t_ms", "track", "sensor", "lat", "lon", "alt", "tfa", "bx", "by", "bz", "b_total", "roll", "pitch", "yaw"):
        assert c in df.columns
    # microtesla -> nanotesla, and the anomaly is the field minus IGRF (a few hundred nT at most)
    assert 50_000 < df["b_total"].median() < 51_000
    assert abs(df["tfa"].median()) < 60 and df["tfa"].abs().max() < 2000     # centred; sensor offset of +-26 nT remains
    # J1 - J2 offset of the pair, read on rows logged at the same instant
    piv = df.pivot_table(index="t_ms", columns="sensor", values="b_total")
    assert 30 < (piv[1] - piv[2]).median() < 70
    # same position for both sensors, 50 Hz, local time near the file name's 13:49:37 start
    assert (df[df.sensor == 1]["lat"].to_numpy() == df[df.sensor == 2]["lat"].to_numpy()).all()
    t1 = df[df.sensor == 1]["t_ms"].to_numpy()
    assert np.isclose(np.median(np.diff(t1)), 40.0, atol=2)             # every 2nd row of 50 Hz
    first = df["timestamp"].min()
    assert first.strftime("%Y-%m-%d") == "2026-10-19" and pd.Timestamp("2026-10-19 13:49:37") < first < pd.Timestamp("2026-10-19 14:05:00")
    # the path gives parallel lines
    tracks = df[df.track > 0]["track"].unique()
    assert len(tracks) >= 2 and (df.track > 0).mean() > 0.5
    info = df.attrs["file_info"]
    assert info["format"] == "log_drone_csv" and info["device"] == "Log_Drone OPM" and "UTC+9" in info["settings"]


def test_segment_tracks_on_a_synthetic_lawnmower():
    # three 40 m lines 2 m apart at 2 m/s, joined by 3 s turns, 50 Hz
    dt, pts = 0.02, []
    for k in range(3):
        y = np.linspace(0, 40, 1000) if k % 2 == 0 else np.linspace(40, 0, 1000)
        pts.append(np.column_stack([np.full(1000, 2.0 * k), y]))
        if k < 2:
            ang = np.linspace(0, np.pi, 150)
            cx, cy = 2.0 * k + 1.0, (40 if k % 2 == 0 else 0)
            pts.append(np.column_stack([cx - np.cos(ang), cy + (np.sin(ang) if k % 2 == 0 else -np.sin(ang))]))
    xy = np.vstack(pts)
    t = np.arange(len(xy)) * dt
    tr = segment_tracks(xy[:, 0], xy[:, 1], t)
    assert tr.max() == 3
    for k in range(3):
        on = tr[k * 1150:k * 1150 + 1000]
        assert (on == k + 1).mean() > 0.9                               # the line itself
    assert (tr[1000:1150] == -1).mean() > 0.6                           # the turn


def test_filename_and_format_detection():
    assert parse_filename("Log_Drone_20261019_134937.csv") == ("2026-10-19", 13 * 3600 + 49 * 60 + 37)
    assert parse_filename("something.csv") == (None, None)
    assert detect_format(LD_FIXTURE.read_bytes()) == "log_drone_csv"
    assert detect_format(R3_FIXTURE.read_bytes()) == "magdrone_r3"


def test_mixed_formats_in_one_project():
    out = load_survey_files([R3_FIXTURE.read_bytes(), LD_FIXTURE.read_bytes()], [R3_FIXTURE.name, LD_FIXTURE.name])
    assert [f["format"] for f in out.attrs["files"]] == ["magdrone_r3", "log_drone_csv"]
    assert out["file_index"].nunique() == 2 and out["point_id"].is_monotonic_increasing
    assert out[out.file_index == 1]["roll"].notna().all() and out[out.file_index == 0]["roll"].isna().all()


def test_api_upload_qc_preprocess_grid_and_targets():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(LD_FIXTURE, "rb") as f:
        r = client.post(f"/api/projects/{pid}/upload", files={"files": (LD_FIXTURE.name, f, "text/csv")})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["sensors"] == [1, 2] and s["sensor_separation_m"] == {} and s["n_tracks"] >= 2
    assert s["files"][0]["format"] == "log_drone_csv" and s["has_vector"] is False
    assert 35 <= s["sample_interval_ms"] <= 45
    qc = client.post(f"/api/projects/{pid}/qc", json={}).json()
    assert qc["files"][0]["n_paired_samples"] > 100
    pre = client.post(f"/api/projects/{pid}/preprocess", json={"lpf_hz": 5.0}).json()
    assert pre["files"][0]["n_lines"] >= 2
    g = client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.25}).json()
    assert g["stats"]["n_points"] > 500 and g["overlay"]["image_data_url"].startswith("data:image/png")
    t = client.post(f"/api/projects/{pid}/targets", json={"sensor_height_m": 1.0, "max_targets": 5})
    assert t.status_code == 200, t.text
    p = client.get(f"/api/projects/{pid}/line-profile", params={"file_index": 0, "line_id": int(s["n_tracks"] and 1)}).json()
    assert {x["sensor"] for x in p["series"]} == {1, 2}
