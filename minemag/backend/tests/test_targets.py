"""Target picking on a synthetic two-probe survey with known dipoles, and
through the API on the real fixture.

The synthetic: three induced dipoles 0.2-0.6 m under the ground, probes
0.5 m above it, 0.5 m tracks 10 m long, 3 nT noise; a second run with
different noise plays the repeat. The picks have to land on the dipoles,
recover depth and moment, be seen in both runs, and grade A - while a
pick that only one run has, or a weak one, must not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.processing.targets import merge_runs, pick_targets, pick_targets_in_run, targets_to_csv
from magcore.dipole_fit import field_direction_enu
from conftest import R3_FIXTURE

INC, DEC = 53.0, -8.0
PROBE_H = 0.5
TRUTH = [  # x, y, burial depth, moment (A m^2)
    (3.0, 3.0, 0.2, 0.3),
    (7.0, 6.0, 0.4, 1.5),
    (4.5, 8.5, 0.6, 4.0),
]


def _dipole_field(x, y, z_probe, sx, sy, sz, moment):
    f = field_direction_enu(INC, DEC)
    rx, ry, rz = x - sx, y - sy, z_probe - sz
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    dot = (f[0] * rx + f[1] * ry + f[2] * rz) / r
    return 1e9 * (4e-7 * np.pi * moment / (4 * np.pi * r**3)) * (3 * dot**2 - 1)


def _run(file_index: int, seed: int, noise_nt: float = 3.0, step: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    t_ms = 0.0
    for i, tx in enumerate(np.arange(0.0, 10.0 + 1e-9, 0.5)):
        ys = np.arange(0.0, 10.0 + 1e-9, step)
        if i % 2:
            ys = ys[::-1]
        for s, off in ((1, -0.5), (2, 0.5)):
            x = np.full(len(ys), tx + off)
            field = sum(_dipole_field(x, ys, PROBE_H, sx, sy, -d, m) for sx, sy, d, m in TRUTH)
            rows.append(pd.DataFrame({
                "t_ms": t_ms + np.arange(len(ys)) * 5.0, "track": i + 1, "sensor": s,
                "x": x, "y": ys, "tfa": 50.0 * s + field + noise_nt * rng.standard_normal(len(ys)),
            }))
        t_ms += len(ys) * 5.0 + 2000.0
    df = pd.concat(rows, ignore_index=True)
    df["file_index"] = file_index
    df["lat"], df["lon"] = 36.0, 127.0
    df["timestamp"] = pd.Timestamp("2026-04-25") + pd.to_timedelta(df["t_ms"], unit="ms")
    return df


def _survey(n_runs=2):
    df = pd.concat([_run(i, seed=i) for i in range(n_runs)], ignore_index=True)
    df.attrs["files"] = [{"filename": f"run{i}.asc"} for i in range(n_runs)]
    return df


def test_the_three_dipoles_are_picked_with_position_depth_and_moment():
    df = _survey(1)
    picks, info = pick_targets_in_run(df, INC, DEC, noise_rms_nt=3.0)
    assert info["threshold_nt"] == 20.0                       # 5 x 3 nT is below the 20 nT floor
    assert len(picks) == 3
    for sx, sy, d, m in TRUTH:
        near = min(picks, key=lambda p: np.hypot(p["x"] - sx, p["y"] - sy))
        assert np.hypot(near["x"] - sx, near["y"] - sy) < 0.1
        assert abs(near["depth_below_probe_m"] - (PROBE_H + d)) < 0.15
        assert 0.7 < near["moment_am2"] / m < 1.4
        assert near["fit_quality"] > 0.6


def test_repeated_runs_are_matched_and_graded():
    df = _survey(2)
    r = pick_targets(df, INC, DEC, 32652, noise_by_file={0: 3.0, 1: 3.0}, sensor_height_m=PROBE_H)
    # a wrong height for run 1 shifts the burial depth by half of it (mean over the two runs)
    shifted = pick_targets(df, INC, DEC, 32652, noise_by_file={0: 3.0, 1: 3.0}, sensor_height_m=PROBE_H,
                           sensor_heights_by_file={1: PROBE_H + 0.2})
    for a, b in zip(r["targets"], shifted["targets"]):
        assert abs((a["burial_depth_m"] - b["burial_depth_m"]) - 0.1) < 0.02
    assert r["n_runs"] == 2 and r["n_targets"] == 3
    for t in r["targets"]:
        assert t["runs_seen"] == [0, 1]
        assert t["grade"] == "A"
        assert t["position_spread_m"] < 0.15
        assert abs(t["burial_depth_m"] - min(TRUTH, key=lambda u: np.hypot(u[0] - t["x"], u[1] - t["y"]))[2]) < 0.15
        assert "lat" in t and t["size_class"]
    assert r["n_by_grade"] == {"A": 3, "B": 0, "C": 0}
    csv = targets_to_csv(r, 32652)
    assert csv.splitlines()[0].startswith("id,grade,lat,lon,x_epsg32652")
    assert len(csv.splitlines()) == 4


def test_a_pick_seen_in_one_run_only_cannot_be_grade_a():
    per_run = {
        0: [{"x": 1.0, "y": 1.0, "depth_below_probe_m": 0.7, "moment_am2": 1.0, "peak_nt": 300.0, "footprint_m": 0.5,
             "fit_quality": 0.9, "rms_residual_nt": 2.0, "n_points": 50, "snr": 100.0}],
        1: [],
    }
    merged = merge_runs(per_run, 0.5)
    assert len(merged) == 1 and merged[0]["runs_seen"] == [0]
    from app.processing.targets import _grade
    assert _grade(1, 2, 100.0, 0.9) == "B"
    assert _grade(2, 2, 100.0, 0.9) == "A"
    assert _grade(2, 2, 3.0, 0.9) == "C"


def test_the_noise_sets_the_threshold_so_a_noisy_run_picks_fewer():
    df = _survey(1)
    quiet, _ = pick_targets_in_run(df, INC, DEC, noise_rms_nt=3.0)
    noisy, info = pick_targets_in_run(df, INC, DEC, noise_rms_nt=40.0)
    assert info["threshold_nt"] == 200.0
    assert len(noisy) < len(quiet)


def test_the_api_picks_on_the_real_fixture_and_exports_csv():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(R3_FIXTURE, "rb") as f:
        client.post(f"/api/projects/{pid}/upload", files={"files": (R3_FIXTURE.name, f, "text/plain")})
    r = client.post(f"/api/projects/{pid}/targets", json={"sensor_height_m": 0.3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_runs"] == 1 and "0" in body["runs"]
    assert body["runs"]["0"]["threshold_nt"] >= 20.0
    for t in body["targets"]:
        assert t["grade"] in ("A", "B", "C") and t["burial_depth_m"] == round(t["depth_below_probe_m"] - 0.3, 2)
    csv = client.get(f"/api/projects/{pid}/targets.csv")
    assert csv.status_code == 200
    assert csv.text.lstrip("﻿").startswith("id,grade,lat,lon")   # BOM so Excel reads the Korean size class
