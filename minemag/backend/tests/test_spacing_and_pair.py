"""Auto max distance from the line spacing, per-sensor heading
correction, and the sensor-pair diagnostic."""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from magcore.survey_files import load_survey_files
from app.main import app
from app.processing.gridding import auto_max_distance, estimate_line_spacing, grid_survey
from app.processing.preprocess import PreprocessParams, preprocess
from magcore.sensor_pair import sensor_pair_report
from conftest import FIXTURES, R3_FIXTURE
from magcore.lines import project_to_local_xy

LD_FIXTURE = FIXTURES / "Log_Drone_20261019_134937_rows6000-8600.csv"


def _lines(spacing: float, n_lines: int = 5, length: float = 30.0, step: float = 0.1) -> pd.DataFrame:
    rows = []
    for k in range(n_lines):
        y = np.arange(0, length, step)
        if k % 2:
            y = y[::-1]
        for yy in y:
            rows.append({"x": k * spacing + np.random.default_rng(k).normal(0, 0.02), "y": yy, "tfa": 0.0,
                         "sensor": 1, "file_index": 0, "line_id": k, "t_ms": len(rows) * 50.0})
    return pd.DataFrame(rows)


def test_auto_max_distance_follows_the_line_spacing():
    for spacing, expect in [(0.5, 0.6), (2.0, 1.5), (3.0, 2.25), (10.0, 5.0)]:
        s, _az = estimate_line_spacing(_lines(spacing))
        assert abs(s - spacing) < 0.1
        assert abs(auto_max_distance(s) - expect) < 0.05
    assert auto_max_distance(None) == 0.6
    grid, stats = grid_survey(_lines(2.0), cell_size_m=0.25)
    assert stats["max_distance_auto"] is True and abs(stats["max_distance_m"] - 1.5) < 0.05 and abs(stats["line_spacing_m"] - 2.0) < 0.1
    assert stats["coverage_pct"] > 90                                   # the gaps between 2 m lines are filled
    _g, stats06 = grid_survey(_lines(2.0), cell_size_m=0.25, max_distance_m=0.6)
    assert stats06["max_distance_auto"] is False and stats06["coverage_pct"] < 70


def _loaded(path):
    df = load_survey_files([path.read_bytes()], [path.name])
    x, y, _ = project_to_local_xy(df["lat"].to_numpy(), df["lon"].to_numpy())
    df["x"], df["y"] = x, y
    return df


def test_r3_pattern_keeps_the_0_6_m_default_and_per_sensor_heading_report():
    df = _loaded(R3_FIXTURE)
    proc, rep = preprocess(df, PreprocessParams(lpf_hz=2.0))
    assert rep["files"][0]["line_spacing_m"] is not None and rep["files"][0]["line_spacing_m"] < 1.0
    _g, stats = grid_survey(proc, cell_size_m=0.2)
    assert stats["max_distance_auto"] and stats["max_distance_m"] == 0.6
    h = rep["files"][0]["heading_correction"]
    assert "per_sensor" in h                                            # two tracks only: nothing applied, but the shape is there


def test_log_drone_gets_a_wider_reach_and_a_pair_report():
    df = _loaded(LD_FIXTURE)
    r = sensor_pair_report(df[df.file_index == 0])
    assert r is not None and r["sensors"] == [1, 2]
    assert -70 < r["offset_nt"] < -30                                   # J2 - J1
    assert r["attitude"] is not None and set(r["attitude"]["per_sensor"]) == {"1", "2"}
    assert r["attitude"]["per_sensor"]["1"]["std_after_nt"] <= r["attitude"]["per_sensor"]["1"]["std_before_nt"] + 1e-6
    assert any("센서" in line for line in r["verdict"])
    s, _ = estimate_line_spacing(df)
    assert s is not None and 0.8 < s < 4.0                            # the slice holds a few lines, 1-1.5 m apart
    proc, rep = preprocess(df, PreprocessParams(lpf_hz=5.0))
    _g, stats = grid_survey(proc, cell_size_m=0.25, file_index=0)
    assert stats["max_distance_auto"] and stats["max_distance_m"] > 0.6
    r3 = sensor_pair_report(_loaded(R3_FIXTURE))
    assert r3 is not None and r3["attitude"] is None                     # the R3 export has no attitude


def test_api_grid_without_max_distance_is_auto_and_qc_carries_the_pair():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(LD_FIXTURE, "rb") as f:
        client.post(f"/api/projects/{pid}/upload", files={"files": (LD_FIXTURE.name, f, "text/csv")})
    g = client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.25}).json()
    assert g["stats"]["max_distance_auto"] is True and g["stats"]["line_spacing_m"] is not None
    g2 = client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.25, "max_distance_m": 0.6}).json()
    assert g2["stats"]["max_distance_auto"] is False and g2["stats"]["coverage_pct"] <= g["stats"]["coverage_pct"]
    qc = client.post(f"/api/projects/{pid}/qc", json={}).json()
    sp = qc["files"][0]["sensor_pair"]
    assert sp and sp["offset_nt"] < 0 and sp["verdict"]
