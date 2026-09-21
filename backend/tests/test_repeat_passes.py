"""Ground flown twice: finding it, measuring it, levelling on it.

The fixture is the situation this exists for - two flights an hour apart,
the second one repeating a few of the first one's lines, and a level
difference between them that no tie line exists to catch.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from app.processing.repeat_passes import find_repeat_passes, solve_offsets, summarize

AZIMUTH = 0.0          # lines run north
SPACING = 50.0


def _geology(x, y):
    return 30.0 * np.sin(2 * np.pi * x / 900.0) * np.cos(2 * np.pi * y / 1300.0) + 0.01 * x


def _flight(line_ids, flight_index, start, offset_nt=0.0, track_shift_m=0.0, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for n, line_id in enumerate(line_ids):
        y = np.arange(-500.0, 500.0, 2.0)
        x = np.full_like(y, line_id * SPACING + track_shift_m)
        rows.append(
            pd.DataFrame(
                {
                    "x": x,
                    "y": y,
                    "line_id": np.full(len(y), 1000 * flight_index + line_id),
                    "source_file_index": np.full(len(y), flight_index),
                    "timestamp": pd.date_range(start + pd.Timedelta(minutes=8 * n), periods=len(y), freq="200ms"),
                    "anomaly": _geology(x, y) + offset_nt + rng.normal(0, noise, len(y)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def survey():
    """Flight 0 flies lines 0-5; flight 1, an hour later, re-flies lines
    3-5 and reads 6 nT high."""
    first = _flight(range(6), 0, pd.Timestamp("2026-09-09 01:00"), seed=1)
    second = _flight(range(3, 6), 1, pd.Timestamp("2026-09-09 02:00"), offset_nt=6.0, seed=2)
    return pd.concat([first, second], ignore_index=True)


def test_it_finds_exactly_the_re_flown_lines(survey):
    pairs = find_repeat_passes(survey, AZIMUTH, SPACING)

    assert len(pairs) == 3
    assert {(p.line_a % 1000, p.line_b % 1000) for p in pairs} == {(3, 3), (4, 4), (5, 5)}
    assert all(p.flight_a != p.flight_b for p in pairs)
    assert all(p.overlap_m > 900 for p in pairs)


def test_neighbouring_lines_are_not_repeats(survey):
    """The test that keeps this from "finding" the whole survey: the next
    line over covers ground 50 m away, not the same ground."""
    pairs = find_repeat_passes(survey, AZIMUTH, SPACING)

    assert all(p.across_track_m < 5.0 for p in pairs)


def test_it_measures_the_level_difference_and_what_is_left(survey):
    pairs = find_repeat_passes(survey, AZIMUTH, SPACING)

    for p in pairs:
        assert abs(abs(p.offset_nt) - 6.0) < 0.5      # the planted step
        assert p.residual_rms_nt < 1.0                 # noise only
        assert 0.5 < p.hours_apart < 2.0


def test_the_offsets_solve_to_one_level_per_flight(survey):
    pairs = find_repeat_passes(survey, AZIMUTH, SPACING)

    offsets = solve_offsets(pairs)

    assert set(offsets) == {0, 1}
    assert abs((offsets[0] - offsets[1]) - (-6.0)) < 0.5
    assert abs(sum(offsets.values())) < 1e-6           # zero-mean datum


def test_applying_the_offsets_makes_the_repeats_agree(survey):
    pairs = find_repeat_passes(survey, AZIMUTH, SPACING)
    offsets = solve_offsets(pairs)

    corrected = survey.copy()
    corrected["anomaly"] -= corrected["source_file_index"].map(offsets).fillna(0.0)
    after = find_repeat_passes(corrected, AZIMUTH, SPACING)

    assert max(abs(p.offset_nt) for p in after) < 0.5


def test_a_survey_with_no_repeats_says_what_to_do_instead():
    single = _flight(range(6), 0, pd.Timestamp("2026-09-09 01:00"))

    pairs = find_repeat_passes(single, AZIMUTH, SPACING)
    report = summarize(pairs, {})

    assert pairs == []
    assert report["n_pairs"] == 0
    assert "다시 날" in report["advice"]


def test_a_stretch_too_short_to_compare_is_not_offered():
    first = _flight([0], 0, pd.Timestamp("2026-09-09 01:00"))
    second = _flight([0], 1, pd.Timestamp("2026-09-09 02:00"))
    second = second[second["y"] < -450]                # 50 m of shared track

    pairs = find_repeat_passes(pd.concat([first, second], ignore_index=True), AZIMUTH, SPACING)

    assert pairs == []


def test_a_repeat_flown_slightly_off_track_still_counts(survey):
    """Real re-flights do not retrace the track exactly; 8 m off on a
    50 m spacing is still the same ground."""
    shifted = _flight(range(3, 6), 1, pd.Timestamp("2026-09-09 02:00"), offset_nt=6.0, track_shift_m=8.0)
    both = pd.concat([_flight(range(6), 0, pd.Timestamp("2026-09-09 01:00")), shifted], ignore_index=True)

    pairs = find_repeat_passes(both, AZIMUTH, SPACING)

    assert len(pairs) == 3
    assert all(6.0 < p.across_track_m < 10.0 for p in pairs)


def test_the_summary_reports_before_and_after(survey):
    pairs = find_repeat_passes(survey, AZIMUTH, SPACING)
    report = summarize(pairs, solve_offsets(pairs))

    assert report["n_pairs"] == 3
    assert abs(report["median_offset_nt"] - 6.0) < 0.5
    assert report["median_offset_after_nt"] < 0.5
    assert report["n_flights_levelled"] == 2
    assert report["total_overlap_m"] > 2700


# ---------------------------------------------------------------- through the app
import pathlib as _pathlib

from fastapi.testclient import TestClient

from app.main import app
from app.store import Project
from app.store import store as project_store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def _processed():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    assert client.post(f"/api/projects/{pid}/process", json={"line_params": {}, "diurnal_params": {}}).status_code == 200
    return client, pid, project_store.get(pid)


def _plant_two_flights(project, step_nt=7.0):
    """Split the survey's lines into two "flights" and put a level step
    between them, with the lines re-flown so there is repeated ground."""
    df = project.processed_base
    on_line = df[df["line_id"] >= 0]
    lines = sorted(on_line["line_id"].unique())
    assert len(lines) >= 2
    longest = on_line["line_id"].value_counts().index[0]
    repeated = df[df["line_id"] == longest].copy()
    repeated["source_file_index"] = 1
    repeated["line_id"] = max(lines) + 1
    repeated["point_id"] = repeated["point_id"] + df["point_id"].max() + 1
    for col in ("anomaly", "tmi"):
        repeated[col] = repeated[col] + step_nt
    df = df.copy()
    df["source_file_index"] = 0
    combined = pd.concat([df, repeated], ignore_index=True)
    project.processed_base = combined
    project.processed = combined.copy()
    project.grid_cache = {}
    return step_nt


def test_the_api_measures_and_levels_a_step_between_flights():
    client, pid, project = _processed()
    step = _plant_two_flights(project)

    found = client.post(f"/api/projects/{pid}/repeat-passes", json={}).json()
    assert found["n_pairs"] >= 1
    assert abs(found["median_offset_nt"] - step) < 1.0
    assert found["applied"] is False

    applied = client.post(f"/api/projects/{pid}/repeat-passes/apply", json={"mode": "apply"})
    assert applied.status_code == 200, applied.text
    assert applied.json()["applied"] is True

    after = client.post(f"/api/projects/{pid}/repeat-passes", json={}).json()
    assert after["median_offset_nt"] < 1.0

    reset = client.post(f"/api/projects/{pid}/repeat-passes/apply", json={"mode": "reset"}).json()
    assert reset["applied"] is False
    assert abs(client.post(f"/api/projects/{pid}/repeat-passes", json={}).json()["median_offset_nt"] - step) < 1.0


def test_levelling_survives_a_save_and_composes_with_structure_removal():
    client, pid, project = _processed()
    _plant_two_flights(project)
    client.post(f"/api/projects/{pid}/repeat-passes/apply", json={"mode": "apply"})
    levelled = project.processed["anomaly"].to_numpy().copy()

    other = Project(id="reloaded")
    other.load_project_bundle(project.save_project_bundle())
    assert other.repeat_pass_offsets

    # point smoothing still applies on top, and undoing it leaves the
    # levelled data rather than the raw data
    ids = [int(i) for i in project.processed["point_id"].iloc[100:160]]
    client.post(f"/api/projects/{pid}/smooth", json={"mode": "point_ids", "point_ids": ids})
    client.post(f"/api/projects/{pid}/smooth", json={"mode": "reset"})
    assert np.allclose(project.processed["anomaly"].to_numpy(), levelled)


def test_a_survey_with_no_repeats_refuses_to_level():
    client, pid, _ = _processed()

    r = client.post(f"/api/projects/{pid}/repeat-passes/apply", json={"mode": "apply"})

    assert r.status_code == 400
    assert "두 번 비행" in r.json()["detail"]
