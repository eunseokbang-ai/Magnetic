"""Integration test for the "반복비행 중복 측선 자동 선택" feature end to
end through the real upload/process API: a survey with a normal 2-line
block plus a *near-duplicate* re-flight of one of those lines (offset by
only a few meters, well inside the tight tolerance meant to catch actual
repeat flights rather than normal line-to-line spacing), where the
duplicate is deliberately much noisier than the original. With
duplicate_line_params.enabled=True, the noisier duplicate's overlapping
points should end up excluded and reported in
process_summary()["duplicate_line_resolution"]; with it left at its
default (disabled), nothing should change from today's behaviour.
"""
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from pyproj import Transformer

from app.main import app

UTM_EPSG = 32648  # zone 48N, matches this app's other synthetic-survey tests
REF_EASTING = 300000.0
REF_NORTHING = 5150000.0


def _line_rows(rng, x_offset_m, y_m, length_m=300.0, speed_mps=8.0, sample_hz=10.0, mag_base=50000.0, noise_std=0.5, start_time=None):
    dt = 1.0 / sample_hz
    n = int(length_m / (speed_mps * dt))
    t = np.arange(n) * dt
    easting = REF_EASTING + x_offset_m + speed_mps * t
    northing = np.full(n, REF_NORTHING + y_m)
    mag = mag_base + 3.0 * np.sin(easting / 40.0) + rng.normal(0, noise_std, n)
    transformer = Transformer.from_crs(f"EPSG:{UTM_EPSG}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    start = start_time if start_time is not None else pd.Timestamp("2026-07-24 06:00:00")
    timestamps = start + pd.to_timedelta(t, unit="s")
    return pd.DataFrame({
        "Date": timestamps.strftime("%Y-%m-%d"),
        "Time": timestamps.strftime("%H:%M:%S.%f").str[:-3],
        "Latitude": lat,
        "Longitude": lon,
        "Mag": mag,
        "Altitude": np.full(n, 50.0),
    }), timestamps[-1] + pd.Timedelta(seconds=5)


def _build_survey_csv(with_duplicate: bool) -> bytes:
    rng = np.random.default_rng(0)
    rows = []
    t = pd.Timestamp("2026-07-24 06:00:00")

    line_a, t = _line_rows(rng, 0.0, 0.0, noise_std=0.3, start_time=t)  # clean line A
    rows.append(line_a)

    line_b, t = _line_rows(rng, 0.0, 60.0, noise_std=0.3, start_time=t)  # normal, distinctly-spaced line B
    rows.append(line_b)

    if with_duplicate:
        # Near-exact re-flight of line A, offset only 3m perpendicular
        # (well under the default 8m tolerance) - much noisier, as if the
        # first pass had a problem and got re-flown.
        dup, t = _line_rows(rng, 0.0, 3.0, noise_std=6.0, start_time=t)
        rows.append(dup)

    df = pd.concat(rows, ignore_index=True)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _upload_and_process(client, csv_bytes, duplicate_line_params=None):
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    r = client.post(
        f"/api/projects/{project_id}/upload/drone",
        files={"files": ("survey.csv", csv_bytes, "text/csv")},
    )
    assert r.status_code == 200, r.text
    body = {"line_params": {}, "diurnal_params": {"mode": "assume_constant"}}
    if duplicate_line_params is not None:
        body["duplicate_line_params"] = duplicate_line_params
    r = client.post(f"/api/projects/{project_id}/process", json=body)
    assert r.status_code == 200, r.text
    return project_id, r.json()


def test_duplicate_line_excluded_when_enabled():
    client = TestClient(app)
    csv_bytes = _build_survey_csv(with_duplicate=True)

    project_id, summary_off = _upload_and_process(client, csv_bytes)
    assert summary_off["duplicate_line_resolution"]["enabled"] is False
    assert summary_off["n_lines"] == 3, "sanity check: all 3 physical passes should be detected as separate lines"

    project_id, summary_on = _upload_and_process(client, csv_bytes, duplicate_line_params={"enabled": True})
    dup_info = summary_on["duplicate_line_resolution"]
    assert dup_info["enabled"] is True
    assert dup_info["n_groups"] == 1, dup_info
    assert dup_info["n_points_excluded"] > 0, dup_info
    group = dup_info["groups"][0]
    assert len(group["line_ids"]) == 2  # line A and its duplicate, not line B
    kept_pass = next(p for p in group["passes"] if p["kept"])
    excluded_pass = next(p for p in group["passes"] if not p["kept"])
    assert kept_pass["quality_nt"] < excluded_pass["quality_nt"], "the cleaner pass must be the one kept"

    # More points should be excluded overall than with detection off, and
    # kept-active point count should drop by roughly the duplicate's size.
    assert summary_on["n_excluded_auto"] > summary_off["n_excluded_auto"]
    assert summary_on["n_kept"] < summary_off["n_kept"]


def test_no_duplicate_group_when_lines_are_normally_spaced():
    """The 2-line survey (no injected duplicate) must report zero groups -
    line A and line B are 60m apart, far outside the tight default 8m
    tolerance, so they must never be treated as the same repeated track."""
    client = TestClient(app)
    csv_bytes = _build_survey_csv(with_duplicate=False)
    project_id, summary = _upload_and_process(client, csv_bytes, duplicate_line_params={"enabled": True})
    dup_info = summary["duplicate_line_resolution"]
    assert dup_info["enabled"] is True
    assert dup_info["n_groups"] == 0
    assert dup_info["n_points_excluded"] == 0
    assert summary["n_lines"] == 2
