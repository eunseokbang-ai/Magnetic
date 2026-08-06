"""End-to-end coverage of manual distortion-removal smoothing through the
actual Project/store.py wiring (set_manual_smoothing, get_line_profile),
not just the isolated processing/manual_smooth.py function - this is the
path the frontend's map-polygon and time-series drag-select tools actually
exercise, and where two real bugs were found and fixed:

1. get_line_profile's "raw_value" (pre-filter reference trace) was
   computed additively from self.processed, so once a stretch was
   smoothed, the "raw" trace mirrored the smoothing edit instead of still
   showing the true original (pre-smoothing) signal.
2. There was no way to undo a smoothing edit once applied - set_manual_
   smoothing's response now returns the full current point_id set so the
   frontend can keep an undo history.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.models import ManualSmoothRequest, ProcessParams
from app.store import Project


def _make_project_with_dipole_distortion() -> tuple[Project, list[int]]:
    """A single straight N-S line with a smooth background field and a
    dipole-shaped bump (rises then dips, like a small buried/nearby
    ferrous object or a house) injected into a short stretch in the
    middle - the same "localized non-geological disturbance" shape the
    manual-smoothing feature exists to remove."""
    lat0, lon0 = 37.5, 127.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))

    n = 150
    y = np.arange(0.0, n * 2.0, 2.0)  # 2 m/s at 1 Hz
    mag = np.full(n, 50000.0)

    # smooth, exactly-zero-outside distortion: cosine-tapered rise, flat
    # plateau, cosine-tapered fall back to zero - like a real localized
    # magnetic disturbance's footprint, but with a *bounded* extent (no
    # long tail) so the true background level right outside it is known
    # exactly, and a sharp step that the despike QC would legitimately
    # flag/remove before manual smoothing ever sees it is avoided.
    start, end, ramp, amplitude = 60, 90, 10, 30.0
    bump = np.zeros(n)
    for i in range(n):
        if start <= i < start + ramp:
            bump[i] = amplitude * 0.5 * (1 - np.cos(np.pi * (i - start) / ramp))
        elif start + ramp <= i < end - ramp:
            bump[i] = amplitude
        elif end - ramp <= i < end:
            bump[i] = amplitude * 0.5 * (1 - np.cos(np.pi * (end - i) / ramp))
    mag = mag + bump
    dipole_ids = list(range(start, end))

    t0 = pd.Timestamp("2026-01-01T00:00:00")
    df = pd.DataFrame(
        {
            "x": 0.0,
            "y": y,
            "mag_raw": mag,
            # 10 Hz sampling (matches a real drone logger, and other tests'
            # convention) - 1 Hz would put the default 1.0 Hz lowpass
            # cutoff right at/above the Nyquist frequency.
            "timestamp": [t0 + pd.Timedelta(milliseconds=100 * i) for i in range(n)],
        }
    )
    df["lat"] = lat0 + df["y"] / m_per_deg_lat
    df["lon"] = lon0
    df["point_id"] = np.arange(n, dtype=np.int64)
    df["altitude_msl_m"] = 100.0
    df["geoid_separation_m"] = 0.0
    df["altitude_ellipsoidal_m"] = 100.0
    df["speed_over_ground"] = np.nan
    df["gyro_mag"] = np.nan
    df["accel_horiz_g"] = np.nan
    df["compass_x"] = np.nan
    df["compass_y"] = np.nan
    df["compass_z"] = np.nan

    base_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(df["timestamp"].min() - pd.Timedelta(minutes=1), df["timestamp"].max() + pd.Timedelta(minutes=1), periods=50),
            "mag": 50000.0,
        }
    )

    project = Project(id="test-manual-smooth")
    project.drone_raw = df
    project.base_raw = base_df

    params = ProcessParams()
    params.line_params.min_speed_mps = 0.1
    params.line_params.min_line_length_m = 50.0
    params.line_params.turn_buffer_m = 2.0
    params.line_params.max_gap_seconds = 5.0
    project.run_pipeline(params)
    return project, dipole_ids


def _positions_for_point_ids(profile: dict, point_ids) -> list[int]:
    """get_line_profile's lists are positional within the (possibly
    trimmed/reordered) line, not aligned to raw pre-pipeline row indices -
    look up each point_id's actual position instead of assuming it."""
    pos_by_id = {pid: i for i, pid in enumerate(profile["point_id"])}
    return [pos_by_id[pid] for pid in point_ids]


def _background_level(profile: dict, dipole_positions: list[int], window: int = 5) -> float:
    """Median of the few points immediately adjacent to the flagged run on
    each side - "anomaly" here sits on a large constant regional-field
    offset (this synthetic setup has no proper IGRF/diurnal cancellation),
    so the bump must be judged relative to its local background, not an
    absolute zero. Using the *immediate* neighbors (not some far point
    elsewhere on the line) also matches what the smoothing algorithm
    itself anchors to - the synthetic dipole's derivative-of-Gaussian
    shape has a long, very-low-amplitude tail, so a far-away reference
    point would include some residual tail the algorithm doesn't try to
    remove either."""
    lo, hi = min(dipole_positions), max(dipole_positions)
    neighbor_positions = list(range(max(0, lo - window), lo)) + list(range(hi + 1, min(len(profile["value"]), hi + 1 + window)))
    return float(np.median([profile["value"][i] for i in neighbor_positions]))


def test_set_manual_smoothing_by_point_ids_flattens_value_but_preserves_raw_reference():
    project, dipole_row_indices = _make_project_with_dipole_distortion()
    line_id = int(project.processed.loc[project.processed["line_id"] >= 0, "line_id"].iloc[0])
    assert line_id >= 0

    point_ids = project.processed.loc[dipole_row_indices, "point_id"].tolist()

    profile_before = project.get_line_profile(line_id, "anomaly")
    pos_before = _positions_for_point_ids(profile_before, point_ids)
    background = _background_level(profile_before, pos_before)
    assert max(profile_before["value"][i] - background for i in pos_before) > 15  # the bump is really there

    resp = project.set_manual_smoothing(ManualSmoothRequest(mode="point_ids", point_ids=point_ids))
    assert set(resp["manual_smooth_point_ids"]) == set(int(p) for p in point_ids)

    profile_after = project.get_line_profile(line_id, "anomaly")
    pos_after = _positions_for_point_ids(profile_after, point_ids)
    smoothed_deviations = [profile_after["value"][i] - background for i in pos_after]
    # flattened to background level, not the original bump
    assert max(abs(v) for v in smoothed_deviations) < 5

    # bug fix: the raw (pre-filter) reference trace must still show the
    # true original distortion, not mirror the smoothing edit.
    raw_after = [profile_after["raw_value"][i] - background for i in pos_after]
    assert max(raw_after) > 15

    assert all(profile_after["smoothed"][i] for i in pos_after)
    untouched_pos = [p for p in range(len(profile_after["point_id"])) if p not in pos_after][:3]
    assert not any(profile_after["smoothed"][i] for i in untouched_pos)


def test_set_manual_smoothing_reset_restores_original_value():
    project, dipole_row_indices = _make_project_with_dipole_distortion()
    line_id = int(project.processed.loc[project.processed["line_id"] >= 0, "line_id"].iloc[0])
    point_ids = project.processed.loc[dipole_row_indices, "point_id"].tolist()

    profile_before = project.get_line_profile(line_id, "anomaly")
    pos = _positions_for_point_ids(profile_before, point_ids)
    background = _background_level(profile_before, pos)

    project.set_manual_smoothing(ManualSmoothRequest(mode="point_ids", point_ids=point_ids))
    resp = project.set_manual_smoothing(ManualSmoothRequest(mode="reset"))
    assert resp["manual_smooth_point_ids"] == []

    profile = project.get_line_profile(line_id, "anomaly")
    restored = [profile["value"][i] - background for i in pos]
    assert max(restored) > 15  # back to the original bump
    assert not any(profile["smoothed"])


def test_set_manual_smoothing_by_polygon_selects_same_points_as_point_ids():
    project, dipole_row_indices = _make_project_with_dipole_distortion()
    line_id = int(project.processed.loc[project.processed["line_id"] >= 0, "line_id"].iloc[0])
    point_ids = set(int(p) for p in project.processed.loc[dipole_row_indices, "point_id"].tolist())

    lats = project.processed.loc[dipole_row_indices, "lat"]
    lons = project.processed.loc[dipole_row_indices, "lon"]
    pad = 0.0005
    polygon = [
        [lats.min() - pad, lons.min() - pad],
        [lats.min() - pad, lons.max() + pad],
        [lats.max() + pad, lons.max() + pad],
        [lats.max() + pad, lons.min() - pad],
    ]

    profile_before = project.get_line_profile(line_id, "anomaly")
    pos = _positions_for_point_ids(profile_before, point_ids)
    background = _background_level(profile_before, pos)

    resp = project.set_manual_smoothing(ManualSmoothRequest(mode="polygon", polygon=polygon))
    assert point_ids.issubset(set(resp["manual_smooth_point_ids"]))

    profile = project.get_line_profile(line_id, "anomaly")
    smoothed_deviations = [profile["value"][i] - background for i in pos]
    assert max(abs(v) for v in smoothed_deviations) < 5
