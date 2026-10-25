"""The MagDrone R3 ASC loader against a real export.

The fixture is two tracks (10 and 11) of the KIGAM test-site quadruped
run of 2026-04-25, every 10th sample, both probes - so the loader is
exercised on the real column layout, real timestamps (ms since midnight
plus the date from the file name) and real probe geometry (1 m apart).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.io_.magdrone_r3 import (
    MagDroneLoadError,
    load_magdrone_r3,
    load_magdrone_r3_files,
    parse_filename,
    summarize,
)
from conftest import R3_FIXTURE


@pytest.fixture(scope="module")
def frame():
    return load_magdrone_r3(R3_FIXTURE.read_bytes(), R3_FIXTURE.name)


def test_both_probes_tracks_vector_and_altitude_are_kept(frame):
    assert list(frame.columns) == ["timestamp", "t_ms", "track", "sensor", "lat", "lon", "alt", "tfa", "bx", "by", "bz", "b_total"]
    assert sorted(frame["sensor"].unique()) == [1, 2]
    assert sorted(frame["track"].unique()) == [10, 11]
    assert (frame["sensor"] == 1).sum() == (frame["sensor"] == 2).sum()
    assert np.isfinite(frame["alt"]).all()
    assert np.isfinite(frame[["bx", "by", "bz"]].to_numpy()).all()
    # a fluxgate's total field at Daejeon is about 50,000 nT
    assert 48000 < frame["b_total"].median() < 53000


def test_timestamps_combine_the_file_date_with_ms_since_midnight(frame):
    first = frame["timestamp"].iloc[0]
    assert first.strftime("%Y-%m-%d") == "2026-04-25"
    assert first.strftime("%H:%M:%S.%f")[:-3] == "00:18:09.870"
    assert frame["timestamp"].is_monotonic_increasing
    assert frame.attrs["file_info"]["start_date"] == "2026-04-25"


def test_filename_tokens_are_recorded():
    info = parse_filename("20260425_001316_MD-R3_#0115_noff_sf5_mm3(ROBOT1)_2Hz.asc")
    assert info.device == "MD-R3" and info.serial == "#0115"
    assert info.start_date == "2026-04-25"
    assert info.settings == ["noff", "sf5", "mm3(ROBOT1)", "2Hz"]
    unknown = parse_filename("export.asc")
    assert unknown.start_date is None


def test_summary_reports_probe_geometry_sampling_and_noise(frame):
    s = summarize(frame)
    assert s["sensors"] == [1, 2]
    assert abs(s["sensor_separation_m"]["2"] - 1.0) < 0.1       # the R3 rod
    assert abs(s["sample_interval_ms"] - 50.0) < 1.0             # every 10th 5 ms sample
    assert s["n_tracks"] == 2
    assert s["has_vector"] is True
    assert 0.5 < s["speed_m_s"]["median"] < 2.0                  # a walking robot
    assert s["altitude_m"]["median"] > 0
    for probe in ("1", "2"):
        assert s["per_sensor"][probe]["noise_4th_diff_nt"] is not None
    assert "2" in s["sensor_offset_nt"]
    assert s["n_gaps"] == 1                                       # the jump between track 10 and 11


def test_an_older_export_without_time_track_or_vector_columns_still_loads():
    text = R3_FIXTURE.read_text(encoding="utf-8")
    df = pd.read_csv(pd.io.common.StringIO(text), sep="\t")
    older = df[["Timestamp [ms]", "Sensor ID", "Latitude [°]", "Longitude [°]", "Total field anomaly [nT]"]]
    frame = load_magdrone_r3(older.to_csv(sep="\t", index=False), "20260425_001316_MD-R3_#0115.asc")
    assert (frame["track"] == -1).all()
    assert frame["alt"].isna().all() and frame["bx"].isna().all()
    assert sorted(frame["sensor"].unique()) == [1, 2]
    assert summarize(frame)["has_vector"] is False


def test_midnight_rollover_moves_to_the_next_day():
    header = "Timestamp [ms]\tSensor ID\tLatitude [°]\tLongitude [°]\tTotal field anomaly [nT]"
    # the export lists probe 1's whole timeline, then probe 2's
    samples = ((86399000, 1.0), (86399500, 2.0), (500, 3.0), (1000, 4.0))
    rows = [f"{t}\t{s}\t36.3789\t127.3610\t{v}" for s in (1, 2) for t, v in samples]
    frame = load_magdrone_r3("\n".join([header, *rows]), "20260425_235959_MD-R3_#0115.asc")
    assert frame["timestamp"].is_monotonic_increasing
    days = frame["timestamp"].dt.strftime("%Y-%m-%d").tolist()
    assert days == ["2026-04-25"] * 4 + ["2026-04-26"] * 4
    assert frame["sensor"].tolist() == [1, 2] * 4


def test_two_files_concatenate_in_time_order_with_file_index():
    data = R3_FIXTURE.read_bytes()
    combined = load_magdrone_r3_files(
        [data, data],
        ["20260426_001316_MD-R3_#0115_run2.asc", "20260425_001316_MD-R3_#0115_run1.asc"],
    )
    assert combined["timestamp"].is_monotonic_increasing
    assert combined["point_id"].is_unique
    assert combined["file_index"].iloc[0] == 1          # the earlier file comes first in time
    assert len(combined.attrs["files"]) == 2
    s = summarize(combined)
    assert s["n_files"] == 2
    assert s["n_tracks"] == 4                       # track ids restart in every export: 2 + 2
    assert abs(s["sensor_separation_m"]["2"] - 1.0) < 0.1


def test_a_file_without_the_required_columns_is_refused_with_the_header():
    with pytest.raises(MagDroneLoadError, match="필수 열"):
        load_magdrone_r3("Timestamp [ms]\tLatitude [°]\n1\t36.0\n", "x.asc")
    with pytest.raises(MagDroneLoadError, match="헤더"):
        load_magdrone_r3("lat,lon,mag\n36,127,50000\n", "x.csv")
