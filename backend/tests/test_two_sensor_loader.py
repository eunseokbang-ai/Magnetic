"""The two-sensor Log_Drone / MicroInfinity CSV in DroneMag Studio: read
through magcore's shared loader (the one MineMag uses), both sensors
kept, the processed channel chosen in the pipeline, and the pair
diagnostic in the processing summary."""
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.io_.drone_loader import load_drone_csv
from app.main import app
from app.store import store

HEADER = ("LaserCurr_J1[mA], LaserCurr_J2[mA], LaserTemp_J1[℃], LaserTemp_J2[℃], ModFreq_J1[KHz], ModFreq_J2[KHz], "
          "ModAmp_J1[mApp], ModAmp_J2[mApp], HTPWR_J1[%], HTPWR_J2[%], CellTemp_J1[Ohm], CellTemp_J2[Ohm], "
          "MagField_J1[uT], MagField_J2[uT], SeqNum, TOW, Lat[deg], Lon[deg], Alt[m], Roll[s], Pitch[uT], Yaw[s], Vel. N, Vel. E, Vel. D")
# 2026-10-19 is a Monday: GPS day-of-week 1. 13:35:06 KST = 04:35:06 UTC = 16506 s; GPS time is 18 s ahead of UTC.
TOW0 = 1 * 86400 + 16506 + 18


def _survey_csv(n_lines=8, line_m=300.0, spacing_m=20.0, hz=10.0, speed=5.0, direction_effect=(-2.0, 9.0), offset=-52.0, lowercase=False):
    """Back-and-forth lines over a dipole-like anomaly. Sensor 2 reads
    `offset` nT below sensor 1 and each sensor carries its own
    direction effect (the reading differs by that much between the two
    flight directions)."""
    rng = np.random.default_rng(0)
    lat0, lon0 = 36.5, 127.2
    rows, t = [], 0.0
    n_per = int(line_m / speed * hz)
    for k in range(n_lines):
        forward = k % 2 == 0
        along = np.linspace(0, line_m, n_per) if forward else np.linspace(line_m, 0, n_per)
        x = k * spacing_m
        for a in along:
            field = 50600.0 + 80.0 * np.exp(-(((a - 150.0) / 40.0) ** 2 + ((x - 70.0) / 40.0) ** 2))
            sign = 0.5 if forward else -0.5
            j1 = field + sign * direction_effect[0] + rng.normal(0, 0.3)
            j2 = field + offset + sign * direction_effect[1] + rng.normal(0, 0.3)
            lat = lat0 + a / 110540.0
            lon = lon0 + x / (111320.0 * np.cos(np.radians(lat0)))
            rows.append(f"83.8,67.0,60.4,18.3,354.6,353.9,45,23,63,60,1350.3,1350.2,{j1 / 1000:.6f},{j2 / 1000:.6f},{len(rows)},{TOW0 + t:.3f},"
                        f"{lat:.8f},{lon:.8f},41.2,1.3,2.9,-16.0,0,0,0")
            t += 1.0 / hz
        t += 8.0                                               # the turn
    header = HEADER.replace("Lat[deg]", "lat[deg]").replace("Lon[deg]", "lon[deg]").replace("Alt[m]", "alt[m]") if lowercase else HEADER
    return (header + "\n" + "\n".join(rows) + "\n").encode("utf-8-sig")


def _named(data: bytes, name: str) -> io.BytesIO:
    buf = io.BytesIO(data)
    buf.name = name
    return buf


def test_both_spellings_of_the_header_load_and_keep_both_sensors():
    for lowercase in (False, True):                             # the Gongju files, and the older firmware
        df = load_drone_csv(_named(_survey_csv(lowercase=lowercase), "Log_Drone_20261019_133506.csv"))
        assert df.attrs["source_format"] == "microinfinity" and df.attrs["sensor_labels"] == ["J1", "J2"]
        assert {"mag_ch1", "mag_ch2"} <= set(df.columns)
        np.testing.assert_allclose(df["mag_raw"], df["mag_ch1"])                 # J1 is the default channel, as before
        assert np.median(df["mag_ch2"] - df["mag_ch1"]) == pytest.approx(-52.0, abs=1.0)
        assert 50500 < df["mag_raw"].median() < 50700                           # microtesla -> nanotesla


def test_timestamps_are_real_utc_taken_from_the_file_name():
    df = load_drone_csv(_named(_survey_csv(), "Log_Drone_20261019_133506.csv"))
    assert df.attrs["placeholder_date"] is False and df.attrs["utc_offset_hours"] == 9.0
    # 13:35:06 local on the file name, UTC+9 -> 04:35:06 UTC on that date
    assert df["timestamp"].iloc[0] == pd.Timestamp("2026-10-19 04:35:06")
    assert df["timestamp"].diff().dropna().median() == pd.Timedelta(milliseconds=100)


def test_a_file_without_a_date_in_its_name_is_flagged_not_silently_dated_today():
    df = load_drone_csv(_named(_survey_csv(), "renamed.csv"))
    assert df.attrs["placeholder_date"] is True and df.attrs["utc_offset_hours"] is None
    assert df["timestamp"].iloc[0].normalize() == pd.Timestamp.utcnow().tz_localize(None).normalize()
    assert df["timestamp"].iloc[0].strftime("%H:%M:%S") == "04:35:06"                 # the time of day is still the real UTC one
    assert df["timestamp"].diff().dropna().median() == pd.Timedelta(milliseconds=100)   # spacing is still exact


def test_upload_process_and_switch_the_sensor_through_the_api():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    r = client.post(f"/api/projects/{pid}/upload/drone", files=[("files", ("Log_Drone_20261019_133506.csv", _survey_csv(), "text/csv"))])
    assert r.status_code == 200, r.text
    summary = r.json()
    assert summary["sensor_labels"] == ["J1", "J2"] and summary["n_files_placeholder_date"] == 0
    assert summary["time_range"][0].startswith("2026-10-19T04:35:06")          # the upload carried the file name through

    means = {}
    for channel in ("1", "2", "mean"):
        r = client.post(f"/api/projects/{pid}/process", json={"sensor_channel": channel, "diurnal_params": {"mode": "assume_constant"}})
        assert r.status_code == 200, r.text
        p = r.json()
        assert p["n_lines"] == 8
        means[channel] = float(store.get(pid).processed["tmi"].mean())
        pair = p["sensor_pair"]
        assert pair["available"] is True and pair["labels"] == ["J1", "J2"] and pair["used"] == channel
        assert pair["offset_nt"] == pytest.approx(-52.0, abs=1.0)
        # each sensor's own direction effect, recovered from the lines flown each way
        assert pair["direction_effect_nt"]["J1"] == pytest.approx(-2.0, abs=0.8)
        assert pair["direction_effect_nt"]["J2"] == pytest.approx(9.0, abs=0.8)
    # the chosen channel is what gets processed: J2 sits 52 nT below J1, the mean halfway
    assert means["2"] - means["1"] == pytest.approx(-52.0, abs=2.0)
    assert means["mean"] - means["1"] == pytest.approx(-26.0, abs=2.0)

    # both channels survive a save / load, so the choice can still be changed afterwards
    data = store.get(pid).save_project_bundle()
    r = client.post(f"/api/projects/{pid}/load", files={"file": ("p.magproj", data, "application/zip")})
    assert r.status_code == 200, r.text
    assert r.json()["drone_summary"]["sensor_labels"] == ["J1", "J2"] and r.json()["params"]["sensor_channel"] == "mean"
    assert "mag_ch2" in store.get(pid).drone_raw.columns


def test_single_sensor_data_is_untouched():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open("tests/fixtures/sample_drone_survey.csv", "rb") as f:
        summary = client.post(f"/api/projects/{pid}/upload/drone", files=[("files", ("d.csv", f.read(), "text/csv"))]).json()
    assert summary["sensor_labels"] is None
    assert "mag_ch1" not in store.get(pid).drone_raw.columns
    p = client.post(f"/api/projects/{pid}/process", json={"sensor_channel": "2", "diurnal_params": {"mode": "assume_constant"}}).json()
    assert p["sensor_pair"] is None and p["n_lines"] == 6                      # the option is simply ignored


GONGJU = pathlib.Path(__file__).resolve().parents[2] / "Landmine" / "Gongju_ADD20261019"


@pytest.mark.skipif(not list(GONGJU.glob("case8_*/Log_Drone_20261019_134937.csv")), reason="Gongju field data not present")
def test_the_gongju_grid_flight_now_opens_in_dronemag_studio():
    """Before 2026-11 this file was refused ("필수 컬럼이 없습니다:
    ['lat[deg]', 'lon[deg]']"): the header spells them Lat / Lon."""
    path = next(GONGJU.glob("case8_*/Log_Drone_20261019_134937.csv"))
    df = load_drone_csv(_named(path.read_bytes(), path.name))
    assert df.attrs["sensor_labels"] == ["J1", "J2"] and df.attrs["placeholder_date"] is False
    assert df["timestamp"].iloc[0].date() == pd.Timestamp("2026-10-19").date() and df.attrs["utc_offset_hours"] == 9.0
    assert len(df) > 10000 and 49000 < df["mag_raw"].median() < 52000
    assert abs(float(np.median(df["mag_ch2"] - df["mag_ch1"]))) > 20.0          # the pair's fixed offset
