"""Tests for the multi-device format auto-detection in io_/drone_loader.py."""
from __future__ import annotations

import io

import numpy as np
import pytest

from app.io_.drone_loader import DroneLoadError, _detect_format, load_drone_csv, load_drone_csvs


def _buf(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode("utf-8"))


def test_detect_generic():
    text = "Date,Time,Latitude,Longitude,Mag\n2024-01-01,00:00:00,37.0,127.0,50000\n"
    assert _detect_format(text) == "generic"


def test_generic_still_works():
    rows = ["Date,Time,Latitude,Longitude,Mag"]
    for i in range(20):
        rows.append(f"2024-01-01,00:00:{i:02d},37.{i:04d},127.{i:04d},{50000 + i}")
    df = load_drone_csv(_buf("\n".join(rows)))
    assert len(df) == 20
    assert df.attrs["source_format"] == "generic"
    assert np.isclose(df["mag_raw"].iloc[0], 50000)


def test_sensys_r1_format():
    header = "Date,Time,Latitude,Longitude,Heading,Next WP,Altitude,Altitude AGL,TMI,Bx,By,Bz"
    rows = [header]
    for i in range(20):
        rows.append(
            f"2024/09/18,07:47:{17+i}.000,56.8632{i:04d},24.1118{i:04d},125.5,,120.0,50.0,"
            f"{50000 + i},47956.8,-13192.4,7945.2"
        )
    text = "\n".join(rows)
    assert _detect_format(text) == "sensys_r1"
    df = load_drone_csv(_buf(text))
    assert len(df) == 20
    assert df.attrs["source_format"] == "sensys_r1"
    assert np.isclose(df["mag_raw"].iloc[0], 50000)
    assert np.isclose(df["lat"].iloc[0], 56.8632)


def test_microinfinity_format():
    header = (
        "LaserCurr_J1[mA],LaserCurr_J2[mA],LaserTemp_J1[],LaserTemp_J2[],ModFreq_J1[KHz],ModFreq_J2[KHz],"
        "ModAmp_J1[mApp],ModAmp_J2[mApp],HTPWR_J1[%],HTPWR_J2[%],CellTemp_J1[Ohm],CellTemp_J2[Ohm],"
        "MagField_J1[uT],MagField_J2[uT],SeqNum,TOW,lat[deg],lon[deg],alt[m],Roll[s],pitch[uT],yaw[s]"
    )
    rows = [header]
    for i in range(20):
        tow = 100568.0 + i * 0.1
        rows.append(
            f"83.8,67.0,60.45,18.29,354.6,353.9,45.00,23.00,63,60,1350.3,1350.2,"
            f"50.66{i:03d},50.56{i:03d},{46699+i},{tow:.3f},36.5111{i:04d},127.1255{i:04d},41.17,1.31,2.92,-0.77"
        )
    text = "\n".join(rows)
    assert _detect_format(text) == "microinfinity"
    df = load_drone_csv(_buf(text))
    assert len(df) == 20
    assert df.attrs["source_format"] == "microinfinity"
    # channel 1 (MagField_J1, uT) converted to nT
    assert df["mag_raw"].iloc[0] > 40000


def test_sensys_r3_raw_format():
    header_block = [
        "20210603_132857_MD-R3_#0058",
        "Field-Nr.: 1",
        "Date: 03.06.2021",
        "Time: 13:45:42",
        "MagDroneR3: 000058",
        "Samples: 200",
        "###---Probe---###",
        "SampleFrequence: 200 [Hz]",
        "###---GPS---###",
        "Timestamp [ms]; B1x [nT]; B1y [nT]; B1z [nT]; B2x [nT]; B2y [nT]; B2z [nT]; AccX [g]; AccY [g]; "
        "AccZ [g]; Temp [Deg]; Latitude [Decimal Degrees]; Longitude [Decimal Degrees]; Altitude [m]; "
        "Satellites; Quality; GPSTime; GPSDate; GPSTime [hh:mm:ss.sss];",
    ]
    data_rows = []
    for i in range(40):
        ms = i * 5
        if i % 20 == 0:
            lat, lon, alt = 52.31 + i * 0.0001, 14.08 + i * 0.0001, 54.2
        else:
            lat, lon, alt = 0, 0, 0
        data_rows.append(
            f"{ms};-2047.18;-18888.29;46071.94;-2973.07;-18510.65;46032.43;-0.003;0.062;1.020;38.0;"
            f"{lat};{lon};{alt};12;12;132859.4;2021/06/03;13:28:59.406;"
        )
    text = "\n".join(header_block + data_rows)
    assert _detect_format(text) == "sensys_r3_raw"
    df = load_drone_csv(_buf(text))
    assert len(df) == 40
    assert df.attrs["source_format"] == "sensys_r3_raw"
    # sqrt(B1x^2+B1y^2+B1z^2) with the fixed sample values above
    expected = np.sqrt(2047.18**2 + 18888.29**2 + 46071.94**2)
    assert np.isclose(df["mag_raw"].iloc[0], expected, rtol=1e-3)
    # interpolated lat/lon should be finite everywhere despite sparse GPS updates
    assert df["lat"].notna().all()
    assert df["lon"].notna().all()


def test_sensys_r3_asc_format():
    header = "Timestamp [ms]  Sensor ID  Latitude [°]  Longitude [°]  Total field anomaly [nT]  Mag-X [nT]  Mag-Y [nT]  Mag-Z [nT]"
    rows = [header]
    for i in range(20):
        ts = 26817194 + i * 5
        rows.append(f"{ts}  1  36.2958{i:04d}  126.9075{i:04d}  50669.42  33733.28  2134.01  37747.87")
    for i in range(20):
        ts = 26817194 + i * 5
        rows.append(f"{ts}  2  36.2958{i:04d}  126.9076{i:04d}  50652.84  34686.00  2086.45  36854.28")
    text = "\n".join(rows)
    assert _detect_format(text) == "sensys_r3_asc"
    df = load_drone_csv(_buf(text))
    # only the first Sensor ID (1) block should be kept
    assert len(df) == 20
    assert df.attrs["source_format"] == "sensys_r3_asc"
    assert np.isclose(df["mag_raw"].iloc[0], 50669.42)


def test_generic_format_extracts_gyro_accel_when_present():
    header = "Date,Time,Latitude,Longitude,Mag,GyroscopeX,GyroscopeY,GyroscopeZ,AccelerometerX,AccelerometerY,AccelerometerZ"
    rows = [header]
    for i in range(20):
        rows.append(f"2024-01-01,00:00:{i:02d},37.{i:04d},127.{i:04d},{50000 + i},0.1,-0.2,0.3,-0.01,0.02,1.0")
    df = load_drone_csv(_buf("\n".join(rows)))
    assert len(df) == 20
    expected_gyro = np.sqrt(0.1**2 + 0.2**2 + 0.3**2)
    expected_accel = np.sqrt(0.01**2 + 0.02**2)
    assert np.allclose(df["gyro_mag"], expected_gyro)
    assert np.allclose(df["accel_horiz_g"], expected_accel)


def test_generic_format_without_gyro_accel_columns_is_nan():
    # test_generic_still_works's fixture has no Gyroscope*/Accelerometer* columns
    rows = ["Date,Time,Latitude,Longitude,Mag"]
    for i in range(20):
        rows.append(f"2024-01-01,00:00:{i:02d},37.{i:04d},127.{i:04d},{50000 + i}")
    df = load_drone_csv(_buf("\n".join(rows)))
    assert df["gyro_mag"].isna().all()
    assert df["accel_horiz_g"].isna().all()


def test_unrecognized_generic_missing_columns_raises():
    text = "foo,bar\n1,2\n"
    with pytest.raises(DroneLoadError):
        load_drone_csv(_buf(text))


def test_generic_prefers_magcomp_over_mag_when_present():
    """A "-comp.csv" written with "Keep Raw Data" on (both the vendor tool
    and the companion MagArrow-heading-error-calibration tool do this)
    keeps "Mag" as the untouched original and puts the corrected value in
    "MagComp" - the loader must use the corrected one."""
    rows = ["Date,Time,Latitude,Longitude,Mag,MagComp"]
    for i in range(20):
        rows.append(f"2024-01-01,00:00:{i:02d},37.{i:04d},127.{i:04d},{50000 + i},{49000 + i}")
    df = load_drone_csv(_buf("\n".join(rows)))
    assert np.isclose(df["mag_raw"].iloc[0], 49000)
    assert df.attrs["mag_source_column"] == "MagComp"


def test_generic_falls_back_to_mag_when_no_magcomp_column():
    rows = ["Date,Time,Latitude,Longitude,Mag"]
    for i in range(20):
        rows.append(f"2024-01-01,00:00:{i:02d},37.{i:04d},127.{i:04d},{50000 + i}")
    df = load_drone_csv(_buf("\n".join(rows)))
    assert np.isclose(df["mag_raw"].iloc[0], 50000)
    assert df.attrs["mag_source_column"] == "Mag"


def _generic_text(mag_base, with_magcomp=False, comp_base=None, n=20):
    header = "Date,Time,Latitude,Longitude,Mag" + (",MagComp" if with_magcomp else "")
    rows = [header]
    for i in range(n):
        line = f"2024-01-01,00:00:{i:02d},37.{i:04d},127.{i:04d},{mag_base + i}"
        if with_magcomp:
            line += f",{comp_base + i}"
        rows.append(line)
    return "\n".join(rows)


def test_load_drone_csvs_tracks_precompensated_mag_per_file():
    """A batch upload mixing a -comp.csv (has MagComp) with a plain -pre.csv
    (no MagComp) must report accurately which is which, not just whether
    *any* file had it - see store.py's drone_summary."""
    buf_with = _buf(_generic_text(50000, with_magcomp=True, comp_base=49000))
    buf_without = _buf(_generic_text(60000, with_magcomp=False))
    combined = load_drone_csvs([buf_with, buf_without])

    assert combined.attrs["n_files_total"] == 2
    assert combined.attrs["n_files_using_precompensated_mag"] == 1

    per_file = combined.groupby("source_file_index")["used_precompensated_mag"].first()
    assert bool(per_file[0]) is True
    assert bool(per_file[1]) is False

    # the file that had MagComp must actually have used it, not raw Mag
    file0 = combined[combined["source_file_index"] == 0]
    assert file0["mag_raw"].min() < 50000  # in the 49000s, not 50000s


def test_load_drone_csvs_all_files_using_precompensated_mag():
    buf_a = _buf(_generic_text(50000, with_magcomp=True, comp_base=49000))
    buf_b = _buf(_generic_text(60000, with_magcomp=True, comp_base=59000))
    combined = load_drone_csvs([buf_a, buf_b])
    assert combined.attrs["n_files_using_precompensated_mag"] == 2
    assert combined.attrs["n_files_total"] == 2
