"""Validates the whitespace-separated "HH MM SS X Y Z F" whole-day base
station format (e.g. Cheongyang base station exports named like
"cyg202607151s.txt": station prefix + YYYYMMDD date + "1s" for the
1-second sampling interval) alongside the pre-existing Korean-AM/PM CSV
format, via app.io_.base_loader's auto-detecting dispatcher."""
import datetime
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from app.io_.base_loader import BaseLoadError, load_base_csv, load_base_csvs

# A handful of lines in the real device's exact column layout (HH MM SS X Y
# Z F, space-separated, no header) - not the real 86400-line file, just
# enough rows to exercise the parser. F (last column) is an independently
# measured scalar total field, distinct from sqrt(X^2+Y^2+Z^2) - real
# fluxgate+scalar-sensor base stations report both and they don't exactly
# agree, which is expected and is why F is used directly rather than
# recomputed from X/Y/Z.
_HMS_XYZF_SAMPLE = (
    "00 00 00 29984.24 -4460.69 40570.81 50646.89\n"
    "00 00 01 29984.25 -4460.68 40570.83 50646.89\n"
    "00 00 02 29984.22 -4460.68 40570.80 50646.89\n"
    "23 59 59 29985.12 -4452.93 40564.14 50641.49\n"
)


def test_detects_hms_xyzf_format_and_uses_scalar_f_column():
    df = load_base_csv(io.BytesIO(_HMS_XYZF_SAMPLE.encode()), filename="cyg202607151s.txt")
    assert list(df.columns) == ["timestamp", "mag"]
    assert len(df) == 4
    # F column values used as-is, NOT recomputed from X/Y/Z (which would
    # differ by ~1.6 nT for this sample - see the module docstring)
    assert np.allclose(df["mag"].to_numpy(), [50646.89, 50646.89, 50646.89, 50641.49])
    computed_f = np.sqrt(29984.24**2 + 4460.69**2 + 40570.81**2)
    assert abs(computed_f - 50646.89) > 1.0  # confirms F is genuinely independent of X/Y/Z here


def test_extracts_date_from_filename():
    df = load_base_csv(io.BytesIO(_HMS_XYZF_SAMPLE.encode()), filename="cyg202607151s.txt")
    assert df["timestamp"].iloc[0] == pd_timestamp("2026-07-15 00:00:00")
    assert df["timestamp"].iloc[-1] == pd_timestamp("2026-07-15 23:59:59")
    assert df.attrs["date_fallback_used"] is False


def test_falls_back_to_todays_date_when_filename_has_no_date():
    df = load_base_csv(io.BytesIO(_HMS_XYZF_SAMPLE.encode()), filename="base_log.txt")
    assert df.attrs["date_fallback_used"] is True
    assert df["timestamp"].iloc[0].date() == datetime.date.today()


def test_falls_back_to_todays_date_when_no_filename_given():
    df = load_base_csv(io.BytesIO(_HMS_XYZF_SAMPLE.encode()))
    assert df.attrs["date_fallback_used"] is True


def test_existing_korean_ampm_csv_format_still_works():
    lines = [f"0,{59000.0 + i},0,오전 6:00:{i:02d},07/24/26,0" for i in range(5)]
    csv_bytes = ("\n".join(lines) + "\n").encode("utf-8-sig")
    df = load_base_csv(io.BytesIO(csv_bytes))
    assert len(df) == 5
    assert df["mag"].iloc[0] == pytest.approx(59000.0)


def test_load_base_csvs_combines_files_and_surfaces_date_fallback_flag():
    good = io.BytesIO(_HMS_XYZF_SAMPLE.encode())
    combined = load_base_csvs([good], filenames=["cyg202607151s.txt"])
    assert combined.attrs["date_fallback_used"] is False
    assert len(combined) == 4

    fallback = io.BytesIO(_HMS_XYZF_SAMPLE.encode())
    combined2 = load_base_csvs([fallback], filenames=["no_date_here.txt"])
    assert combined2.attrs["date_fallback_used"] is True


def test_detects_own_timestamp_mag_csv_export_format():
    """This app's own "주변 관측소 자료" export (store.py::export_
    nearest_intermagnet_csv) - a plain header'd (timestamp, mag_nT) CSV -
    must load back in cleanly through the standard base-upload path, not
    just the export's original intended re-import path, since users
    naturally expect a file they saved as a "base station replacement" to
    work as one."""
    lines = ["timestamp,mag_nT"] + [f"2026-07-24 06:00:{i:02d},{59000.0 + i}" for i in range(5)]
    csv_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    df = load_base_csv(io.BytesIO(csv_bytes))
    assert list(df.columns) == ["timestamp", "mag"]
    assert len(df) == 5
    assert df["mag"].iloc[0] == pytest.approx(59000.0)
    assert df["timestamp"].iloc[0] == pd_timestamp("2026-07-24 06:00:00")


def test_timestamp_mag_csv_format_is_case_and_column_order_insensitive():
    lines = ["Mag_nT,Timestamp"] + [f"{59000.0 + i},2026-07-24 06:00:{i:02d}" for i in range(3)]
    csv_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    df = load_base_csv(io.BytesIO(csv_bytes))
    assert len(df) == 3
    assert df["mag"].iloc[0] == pytest.approx(59000.0)


def test_unrecognized_format_raises_clear_error():
    with pytest.raises(BaseLoadError):
        load_base_csv(io.BytesIO(b"this is not a recognized base file format at all\n"))


def pd_timestamp(s):
    import pandas as pd

    return pd.Timestamp(s)


if __name__ == "__main__":
    test_detects_hms_xyzf_format_and_uses_scalar_f_column()
    test_extracts_date_from_filename()
    test_falls_back_to_todays_date_when_filename_has_no_date()
    test_falls_back_to_todays_date_when_no_filename_given()
    test_existing_korean_ampm_csv_format_still_works()
    test_load_base_csvs_combines_files_and_surfaces_date_fallback_flag()
    test_unrecognized_format_raises_clear_error()
    print("ALL CHECKS PASSED")
