"""Rows the logger wrote with a timestamp it had already used.

Just before a ~1 s dropout the MagArrow writes two or three rows carrying
the Counter and Time of rows 0.4 s earlier, while their position and field
are those of the moment they were really taken, about 8 m further along.
Seen five times on the 2026-09 HaeNam block (ACQU433, ACQU451).

The first-written occurrence is the genuine sample. Deciding which to keep
after an unstable sort made it arbitrary, and for two of the five the
re-stamped row was kept - a sample 8 m out of place in the profile.
"""
from __future__ import annotations

import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.io_.drone_loader import load_drone_csv, load_drone_csvs

HEADER = "Counter,Date,Time,Latitude,Longitude,Mag,MagValid\n"


def _row(counter, time, lat, mag):
    return f"{counter},2026-07-15,{time},{lat:.6f},126.355770,{mag:.3f},1\n"


def _restamped_file():
    rows = []
    # a steady track north, 10 Hz
    for i in range(20):
        rows.append(_row(100 + i, f"06:46:5{i // 10}.{i % 10}00", 34.589000 + i * 0.000006,
                         49840.0 + i * 0.25))
    # the glitch: the next three samples are stamped with the times of
    # samples 15-17 but were taken after sample 19
    for j, i in enumerate((15, 16, 17)):
        rows.append(_row(100 + i, f"06:46:5{i // 10}.{i % 10}00",
                         34.589000 + (20 + j) * 0.000006, 49840.0 + (20 + j) * 0.25))
    # then a dropout and the track resumes
    for i in range(33, 40):
        rows.append(_row(100 + i, f"06:46:5{i // 10}.{i % 10}00", 34.589000 + i * 0.000006,
                         49840.0 + i * 0.25))
    return io.StringIO(HEADER + "".join(rows))


def test_the_genuine_sample_is_kept_not_the_restamped_one():
    df = load_drone_csv(_restamped_file())

    for i in (15, 16, 17):
        t = f"2026-07-15 06:46:5{i // 10}.{i % 10}00"
        kept = df.loc[df["timestamp"].astype(str).str.startswith(t[:21]), "lat"]
        assert len(kept) == 1
        assert abs(float(kept.iloc[0]) - (34.589000 + i * 0.000006)) < 1e-7, t


def test_the_track_comes_out_monotonic():
    df = load_drone_csv(_restamped_file())

    assert (df["lat"].diff().dropna() > 0).all()
    assert (df["timestamp"].diff().dropna().dt.total_seconds() > 0).all()


def test_restamped_rows_are_counted_with_the_duplicates():
    combined = load_drone_csvs([_restamped_file()])

    assert combined.attrs["n_duplicate_timestamps_removed"] == 3
