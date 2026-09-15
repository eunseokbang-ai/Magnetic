"""Keeping the blended base series continuous when stations come and go.

Observatory data is fetched one calendar day at a time per station, so a
station's coverage starts and stops at midnight and the set contributing
to the blend can only change there. If the stations are not on a common
level when that happens, the blend steps - and the step lands in the
survey one for one, because the diurnal correction's reference cancels
out of any difference between two times.

The 2026-09 HaeNam base file had exactly that: a 20.86 nT jump at
2026-09-09 00:00:00 and 26.34 nT at 2026-08-03 00:00:00, against 1-minute
changes with a median of 0.13 nT. The first of those measured as a
20.86 nT level step between the survey blocks flown either side of that
midnight.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from app.processing.intermagnet import IagaObservatoryData, estimate_base_from_observatories

TARGET_LAT, TARGET_LON = 34.55, 126.55


def _station(code, lat, lon, start, end, level, diurnal_amplitude, phase_hours=0.0, seed=0):
    """One observatory: a daily Sq curve on its own absolute level."""
    t = pd.date_range(start, end, freq="1min", inclusive="left")
    hours = t.hour + t.minute / 60.0
    sq = diurnal_amplitude * np.sin(2 * np.pi * (hours - phase_hours) / 24.0)
    noise = np.random.default_rng(seed).normal(0.0, 0.05, len(t))
    return IagaObservatoryData(
        station_name=code, iaga_code=code, lat=lat, lon=lon, elevation_m=0.0,
        reported="F", df=pd.DataFrame({"timestamp": t, "mag": level + sq + noise}),
    )


def _largest_one_minute_jump(base: pd.DataFrame) -> tuple[float, pd.Timestamp]:
    d = base["mag"].diff().abs()
    k = int(d.iloc[1:].idxmax())
    return float(d.iloc[k]), base["timestamp"].iloc[k]


def test_a_station_dropping_out_at_midnight_does_not_step_the_blend():
    """The real failure, reproduced: the near station covers only the
    first day, so from midnight the blend is the far station alone. Their
    diurnal curves differ, so without a common level the handover is a
    step."""
    near = _station("NEA", 34.6, 126.6, "2026-09-08", "2026-09-09", 50130.0, 20.0)
    far = _station("FAR", 36.0, 128.0, "2026-09-08", "2026-09-10", 49800.0, 32.0,
                   phase_hours=1.5, seed=1)

    base = estimate_base_from_observatories([near, far], TARGET_LAT, TARGET_LON)

    jump, when = _largest_one_minute_jump(base)
    assert jump < 2.0, f"blend steps {jump:.2f} nT at {when}"


def test_the_handover_still_tracks_the_remaining_station():
    """Continuity must not come from flattening the series - after the
    near station goes, the blend has to follow the far one's variation."""
    near = _station("NEA", 34.6, 126.6, "2026-09-08", "2026-09-09", 50130.0, 20.0)
    far = _station("FAR", 36.0, 128.0, "2026-09-08", "2026-09-10", 49800.0, 32.0,
                   phase_hours=1.5, seed=1)

    base = estimate_base_from_observatories([near, far], TARGET_LAT, TARGET_LON)

    after = base[base["timestamp"] >= "2026-09-09"]
    far_after = far.df[far.df["timestamp"] >= "2026-09-09"]
    merged = after.merge(far_after, on="timestamp", suffixes=("_blend", "_far"))
    # same shape, just shifted onto the blend's level
    assert np.corrcoef(merged["mag_blend"], merged["mag_far"])[0, 1] > 0.999


def test_three_stations_handing_over_one_after_another():
    """A station that never overlaps the anchor is levelled through the
    one that does, rather than being left on its own mean."""
    a = _station("AAA", 34.6, 126.6, "2026-09-08", "2026-09-09", 50130.0, 20.0)
    b = _station("BBB", 35.2, 127.1, "2026-09-08", "2026-09-10", 49950.0, 26.0,
                 phase_hours=0.8, seed=2)
    c = _station("CCC", 36.4, 128.4, "2026-09-09", "2026-09-11", 49700.0, 34.0,
                 phase_hours=2.0, seed=3)

    base = estimate_base_from_observatories([a, b, c], TARGET_LAT, TARGET_LON)

    jump, when = _largest_one_minute_jump(base)
    assert jump < 2.0, f"blend steps {jump:.2f} nT at {when}"


def test_the_blend_sits_at_the_nearest_stations_own_level():
    """Absolute level means nothing to the diurnal correction, which only
    uses (base - reference) - but the series should still read as a
    plausible field strength rather than an average of stations hundreds
    of kilometres apart."""
    near = _station("NEA", 34.6, 126.6, "2026-09-08", "2026-09-10", 50130.0, 20.0)
    far = _station("FAR", 36.0, 128.0, "2026-09-08", "2026-09-10", 49800.0, 32.0, seed=4)

    base = estimate_base_from_observatories([near, far], TARGET_LAT, TARGET_LON)

    assert abs(base["mag"].mean() - 50130.0) < 5.0


def test_a_station_sharing_no_minute_with_the_others_is_still_usable():
    """Nothing to measure a level against, so it keeps its own mean - the
    old behaviour, and the best available. It must not crash or drop the
    station."""
    a = _station("AAA", 34.6, 126.6, "2026-09-08", "2026-09-09", 50130.0, 20.0)
    b = _station("BBB", 35.2, 127.1, "2026-09-10", "2026-09-11", 49950.0, 26.0, seed=5)

    base = estimate_base_from_observatories([a, b], TARGET_LAT, TARGET_LON)

    assert base["timestamp"].min() < pd.Timestamp("2026-09-09")
    assert base["timestamp"].max() > pd.Timestamp("2026-09-10")


def test_one_station_alone_is_unchanged_by_any_of_this():
    only = _station("ONE", 34.6, 126.6, "2026-09-08", "2026-09-09", 50130.0, 20.0)

    base = estimate_base_from_observatories([only], TARGET_LAT, TARGET_LON)

    merged = base.merge(only.df, on="timestamp", suffixes=("_blend", "_raw"))
    assert np.allclose(merged["mag_blend"], merged["mag_raw"], atol=1e-6)


def test_no_stations_is_an_error_not_an_empty_series():
    with pytest.raises(Exception):
        estimate_base_from_observatories([], TARGET_LAT, TARGET_LON)
