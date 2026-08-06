"""Tests processing/intermagnet.py's "nearest observatories" base-station
estimation: probing a shortlist of candidate observatories for their real
(header-reported) coordinates, picking a directionally-spread subset, and
combining their data via inverse-distance weighting. requests.get is
mocked throughout (this sandbox's own network policy blocks the real BGS
GIN service - see intermagnet.py's module docstring); only synthetic,
format-accurate IAGA-2002 text is used, never real observatory data."""
import pathlib
import sys
from datetime import date
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from app.processing.intermagnet import (
    IagaObservatoryData,
    IntermagnetFetchError,
    _bearing_deg,
    _pick_direction_diverse,
    estimate_base_from_observatories,
    fill_missing_days,
    select_nearest_observatories,
)


def _make_iaga_text(code: str, lat: float, lon: float, base_f: float, n: int = 5) -> str:
    lines = [
        " Format                 IAGA-2002                                    |",
        f" Station Name           {code} Test Station                          |",
        f" IAGA CODE              {code}                                           |",
        f" Geodetic Latitude      {lat:.3f}                                        |",
        f" Geodetic Longitude     {lon:.3f}                                        |",
        " Elevation              100                                           |",
        " Reported                XYZF                                        |",
        f"DATE       TIME         DOY     {code}X      {code}Y      {code}Z      {code}F   |",
    ]
    for i in range(n):
        lines.append(f"2026-07-24 00:{i:02d}:00.000 205     20000.00  0.00      45000.00  {base_f + i:.2f}")
    return "\n".join(lines) + "\n"


# Roughly spread around a Korea-ish target (36.5N, 127.9E): CYG at the
# target itself, KAK to the east (Japan), BMT to the west (China), IRT to
# the north (Russia), GZH to the south (China, but far enough south).
_TARGET_LAT, _TARGET_LON = 36.5, 127.9
_MOCK_STATIONS = {
    "CYG": (36.37, 126.80, 50000.0),
    "KAK": (36.23, 140.19, 46000.0),
    "BMT": (40.30, 116.20, 54000.0),
    "IRT": (52.27, 104.45, 58000.0),
    "GZH": (23.09, 113.34, 42000.0),
}


def _fake_requests_get(url, params=None, timeout=None):
    code = (params or {}).get("observatoryIagaCode")
    resp = MagicMock()
    if code in _MOCK_STATIONS:
        lat, lon, base_f = _MOCK_STATIONS[code]
        resp.status_code = 200
        resp.text = _make_iaga_text(code, lat, lon, base_f)
    else:
        resp.status_code = 404
        resp.text = "not found"
    return resp


def test_bearing_deg_cardinal_directions():
    # due east and due north from the origin
    assert _bearing_deg(0.0, 0.0, 0.0, 10.0) == pytest.approx(90.0, abs=1.0)
    assert _bearing_deg(0.0, 0.0, 10.0, 0.0) == pytest.approx(0.0, abs=1.0)
    assert _bearing_deg(0.0, 0.0, 0.0, -10.0) == pytest.approx(270.0, abs=1.0)
    assert _bearing_deg(0.0, 0.0, -10.0, 0.0) == pytest.approx(180.0, abs=1.0)


def test_pick_direction_diverse_prefers_one_per_quadrant():
    def _entry(code, d, bearing):
        return (d, bearing, IagaObservatoryData(code, code, 0.0, 0.0, 0.0, "F", pd.DataFrame()))

    probed = [
        _entry("N1", 100, 10),  # north
        _entry("N2", 50, 20),  # also north, closer - should win over N1
        _entry("E1", 200, 100),  # east
        _entry("S1", 300, 190),  # south
        _entry("W1", 400, 280),  # west
    ]
    selected = _pick_direction_diverse(probed, n_stations=4)
    codes = {data.iaga_code for _, _, data in selected}
    assert codes == {"N2", "E1", "S1", "W1"}


def test_pick_direction_diverse_fills_remaining_slots_from_same_quadrant():
    def _entry(code, d, bearing):
        return (d, bearing, IagaObservatoryData(code, code, 0.0, 0.0, 0.0, "F", pd.DataFrame()))

    # only two distinct quadrants represented, but 3 stations requested
    probed = [_entry("N1", 100, 10), _entry("N2", 150, 20), _entry("E1", 200, 100)]
    selected = _pick_direction_diverse(probed, n_stations=3)
    assert len(selected) == 3
    codes = {data.iaga_code for _, _, data in selected}
    assert codes == {"N1", "N2", "E1"}


def test_select_nearest_observatories_picks_real_station_near_target():
    with patch("app.processing.intermagnet.requests.get", side_effect=_fake_requests_get):
        stations, estimated_dates_by_code = select_nearest_observatories(
            _TARGET_LAT, _TARGET_LON, date(2026, 7, 24), n_stations=4, max_candidates=60
        )

    assert 1 <= len(stations) <= 4
    codes = {s.iaga_code for s in stations}
    # CYG is essentially at the target - it must be the closest candidate
    # whenever it's within the probed shortlist.
    assert codes.issubset(set(_MOCK_STATIONS))
    # a single-day request (no end_date) never triggers day-filling
    assert estimated_dates_by_code == {}


def test_select_nearest_observatories_fills_missing_days_when_end_date_given():
    # Unlike _fake_requests_get's compact 5-point fixture (fine for the
    # coordinate-probing stage), fill_missing_days needs enough coverage
    # on the one day the station actually has to treat it as a valid
    # template source - so this mock returns a full day (1440 one-minute
    # points) for 2026-07-24, always, regardless of the requested
    # start_date/days - simulating a station whose feed only actually has
    # that one day published, out of the 3-day range requested here.
    def _make_full_day_text(code: str, lat: float, lon: float, base_f: float) -> str:
        lines = [
            " Format                 IAGA-2002                                    |",
            f" Station Name           {code} Test Station                          |",
            f" IAGA CODE              {code}                                           |",
            f" Geodetic Latitude      {lat:.3f}                                        |",
            f" Geodetic Longitude     {lon:.3f}                                        |",
            " Elevation              100                                           |",
            " Reported                XYZF                                        |",
            f"DATE       TIME         DOY     {code}X      {code}Y      {code}Z      {code}F   |",
        ]
        for i in range(24 * 60):
            hh, mm = divmod(i, 60)
            lines.append(
                f"2026-07-24 {hh:02d}:{mm:02d}:00.000 205     20000.00  0.00      45000.00  {base_f + i * 0.01:.2f}"
            )
        return "\n".join(lines) + "\n"

    def _fake_requests_get_full_day(url, params=None, timeout=None):
        code = (params or {}).get("observatoryIagaCode")
        resp = MagicMock()
        if code in _MOCK_STATIONS:
            lat, lon, base_f = _MOCK_STATIONS[code]
            resp.status_code = 200
            resp.text = _make_full_day_text(code, lat, lon, base_f)
        else:
            resp.status_code = 404
            resp.text = "not found"
        return resp

    with patch("app.processing.intermagnet.requests.get", side_effect=_fake_requests_get_full_day):
        stations, estimated_dates_by_code = select_nearest_observatories(
            _TARGET_LAT, _TARGET_LON, date(2026, 7, 23), end_date=date(2026, 7, 25), n_stations=2, max_candidates=60
        )

    assert 1 <= len(stations) <= 2
    for s in stations:
        assert s.iaga_code in estimated_dates_by_code
        assert set(estimated_dates_by_code[s.iaga_code]) == {"2026-07-23", "2026-07-25"}
        covered_days = {ts.date() for ts in s.df["timestamp"]}
        assert {date(2026, 7, 23), date(2026, 7, 24), date(2026, 7, 25)}.issubset(covered_days)


def test_select_nearest_observatories_raises_clear_error_when_nothing_found():
    def _all_404(url, params=None, timeout=None):
        resp = MagicMock(status_code=404, text="not found")
        return resp

    with patch("app.processing.intermagnet.requests.get", side_effect=_all_404):
        with pytest.raises(IntermagnetFetchError, match="관측소"):
            select_nearest_observatories(_TARGET_LAT, _TARGET_LON, date(2026, 7, 24), n_stations=4, max_candidates=10)


def test_estimate_base_from_observatories_weights_toward_closer_station():
    t = pd.date_range("2026-07-24", periods=5, freq="1min")
    near = IagaObservatoryData("NEAR", "NEAR", 36.5, 127.9, 0.0, "F", pd.DataFrame({"timestamp": t, "mag": 50000.0}))
    far = IagaObservatoryData("FAR", "FAR", 60.0, 100.0, 0.0, "F", pd.DataFrame({"timestamp": t, "mag": 40000.0}))

    out = estimate_base_from_observatories([near, far], 36.5, 127.9)
    assert len(out) == 5
    # target coincides exactly with "near" - IDW weight for it should
    # dominate heavily, pulling the estimate close to 50000, not the
    # midpoint (45000).
    assert out["mag"].mean() > 48000


def test_estimate_base_from_observatories_equal_distance_is_plain_average():
    t = pd.date_range("2026-07-24", periods=3, freq="1min")
    a = IagaObservatoryData("A", "A", 0.0, 1.0, 0.0, "F", pd.DataFrame({"timestamp": t, "mag": 100.0}))
    b = IagaObservatoryData("B", "B", 0.0, -1.0, 0.0, "F", pd.DataFrame({"timestamp": t, "mag": 200.0}))

    out = estimate_base_from_observatories([a, b], 0.0, 0.0)
    assert np.allclose(out["mag"].to_numpy(), 150.0)


def test_estimate_base_from_observatories_interpolates_small_gaps_not_extrapolate():
    t_a = pd.date_range("2026-07-24 00:00", periods=10, freq="1min")
    a_vals = np.full(10, 100.0)
    a_vals[4] = np.nan  # a single dropped sample in the middle
    a = IagaObservatoryData("A", "A", 0.0, 0.0, 0.0, "F", pd.DataFrame({"timestamp": t_a, "mag": a_vals}).dropna())

    out = estimate_base_from_observatories([a], 0.0, 0.0)
    # the single-sample gap at minute 4 should be bridged by interpolation
    assert t_a[4] in set(out["timestamp"])


def test_estimate_base_from_observatories_raises_on_empty_station_list():
    with pytest.raises(IntermagnetFetchError):
        estimate_base_from_observatories([], 0.0, 0.0)


def test_fill_missing_days_reconstructs_diurnal_shape_not_flat_line():
    # two known days sharing an identical simple diurnal-like pattern
    # (a smooth bump peaking around noon), with the day between them
    # entirely missing.
    minutes = np.arange(24 * 60)
    pattern = 100.0 + 10.0 * np.sin(np.pi * minutes / (24 * 60))
    day1 = pd.date_range("2026-07-23", periods=24 * 60, freq="1min")
    day3 = pd.date_range("2026-07-25", periods=24 * 60, freq="1min")
    df = pd.DataFrame({"timestamp": list(day1) + list(day3), "mag": np.concatenate([pattern, pattern])})

    filled, estimated_dates = fill_missing_days(df, date(2026, 7, 23), date(2026, 7, 25))

    assert estimated_dates == ["2026-07-24"]
    day2_vals = filled.loc[filled["timestamp"].dt.date == date(2026, 7, 24), "mag"].to_numpy()
    assert len(day2_vals) == 24 * 60
    # tracks the diurnal shape from the known days, not a flat/straight line
    assert np.allclose(day2_vals, pattern, atol=1.0)


def test_fill_missing_days_anchors_estimate_to_nearby_actual_level():
    minutes = np.arange(24 * 60)
    pattern = 100.0 + 5.0 * np.sin(np.pi * minutes / (24 * 60))
    day1 = pd.date_range("2026-07-23", periods=24 * 60, freq="1min")
    day2 = pd.date_range("2026-07-24", periods=24 * 60, freq="1min")
    # day2 is mostly missing (well under the 50% coverage threshold) but
    # has a few real samples, all shifted +20 above the template pattern -
    # the fill should track that offset, not the raw template level.
    sparse_idx = [0, 100, 200]
    df = pd.DataFrame(
        {
            "timestamp": list(day1) + [day2[i] for i in sparse_idx],
            "mag": np.concatenate([pattern, pattern[sparse_idx] + 20.0]),
        }
    )

    filled, estimated_dates = fill_missing_days(df, date(2026, 7, 23), date(2026, 7, 24))

    assert estimated_dates == ["2026-07-24"]
    day2_out = filled.set_index("timestamp")["mag"]
    far_minute = day2[500]  # far from any of the sparse real samples
    assert day2_out.loc[far_minute] == pytest.approx(pattern[500] + 20.0, abs=1.0)


def test_fill_missing_days_leaves_data_unchanged_when_no_good_day_available():
    # the only day present is itself too sparse to serve as a template
    # source, so there is nothing to estimate from - must not invent data.
    t = pd.date_range("2026-07-23", periods=5, freq="1min")
    df = pd.DataFrame({"timestamp": t, "mag": [1.0, 2.0, 3.0, 4.0, 5.0]})

    filled, estimated_dates = fill_missing_days(df, date(2026, 7, 23), date(2026, 7, 23))

    assert estimated_dates == []
    assert len(filled) == 5
    assert np.allclose(filled["mag"].to_numpy(), [1.0, 2.0, 3.0, 4.0, 5.0])


def test_fill_missing_days_handles_empty_dataframe():
    df = pd.DataFrame({"timestamp": pd.to_datetime([]), "mag": []})
    filled, estimated_dates = fill_missing_days(df, date(2026, 7, 23), date(2026, 7, 24))
    assert filled.empty
    assert estimated_dates == []
