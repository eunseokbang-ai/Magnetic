"""Building a base from observatories over a connection that is not
perfect.

A briefly heavier fetch (every calendar day of the project, sequentially,
all-or-nothing per station) turned a 40-request job into a 228-request one
on the HaeNam survey. On a slow connection that took a long time and ended
with two stations instead of four, both partly estimated. What has to hold
instead: transient failures are retried, a station that cannot supply
every flight date is replaced rather than lost, what was downloaded is
kept, and the operator is told what was dropped and why.
"""
from __future__ import annotations

import pathlib
import sys
from datetime import date
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
import requests

from app.processing import intermagnet
from app.processing.intermagnet import fetch_iaga2002_text, select_nearest_observatories

TARGET = (36.5, 127.9)
DATES = [date(2026, 9, 7), date(2026, 9, 8), date(2026, 9, 9)]

# code: (lat, lon, base F) - CYG and SEO both north-west-ish of the target
# in the same quadrant, CYG nearer; KAK east.
STATIONS = {
    "CYG": (36.37, 126.80, 50000.0),
    "SEO": (37.50, 126.90, 50100.0),
    "KAK": (36.23, 140.19, 46000.0),
}


def _day_text(code: str, d: date) -> str:
    lat, lon, f = STATIONS[code]
    head = [
        " Format                 IAGA-2002                                    |",
        f" Station Name           {code} Test                                  |",
        f" IAGA CODE              {code}                                           |",
        f" Geodetic Latitude      {lat:.3f}                                        |",
        f" Geodetic Longitude     {lon:.3f}                                        |",
        " Elevation              100                                           |",
        " Reported               F                                             |",
        f"DATE       TIME         DOY     {code}F   |",
    ]
    rows = [f"{d.isoformat()} {m // 60:02d}:{m % 60:02d}:00.000 001     {f + (m % 60) * 0.01:.2f}"
            for m in range(1440)]
    return "\n".join(head + rows) + "\n"


class FakeGin:
    """A GIN service with per-(station, date) outages and a request log."""

    def __init__(self, missing=(), flaky=()):
        self.missing = set(missing)      # (code, date) that never exist -> 404
        self.flaky = dict(flaky)         # (code, date) -> timeouts before success
        self.calls: list[tuple[str, date]] = []

    def get(self, url, params=None, timeout=None, **_kwargs):
        code = params["observatoryIagaCode"]
        d = date.fromisoformat(params["dataStartDate"][:10])
        self.calls.append((code, d))
        if self.flaky.get((code, d), 0) > 0:
            self.flaky[(code, d)] -= 1
            raise requests.Timeout("simulated timeout")
        resp = MagicMock()
        if code not in STATIONS or (code, d) in self.missing:
            resp.status_code, resp.text = 404, "not found"
        else:
            resp.status_code, resp.text = 200, _day_text(code, d)
        return resp


@pytest.fixture(autouse=True)
def _only_our_candidates(monkeypatch):
    """Keep the candidate list to the three fake stations."""
    from app.processing.intermagnet_stations import ObservatoryRosterEntry

    entries = [ObservatoryRosterEntry(iaga_code=c, name=c, country="Republic of Korea") for c in STATIONS]
    monkeypatch.setattr(intermagnet, "_candidates_by_rough_distance", lambda lat, lon, limit: entries)


def _select(gin, report=None, n=2):
    with patch("app.processing.intermagnet.requests.get", side_effect=gin.get):
        return select_nearest_observatories(*TARGET, DATES, n_stations=n, report=report)


def test_a_timeout_is_retried_rather_than_treated_as_missing_data():
    gin = FakeGin(flaky={("CYG", DATES[0]): 2})     # two timeouts, then fine

    with patch("app.processing.intermagnet.requests.get", side_effect=gin.get):
        text = fetch_iaga2002_text("CYG", DATES[0], days=1)

    assert "CYGF" in text
    assert gin.calls.count(("CYG", DATES[0])) == 3


def test_missing_data_is_not_retried():
    gin = FakeGin(missing={("CYG", DATES[0])})

    with patch("app.processing.intermagnet.requests.get", side_effect=gin.get):
        with pytest.raises(intermagnet.IntermagnetFetchError):
            fetch_iaga2002_text("CYG", DATES[0], days=1)

    assert gin.calls.count(("CYG", DATES[0])) == 1


def test_a_station_missing_a_flight_date_is_replaced_by_the_next_in_its_direction():
    """CYG is nearest in its quadrant but has no data for the middle
    flight date, and neither do its neighbours' days help (all of 9/7-9/9
    around it are asked for; 9/8 alone is missing, so it gets estimated -
    make it missing on every day but the probe day to force a drop)."""
    gin = FakeGin(missing={("CYG", DATES[1]), ("CYG", DATES[2]),
                           ("CYG", date(2026, 9, 10)), ("CYG", date(2026, 9, 6))})
    report: dict = {}

    stations, _ = _select(gin, report)

    codes = {s.iaga_code for s in stations}
    assert "CYG" not in codes
    assert codes == {"SEO", "KAK"}
    dropped = {d["iaga_code"]: d["reason"] for d in report["dropped"]}
    assert "CYG" in dropped


def test_every_chosen_station_covers_every_flight_date():
    gin = FakeGin(missing={("KAK", DATES[2]), ("KAK", date(2026, 9, 10)), ("KAK", DATES[1])})

    stations, _ = _select(gin, n=2)

    for s in stations:
        days = {t.date() for t in s.df["timestamp"]}
        assert set(DATES) <= days, s.iaga_code


def test_a_second_run_is_served_from_the_cache():
    first = FakeGin()
    _select(first)
    assert first.calls

    second = FakeGin()
    _select(second)

    assert second.calls == [], f"re-downloaded {len(second.calls)} days"


def test_a_run_cut_short_keeps_what_it_got():
    """A day that failed is asked for again next time; the days that
    arrived are not."""
    first = FakeGin(flaky={("SEO", DATES[1]): 99})    # SEO 9/8 never arrives
    _select(first)

    second = FakeGin()
    _select(second)

    assert ("CYG", DATES[0]) not in second.calls
    assert ("SEO", DATES[1]) in second.calls


def test_only_flight_dates_are_requested_for_the_selected_stations():
    """Not the weeks in between - the survey here is three consecutive
    days, so nothing outside them (and the probe) should be asked for."""
    gin = FakeGin()

    _select(gin)

    assert {d for _, d in gin.calls} <= set(DATES)


def test_choosing_stations_reads_only_the_header_of_a_day():
    """Positions sit in the first ~2 KB; the minute data after them is
    ~100 KB. The probe must stop reading at the column-heading line."""
    text = _day_text("KAK", DATES[0]).encode()
    header_end = text.index(b"\nDATE")
    served = {"bytes": 0}

    def chunks(size):
        for i in range(0, len(text), size):
            served["bytes"] += min(size, len(text) - i)
            yield text[i:i + size]

    resp = MagicMock()
    resp.status_code = 200
    resp.iter_content.side_effect = chunks
    with patch("app.processing.intermagnet.requests.get", return_value=resp):
        station = intermagnet.fetch_observatory_header("KAK", DATES[0])

    assert station is not None and abs(station.lat - 36.23) < 1e-6
    assert served["bytes"] < header_end + 4096
    resp.close.assert_called()


def test_a_known_station_is_not_asked_where_it_is_again():
    first = FakeGin()
    with patch("app.processing.intermagnet.requests.get", side_effect=first.get):
        intermagnet.fetch_observatory_header("KAK", DATES[0])
    assert first.calls

    second = FakeGin()
    with patch("app.processing.intermagnet.requests.get", side_effect=second.get):
        station = intermagnet.fetch_observatory_header("KAK", DATES[1])

    assert station is not None and station.iaga_code == "KAK"
    assert second.calls == []
