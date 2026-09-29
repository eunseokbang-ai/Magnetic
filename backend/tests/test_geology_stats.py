"""Joining the readings to the geology map.

No network here: the polygons are a hand-made FeatureCollection in the
shape the KIGAM WFS returns (measured attributes: lithoidx, lithoname,
age, mapname), so what is tested is the join and the arithmetic, not
the service.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from app.processing import geology_stats as gs


def _square(lon0, lat0, size, **props):
    ring = [[lon0, lat0], [lon0 + size, lat0], [lon0 + size, lat0 + size], [lon0, lat0 + size], [lon0, lat0]]
    return {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [[ring]]},
            "properties": props}


@pytest.fixture
def two_units():
    return {"type": "FeatureCollection", "features": [
        _square(126.50, 34.30, 0.05, lithoidx="Kj", lithoname="화산력 응회암", age="백악기", mapname="남창"),
        _square(126.55, 34.30, 0.05, lithoidx="Kg", lithoname="흑운모 화강암", age="백악기", mapname="남창"),
        # the same unit as the first, as a second polygon - one unit is
        # usually several polygons and they must be summed, not listed twice
        _square(126.50, 34.35, 0.05, lithoidx="Kj", lithoname="화산력 응회암", age="백악기", mapname="남창"),
    ]}


def _readings():
    rng = np.random.default_rng(0)
    lon = np.concatenate([rng.uniform(126.51, 126.54, 300), rng.uniform(126.56, 126.59, 200), rng.uniform(126.51, 126.54, 100)])
    lat = np.concatenate([rng.uniform(34.31, 34.34, 300), rng.uniform(34.31, 34.34, 200), rng.uniform(34.36, 34.39, 100)])
    values = np.concatenate([rng.normal(50.0, 5.0, 300), rng.normal(-20.0, 3.0, 200), rng.normal(50.0, 5.0, 100)])
    line = np.concatenate([np.repeat([0, 1, 2], 100), np.repeat([3, 4], 100), np.repeat([5], 100)])
    return lon, lat, values, line


def test_each_reading_lands_in_the_polygon_it_is_over(two_units):
    lon, lat, _, _ = _readings()
    unit, attrs = gs.assign_units(lon, lat, two_units)

    assert (unit[:300] == 0).all()
    assert (unit[300:500] == 1).all()
    assert (unit[500:] == 2).all()
    assert attrs[0]["symbol"] == "Kj" and attrs[1]["name"] == "흑운모 화강암"


def test_a_reading_over_no_polygon_is_outside_not_misassigned(two_units):
    unit, _ = gs.assign_units(np.array([127.0]), np.array([35.0]), two_units)
    assert unit[0] == -1


def test_units_are_merged_by_symbol_and_the_field_summarised(two_units):
    lon, lat, values, line = _readings()

    units, report = gs.summarize_by_unit(lon, lat, values, line, two_units)

    assert [u.symbol for u in units] == ["Kj", "Kg"]           # strongest count first
    kj, kg = units
    assert kj.n_points == 400 and kj.n_lines == 4               # both polygons, summed
    assert abs(kj.mean_nt - 50.0) < 1.5 and abs(kg.mean_nt - (-20.0)) < 1.0
    assert kj.p10_nt < kj.median_nt < kj.p90_nt
    assert abs(kj.area_share_pct - 400 / 6) < 0.1
    assert report["n_units"] == 2 and report["n_polygons"] == 3 and report["n_outside"] == 0
    assert report["sheets"] == ["남창"]


def test_the_signal_column_follows_the_same_join(two_units):
    lon, lat, values, line = _readings()
    signal = np.where(np.arange(600) < 300, 1.0, 0.1)

    units, _ = gs.summarize_by_unit(lon, lat, values, line, two_units, signal=signal)

    assert abs(units[0].mean_signal - (300 * 1.0 + 100 * 0.1) / 400) < 1e-9
    assert abs(units[1].mean_signal - 0.1) < 1e-9


def test_nan_readings_are_left_out_of_the_statistics(two_units):
    lon, lat, values, line = _readings()
    values = values.copy()
    values[:50] = np.nan

    units, report = gs.summarize_by_unit(lon, lat, values, line, two_units)

    assert report["n_points"] == 550
    assert units[0].n_points == 350


def test_the_map_layer_carries_only_what_the_map_needs(two_units):
    out = gs.units_geojson(two_units)
    assert len(out["features"]) == 3
    assert set(out["features"][0]["properties"]) == {"symbol", "name", "age"}


def test_the_table_exports_with_korean_headings(two_units):
    lon, lat, values, line = _readings()
    units, _ = gs.summarize_by_unit(lon, lat, values, line, two_units)

    text = gs.stats_to_csv(units).decode("utf-8-sig")

    assert text.splitlines()[0].startswith("기호,암상,시대,측점수")
    assert "Kj" in text and "화산력 응회암" in text


def test_the_service_request_is_shaped_the_way_the_server_accepts(monkeypatch, tmp_path):
    """Measured against the live service: a plain lon,lat BBOX with the
    CRS named returned the sheet, lat,lon returned nothing, and the
    request has to carry a browser User-Agent or it is refused."""
    monkeypatch.setattr(gs.paths, "data_dir", lambda: tmp_path)
    seen = {}

    class Response:
        status_code = 200

        def json(self):
            return {"type": "FeatureCollection", "features": []}

    def fake_get(url, timeout, headers):
        seen["url"] = url
        seen["ua"] = headers["User-Agent"]
        return Response()

    monkeypatch.setattr(gs.requests, "get", fake_get)

    gs.fetch_lithology(34.30, 126.50, 34.40, 126.60, scale="50k")

    assert "BBOX=126.5%2C34.3%2C126.6%2C34.4%2CEPSG%3A4326" in seen["url"]
    assert "l_50k_geology_litho_view_latest" in seen["url"]
    assert "Mozilla" in seen["ua"]
    # and the second call is served from disk, not the service
    seen.clear()
    gs.fetch_lithology(34.30, 126.50, 34.40, 126.60, scale="50k")
    assert seen == {}
