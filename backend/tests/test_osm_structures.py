"""Tests for the OpenStreetMap structure-footprint fetch/buffer module
that feeds the structure-distortion auto-scan feature: mocked Overpass
responses only, since a live outbound request to a third-party public
service (see osm_structures.py's fetch_osm_structures docstring) is not
something a unit test should depend on.
"""
import pathlib
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from app.processing.osm_structures import (
    OsmFetchError,
    OsmStructures,
    buffer_structures_to_polygons,
    fetch_osm_structures,
)

_SAMPLE_OVERPASS_RESPONSE = {
    "elements": [
        {
            "type": "way",
            "tags": {"building": "yes"},
            "geometry": [
                {"lat": 46.4840, "lon": 106.2730},
                {"lat": 46.4840, "lon": 106.2732},
                {"lat": 46.4842, "lon": 106.2732},
                {"lat": 46.4842, "lon": 106.2730},
                {"lat": 46.4840, "lon": 106.2730},
            ],
        },
        {
            "type": "way",
            "tags": {"highway": "residential"},
            "geometry": [
                {"lat": 46.4850, "lon": 106.2725},
                {"lat": 46.4851, "lon": 106.2728},
            ],
        },
        # a way with no relevant tags - must be ignored
        {"type": "way", "tags": {"landuse": "farmland"}, "geometry": [{"lat": 46.49, "lon": 106.28}, {"lat": 46.491, "lon": 106.281}]},
        # a node - must be ignored (only "way" elements are handled)
        {"type": "node", "tags": {"building": "yes"}, "lat": 46.49, "lon": 106.28},
    ]
}


def _mock_post(status_code=200, json_body=None):
    resp = MagicMock(status_code=status_code)
    resp.json.return_value = json_body if json_body is not None else _SAMPLE_OVERPASS_RESPONSE
    return MagicMock(return_value=resp)


def test_fetch_osm_structures_parses_buildings_and_roads():
    with patch("app.processing.osm_structures.requests.post", _mock_post()):
        result = fetch_osm_structures(46.48, 106.27, 46.49, 106.28)
    assert len(result.buildings) == 1
    assert len(result.roads) == 1
    assert result.buildings[0][0] == [46.4840, 106.2730]


def test_fetch_osm_structures_rejects_invalid_bbox():
    with pytest.raises(OsmFetchError):
        fetch_osm_structures(46.49, 106.27, 46.48, 106.28)  # south > north


def test_fetch_osm_structures_rejects_oversized_bbox():
    with pytest.raises(OsmFetchError):
        fetch_osm_structures(0.0, 0.0, 10.0, 10.0)  # far larger than the sanity cap


def test_fetch_osm_structures_raises_clear_error_on_http_failure():
    with patch("app.processing.osm_structures.requests.post", _mock_post(status_code=504)):
        with pytest.raises(OsmFetchError):
            fetch_osm_structures(46.48, 106.27, 46.49, 106.28)


def test_fetch_osm_structures_raises_on_network_error():
    import requests

    with patch("app.processing.osm_structures.requests.post", side_effect=requests.ConnectionError("blocked")):
        with pytest.raises(OsmFetchError):
            fetch_osm_structures(46.48, 106.27, 46.49, 106.28)


def test_buffer_structures_to_polygons_produces_valid_rings():
    with patch("app.processing.osm_structures.requests.post", _mock_post()):
        structures = fetch_osm_structures(46.48, 106.27, 46.49, 106.28)
    polygons = buffer_structures_to_polygons(structures, utm_epsg=32648, building_buffer_m=10.0, road_buffer_m=5.0)
    assert len(polygons) == 2  # building and road buffers here don't overlap
    for ring in polygons:
        assert len(ring) >= 4
        for lat, lon in ring:
            # buffered points should stay in the immediate vicinity of the
            # original structures, not jump continents from a coordinate
            # order bug (lat/lon swapped, etc.)
            assert 46.0 < lat < 47.0
            assert 106.0 < lon < 107.0


def test_buffer_structures_merges_overlapping_regions():
    close_buildings = OsmStructures(
        buildings=[
            [[46.484, 106.273], [46.484, 106.2731], [46.4841, 106.2731], [46.4841, 106.273], [46.484, 106.273]],
            [[46.4841, 106.2731], [46.4841, 106.2732], [46.4842, 106.2732], [46.4842, 106.2731], [46.4841, 106.2731]],
        ],
        roads=[],
    )
    polygons = buffer_structures_to_polygons(close_buildings, utm_epsg=32648, building_buffer_m=15.0, road_buffer_m=5.0)
    assert len(polygons) == 1


def test_buffer_structures_returns_empty_for_no_structures():
    empty = OsmStructures(buildings=[], roads=[])
    assert buffer_structures_to_polygons(empty, utm_epsg=32648, building_buffer_m=10.0, road_buffer_m=5.0) == []
