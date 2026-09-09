"""Automatic display boundary derived from the flown lines
(processing/boundary.py + store.auto_display_boundary + the shapefile/KML
export), and the multi-part boundary handling that came with it.

The point of the feature is that gridding's convex-hull cap over-fills a
concave footprint, so the map shows interpolated surface over ground
nobody flew; these tests pin the geometry that fixes that (buffer
distance, solid interior across the normal line spacing, concave notch
excluded, genuinely unflown gaps left out) rather than just "a polygon
came back".
"""
import io
import pathlib
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import shapefile as pyshp
from fastapi.testclient import TestClient
from shapely import unary_union
from shapely.geometry import Point, Polygon

from app.main import app
from app.processing.boundary import auto_survey_boundary
from app.processing.boundary_export import boundary_to_kml, boundary_to_shapefile_zip
from app.store import boundary_rings
from app.store import store as project_store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def _lines(specs):
    """specs: [(x_start, x_end, y), ...] -> (x, y, line_id) sampled every 2 m."""
    xs, ys, ids = [], [], []
    for i, (x0, x1, yv) in enumerate(specs):
        n = max(2, int(abs(x1 - x0) / 2))
        xs.append(np.linspace(x0, x1, n))
        ys.append(np.full(n, float(yv)))
        ids.append(np.full(n, i))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)


def _area(result):
    return unary_union([Polygon(r) for r in result.rings_xy])


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_boundary_sits_exactly_buffer_m_outside_the_outermost_line():
    x, y, lid = _lines([(0, 400, 0), (0, 400, 50), (0, 400, 100)])

    result = auto_survey_boundary(x, y, lid, buffer_m=10.0, line_spacing_m=50.0)

    ring = result.rings_xy[0]
    # lines span y = 0..100, so the ring must reach exactly 10 m beyond each
    assert ring[:, 1].min() == pytest.approx(-10.0, abs=0.5)
    assert ring[:, 1].max() == pytest.approx(110.0, abs=0.5)


def test_interior_stays_solid_across_the_normal_line_spacing():
    """Buffering each line by 10 m alone would leave three separate 20 m
    strips with unflown gaps between them - the grid would come out
    striped. The closing operation has to fuse them into one block."""
    x, y, lid = _lines([(0, 400, 0), (0, 400, 50), (0, 400, 100)])

    result = auto_survey_boundary(x, y, lid, buffer_m=10.0, line_spacing_m=50.0)

    assert result.n_parts == 1
    poly = Polygon(result.rings_xy[0])
    for y_between in (25.0, 75.0):  # midway between adjacent lines
        assert poly.contains(Point(200.0, y_between))


def test_concave_notch_is_excluded_unlike_a_convex_hull():
    """The whole reason the feature exists: an L-shaped block's notch is
    inside the convex hull, so hull-capped gridding fills it in."""
    x, y, lid = _lines(
        [(0, 400, 0), (0, 400, 50), (0, 400, 100), (0, 200, 150), (0, 200, 200), (0, 200, 250)]
    )

    result = auto_survey_boundary(x, y, lid, buffer_m=10.0, line_spacing_m=50.0)

    poly = Polygon(result.rings_xy[0])
    assert poly.contains(Point(100.0, 220.0))  # flown part of the short legs
    assert not poly.contains(Point(350.0, 220.0))  # the notch - never flown
    assert not poly.contains(Point(350.0, 150.0))
    # and it is genuinely concave, not just a hull that happens to fit
    assert poly.area < 0.95 * poly.convex_hull.area


def test_a_skipped_line_keeps_both_blocks_instead_of_dropping_one():
    """A gap far wider than the line spacing splits the survey. Keeping
    only the largest block would silently hide flown data from the map, so
    every block is kept - while the gap itself stays excluded."""
    x, y, lid = _lines([(0, 400, 0), (0, 400, 50), (0, 400, 250), (0, 400, 300)])

    result = auto_survey_boundary(x, y, lid, buffer_m=10.0, line_spacing_m=50.0)

    assert result.n_parts == 2
    area = _area(result)
    assert area.contains(Point(200.0, 25.0))  # first block
    assert area.contains(Point(200.0, 275.0))  # second block
    assert not area.contains(Point(200.0, 150.0))  # the unflown gap
    assert any("2개 구역" in w for w in result.warnings)


def test_boundary_is_simplified_to_a_workable_vertex_count():
    """Buffering keeps one vertex per input sample, which on a real survey
    produced a ~10,000-vertex ring that then rides along in every process
    summary, save file and export, and is point-in-polygon tested per grid
    cell. Simplification has to cut that hard while staying well inside
    the buffer distance."""
    n = 4000
    x = np.concatenate([np.linspace(0, 400, n), np.linspace(0, 400, n)])
    y = np.concatenate([np.zeros(n), np.full(n, 50.0)])
    lid = np.concatenate([np.zeros(n, int), np.ones(n, int)])

    result = auto_survey_boundary(x, y, lid, buffer_m=10.0, line_spacing_m=50.0)

    ring = result.rings_xy[0]
    assert len(ring) < 200, f"ring still has {len(ring)} vertices"
    # simplification must not eat into the promised buffer distance
    assert ring[:, 1].min() == pytest.approx(-10.0, abs=0.5)


def test_buffer_must_be_positive():
    x, y, lid = _lines([(0, 400, 0)])
    with pytest.raises(ValueError):
        auto_survey_boundary(x, y, lid, buffer_m=0.0, line_spacing_m=50.0)


def test_unknown_line_spacing_warns_that_lines_may_not_merge():
    x, y, lid = _lines([(0, 400, 0), (0, 400, 50)])

    result = auto_survey_boundary(x, y, lid, buffer_m=10.0, line_spacing_m=None)

    assert any("측선 간격" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# ring normalization (single ring vs multi-part, both stored shapes)
# ---------------------------------------------------------------------------


def test_boundary_rings_normalizes_both_stored_shapes():
    single = [[37.0, 126.0], [37.1, 126.0], [37.1, 126.1]]
    multi = [single, [[38.0, 127.0], [38.1, 127.0], [38.1, 127.1]]]

    assert boundary_rings(single) == [single]
    assert boundary_rings(multi) == multi
    assert boundary_rings(None) == []
    assert boundary_rings([]) == []


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _ring_latlon(n=5, lat0=37.0, lon0=126.0):
    return [[lat0 + 0.001 * i, lon0 + 0.001 * (i % 2)] for i in range(n)]


def test_shapefile_export_carries_every_block_as_one_multipart_polygon():
    rings = [_ring_latlon(), _ring_latlon(lat0=38.0, lon0=127.0)]

    data = boundary_to_shapefile_zip(rings, "haenam")

    zf = zipfile.ZipFile(io.BytesIO(data))
    assert set(zf.namelist()) == {"haenam.shp", "haenam.shx", "haenam.dbf", "haenam.prj"}
    reader = pyshp.Reader(
        shp=io.BytesIO(zf.read("haenam.shp")),
        dbf=io.BytesIO(zf.read("haenam.dbf")),
        shx=io.BytesIO(zf.read("haenam.shx")),
    )
    shape = reader.shape(0)
    assert shape.shapeType == pyshp.POLYGON
    assert len(shape.parts) == 2  # both blocks, not just the first
    assert "WGS 84" in zf.read("haenam.prj").decode()


def test_kml_export_is_valid_xml_and_uses_multigeometry_only_when_needed():
    import xml.dom.minidom as minidom

    one = boundary_to_kml([_ring_latlon()], "haenam").decode()
    two = boundary_to_kml([_ring_latlon(), _ring_latlon(lat0=38.0)], "haenam").decode()

    minidom.parseString(one)  # raises if malformed
    minidom.parseString(two)
    assert "<MultiGeometry>" not in one and one.count("<Polygon>") == 1
    assert "<MultiGeometry>" in two and two.count("<Polygon>") == 2


def test_export_closes_the_ring_and_rejects_a_degenerate_one():
    kml = boundary_to_kml([_ring_latlon(n=3)], "b").decode()
    coords = kml.split("<coordinates>")[1].split("</coordinates>")[0].split()
    assert coords[0] == coords[-1], "ring must be explicitly closed"

    with pytest.raises(ValueError):
        boundary_to_kml([[[37.0, 126.0], [37.1, 126.0]]], "b")  # only 2 vertices


# ---------------------------------------------------------------------------
# end to end through the API
# ---------------------------------------------------------------------------


def _processed_project(**process_overrides):
    client = TestClient(app)
    project_id = client.post("/api/projects").json()["project_id"]
    for endpoint, path in (("drone", DRONE_CSV), ("base", BASE_CSV)):
        with open(path, "rb") as f:
            client.post(f"/api/projects/{project_id}/upload/{endpoint}", files={"files": ("f.csv", f, "text/csv")})
    body = {"line_params": {}, "diurnal_params": {}, "heading_correction": {}}
    body.update(process_overrides)
    r = client.post(f"/api/projects/{project_id}/process", json=body)
    assert r.status_code == 200, r.text
    return client, project_id, r.json()


def test_processing_applies_an_auto_boundary_by_default_and_it_clips_the_grid():
    client, project_id, summary = _processed_project()

    assert summary["display_boundary_polygon"], "auto boundary should be on by default"
    assert summary["auto_boundary"]["buffer_m"] == 10.0
    project = project_store.get(project_id)
    grid_request = {"value": "anomaly", "cell_size_m": 20.0, "method": "nearest"}

    client.post(f"/api/projects/{project_id}/grid", json=grid_request)
    clipped = int(np.isfinite(project.last_overlay.values).sum())
    client.post(f"/api/projects/{project_id}/display-boundary", json={"polygon": None})
    client.post(f"/api/projects/{project_id}/grid", json=grid_request)
    unclipped = int(np.isfinite(project.last_overlay.values).sum())

    assert 0 < clipped < unclipped, "the auto boundary must actually remove over-filled cells"


def test_auto_boundary_can_be_turned_off_to_keep_a_hand_drawn_one():
    _client, _project_id, summary = _processed_project(auto_display_boundary=False)

    assert summary["display_boundary_polygon"] is None
    assert summary["auto_boundary"] is None


def test_regenerating_with_a_bigger_buffer_grows_the_boundary():
    client, project_id, _summary = _processed_project()

    small = client.post(f"/api/projects/{project_id}/display-boundary/auto", json={"buffer_m": 5.0}).json()
    large = client.post(f"/api/projects/{project_id}/display-boundary/auto", json={"buffer_m": 40.0}).json()

    assert large["area_km2"] > small["area_km2"]
    assert large["buffer_m"] == 40.0


def test_boundary_export_endpoint_serves_both_formats_and_rejects_others():
    client, project_id, _summary = _processed_project()

    shp = client.get(f"/api/projects/{project_id}/display-boundary/export", params={"format": "shp", "name": "haenam"})
    kml = client.get(f"/api/projects/{project_id}/display-boundary/export", params={"format": "kml", "name": "haenam"})
    bad = client.get(f"/api/projects/{project_id}/display-boundary/export", params={"format": "geojson"})

    assert shp.status_code == 200 and shp.content[:2] == b"PK"  # zip magic
    assert "haenam_shapefile.zip" in shp.headers["content-disposition"]
    assert kml.status_code == 200 and b"<kml" in kml.content
    assert bad.status_code == 400


def test_save_load_restores_the_boundary_state_including_a_cleared_one():
    """Loading replays run_pipeline, which regenerates an auto boundary -
    that must not overwrite a hand-drawn boundary, nor resurrect one the
    user deliberately cleared before saving."""
    client, project_id, _summary = _processed_project()
    drawn = [[46.5, 106.27], [46.52, 106.27], [46.52, 106.29], [46.5, 106.29]]
    client.post(f"/api/projects/{project_id}/display-boundary", json={"polygon": drawn})

    def save_and_reload(source_project_id):
        blob = client.get(f"/api/projects/{source_project_id}/save").content
        target = client.post("/api/projects").json()["project_id"]
        r = client.post(
            f"/api/projects/{target}/load", files={"file": ("p.zip", io.BytesIO(blob), "application/zip")}
        )
        assert r.status_code == 200, r.text
        return project_store.get(target).display_boundary_polygon

    assert save_and_reload(project_id) == drawn

    # now the cleared case
    client.post(f"/api/projects/{project_id}/display-boundary", json={"polygon": None})
    assert save_and_reload(project_id) is None


def test_a_failed_auto_boundary_does_not_fail_the_whole_processing_run():
    """The boundary is a display refinement; a survey it can't outline must
    still process and grid (capped at the hull as before), with the reason
    reported instead of the run erroring out."""
    client, project_id, _summary = _processed_project()
    project = project_store.get(project_id)
    # every line excluded -> nothing to build an outline from
    project.processed["line_id"] = -1

    with pytest.raises(Exception):
        project.auto_display_boundary(10.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
