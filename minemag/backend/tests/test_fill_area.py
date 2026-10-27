"""Filling inside the area of interest: cells inside the polygon keep
their interpolated value whatever their distance to a reading; outside
the polygon the distance mask still applies."""
from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.processing.gridding import grid_survey
from conftest import FIXTURES
from test_spacing_and_pair import _lines

LD_FIXTURE = FIXTURES / "Log_Drone_20261019_134937_rows6000-8600.csv"


def test_polygon_lifts_the_distance_mask_only_inside():
    df = _lines(2.0)                                          # 5 lines, 2 m apart, 30 m long
    grid0, s0 = grid_survey(df, cell_size_m=0.25, max_distance_m=0.6)
    assert s0["coverage_pct"] < 70 and s0["fill_inside_area"] is False
    poly = [(-0.5, 5.0), (8.5, 5.0), (8.5, 20.0), (-0.5, 20.0)]   # the middle of the block
    grid1, s1 = grid_survey(df, cell_size_m=0.25, max_distance_m=0.6, fill_polygon_xy=poly)
    assert s1["fill_inside_area"] is True and s1["n_filled_cells"] > 100
    gx, gy = np.meshgrid(grid1.easting, grid1.northing)
    if gx.shape != grid1.values.shape:
        gx, gy = gx.T, gy.T
    inside = (gx > 0) & (gx < 8) & (gy > 5.5) & (gy < 19.5)
    assert np.isfinite(grid1.values[inside]).mean() > 0.99          # no holes inside the polygon
    outside_gap = (np.abs(gx - 1.0) < 0.3) & (gy > 22) & (gy < 28)   # between lines 0 and 1, outside the polygon
    assert np.isnan(grid1.values[outside_gap]).all()                 # the distance mask still rules there
    assert np.isnan(grid0.values[inside]).any()                      # and the hole really was there before


def test_api_fill_inside_area_needs_an_area_and_raises_coverage():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(LD_FIXTURE, "rb") as f:
        client.post(f"/api/projects/{pid}/upload", files={"files": (LD_FIXTURE.name, f, "text/csv")})
    base = client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.25, "max_distance_m": 0.5}).json()["stats"]
    no_area = client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.25, "max_distance_m": 0.5, "fill_inside_area": True}).json()["stats"]
    assert no_area["fill_inside_area"] is False and no_area["coverage_pct"] == base["coverage_pct"]
    pts = client.get(f"/api/projects/{pid}/points", params={"max_points": 3000}).json()
    lat, lon = np.array(pts["lat"]), np.array(pts["lon"])
    poly = [[lat.min(), lon.min()], [lat.min(), lon.max()], [lat.max(), lon.max()], [lat.max(), lon.min()]]
    client.put(f"/api/projects/{pid}/area", json={"polygon": poly})
    filled = client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": 0.25, "max_distance_m": 0.5, "fill_inside_area": True}).json()["stats"]
    assert filled["fill_inside_area"] is True and filled["n_filled_cells"] > 0 and filled["coverage_pct"] > base["coverage_pct"]
    t = client.post(f"/api/projects/{pid}/targets", json={"sensor_height_m": 1.0, "max_targets": 3, "fill_inside_area": True})
    assert t.status_code == 200, t.text
