"""The two choices about what derived grids are made from.

  regional_removal  the base grid: the anomaly as it is, or with a
                    background (median level, robust plane or quadratic)
                    taken out of it first. RTP and upward continuation
                    carry a level straight through, so on a survey that
                    sits at -100 nT a local high stays negative in them.
  derive_from_rtp   the edge- and peak-reading products (tilt, theta, the
                    derivatives) made from the RTP of the base grid, where
                    edges sit over the bodies, instead of from the grid.

Both are off by default, and off must leave every result what it was.
Checked on the sample survey through the API, and on the survey the
question came from (GaSaDo; skipped where the field data is absent).
"""
import io
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient

from app.main import app
from app.models import GridRequest, TransformConfidenceRequest, TransformRequest
from app.store import store
from magcore.transforms import reduction_to_pole, tilt_angle, upward_continuation
from magcore.trend import remove_regional_trend, robust_trend_surface

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"
CELL = 10.0
METHOD = GridRequest().method          # the gridding the map and the API use by default


@pytest.fixture(scope="module")
def project():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{pid}/process", json={"line_params": {}, "diurnal_params": {}, "heading_correction": {}})
    assert r.status_code == 200, r.text
    return client, pid, store.get(pid)


def values(proj, transform, **kw):
    grid = proj._grid_for("anomaly", CELL, METHOD)
    return proj._transform_values(grid, TransformRequest(transform=transform, cell_size_m=CELL, **kw))[0]


def base(proj, **kw):
    grid = proj._grid_for("anomaly", CELL, METHOD)
    return proj._regional_split(grid, GridRequest(cell_size_m=CELL, **kw))


def finite_equal(a, b, atol):
    ok = np.isfinite(a) & np.isfinite(b)
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    return np.allclose(a[ok], b[ok], rtol=0, atol=atol)


# ------------------------------------------------------------------ the base grid
def test_off_by_default_and_off_changes_nothing(project):
    _client, _pid, proj = project
    assert GridRequest().regional_removal == "none" and TransformRequest(transform="rtp").regional_removal == "none"
    assert TransformRequest(transform="tilt").derive_from_rtp is False
    grid = proj._grid_for("anomaly", CELL, METHOD)
    same, info = base(proj)
    assert same is grid and info["method"] == "none" and info["level_nt"] == 0.0
    assert finite_equal(values(proj, "rtp"), values(proj, "rtp", regional_removal="none", derive_from_rtp=False), 0.0)


def test_the_level_goes_through_rtp_and_upward_continuation_unchanged(project):
    """Why the option exists: these two hand the background straight on."""
    _client, _pid, proj = project
    residual, info = base(proj, regional_removal="level")
    level = info["level_nt"]
    assert abs(level) > 1.0 and info["data_median_after_nt"] == pytest.approx(0.0, abs=1e-9)
    for name, kw in (("rtp", {}), ("upward_continuation", {"continuation_height_m": 40.0})):
        plain = values(proj, name, **kw)
        from_residual = values(proj, name, regional_removal="level", **kw)
        diff = (plain - from_residual)[np.isfinite(plain)]
        assert np.median(diff) == pytest.approx(level, abs=0.02 * abs(level)), name
        assert np.percentile(np.abs(diff - level), 90) < 0.05 * abs(level), name


def test_products_that_differentiate_do_not_care_about_a_level(project):
    _client, _pid, proj = project
    for name in ("as", "1vd", "thdr", "tilt"):
        plain, from_residual = values(proj, name), values(proj, name, regional_removal="level")
        ok = np.isfinite(plain)
        scale = np.nanpercentile(np.abs(plain), 95)
        assert np.nanpercentile(np.abs(plain - from_residual)[ok], 95) < 0.02 * scale, name


def test_the_map_its_exports_and_the_derived_grids_share_one_base(project):
    client, pid, proj = project
    residual, info = base(proj, regional_removal="plane")
    assert info["method"] == "plane" and info["min_nt"] < info["max_nt"]

    r = client.post(f"/api/projects/{pid}/grid", json={"value": "anomaly", "cell_size_m": CELL, "regional_removal": "plane"})
    assert r.status_code == 200, r.text
    overlay = r.json()
    assert overlay["regional_removal"]["method"] == "plane"
    assert overlay["regional_removal"]["level_nt"] == pytest.approx(info["level_nt"])
    shown = proj._apply_display_boundary(residual.values, residual)            # the map is clipped to the survey outline
    assert overlay["stats"]["mean"] == pytest.approx(float(np.nanmean(shown)), abs=1e-6)

    r = client.post(f"/api/projects/{pid}/grid/geotiff", json={"value": "anomaly", "cell_size_m": CELL, "regional_removal": "plane"})
    assert r.status_code == 200, r.text
    with rasterio.open(io.BytesIO(r.content)) as ds:
        exported = ds.read(1, masked=True).astype(float).filled(np.nan)
    assert np.nanmean(exported) == pytest.approx(float(np.nanmean(shown)), abs=0.05)

    # the detrend transform of a grid that has had its plane removed has nothing left to remove
    again = values(proj, "detrend", trend_order=1, regional_removal="plane")
    assert finite_equal(again, residual.values, 1e-3)

    r = client.post(f"/api/projects/{pid}/transform", json={"transform": "rtp", "cell_size_m": CELL, "regional_removal": "level"})
    assert r.status_code == 200, r.text
    assert r.json()["regional_removal"]["method"] == "level"
    r = client.post(f"/api/projects/{pid}/grid", json={"cell_size_m": CELL, "regional_removal": "cubic"})
    assert r.status_code == 422


def test_the_detrend_transform_follows_the_background_too(project):
    """It used a plain least-squares surface, which leans toward the
    strongest anomalies; it is now the same robust surface as the base
    grid's."""
    _client, _pid, proj = project
    grid = proj._grid_for("anomaly", CELL, METHOD)
    detrended = values(proj, "detrend", trend_order=1)
    robust = grid.values - robust_trend_surface(grid.values, grid.easting, grid.northing, 1)
    assert finite_equal(detrended, robust, 1e-9)
    least_squares, _ = remove_regional_trend(grid.values, grid.easting, grid.northing, order=1)
    assert not finite_equal(detrended, least_squares, 1.0)
    # the robust residual sits on zero where the background is; the least-squares one has mean zero instead
    assert abs(np.nanmedian(detrended)) < abs(np.nanmedian(least_squares))


# ------------------------------------------------------------------ derived from RTP
def test_tilt_from_rtp_is_the_tilt_of_the_rtp_grid(project):
    _client, _pid, proj = project
    plain = values(proj, "tilt", boundary_continuation=False, derivative_presmooth=False)
    via = values(proj, "tilt", derive_from_rtp=True, boundary_continuation=False, derivative_presmooth=False)
    grid = proj._grid_for("anomaly", CELL, METHOD)
    expected = tilt_angle(reduction_to_pole(grid.values, CELL, proj.inclination_deg, proj.declination_deg), CELL)
    expected = np.where(np.isnan(grid.values), np.nan, expected)
    assert finite_equal(via, expected, 1e-9)
    assert not finite_equal(via, plain, 1.0)


def test_it_applies_only_to_the_products_it_is_meant_for(project):
    client, pid, proj = project
    for name in ("1vd", "thdr", "theta", "dx", "dyz"):
        assert not finite_equal(values(proj, name), values(proj, name, derive_from_rtp=True), 1e-9), name
    for name in ("as", "rtp", "upward_continuation", "detrend"):
        assert finite_equal(values(proj, name), values(proj, name, derive_from_rtp=True), 0.0), name

    r = client.post(f"/api/projects/{pid}/transform", json={"transform": "tilt", "cell_size_m": CELL, "derive_from_rtp": True})
    assert r.status_code == 200 and r.json()["derived_from_rtp"] == {"requested": True, "applied": True, "reason": None}
    r = client.post(f"/api/projects/{pid}/transform", json={"transform": "as", "cell_size_m": CELL, "derive_from_rtp": True})
    report = r.json()["derived_from_rtp"]
    assert (report["requested"], report["applied"]) == (True, False) and "RTP" in report["reason"]
    r = client.post(f"/api/projects/{pid}/transform", json={"transform": "tilt", "cell_size_m": CELL})
    assert r.json()["derived_from_rtp"] == {"requested": False, "applied": False, "reason": None}


def test_the_confidence_layer_goes_through_the_pole_as_well(project):
    _client, _pid, proj = project
    plain = proj._transform_components("1vd", TransformConfidenceRequest(transform="1vd", cell_size_m=CELL), CELL)
    via = proj._transform_components("1vd", TransformConfidenceRequest(transform="1vd", cell_size_m=CELL, derive_from_rtp=True), CELL)
    rng = np.random.default_rng(0)
    field = rng.normal(size=(64, 64))
    assert len(plain) == len(via) == 1
    assert not np.allclose(plain[0](field), via[0](field))
    untouched = proj._transform_components("as", TransformConfidenceRequest(transform="as", cell_size_m=CELL, derive_from_rtp=True), CELL)
    reference = proj._transform_components("as", TransformConfidenceRequest(transform="as", cell_size_m=CELL), CELL)
    assert all(np.allclose(a(field), b(field)) for a, b in zip(untouched, reference))


def test_both_together_and_cached_apart(project):
    _client, _pid, proj = project
    a = values(proj, "tilt")
    b = values(proj, "tilt", regional_removal="level", derive_from_rtp=True)
    c = values(proj, "tilt", derive_from_rtp=True)
    d = values(proj, "tilt")                                 # the first again: not the cached answer to another question
    assert finite_equal(a, d, 0.0) and not finite_equal(a, c, 1.0)
    assert np.isfinite(b).sum() == np.isfinite(c).sum() > 0
    assert not finite_equal(b, a, 1.0)


# ---------------------------------------------------------------------------
# GaSaDo: the local high at 34.466633 N 126.048416 E. About 85 nT above its
# surroundings (108 nT in RTP) and negative in the anomaly map, because the
# survey sits at about -101 nT - and negative in RTP for the same reason.
# ---------------------------------------------------------------------------
GASADO = pathlib.Path(os.environ.get("MAGNETIC_GASADO_DIR") or pathlib.Path(__file__).resolve().parents[2] / "GaSaDo")
COMPENSATED = GASADO / "compensated"


@pytest.mark.skipif(len(list(COMPENSATED.glob("SRVY0-ACQU*-comp.csv"))) < 41, reason=f"GaSaDo compensated flights not present ({COMPENSATED})")
def test_gasado_local_high_reads_positive_once_the_level_is_out():
    from pyproj import Transformer

    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    number = lambda f: int(f.name.split("ACQU")[1].split("-")[0])  # noqa: E731
    files = sorted((f for f in COMPENSATED.glob("SRVY0-ACQU*-comp.csv") if 85 <= number(f) <= 125), key=number)
    r = client.post(f"/api/projects/{pid}/upload/drone", files=[("files", (f.name, f.read_bytes(), "text/csv")) for f in files])
    assert r.status_code == 200, r.text
    r = client.post(f"/api/projects/{pid}/process", json={"diurnal_params": {"mode": "assume_constant"}})
    assert r.status_code == 200, r.text
    proj = store.get(pid)
    grid = proj._grid_for("anomaly", CELL, METHOD)
    x0, y0 = Transformer.from_crs("EPSG:4326", f"EPSG:{proj.utm_epsg}", always_xy=True).transform(126.048416, 34.466633)
    e, n = np.meshgrid(grid.easting, grid.northing)
    dist = np.hypot(e - x0, n - y0)

    def peak(v):
        return float(np.nanmax(np.where(dist <= 170, v, -np.inf)))

    def ring(v):
        return float(np.nanmedian(v[(dist > 300) & (dist <= 600)]))

    residual, info = proj._regional_split(grid, GridRequest(cell_size_m=CELL, regional_removal="level"))
    assert -125 < info["level_nt"] < -85
    rtp = proj._transform_values(grid, TransformRequest(transform="rtp", cell_size_m=CELL))[0]
    rtp_residual = proj._transform_values(grid, TransformRequest(transform="rtp", cell_size_m=CELL, regional_removal="level"))[0]

    # a high, by 40 nT or more over its surroundings, in every version
    for v in (grid.values, residual.values, rtp, rtp_residual):
        assert peak(v) - ring(v) > 40
    # negative as it comes, in the anomaly and in its RTP (-39 and -26 nT at the peak)
    assert peak(grid.values) < -20 and peak(rtp) < -10
    # positive once the level is out, in both
    assert peak(residual.values) > 15 and peak(rtp_residual) > 20
    # and most of the survey is no longer "negative"
    assert np.nanmean(rtp[np.isfinite(rtp)] < 0) > 0.85
    assert 0.35 < np.nanmean(rtp_residual[np.isfinite(rtp_residual)] < 0) < 0.65
