"""Tests for the three QC/export additions built on top of the existing
pipeline: (1) exporting a detected-target list as CSV/shapefile, (2) a
continuous grid-confidence companion layer to the value grid, and (3) the
standard QC pass/fail certificate that bundles existing QC indicators
against configurable acceptance thresholds. See processing/qc_certificate.py,
processing/gridding.py::grid_confidence, and store.py's
export_targets_csv/export_targets_shapefile/get_grid_confidence_overlay/
generate_qc_certificate for the implementations under test.
"""
import io
import pathlib
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.models import GridConfidenceRequest, QcCertificateRequest
from app.processing.gridding import grid_confidence, grid_points
from app.processing.qc_certificate import evaluate_qc_certificate
from app.store import ProjectError, store as project_store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def _make_processed_project():
    client = TestClient(app)
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{project_id}/process", json={"line_params": {}, "diurnal_params": {}, "heading_correction": {}})
    assert r.status_code == 200, r.text
    return client, project_id


# ---------------------------------------------------------------------------
# Target CSV/shapefile export
# ---------------------------------------------------------------------------

_FAKE_TARGETS = [
    {
        "lat": 35.1, "lon": 126.5, "depth_m": 1.2, "moment_am2": 3.4, "size_class": "small_metal",
        "peak_anomaly_nt": 120.0, "footprint_m": 2.0, "fit_quality": 0.8, "background_nt": 50000.0,
    },
    {
        "lat": 35.2, "lon": 126.6, "depth_m": 0.5, "moment_am2": 1.1, "size_class": "very_small",
        "peak_anomaly_nt": 60.0, "footprint_m": 1.0, "fit_quality": 0.6, "background_nt": 50000.0,
    },
]


def test_export_targets_csv_contains_every_target():
    project = project_store.create()
    project.target_summary_cache = {"n_targets": 2, "amplitude_threshold_nt": 10.0, "cell_size_m": 1.0, "targets": _FAKE_TARGETS}
    data = project.export_targets_csv()
    df = pd.read_csv(io.BytesIO(data))
    assert len(df) == 2
    assert list(df["target_id"]) == [1, 2]
    assert set(["lat", "lon", "depth_m", "moment_am2", "size_class", "fit_quality"]).issubset(df.columns)
    assert abs(df.iloc[0]["lat"] - 35.1) < 1e-9


def test_export_targets_csv_header_only_when_zero_targets():
    project = project_store.create()
    project.target_summary_cache = {"n_targets": 0, "amplitude_threshold_nt": 10.0, "cell_size_m": 1.0, "targets": []}
    data = project.export_targets_csv()
    df = pd.read_csv(io.BytesIO(data))
    assert len(df) == 0
    assert "target_id" in df.columns


def test_export_targets_csv_requires_detection_first():
    project = project_store.create()
    try:
        project.export_targets_csv()
        assert False, "should have raised"
    except ProjectError:
        pass


def test_export_targets_shapefile_is_a_valid_zip_with_matching_point_count():
    import shapefile as pyshp

    project = project_store.create()
    project.target_summary_cache = {"n_targets": 2, "amplitude_threshold_nt": 10.0, "cell_size_m": 1.0, "targets": _FAKE_TARGETS}
    data = project.export_targets_shapefile()

    zf = zipfile.ZipFile(io.BytesIO(data))
    names = set(zf.namelist())
    assert {"targets.shp", "targets.shx", "targets.dbf", "targets.prj"}.issubset(names)

    reader = pyshp.Reader(
        shp=io.BytesIO(zf.read("targets.shp")), shx=io.BytesIO(zf.read("targets.shx")), dbf=io.BytesIO(zf.read("targets.dbf"))
    )
    shapes = reader.shapes()
    assert len(shapes) == 2
    points = sorted(s.points[0] for s in shapes)
    assert abs(points[0][0] - 126.5) < 1e-6 and abs(points[0][1] - 35.1) < 1e-6


def test_target_export_endpoints_via_api():
    client, project_id = _make_processed_project()
    project = project_store.get(project_id)
    project.target_summary_cache = {"n_targets": 2, "amplitude_threshold_nt": 10.0, "cell_size_m": 1.0, "targets": _FAKE_TARGETS}

    r = client.get(f"/api/projects/{project_id}/target-detection/csv")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    df = pd.read_csv(io.BytesIO(r.content))
    assert len(df) == 2

    r = client.get(f"/api/projects/{project_id}/target-detection/shapefile")
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert "targets.shp" in zf.namelist()


# ---------------------------------------------------------------------------
# Grid confidence layer
# ---------------------------------------------------------------------------

def _two_line_survey(line_spacing_m=50.0):
    x = np.linspace(0, 400, 80)
    y0 = np.zeros_like(x)
    y1 = np.full_like(x, line_spacing_m)
    xs = np.concatenate([x, x])
    ys = np.concatenate([y0, y1])
    line_id = np.concatenate([np.zeros_like(x, dtype=int), np.ones_like(x, dtype=int)])
    values = 100 + np.sin(xs / 30.0)
    return xs, ys, values, line_id


def test_grid_confidence_is_high_at_data_and_fades_toward_midline():
    x, y, values, line_id = _two_line_survey(line_spacing_m=50.0)
    cell_size_m = 5.0
    grid = grid_points(x, y, values, cell_size_m, method="nearest", line_id=line_id)
    easting_2d, northing_2d = np.meshgrid(grid.easting, grid.northing)

    confidence = grid_confidence(x, y, easting_2d, northing_2d, cell_size_m, line_id=line_id)
    assert confidence.shape == grid.values.shape

    # Confidence is NaN exactly where the value grid is NaN (they must
    # agree on the fill boundary - see _resolve_auto_mask).
    assert np.array_equal(np.isnan(confidence), np.isnan(grid.values))

    # A row that actually sits on line 0 (y=0) should be near-maximum
    # confidence; the midline row (y=25, farthest from both lines) should
    # be markedly lower.
    row_on_line = np.argmin(np.abs(grid.northing - 0.0))
    row_midline = np.argmin(np.abs(grid.northing - 25.0))
    on_line_conf = np.nanmean(confidence[row_on_line, :])
    midline_conf = np.nanmean(confidence[row_midline, :])
    assert on_line_conf > 0.9
    assert midline_conf < on_line_conf
    assert np.nanmax(confidence) <= 1.0 and np.nanmin(confidence) >= 0.0


def test_grid_confidence_reuses_grid_points_cached_auto_mask():
    """Regression/perf test: grid_points already computes the nearest-
    point-distance field and auto-mask (tree_dist/max_distance_grid/
    hull_mask) to build its own NaN mask - the dominant cost of gridding
    a survey (a cKDTree build/query plus, in auto mode, the per-line
    local-gap computation). grid_confidence used to always redo that
    same work from scratch even when called right after grid_points for
    the identical point set/parameters. Confirms GridResult actually
    carries those fields, and that passing them through gives byte-
    identical output to the from-scratch path (so reusing them is safe,
    not just faster)."""
    x, y, values, line_id = _two_line_survey(line_spacing_m=50.0)
    cell_size_m = 5.0
    grid = grid_points(x, y, values, cell_size_m, method="nearest", line_id=line_id)
    assert grid.tree_dist is not None
    assert grid.max_distance_grid is not None
    assert grid.hull_mask is not None

    easting_2d, northing_2d = np.meshgrid(grid.easting, grid.northing)
    from_scratch = grid_confidence(x, y, easting_2d, northing_2d, cell_size_m, line_id=line_id)
    reused = grid_confidence(
        x, y, easting_2d, northing_2d, cell_size_m, line_id=line_id,
        tree_dist=grid.tree_dist, max_distance_grid=grid.max_distance_grid, hull_mask=grid.hull_mask,
    )
    assert np.array_equal(np.isnan(from_scratch), np.isnan(reused))
    finite = ~np.isnan(from_scratch)
    assert np.allclose(from_scratch[finite], reused[finite])


def test_grid_confidence_overlay_endpoint_via_api():
    client, project_id = _make_processed_project()
    r = client.post(
        f"/api/projects/{project_id}/grid/confidence",
        json=GridConfidenceRequest(value="anomaly", cell_size_m=20.0, method="nearest").model_dump(),
    )
    assert r.status_code == 200, r.text
    resp = r.json()
    assert "image_base64" in resp or "png_base64" in resp or "stats" in resp
    assert "cell_size_m" in resp


# ---------------------------------------------------------------------------
# QC certificate
# ---------------------------------------------------------------------------

def test_qc_certificate_all_not_evaluated_on_empty_summary():
    result = evaluate_qc_certificate({"noise_qc": {"available": False}}, None)
    assert result["overall_status"] == "not_evaluated"
    assert result["n_pass"] == 0
    assert result["n_fail"] == 0
    assert all(c["status"] == "not_evaluated" for c in result["criteria"])


def test_qc_certificate_passes_when_metrics_are_comfortably_within_threshold():
    process_summary = {
        "noise_qc": {
            "available": True, "overall_rms_4th_diff_nt": 0.5, "reference_threshold_4th_diff_nt": 2.0,
            "sample_rate_hz": 10.0, "n_lines_flagged": 0,
        },
        "sampling_qc": {"available": True, "pct_gaps_exceeding_tolerance": 0.1, "gap_tolerance_m": 20.0, "n_gaps_exceeding_tolerance": 1},
        "file_level_check": {"available": True, "flagged_any": False, "n_files": 2, "flag_threshold_nt": 5.0},
        "heading_effect_calibration": {"enabled": False},
        "n_kept": 950,
        "n_excluded_auto": 50,
    }
    result = evaluate_qc_certificate(process_summary, None)
    assert result["overall_status"] == "pass"
    assert result["n_fail"] == 0
    by_name = {c["name"]: c for c in result["criteria"]}
    assert by_name["자력 노이즈 수준 (4차 차분 RMS)"]["status"] == "pass"
    assert by_name["샘플링 간격 계약기준 초과 비율"]["status"] == "pass"
    assert by_name["파일간 DC 레벨 오프셋"]["status"] == "pass"
    assert by_name["헤딩효과 캘리브레이션 교차검증"]["status"] == "not_evaluated"


def test_qc_certificate_fails_criteria_that_exceed_threshold():
    process_summary = {
        "noise_qc": {
            "available": True, "overall_rms_4th_diff_nt": 50.0, "reference_threshold_4th_diff_nt": 2.0,
            "sample_rate_hz": 10.0, "n_lines_flagged": 3,
        },
        "sampling_qc": {"available": True, "pct_gaps_exceeding_tolerance": 20.0, "gap_tolerance_m": 20.0, "n_gaps_exceeding_tolerance": 50},
        "file_level_check": {"available": True, "flagged_any": True, "n_files": 3, "flag_threshold_nt": 5.0},
        "heading_effect_calibration": {"enabled": False},
        "n_kept": 500,
        "n_excluded_auto": 500,
    }
    result = evaluate_qc_certificate(process_summary, None, max_excluded_pct=30.0)
    assert result["overall_status"] == "fail"
    assert result["n_fail"] >= 4
    by_name = {c["name"]: c for c in result["criteria"]}
    assert by_name["자력 노이즈 수준 (4차 차분 RMS)"]["status"] == "fail"
    assert by_name["자동 제외 자료 비율"]["status"] == "fail"


def test_qc_certificate_endpoint_via_api():
    client, project_id = _make_processed_project()
    r = client.post(f"/api/projects/{project_id}/qc-certificate", json=QcCertificateRequest().model_dump())
    assert r.status_code == 200, r.text
    resp = r.json()
    assert "criteria" in resp and "overall_status" in resp
    assert resp["overall_status"] in ("pass", "fail", "not_evaluated")

    project = project_store.get(project_id)
    assert project.qc_certificate_cache == resp
    report = project.generate_report()
    assert "QC 인증서" in report
