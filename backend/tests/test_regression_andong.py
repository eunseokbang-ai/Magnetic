"""Real-data regression: the Andong 2026-10-28 MagArrow survey through the
whole API, with the numbers it produced on 2026-11-10 pinned.

The synthetic fixtures prove each step does what it says; they cannot say
whether a change moved the result on a real survey. This does. It is also
how the line-resolution diagnostic was found dead: the fixtures never
looked at it, and on this block it had been reporting "진단 실패" since
the day it was added (an undefined name, and behind it an azimuth passed
in the wrong convention).

The survey (18 files, 48 MB, 266,672 samples, N-S lines 100 m apart over
9 km2) is field data and is not in the repository, so the module skips
when the folder is absent. Point MAGNETIC_ANDONG_DIR at it to run
elsewhere. Diurnal correction is off ("assume_constant") so that nothing
depends on the network - the survey has no base station of its own and
the app would otherwise fetch observatory data.

Re-pinned 2026-11-11, when line detection began taking out the ferry legs
between the takeoff point and the block (magcore.lines._exclude_transit_runs):
11 legs, 2.1 km, 2,200 samples that had been kept as four "lines" and as
tails on others. Everything downstream of them moved, and in the direction
of a cleaner survey - lines flagged noisy 7 -> 1, overall 4th-difference
noise 0.433 -> 0.036 nT, roughness after levelling 3.91 -> 2.61 nT, the
line-spacing estimate 99.1 -> 99.5 m (flown at 100), and the block lost the
strip west of it that only a ferry leg had crossed (9.01 -> 8.81 km2).

A pinned number that moves is not automatically a bug. If the change was
meant to move it, update EXPECTED in the same commit and say why; if it
was not, that is the regression this file exists to catch. Tolerances
are loose enough for a different BLAS or numpy build (1e-3 relative on
field statistics), not for a change of method.
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

DATA = pathlib.Path(os.environ.get("MAGNETIC_ANDONG_DIR") or pathlib.Path(__file__).resolve().parents[2] / "AnDong_Mag" / "20261028")
pytestmark = pytest.mark.skipif(not list(DATA.glob("SRVY0-ACQU*.csv")), reason=f"Andong field data not present ({DATA})")

CELL_M = 10.0


def _grid_stats(content: bytes) -> dict:
    with rasterio.open(io.BytesIO(content)) as ds:
        a = ds.read(1, masked=True).astype(float).filled(np.nan)
        p5, p95 = np.nanpercentile(a, [5, 95])
        return {"crs": str(ds.crs), "shape": list(a.shape), "n_valid": int(np.isfinite(a).sum()),
                "median": float(np.nanmedian(a)), "p5": float(p5), "p95": float(p95), "std": float(np.nanstd(a))}


@pytest.fixture(scope="module")
def andong():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    files = sorted(DATA.glob("SRVY0-ACQU*.csv"))
    r = client.post(f"/api/projects/{pid}/upload/drone", files=[("files", (f.name, f.read_bytes(), "text/csv")) for f in files])
    assert r.status_code == 200, r.text
    upload = r.json()
    r = client.post(f"/api/projects/{pid}/process", json={"diurnal_params": {"mode": "assume_constant"}})
    assert r.status_code == 200, r.text
    return {"client": client, "pid": pid, "upload": upload, "process": r.json()}


def _close(actual, expected, rel=1e-3):
    assert actual == pytest.approx(expected, rel=rel), (actual, expected)


def test_loading_reads_every_file_and_sample(andong):
    u = andong["upload"]
    assert (u["n_points"], u["n_files_total"], u["n_duplicate_timestamps_removed"]) == (266672, 18, 0)
    _close(u["median_speed_mps"], 4.1379)


def test_line_detection_and_geometry(andong):
    p = andong["process"]
    assert (p["n_lines"], p["n_kept"], p["n_excluded_auto"]) == (37, 212609, 54063)
    assert p["utm_epsg"] == 32652
    assert p["dominant_azimuth_deg"] == 89.0            # atan2(dy, dx): the lines run north-south
    t = p["transit"]
    assert (t["n_removed"], t["n_points_removed"], t["n_suspect"]) == (11, 2200, 1)
    _close(t["length_removed_m"], 2110.9)
    _close(p["line_spacing_m"], 99.512)
    _close(p["inclination_deg"], 53.0445)
    _close(p["declination_deg"], -8.8209)
    _close(p["auto_boundary"]["area_km2"], 8.8076)


def test_corrections(andong):
    p = andong["process"]
    assert p["despike"]["n_spikes_removed"] == 5156
    h = p["heading_correction"]
    assert h["applied"] is False and (h["n_matched_pairs"], h["n_quiet_pairs"]) == (853, 341)
    _close(h["offset_nt"], 3.4078)
    lv = p["statistical_leveling"]
    assert lv["applied"] is True and lv["n_pairs"] == 36
    _close(lv["max_shift_nt"], 17.332)
    _close(lv["rms_shift_nt"], 6.6166)
    _close(lv["roughness_before_nt"], 11.637)
    _close(lv["roughness_after_nt"], 2.6092)
    a = p["anomaly_stats"]
    _close(a["min"], -1382.93)
    _close(a["max"], 1742.38)
    _close(a["std"], 470.309)
    assert a["mean"] == pytest.approx(12.681, abs=0.05)


def test_diagnostics_all_ran(andong):
    p = andong["process"]
    s = p["striping"]
    assert (s["line_level_jitter_nt"], s["signal_range_nt"], s["stripe_ratio_pct"]) == pytest.approx((11.637, 1603.62, 0.73), rel=1e-3)
    n = p["noise_qc"]
    assert n["n_lines_flagged"] == 1
    _close(n["overall_rms_4th_diff_nt"], 0.036494)
    # The line-spacing diagnostic: 100 m lines leave 42 % of each line's
    # 40-600 m signal unpredictable from its neighbours (sources ~38 m
    # away, so the spacing is 2.6 source distances; 19 m would resolve it).
    lr = p["line_resolution"]
    assert lr["available"] is True, lr
    assert lr["n_triples"] == 34
    assert (lr["unresolved_pct"], lr["line_spacing_m"], lr["line_content_nt"], lr["source_distance_m"],
            lr["spacing_over_distance"], lr["recommended_spacing_m"]) == pytest.approx((42.0, 99.6, 140.5, 38.2, 2.6, 19.0), rel=1e-3)
    # no diagnostic anywhere in the summary swallowed an exception
    import json
    assert "진단 실패" not in json.dumps(p, ensure_ascii=False)


def test_anomaly_grid(andong):
    r = andong["client"].post(f"/api/projects/{andong['pid']}/grid/geotiff", json={"value": "anomaly", "cell_size_m": CELL_M})
    assert r.status_code == 200, r.text
    g = _grid_stats(r.content)
    assert (g["crs"], g["shape"], g["n_valid"]) == ("EPSG:32652", [288, 352], 81746)
    _close(g["median"], -104.281)
    _close(g["p5"], -557.72)
    _close(g["p95"], 934.99)
    _close(g["std"], 436.203)


def test_reduction_to_pole_grid(andong):
    r = andong["client"].post(f"/api/projects/{andong['pid']}/transform/geotiff",
                              json={"transform": "rtp", "value": "anomaly", "cell_size_m": CELL_M})
    assert r.status_code == 200, r.text
    g = _grid_stats(r.content)
    assert (g["crs"], g["shape"], g["n_valid"]) == ("EPSG:32652", [288, 352], 81746)
    _close(g["median"], -172.635)
    _close(g["p5"], -371.39)
    _close(g["p95"], 992.06)
    _close(g["std"], 440.610)


def test_euler_deconvolution(andong):
    r = andong["client"].post(f"/api/projects/{andong['pid']}/euler-deconvolution",
                              json={"cell_size_m": CELL_M, "structural_index": 1.0, "window_size_m": 100.0})
    assert r.status_code == 200, r.text
    e = r.json()
    # window acceptance sits on thresholds, so a count is looser than a field statistic
    assert e["n_solutions"] == pytest.approx(2034, rel=0.02)
    assert float(np.median([s["depth_m"] for s in e["solutions"]])) == pytest.approx(63.18, rel=0.02)
