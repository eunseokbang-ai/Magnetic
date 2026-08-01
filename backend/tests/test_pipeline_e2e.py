"""End-to-end smoke test of the full API against the real sample survey.

Not a pytest suite (no pytest in requirements) - run directly:
    python3 tests/test_pipeline_e2e.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _multi_files(field: str, *paths: pathlib.Path):
    return [(field, (p.name, open(p, "rb"), "text/csv")) for p in paths]


def main():
    client = TestClient(app)

    r = client.post("/api/projects")
    assert r.status_code == 200, r.text
    project_id = r.json()["project_id"]
    print("project_id:", project_id)

    drone_path = FIXTURES / "sample_drone_survey.csv"
    base_path = FIXTURES / "sample_base_station.csv"

    # multi-file upload: same fixture supplied twice exercises both file
    # concatenation and exact-timestamp-duplicate QC removal at once - since
    # the two "files" here are byte-identical, every row collides on
    # timestamp with its twin and gets deduplicated back down to the
    # original single-file row count (a real accidental double-upload
    # would collapse the same way; two genuinely different overlapping
    # flights would not, since their timestamps wouldn't line up exactly).
    r = client.post(f"/api/projects/{project_id}/upload/drone", files=_multi_files("files", drone_path, drone_path))
    assert r.status_code == 200, r.text
    drone_summary_multi = r.json()
    print("drone summary (2 files):", drone_summary_multi)
    assert drone_summary_multi["n_points"] == 22025, drone_summary_multi["n_points"]
    assert drone_summary_multi["n_duplicate_timestamps_removed"] == 22025, drone_summary_multi

    r = client.post(f"/api/projects/{project_id}/upload/base", files=_multi_files("files", base_path, base_path))
    assert r.status_code == 200, r.text
    base_summary_multi = r.json()
    print("base summary (2 files):", base_summary_multi)
    assert base_summary_multi["n_points"] == 20314, base_summary_multi["n_points"]
    assert base_summary_multi["n_duplicate_timestamps_removed"] == 20314, base_summary_multi

    # re-upload single files for the rest of the pipeline (multi-file loading
    # already proven above; a single real flight keeps line detection sane)
    r = client.post(f"/api/projects/{project_id}/upload/drone", files=_multi_files("files", drone_path))
    assert r.status_code == 200, r.text
    print("drone summary (1 file):", r.json())

    r = client.post(f"/api/projects/{project_id}/upload/base", files=_multi_files("files", base_path))
    assert r.status_code == 200, r.text
    print("base summary (1 file):", r.json())

    r = client.post(
        f"/api/projects/{project_id}/process",
        json={
            "filter_cutoff_hz": 1.0,
            "line_params": {},
            "diurnal_params": {"time_offset_seconds": 0.0, "reference": "mean"},
            "heading_correction": {"enabled": True, "quiet_percentile": 40.0},
        },
    )
    assert r.status_code == 200, r.text
    summary = r.json()
    print("process summary:", {k: v for k, v in summary.items() if k != "lines"})
    print("lines:", summary["lines"])
    assert summary["n_lines"] >= 1
    assert summary["diurnal"]["has_overlap"] is False  # known clock mismatch in sample data
    assert summary["line_spacing_m"] is not None and summary["line_spacing_m"] > 0
    assert summary["heading_correction"]["applied"] is True, summary["heading_correction"]
    assert summary["heading_correction"]["offset_nt"] is not None
    assert all(l["heading_group"] in ("A", "B") for l in summary["lines"])
    assert all(l["heading_shift_nt"] is not None for l in summary["lines"])
    assert summary["despike"]["enabled"] is True
    assert 0 <= summary["despike"]["n_spikes_removed"] < summary["n_points"]
    assert summary["despike"]["pct_spikes_removed"] < 20.0  # sanity: not flagging most of the survey as spikes

    r = client.post(
        f"/api/projects/{project_id}/process",
        json={
            "filter_cutoff_hz": 1.0,
            "despike_params": {"enabled": False},
            "line_params": {},
            "diurnal_params": {"time_offset_seconds": 0.0, "reference": "mean"},
            "heading_correction": {"enabled": True, "quiet_percentile": 40.0},
        },
    )
    assert r.status_code == 200, r.text
    summary_no_despike = r.json()
    assert summary_no_despike["despike"] == {"enabled": False, "n_spikes_removed": 0, "pct_spikes_removed": 0.0}

    r = client.post(
        f"/api/projects/{project_id}/process",
        json={
            "filter_cutoff_hz": 1.0,
            "gps_mag_lag_seconds": 0.5,
            "line_params": {},
            "diurnal_params": {"time_offset_seconds": 0.0, "reference": "mean"},
            "heading_correction": {"enabled": True, "quiet_percentile": 40.0},
        },
    )
    assert r.status_code == 200, r.text
    lag_summary = r.json()
    print("gps_mag_lag summary:", lag_summary["gps_mag_lag"])
    assert lag_summary["gps_mag_lag"]["lag_seconds"] == 0.5
    assert lag_summary["gps_mag_lag"]["n_points_dropped"] > 0
    assert lag_summary["n_points"] < summary["n_points"]  # trimmed rows at the shifted-out end of the track

    # re-run with despiking back on so the rest of the pipeline (grids, inversion) uses the cleaned default
    r = client.post(
        f"/api/projects/{project_id}/process",
        json={
            "filter_cutoff_hz": 1.0,
            "line_params": {},
            "diurnal_params": {"time_offset_seconds": 0.0, "reference": "mean"},
            "heading_correction": {"enabled": True, "quiet_percentile": 40.0},
        },
    )
    assert r.status_code == 200, r.text

    r = client.get(f"/api/projects/{project_id}/points", params={"value": "anomaly"})
    assert r.status_code == 200, r.text
    pts = r.json()
    print("n points returned:", len(pts))
    assert len(pts) == summary["n_points"]
    n_excluded = sum(1 for p in pts if p["excluded"])
    assert n_excluded == summary["n_points"] - summary["n_kept"]

    # manual exclude by line id
    first_line = summary["lines"][0]["line_id"]
    r = client.post(
        f"/api/projects/{project_id}/manual-exclude",
        json={"mode": "lines", "action": "exclude", "line_ids": [first_line]},
    )
    assert r.status_code == 200, r.text
    summary2 = r.json()
    print("after manual exclude, n_kept:", summary2["n_kept"], "n_manual_excluded:", summary2["n_manual_excluded"])
    assert summary2["n_kept"] < summary["n_kept"]

    # manual-exclude responses carry a lightweight (point_id, excluded)
    # delta so the frontend can patch its point cache in place instead of
    # re-fetching the full point list (slow at 100k+ points) after every edit.
    exclusion = summary2["exclusion"]
    assert len(exclusion["point_id"]) == len(exclusion["excluded"]) == summary["n_points"]
    n_excluded_from_delta = sum(1 for e in exclusion["excluded"] if e)
    assert n_excluded_from_delta == summary["n_points"] - summary2["n_kept"]

    # manual polygon exclude (small box around the survey centroid)
    lats = [p["lat"] for p in pts]
    lons = [p["lon"] for p in pts]
    clat, clon = sum(lats) / len(lats), sum(lons) / len(lons)
    d = 0.001
    polygon = [[clat - d, clon - d], [clat - d, clon + d], [clat + d, clon + d], [clat + d, clon - d]]
    r = client.post(
        f"/api/projects/{project_id}/manual-exclude",
        json={"mode": "polygon", "action": "exclude", "polygon": polygon},
    )
    assert r.status_code == 200, r.text
    print("after polygon exclude, n_kept:", r.json()["n_kept"])

    # force-include a batch of auto-excluded points (e.g. a turbulence
    # segment the auto line-detector correctly dropped but the user wants
    # back) - pick a polygon around some points with line_id == -1
    excluded_pts = [p for p in pts if p["excluded"] and p["line_id"] < 0]
    assert excluded_pts, "expected some auto-excluded points to restore"
    target = excluded_pts[len(excluded_pts) // 2]
    d2 = 0.0005
    restore_polygon = [
        [target["lat"] - d2, target["lon"] - d2],
        [target["lat"] - d2, target["lon"] + d2],
        [target["lat"] + d2, target["lon"] + d2],
        [target["lat"] + d2, target["lon"] - d2],
    ]
    before_kept = r.json()["n_kept"]
    r = client.post(
        f"/api/projects/{project_id}/manual-exclude",
        json={"mode": "polygon", "action": "include", "polygon": restore_polygon},
    )
    assert r.status_code == 200, r.text
    summary_restored = r.json()
    print("after force-include, n_kept:", summary_restored["n_kept"], "n_manual_included:", summary_restored["n_manual_included"])
    assert summary_restored["n_kept"] > before_kept, "force-including auto-excluded points should raise n_kept"
    assert summary_restored["n_manual_included"] > 0

    # full reset should return to the original automatic n_kept
    r = client.post(f"/api/projects/{project_id}/manual-exclude", json={"mode": "reset"})
    assert r.status_code == 200, r.text
    summary_reset = r.json()
    print("after reset, n_kept:", summary_reset["n_kept"])
    assert summary_reset["n_kept"] == summary["n_kept"]
    assert summary_reset["n_manual_included"] == 0 and summary_reset["n_manual_excluded"] == 0

    for method in ["nearest", "spline", "linear", "cubic"]:
        r = client.post(f"/api/projects/{project_id}/grid", json={"value": "anomaly", "cell_size_m": 10.0, "method": method})
        assert r.status_code == 200, r.text
        grid_resp = r.json()
        print(f"grid[{method}] bounds:", grid_resp["bounds"], "stats:", grid_resp["stats"])
        assert grid_resp["image_data_url"].startswith("data:image/png;base64,")

    # manual vmin/vmax should be honored as-is (not overridden by the
    # symmetric-around-zero auto-expansion for anomaly grids)
    r = client.post(
        f"/api/projects/{project_id}/grid",
        json={"value": "anomaly", "cell_size_m": 10.0, "method": "nearest", "vmin": 1000.0, "vmax": 2000.0},
    )
    assert r.status_code == 200, r.text
    manual_range_resp = r.json()
    print("manual vmin/vmax grid response range:", manual_range_resp["vmin"], manual_range_resp["vmax"])
    assert manual_range_resp["vmin"] == 1000.0 and manual_range_resp["vmax"] == 2000.0

    # colormap selection should be honored, including the custom
    # geosoft_rainbow palette registered at app startup
    for cmap_name in ["turbo", "geosoft_rainbow"]:
        r = client.post(
            f"/api/projects/{project_id}/grid",
            json={"value": "anomaly", "cell_size_m": 10.0, "method": "nearest", "cmap": cmap_name},
        )
        assert r.status_code == 200, r.text
        assert r.json()["cmap"] == cmap_name

    r = client.post(
        f"/api/projects/{project_id}/grid",
        json={"value": "anomaly", "cell_size_m": 10.0, "method": "nearest", "show_contours": True, "contour_n_levels": 8},
    )
    assert r.status_code == 200, r.text
    contours = r.json()["contours"]
    print("n contour levels:", len(contours["levels"]), "n contour lines:", len(contours["features"]))
    assert len(contours["levels"]) > 0
    assert len(contours["features"]) > 0
    for feat in contours["features"][:3]:
        assert len(feat["path"]) >= 2
        lat, lon = feat["path"][0]
        assert 40 < lat < 55 and 100 < lon < 115  # sanity: within sample survey's rough lat/lon range

    r = client.post(
        f"/api/projects/{project_id}/grid",
        json={"value": "anomaly", "cell_size_m": 10.0, "method": "nearest", "show_contours": True, "contour_interval_nt": 200.0},
    )
    assert r.status_code == 200, r.text
    interval_contours = r.json()["contours"]
    assert all(abs((interval_contours["levels"][i + 1] - interval_contours["levels"][i]) - 200.0) < 1e-6 for i in range(len(interval_contours["levels"]) - 1))

    # line summaries should carry a centroid for map number labels
    assert all(l["centroid_lat"] is not None and l["centroid_lon"] is not None for l in summary["lines"])

    # confirm the auto max_distance (line-spacing based) fills far more of
    # the block than the old fixed 2*cell_size default would have
    from app.store import store as project_store

    project = project_store.get(project_id)
    auto_grid = project._grid_for("anomaly", 10.0, "spline", None)
    tight_grid = project._grid_for("anomaly", 10.0, "spline", 20.0)
    import numpy as np

    auto_nan_pct = float(np.isnan(auto_grid.values).mean())
    tight_nan_pct = float(np.isnan(tight_grid.values).mean())
    print(f"grid NaN%% auto={auto_nan_pct:.3f} vs old-style tight={tight_nan_pct:.3f}")
    assert auto_nan_pct < tight_nan_pct - 0.2, "line-spacing-based mask should fill noticeably more of the block"

    for transform_name in ["rtp", "rte", "1vd", "as"]:
        r = client.post(
            f"/api/projects/{project_id}/transform",
            json={"transform": transform_name, "value": "anomaly", "cell_size_m": 10.0},
        )
        assert r.status_code == 200, r.text
        tr = r.json()
        print(f"{transform_name} stats:", tr["stats"])
        assert tr["image_data_url"].startswith("data:image/png;base64,")

    # hillshade color-shaded relief option (Geosoft Oasis Montaj style)
    r = client.post(
        f"/api/projects/{project_id}/grid",
        json={"value": "anomaly", "cell_size_m": 10.0, "hillshade": True, "hillshade_exaggeration": 5.0},
    )
    assert r.status_code == 200, r.text
    assert r.json()["image_data_url"].startswith("data:image/png;base64,")

    # histogram-equalized stretch option
    r = client.post(f"/api/projects/{project_id}/grid", json={"value": "anomaly", "cell_size_m": 10.0, "stretch": "equalize"})
    assert r.status_code == 200, r.text
    eq_resp = r.json()
    assert eq_resp["stretch"] == "equalize"
    assert eq_resp["image_data_url"].startswith("data:image/png;base64,")
    r = client.post(f"/api/projects/{project_id}/grid", json={"value": "anomaly", "cell_size_m": 10.0, "stretch": "linear"})
    assert r.status_code == 200, r.text
    assert r.json()["stretch"] == "linear"

    _check_geotiff_export(client, project_id)

    r = client.post(
        f"/api/projects/{project_id}/euler-deconvolution",
        json={"value": "anomaly", "cell_size_m": 10.0, "structural_index": 1.0, "window_size_m": 100.0},
    )
    assert r.status_code == 200, r.text
    euler = r.json()
    print("euler deconvolution:", {k: v for k, v in euler.items() if k != "solutions"}, "n solutions:", len(euler["solutions"]))
    assert euler["n_solutions"] == len(euler["solutions"])
    for sol in euler["solutions"][:3]:
        assert 40 < sol["lat"] < 55 and 100 < sol["lon"] < 115
        assert sol["depth_m"] > 0

    # error path: unknown project id
    r = client.get("/api/projects/does-not-exist/summary")
    assert r.status_code == 400, r.text

    _check_overlay_image_upload(client)
    _check_inversion(client, project_id, pts)

    r = client.get(f"/api/projects/{project_id}/report")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/markdown")
    report_md = r.content.decode("utf-8")
    print("report length:", len(report_md))
    for expected in ["드론 자력탐사 자료 처리 보고서", "측선 판별 결과", "자력값 통계", "3차원 역산 결과", "오일러 디컨볼루션 결과"]:
        assert expected in report_md, f"report missing section: {expected}"
    _check_project_save_load(client, project_id, summary)

    print("\nALL CHECKS PASSED")


def _check_project_save_load(client, project_id, summary):
    """Full project save/load: resume a project (raw data, params, manual
    edits, DEM, inversion) without re-uploading or re-running anything."""
    r = client.post(
        f"/api/projects/{project_id}/manual-exclude",
        json={"mode": "lines", "action": "exclude", "line_ids": [summary["lines"][0]["line_id"]]},
    )
    assert r.status_code == 200, r.text
    excluded_summary = r.json()

    r = client.get(f"/api/projects/{project_id}/save")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    bundle = r.content

    r = client.post("/api/projects")
    fresh_id = r.json()["project_id"]
    r = client.post(f"/api/projects/{fresh_id}/load", files={"file": ("project.zip", bundle, "application/zip")})
    assert r.status_code == 200, r.text
    restored = r.json()
    assert restored["processed"] is True
    assert restored["inversion_restored"] is True
    assert restored["process_summary"]["n_kept"] == excluded_summary["n_kept"]
    assert restored["process_summary"]["n_manual_excluded"] == excluded_summary["n_manual_excluded"]

    # restored inversion must be immediately usable, not just accepted
    r = client.post(f"/api/projects/{fresh_id}/inversion/slice", json={"layer_index": 0})
    assert r.status_code == 200, r.text

    # a project with nothing uploaded yet has nothing to save
    r = client.post("/api/projects")
    empty_id = r.json()["project_id"]
    r = client.get(f"/api/projects/{empty_id}/save")
    assert r.status_code == 400, r.text

    # a corrupt/non-zip upload should fail cleanly, not 500
    r = client.post(f"/api/projects/{empty_id}/load", files={"file": ("bad.zip", b"not a zip file", "application/zip")})
    assert r.status_code == 400, r.text


def _check_inversion(client, project_id, pts):
    """3D inversion tab: run + horizontal slice + vertical section + 3D
    volume + DEM upload/use/clear + guardrail error paths."""
    r = client.post(
        f"/api/projects/{project_id}/inversion",
        json={"obs_cell_size_m": 40.0, "depth_extent_m": 150.0, "n_layers": 8},
    )
    assert r.status_code == 200, r.text
    inv_summary = r.json()
    print("inversion summary:", {k: v for k, v in inv_summary.items() if k != "field"})
    assert inv_summary["n_obs"] > 0 and inv_summary["n_active_cells"] > 0
    assert inv_summary["susceptibility_stats"]["max"] is not None

    r = client.post(f"/api/projects/{project_id}/inversion/slice", json={"layer_index": 2})
    assert r.status_code == 200, r.text
    slice_resp = r.json()
    assert slice_resp["image_data_url"].startswith("data:image/png;base64,")
    assert slice_resp["n_layers"] == 8

    r = client.post(f"/api/projects/{project_id}/inversion/slice", json={"layer_index": 2, "threshold": 0.05})
    assert r.status_code == 200, r.text

    lats = [p["lat"] for p in pts]
    lons = [p["lon"] for p in pts]
    clat, clon = sum(lats) / len(lats), sum(lons) / len(lons)
    d = 0.003
    path = [[clat - d, clon - d], [clat + d, clon + d]]
    r = client.post(f"/api/projects/{project_id}/inversion/section", json={"path": path})
    assert r.status_code == 200, r.text
    section_resp = r.json()
    assert section_resp["image_data_url"].startswith("data:image/png;base64,")
    assert len(section_resp["elevation_m"]) == 8

    r = client.get(f"/api/projects/{project_id}/inversion/volume")
    assert r.status_code == 200, r.text
    volume = r.json()
    ny, nx, nz = volume["shape"]
    assert len(volume["x"]) == ny * nx * nz == len(volume["value"])
    # isosurface is upsampled well beyond the raw 8-layer mesh for a
    # smooth render, and the observed-anomaly map is draped on the
    # terrain-following top for geographic context.
    assert ny * nx * nz > 8 * inv_summary["n_active_cells"] / inv_summary["n_layers"] * 4
    assert volume["top"] is not None
    top = volume["top"]
    top_ny, top_nx = len(top["y"]), len(top["x"])
    assert top_ny > 0 and top_nx > 0
    assert len(top["z"]) == top_ny and len(top["z"][0]) == top_nx
    assert len(top["color"]) == top_ny and len(top["color"][0]) == top_nx

    # min+max SI range filtering: values outside [threshold, threshold_max]
    # should be excluded from the returned slice/volume.
    r = client.post(
        f"/api/projects/{project_id}/inversion/slice",
        json={"layer_index": 2, "threshold": 0.02, "threshold_max": 0.1},
    )
    assert r.status_code == 200, r.text
    ranged_stats = r.json()["stats"]
    if ranged_stats["max"] is not None:
        assert ranged_stats["min"] >= 0.02 - 1e-9 and ranged_stats["max"] <= 0.1 + 1e-9

    r = client.get(f"/api/projects/{project_id}/inversion/volume", params={"threshold": 0.02, "threshold_max": 0.1})
    assert r.status_code == 200, r.text

    # fixed east-west / north-south section profiles (no manual drawing)
    r = client.post(f"/api/projects/{project_id}/inversion/section", json={"profile": "ew", "position_frac": 0.5})
    assert r.status_code == 200, r.text
    ew = r.json()
    assert ew["profile"] == "ew"
    assert ew["image_data_url"].startswith("data:image/png;base64,")

    r = client.post(f"/api/projects/{project_id}/inversion/section", json={"profile": "ns", "position_frac": 0.5})
    assert r.status_code == 200, r.text
    assert r.json()["profile"] == "ns"

    # ew/ns without position_frac, and custom without a path, are rejected
    r = client.post(f"/api/projects/{project_id}/inversion/section", json={"profile": "ew"})
    assert r.status_code == 400, r.text
    r = client.post(f"/api/projects/{project_id}/inversion/section", json={"profile": "custom"})
    assert r.status_code == 400, r.text

    # auto-parameter mode: omitting the mesh params should pick sensible
    # values from line spacing / spectral depth and run successfully.
    r = client.post(f"/api/projects/{project_id}/inversion", json={})
    assert r.status_code == 200, r.text
    auto_summary = r.json()
    print("auto-param inversion summary:", {k: v for k, v in auto_summary.items() if k != "field"})
    assert auto_summary["auto_params"] is True
    assert auto_summary["obs_cell_size_m"] > 0
    assert auto_summary["depth_extent_m"] > 0
    assert auto_summary["n_layers"] >= 4

    # re-run explicit params so the rest of this helper (DEM checks etc.)
    # operates on the same fixed mesh as before.
    r = client.post(
        f"/api/projects/{project_id}/inversion",
        json={"obs_cell_size_m": 40.0, "depth_extent_m": 150.0, "n_layers": 8},
    )
    assert r.status_code == 200, r.text

    # DEM upload: build a synthetic GeoTIFF covering the survey extent,
    # confirm it's actually used, then clear it back to GPS-AGL estimation.
    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    from app.store import store as project_store

    project = project_store.get(project_id)
    df = project.processed
    xmin, xmax = df["x"].min() - 500, df["x"].max() + 500
    ymin, ymax = df["y"].min() - 500, df["y"].max() + 500
    dem_path = "/tmp/_e2e_inversion_dem.tif"
    w, h = 60, 60
    transform = from_origin(xmin, ymax, (xmax - xmin) / w, (ymax - ymin) / h)
    crs = CRS.from_epsg(project.utm_epsg)
    elev = (1350 + 0.01 * np.random.randn(h, w)).astype("float32")
    with rasterio.open(dem_path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=crs, transform=transform) as dst:
        dst.write(elev, 1)

    with open(dem_path, "rb") as f:
        r = client.post(f"/api/projects/{project_id}/upload/dem", files={"file": ("dem.tif", f, "image/tiff")})
    assert r.status_code == 200, r.text

    r = client.post(
        f"/api/projects/{project_id}/inversion",
        json={"obs_cell_size_m": 40.0, "depth_extent_m": 150.0, "n_layers": 6},
    )
    assert r.status_code == 200, r.text
    dem_inv_summary = r.json()
    assert dem_inv_summary["used_dem"] is True

    r = client.delete(f"/api/projects/{project_id}/dem")
    assert r.status_code == 200, r.text
    assert r.json()["cleared"] is True

    # guardrail: an unreasonably fine observation grid should be rejected
    # with a friendly error rather than hanging on a huge dense mesh.
    r = client.post(
        f"/api/projects/{project_id}/inversion",
        json={"obs_cell_size_m": 2.0, "depth_extent_m": 100.0, "n_layers": 8},
    )
    assert r.status_code == 400, r.text

    # error path: slice/section before inversion has been run
    r2 = client.post("/api/projects")
    fresh_id = r2.json()["project_id"]
    r = client.post(f"/api/projects/{fresh_id}/inversion/slice", json={"layer_index": 0})
    assert r.status_code == 400, r.text

    # inversion result export/import: reload without rerunning the solve
    r = client.get(f"/api/projects/{project_id}/inversion/export")
    assert r.status_code == 200, r.text
    export_bytes = r.content
    assert r.headers["content-type"] == "application/octet-stream"

    r2 = client.post("/api/projects")
    fresh_id2 = r2.json()["project_id"]
    r = client.post(
        f"/api/projects/{fresh_id2}/inversion/import",
        files={"file": ("inversion.npz", export_bytes, "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    imported_summary = r.json()
    assert imported_summary["n_active_cells"] == dem_inv_summary["n_active_cells"]
    assert abs(imported_summary["rms_misfit_nt"] - dem_inv_summary["rms_misfit_nt"]) < 1e-6

    # the imported result must actually be usable (slice/volume), not just accepted
    r = client.post(f"/api/projects/{fresh_id2}/inversion/slice", json={"layer_index": 0})
    assert r.status_code == 200, r.text

    # importing a garbage file should fail cleanly, not 500
    r = client.post(
        f"/api/projects/{fresh_id2}/inversion/import",
        files={"file": ("bad.npz", b"not a real npz file", "application/octet-stream")},
    )
    assert r.status_code == 400, r.text

    # inversion horizontal slice as GeoTIFF (single-band, real SI values)
    import rasterio
    from rasterio.io import MemoryFile

    r = client.post(f"/api/projects/{project_id}/inversion/slice/geotiff", json={"layer_index": 2})
    assert r.status_code == 200, r.text
    with MemoryFile(r.content) as memfile, memfile.open() as src:
        assert src.crs is not None
        assert src.count == 1


def _check_geotiff_export(client, project_id):
    """Grid/derivative GeoTIFF export - single-band float32, real georeferenced
    values (not just a colored PNG), openable in Oasis Montaj/QGIS/ArcGIS/etc."""
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile

    r = client.post(f"/api/projects/{project_id}/grid/geotiff", json={"value": "anomaly", "cell_size_m": 10.0})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/tiff"
    with MemoryFile(r.content) as memfile, memfile.open() as src:
        assert src.crs is not None
        assert src.count == 1
        data = src.read(1)
        finite = data[np.isfinite(data)]
        assert finite.size > 0
        # should be real anomaly values (nT), not colormap indices/RGB bytes
        assert finite.max() > 100

    r = client.post(f"/api/projects/{project_id}/transform/geotiff", json={"transform": "rtp", "value": "anomaly", "cell_size_m": 10.0})
    assert r.status_code == 200, r.text
    with MemoryFile(r.content) as memfile, memfile.open() as src:
        assert src.count == 1

    # colored (RGBA) GeoTIFF variant - bakes in the on-screen colormap/hillshade, for viewing as-is elsewhere
    r = client.post(f"/api/projects/{project_id}/grid/geotiff", json={"value": "anomaly", "cell_size_m": 10.0, "colored": True})
    assert r.status_code == 200, r.text
    with MemoryFile(r.content) as memfile, memfile.open() as src:
        assert src.crs is not None
        assert src.count == 4
        assert src.dtypes[0] == "uint8"
        rgb = src.read([1, 2, 3])
        assert rgb.max() > 0  # not all-black/empty

    r = client.post(
        f"/api/projects/{project_id}/grid/geotiff",
        json={"value": "anomaly", "cell_size_m": 10.0, "colored": True, "hillshade": True},
    )
    assert r.status_code == 200, r.text
    with MemoryFile(r.content) as memfile, memfile.open() as src:
        assert src.count == 4

    r = client.post(
        f"/api/projects/{project_id}/transform/geotiff",
        json={"transform": "rtp", "value": "anomaly", "cell_size_m": 10.0, "colored": True},
    )
    assert r.status_code == 200, r.text
    with MemoryFile(r.content) as memfile, memfile.open() as src:
        assert src.count == 4


def _check_overlay_image_upload(client):
    """GeoTIFF overlay layer endpoint - stateless, not project-scoped."""
    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    path = "/tmp/_e2e_test_overlay.tif"
    transform = from_origin(590000, 5155000, 10, 10)
    crs = CRS.from_epsg(32648)
    data = (np.indices((50, 50)).sum(axis=0) % 3).astype("uint8")
    with rasterio.open(path, "w", driver="GTiff", height=50, width=50, count=1, dtype="uint8", crs=crs, transform=transform) as dst:
        dst.write(data, 1)
        dst.write_colormap(1, {0: (255, 0, 0, 255), 1: (0, 255, 0, 255), 2: (0, 0, 255, 255)})

    with open(path, "rb") as f:
        r = client.post("/api/overlay-images", files={"file": ("geology.tif", f, "image/tiff")})
    assert r.status_code == 200, r.text
    resp = r.json()
    print("overlay image bounds:", resp["bounds"])
    assert resp["image_data_url"].startswith("data:image/png;base64,")
    assert resp["name"] == "geology.tif"
    assert len(resp["bounds"]) == 2 and len(resp["bounds"][0]) == 2


if __name__ == "__main__":
    main()
