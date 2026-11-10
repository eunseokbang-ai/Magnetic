"""Seminar figures from the Andong 2026-10-28 drone survey, produced by
driving DroneMag Studio's own backend in-process (no server).

Stage 1 (once, cached in fig/cache/state.pkl): upload -> INTERMAGNET
virtual base -> processing -> grid/transform GeoTIFFs -> Euler.
Stage 2 (every run): the figures, from the cached outputs. Delete the
cache to redo stage 1. Figures land in fig/, numbers in
fig/andong_summary.json.
"""
from __future__ import annotations

import io
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from PIL import Image
from rasterio.warp import Resampling, reproject, transform_bounds

sys.path.insert(0, r"C:\magnetic\backend")
sys.path.insert(0, r"C:\magnetic")
import app  # noqa: E402,F401 - sys.path shim + magcore data dir

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from magcore.vector_layer import load_vector_layer  # noqa: E402

for f in font_manager.findSystemFonts():
    if "malgun" in f.lower():
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
FIG = HERE / "fig"
CACHE = FIG / "cache"
FIG.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
DATA = Path(r"C:\magnetic\AnDong_Mag\20261028")
KMZ = next(Path(r"C:\magnetic\AnDong_Mag").glob("안동시추위치*.kmz"))
AIRBORNE = Path(r"C:\magnetic\AnDong_Mag\AnDong_RTP_EPSG5187.tif")
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
UA = "DroneMag-Studio/1.0 (seminar figures)"
ORIGIN = 20037508.342789244
DPI = 200
CELL = 5.0
KINDS = [("grid", {}), ("rtp", {}), ("as", {}), ("thdr", {}), ("1vd", {}), ("tilt", {}), ("upward_continuation", {"continuation_height_m": 50.0})]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------ basemap helpers
def tile_xy(lat, lon, z):
    n = 2 ** z
    return (lon + 180.0) / 360.0 * n, (1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n


def esri_mosaic(west, south, east, north, z):
    x0, y1 = tile_xy(south, west, z)
    x1, y0 = tile_xy(north, east, z)
    xs, ys = range(int(x0), int(x1) + 1), range(int(y0), int(y1) + 1)
    im = Image.new("RGB", (256 * len(xs), 256 * len(ys)), (40, 40, 40))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            tile_path = CACHE / "esri" / f"{z}_{x}_{y}.jpg"
            if tile_path.exists():
                im.paste(Image.open(tile_path).convert("RGB"), (256 * i, 256 * j))
                continue
            for attempt in range(4):
                try:
                    r = requests.get(ESRI.format(z=z, x=x, y=y), headers={"User-Agent": UA}, timeout=40)
                    if r.status_code == 200:
                        tile_path.parent.mkdir(exist_ok=True)
                        tile_path.write_bytes(r.content)
                        im.paste(Image.open(io.BytesIO(r.content)).convert("RGB"), (256 * i, 256 * j))
                        break
                except requests.RequestException:
                    time.sleep(2 + 3 * attempt)
    span = 2 * ORIGIN / 2 ** z
    bounds = (-ORIGIN + xs[0] * span, ORIGIN - (ys[-1] + 1) * span, -ORIGIN + (xs[-1] + 1) * span, ORIGIN - ys[0] * span)
    return im, bounds


def merc(lat, lon):
    return lon * ORIGIN / 180.0, math.log(math.tan((90 + lat) * math.pi / 360.0)) * ORIGIN / math.pi


# ------------------------------------------------------------ stage 1: pipeline
def run_pipeline() -> dict:
    from fastapi.testclient import TestClient

    from app.main import app as fastapi_app

    client = TestClient(fastapi_app)
    pid = client.post("/api/projects").json()["project_id"]
    files = sorted(DATA.glob("*.csv"))
    log("uploading", len(files), "files")
    r = client.post(f"/api/projects/{pid}/upload/drone", files=[("files", (f.name, f.read_bytes(), "text/csv")) for f in files])
    assert r.status_code == 200, r.text
    drone = r.json()
    log("INTERMAGNET nearest fetch")
    r = client.post(f"/api/projects/{pid}/base/intermagnet/nearest/fetch", json={"n_stations": 4, "max_distance_km": 2000.0})
    assert r.status_code == 200, r.text
    near = r.json()
    base_csv = client.get(f"/api/projects/{pid}/base/intermagnet/nearest/csv").content
    assert client.post(f"/api/projects/{pid}/base/intermagnet/nearest/apply").status_code == 200
    log("processing")
    r = client.post(f"/api/projects/{pid}/process", json={"filter_cutoff_hz": 1.0, "line_params": {},
                                                          "diurnal_params": {"time_offset_seconds": 0.0, "reference": "mean"},
                                                          "heading_correction": {"enabled": True, "quiet_percentile": 40.0}})
    assert r.status_code == 200, r.text
    proc = r.json()
    pts_csv = client.get(f"/api/projects/{pid}/points/csv").content
    grids = {}
    for kind, extra in KINDS:
        log("grid", kind)
        if kind == "grid":
            r = client.post(f"/api/projects/{pid}/grid/geotiff", json={"value": "anomaly", "cell_size_m": CELL})
        else:
            r = client.post(f"/api/projects/{pid}/transform/geotiff", json={"transform": kind, "value": "anomaly", "cell_size_m": CELL, **extra})
        assert r.status_code == 200, (kind, r.text[:300])
        grids[kind] = r.content
    log("euler")
    r = client.post(f"/api/projects/{pid}/euler-deconvolution", json={"cell_size_m": CELL, "structural_index": 1.0, "window_size_m": 100.0})
    euler = r.json() if r.status_code == 200 else None
    return {"drone": drone, "near": near, "base_csv": base_csv, "proc": proc, "pts_csv": pts_csv, "grids": grids, "euler": euler, "n_files": len(files)}


state_file = CACHE / "state.pkl"
if state_file.exists():
    state = pickle.loads(state_file.read_bytes())
    log("using cached pipeline outputs")
else:
    state = run_pipeline()
    state_file.write_bytes(pickle.dumps(state))
    log("pipeline outputs cached")

drone, near, proc, euler = state["drone"], state["near"], state["proc"], state["euler"]
base = pd.read_csv(io.BytesIO(state["base_csv"]))
pts = pd.read_csv(io.BytesIO(state["pts_csv"]))
log("base columns", list(base.columns))
log("points columns", list(pts.columns))
(FIG / "andong_process_summary.json").write_text(json.dumps(proc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

summary = {
    "n_files": state["n_files"],
    "drone": {k: v for k, v in drone.items() if not isinstance(v, (list, dict))},
    "intermagnet": {"stations": [{k: s.get(k) for k in ("iaga_code", "station_name", "distance_km", "bearing_deg")} for s in near["stations"]],
                    "n_points": near["n_points"], "time_range": near["time_range"], "dropped": near.get("dropped_stations")},
    "process": {k: proc.get(k) for k in ("n_points", "n_lines", "n_kept", "n_excluded_auto", "line_spacing_m", "utm_epsg", "dominant_azimuth_deg",
                                         "inclination_deg", "declination_deg", "despike", "heading_correction", "statistical_leveling", "diurnal", "anomaly_stats", "tmi_stats")},
    "noise_qc": {k: v for k, v in (proc.get("noise_qc") or {}).items() if not isinstance(v, list)},
}


def _open(data):
    with rasterio.open(io.BytesIO(data)) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.bounds, src.crs


grids = {k: _open(v) for k, v in state["grids"].items()}
for k, (arr, b, crs) in grids.items():
    summary.setdefault("grids", {})[k] = {"shape": list(arr.shape), "crs": str(crs), "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr))}

# ------------------------------------------------------------ stage 2: figures
anom, b, crs = grids["grid"]
x0, x1, y0, y1 = b.left, b.right, b.bottom, b.top
ext_km = (0, (x1 - x0) / 1000, 0, (y1 - y0) / 1000)


def panel(ax, arr, title, unit, cmap="RdBu_r", sym=True, contour=False):
    v = arr[np.isfinite(arr)]
    lo, hi = np.percentile(v, [1, 99])
    if sym:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    im = ax.imshow(arr, extent=ext_km, origin="upper", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    if contour:
        ax.contour(np.linspace(ext_km[0], ext_km[1], arr.shape[1]), np.linspace(ext_km[3], ext_km[2], arr.shape[0]), arr, levels=12, colors="k", linewidths=0.3, alpha=0.6)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("동 (km)")
    ax.set_ylabel("북 (km)")
    ax.set_aspect("equal")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(unit, fontsize=9)
    cb.ax.tick_params(labelsize=8)


def fig_anomaly():
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    panel(ax, anom, "총자기장 이상 (일변화·IGRF·레벨링 후, 셀 5 m)", "nT", contour=True)
    fig.tight_layout()
    fig.savefig(FIG / "andong_anomaly.png", dpi=DPI)
    plt.close(fig)


def fig_derivatives():
    spec = [("rtp", "극자화 변환 (RTP)", "nT", "RdBu_r", True), ("as", "해석신호 (AS)", "nT/m", "magma", False),
            ("thdr", "총수평미분 (THDR)", "nT/m", "magma", False), ("1vd", "1차 수직미분", "nT/m", "RdBu_r", True),
            ("tilt", "틸트각", "°", "RdBu_r", True), ("upward_continuation", "상향연속 50 m", "nT", "RdBu_r", True)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))
    for ax, (k, t, u, cm, sym) in zip(axes.ravel(), spec):
        panel(ax, grids[k][0], t, u, cm, sym)
    fig.tight_layout()
    fig.savefig(FIG / "andong_derivatives.png", dpi=DPI)
    plt.close(fig)


def _latlon_cols():
    latc = next(c for c in pts.columns if c.lower().startswith("lat"))
    lonc = next(c for c in pts.columns if c.lower().startswith("lon"))
    return latc, lonc


def fig_lines():
    latc, lonc = _latlon_cols()
    valc = "tmi" if "tmi" in pts.columns else next(c for c in pts.columns if "mag" in c.lower() or "tmi" in c.lower())
    lat, lon, val = pts[latc].to_numpy(), pts[lonc].to_numpy(), pts[valc].to_numpy()
    pad = 0.004
    base_img, mb = esri_mosaic(lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad, 16)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(base_img, extent=(mb[0], mb[2], mb[1], mb[3]))
    mx, my = zip(*[merc(a, o) for a, o in zip(lat[::5], lon[::5])])
    sc = ax.scatter(mx, my, c=val[::5], s=1.2, cmap="RdBu_r", vmin=np.percentile(val, 2), vmax=np.percentile(val, 98))
    xmin, ymin = merc(lat.min() - pad, lon.min() - pad)
    xmax, ymax = merc(lat.max() + pad, lon.max() + pad)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_xticks([]); ax.set_yticks([])
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("총자기장 (nT)")
    ax.set_title(f"비행 측선과 원시 총자기장 — {state['n_files']}개 비행, {len(pts):,}점", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "andong_lines.png", dpi=DPI)
    plt.close(fig)


def fig_base():
    tcol = next((c for c in base.columns if any(k in c.lower() for k in ("time", "date", "utc", "timestamp"))), base.columns[0])
    vcols = [c for c in base.columns if c != tcol and pd.api.types.is_numeric_dtype(base[c])]
    ycol = next((c for c in vcols if any(k in c.lower() for k in ("estimate", "combined", "total", "tmi", "base", "value"))), vcols[0])
    t = pd.to_datetime(base[tcol], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(t, base[ycol] - base[ycol].median(), lw=0.8, color="#1f4e79")
    ax.set_ylabel("일변화 (nT, 중앙값 기준)")
    ax.set_xlabel("시각 (UTC)")
    codes = ", ".join(s["iaga_code"] for s in near["stations"])
    ax.set_title(f"INTERMAGNET 4개 관측소 거리가중 결합 가상 기준국 — {codes}", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "andong_base.png", dpi=DPI)
    plt.close(fig)


def _holes():
    return load_vector_layer(KMZ.read_bytes(), KMZ.name)["features"]


def fig_anomaly_on_imagery():
    with rasterio.open(io.BytesIO(state["grids"]["grid"])) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        bimg, mb2 = esri_mosaic(west - 0.003, south - 0.003, east + 0.003, north + 0.003, 16)
        w, h = bimg.size
        dst_tr = rasterio.transform.from_bounds(*mb2, w, h)
        out = np.full((h, w), np.nan, dtype="float32")
        reproject(rasterio.band(src, 1), out, dst_transform=dst_tr, dst_crs="EPSG:3857", resampling=Resampling.bilinear, dst_nodata=np.nan)
        if src.nodata is not None:
            out[out == src.nodata] = np.nan
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(bimg, extent=(mb2[0], mb2[2], mb2[1], mb2[3]))
    v = out[np.isfinite(out)]
    m = max(abs(np.percentile(v, 1)), abs(np.percentile(v, 99)))
    im = ax.imshow(out, extent=(mb2[0], mb2[2], mb2[1], mb2[3]), cmap="RdBu_r", vmin=-m, vmax=m, alpha=0.65)
    for f in _holes():
        la, lo = f["coordinates"][0]
        hx, hy = merc(la, lo)
        ax.plot(hx, hy, marker="o", ms=9, mfc="yellow", mec="k")
        ax.annotate("시추", (hx, hy), xytext=(6, 6), textcoords="offset points", fontsize=9, color="yellow", weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("자기이상 (nT)")
    ax.set_title("자기이상 격자 + 위성영상·시추 위치 (참조 레이어)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "andong_anomaly_on_imagery.png", dpi=DPI)
    plt.close(fig)


def fig_airborne_vs_drone():
    with rasterio.open(io.BytesIO(state["grids"]["rtp"])) as src:
        rw, rs, re_, rn = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        padd = 0.006
        win_w, win_s, win_e, win_n = rw - padd, rs - padd, re_ + padd, rn + padd
        mw, ms = merc(win_s, win_w)
        me, mn = merc(win_n, win_e)
        W = 900
        H = int(900 * (mn - ms) / (me - mw))
        dst_tr = rasterio.transform.from_bounds(mw, ms, me, mn, W, H)
        out2 = np.full((H, W), np.nan, dtype="float32")
        reproject(rasterio.band(src, 1), out2, dst_transform=dst_tr, dst_crs="EPSG:3857", resampling=Resampling.bilinear, dst_nodata=np.nan)
        if src.nodata is not None:
            out2[out2 == src.nodata] = np.nan
    with rasterio.open(AIRBORNE) as src:
        air = np.zeros((4, H, W), dtype="uint8")
        for bnd in range(4):
            reproject(rasterio.band(src, bnd + 1), air[bnd], dst_transform=dst_tr, dst_crs="EPSG:3857", resampling=Resampling.bilinear)
    bimg3, mb3 = esri_mosaic(win_w, win_s, win_e, win_n, 15)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(bimg3, extent=(mb3[0], mb3[2], mb3[1], mb3[3]))
    axes[0].imshow(np.moveaxis(air, 0, -1), extent=(mw, me, ms, mn))
    axes[0].set_title("기존 항공 자력탐사 RTP (측선 간격 수백 m, 고도 수백 m)", fontsize=11)
    axes[1].imshow(bimg3, extent=(mb3[0], mb3[2], mb3[1], mb3[3]))
    v = out2[np.isfinite(out2)]
    m = max(abs(np.percentile(v, 1)), abs(np.percentile(v, 99)))
    im = axes[1].imshow(out2, extent=(mw, me, ms, mn), cmap="RdBu_r", vmin=-m, vmax=m, alpha=0.8)
    axes[1].set_title("드론 자력탐사 RTP (DroneMag Studio, 셀 5 m)", fontsize=11)
    for ax in axes:
        ax.set_xlim(mw, me); ax.set_ylim(ms, mn); ax.set_xticks([]); ax.set_yticks([])
        for f in _holes():
            la, lo = f["coordinates"][0]
            hx, hy = merc(la, lo)
            ax.plot(hx, hy, marker="o", ms=7, mfc="yellow", mec="k")
    cb = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    cb.set_label("RTP (nT)")
    fig.tight_layout()
    fig.savefig(FIG / "andong_airborne_vs_drone.png", dpi=DPI)
    plt.close(fig)


def fig_euler():
    if not euler or not isinstance(euler.get("solutions"), list) or not euler["solutions"]:
        log("no euler solutions")
        return
    sol = pd.DataFrame(euler["solutions"])
    log("euler columns", list(sol.columns))
    dc = next(c for c in sol.columns if c.startswith("depth"))
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    sx, sy = tr.transform(sol["lon"].to_numpy(), sol["lat"].to_numpy())
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    v = anom[np.isfinite(anom)]; m = max(abs(np.percentile(v, 1)), abs(np.percentile(v, 99)))
    ax.imshow(anom, extent=ext_km, origin="upper", cmap="Greys", vmin=-m, vmax=m, interpolation="nearest", alpha=0.6)
    ax.set_title("오일러 디컨볼루션 해 (SI=1, 창 100 m) — 바탕: 자기이상(회색)", fontsize=11)
    ax.set_xlabel("동 (km)"); ax.set_ylabel("북 (km)"); ax.set_aspect("equal")
    order = np.argsort(sol[dc].to_numpy())[::-1]
    sc = ax.scatter((sx[order] - x0) / 1000, (sy[order] - y0) / 1000, c=sol[dc].to_numpy()[order], cmap="viridis", s=9, alpha=0.85, linewidth=0)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("심도 (m)")
    fig.tight_layout()
    fig.savefig(FIG / "andong_euler.png", dpi=DPI)
    plt.close(fig)
    summary["euler"] = {"n_solutions": int(len(sol)), "depth_median_m": float(sol[dc].median()), "depth_stats": euler.get("depth_stats")}


def fig_profile():
    latc, lonc = _latlon_cols()
    rawc = next((c for c in pts.columns if "raw" in c.lower()), None)
    procc = next((c for c in pts.columns if c.lower() in ("anomaly", "anomaly_nt", "mag_filtered", "tmi_processed")), None)
    linec = next((c for c in pts.columns if "line" in c.lower()), None)
    if linec is None or procc is None:
        log("profile skipped: columns", list(pts.columns))
        return
    valid = pts[pts[linec] >= 0]
    lid = valid.groupby(linec).size().idxmax()
    g = pts[pts[linec] == lid]
    d = np.r_[0, np.cumsum(np.hypot(np.diff(g[lonc]) * 111320 * np.cos(np.radians(g[latc].mean())), np.diff(g[latc]) * 110540))]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    if "tmi_nt" in g.columns:
        ax.plot(d, g["tmi_nt"] - g["tmi_nt"].median(), color="#999", lw=0.7, label="총자기장 (중앙값 기준)")
    ax.plot(d, g[procc], color="#b42318", lw=1.0, label="자기이상 (일변화·IGRF·레벨링 후)")
    ax.set_xlabel("측선 거리 (m)"); ax.set_ylabel("nT"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title(f"측선 {lid}: 총자기장과 처리 후 자기이상", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "andong_profile.png", dpi=DPI)
    plt.close(fig)


for fn in (fig_anomaly, fig_derivatives, fig_lines, fig_base, fig_anomaly_on_imagery, fig_airborne_vs_drone, fig_euler, fig_profile):
    try:
        fn()
        log("ok", fn.__name__)
    except Exception as exc:  # noqa: BLE001 - one figure must not kill the rest
        import traceback
        log("FAILED", fn.__name__, repr(exc))
        traceback.print_exc()

(FIG / "andong_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
log("done:", sorted(p.name for p in FIG.glob("andong_*.png")))
