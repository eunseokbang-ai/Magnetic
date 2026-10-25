"""Target picking: every compact anomaly fitted as an induced dipole,
graded, and - when the ground was surveyed more than once - checked
against the other runs.

The fit itself is magcore.dipole_fit.detect_targets, the same model the
mineral program uses in its target mode (one induced dipole per
anomaly, position, depth below the probe plane and moment solved on the
raw readings near the peak). What this module adds is what a clearance
list needs and a one-off survey does not:

- the detection threshold comes from the platform's own quiet-zone noise
  (processing/platform_qc.py), `threshold_k` times its rms, so a quiet
  drone flight picks 35 nT anomalies and a walking robot with 50 nT of
  gait noise is not flooded with false picks;
- each run is picked separately and the picks are matched across runs
  (`match_radius_m`); "seen in 2 of 2 runs" is worth more than any fit
  statistic, because the gait noise does not repeat and the targets do;
- the probe height above ground turns the fitted depth below the probe
  into a burial depth;
- a grade: A = seen in every run (or the only run) with signal-to-noise
  >= 10 and fit quality >= 0.5; B = SNR >= 5 and quality >= 0.3; C = the
  rest. The thresholds are the KIGAM test site's: its four known targets
  come out A on the drone and on both robot runs; the robot's noise
  blobs that pass the threshold at all come out C.

Measured on the synthetic of tests/test_targets.py (three induced
dipoles 0.2-0.6 m under a 0.5 m probe plane, 0.5 m tracks, 3 nT noise):
positions within 0.05 m, depths within 0.1 m, moments within 15%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer

from magcore.dipole_fit import classify_moment, detect_targets

from .gridding import grid_survey

DEFAULT_THRESHOLD_K = 5.0
DEFAULT_MIN_THRESHOLD_NT = 20.0
DEFAULT_SENSOR_HEIGHT_M = 0.5
DEFAULT_MATCH_RADIUS_M = 0.5
DEFAULT_FIT_WINDOW_M = 1.5
DEFAULT_MAX_DEPTH_M = 2.0
DEFAULT_MIN_FOOTPRINT_M = 0.15
DEFAULT_MAX_FOOTPRINT_M = 3.0
DEFAULT_MIN_FIT_QUALITY = 0.3
GRADE_A = {"snr": 10.0, "quality": 0.5}
GRADE_B = {"snr": 5.0, "quality": 0.3}


def _robust_std(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else 0.0


def _levelled(sub: pd.DataFrame) -> np.ndarray:
    v = sub["tfa"].to_numpy(dtype=float).copy()
    for _, idx in sub.groupby(["file_index", "sensor"]).indices.items():
        v[idx] -= np.median(v[idx])
    return v


def pick_targets_in_run(
    sub: pd.DataFrame,
    inclination_deg: float,
    declination_deg: float,
    noise_rms_nt: float | None,
    *,
    cell_size_m: float = 0.1,
    threshold_k: float = DEFAULT_THRESHOLD_K,
    threshold_nt: float | None = None,
    fit_window_m: float = DEFAULT_FIT_WINDOW_M,
    max_depth_m: float = DEFAULT_MAX_DEPTH_M,
    min_footprint_m: float = DEFAULT_MIN_FOOTPRINT_M,
    max_footprint_m: float = DEFAULT_MAX_FOOTPRINT_M,
    min_fit_quality: float = DEFAULT_MIN_FIT_QUALITY,
) -> tuple[list[dict], dict]:
    """Picks of one run (one file), on its own grid and its own noise."""
    grid, _stats = grid_survey(sub, cell_size_m=cell_size_m)
    noise = float(noise_rms_nt) if noise_rms_nt else _robust_std(grid.values)
    noise = max(noise, 1e-3)
    thr = float(threshold_nt) if threshold_nt else max(threshold_k * noise, DEFAULT_MIN_THRESHOLD_NT)
    values = _levelled(sub)
    found = detect_targets(
        sub["x"].to_numpy(dtype=float), sub["y"].to_numpy(dtype=float), values,
        grid.values, grid.easting, grid.northing, grid.cell_size_m,
        inclination_deg, declination_deg,
        threshold_nt=thr, min_footprint_m=min_footprint_m, max_footprint_m=max_footprint_m,
        fit_window_m=fit_window_m, max_depth_m=max_depth_m, min_fit_quality=min_fit_quality,
    )
    picks = []
    for t in found:
        picks.append({
            "x": t.x, "y": t.y, "depth_below_probe_m": t.depth_m, "moment_am2": t.moment_am2,
            "peak_nt": t.peak_anomaly_nt, "footprint_m": t.footprint_m, "fit_quality": t.fit_quality,
            "rms_residual_nt": t.rms_residual_nt, "n_points": t.n_points,
            "snr": abs(t.peak_anomaly_nt) / noise,
        })
    return picks, {"noise_rms_nt": round(noise, 2), "threshold_nt": round(thr, 1), "n_picks": len(picks)}


def _grade(seen: int, n_runs: int, snr: float, quality: float) -> str:
    if seen == n_runs and snr >= GRADE_A["snr"] and quality >= GRADE_A["quality"]:
        return "A"
    if snr >= GRADE_B["snr"] and quality >= GRADE_B["quality"]:
        return "B"
    return "C"


def merge_runs(per_run: dict[int, list[dict]], match_radius_m: float) -> list[dict]:
    """Picks of all runs clustered within `match_radius_m`: one target per
    cluster, its position the mean of the members, its statistics the
    best member's, and the list of runs it was seen in."""
    runs = sorted(per_run)
    clusters: list[dict] = []
    for r in runs:
        for p in per_run[r]:
            best = None
            for c in clusters:
                d = np.hypot(c["x"] - p["x"], c["y"] - p["y"])
                if d <= match_radius_m and r not in c["runs"] and (best is None or d < best[0]):
                    best = (d, c)
            if best is None:
                clusters.append({"x": p["x"], "y": p["y"], "runs": [r], "members": [p]})
            else:
                c = best[1]
                c["members"].append(p)
                c["runs"].append(r)
                c["x"] = float(np.mean([m["x"] for m in c["members"]]))
                c["y"] = float(np.mean([m["y"] for m in c["members"]]))
    out = []
    for c in clusters:
        best = max(c["members"], key=lambda m: m["fit_quality"])
        out.append({
            "x": c["x"], "y": c["y"],
            "position_spread_m": round(float(np.max([np.hypot(m["x"] - c["x"], m["y"] - c["y"]) for m in c["members"]])), 2),
            "runs_seen": sorted(c["runs"]),
            "depth_by_run": {int(r): float(m["depth_below_probe_m"]) for r, m in zip(c["runs"], c["members"])},
            "depth_below_probe_m": float(np.mean([m["depth_below_probe_m"] for m in c["members"]])),
            "moment_am2": float(np.exp(np.mean([np.log(max(m["moment_am2"], 1e-9)) for m in c["members"]]))),
            "peak_nt": float(max((m["peak_nt"] for m in c["members"]), key=abs)),
            "footprint_m": float(np.mean([m["footprint_m"] for m in c["members"]])),
            "fit_quality": float(best["fit_quality"]),
            "rms_residual_nt": float(best["rms_residual_nt"]),
            "snr": float(max(m["snr"] for m in c["members"])),
        })
    return out


def pick_targets(
    df: pd.DataFrame,
    inclination_deg: float,
    declination_deg: float,
    utm_epsg: int,
    noise_by_file: dict[int, float] | None = None,
    *,
    sensor_height_m: float = DEFAULT_SENSOR_HEIGHT_M,
    sensor_heights_by_file: dict[int, float] | None = None,
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
    **run_kwargs,
) -> dict:
    """`sensor_height_m` is the probe height above ground for every run
    unless `sensor_heights_by_file` names a run's own (a drone at 0.5 m
    and a robot at 0.3 m surveyed the KIGAM site); burial depth is the
    mean over the runs that saw the target of (depth below probe - that
    run's probe height)."""
    files = sorted(int(f) for f in df["file_index"].unique())
    heights = {f: float((sensor_heights_by_file or {}).get(f, sensor_height_m)) for f in files}
    per_run, run_info = {}, {}
    for f in files:
        sub = df[df["file_index"] == f]
        noise = (noise_by_file or {}).get(f)
        picks, info = pick_targets_in_run(sub, inclination_deg, declination_deg, noise, **run_kwargs)
        per_run[f] = picks
        run_info[f] = info
    merged = merge_runs(per_run, match_radius_m)
    n_runs = len(files)
    to_ll = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    targets = []
    for t in merged:
        lon, lat = to_ll.transform(t["x"], t["y"])
        grade = _grade(len(t["runs_seen"]), n_runs, t["snr"], t["fit_quality"])
        targets.append({
            **t,
            "lat": float(lat), "lon": float(lon),
            "burial_depth_m": round(float(np.mean([d - heights[r] for r, d in t["depth_by_run"].items()])), 2),
            "size_class": classify_moment(t["moment_am2"]),
            "n_runs": n_runs,
            "grade": grade,
        })
    order = {"A": 0, "B": 1, "C": 2}
    targets.sort(key=lambda t: (order[t["grade"]], -abs(t["peak_nt"])))
    for i, t in enumerate(targets, start=1):
        t["id"] = i
        t["depth_by_run"] = {str(k): round(v, 3) for k, v in t["depth_by_run"].items()}
        for k in ("x", "y", "depth_below_probe_m", "moment_am2", "peak_nt", "footprint_m", "fit_quality",
                  "rms_residual_nt", "snr"):
            t[k] = round(float(t[k]), 3)
    counts = {g: sum(1 for t in targets if t["grade"] == g) for g in ("A", "B", "C")}
    return {
        "n_targets": len(targets),
        "n_by_grade": counts,
        "n_runs": n_runs,
        "sensor_height_m": sensor_height_m,
        "sensor_heights_by_file": {str(k): v for k, v in heights.items()},
        "match_radius_m": match_radius_m,
        "runs": {str(k): v for k, v in run_info.items()},
        "grades": {"A": GRADE_A, "B": GRADE_B},
        "targets": targets,
    }


def targets_to_csv(result: dict, utm_epsg: int) -> str:
    cols = ["id", "grade", "lat", "lon", "x", "y", "peak_nt", "snr", "depth_below_probe_m", "burial_depth_m",
            "moment_am2", "size_class", "fit_quality", "footprint_m", "runs_seen", "n_runs", "position_spread_m"]
    lines = [",".join(cols).replace("x,y", f"x_epsg{utm_epsg},y_epsg{utm_epsg}")]
    for t in result["targets"]:
        row = []
        for c in cols:
            v = t.get(c)
            if isinstance(v, list):
                v = " ".join(str(i) for i in v)
            row.append(f"{v:.7f}" if c in ("lat", "lon") else str(v))
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"
