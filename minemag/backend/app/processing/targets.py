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
- a grade: A = seen in every run that covered the spot (a reading
  within the fit window of it) with K >= 20 (THD picks) or SNR >= 10
  (amplitude picks), B = K >= 10 or SNR >= 5, C = the rest; a fit
  quality below the grade's floor drops it one step. The denominator is
  the covering runs, not all runs: on the KIGAM site the drone flew
  38 x 67 m and the robot 20 x 31 m, and a target the robot never walked
  over cannot be held against it.

Measured on the synthetic of tests/test_targets.py (three induced
dipoles 0.2-0.6 m under a 0.5 m probe plane, 0.5 m tracks, 3 nT noise):
positions within 0.05 m, depths within 0.1 m, moments within 15%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer

from magcore.dipole_fit import _fit_dipole, classify_moment, detect_targets, field_direction_enu
from scipy.spatial import cKDTree

from .gridding import DEFAULT_MAX_DISTANCE_M, grid_survey
from .peaks import derivative_grid, find_peaks, type_class

DEFAULT_THRESHOLD_K = 5.0
# The paper's locator: THD peaks with prominence >= K x MAD. The amplitude
# method (magcore's local maxima of |anomaly| above k x noise) remains as
# an alternative.
DEFAULT_METHOD = "thd"
DEFAULT_PEAK_K = 20.0
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
# For THD picks the strength is the K the pick would need (prominence /
# MAD); the paper's references at 0.5 m were K = 65-71 for anti-tank and
# 19-20 for anti-personnel mines.
GRADE_A_K = 20.0
GRADE_B_K = 10.0


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
    method: str = DEFAULT_METHOD,
    peak_k: float = DEFAULT_PEAK_K,
    derivative: str = "thd",
    min_separation_m: float = 1.2,
    grid_method: str = "minimum_curvature",
    detrend_order: int = 1,
    inclination_declination=None,
) -> tuple[list[dict], dict]:
    """Picks of one run (one file), on its own grid and its own noise."""
    grid, _stats = grid_survey(sub, cell_size_m=cell_size_m, method=grid_method, detrend_order=detrend_order)
    noise = float(noise_rms_nt) if noise_rms_nt else _robust_std(grid.values)
    noise = max(noise, 1e-3)
    values = _levelled(sub)
    if method == "thd":
        return _pick_by_derivative(
            sub, values, grid, noise, inclination_deg, declination_deg, peak_k, derivative,
            min_separation_m, fit_window_m, max_depth_m,
        )
    thr = float(threshold_nt) if threshold_nt else max(threshold_k * noise, DEFAULT_MIN_THRESHOLD_NT)
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
    return picks, {"method": "amplitude", "noise_rms_nt": round(noise, 2), "threshold_nt": round(thr, 1), "n_picks": len(picks)}


def _pick_by_derivative(sub, values, grid, noise, inclination_deg, declination_deg, peak_k, derivative,
                        min_separation_m, fit_window_m, max_depth_m):
    """The paper's way: prominence peaks of the THD (or ASA) of the
    residual grid locate the targets; a dipole fitted to the raw readings
    around each peak adds depth and moment where the fit works."""
    deriv = derivative_grid(grid.values, grid.cell_size_m, derivative, inclination_deg, declination_deg)
    peaks, info = find_peaks(deriv, grid.values, grid.easting, grid.northing, grid.cell_size_m,
                             k=peak_k, min_separation_m=min_separation_m)
    x = sub["x"].to_numpy(dtype=float)
    y = sub["y"].to_numpy(dtype=float)
    tree = cKDTree(np.column_stack([x, y]))
    f_hat = field_direction_enu(inclination_deg, declination_deg)
    picks = []
    for p in peaks:
        idx = tree.query_ball_point([p["x"], p["y"]], r=fit_window_m)
        fit = None
        if len(idx) >= 6:
            idx = np.asarray(idx)
            depth_guess = float(np.clip(p["extent_m"] / 1.3, 0.3, 0.7 * max_depth_m))
            fit = _fit_dipole(x[idx], y[idx], values[idx], f_hat, p["x"], p["y"], depth_guess, max_depth_m, fit_window_m)
        picks.append({
            "x": p["x"], "y": p["y"],
            "fit_x": fit["x"] if fit else None, "fit_y": fit["y"] if fit else None,
            "depth_below_probe_m": fit["depth_m"] if fit else float("nan"),
            "moment_am2": fit["moment_am2"] if fit else float("nan"),
            "peak_nt": p["p2p_nt"], "p2p_nt": p["p2p_nt"], "footprint_m": p["extent_m"], "extent_m": p["extent_m"],
            "fit_quality": fit["fit_quality"] if fit else float("nan"),
            "rms_residual_nt": fit["rms_residual_nt"] if fit else float("nan"),
            "n_points": int(len(idx)),
            "snr": p["p2p_nt"] / noise if np.isfinite(p["p2p_nt"]) else float("nan"),
            "k_needed": p["k_needed"], "prominence": p["prominence"], "type_class": p["type_class"],
            "at_edge": p["at_edge"],
        })
    return picks, {"method": derivative, "noise_rms_nt": round(noise, 2), "peak_k": peak_k, "mad": info["mad"],
                   "prominence_threshold": info["threshold"], "n_local_maxima": info["n_local_maxima"], "n_picks": len(picks)}


def _grade(seen: int, n_runs: int, snr: float, quality: float, k_needed: float | None = None) -> str:
    """A: seen in every run and strong; B: strong enough; C: the rest.
    Strength is the K the pick needs when it came from the THD, the
    signal-to-noise ratio when it came from the amplitude method. A fit
    quality below the grade's floor drops the grade one step, but a pick
    without a fit is not penalised for it (the lobes of a shallow mine do
    not always fit one induced dipole)."""
    if k_needed is not None and np.isfinite(k_needed):
        strong_a, strong_b = k_needed >= GRADE_A_K, k_needed >= GRADE_B_K
    else:
        strong_a, strong_b = snr >= GRADE_A["snr"], snr >= GRADE_B["snr"]
    has_fit = quality is not None and np.isfinite(quality)
    if seen == n_runs and strong_a and (not has_fit or quality >= GRADE_A["quality"]):
        return "A"
    if strong_b and (not has_fit or quality >= GRADE_B["quality"]):
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

    def finite_mean(key):
        v = np.array([m.get(key, np.nan) for m in c["members"]], dtype=float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    for c in clusters:
        fits = [m for m in c["members"] if np.isfinite(m.get("fit_quality", np.nan))]
        best = max(fits, key=lambda m: m["fit_quality"]) if fits else None
        moments = np.array([m.get("moment_am2", np.nan) for m in c["members"]], dtype=float)
        moments = moments[np.isfinite(moments) & (moments > 0)]
        k_vals = np.array([m.get("k_needed", np.nan) for m in c["members"]], dtype=float)
        out.append({
            "x": c["x"], "y": c["y"],
            "position_spread_m": round(float(np.max([np.hypot(m["x"] - c["x"], m["y"] - c["y"]) for m in c["members"]])), 2),
            "runs_seen": sorted(c["runs"]),
            "depth_by_run": {int(r): float(m["depth_below_probe_m"]) for r, m in zip(c["runs"], c["members"])},
            "depth_below_probe_m": finite_mean("depth_below_probe_m"),
            "moment_am2": float(np.exp(np.mean(np.log(moments)))) if len(moments) else float("nan"),
            "peak_nt": float(max((m["peak_nt"] for m in c["members"]), key=abs)),
            # Noise only inflates a max-minus-min, so the smallest member is
            # the best estimate of the anomaly's own peak-to-peak.
            "p2p_nt": float(np.nanmin([m.get("p2p_nt", np.nan) for m in c["members"]]))
            if np.isfinite([m.get("p2p_nt", np.nan) for m in c["members"]]).any() else float("nan"),
            "footprint_m": finite_mean("footprint_m"),
            "extent_m": finite_mean("extent_m"),
            "fit_quality": float(best["fit_quality"]) if best else float("nan"),
            "rms_residual_nt": float(best["rms_residual_nt"]) if best else float("nan"),
            "snr": float(np.nanmax([m["snr"] for m in c["members"]])),
            "k_needed": float(np.nanmax(k_vals)) if np.isfinite(k_vals).any() else None,
            "at_edge": bool(any(m.get("at_edge") for m in c["members"])),
        })
        out[-1]["type_class"] = type_class(out[-1]["p2p_nt"]) if any(m.get("type_class") for m in c["members"]) else None
    return out


def runs_covering(df: pd.DataFrame, files: list[int], x: float, y: float, reach_m: float) -> list[int]:
    """The runs with a reading within `reach_m` of (x, y): only those could
    have seen a target there. A drone that flew the whole field and a
    robot that walked its middle do not share a denominator at the edges."""
    out = []
    for f in files:
        tree = _run_trees(df).get(f)
        if tree is not None and tree.query([x, y], k=1)[0] <= reach_m:
            out.append(f)
    return out


_TREE_CACHE: dict = {}


def _run_trees(df: pd.DataFrame) -> dict:
    key = id(df)
    if key not in _TREE_CACHE or _TREE_CACHE[key][0] != len(df):
        trees = {}
        for f, idx in df.groupby("file_index").indices.items():
            pts = np.column_stack([df["x"].to_numpy()[idx], df["y"].to_numpy()[idx]])
            trees[int(f)] = cKDTree(pts[::max(1, len(pts) // 100_000)])
        _TREE_CACHE.clear()
        _TREE_CACHE[key] = (len(df), trees)
    return _TREE_CACHE[key][1]


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
    max_targets: int | None = None,
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
    reach = DEFAULT_MAX_DISTANCE_M          # a run covers a spot it has a gridded reading for
    to_ll = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    targets = []
    for t in merged:
        lon, lat = to_ll.transform(t["x"], t["y"])
        covering = runs_covering(df, files, t["x"], t["y"], reach) or sorted(t["runs_seen"])
        t["runs_covering"] = covering
        grade = _grade(len(t["runs_seen"]), len(covering), t["snr"], t["fit_quality"], t.get("k_needed"))
        burial = [d - heights[r] for r, d in t["depth_by_run"].items() if np.isfinite(d)]
        targets.append({
            **t,
            "lat": float(lat), "lon": float(lon),
            "burial_depth_m": round(float(np.mean(burial)), 2) if burial else None,
            "size_class": classify_moment(t["moment_am2"]) if np.isfinite(t["moment_am2"]) else None,
            "n_runs": n_runs,
            "n_runs_covering": len(covering),
            "grade": grade,
        })
    # Strongest first - the K the pick needs when it came from a derived
    # grid, the signal-to-noise ratio otherwise - so a top-N cut keeps the
    # biggest, whatever their grade.
    def strength(t):
        k = t.get("k_needed")
        return float(k) if k is not None and np.isfinite(k) else float(t["snr"]) if np.isfinite(t["snr"]) else 0.0

    targets.sort(key=lambda t: -strength(t))
    n_before_cut = len(targets)
    if max_targets is not None and max_targets > 0:
        targets = targets[:max_targets]
    for i, t in enumerate(targets, start=1):
        t["id"] = i
        t["depth_by_run"] = {str(k): round(v, 3) for k, v in t["depth_by_run"].items()}
        for k in ("x", "y", "depth_below_probe_m", "moment_am2", "peak_nt", "p2p_nt", "footprint_m", "extent_m",
                  "fit_quality", "rms_residual_nt", "snr", "k_needed"):
            v = t.get(k)
            t[k] = round(float(v), 3) if v is not None and np.isfinite(float(v)) else None
    counts = {g: sum(1 for t in targets if t["grade"] == g) for g in ("A", "B", "C")}
    return {
        "n_targets": len(targets),
        "n_before_cut": n_before_cut,
        "max_targets": max_targets,
        "n_by_grade": counts,
        "n_runs": n_runs,
        "sensor_height_m": sensor_height_m,
        "sensor_heights_by_file": {str(k): v for k, v in heights.items()},
        "match_radius_m": match_radius_m,
        "runs": {str(k): v for k, v in run_info.items()},
        "grades": {"A": GRADE_A, "B": GRADE_B, "A_k": GRADE_A_K, "B_k": GRADE_B_K},
        "targets": targets,
    }


def targets_to_csv(result: dict, utm_epsg: int) -> str:
    cols = ["id", "grade", "lat", "lon", "x", "y", "p2p_nt", "k_needed", "type_class", "extent_m", "snr",
            "depth_below_probe_m", "burial_depth_m", "moment_am2", "size_class", "fit_quality", "runs_seen",
            "runs_covering", "n_runs", "position_spread_m"]
    lines = [",".join(cols).replace("x,y", f"x_epsg{utm_epsg},y_epsg{utm_epsg}")]
    for t in result["targets"]:
        row = []
        for c in cols:
            v = t.get(c)
            if isinstance(v, list):
                v = " ".join(str(i) for i in v)
            row.append(f"{v:.7f}" if c in ("lat", "lon") else ("" if v is None else str(v)))
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def targets_to_geojson(result: dict) -> dict:
    """The clearance list as points, for GIS and the field tablet."""
    features = []
    for t in result["targets"]:
        props = {k: v for k, v in t.items() if k not in ("lat", "lon", "depth_by_run")}
        props["depth_by_run"] = t.get("depth_by_run")
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [t["lon"], t["lat"]]},
                         "properties": props})
    return {"type": "FeatureCollection", "features": features,
            "properties": {k: v for k, v in result.items() if k != "targets"}}
