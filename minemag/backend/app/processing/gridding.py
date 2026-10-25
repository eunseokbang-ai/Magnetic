"""Gridding for target detection: fine cells, honest coverage, and the
comparison of repeated runs.

Cells are 0.1 m by default - a mine's anomaly at 0.3-0.5 m probe height
is about a metre wide and the two probes' lines are 0.5 m apart. The
interpolation is magcore's linear gridder (block mean, then Delaunay
linear), the same engine as the mineral program, with one difference in
how far it may reach: `max_distance_m` from the nearest reading (default
0.6 m, a little over the half-spacing of a 0.5 m track pattern), beyond
which a cell is left empty. A walking robot does not hold its line, so
the gaps are real and the coverage map says where nothing was measured.

Repeated runs are compared cell by cell on block means (no interpolation,
see repeat_comparison): the rms of the difference against the rms of the
signal, and their correlation. On the KIGAM robot runs the targets
repeated and the background did not - which is how the gait noise was
found - and the difference rms is the honest repeatability to quote.

Measured on the two KIGAM robot runs (2026-04-25, 0.5 m tracks, probes
1 m apart) and the drone run; correlation of the common cells:

    method                       frame        robot1-robot2   drone-robot1
    interpolated, 0.1 m cells    UTM 52N          0.38           0.35
    interpolated, 0.1 m cells    UTM-K            0.53             -
    interpolated, 0.1 m cells    turned 1 deg     0.25             -
    block mean, 0.25 m cells     UTM 52N          0.65           0.49
    block mean, 0.25 m cells     UTM-K            0.73           0.39
    block mean, 0.25 m cells     turned 1 deg     0.71           0.50
    block mean, 0.10 m cells     UTM 52N          0.62           0.43
    block mean, 0.50 m cells     UTM 52N          0.84           0.64

The interpolated figure moves by 0.28 with the frame alone; the block
mean by 0.08. The block mean does depend on the cell (a bigger cell
averages more of the gait noise away), so the cell is part of the
number and is reported with it. Drone-robot pairs share few cells (the
drone's probe lines are 1 m apart) and their figure is correspondingly
rough.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from magcore.gridding import GridResult, grid_points

DEFAULT_CELL_SIZE_M = 0.1
DEFAULT_MAX_DISTANCE_M = 0.6
# Cell of the run-to-run comparison: about half the track spacing of a
# walking survey, so each cell holds readings of both runs.
DEFAULT_COMPARE_CELL_M = 0.25


def _select(df: pd.DataFrame, sensors, file_index) -> pd.DataFrame:
    sub = df
    if sensors:
        sub = sub[sub["sensor"].isin(list(sensors))]
    if file_index is not None:
        sub = sub[sub["file_index"] == file_index]
    sub = sub[np.isfinite(sub["tfa"]) & np.isfinite(sub["x"]) & np.isfinite(sub["y"])]
    if len(sub) < 10:
        raise ValueError("격자를 만들 측점이 너무 적습니다.")
    return sub


def _levelled(sub: pd.DataFrame, level: str) -> np.ndarray:
    """Per-file, per-probe median removed, so a probe offset or a file's
    datum does not become a step in the grid."""
    v = sub["tfa"].to_numpy(dtype=float).copy()
    if level == "median":
        for _, idx in sub.groupby(["file_index", "sensor"]).indices.items():
            v[idx] -= np.median(v[idx])
    return v


def grid_survey(
    df: pd.DataFrame,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    sensors=None,
    file_index: int | None = None,
    max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
    level: str = "median",
) -> tuple[GridResult, dict]:
    sub = _select(df, sensors, file_index)
    values = _levelled(sub, level)
    grid = grid_points(
        sub["x"].to_numpy(dtype=float), sub["y"].to_numpy(dtype=float), values,
        cell_size_m, method="linear", max_distance_m=max_distance_m,
    )
    finite = np.isfinite(grid.values)
    stats = {
        "n_points": int(len(sub)),
        "n_cells": int(grid.values.size),
        "coverage_pct": round(100.0 * float(finite.mean()), 1),
        "rms_nt": round(float(np.nanstd(grid.values)), 2) if finite.any() else None,
        "min_nt": round(float(np.nanmin(grid.values)), 1) if finite.any() else None,
        "max_nt": round(float(np.nanmax(grid.values)), 1) if finite.any() else None,
        "cell_size_m": cell_size_m,
        "max_distance_m": max_distance_m,
        "extent_m": {"x": round(float(np.ptp(grid.easting)) + cell_size_m, 1), "y": round(float(np.ptp(grid.northing)) + cell_size_m, 1)},
        "sensors": sorted(int(s) for s in sub["sensor"].unique()),
        "file_index": file_index,
        "level": level,
    }
    return grid, stats


def _block_means(x, y, v, x0, y0, cell, nx, ny):
    ix = np.floor((x - x0) / cell).astype(np.int64)
    iy = np.floor((y - y0) / cell).astype(np.int64)
    key = iy * nx + ix
    sums = np.bincount(key, weights=v, minlength=nx * ny)
    counts = np.bincount(key, minlength=nx * ny)
    means = np.full(nx * ny, np.nan)
    means[counts > 0] = sums[counts > 0] / counts[counts > 0]
    return means.reshape(ny, nx), counts.reshape(ny, nx)


def repeat_comparison(
    df: pd.DataFrame,
    compare_cell_m: float = DEFAULT_COMPARE_CELL_M,
    sensors=None,
    min_samples: int = 3,
) -> dict:
    """Every pair of files compared cell by cell, on the mean of the
    readings each run put into the same `compare_cell_m` cell, only where
    both runs have at least `min_samples` readings there.

    No interpolation on purpose. An interpolated comparison (Delaunay
    linear, 0.1 m cells) of the two KIGAM robot runs gave a correlation of
    0.53 in one projection, 0.38 in another and 0.25 with the grid turned
    by one degree - sliver triangles across the 0.5 m track gaps - so it
    measured the gridder, not the survey. Block means move by 0.08 over
    the same three frames (table in the module docstring).
    """
    files = sorted(int(f) for f in df["file_index"].unique())
    if len(files) < 2:
        return {"pairs": [], "note": "반복 측정 비교에는 파일이 둘 이상 필요합니다."}
    sub = _select(df, sensors, None)
    values = _levelled(sub, "median")
    x, y = sub["x"].to_numpy(dtype=float), sub["y"].to_numpy(dtype=float)
    cell = float(compare_cell_m)
    x0, y0 = float(x.min()), float(y.min())
    nx = int((x.max() - x0) / cell) + 2
    ny = int((y.max() - y0) / cell) + 2
    fi_all = sub["file_index"].to_numpy()
    grids = {}
    for f in files:
        m = fi_all == f
        grids[f] = _block_means(x[m], y[m], values[m], x0, y0, cell, nx, ny)
    infos = {i: f for i, f in enumerate(df.attrs.get("files") or [])}
    pairs = []
    for i, a in enumerate(files):
        for b in files[i + 1:]:
            (ga, ca), (gb, cb) = grids[a], grids[b]
            both = (ca >= min_samples) & (cb >= min_samples)
            if both.sum() < 50:
                continue
            da, db = ga[both], gb[both]
            d = da - db
            pairs.append({
                "file_a": a, "file_b": b,
                "name_a": infos.get(a, {}).get("filename"), "name_b": infos.get(b, {}).get("filename"),
                "n_cells": int(both.sum()),
                "common_area_m2": round(float(both.sum()) * cell * cell, 1),
                "diff_rms_nt": round(float(np.sqrt(np.mean(d * d))), 2),
                "diff_mean_nt": round(float(d.mean()), 2),
                "signal_rms_a_nt": round(float(da.std()), 2),
                "signal_rms_b_nt": round(float(db.std()), 2),
                "correlation": round(float(np.corrcoef(da, db)[0, 1]), 3) if da.std() > 0 and db.std() > 0 else None,
            })
    return {"compare_cell_m": cell, "min_samples": min_samples, "pairs": pairs, "note": _repeat_note(pairs)}


def _repeat_note(pairs) -> str | None:
    if not pairs:
        return None
    worst = min(p["correlation"] or 0.0 for p in pairs)
    ratio = max(p["diff_rms_nt"] / max(1e-9, 0.5 * (p["signal_rms_a_nt"] + p["signal_rms_b_nt"])) for p in pairs)
    if worst >= 0.9:
        return f"반복 상관 {worst:.2f}: 두 측정이 거의 같습니다. 차이의 rms가 이 탐사의 반복도입니다."
    if worst >= 0.7:
        return f"반복 상관 {worst:.2f}: 표적은 반복되지만 배경에 반복되지 않는 성분이 있습니다(차/신호 {ratio:.2f})."
    return (f"반복 상관 {worst:.2f}, 차/신호 {ratio:.2f}: 배경이 반복되지 않습니다. 플랫폼 잡음 QC의 스펙트럼과 "
            "탐침 상관을 확인하세요(시험장 로봇 사례 0.50).")
