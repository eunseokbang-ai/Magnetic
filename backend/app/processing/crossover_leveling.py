"""Tie-line (crossover) leveling - an optional correction that only does
anything when perpendicular calibration ("tie") lines were actually flown
alongside the main parallel survey lines (see lines.py:detect_tie_lines).

At every point where a tie line crosses a survey line, both should read
the same anomaly value (real geology doesn't care which line flew over
it); any difference is a leveling error. This module estimates one
constant nT shift per survey line from its crossover differences against
the (fixed, unshifted) tie-line network - a simplified, per-line version
of the classic crossover leveling technique. It does not solve a full
simultaneous network adjustment across tie lines too, which is the
standard simplification when tie lines are treated as the reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass
class CrossoverLevelingResult:
    applied: bool
    reason: str | None
    n_tie_lines: int = 0
    n_crossovers: int = 0
    n_survey_lines_corrected: int = 0
    rms_before_nt: float | None = None
    rms_after_nt: float | None = None
    line_shifts: dict = field(default_factory=dict)  # line_id -> applied shift (nT)


def _window_value(values: np.ndarray, center_idx: int, half_window: int = 2) -> float:
    lo, hi = max(0, center_idx - half_window), min(len(values), center_idx + half_window + 1)
    return float(np.median(values[lo:hi]))


def compute_crossover_leveling(
    df: pd.DataFrame,
    tie_line_id: pd.Series,
    value_col: str,
    max_crossover_distance_m: float = 15.0,
) -> CrossoverLevelingResult:
    tie_points = df.loc[tie_line_id.to_numpy() >= 0].copy()
    tie_points["tie_line_id"] = tie_line_id[tie_line_id >= 0]
    n_tie_lines = int(tie_points["tie_line_id"].nunique()) if len(tie_points) else 0
    if n_tie_lines == 0:
        return CrossoverLevelingResult(False, "타이라인(측선과 교차하는 검교정 측선)이 탐지되지 않았습니다.")

    survey = df[df["line_id"] >= 0]
    if survey["line_id"].nunique() < 1:
        return CrossoverLevelingResult(False, "유효한 측선이 없습니다.")

    diffs_by_line: dict[int, list[float]] = {}
    n_crossovers = 0
    for tie_id, tie_sub in tie_points.groupby("tie_line_id"):
        tie_xy = tie_sub[["x", "y"]].to_numpy()
        if len(tie_xy) < 2:
            continue
        tree = cKDTree(tie_xy)
        tie_vals = tie_sub[value_col].to_numpy()
        for line_id, line_sub in survey.groupby("line_id"):
            line_xy = line_sub[["x", "y"]].to_numpy()
            dist, idx = tree.query(line_xy)
            best = int(np.argmin(dist))
            if dist[best] > max_crossover_distance_m:
                continue
            survey_val = _window_value(line_sub[value_col].to_numpy(), best)
            tie_val = _window_value(tie_vals, int(idx[best]))
            diffs_by_line.setdefault(int(line_id), []).append(tie_val - survey_val)
            n_crossovers += 1

    if not diffs_by_line:
        return CrossoverLevelingResult(
            False,
            f"타이라인 {n_tie_lines}개를 탐지했지만 {max_crossover_distance_m}m 이내에서 측선과 교차하는 지점을 찾지 못했습니다.",
            n_tie_lines=n_tie_lines,
        )

    line_shifts = {lid: float(np.median(diffs)) for lid, diffs in diffs_by_line.items()}

    before = np.concatenate([np.array(v) for v in diffs_by_line.values()])
    after = np.concatenate([np.array(v) - line_shifts[lid] for lid, v in diffs_by_line.items()])
    rms_before = float(np.sqrt(np.mean(before**2)))
    rms_after = float(np.sqrt(np.mean(after**2)))

    return CrossoverLevelingResult(
        applied=True,
        reason=None,
        n_tie_lines=n_tie_lines,
        n_crossovers=n_crossovers,
        n_survey_lines_corrected=len(line_shifts),
        rms_before_nt=rms_before,
        rms_after_nt=rms_after,
        line_shifts=line_shifts,
    )


def apply_crossover_leveling(df: pd.DataFrame, value_col: str, result: CrossoverLevelingResult) -> np.ndarray:
    values = df[value_col].to_numpy(copy=True)
    if not result.applied or not result.line_shifts:
        return values
    shift = df["line_id"].map(result.line_shifts).fillna(0.0).to_numpy()
    return values + shift
