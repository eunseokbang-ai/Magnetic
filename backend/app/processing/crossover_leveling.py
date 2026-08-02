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
    line_shifts: dict = field(default_factory=dict)  # line_id -> applied constant shift (nT)
    # Populated only when leveling_order=1 (polynomial/linear levelling):
    # line_id -> (intercept_nt, slope_nt_per_m, centroid_x, centroid_y,
    # direction_x, direction_y) - a spatially-varying shift(s) = intercept
    # + slope*s along the line's own principal axis, instead of one
    # constant per line. None (default) means every applied correction is
    # the plain constant in line_shifts above.
    line_shifts_linear: dict | None = None


def _window_value(values: np.ndarray, center_idx: int, half_window: int = 2) -> float:
    lo, hi = max(0, center_idx - half_window), min(len(values), center_idx + half_window + 1)
    return float(np.median(values[lo:hi]))


def _line_axis(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and principal-axis unit direction of a point cloud (via
    PCA), used to define a consistent along-line arc-length coordinate s
    for the polynomial (leveling_order=1) correction below."""
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    return centroid, direction


def compute_crossover_leveling(
    df: pd.DataFrame,
    tie_line_id: pd.Series,
    value_col: str,
    max_crossover_distance_m: float = 15.0,
    iterative: bool = True,
    max_iterations: int = 20,
    tol_nt: float = 0.01,
    leveling_order: int = 0,
) -> CrossoverLevelingResult:
    """iterative=True (default) solves a small network adjustment instead
    of treating the tie-line network as a perfect fixed reference: survey
    and tie line shifts are refined in alternating passes (each survey
    line's shift is re-estimated against the tie lines' *current* shift
    estimate, then each tie line's shift against the survey lines'
    current estimate, and so on) until the largest per-line change drops
    below tol_nt or max_iterations is reached. A constant offset added to
    every line (survey and tie alike) doesn't change any crossover
    difference, so that overall mean shift is unobservable and is
    subtracted out after every pass to anchor the solution - the network
    equivalent of the single-pass version's "tie lines are fixed at 0"
    anchor, but let error distribute across both line families instead of
    assuming the tie-line measurements are error-free. Tie-line points are
    never written back into the survey grid (they're excluded from
    gridding entirely, see store.py), so only the resulting survey line
    shifts are returned/applied - the tie shifts only exist internally to
    make those survey shifts more robust.

    iterative=False keeps the original simplified behavior: one median
    (tie value - survey value) shift per survey line against the raw,
    unshifted tie-line crossover values.

    leveling_order: 0 (default) = the plain constant-shift-per-line
    ("zero order") behavior above, either iterative or not. 1 =
    polynomial/linear levelling (per the UAV magnetics guidelines):
    instead of one constant shift per survey line, fits shift(s) =
    intercept + slope*s, where s is the crossover point's own position
    along that line's principal axis - lets the correction vary smoothly
    along a line instead of only as a DC jump, at the cost of needing at
    least 2 tie-line crossings per survey line to fit a slope (falls back
    to a plain constant for any line with fewer). Only supported in the
    non-iterative form: fitting a slope through the alternating
    survey/tie network refinement above would need re-deriving the whole
    scheme, so leveling_order=1 always uses the single-pass fit against
    the raw, unshifted tie-line values (mirrors iterative=False's scope,
    regardless of what iterative is set to).
    """
    tie_points = df.loc[tie_line_id.to_numpy() >= 0].copy()
    tie_points["tie_line_id"] = tie_line_id[tie_line_id >= 0]
    n_tie_lines = int(tie_points["tie_line_id"].nunique()) if len(tie_points) else 0
    if n_tie_lines == 0:
        return CrossoverLevelingResult(False, "타이라인(측선과 교차하는 검교정 측선)이 탐지되지 않았습니다.")

    survey = df[df["line_id"] >= 0]
    if survey["line_id"].nunique() < 1:
        return CrossoverLevelingResult(False, "유효한 측선이 없습니다.")

    # (survey_line_id, tie_line_id, survey_val, tie_val, crossover_x, crossover_y)
    records: list[tuple[int, int, float, float, float, float]] = []
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
            cx, cy = line_xy[best]
            records.append((int(line_id), int(tie_id), survey_val, tie_val, float(cx), float(cy)))

    if not records:
        return CrossoverLevelingResult(
            False,
            f"타이라인 {n_tie_lines}개를 탐지했지만 {max_crossover_distance_m}m 이내에서 측선과 교차하는 지점을 찾지 못했습니다.",
            n_tie_lines=n_tie_lines,
        )

    if leveling_order == 1:
        line_axes = {int(lid): _line_axis(sub[["x", "y"]].to_numpy()) for lid, sub in survey.groupby("line_id")}

        by_survey_pts: dict[int, list[tuple[float, float]]] = {}
        for lid, _tid, sv, tv, cx, cy in records:
            centroid, direction = line_axes[lid]
            s = float(np.dot(np.array([cx, cy]) - centroid, direction))
            by_survey_pts.setdefault(lid, []).append((s, tv - sv))

        survey_shift: dict[int, float] = {}
        survey_shift_linear: dict[int, tuple] = {}
        for lid, pts in by_survey_pts.items():
            centroid, direction = line_axes[lid]
            s_arr = np.array([p[0] for p in pts])
            d_arr = np.array([p[1] for p in pts])
            if len(pts) >= 2 and np.ptp(s_arr) > 1e-6:
                slope, intercept = np.polyfit(s_arr, d_arr, 1)
            else:
                intercept, slope = float(np.median(d_arr)), 0.0
            survey_shift[lid] = float(intercept)
            survey_shift_linear[lid] = (
                float(intercept), float(slope), float(centroid[0]), float(centroid[1]),
                float(direction[0]), float(direction[1]),
            )
        tie_shift: dict[int, float] = {}

        def _applied(lid, cx, cy):
            intercept, slope, cx0, cy0, dx, dy = survey_shift_linear[lid]
            s = (cx - cx0) * dx + (cy - cy0) * dy
            return intercept + slope * s

        before = np.array([tv - sv for _lid, _tid, sv, tv, _cx, _cy in records])
        after = np.array([(tv) - (sv + _applied(lid, cx, cy)) for lid, _tid, sv, tv, cx, cy in records])
        rms_before = float(np.sqrt(np.mean(before**2)))
        rms_after = float(np.sqrt(np.mean(after**2)))

        return CrossoverLevelingResult(
            applied=True,
            reason=None,
            n_tie_lines=n_tie_lines,
            n_crossovers=len(records),
            n_survey_lines_corrected=len(survey_shift),
            rms_before_nt=rms_before,
            rms_after_nt=rms_after,
            line_shifts=survey_shift,
            line_shifts_linear=survey_shift_linear,
        )

    if iterative:
        survey_ids = sorted({lid for lid, _tid, _sv, _tv, _cx, _cy in records})
        tie_ids = sorted({tid for _lid, tid, _sv, _tv, _cx, _cy in records})
        survey_shift = {lid: 0.0 for lid in survey_ids}
        tie_shift = {tid: 0.0 for tid in tie_ids}

        for _ in range(max(1, max_iterations)):
            by_survey: dict[int, list[float]] = {}
            for lid, tid, sv, tv, _cx, _cy in records:
                by_survey.setdefault(lid, []).append((tv + tie_shift[tid]) - sv)
            new_survey_shift = {lid: float(np.median(diffs)) for lid, diffs in by_survey.items()}

            by_tie: dict[int, list[float]] = {}
            for lid, tid, sv, tv, _cx, _cy in records:
                by_tie.setdefault(tid, []).append((sv + new_survey_shift[lid]) - tv)
            new_tie_shift = {tid: float(np.median(diffs)) for tid, diffs in by_tie.items()}

            mean_shift = float(np.mean(list(new_survey_shift.values()) + list(new_tie_shift.values())))
            new_survey_shift = {k: v - mean_shift for k, v in new_survey_shift.items()}
            new_tie_shift = {k: v - mean_shift for k, v in new_tie_shift.items()}

            max_delta = max(
                [abs(new_survey_shift[k] - survey_shift[k]) for k in new_survey_shift]
                + [abs(new_tie_shift[k] - tie_shift[k]) for k in new_tie_shift]
            )
            survey_shift, tie_shift = new_survey_shift, new_tie_shift
            if max_delta < tol_nt:
                break
    else:
        by_survey = {}
        for lid, _tid, sv, tv, _cx, _cy in records:
            by_survey.setdefault(lid, []).append(tv - sv)
        survey_shift = {lid: float(np.median(diffs)) for lid, diffs in by_survey.items()}
        tie_shift = {}

    before = np.array([tv - sv for _lid, _tid, sv, tv, _cx, _cy in records])
    after = np.array(
        [(tv + tie_shift.get(tid, 0.0)) - (sv + survey_shift.get(lid, 0.0)) for lid, tid, sv, tv, _cx, _cy in records]
    )
    rms_before = float(np.sqrt(np.mean(before**2)))
    rms_after = float(np.sqrt(np.mean(after**2)))

    return CrossoverLevelingResult(
        applied=True,
        reason=None,
        n_tie_lines=n_tie_lines,
        n_crossovers=len(records),
        n_survey_lines_corrected=len(survey_shift),
        rms_before_nt=rms_before,
        rms_after_nt=rms_after,
        line_shifts=survey_shift,
    )


def apply_crossover_leveling(df: pd.DataFrame, value_col: str, result: CrossoverLevelingResult) -> np.ndarray:
    values = df[value_col].to_numpy(copy=True)
    if not result.applied or not result.line_shifts:
        return values

    if result.line_shifts_linear:
        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        line_id = df["line_id"].to_numpy()
        shift = np.zeros(len(df), dtype=float)
        for lid, (intercept, slope, cx0, cy0, dx, dy) in result.line_shifts_linear.items():
            mask = line_id == lid
            if not mask.any():
                continue
            s = (x[mask] - cx0) * dx + (y[mask] - cy0) * dy
            shift[mask] = intercept + slope * s
        return values + shift

    shift = df["line_id"].map(result.line_shifts).fillna(0.0).to_numpy()
    return values + shift
