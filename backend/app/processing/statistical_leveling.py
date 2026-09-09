"""Statistical ("loose") leveling: per-line level corrections estimated
from the survey's own adjacent lines, with no tie lines and no
calibration flight required.

This fills the gap between the two per-line corrections already here:
processing/crossover_leveling.py needs tie lines to have been flown, and
processing/leveling.py estimates a single forward/reverse offset shared
by the whole survey. Neither can remove what actually produces most
visible striping - a level error that differs from line to line
(residual diurnal, sensor drift over a long flight, per-line altitude
differences, heading effect that varies with the drone's attitude rather
than just its direction). A single A/B offset cannot make one line high
and its neighbour low while a third sits in between; only a per-line
estimate can.

Method (the classic aeromagnetic "loose"/statistical leveling):

1. Order lines across-track and, for each adjacent pair, measure the
   robust difference between them over the length they share. That
   difference is (real geological change across the gap) + (difference
   in the two lines' level errors).
2. Chain those pairwise differences into an apparent level per line.
3. Split that per-line sequence into a smooth part and a
   line-to-line-jitter part by fitting a local trend across
   `trend_window_lines` neighbouring lines and subtracting it. This is
   the step that separates the two contributions above: real geology is
   spatially coherent, so it varies smoothly from one line to the next,
   whereas a leveling error is independent per line and lands almost
   entirely in the jitter.
4. Apply the jitter, negated and demeaned, as the correction.

The assumption in step 3 is the method's one real limitation, and it is
worth stating plainly: geology that is genuinely narrow *and* runs
parallel to the flight lines (a line-parallel dike a line or two wide)
looks exactly like a leveling error and will be partly removed along
with it.

trend_window_lines is the knob for that trade-off, and it works the way
a filter cutoff does, not the way "more smoothing is gentler" intuition
suggests: it sets how many lines a feature must span to count as
geology. A *narrow* window lets the trend follow fast cross-line
variation, so only the very fastest line-to-line jitter is left to
correct - conservative. A *wide* window holds the trend flat over more
lines, so more of what varies across them is called error and removed -
aggressive, and more likely to take line-parallel geology with it. This
is why the correction is off by default and reports what it changed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .lines import cross_track_coordinate

# Fewer lines than this and "smooth across lines vs. jitter across lines"
# is not a distinction the data can support - a local trend fitted to 3
# lines has nothing to average.
_MIN_LINES = 4
# An adjacent pair sharing less than this fraction of the shorter line's
# length is not compared: a handful of overlapping samples at one end
# gives a difference dominated by whatever geology sits there.
_MIN_OVERLAP_FRACTION = 0.25
# Along-line resampling step, as a fraction of the line spacing. Finer
# than this just re-samples the same information (points on one line are
# far denser than the line spacing) while making the interpolation
# noisier.
_RESAMPLE_STEP_FRACTION = 0.5
_MIN_SAMPLES_PER_PAIR = 8
# Narrower than this and the local quadratic in _running_trend has too
# few points to be stable: measured on synthetic surveys, a 5-line window
# leaves ~40% of the striping behind where a 9-line window leaves ~10%,
# because the quadratic starts following the very jitter it should be
# averaging over. 9 is the default for the same reason - it was the best
# measured balance of striping removed against clean-data bias
# (~10% residual striping, under 1 nT moved on data with no error in it).
_MIN_TREND_WINDOW = 7


@dataclass
class StatisticalLevelingResult:
    applied: bool
    reason: str | None
    n_lines: int = 0
    n_pairs: int = 0
    trend_window_lines: int = 0
    # Robustness diagnostics, in nT: how big the corrections were, and how
    # much line-to-line jitter they removed (see _cross_line_roughness).
    max_shift_nt: float | None = None
    rms_shift_nt: float | None = None
    roughness_before_nt: float | None = None
    roughness_after_nt: float | None = None
    line_shifts: dict = field(default_factory=dict)  # line_id -> constant shift (nT)
    # order=1 only, same shape as CrossoverLevelingResult.line_shifts_linear:
    # line_id -> (intercept_nt, slope_nt_per_m, x0, y0, dir_x, dir_y).
    line_shifts_linear: dict | None = None
    warnings: list[str] = field(default_factory=list)


def _along_track_unit(dominant_azimuth_deg: float) -> np.ndarray:
    """Unit vector along the line azimuth, in the atan2(dy, dx) convention
    lines.py uses (0 = along +x/easting, 90 = along +y/northing) - the
    companion of that module's _perp_vector."""
    az = np.radians(dominant_azimuth_deg)
    return np.array([np.cos(az), np.sin(az)])


def _profile(s: np.ndarray, v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Median value of `v` in each along-track bin defined by `edges`, NaN
    for empty bins. A median rather than an interpolation so a single
    spike or a stray off-line sample can't move the bin, and so the two
    lines of a pair are reduced the same way regardless of how differently
    they happen to be sampled."""
    idx = np.digitize(s, edges) - 1
    n_bins = len(edges) - 1
    out = np.full(n_bins, np.nan)
    order = np.argsort(idx, kind="stable")
    idx_sorted, v_sorted = idx[order], v[order]
    # One pass over the contiguous runs of equal bin index.
    starts = np.flatnonzero(np.r_[True, idx_sorted[1:] != idx_sorted[:-1]])
    for start, stop in zip(starts, np.r_[starts[1:], len(idx_sorted)]):
        b = idx_sorted[start]
        if 0 <= b < n_bins:
            out[b] = np.median(v_sorted[start:stop])
    return out


def _running_trend(values: np.ndarray, window: int, poly_order: int = 2, n_iter: int = 3) -> np.ndarray:
    """Robust local polynomial trend across the line sequence: for each
    line, fit a low-order polynomial to the levels of the lines around it
    and take its value there. Windows shrink at the ends, so every line
    gets a trend estimate (an edge line has neighbours on one side only,
    but it still has neighbours).

    A local *fit* rather than the more obvious running median, for two
    reasons that both showed up as visibly wrong corrections:

    - A median tracks curvature badly. Real geology curving across the
      survey then lands in the residual and gets "corrected" away, moving
      lines by several nT on data with no leveling error in it at all.
    - On an alternating high/low/high sequence - which is exactly what a
      direction-dependent offset looks like across lines - a median
      returns the majority value, so it follows the very pattern it is
      supposed to be separating out, and the alternation survives the
      correction. A least-squares fit averages the two instead.

    Robustness against a single line sitting on a real anomaly (which a
    median would have given for free) is recovered with a few
    reweighting passes: residuals far outside the local scatter lose
    weight, so one wild line bends the trend much less than it would in a
    plain fit.
    """
    n = len(values)
    half = max(1, window // 2)
    positions = np.arange(n, dtype=float)
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        xs = positions[lo:hi] - i
        ys = values[lo:hi]
        # Leave at least 2 degrees of freedom: a polynomial with as many
        # coefficients as points interpolates them exactly, which would
        # make the residual - and so the whole correction - identically
        # zero.
        order = min(poly_order, len(xs) - 2)
        if order < 1:
            out[i] = float(np.median(ys))
            continue
        weights = np.ones(len(xs))
        for _ in range(n_iter):
            coeff = np.polyfit(xs, ys, order, w=np.sqrt(weights))
            residual = ys - np.polyval(coeff, xs)
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            if scale <= 1e-12:
                break
            # Cauchy weights: smooth, and never fully discards a point.
            weights = 1.0 / (1.0 + (residual / (2.0 * scale)) ** 2)
        out[i] = float(np.polyval(np.polyfit(xs, ys, order, w=np.sqrt(weights)), 0.0))
    return out


def _cross_line_roughness(levels: np.ndarray) -> float:
    """Mean |second difference| of the per-line levels across the survey -
    a scalar for "how much do these lines disagree with their neighbours
    beyond a smooth trend", which is what striping looks like. Second
    (not first) difference so a genuine steady geological gradient across
    the survey doesn't register as roughness."""
    if len(levels) < 3:
        return 0.0
    return float(np.mean(np.abs(levels[2:] - 2.0 * levels[1:-1] + levels[:-2])))


def compute_statistical_leveling(
    df: pd.DataFrame,
    value_col: str,
    dominant_azimuth_deg: float,
    line_spacing_m: float | None,
    trend_window_lines: int = 9,
    order: int = 0,
    n_segments: int = 4,
    max_shift_nt: float | None = None,
) -> StatisticalLevelingResult:
    """trend_window_lines: how many lines a feature must span to be
    treated as geology rather than leveling error - the cross-line cutoff
    described in the module docstring. Smaller = more conservative
    (corrects only the fastest line-to-line jitter); larger = removes more
    error and more line-parallel signal with it.

    order: 0 (default) = one constant shift per line. 1 = the shift is
    allowed to vary linearly along each line, which catches drift within a
    single long line rather than only a whole-line DC jump; estimated by
    running the same chain-and-detrend over `n_segments` along-line
    segments and fitting a straight line through the resulting per-segment
    corrections.

    max_shift_nt: optional clamp. A per-line correction larger than this
    is more likely a real anomaly the line happened to sit on than a
    leveling error, so it is limited and reported in warnings rather than
    applied in full.
    """
    if trend_window_lines < _MIN_TREND_WINDOW:
        raise ValueError(
            f"trend_window_lines는 {_MIN_TREND_WINDOW} 이상이어야 합니다 "
            "(그보다 좁으면 국소 2차 추세를 안정적으로 맞출 수 없습니다)."
        )
    if order not in (0, 1):
        raise ValueError("order는 0(측선당 상수) 또는 1(측선 방향 1차)이어야 합니다.")

    kept = df[df["line_id"] >= 0]
    line_ids = sorted(int(v) for v in kept["line_id"].unique())
    if len(line_ids) < _MIN_LINES:
        return StatisticalLevelingResult(
            False,
            f"통계적 레벨링에는 측선이 최소 {_MIN_LINES}개 필요합니다 (현재 {len(line_ids)}개) - "
            "인접 측선의 추세와 측선별 오차를 구분할 수 없습니다.",
            n_lines=len(line_ids),
        )

    along = _along_track_unit(dominant_azimuth_deg)
    x = kept["x"].to_numpy(dtype=float)
    y = kept["y"].to_numpy(dtype=float)
    s_all = x * along[0] + y * along[1]
    cross_all = cross_track_coordinate(x, y, dominant_azimuth_deg)
    v_all = kept[value_col].to_numpy(dtype=float)
    lid_all = kept["line_id"].to_numpy()

    finite = np.isfinite(s_all) & np.isfinite(v_all)
    per_line: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    line_cross: dict[int, float] = {}
    for lid in line_ids:
        m = (lid_all == lid) & finite
        if m.sum() < _MIN_SAMPLES_PER_PAIR:
            continue
        per_line[lid] = (s_all[m], v_all[m])
        line_cross[lid] = float(np.median(cross_all[m]))
    if len(per_line) < _MIN_LINES:
        return StatisticalLevelingResult(
            False, "통계적 레벨링에 쓸 수 있는 유효 측선이 부족합니다.", n_lines=len(per_line)
        )

    ordered = sorted(per_line, key=lambda lid: line_cross[lid])
    step = max(1.0, (line_spacing_m or 50.0) * _RESAMPLE_STEP_FRACTION)
    n_seg = max(1, n_segments) if order == 1 else 1

    # deltas[k][i] = (line i's level - line i+1's level) in segment k, NaN
    # where the pair doesn't share enough of that segment to compare.
    deltas = np.full((n_seg, len(ordered) - 1), np.nan)
    n_pairs = 0
    for i in range(len(ordered) - 1):
        s_a, v_a = per_line[ordered[i]]
        s_b, v_b = per_line[ordered[i + 1]]
        lo, hi = max(s_a.min(), s_b.min()), min(s_a.max(), s_b.max())
        shared = hi - lo
        shorter = min(np.ptp(s_a), np.ptp(s_b))
        if shared <= 0 or shorter <= 0 or shared < _MIN_OVERLAP_FRACTION * shorter:
            continue

        edges = np.arange(lo, hi + step, step)
        if len(edges) < _MIN_SAMPLES_PER_PAIR + 1:
            edges = np.linspace(lo, hi, _MIN_SAMPLES_PER_PAIR + 1)
        prof_a = _profile(s_a, v_a, edges)
        prof_b = _profile(s_b, v_b, edges)
        diff = prof_a - prof_b
        if np.isfinite(diff).sum() < _MIN_SAMPLES_PER_PAIR:
            continue
        n_pairs += 1

        centres = 0.5 * (edges[:-1] + edges[1:])
        # Segment boundaries are taken over the pair's own shared span, so
        # segment k means "the k-th fraction of what these two lines have
        # in common" for every pair alike.
        seg_idx = np.clip(((centres - lo) / max(shared, 1e-9) * n_seg).astype(int), 0, n_seg - 1)
        for k in range(n_seg):
            d = diff[(seg_idx == k) & np.isfinite(diff)]
            if d.size >= max(3, _MIN_SAMPLES_PER_PAIR // n_seg):
                deltas[k, i] = float(np.median(d))

    if n_pairs == 0:
        return StatisticalLevelingResult(
            False,
            "인접 측선끼리 겹치는 구간이 부족해 통계적 레벨링을 계산할 수 없습니다 "
            "(측선이 서로 다른 구역을 날았는지 확인하세요).",
            n_lines=len(ordered),
        )

    warnings: list[str] = []
    # Corrections per segment, then reduced to a constant (order=0) or a
    # straight line in s (order=1) per survey line.
    corrections = np.zeros((n_seg, len(ordered)))
    for k in range(n_seg):
        d = deltas[k]
        # A pair we couldn't measure contributes no step, which chains the
        # two sides together at whatever level they already had rather
        # than breaking the sequence in two.
        levels = -np.concatenate([[0.0], np.nancumsum(np.where(np.isfinite(d), d, 0.0))])
        jitter = levels - _running_trend(levels, trend_window_lines)
        corrections[k] = -(jitter - np.mean(jitter))

    if max_shift_nt is not None and max_shift_nt > 0:
        clamped = int(np.sum(np.abs(corrections) > max_shift_nt))
        if clamped:
            warnings.append(
                f"측선 보정량 {clamped}개가 상한({max_shift_nt:.1f}nT)을 넘어 잘렸습니다 - "
                "실제 이상대를 레벨 오차로 오인했을 수 있으니 추세 창(측선 수)을 넓혀보세요."
            )
        corrections = np.clip(corrections, -max_shift_nt, max_shift_nt)

    line_shifts = {lid: float(corrections[:, i].mean()) for i, lid in enumerate(ordered)}
    line_shifts_linear: dict | None = None
    if order == 1 and n_seg > 1:
        line_shifts_linear = {}
        for i, lid in enumerate(ordered):
            s_line, _v = per_line[lid]
            lo, hi = float(s_line.min()), float(s_line.max())
            centre = 0.5 * (lo + hi)
            # Segment k of this line spans the k-th fraction of its own
            # length; place each correction at that segment's midpoint,
            # measured from the line centre so `intercept` is the shift at
            # the middle of the line and stays comparable to order=0.
            seg_s = lo + (np.arange(n_seg) + 0.5) * (hi - lo) / n_seg - centre
            if hi - lo > 1e-6:
                slope, intercept = np.polyfit(seg_s, corrections[:, i], 1)
            else:
                slope, intercept = 0.0, float(np.mean(corrections[:, i]))
            line_shifts[lid] = float(intercept)
            line_shifts_linear[lid] = (
                float(intercept), float(slope),
                float(centre * along[0]), float(centre * along[1]),
                float(along[0]), float(along[1]),
            )

    levels_before = -np.concatenate([[0.0], np.nancumsum(np.where(np.isfinite(deltas[0]), deltas[0], 0.0))])
    shifts_ordered = np.array([line_shifts[lid] for lid in ordered])
    result = StatisticalLevelingResult(
        applied=True,
        reason=None,
        n_lines=len(ordered),
        n_pairs=n_pairs,
        trend_window_lines=trend_window_lines,
        max_shift_nt=float(np.max(np.abs(shifts_ordered))),
        rms_shift_nt=float(np.sqrt(np.mean(shifts_ordered**2))),
        roughness_before_nt=_cross_line_roughness(levels_before),
        roughness_after_nt=_cross_line_roughness(levels_before + shifts_ordered),
        line_shifts=line_shifts,
        line_shifts_linear=line_shifts_linear,
        warnings=warnings,
    )
    if len(ordered) < trend_window_lines + 2:
        # The running window then spans the whole survey, so there is no
        # "smooth across the survey vs. jitter between neighbours" split
        # left to make - every line is inside every other line's trend
        # window, and what comes out is whatever the survey-wide fit
        # happens not to explain. Still applied (it is small and the user
        # asked for it), but it should not be read as a leveling estimate.
        result.warnings.append(
            f"측선이 {len(ordered)}개뿐이라 추세 창({trend_window_lines}개)이 탐사 전체를 덮습니다 - "
            "측선별 오차와 지질 추세를 구분할 수 없으니 보정량을 신뢰하지 마세요 "
            f"(추세 창을 줄이거나, 측선이 {trend_window_lines + 2}개 이상인 자료에 사용하세요)."
        )
    if n_pairs < len(ordered) - 1:
        result.warnings.append(
            f"인접 측선 {len(ordered) - 1}쌍 중 {n_pairs}쌍만 비교할 수 있었습니다 - "
            "겹치지 않는 측선 구간이 있어 일부 측선은 이웃을 통해 간접적으로만 보정됩니다."
        )
    return result


def apply_statistical_leveling(df: pd.DataFrame, value_col: str, result: StatisticalLevelingResult) -> np.ndarray:
    values = df[value_col].to_numpy(copy=True)
    if not result.applied or not result.line_shifts:
        return values

    if result.line_shifts_linear:
        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        line_id = df["line_id"].to_numpy()
        shift = np.zeros(len(df), dtype=float)
        for lid, (intercept, slope, x0, y0, dx, dy) in result.line_shifts_linear.items():
            mask = line_id == lid
            if not mask.any():
                continue
            s = (x[mask] - x0) * dx + (y[mask] - y0) * dy
            shift[mask] = intercept + slope * s
        return values + shift

    shift = df["line_id"].map(result.line_shifts).fillna(0.0).to_numpy()
    return values + shift
