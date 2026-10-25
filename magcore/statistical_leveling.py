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

trend_window_lines is the knob for that trade-off, and it does not behave
the way either intuition about it suggests. It is not "more smoothing is
gentler", and it is not a clean filter cutoff either. Measured on
synthetic surveys (30 lines, geology curving over ~18 of them):

  window   per-line error left   invented on clean data   line-parallel
                                                          anomaly kept
       7                   20%                  0.5 nT              6%
       9                   16%                  1.1 nT              7%
      15                   10%                  3.3 nT             20%
      21                   13%                 11.9 nT             52%
      27                   18%                 25.0 nT             96%

Too narrow and the local trend follows the per-line error itself, so
little is corrected. Too wide and the local quadratic can no longer
follow the geology across the window's span; its own misfit is then what
gets subtracted, and the correction becomes invented structure - 25 nT of
it on data with no leveling error in it at all. Note the last column
rising with it: a wide window does preserve a line-parallel anomaly, but
only because the trend has stopped tracking anything, which is not a
safety margin worth having.

So there is a sweet spot rather than a direction, and it sits a little
below the number of lines the geology takes to curve. The default of 9
is near it and errs narrow. Because a too-wide window is actively
harmful rather than merely aggressive, the correction also checks itself:
it recomputes at the narrowest supported window and warns when widening
has inflated the correction far more than finding real error ever does
(see _WINDOW_STABILITY_RATIO).

This correction runs by default. It declines outright, changing nothing,
on any survey with fewer lines than the window can be run over.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .lines import cross_track_coordinate

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
# The trend window has to be narrower than the survey, or every line sits
# inside every other line's window and there is no "smooth across the
# survey vs. jitter between neighbours" split left to make. Two spare
# lines is the minimum that leaves the running window something to run
# over; below it the correction is declined rather than applied - it is on
# by default now, and a correction nobody can check is worse than none.
_TREND_WINDOW_HEADROOM_LINES = 2


# How much bigger the correction may get, relative to the one the
# narrowest supported window produces, before the extra width is treated
# as misfit rather than as more leveling error found. Measured on
# synthetic surveys: with real per-line error present, widening the window
# from 7 to 27 lines changes the correction by at most ~2.5x, while on a
# survey with no error at all the same widening inflates it 66x.
_WINDOW_STABILITY_RATIO = 3.0


def required_lines(trend_window_lines: int) -> int:
    """How many survey lines this correction needs at a given window."""
    return trend_window_lines + _TREND_WINDOW_HEADROOM_LINES


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


def _line_profiles(
    kept: pd.DataFrame, value_col: str, dominant_azimuth_deg: float
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, float]]:
    """Each line reduced to (along-track coordinate, value) arrays, plus
    where it sits on the cross-line axis so lines can be put in
    across-survey order. Lines too short to compare are dropped."""
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
    for lid in sorted(int(v) for v in kept["line_id"].unique()):
        m = (lid_all == lid) & finite
        if m.sum() < _MIN_SAMPLES_PER_PAIR:
            continue
        per_line[lid] = (s_all[m], v_all[m])
        line_cross[lid] = float(np.median(cross_all[m]))
    return per_line, line_cross


def _pairwise_deltas(
    per_line: dict[int, tuple[np.ndarray, np.ndarray]],
    ordered: list[int],
    line_spacing_m: float | None,
    n_seg: int,
) -> tuple[np.ndarray, int]:
    """deltas[k][i] = (line i's level - line i+1's level) over segment k of
    what the two lines share, NaN where they don't share enough of it to
    compare. Also returns how many adjacent pairs could be compared at
    all. Both lines are reduced the same way (median per along-track bin)
    so a difference reflects their levels, not their sampling."""
    step = max(1.0, (line_spacing_m or 50.0) * _RESAMPLE_STEP_FRACTION)
    deltas = np.full((n_seg, max(len(ordered) - 1, 0)), np.nan)
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
        diff = _profile(s_a, v_a, edges) - _profile(s_b, v_b, edges)
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
    return deltas, n_pairs


def _chain(deltas_row: np.ndarray) -> np.ndarray:
    """Turn the pairwise differences into an apparent level per line. A
    pair that couldn't be measured contributes no step, which chains the
    two sides together at whatever level they already had rather than
    breaking the sequence in two."""
    return -np.concatenate([[0.0], np.nancumsum(np.where(np.isfinite(deltas_row), deltas_row, 0.0))])


def _corrections_for_window(deltas: np.ndarray, trend_window_lines: int) -> np.ndarray:
    """Per-segment, per-line corrections at one trend window: chain the
    pairwise differences into levels, subtract the local trend, and negate
    what's left. Demeaned so the survey's overall level is unchanged."""
    corrections = np.zeros_like(deltas, shape=(deltas.shape[0], deltas.shape[1] + 1))
    for k in range(deltas.shape[0]):
        levels = _chain(deltas[k])
        jitter = levels - _running_trend(levels, trend_window_lines)
        corrections[k] = -(jitter - np.mean(jitter))
    return corrections


def _chained_line_levels(
    kept: pd.DataFrame, value_col: str, dominant_azimuth_deg: float, line_spacing_m: float | None
) -> np.ndarray | None:
    """The per-line level sequence on its own, for measurement rather than
    correction. None when too few lines could be compared."""
    per_line, line_cross = _line_profiles(kept, value_col, dominant_azimuth_deg)
    if len(per_line) < 3:
        return None
    ordered = sorted(per_line, key=lambda lid: line_cross[lid])
    deltas, n_pairs = _pairwise_deltas(per_line, ordered, line_spacing_m, 1)
    if n_pairs == 0:
        return None
    return _chain(deltas[0])


def measure_striping(
    df: pd.DataFrame,
    value_col: str,
    dominant_azimuth_deg: float,
    line_spacing_m: float | None,
) -> dict:
    """How badly this survey stripes, as a number, independent of whether
    any correction is applied.

    Reported for every survey so the effect can be compared rather than
    eyeballed - between two flights, between processing settings, or
    between two aircraft, since striping severity is a property of the
    platform (vibration, current draw, how far the sensor hangs below the
    airframe) as much as of the magnetometer.

    Two figures come back. `line_level_jitter_nt` is the mean disagreement
    between a line and its immediate neighbours beyond a smooth trend, in
    nT - the absolute size of the problem. `stripe_ratio_pct` is that as a
    percentage of the survey's own anomaly range, which is the one to
    compare across surveys: the same 2 nT of jitter is invisible over a
    strongly magnetic area and dominates a quiet one.

    Needs only three lines (it is a second difference), so it is available
    on surveys far too small for the correction itself.
    """
    kept = df[df["line_id"] >= 0]
    values = kept[value_col].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    line_ids = sorted(int(v) for v in kept["line_id"].unique())
    if len(line_ids) < 3 or values.size == 0:
        return {
            "available": False,
            "reason": "줄무늬 세기를 재려면 측선이 3개 이상 필요합니다.",
            "n_lines": len(line_ids),
        }

    levels = _chained_line_levels(kept, value_col, dominant_azimuth_deg, line_spacing_m)
    if levels is None or len(levels) < 3:
        return {
            "available": False,
            "reason": "인접 측선끼리 겹치는 구간이 부족해 줄무늬 세기를 잴 수 없습니다.",
            "n_lines": len(line_ids),
        }

    jitter = _cross_line_roughness(levels)
    # Robust spread of the field itself, so one spike can't make a badly
    # striped survey look clean by inflating the denominator.
    signal = float(np.percentile(values, 95) - np.percentile(values, 5))
    return {
        "available": True,
        "n_lines": len(line_ids),
        "line_level_jitter_nt": round(jitter, 3),
        "signal_range_nt": round(signal, 3),
        "stripe_ratio_pct": round(100.0 * jitter / signal, 2) if signal > 1e-9 else None,
    }


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
    """trend_window_lines: how many lines the local trend is fitted over -
    see the module docstring for the measured trade-off and why there is a
    sweet spot rather than a "safe direction". Needs the survey to have
    two more lines than this, or the correction is declined.

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

    required = required_lines(trend_window_lines)
    kept = df[df["line_id"] >= 0]
    line_ids = sorted(int(v) for v in kept["line_id"].unique())
    if len(line_ids) < required:
        return StatisticalLevelingResult(
            False,
            f"측선이 {len(line_ids)}개뿐이라 통계적 레벨링을 건너뛰었습니다 "
            f"(추세 창 {trend_window_lines}개에는 측선 {required}개 이상 필요) - "
            "창이 탐사 전체를 덮으면 측선별 오차와 지질 추세를 구분할 수 없어, "
            "믿을 수 없는 보정을 적용하는 대신 자료를 그대로 둡니다. "
            f"추세 창을 최소값({_MIN_TREND_WINDOW})까지 줄이면 측선 "
            f"{required_lines(_MIN_TREND_WINDOW)}개부터 사용할 수 있습니다.",
            n_lines=len(line_ids),
            trend_window_lines=trend_window_lines,
        )

    per_line, line_cross = _line_profiles(kept, value_col, dominant_azimuth_deg)
    if len(per_line) < required:
        return StatisticalLevelingResult(
            False,
            f"통계적 레벨링에 쓸 수 있는 유효 측선이 부족합니다 "
            f"({len(per_line)}개, {required}개 이상 필요).",
            n_lines=len(per_line)
        )

    ordered = sorted(per_line, key=lambda lid: line_cross[lid])
    n_seg = max(1, n_segments) if order == 1 else 1
    deltas, n_pairs = _pairwise_deltas(per_line, ordered, line_spacing_m, n_seg)

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
    corrections = _corrections_for_window(deltas, trend_window_lines)

    # Widening the trend window past what the geology supports is the one
    # way this correction can do real damage: once a local quadratic can no
    # longer follow the field over the window's span, its own misfit is
    # what gets removed, and the result is invented structure rather than
    # leveling. The symptom is specific enough to test for - a survey with
    # genuine per-line error gives nearly the same correction at any
    # window (the error is there to be found either way), while on a
    # survey without it the correction balloons with the window. So
    # compare against the narrowest supported window and say so.
    if trend_window_lines > _MIN_TREND_WINDOW:
        reference = _corrections_for_window(deltas, _MIN_TREND_WINDOW)
        ref_rms = float(np.sqrt(np.mean(reference**2)))
        this_rms = float(np.sqrt(np.mean(corrections**2)))
        if ref_rms > 1e-9 and this_rms / ref_rms > _WINDOW_STABILITY_RATIO:
            warnings.append(
                f"추세 창을 {trend_window_lines}개로 넓히자 보정량이 최소 창({_MIN_TREND_WINDOW}개) 대비 "
                f"{this_rms / ref_rms:.1f}배로 커졌습니다 - 창이 지질 변화를 따라가지 못해 "
                "레벨 오차가 아니라 추세 맞춤 오차를 지우고 있을 가능성이 높습니다. "
                f"추세 창을 {_MIN_TREND_WINDOW}~11개로 줄이세요."
            )

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
        along = _along_track_unit(dominant_azimuth_deg)
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

    levels_before = _chain(deltas[0])
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
