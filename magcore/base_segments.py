"""Telling a moved base station apart from a real change in the field.

The diurnal correction subtracts (base reading - one reference level taken
over the whole survey) from every drone sample. Two different things can
make one day's base readings sit at a different level from another's, and
they need opposite treatment:

* **The field really was different that day.** Day-to-day Sq and storm
  activity move the whole region, base and survey area alike, so the base
  captures it and subtracting it is exactly what the correction is for.
  This must be carried through to the drone data, not removed. Measured on
  ten days of Cheongyang observatory 1 s data - a station that never moves
  - the 09-11 h level differed between consecutive days by a median of
  5.1 nT, at most 22.0 nT, with a total spread of 22.9 nT across a month.

* **The base was moved.** A different spot reads a different absolute
  field, and that step is an artifact of where the instrument sat. It does
  not cancel against one survey-wide reference, so it lands in the anomaly
  as a constant per-flight offset, ahead of every levelling step and
  indistinguishable from real field there. On a two-day synthetic with no
  anomaly in it and the base moved 200 nT between days, the corrected data
  came out with a -200.00 nT step between the days.

Nothing in the base log alone settles which of the two happened, so this
module does not pretend otherwise. What it does is separate the evidence:

* A **step inside continuous logging** cannot be the field. Diurnal
  variation runs at roughly 0.1-2 nT per minute; a jump of tens of nT
  between consecutive samples means the instrument's surroundings changed
  - it was carried somewhere else, or something ferrous was parked beside
  it. Levelling those is safe and is the default.

* A **level difference across a gap in logging** is ambiguous. The unit
  being off overnight is the normal way a base station is run, and the
  level difference across that gap is usually the real day-to-day field
  change above. So gaps are measured and reported but *not* levelled
  unless the operator says the base moved (`mode="all"`), because
  levelling them away would throw out the very signal the base was
  deployed to capture. When such a difference is far outside what a fixed
  station does - the 22.9 nT above sets that scale - it is flagged, with
  the question put to the operator rather than answered here.

An observatory feed (processing/intermagnet.py) never moves, so it should
be left on the default rather than "all".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A base log is normally continuous; this much silence means the unit was
# off - which happens both overnight and while being carried.
DEFAULT_GAP_MINUTES = 10.0
# Minimum instantaneous level change to call a discontinuity. Well above
# any between-sample diurnal change, well below the tens to hundreds of nT
# a different location gives.
DEFAULT_STEP_THRESHOLD_NT = 20.0
# ...and it must also stand out against how much this particular log moves
# between samples, so a noisy logger does not fragment.
_STEP_ROBUST_K = 8.0

# Floor for a jump landing exactly at 00:00 UT - see find_base_segments
# for why a seam there is held to a different standard than one in the
# middle of a day.
DEFAULT_DAY_BOUNDARY_STEP_NT = 3.0

# Samples either side of a break used to measure how far the record jumps
# there. Enough to average out ordinary sample-to-sample noise, short
# enough that the diurnal curve barely bends across it.
_STEP_WINDOW_SAMPLES = 30
_MIN_STEP_SAMPLES = 10
# Above this, a level difference across a logging gap is larger than a
# fixed station's own day-to-day range (22.9 nT measured over ten days at
# Cheongyang) and is worth asking the operator about.
DEFAULT_GAP_SUSPICIOUS_NT = 30.0

SegmentMode = str  # "off" | "steps" | "all"


@dataclass
class BaseSegment:
    index: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_samples: int
    level_nt: float
    offset_applied_nt: float
    split_reason: str | None   # "gap" | "step" | None for the first segment
    levelled: bool


@dataclass
class BaseLevelingResult:
    base: pd.DataFrame
    mode: SegmentMode = "steps"
    segments: list[BaseSegment] = field(default_factory=list)
    common_level_nt: float | None = None
    max_offset_nt: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "n_segments": self.n_segments,
            "n_levelled": sum(1 for s in self.segments if s.levelled),
            "common_level_nt": self.common_level_nt,
            "max_offset_nt": self.max_offset_nt,
            "segments": [
                {
                    "index": s.index,
                    "start": str(s.start),
                    "end": str(s.end),
                    "n_samples": s.n_samples,
                    "level_nt": s.level_nt,
                    "offset_applied_nt": s.offset_applied_nt,
                    "split_reason": s.split_reason,
                    "levelled": s.levelled,
                }
                for s in self.segments
            ],
            "warnings": self.warnings,
        }


def step_across_break(
    values: np.ndarray, k: int, window: int = _STEP_WINDOW_SAMPLES
) -> float | None:
    """How far the record jumps at index `k`, measured locally.

    Fits a straight line to the `window` samples on each side of the break
    and reads both at the break. The difference of two medians would
    measure the step plus half a window of slope mismatch, and a difference
    of whole-segment medians - which is what this module used to apply -
    measures something else entirely: two segments covering different
    numbers of days, or different parts of the diurnal curve, have
    different medians for reasons that have nothing to do with the step.

    That mattered. On the 2026-09 HaeNam base the discontinuity at
    2026-09-09 00:00 UT measures -19.1 nT locally; the segment-median
    difference was -5.2 nT, so only a fifth of it was taken out, and what
    the correction applied (+3.96 nT) bore no relation to what it had
    detected.

    None when either side is too short to fit.
    """
    before = values[max(0, k - window):k]
    after = values[k:k + window]
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]
    if len(before) < _MIN_STEP_SAMPLES or len(after) < _MIN_STEP_SAMPLES:
        return None

    def edge(v: np.ndarray, at_start: bool) -> float:
        x = np.arange(len(v), dtype=float)
        slope, intercept = np.polyfit(x, v, 1)
        return float(intercept + slope * (0.0 if at_start else len(v)))

    return edge(after, True) - edge(before, False)


def find_base_segments(
    base_df: pd.DataFrame,
    gap_minutes: float = DEFAULT_GAP_MINUTES,
    step_threshold_nt: float = DEFAULT_STEP_THRESHOLD_NT,
    day_boundary_step_threshold_nt: float = DEFAULT_DAY_BOUNDARY_STEP_NT,
) -> tuple[np.ndarray, list[str]]:
    """Segment id per row, plus why each new segment started."""
    base = base_df.sort_values("timestamp")
    t = base["timestamp"].astype("int64").to_numpy()
    mag = base["mag"].to_numpy(dtype=float)
    n = len(mag)
    if n == 0:
        return np.zeros(0, dtype=int), []

    dt_s = np.diff(t) / 1e9
    dmag = np.diff(mag)
    mad = float(np.median(np.abs(dmag - np.median(dmag)))) if len(dmag) else 0.0
    robust_step = max(step_threshold_nt, _STEP_ROBUST_K * 1.4826 * mad)

    is_gap = dt_s > gap_minutes * 60.0
    is_step = np.abs(dmag) > robust_step

    # A discontinuity exactly at UT midnight gets a much lower threshold.
    # Observatory data is published one UT day at a time, so a record
    # assembled from it carries its seams at 00:00 UT - and the field does
    # not know about UT midnight, so a jump landing precisely there, many
    # times the record's own sample-to-sample variability, is the join and
    # not the field. Away from that boundary the conservative threshold
    # stands, because a severe storm really can move the field several nT
    # in a minute and must not be levelled away.
    #
    # This is not a corner case at this longitude: 00:00 UT is 09:00 in
    # Korea, so every Korean morning flight spans it. On the 2026-09
    # HaeNam survey four of the nine flight sessions crossed UT midnight,
    # and the base carried a 20.9 nT seam there.
    ts = base["timestamp"]
    at_day_boundary = (
        (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)
    ).to_numpy()[1:]
    boundary_step = max(day_boundary_step_threshold_nt, _STEP_ROBUST_K * 1.4826 * mad)
    is_step |= at_day_boundary & (np.abs(dmag) > boundary_step)

    seg = np.zeros(n, dtype=int)
    reasons: list[str] = []
    current = 0
    for i in range(1, n):
        if is_gap[i - 1] or is_step[i - 1]:
            current += 1
            reasons.append("gap" if is_gap[i - 1] else "step")
        seg[i] = current
    return seg, reasons


def level_base_segments(
    base_df: pd.DataFrame,
    mode: SegmentMode = "steps",
    gap_minutes: float = DEFAULT_GAP_MINUTES,
    step_threshold_nt: float = DEFAULT_STEP_THRESHOLD_NT,
    gap_suspicious_nt: float = DEFAULT_GAP_SUSPICIOUS_NT,
    min_segment_samples: int = 30,
    day_boundary_step_threshold_nt: float = DEFAULT_DAY_BOUNDARY_STEP_NT,
) -> BaseLevelingResult:
    """Report the base log's deployments, and level the ones `mode` allows.

    mode="off"    measure and report only.
    mode="steps"  (default) level discontinuities inside continuous
                  logging, which cannot be the field.
    mode="all"    also level across logging gaps - for a base the operator
                  knows was moved. This necessarily removes real
                  day-to-day field change along with the relocation, so it
                  is not the default.
    """
    base = base_df.sort_values("timestamp").reset_index(drop=True).copy()
    if base.empty:
        return BaseLevelingResult(base=base, mode=mode)

    seg_id, reasons = find_base_segments(
        base, gap_minutes, step_threshold_nt, day_boundary_step_threshold_nt)
    base["_segment"] = seg_id

    # Fold a too-short segment into its predecessor: a handful of samples
    # cannot establish a level.
    counts = base["_segment"].value_counts().sort_index()
    keep = [s for s, c in counts.items() if c >= min_segment_samples] or [int(counts.idxmax())]
    remap, last_kept = {}, keep[0]
    for s in counts.index:
        if s in keep:
            last_kept = s
        remap[s] = last_kept
    base["_segment"] = base["_segment"].map(remap)

    # Reasons survive only for boundaries that still separate two segments.
    kept_ids = list(dict.fromkeys(base["_segment"].tolist()))
    reason_by_id: dict[int, str | None] = {kept_ids[0]: None}
    for i, sid in enumerate(kept_ids[1:], start=1):
        reason_by_id[sid] = reasons[sid - 1] if sid - 1 < len(reasons) else "gap"

    levels = base.groupby("_segment")["mag"].median()
    sizes = base.groupby("_segment")["mag"].size()

    def should_level(sid: int) -> bool:
        if mode == "off":
            return False
        if mode == "all":
            return True
        return reason_by_id.get(sid) == "step"

    # Walk the record forwards, and at each boundary that is to be
    # levelled, shift everything after it by the step measured at that
    # boundary (step_across_break). The shift accumulates, so the record
    # comes out continuous and everything before the first levelled
    # boundary keeps the level it was recorded at.
    #
    # The offset used to be (weighted common level - this segment's own
    # median), which is a different quantity from the step it was detected
    # by: segments covering different numbers of days, or different parts
    # of the diurnal curve, have different medians for legitimate reasons.
    # See step_across_break for what that cost on real data.
    values = base["mag"].to_numpy(dtype=float)
    seg_start = {sid: int(np.flatnonzero((base["_segment"] == sid).to_numpy())[0])
                 for sid in kept_ids}
    cumulative = 0.0
    offset_by_sid: dict[int, float] = {}
    for sid in kept_ids:
        if should_level(sid):
            measured = step_across_break(values, seg_start[sid])
            if measured is not None:
                cumulative -= measured
        offset_by_sid[sid] = cumulative

    segments: list[BaseSegment] = []
    for i, sid in enumerate(kept_ids):
        rows = base["_segment"] == sid
        level = float(levels[sid])
        offset = offset_by_sid[sid]
        if offset:
            base.loc[rows, "mag"] = base.loc[rows, "mag"] + offset
        sub = base.loc[rows, "timestamp"]
        segments.append(BaseSegment(
            index=i, start=sub.min(), end=sub.max(), n_samples=int(rows.sum()),
            level_nt=level, offset_applied_nt=offset,
            split_reason=reason_by_id.get(sid), levelled=bool(offset),
        ))

    warnings: list[str] = []
    n_steps = sum(1 for s in segments if s.split_reason == "step")
    if n_steps and mode != "off":
        warnings.append(
            f"기준국 기록 중간에 불연속 단차가 {n_steps}곳 있어 해당 구간을 공통 레벨로 "
            f"맞췄습니다 - 일변화는 분당 0.1~2nT로 변하므로 샘플 사이 수십 nT 도약은 실제 "
            f"자기장이 아니라 기준국 주변 환경이 바뀐 것(이동, 차량 접근 등)입니다."
        )
    elif n_steps:
        warnings.append(f"기준국 기록 중간에 불연속 단차가 {n_steps}곳 있으나 보정하지 않았습니다(mode=off).")

    gap_jumps = [
        (segments[i - 1], segments[i], abs(segments[i].level_nt - segments[i - 1].level_nt))
        for i in range(1, len(segments)) if segments[i].split_reason == "gap"
    ]
    big = [g for g in gap_jumps if g[2] > gap_suspicious_nt]
    if big and mode != "all":
        worst = max(big, key=lambda g: g[2])
        warnings.append(
            f"기록이 끊긴 뒤 기준국 레벨이 {worst[2]:.1f}nT 달라졌습니다 "
            f"({worst[0].end} → {worst[1].start}). 위치가 고정된 관측소의 날짜 간 변화는 "
            f"보통 20nT 이내이므로, 이 정도 차이는 기준국을 옮겼을 가능성이 있습니다. "
            f"옮긴 것이 맞다면 기준국 구간 보정을 '이동함(all)'으로 설정하세요. "
            f"옮기지 않았다면 이것은 실제 지자기 변화이므로 그대로 두는 것이 맞습니다."
        )
    elif big:
        warnings.append(
            f"기록 공백 전후 레벨 차이({max(g[2] for g in big):.1f}nT)를 이동으로 보고 "
            f"공통 레벨에 맞췄습니다 - 이 경우 실제 날짜 간 지자기 변화도 함께 제거됩니다."
        )

    base = base.drop(columns=["_segment"])
    max_offset = max((abs(s.offset_applied_nt) for s in segments), default=0.0)
    return BaseLevelingResult(base=base, mode=mode, segments=segments,
                              common_level_nt=float(base["mag"].mean()),
                              max_offset_nt=max_offset,
                              warnings=warnings)
