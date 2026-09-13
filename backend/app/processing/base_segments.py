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


def find_base_segments(
    base_df: pd.DataFrame,
    gap_minutes: float = DEFAULT_GAP_MINUTES,
    step_threshold_nt: float = DEFAULT_STEP_THRESHOLD_NT,
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

    seg_id, reasons = find_base_segments(base, gap_minutes, step_threshold_nt)
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

    # Anchor: the weighted level of the segments that are staying put, so
    # levelling a step never shifts the rest of the record.
    anchored = [sid for sid in kept_ids if not should_level(sid)]
    if anchored:
        common = float((levels[anchored] * sizes[anchored]).sum() / sizes[anchored].sum())
    else:
        common = float((levels * sizes).sum() / sizes.sum())

    segments: list[BaseSegment] = []
    for i, sid in enumerate(kept_ids):
        rows = base["_segment"] == sid
        level = float(levels[sid])
        do_level = should_level(sid)
        offset = (common - level) if do_level else 0.0
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
                              common_level_nt=common, max_offset_nt=max_offset,
                              warnings=warnings)
