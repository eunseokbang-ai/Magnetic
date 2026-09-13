"""Levelling a base station log that was recorded at more than one spot.

The diurnal correction subtracts (base reading - a reference level) from
every drone sample, and takes that reference once, as the mean of the base
over the whole span of drone data. That is right while the base sits in
one place: its absolute level then cancels, and only the time variation -
which is what the correction is for - is removed.

It stops being right the moment the base is moved. A different spot reads
a different absolute field, often by tens or hundreds of nT, and that step
does not cancel against a single survey-wide reference. Each flight is
then shifted by the difference between its own base level and that one
reference - a constant, whole-day offset in the anomaly, applied before
any levelling step runs and indistinguishable from real field there.
Measured on a two-day synthetic with no anomaly in it at all and the base
moved 200 nT between days: the corrected data came out with a -200.00 nT
step between the days, exactly the move.

So the base is levelled here first, before it is ever used as a reference:
each deployment is found, and each is shifted to one common level. What
survives is the time variation, which is the only part of a base log that
was ever transferable between locations.

Two things mark a deployment boundary, and both are required to be the
base's doing rather than the field's:

* a gap in logging - the unit was off, which is what happens when it is
  picked up and carried;
* a step in level far faster than the field itself moves. Diurnal
  variation runs at roughly 0.1-2 nT per minute; setting a magnetometer
  down somewhere else shows up as tens of nT between consecutive samples.
  Requiring the jump to be both large in absolute terms and large against
  the log's own sample-to-sample behaviour keeps a magnetic storm - which
  is fast but not discontinuous - from being read as a move.

What this deliberately does not do is decide whether the base moved at
all. Segments are always reported, even when only one is found, so
"the base stayed put" is something the operator reads off the summary
rather than something assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A base log is normally continuous; this much silence means the unit was
# off, which for a field base almost always means it was being moved.
DEFAULT_GAP_MINUTES = 10.0
# Minimum size of an instantaneous level change to call it a relocation.
# Well above any real between-sample diurnal change, well below the tens
# to hundreds of nT a different location gives.
DEFAULT_STEP_THRESHOLD_NT = 20.0
# ...and it must also stand out against how much this particular log
# normally moves between samples, so a noisy logger does not fragment.
_STEP_ROBUST_K = 8.0


@dataclass
class BaseSegment:
    index: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_samples: int
    level_nt: float
    offset_applied_nt: float
    split_reason: str | None  # why this segment started: "gap" | "step" | None


@dataclass
class BaseLevelingResult:
    base: pd.DataFrame          # levelled copy, safe to use as the reference
    segments: list[BaseSegment] = field(default_factory=list)
    common_level_nt: float | None = None
    max_offset_nt: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    def summary(self) -> dict:
        return {
            "n_segments": self.n_segments,
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
    """Segment id per row, plus the reason each new segment started."""
    base = base_df.sort_values("timestamp")
    t = base["timestamp"].astype("int64").to_numpy()
    mag = base["mag"].to_numpy(dtype=float)
    n = len(mag)
    if n == 0:
        return np.zeros(0, dtype=int), []

    dt_s = np.diff(t) / 1e9
    dmag = np.diff(mag)

    # How much this log normally moves between samples, measured robustly
    # so the very steps being looked for do not inflate the threshold.
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
    gap_minutes: float = DEFAULT_GAP_MINUTES,
    step_threshold_nt: float = DEFAULT_STEP_THRESHOLD_NT,
    min_segment_samples: int = 30,
) -> BaseLevelingResult:
    """Shift each base deployment onto one common level.

    The common level is the sample-count-weighted mean of the segment
    levels, so the longest deployment - usually the one the survey is
    mostly referenced against anyway - moves least.
    """
    base = base_df.sort_values("timestamp").reset_index(drop=True).copy()
    if base.empty:
        return BaseLevelingResult(base=base)

    seg_id, reasons = find_base_segments(base, gap_minutes, step_threshold_nt)
    base["_segment"] = seg_id

    # Fold a too-short segment into its predecessor: a handful of samples
    # cannot establish a level, and treating one as its own deployment
    # would shift it by whatever its noise happened to average to.
    counts = base["_segment"].value_counts().sort_index()
    keep = [s for s, c in counts.items() if c >= min_segment_samples]
    if not keep:
        keep = [int(counts.idxmax())]
    remap = {}
    last_kept = keep[0]
    for s in counts.index:
        if s in keep:
            last_kept = s
        remap[s] = last_kept
    base["_segment"] = base["_segment"].map(remap)

    levels = base.groupby("_segment")["mag"].median()
    sizes = base.groupby("_segment")["mag"].size()
    common = float((levels * sizes).sum() / sizes.sum())

    segments: list[BaseSegment] = []
    for i, (sid, level) in enumerate(levels.items()):
        rows = base["_segment"] == sid
        offset = common - float(level)
        base.loc[rows, "mag"] = base.loc[rows, "mag"] + offset
        sub = base.loc[rows, "timestamp"]
        reason = None
        if i > 0:
            reason = reasons[i - 1] if i - 1 < len(reasons) else "gap"
        segments.append(BaseSegment(
            index=i,
            start=sub.min(),
            end=sub.max(),
            n_samples=int(rows.sum()),
            level_nt=float(level),
            offset_applied_nt=offset,
            split_reason=reason,
        ))

    max_offset = max((abs(s.offset_applied_nt) for s in segments), default=0.0)
    warnings: list[str] = []
    if len(segments) > 1:
        spread = float(levels.max() - levels.min())
        warnings.append(
            f"기준국 자료가 {len(segments)}개 구간으로 나뉘고 구간별 레벨이 최대 {spread:.1f}nT "
            f"차이납니다 - 기준국을 옮긴 것으로 보입니다. 각 구간을 공통 레벨로 맞춘 뒤 "
            f"일변화 보정에 사용합니다(보정하지 않으면 이 차이가 비행일별 상수 오프셋으로 "
            f"자기이상도에 그대로 남습니다)."
        )

    base = base.drop(columns=["_segment"])
    return BaseLevelingResult(base=base, segments=segments,
                              common_level_nt=common, max_offset_nt=max_offset,
                              warnings=warnings)
