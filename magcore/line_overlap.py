"""Choosing between two recordings of the same stretch of one line.

A flight interrupted mid-line and resumed rarely picks up exactly where it
stopped. The drone usually backs up first, so when the two halves are
rejoined into one line (processing/lines.py::merge_continued_lines) there
is a stretch the survey flew twice - 116 m of it on the 2026-08 HaeNam
block, after a 6.3 minute stop.

Gridding does not handle that stretch sensibly on its own: whichever
sample happens to land in a cell wins, so the two recordings are
interleaved cell by cell, which is the one outcome nobody would choose.
And they are not equally good. An abrupt stop and a restart are exactly
when a slung sensor swings, so the end of the first recording and the
start of the second are the two places on the line most likely to be
disturbed - and the overlap is made of precisely those two pieces.

So the overlap is resolved rather than averaged or left to chance: one
recording is kept whole and the other's samples there are excluded, the
same way turn and takeoff samples already are. Keeping one whole avoids a
seam mid-overlap, which switching per sample would introduce.

Which one is kept is decided by the same normalised 4th-difference noise
metric the rest of this app uses for data quality (processing/noise_qc.py,
after Denisov et al. 2006): neighbouring samples of a potential field
cannot differ randomly, so the recording with the larger 4th difference
over the shared ground is the one that was shaking. That is a measurement,
not a guess, but it is still only a proxy - so which way each overlap went
is reported per line, and `overrides` lets the operator overrule any of
them after looking at the profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .noise_qc import normalized_4th_difference_rms

# Below this there is nothing worth resolving - a few samples of overlap is
# ordinary sample-spacing jitter at a join, not a re-flown stretch.
MIN_OVERLAP_M = 5.0
MIN_OVERLAP_SAMPLES = 10


@dataclass
class OverlapDecision:
    line_id: int
    overlap_m: float
    kept: str                 # "first" | "second"
    reason: str               # "noise" | "override" | "only_option"
    noise_first_nt: float | None
    noise_second_nt: float | None
    n_points_excluded: int
    first_time: str
    second_time: str


@dataclass
class OverlapResult:
    line_id: np.ndarray
    exclusion_reason: np.ndarray
    decisions: list[OverlapDecision] = field(default_factory=list)

    def summary(self) -> list[dict]:
        return [
            {
                "line_id": d.line_id,
                "overlap_m": d.overlap_m,
                "kept": d.kept,
                "reason": d.reason,
                "noise_first_nt": d.noise_first_nt,
                "noise_second_nt": d.noise_second_nt,
                "n_points_excluded": d.n_points_excluded,
                "first_time": d.first_time,
                "second_time": d.second_time,
            }
            for d in self.decisions
        ]


def resolve_line_overlaps(
    out: pd.DataFrame,
    value_col: str,
    dominant_azimuth_deg: float,
    line_id: np.ndarray,
    exclusion_reason: np.ndarray,
    overrides: dict[int, str] | None = None,
    min_overlap_m: float = MIN_OVERLAP_M,
) -> OverlapResult:
    """Within each line, drop the worse of two recordings of shared ground.

    A line is only examined when it was assembled from segments separated
    in time (the stop), which is what makes "the same ground twice" mean
    two recordings rather than ordinary along-track sampling.

    overrides: {line_id: "first" | "second"} to overrule the noise
    comparison for that line - the operator's call after seeing the
    profile.
    """
    line_id = line_id.copy()
    exclusion_reason = exclusion_reason.copy()
    overrides = overrides or {}

    az = np.radians(dominant_azimuth_deg)
    x, y = out["x"].to_numpy(), out["y"].to_numpy()
    along = x * np.cos(az) + y * np.sin(az)
    values = out[value_col].to_numpy(dtype=float)
    times = out["timestamp"]

    decisions: list[OverlapDecision] = []

    for lid in sorted({int(v) for v in np.unique(line_id) if v >= 0}):
        idx = np.flatnonzero(line_id == lid)
        if len(idx) < 2 * MIN_OVERLAP_SAMPLES:
            continue
        order = idx[np.argsort(times.to_numpy()[idx])]

        # Split at the largest time break; that is the stop, if there was
        # one. Everything before it is the first recording.
        t = times.to_numpy()[order].astype("datetime64[ns]").astype("int64") / 1e9
        gaps = np.diff(t)
        if len(gaps) == 0:
            continue
        k = int(np.argmax(gaps))
        if gaps[k] < 5.0:          # no real interruption on this line
            continue
        first, second = order[: k + 1], order[k + 1:]
        if len(first) < MIN_OVERLAP_SAMPLES or len(second) < MIN_OVERLAP_SAMPLES:
            continue

        a_lo, a_hi = along[first].min(), along[first].max()
        b_lo, b_hi = along[second].min(), along[second].max()
        lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
        if hi - lo < min_overlap_m:
            continue

        in_a = first[(along[first] >= lo) & (along[first] <= hi)]
        in_b = second[(along[second] >= lo) & (along[second] <= hi)]
        if len(in_a) < MIN_OVERLAP_SAMPLES or len(in_b) < MIN_OVERLAP_SAMPLES:
            continue

        noise_a = normalized_4th_difference_rms(values[in_a])
        noise_b = normalized_4th_difference_rms(values[in_b])

        override = overrides.get(lid) or overrides.get(str(lid))
        if override in ("first", "second"):
            kept, why = override, "override"
        elif not np.isfinite(noise_a) and not np.isfinite(noise_b):
            kept, why = "second", "only_option"
        elif not np.isfinite(noise_a):
            kept, why = "second", "only_option"
        elif not np.isfinite(noise_b):
            kept, why = "first", "only_option"
        else:
            kept, why = ("first", "noise") if noise_a <= noise_b else ("second", "noise")

        drop = in_b if kept == "first" else in_a
        line_id[drop] = -1
        exclusion_reason[drop] = "overlap_duplicate"

        decisions.append(OverlapDecision(
            line_id=lid,
            overlap_m=float(hi - lo),
            kept=kept,
            reason=why,
            noise_first_nt=float(noise_a) if np.isfinite(noise_a) else None,
            noise_second_nt=float(noise_b) if np.isfinite(noise_b) else None,
            n_points_excluded=int(len(drop)),
            first_time=str(pd.Timestamp(times.to_numpy()[first][-1])),
            second_time=str(pd.Timestamp(times.to_numpy()[second][0])),
        ))

    return OverlapResult(line_id=line_id, exclusion_reason=exclusion_reason,
                         decisions=decisions)
