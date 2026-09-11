"""Detects when two (or more) flight lines in the *main survey* are repeat
passes over the same physical track - flown twice for whatever reason
(reflight after a bad first pass, an accidental duplicate, a deliberate
repeat folded into the main survey instead of a separate dedicated
repeatability-test flight - see processing/repeatability.py, which shares
the same track-grouping primitive but with a much looser tolerance meant
for a hand-flown test box's drift) - and automatically keeps only the
better-quality pass for the overlapping stretch, instead of gridding both
and letting whichever line's value happens to land on a given cell win
arbitrarily (or letting interpolation blend a good pass with a noisy one).

"Better quality" is the same normalized 4th-difference noise metric used
elsewhere in this app for QC (processing/noise_qc.py) - the standard
airborne-magnetics roughness indicator - computed over just the
overlapping stretch of each pass (not the whole line, in case only part
of a line needed re-flying), lower is better. Only the overlapping
stretch of the worse pass(es) is excluded, never a whole line - a partial
re-flight (e.g. only the second half of a line was noisy and got
re-flown) keeps the non-overlapping parts of every pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .line_grouping import MIN_POINTS_PER_PASS, LinePass, group_passes, line_pass
from .noise_qc import normalized_4th_difference_rms

# Much tighter than repeatability.py's 10m/15deg default - the job here is
# telling apart "the exact same line, flown twice" from "just the next
# survey line over", not tolerating a hand-flown test box's drift.
# RTK-guided autopilot repeats of the same waypoint line are commonly
# within 1-3m of each other; even a hand-guided repeat is much closer than
# a normal line-to-line spacing (typically tens of meters or more).
DEFAULT_PERP_TOLERANCE_M = 8.0
DEFAULT_ANGLE_TOLERANCE_DEG = 15.0


def resolve_duplicate_lines(
    df: pd.DataFrame,
    value_col: str,
    perp_tolerance_m: float = DEFAULT_PERP_TOLERANCE_M,
    angle_tolerance_deg: float = DEFAULT_ANGLE_TOLERANCE_DEG,
) -> dict:
    """df must already have line_id assigned (processing/lines.py) and
    point_id/x/y/timestamp/value_col columns; only rows with line_id >= 0
    are considered. Returns a summary dict - "excluded_point_ids" (empty
    if no duplicate track was found) is what the caller should mark
    excluded, the same way other automatic QC steps (e.g. sway detection)
    do."""
    empty = {"available": True, "n_groups": 0, "n_points_excluded": 0, "groups": [], "excluded_point_ids": []}
    active = df[df["line_id"] >= 0]
    if active.empty:
        return empty

    passes = [p for lid, group in active.groupby("line_id") if (p := line_pass(int(lid), group, value_col=value_col))]
    if len(passes) < 2:
        return empty

    groups = group_passes(passes, perp_tolerance_m, angle_tolerance_deg)
    if not groups:
        return empty

    excluded_point_ids: list[int] = []
    group_summaries = []
    for group in groups:
        summary = _resolve_group(group)
        if summary is None:
            continue
        group_summaries.append(summary)
        excluded_point_ids.extend(summary.pop("excluded_point_ids"))

    return {
        "available": True,
        "n_groups": len(group_summaries),
        "n_points_excluded": len(excluded_point_ids),
        "groups": group_summaries,
        "excluded_point_ids": excluded_point_ids,
    }


def _resolve_group(group: list[LinePass]) -> dict | None:
    """For a group of >=2 passes over the same track, find their common
    overlapping stretch (projected onto the shared canonical axis), rank
    each pass's quality *within just that stretch*, and return the
    point_ids of every pass but the best one, restricted to that stretch."""
    ref_direction = group[0].canonical_direction
    ref_centroid = group[0].centroid

    s_values = [(np.column_stack([p.x, p.y]) - ref_centroid) @ ref_direction for p in group]
    lo = max(float(s.min()) for s in s_values)
    hi = min(float(s.max()) for s in s_values)
    if hi <= lo:
        return None

    ranked = []
    for p, s in zip(group, s_values):
        in_overlap = (s >= lo) & (s <= hi)
        if int(in_overlap.sum()) < MIN_POINTS_PER_PASS:
            return None
        rms = normalized_4th_difference_rms(p.value[in_overlap])
        if not np.isfinite(rms):
            return None
        ranked.append((p, in_overlap, rms))

    # Lower normalized 4th-difference RMS = smoother = better quality -
    # same convention as noise_qc.py's per-line flagging.
    ranked.sort(key=lambda item: item[2])
    best_pass = ranked[0][0]

    excluded_point_ids: list[int] = []
    for p, in_overlap, _rms in ranked[1:]:
        excluded_point_ids.extend(int(pid) for pid in p.point_ids[in_overlap])

    return {
        "line_ids": [p.line_id for p in group],
        "best_line_id": int(best_pass.line_id),
        "overlap_length_m": float(hi - lo),
        "passes": [
            {
                "line_id": int(p.line_id),
                "quality_nt": rms,
                "n_overlap_points": int(in_overlap.sum()),
                "kept": bool(p.line_id == best_pass.line_id),
            }
            for p, in_overlap, rms in ranked
        ],
        "excluded_point_ids": excluded_point_ids,
    }
