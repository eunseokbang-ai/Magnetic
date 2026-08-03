"""User-driven distortion removal: the user marks a stretch of a survey
line (by point_id, e.g. from a drag-select on the time-series chart, or a
polygon drawn over a house/building on the map) where a ground structure
is visibly distorting the magnetic signal, and this module removes that
distortion by linearly interpolating across the marked stretch between
the last good value before it and the first good value after it - the
same "bridge across the gap" idea used for despiking a single point,
just applied to a user-chosen run of points instead of an automatically
detected spike.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _interpolate_flagged_runs(values: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """Replace each contiguous run of flagged samples with a straight line
    between the last unflagged sample before the run and the first
    unflagged sample after it. A run touching one end of the array (no
    anchor on that side) is held flat at the anchor that does exist; a
    run spanning the entire array is left unchanged (nothing to anchor
    to)."""
    out = values.copy()
    n = len(values)
    i = 0
    while i < n:
        if not flagged[i]:
            i += 1
            continue
        j = i
        while j < n and flagged[j]:
            j += 1
        left, right = i - 1, j
        if left >= 0 and right < n:
            out[i:j] = np.linspace(values[left], values[right], j - i + 2)[1:-1]
        elif left >= 0:
            out[i:j] = values[left]
        elif right < n:
            out[i:j] = values[right]
        i = j
    return out


def apply_manual_smoothing(
    df: pd.DataFrame,
    point_ids: set[int],
    columns: tuple[str, ...] = ("anomaly", "tmi"),
) -> pd.DataFrame:
    """Return a copy of df with `columns` smoothed over the point_ids in
    `point_ids`, run per line_id in timestamp order (matching how the
    user sees/selects the stretch, on the time-series chart or on the
    map). Excluded/ramp rows (line_id < 0) are skipped - there is no
    coherent "surrounding trend" to interpolate against there.

    Always returns a copy, even when point_ids is empty - callers (see
    store.py::set_manual_smoothing) rely on the result never aliasing df,
    since df is often a pristine snapshot (processed_base) that must stay
    independent from the mutable self.processed it's assigned to."""
    if not point_ids:
        return df.copy()
    out = df.copy()
    for line_id, group in out.groupby("line_id"):
        if line_id < 0:
            continue
        ordered = group.sort_values("timestamp")
        flagged = ordered["point_id"].isin(point_ids).to_numpy()
        if not flagged.any():
            continue
        for col in columns:
            out.loc[ordered.index, col] = _interpolate_flagged_runs(ordered[col].to_numpy(), flagged)
    return out
