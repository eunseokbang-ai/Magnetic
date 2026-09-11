"""User-driven distortion removal: the user marks a stretch of a survey
line (by point_id, e.g. from a drag-select on the time-series chart, or a
polygon drawn over a house/building on the map) where a ground structure
is visibly distorting the magnetic signal (typically a dipole-shaped
bump/dip), and this module removes that distortion by linearly
interpolating across the marked stretch between a robust "background
level" estimated just before it and just after it - the same "bridge
across the gap" idea used for despiking a single point, just applied to a
user-chosen run of points instead of an automatically detected spike.

The background level on each side is the median of a small window of
unflagged samples immediately adjacent to the run, not just the single
boundary sample - a dipole's tails often still slightly perturb the very
last point right at the edge of what the user selected, so anchoring to
one noisy sample instead of a short, more stable window can leave a
visible kink or residual offset instead of a clean match to the true
background.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Number of unflagged samples averaged (via median) on each side of a
# flagged run to estimate the local background level - see module
# docstring for why a window instead of just the single boundary sample.
_BACKGROUND_WINDOW = 5


def _local_background(values: np.ndarray, flagged: np.ndarray, start: int, step: int, window: int) -> float | None:
    """Median of up to `window` unflagged samples starting at `start` and
    walking in `step` direction (+1 = forward, -1 = backward). Returns
    None if `start` is out of bounds or no unflagged samples are found in
    that direction (e.g. the flagged run touches the very edge of a short
    line)."""
    collected = []
    n = len(values)
    k = start
    while 0 <= k < n and len(collected) < window:
        if not flagged[k]:
            collected.append(values[k])
        k += step
    return float(np.median(collected)) if collected else None


def _interpolate_flagged_runs(values: np.ndarray, flagged: np.ndarray, background_window: int = _BACKGROUND_WINDOW) -> np.ndarray:
    """Replace each contiguous run of flagged samples with a straight line
    between the local background level just before the run and just after
    it (see _local_background). A run touching one end of the array (no
    background estimate on that side) is held flat at the estimate that
    does exist; a run spanning the entire array is left unchanged (nothing
    to anchor to)."""
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
        left_bg = _local_background(values, flagged, i - 1, -1, background_window) if i - 1 >= 0 else None
        right_bg = _local_background(values, flagged, j, 1, background_window) if j < n else None
        if left_bg is not None and right_bg is not None:
            out[i:j] = np.linspace(left_bg, right_bg, j - i + 2)[1:-1]
        elif left_bg is not None:
            out[i:j] = left_bg
        elif right_bg is not None:
            out[i:j] = right_bg
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
