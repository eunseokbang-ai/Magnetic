"""Base (diurnal) station QC: trims the installation/pickup transients at
the start and end of the log and despikes the interior, before the base
series is used for diurnal correction. Without this, a base logger's own
handling noise (setting the unit down, walking away, picking it back up)
gets interpolated straight into the diurnal correction applied to every
drone sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .despike import despike


@dataclass
class BaseQCResult:
    raw: pd.DataFrame  # untouched, as loaded
    corrected: pd.DataFrame  # trimmed + despiked
    n_trimmed_start: int
    n_trimmed_end: int
    n_spikes_removed: int


def _first_settled_index(below_threshold: np.ndarray, confirm_n: int, limit: int) -> int:
    """First index i (<= limit) such that below_threshold[i:i+confirm_n] is
    entirely True - i.e. the record has been "settled" (locally quiet) for
    at least confirm_n consecutive samples. Falls back to `limit` if no
    such stretch exists within the search range, so a genuinely noisy
    start/end is capped rather than trimming the whole record."""
    n = len(below_threshold)
    confirm_n = max(1, min(confirm_n, n))
    if confirm_n == 1:
        idx = np.flatnonzero(below_threshold[: limit + 1])
        return int(idx[0]) if idx.size else limit
    window_sum = np.convolve(below_threshold.astype(int), np.ones(confirm_n, dtype=int), mode="valid")
    candidates = np.flatnonzero(window_sum == confirm_n)
    candidates = candidates[candidates <= limit]
    return int(candidates[0]) if candidates.size else limit


def trim_base_transients(
    df: pd.DataFrame,
    window_seconds: float = 30.0,
    threshold_k: float = 6.0,
    confirm_seconds: float = 60.0,
    max_trim_fraction: float = 0.2,
) -> dict:
    """Detect and drop the initial "installation" and final "pickup"
    transient: a period of anomalously high short-term variability
    (handling, mechanical shock, proximity to the operator/vehicle while
    walking away) relative to the settled interior of the log. Uses the
    same robust local-median/MAD machinery as despike(), just applied to
    find a *stretch* to drop rather than individual spikes to replace.

    max_trim_fraction caps how much can be trimmed from each end (as a
    fraction of the whole record), so a record that never looks "settled"
    within that budget is trimmed only up to the cap, not indefinitely."""
    n = len(df)
    if n < 10:
        return {"trimmed": df.reset_index(drop=True), "n_trimmed_start": 0, "n_trimmed_end": 0}

    t = df["timestamp"]
    dt = t.diff().dt.total_seconds().dropna()
    dt_median = float(dt[dt > 0].median()) if not dt.empty else 1.0
    window = max(3, int(round(window_seconds / dt_median)) | 1)
    confirm_n = max(1, int(round(confirm_seconds / dt_median)))
    max_trim_n = max(0, int(n * max_trim_fraction))

    mag = pd.Series(df["mag"].to_numpy(), dtype=float)
    med = mag.rolling(window=window, center=True, min_periods=1).median()
    local_var = (mag - med).abs().rolling(window=window, center=True, min_periods=1).median().to_numpy()

    # "settled" baseline variability from the middle 60% of the record,
    # away from the transients we're trying to characterize.
    lo, hi = int(n * 0.2), int(n * 0.8)
    baseline = float(np.nanmedian(local_var[lo:hi])) if hi > lo else float(np.nanmedian(local_var))
    baseline = baseline if baseline > 0 else 1e-9
    threshold = threshold_k * baseline

    below = local_var <= threshold
    n_start = _first_settled_index(below, confirm_n, max_trim_n)
    n_end = _first_settled_index(below[::-1], confirm_n, max_trim_n)

    if n_start + n_end >= n:
        n_start, n_end = 0, 0

    trimmed = df.iloc[n_start : n - n_end].reset_index(drop=True)
    return {"trimmed": trimmed, "n_trimmed_start": n_start, "n_trimmed_end": n_end}


def process_base_station(
    df: pd.DataFrame,
    trim_enabled: bool = True,
    trim_window_seconds: float = 30.0,
    trim_threshold_k: float = 6.0,
    trim_confirm_seconds: float = 60.0,
    trim_max_fraction: float = 0.2,
    despike_enabled: bool = True,
    despike_window_size: int = 11,
    despike_threshold_k: float = 5.0,
) -> BaseQCResult:
    raw = df.sort_values("timestamp").reset_index(drop=True)

    if trim_enabled:
        trim_result = trim_base_transients(raw, trim_window_seconds, trim_threshold_k, trim_confirm_seconds, trim_max_fraction)
        working = trim_result["trimmed"]
        n_trimmed_start = trim_result["n_trimmed_start"]
        n_trimmed_end = trim_result["n_trimmed_end"]
    else:
        working = raw.copy()
        n_trimmed_start = n_trimmed_end = 0

    if despike_enabled and len(working) >= 5:
        cleaned, spike_mask = despike(working["mag"].to_numpy(), despike_window_size, despike_threshold_k)
        working = working.copy()
        working["mag"] = cleaned
        n_spikes_removed = int(spike_mask.sum())
    else:
        n_spikes_removed = 0

    return BaseQCResult(
        raw=raw,
        corrected=working.reset_index(drop=True),
        n_trimmed_start=n_trimmed_start,
        n_trimmed_end=n_trimmed_end,
        n_spikes_removed=n_spikes_removed,
    )
