"""Spike/outlier removal from the raw magnetometer time series, run before
the low-pass filter so a single bad sample (cultural noise - fences,
vehicles, power lines, electrical equipment) doesn't get smeared across a
wide window by the Butterworth filtfilt (a zero-phase IIR filter spreads
a spike's influence forward and backward through the whole segment, which
a windowed median never would).

Uses a rolling median + MAD (median absolute deviation) threshold, which
is itself robust to the spikes it's trying to detect - unlike a plain
rolling std, where a few huge spikes can inflate the std enough to hide
themselves from a std-based test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Converts MAD to a consistent estimator of standard deviation under a
# normal distribution - the standard robust-statistics scale factor.
_MAD_TO_STD = 1.4826


def _local_stats(s: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    med = s.rolling(window=window, center=True, min_periods=1).median()
    resid = s - med
    mad = resid.abs().rolling(window=window, center=True, min_periods=1).median()
    robust_std = (mad * _MAD_TO_STD).replace(0, np.nan)
    return med, resid, robust_std


def despike(
    values: np.ndarray,
    window_size: int = 11,
    threshold_k: float = 4.0,
    adaptive: bool = True,
    adaptive_gradient_threshold: float = 5.0,
    adaptive_expand_samples: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace samples that deviate from their local rolling median by
    more than threshold_k robust-standard-deviations with that median.
    Returns (cleaned_values, spike_mask).

    adaptive: when True (default), samples sitting in a steep along-track
    gradient region (adjacent-sample |difference| > adaptive_gradient_threshold,
    default 5.0 nT/sample) use a widened window
    (window_size + adaptive_expand_samples) instead of the base window for
    their local median/MAD. A steep-but-real ramp inflates the residual
    against a too-narrow median baseline and can trigger false-positive
    spike flags along its whole length; widening the baseline just in
    that region gives the median more real "ground" on either side of the
    ramp to sit on, without loosening detection anywhere else.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 5:
        return values.copy(), np.zeros(n, dtype=bool)

    window = max(3, int(window_size) | 1)  # force odd, so "center" has a true middle sample
    s = pd.Series(values)
    med, resid, robust_std = _local_stats(s, window)

    if adaptive and n >= 5:
        wide_window = window + max(0, int(adaptive_expand_samples))
        wide_window = wide_window if wide_window % 2 == 1 else wide_window + 1
        if wide_window > window:
            med_wide, resid_wide, robust_std_wide = _local_stats(s, wide_window)

            gradient = np.abs(np.diff(values, prepend=values[0]))
            # a sample counts as "steep" if either its own step-in or the
            # next sample's step-in exceeds the threshold, so the widened
            # window is used on both sides of a ramp, not just its start.
            steep = pd.Series(
                (gradient > adaptive_gradient_threshold) | (np.roll(gradient, -1) > adaptive_gradient_threshold)
            )
            med = med.where(~steep, med_wide)
            resid = resid.where(~steep, resid_wide)
            robust_std = robust_std.where(~steep, robust_std_wide)

    spike_mask = (resid.abs() > threshold_k * robust_std).fillna(False).to_numpy()

    cleaned = values.copy()
    cleaned[spike_mask] = med.to_numpy()[spike_mask]
    return cleaned, spike_mask
