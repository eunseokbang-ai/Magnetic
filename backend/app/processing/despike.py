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


def despike(values: np.ndarray, window_size: int = 11, threshold_k: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Replace samples that deviate from their local rolling median by
    more than threshold_k robust-standard-deviations with that median.
    Returns (cleaned_values, spike_mask)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 5:
        return values.copy(), np.zeros(n, dtype=bool)

    window = max(3, int(window_size) | 1)  # force odd, so "center" has a true middle sample
    s = pd.Series(values)
    med = s.rolling(window=window, center=True, min_periods=1).median()
    resid = s - med
    mad = resid.abs().rolling(window=window, center=True, min_periods=1).median()
    robust_std = (mad * _MAD_TO_STD).replace(0, np.nan)
    spike_mask = (resid.abs() > threshold_k * robust_std).fillna(False).to_numpy()

    cleaned = values.copy()
    cleaned[spike_mask] = med.to_numpy()[spike_mask]
    return cleaned, spike_mask
