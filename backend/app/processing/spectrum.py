"""Power spectrum diagnostic for a single flight line's raw magnetometer
signal, to visually identify UAV electromagnetic interference: the UAV
magnetics guidelines describe both a static/low-frequency and a dynamic,
higher-frequency component (motor mechanical rotation, typically ~45-60Hz
for common commercial UAVs, and sensor-swing frequencies around
0.1-0.6Hz for a suspended sensor) - both show up as narrow peaks in the
power spectral density, distinct from the broadband instrument noise
floor and the smooth low-frequency geological signal. Once identified
here, a candidate frequency can be removed with
processing/filters.py:notch_filter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

from .filters import estimate_sample_rate_hz

# A peak has to stand out from the local noise floor by at least this
# multiple to be reported as a candidate interference frequency - avoids
# flagging ordinary spectral wiggle as if it were a discrete tone.
_PEAK_PROMINENCE_FACTOR = 5.0
_MAX_PEAKS_REPORTED = 8


def compute_power_spectrum(values: np.ndarray, timestamps: pd.Series) -> dict:
    """Welch power spectral density of a single line's signal, plus a
    simple local-maxima peak picker for candidate notch-filter frequencies.
    available=False when there isn't enough data to estimate a spectrum."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 32:
        return {"available": False}
    values = values[finite]

    fs = estimate_sample_rate_hz(timestamps)
    nperseg = min(len(values), max(32, int(2 ** np.floor(np.log2(len(values) / 4)))))
    freqs, psd = welch(values, fs=fs, nperseg=nperseg)

    # Skip the DC bin (index 0) - it reflects the mean field level, not
    # noise, and would otherwise dominate/mask everything else.
    freqs, psd = freqs[1:], psd[1:]
    if freqs.size == 0:
        return {"available": False}

    log_psd = np.log10(np.maximum(psd, 1e-30))
    median_log = float(np.median(log_psd))
    mad_log = float(np.median(np.abs(log_psd - median_log))) * 1.4826 or 1e-9

    is_local_max = np.r_[False, (psd[1:-1] > psd[:-2]) & (psd[1:-1] > psd[2:]), False]
    prominent = is_local_max & (log_psd > median_log + np.log10(_PEAK_PROMINENCE_FACTOR))
    peak_idx = np.flatnonzero(prominent)
    peak_idx = peak_idx[np.argsort(psd[peak_idx])[::-1]][:_MAX_PEAKS_REPORTED]
    peak_idx = np.sort(peak_idx)

    return {
        "available": True,
        "sample_rate_hz": fs,
        "freqs_hz": freqs.tolist(),
        "psd": psd.tolist(),
        "peak_frequencies_hz": [float(freqs[i]) for i in peak_idx],
    }
