"""Temporal low-pass filtering of the raw magnetometer time series."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter


def estimate_sample_rate_hz(timestamps: pd.Series) -> float:
    dt = timestamps.diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]
    if dt.empty:
        raise ValueError("샘플링 주기를 추정할 수 없습니다.")
    return 1.0 / dt.median()


def lowpass_filter(
    values: np.ndarray,
    timestamps: pd.Series,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter over a (near-)uniformly
    sampled time series. Falls back to returning the input unchanged for
    series too short to filter."""
    n = len(values)
    if n < order * 3 + 1:
        return np.asarray(values, dtype=float)

    fs = estimate_sample_rate_hz(timestamps)
    nyquist = fs / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        raise ValueError(
            f"차단주파수(cutoff_hz={cutoff_hz})는 0보다 크고 나이퀴스트 주파수({nyquist:.3f} Hz)보다 작아야 합니다."
        )

    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return filtfilt(b, a, np.asarray(values, dtype=float))


def notch_filter(
    values: np.ndarray,
    timestamps: pd.Series,
    freq_hz: float,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Zero-phase IIR notch filter removing a single known interference
    frequency (e.g. UAV motor rotation frequency, or a cultural noise
    source identified from the power spectrum - see processing/spectrum.py)
    from a (near-)uniformly sampled time series, per the UAV magnetics
    guidelines' recommendation to design a notch filter once a specific
    frequency is identified in the power spectrum. A no-op when freq_hz is
    at or above the Nyquist frequency (can't be represented at this
    sampling rate) or the series is too short to filter."""
    n = len(values)
    if n < 19:  # filtfilt needs > 3*max(len(a), len(b)); iirnotch order 2 -> len 3
        return np.asarray(values, dtype=float)

    fs = estimate_sample_rate_hz(timestamps)
    nyquist = fs / 2.0
    if freq_hz <= 0 or freq_hz >= nyquist:
        return np.asarray(values, dtype=float)

    b, a = iirnotch(freq_hz / nyquist, quality_factor)
    return filtfilt(b, a, np.asarray(values, dtype=float))


def _window_samples(timestamps: pd.Series, window_seconds: float, min_samples: int) -> int:
    fs = estimate_sample_rate_hz(timestamps)
    n = max(min_samples, int(round(window_seconds * fs)))
    return n if n % 2 == 1 else n + 1  # both filters below require an odd window length


def moving_average_filter(values: np.ndarray, timestamps: pd.Series, window_seconds: float) -> np.ndarray:
    """Simple centered moving-average smoother, sized in seconds (rather
    than a raw sample count) so it behaves consistently regardless of the
    logger's sampling rate. The crudest of the three filter options -
    unlike the zero-phase Butterworth low-pass or Savitzky-Golay below,
    it has no sharp frequency cutoff and does mild peak-broadening, but
    it's simple, fast, and a common baseline in aeromag processing
    software."""
    values = np.asarray(values, dtype=float)
    window = _window_samples(timestamps, window_seconds, min_samples=3)
    if window >= len(values):
        return values
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def savgol_filter_1d(
    values: np.ndarray, timestamps: pd.Series, window_seconds: float, polyorder: int = 3
) -> np.ndarray:
    """Savitzky-Golay smoother: fits a local polynomial (degree
    `polyorder`) over a sliding window and evaluates it at the window
    center, sized in seconds like moving_average_filter above. Unlike a
    plain moving average, it preserves peak height/width much better
    (it's a common choice specifically because it smooths noise without
    flattening real anomalies as aggressively)."""
    values = np.asarray(values, dtype=float)
    window = _window_samples(timestamps, window_seconds, min_samples=polyorder + 2)
    if window > len(values):
        return values
    if window <= polyorder:
        raise ValueError(f"Savitzky-Golay 창 크기({window}개 샘플)가 다항식 차수({polyorder})보다 커야 합니다.")
    return savgol_filter(values, window_length=window, polyorder=polyorder)
