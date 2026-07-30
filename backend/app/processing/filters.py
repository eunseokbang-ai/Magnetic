"""Temporal low-pass filtering of the raw magnetometer time series."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


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
