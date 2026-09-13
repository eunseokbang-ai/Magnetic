"""Validation of the power-spectrum diagnostic and the notch filter it's
meant to feed: inject a known sinusoidal interference tone into an
otherwise smooth signal, confirm compute_power_spectrum finds it as a
peak, then confirm notch_filter removes it while leaving a signal with no
matching tone (near-)unchanged."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.processing.filters import notch_filter
from app.processing.spectrum import compute_power_spectrum

RNG = np.random.default_rng(0)


def _make_series(n=2000, fs=50.0, tone_hz=None, tone_amp=5.0, noise_std=0.1):
    t = pd.Series(pd.date_range("2024-01-01", periods=n, freq=pd.Timedelta(seconds=1.0 / fs)))
    smooth = 50000.0 + 2.0 * np.sin(np.linspace(0, 2 * np.pi, n))
    values = smooth + RNG.normal(0, noise_std, n)
    if tone_hz is not None:
        seconds = np.arange(n) / fs
        values = values + tone_amp * np.sin(2 * np.pi * tone_hz * seconds)
    return values, t


def test_power_spectrum_detects_injected_tone():
    tone_hz = 12.5
    values, t = _make_series(tone_hz=tone_hz, tone_amp=8.0, noise_std=0.05)
    result = compute_power_spectrum(values, t)
    assert result["available"]
    peaks = result["peak_frequencies_hz"]
    assert peaks, "expected at least one detected peak"
    closest = min(peaks, key=lambda f: abs(f - tone_hz))
    print(f"peaks={peaks} closest_to_{tone_hz}={closest}")
    assert abs(closest - tone_hz) < 1.0


def test_power_spectrum_unavailable_for_short_series():
    values, t = _make_series(n=10)
    result = compute_power_spectrum(values, t)
    assert result["available"] is False


def test_notch_filter_removes_injected_tone():
    tone_hz = 12.5
    values, t = _make_series(n=4000, tone_hz=tone_hz, tone_amp=8.0, noise_std=0.05)
    filtered = notch_filter(values, t, tone_hz, quality_factor=30.0)

    spectrum_before = compute_power_spectrum(values, t)
    spectrum_after = compute_power_spectrum(filtered, t)

    freqs = np.array(spectrum_before["freqs_hz"])
    idx = int(np.argmin(np.abs(freqs - tone_hz)))
    psd_before = spectrum_before["psd"][idx]
    psd_after = spectrum_after["psd"][idx]
    print(f"psd at {tone_hz}Hz: before={psd_before:.4g} after={psd_after:.4g}")
    assert psd_after < psd_before * 0.1, "notch filter should strongly attenuate the tone's power"


def test_notch_filter_noop_above_nyquist():
    values, t = _make_series(n=200, fs=10.0)
    filtered = notch_filter(values, t, freq_hz=100.0)  # far above Nyquist (5Hz)
    assert np.allclose(filtered, values)


def test_notch_filter_leaves_clean_signal_mostly_unchanged():
    values, t = _make_series(n=2000, tone_hz=None, noise_std=0.05)
    filtered = notch_filter(values, t, freq_hz=15.0, quality_factor=30.0)
    assert np.std(filtered - values) < 0.5


if __name__ == "__main__":
    test_power_spectrum_detects_injected_tone()
    test_power_spectrum_unavailable_for_short_series()
    test_notch_filter_removes_injected_tone()
    test_notch_filter_noop_above_nyquist()
    test_notch_filter_leaves_clean_signal_mostly_unchanged()
    print("ALL CHECKS PASSED")
