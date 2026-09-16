"""Despiking a clean record with one bad sample in it.

This is the case the detector most needs to get right and the one it used
to fail silently: on a smooth series the centred rolling median equals the
middle sample for most windows, so more than half the residuals in a
window are exactly zero and the local MAD is zero too. That zero used to
be replaced with NaN, and `resid > k * NaN` is False - so detection
switched itself off precisely where a spike is most obvious.

Real consequence: Cheongyang observatory jumped 39 nT for a single minute
on 2026-09-08 and went through despiking untouched into a base series.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from app.processing.despike import despike


def _smooth_day(n=1440, amplitude=20.0, noise=0.0, seed=0):
    """A day of 1-minute observatory data: a diurnal curve, nothing else."""
    t = np.arange(n)
    values = 50130.0 + amplitude * np.sin(2 * np.pi * t / n)
    if noise:
        values = values + np.random.default_rng(seed).normal(0, noise, n)
    return values


def test_one_bad_sample_in_an_otherwise_perfect_record_is_caught():
    values = _smooth_day()
    values[500] += 39.0

    cleaned, mask = despike(values, window_size=11, threshold_k=6.0, adaptive=False)

    assert mask[500], "the spike was not flagged"
    assert abs(cleaned[500] - 50130.0 - 20.0 * np.sin(2 * np.pi * 500 / 1440)) < 1.0


def test_it_does_not_flag_the_whole_record_when_the_local_scale_is_zero():
    """The other failure mode of a zero local scale: `resid > 0` would be
    true for every sample that is not exactly on its rolling median."""
    values = _smooth_day()

    _cleaned, mask = despike(values, window_size=11, threshold_k=6.0, adaptive=False)

    assert mask.sum() == 0, f"{mask.sum()} samples flagged in a clean record"


def test_a_noisy_record_still_behaves(seed=3):
    values = _smooth_day(noise=0.4, seed=seed)
    values[200] += 25.0
    values[900] -= 30.0

    cleaned, mask = despike(values, window_size=11, threshold_k=6.0, adaptive=False)

    assert mask[200] and mask[900]
    assert mask.sum() < 20, f"{mask.sum()} flagged - too eager on ordinary noise"
    assert abs(cleaned[200] - values[200]) > 20


def test_a_flat_record_with_one_spike():
    """No variation at all anywhere, so there is no local scale and no
    series-wide scale either. The spike must still not slip through as a
    silent NaN comparison."""
    values = np.full(600, 50130.0)
    values[300] += 15.0

    _cleaned, mask = despike(values, window_size=11, threshold_k=6.0, adaptive=False)

    assert mask[300]
