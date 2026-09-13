"""Regression tests for the along-line low-pass smoothing that prevents
flight-line-parallel "corrugation" in gridded output (see
processing.gridding._along_line_lowpass and grid_points' line_id /
along_line_smooth_wavelength_m parameters).
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from app.processing.gridding import _along_line_lowpass, grid_points


def _synthetic_survey(n_lines=10, line_spacing=100.0, along_spacing=2.0, line_len=2000.0, noise_std=5.0, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys, vs, lids = [], [], [], []
    for i in range(n_lines):
        n = int(line_len / along_spacing)
        y = np.arange(n) * along_spacing
        x = np.full(n, i * line_spacing)
        trend = 0.01 * x + 0.005 * y  # smooth regional signal, resolvable cross-line too
        noise = rng.normal(0, noise_std, n)  # per-line-only high-frequency detail - the corrugation source
        xs.append(x)
        ys.append(y)
        vs.append(trend + noise)
        lids.append(np.full(n, i))
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(vs),
        np.concatenate(lids),
        line_spacing,
    )


def _row_roughness(values_2d):
    """Row-to-row (along-northing) difference std, as a stand-in for
    corrugation amplitude - a smaller value means adjacent grid rows agree
    more, i.e. less fine line-parallel ridging."""
    finite_rows = np.isfinite(values_2d).all(axis=1)
    sub = values_2d[finite_rows]
    return float(np.nanstd(np.diff(sub, axis=0)))


def test_along_line_lowpass_reduces_line_parallel_noise():
    x, y, v, lid, spacing = _synthetic_survey()
    smoothed = _along_line_lowpass(x, y, v, lid, wavelength_m=spacing)
    # per-line variance of the smoothed values should drop sharply relative
    # to the raw noisy values, since the injected noise has no structure
    # below the smoothing wavelength
    raw_std = np.std(v)
    smoothed_std = np.std(smoothed)
    assert smoothed_std < raw_std * 0.7


def test_along_line_lowpass_preserves_broad_signal():
    """A real anomaly that varies smoothly over many along-line samples
    (wavelength much longer than the smoothing window) should survive
    smoothing largely intact - this isn't supposed to flatten everything,
    only the fine detail no cross-line interpolation could resolve anyway."""
    rng = np.random.default_rng(1)
    n = 500
    y = np.arange(n) * 2.0  # 1000m line, 2m spacing
    x = np.zeros(n)
    broad_signal = 50.0 * np.sin(y / 300.0)  # ~1885m wavelength, much longer than 100m smoothing window
    values = broad_signal + rng.normal(0, 1.0, n)
    lid = np.zeros(n, dtype=int)

    smoothed = _along_line_lowpass(x, y, values, lid, wavelength_m=100.0)
    corr = np.corrcoef(smoothed, broad_signal)[0, 1]
    assert corr > 0.99


def test_along_line_lowpass_skips_tiny_lines_without_crashing():
    x = np.array([0.0, 0.0, 100.0, 100.0, 100.0, 100.0])
    y = np.array([0.0, 1.0, 0.0, 1.0, 2.0, 3.0])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    lid = np.array([0, 0, 1, 1, 1, 1])  # line 0 has only 2 points (< 3, skipped)
    out = _along_line_lowpass(x, y, values, lid, wavelength_m=50.0)
    assert out.shape == values.shape
    assert np.array_equal(out[:2], values[:2])  # untouched (too few points to smooth)
    assert np.isfinite(out).all()


def test_grid_points_along_line_smooth_reduces_grid_corrugation():
    x, y, v, lid, spacing = _synthetic_survey()
    cell = 10.0
    g_raw = grid_points(x, y, v, cell, method="nearest", max_distance_m=60.0)
    g_smooth = grid_points(
        x, y, v, cell, method="nearest", max_distance_m=60.0,
        line_id=lid, along_line_smooth_wavelength_m=spacing,
    )
    raw_roughness = _row_roughness(g_raw.values)
    smooth_roughness = _row_roughness(g_smooth.values)
    assert smooth_roughness < raw_roughness * 0.5
    # the underlying regional trend should still be there, not just noise
    # replaced by a flat/degenerate surface
    assert np.nanstd(g_smooth.values) > 1.0


def test_grid_points_without_line_id_is_unaffected():
    """Omitting line_id (or the wavelength) must behave exactly as before
    - along-line smoothing is opt-in per call, not implicitly forced on
    just because grid_points supports it now."""
    x, y, v, lid, spacing = _synthetic_survey(n_lines=3, line_len=200.0)
    g1 = grid_points(x, y, v, 10.0, method="nearest", max_distance_m=60.0)
    g2 = grid_points(x, y, v, 10.0, method="nearest", max_distance_m=60.0, line_id=lid)  # no wavelength given
    np.testing.assert_array_equal(g1.values, g2.values)
