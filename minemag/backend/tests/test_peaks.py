"""THD/ASA prominence peaks on gridded synthetic dipoles."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from app.processing.peaks import _prominences, derivative_grid, find_peaks, type_class
from magcore.dipole_fit import field_direction_enu

INC, DEC = 53.0, -8.0


def _dipole(x, y, h, sx, sy, depth, moment):
    f = field_direction_enu(INC, DEC)
    rx, ry, rz = x - sx, y - sy, h + depth
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    dot = (f[0] * rx + f[1] * ry + f[2] * rz) / r
    return 1e9 * (4e-7 * np.pi * moment / (4 * np.pi * r**3)) * (3 * dot**2 - 1)


def _grid(cell=0.1, noise=2.0, sources=((3.0, 3.0, 0.2, 4.0), (7.0, 7.0, 0.3, 0.6))):
    """Gridded data is smooth at the cell scale (the readings were
    low-passed and gridded), so the noise is spatially correlated over a
    few cells rather than white per cell."""
    e = np.arange(0, 10 + cell, cell)
    n = np.arange(0, 10 + cell, cell)
    X, Y = np.meshgrid(e, n)
    rng = np.random.default_rng(0)
    w = gaussian_filter(rng.standard_normal(X.shape), 3)
    g = sum(_dipole(X, Y, 0.5, *s) for s in sources) + noise * w / w.std()
    return g, e, n


def test_thd_peaks_sit_over_the_dipoles_and_the_anomaly_maximum_does_not():
    g, e, n = _grid()
    thd = derivative_grid(g, 0.1, "thd")
    peaks, info = find_peaks(thd, g, e, n, 0.1, k=20.0)
    assert info["mad"] > 0 and info["n_local_maxima"] >= 2
    assert len(peaks) == 2
    for sx, sy, _d, _m in ((3.0, 3.0, 0, 0), (7.0, 7.0, 0, 0)):
        near = min(peaks, key=lambda p: np.hypot(p["x"] - sx, p["y"] - sy))
        assert np.hypot(near["x"] - sx, near["y"] - sy) < 0.3      # the paper's THD errors: 0.13-0.37 m
        assert near["k_needed"] >= 20.0
    # the anomaly itself peaks off the object (its lobes are north and south)
    r, c = np.unravel_index(np.nanargmax(np.abs(g)), g.shape)
    assert np.hypot(e[c] - 3.0, n[r] - 3.0) >= 0.15
    big = max(peaks, key=lambda p: p["p2p_nt"])
    assert big["p2p_nt"] > 500 and big["type_class"] == "대전차지뢰급"
    small = min(peaks, key=lambda p: p["p2p_nt"])
    assert small["type_class"] in ("대인지뢰급", "소형 금속/미상")


def test_k_controls_how_many_peaks_pass_and_noise_bumps_need_small_k():
    g, e, n = _grid(noise=5.0)
    thd = derivative_grid(g, 0.1, "thd")
    loose, _ = find_peaks(thd, g, e, n, 0.1, k=2.0)
    strict, _ = find_peaks(thd, g, e, n, 0.1, k=20.0)
    assert len(loose) > len(strict) >= 2
    # every noise peak that passed the loose threshold has k_needed below the strict one
    assert all(p["k_needed"] < 20.0 for p in loose if np.hypot(p["x"] - 3, p["y"] - 3) > 0.5 and np.hypot(p["x"] - 7, p["y"] - 7) > 0.5)


def test_asa_also_locates():
    g, e, n = _grid(noise=0.5)
    asa = derivative_grid(g, 0.1, "asa")
    peaks, _ = find_peaks(asa, g, e, n, 0.1, k=20.0)
    assert len(peaks) == 2
    for sx, sy in ((3.0, 3.0), (7.0, 7.0)):
        assert min(np.hypot(p["x"] - sx, p["y"] - sy) for p in peaks) < 0.3


def test_prominence_is_height_above_the_saddle_to_a_higher_peak():
    """Two hills joined by a ridge: the lower one's prominence is its
    height above the ridge's saddle, the higher one's is its height
    above the grid minimum."""
    e = np.arange(0, 20.1, 0.1)
    X, Y = np.meshgrid(e, e)
    g = 100 * np.exp(-((X - 6) ** 2 + (Y - 10) ** 2) / 4.0) + 60 * np.exp(-((X - 14) ** 2 + (Y - 10) ** 2) / 4.0)
    ridge = 30 * np.exp(-((Y - 10) ** 2) / 2.0) * np.clip(np.sin(np.pi * (X - 7) / 6.0), 0, 1) * ((X > 7) & (X < 13))
    g += ridge
    peaks = [(100, 60), (100, 140)]
    assert g[100, 60] == g.max() and g[100, 140] == g[95:106, 130:150].max()   # the two hills' tops
    prom = _prominences(g, peaks)
    assert abs(prom[0] - (g[100, 60] - g.min())) < 2.0
    saddle = g[100, 70:130].min()
    assert abs(prom[1] - (g[100, 140] - saddle)) < 3.0
    assert prom[1] < g[100, 140] - 20


def test_type_classes_follow_the_paper_ranges():
    assert type_class(1500.0) == "대전차지뢰급"
    assert type_class(300.0) == "대인지뢰급"
    assert type_class(50.0) == "소형 금속/미상"
    assert type_class(float("nan")) == "미상"
