"""The synthetic the height normalisation is measured on: a terrain-
following flight over a hill, with a known field to compare against.

Two shallow induced dipoles (40 m and 60 m under the ground) and a
regional "geology" - a deep layer of dipoles whose moments vary with a
1.4 km wavelength - under a ground surface that undulates by +-30 m.
Lines 50 m apart run north-south, sampled every 2 m, flown 50 m above
the ground, so the sensor altitude varies by 60 m across the survey.

Because everything is dipoles, the true field at any altitude comes from
the same kernel the normalisation uses (dipole_layer_field), and "what
the readings would have been at z_ref" is exact.
"""
from __future__ import annotations

import numpy as np

from app.processing.height_normalization import dipole_layer_field

INC, DEC = 53.0, -8.0
LINE_SPACING = 50.0
AGL = 50.0
RELIEF = 30.0


def ground(x, y):
    return RELIEF * np.sin(2 * np.pi * x / 700.0) * np.cos(2 * np.pi * y / 900.0)


def true_sources():
    """(sources_xyz, moments) of the whole model."""
    shallow_xy = [(430.0, 520.0), (700.0, 300.0)]
    shallow_depth = [40.0, 60.0]
    shallow_m = [6e4, 1.2e5]
    src = [[sx, sy, ground(sx, sy) - dz] for (sx, sy), dz in zip(shallow_xy, shallow_depth)]
    mom = list(shallow_m)
    # Regional geology: a deep sheet, 100 m grid over 3 x 3 km, moments
    # varying with a 1.4 km wavelength.
    gx = np.arange(-1000.0, 2000.0 + 1, 100.0)
    gy = np.arange(-1000.0, 2000.0 + 1, 100.0)
    GX, GY = np.meshgrid(gx, gy)
    deep_z = -300.0
    amp = 4e6
    for xx, yy in zip(GX.ravel(), GY.ravel()):
        src.append([xx, yy, deep_z])
        mom.append(amp * np.sin(2 * np.pi * xx / 1400.0) * np.cos(2 * np.pi * yy / 1400.0 + 0.7))
    return np.asarray(src, dtype=float), np.asarray(mom, dtype=float)


def true_field(x, y, z):
    s, m = true_sources()
    return dipole_layer_field(x, y, z, s, m, INC, DEC)


def hill_survey(noise_nt: float = 0.3, relief: float | None = None, seed: int = 0, extent_m: float = 1000.0,
                sample_m: float = 2.0):
    """x, y, z, line_id, anomaly of a survey flown AGL m over the ground.
    `relief` scales the terrain (None = the module default of +-30 m)."""
    rng = np.random.default_rng(seed)
    xs = np.arange(0.0, extent_m + 1e-9, LINE_SPACING)
    ys = np.arange(0.0, extent_m + 1e-9, sample_m)
    x = np.repeat(xs, len(ys))
    y = np.tile(ys, len(xs))
    line_id = np.repeat(np.arange(len(xs)), len(ys))
    g = ground(x, y)
    if relief is not None:
        g = g * (relief / RELIEF)
    z = g + AGL
    anomaly = true_field(x, y, z) + noise_nt * rng.standard_normal(len(x))
    return x, y, z, line_id, anomaly
