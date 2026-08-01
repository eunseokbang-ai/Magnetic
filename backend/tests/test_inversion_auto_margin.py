"""Regression test: suggest_mesh_params must always propose a cell size
whose actual gridded shape respects n_obs_cap, even for near-square
survey footprints where the "+1 per axis" discrete grid rounding effect
is large relative to the analytic area/cell^2 estimate.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from app.processing.inversion_auto import _grid_shape, suggest_mesh_params


def _check(width, height, n_obs_cap=3500, n_active_cap=90000, line_spacing_m=99.0, seed=0):
    rng = np.random.default_rng(seed)
    n = 5000
    x = rng.uniform(0, width, n)
    y = rng.uniform(0, height, n)
    values = rng.normal(0, 50, n)
    sugg = suggest_mesh_params(x, y, values, line_spacing_m, n_obs_cap, n_active_cap)
    nx, ny = _grid_shape(width, height, sugg["obs_cell_size_m"])
    assert nx * ny <= n_obs_cap, f"{width}x{height}: nx*ny={nx*ny} exceeds cap {n_obs_cap}"
    return sugg


def test_near_square_survey_respects_obs_cap():
    _check(6000.0, 5800.0)


def test_various_aspect_ratios_respect_obs_cap():
    for width, height in [(500, 500), (2000, 300), (300, 2000), (10000, 9500), (1564, 4447)]:
        _check(width, height)


def test_elongated_survey_still_uses_half_line_spacing_floor():
    sugg = _check(1564.0, 4447.0, line_spacing_m=98.97)
    assert sugg["obs_cell_size_m"] >= 98.97 / 2.0 - 1e-6
