"""Registers custom matplotlib colormaps not built into matplotlib itself,
so `matplotlib.colormaps[name]` works for them the same as any built-in name.
Call register_custom_colormaps() once at app startup.
"""
from __future__ import annotations

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

# Approximates the classic Geosoft Oasis Montaj default grid color table:
# dark blue -> blue -> cyan -> green -> yellow -> orange -> red -> magenta
# -> pink, very common for magnetic/gravity anomaly maps in the industry.
_GEOSOFT_RAINBOW_STOPS = [
    (0.00, (10, 10, 120)),
    (0.10, (20, 60, 200)),
    (0.20, (0, 160, 220)),
    (0.30, (0, 210, 190)),
    (0.40, (0, 200, 90)),
    (0.50, (140, 220, 40)),
    (0.58, (255, 255, 0)),
    (0.66, (255, 180, 0)),
    (0.74, (255, 90, 0)),
    (0.82, (230, 20, 20)),
    (0.90, (200, 20, 160)),
    (1.00, (255, 200, 235)),
]


def _make_geosoft_rainbow() -> LinearSegmentedColormap:
    colors = [(pos, tuple(c / 255.0 for c in rgb)) for pos, rgb in _GEOSOFT_RAINBOW_STOPS]
    return LinearSegmentedColormap.from_list("geosoft_rainbow", colors, N=256)


def register_custom_colormaps() -> None:
    if "geosoft_rainbow" not in matplotlib.colormaps:
        matplotlib.colormaps.register(name="geosoft_rainbow", cmap=_make_geosoft_rainbow())
