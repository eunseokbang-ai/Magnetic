"""grid_to_png_overlay must report legend_ticks - the data value whose
color sits at each of 5 equally spaced color-bar positions - because the
frontend legend draws a uniform gradient and labels positions along it.
With only vmin/vmax labeled, the legend implicitly claims the color-value
mapping between them is linear; for the equalize (empirical-rank/CDF) and
normal (Gaussian-CDF) stretches that's wrong - the bar's midpoint color
corresponds to the data's median / mean, not (vmin+vmax)/2 - so a reader
interpolating a value from the legend misreads the map. The ticks must
follow each stretch's own mapping (matching render.py's _EqualizeNorm /
_NormalNorm definitions), not a linspace."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from app.processing.render import grid_to_png_overlay, robust_center_scale

UTM_EPSG = 32652


def _skewed_grid():
    # Long-tailed values (squared uniform) so median != midpoint - the
    # exact situation where a linear legend misleads under equalize.
    rng = np.random.default_rng(42)
    values = (rng.uniform(0.0, 1.0, size=(20, 25)) ** 4) * 100.0
    values[0, 0] = np.nan  # keep a NaN cell in play, as real grids have
    easting = np.arange(25) * 10.0 + 300000.0
    northing = np.arange(20) * 10.0 + 3900000.0
    return values, easting, northing


def test_linear_ticks_are_evenly_spaced_between_vmin_vmax():
    values, easting, northing = _skewed_grid()
    overlay = grid_to_png_overlay(values, easting, northing, UTM_EPSG, stretch="linear")

    ticks = overlay["legend_ticks"]
    assert len(ticks) == 5
    expected = np.linspace(overlay["vmin"], overlay["vmax"], 5)
    assert np.allclose(ticks, expected)


def test_equalize_ticks_are_quantiles_not_linspace():
    values, easting, northing = _skewed_grid()
    overlay = grid_to_png_overlay(values, easting, northing, UTM_EPSG, stretch="equalize")

    ticks = np.array(overlay["legend_ticks"])
    finite = values[np.isfinite(values)]
    expected = np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(ticks, expected)
    # the whole point: for this skewed data the median (color-bar midpoint)
    # sits far from the linear midpoint a two-end legend would imply
    linear_mid = (overlay["vmin"] + overlay["vmax"]) / 2.0
    assert abs(ticks[2] - linear_mid) > 0.2 * (overlay["vmax"] - overlay["vmin"])


def test_normal_ticks_follow_gaussian_cdf_inverse():
    values, easting, northing = _skewed_grid()
    overlay = grid_to_png_overlay(values, easting, northing, UTM_EPSG, stretch="normal")

    ticks = np.array(overlay["legend_ticks"])
    center, scale = robust_center_scale(values[np.isfinite(values)])
    # ends land exactly on the reported +-3 sigma display range, the
    # midpoint on the center, interior ticks at center +- 0.6745 sigma
    # (Phi^-1(0.75))
    assert np.isclose(ticks[0], overlay["vmin"])
    assert np.isclose(ticks[-1], overlay["vmax"])
    assert np.isclose(ticks[2], center)
    assert np.isclose(ticks[3] - center, 0.6744897501960817 * scale, rtol=1e-6)


if __name__ == "__main__":
    test_linear_ticks_are_evenly_spaced_between_vmin_vmax()
    test_equalize_ticks_are_quantiles_not_linspace()
    test_normal_ticks_follow_gaussian_cdf_inverse()
    print("ALL CHECKS PASSED")
