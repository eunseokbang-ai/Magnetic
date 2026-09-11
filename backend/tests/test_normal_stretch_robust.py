"""The "normal" stretch fits a Gaussian CDF to the grid's background level
and spread. Using the mean/std for that pair defeats the stretch's own
premise - "background noise plus a few real anomalies" is exactly the case
where a handful of strong anomalies inflates the std, widening the +-3
sigma display range and flattening the color contrast across the
background the map is meant to show. The median/MAD are the robust
estimators of the same two quantities, so they agree for genuinely
Gaussian data and only differ where the non-robust pair was being skewed.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from app.processing.render import grid_to_png_overlay, robust_center_scale

UTM_EPSG = 32652


def _grid_axes(n):
    return np.arange(n) * 10.0 + 300000.0, np.arange(n) * 10.0 + 3900000.0


def test_robust_estimators_match_mean_std_for_clean_gaussian_data():
    rng = np.random.default_rng(7)
    values = rng.normal(100.0, 5.0, size=20000)

    center, scale = robust_center_scale(values)

    assert np.isclose(center, np.mean(values), rtol=0.05)
    assert np.isclose(scale, np.std(values), rtol=0.05)


def test_a_few_strong_anomalies_no_longer_blow_out_the_display_range():
    rng = np.random.default_rng(11)
    n = 60
    values = rng.normal(0.0, 1.0, size=(n, n))
    # 1% of cells carry a huge real anomaly - the situation the stretch is
    # meant to handle, and the one that wrecks a mean/std fit.
    values.ravel()[: int(0.01 * values.size)] = 500.0
    easting, northing = _grid_axes(n)

    overlay = grid_to_png_overlay(values, easting, northing, UTM_EPSG, stretch="normal")
    robust_span = overlay["vmax"] - overlay["vmin"]

    non_robust_span = 6.0 * float(np.std(values))
    background_span = 6.0 * 1.0  # the +-3 sigma range of the real background

    # The robust range tracks the background it's supposed to (within a
    # factor of ~2), while the mean/std range is dragged an order of
    # magnitude wider by the 1% of anomalous cells.
    assert robust_span < 2.0 * background_span
    assert non_robust_span > 10.0 * robust_span


def test_zero_mad_falls_back_instead_of_collapsing_the_range():
    """More than half the cells sharing one exact value (a mostly-flat
    mask, a heavily quantized grid) drives the MAD to 0; without a
    fallback the whole color range would collapse onto that single
    value."""
    values = np.zeros(400)
    values[:50] = np.linspace(1.0, 50.0, 50)

    center, scale = robust_center_scale(values)

    assert center == 0.0
    assert scale > 0
    assert np.isclose(scale, np.std(values))


if __name__ == "__main__":
    test_robust_estimators_match_mean_std_for_clean_gaussian_data()
    test_a_few_strong_anomalies_no_longer_blow_out_the_display_range()
    test_zero_mad_falls_back_instead_of_collapsing_the_range()
    print("ALL CHECKS PASSED")
