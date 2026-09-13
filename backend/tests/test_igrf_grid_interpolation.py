"""compute_igrf_total_field evaluates ppigrf on a coarse regular grid and
interpolates rather than calling it per-point (see processing/igrf.py) -
ppigrf's own cost scales far worse than linearly with point count, so this
matters a lot for a real multi-hundred-thousand-point survey. Verify the
interpolated result stays numerically indistinguishable from calling
ppigrf directly, across both a single-day and a multi-day survey.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import ppigrf

from app.processing.igrf import compute_igrf_total_field


def _reference_total_field(lat, lon, alt_m, date) -> np.ndarray:
    be, bn, bu = ppigrf.igrf(lon, lat, np.asarray(alt_m) / 1000.0, date)
    return np.sqrt(np.ravel(be) ** 2 + np.ravel(bn) ** 2 + np.ravel(bu) ** 2)


def test_interpolated_field_matches_direct_ppigrf_within_a_fraction_of_a_nt():
    rng = np.random.default_rng(0)
    n = 500
    lat = rng.uniform(34.54, 34.60, n)
    lon = rng.uniform(126.35, 126.42, n)
    alt = rng.uniform(50.0, 150.0, n)
    day = pd.Timestamp("2026-07-15")
    dates = pd.Series([day] * n)

    got = compute_igrf_total_field(lat, lon, alt, dates)
    ref = _reference_total_field(lat, lon, alt, day.to_pydatetime())

    assert np.max(np.abs(got - ref)) < 0.01, "interpolation error should be a small fraction of a nT"


def test_multi_day_survey_still_grouped_and_accurate_per_day():
    rng = np.random.default_rng(1)
    n = 300
    lat = rng.uniform(34.54, 34.60, n)
    lon = rng.uniform(126.35, 126.42, n)
    alt = rng.uniform(50.0, 150.0, n)
    days = pd.to_datetime(rng.choice(["2026-07-15", "2026-07-16", "2026-08-03"], size=n))
    dates = pd.Series(days)

    got = compute_igrf_total_field(lat, lon, alt, dates)
    assert got.shape == (n,)
    assert np.all(np.isfinite(got))

    for day in dates.dt.floor("D").unique():
        mask = (dates.dt.floor("D") == day).to_numpy()
        ref = _reference_total_field(lat[mask], lon[mask], alt[mask], pd.Timestamp(day).to_pydatetime())
        assert np.max(np.abs(got[mask] - ref)) < 0.01


def test_handles_single_point_and_zero_span_without_degenerate_grid_error():
    dates = pd.Series([pd.Timestamp("2026-07-15")])
    got = compute_igrf_total_field(np.array([34.57]), np.array([126.37]), np.array([100.0]), dates)
    assert got.shape == (1,)
    assert np.isfinite(got[0])


def test_some_nan_altitudes_dont_poison_the_whole_days_grid():
    # A loader format that never populates altitude leaves it all-NaN for
    # some rows - np.min/np.max over an array containing any NaN returns
    # NaN, which used to turn that day's whole interpolation grid axis
    # into NaN and made RegularGridInterpolator raise for every point
    # that day, not just the NaN ones. The NaN rows should come back NaN
    # (matching what a direct per-point ppigrf call would do); the finite
    # rows must still be computed normally.
    rng = np.random.default_rng(2)
    n = 50
    lat = rng.uniform(34.54, 34.60, n)
    lon = rng.uniform(126.35, 126.42, n)
    alt = rng.uniform(50.0, 150.0, n)
    alt[::5] = np.nan  # every 5th point has no altitude
    dates = pd.Series([pd.Timestamp("2026-07-15")] * n)

    got = compute_igrf_total_field(lat, lon, alt, dates)

    nan_mask = np.isnan(alt)
    assert np.all(np.isnan(got[nan_mask]))
    assert np.all(np.isfinite(got[~nan_mask]))
    ref = _reference_total_field(lat[~nan_mask], lon[~nan_mask], alt[~nan_mask], pd.Timestamp("2026-07-15").to_pydatetime())
    assert np.max(np.abs(got[~nan_mask] - ref)) < 0.01


def test_all_nan_altitudes_return_all_nan_not_a_crash():
    dates = pd.Series([pd.Timestamp("2026-07-15")] * 5)
    got = compute_igrf_total_field(
        np.full(5, 34.57), np.full(5, 126.37), np.full(5, np.nan), dates
    )
    assert got.shape == (5,)
    assert np.all(np.isnan(got))
