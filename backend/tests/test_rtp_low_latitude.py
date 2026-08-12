"""RTP's wavenumber operator k^2 / (i*(kx*a + ky*b) + |k|*c)^2 has an
unbounded singularity as the inclination approaches the magnetic equator
(c = sin(I) -> 0), so wavenumbers perpendicular to the declination get
divided by a vanishing denominator and the operator's gain diverges -
amplifying noise into the classic low-latitude RTP streaking artefact.
reduction_to_pole caps the operator's magnitude (preserving phase) to
bound that blowup, and store surfaces a warning steering the user to RTE,
which is stable at any latitude.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from app.processing.transforms import reduction_to_equator, reduction_to_pole

CELL = 10.0

# Measured on the synthetic grid below: the capped operator produces an
# output span ~4.8x the input's at 2 deg inclination, the uncapped one
# ~142x. This threshold sits well clear of both, so the assertion tracks
# the cap actually working rather than restating the cap constant (which
# would pass no matter what value the constant held).
_MAX_REASONABLE_SPAN_RATIO = 20.0


def _noisy_grid(seed=0, n=48):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    anomaly = 60.0 * np.exp(-(((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (2 * 5.0**2)))
    return anomaly + rng.normal(0.0, 1.0, size=(n, n))


def test_low_latitude_rtp_output_stays_bounded():
    grid = _noisy_grid()
    input_span = float(np.nanmax(grid) - np.nanmin(grid))

    # 2 deg inclination: essentially at the magnetic equator, the worst
    # case for the RTP operator.
    out = reduction_to_pole(grid, CELL, inclination_deg=2.0, declination_deg=0.0)

    assert np.all(np.isfinite(out))
    output_span = float(np.nanmax(out) - np.nanmin(out))
    # Without the cap this runs away by ~30x more than with it (see the
    # constant's comment above for the measured numbers).
    assert output_span < _MAX_REASONABLE_SPAN_RATIO * input_span


def test_rte_is_far_better_behaved_than_rtp_at_low_latitude():
    grid = _noisy_grid()
    rtp = reduction_to_pole(grid, CELL, inclination_deg=2.0, declination_deg=0.0)
    rte = reduction_to_equator(grid, CELL, inclination_deg=2.0, declination_deg=0.0)

    # The point of the warning the app shows: at this latitude RTE is the
    # transform that actually behaves, which is why users are steered to
    # it rather than being left with a capped-but-still-streaky RTP.
    assert np.nanstd(rte) < np.nanstd(rtp)


def test_high_latitude_rtp_is_untouched_by_the_cap():
    """At Korea's ~53 deg inclination the operator's true gain never
    approaches the cap, so the capped implementation must reproduce the
    uncapped result exactly - the fix is inert everywhere it should be."""
    grid = _noisy_grid()
    out = reduction_to_pole(grid, CELL, inclination_deg=53.0, declination_deg=-8.0)

    # Recompute without the magnitude clamp, mirroring the module's filter.
    from app.processing.transforms import _apply_filter, _direction_coeffs

    a, b, c = _direction_coeffs(53.0, -8.0)

    def uncapped(kx, ky, k_mag):
        denom = 1j * (kx * a + ky * b) + k_mag * c
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = (k_mag**2) / (denom**2)
        theta[(k_mag == 0)] = 1.0
        theta[~np.isfinite(theta)] = 0.0
        return theta

    expected = _apply_filter(grid, CELL, uncapped)
    assert np.allclose(out, expected)


if __name__ == "__main__":
    test_low_latitude_rtp_output_stays_bounded()
    test_rte_is_far_better_behaved_than_rtp_at_low_latitude()
    test_high_latitude_rtp_is_untouched_by_the_cap()
    print("ALL CHECKS PASSED")
