"""How far a grid cell's trustworthiness carries into what is computed
from it.

`gridding.grid_confidence` scores each cell of the anomaly grid from 0 to
1 by how well the readings support it (distance to the nearest reading
and to the nearest line). Until 2026-11 the score stopped there: an RTP
or derivative grid drawn from the same cells showed no such layer, and an
Euler solution in a gap between lines looked exactly like one on a line.

A derived cell is not supported by one base cell but by all the base
cells its operator reads, in proportion to how strongly it reads them.
For a linear, shift-invariant transform with impulse response h that is

    c_derived(i) = sum_j h(i - j)^2 c(j) / sum_j h(j)^2

- the base confidence averaged under the square of the operator's
footprint. The square, because that is how errors travel through a linear
combination: the variance of sum h_j e_j, for independent errors, is
sum h_j^2 var(e_j), so each input matters in proportion to the square of
its weight. (Weighting by |h| instead was tried first and is wrong for
exactly the operators this is for: a spectral derivative's impulse
response falls off as 1/n with alternating sign, whose magnitudes do not
sum - measured on a 128-cell row, 90 % of sum |h| lay within 36 cells,
which would smear a gap across the whole survey. Under h^2 the same
operator's 90 % radius is a couple of cells, as it should be.)
Cells with no value at all (outside the survey) count as confidence 0:
whatever the transform read there was padding.

What follows from the rule, without being built in: a first derivative,
whose footprint is a few cells wide, keeps the base layer nearly as it
is; a reduction to the pole or an upward continuation, whose footprint
decays slowly, smears a gap's low confidence well into the supported
cells around it - which is what those transforms do to the gap's
interpolated values too.

The footprint is measured, not derived: the transform is applied to a
unit impulse on a grid of the survey's own shape and cell size, so
whatever the implementation does (its wavenumber response, its low-pass
at the line spacing) is what gets propagated. Products that are not
linear in the field - analytic signal, total horizontal derivative, tilt,
theta - are built from the field's derivatives, and take the smallest of
those derivatives' confidences: a ratio or a root of three numbers is no
better known than the worst-known of them.

This is a support measure, not an error bar in nT. It says where a value
rests on readings and where on interpolation, in the same 0-1 terms as
the base layer; it does not say how wrong the value is.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.signal import fftconvolve

# Footprint weights below this share of the largest are dropped: they are
# the transform's numerical floor spread over the whole grid.
_FOOTPRINT_FLOOR = 1e-6


def footprint(transform: Callable[[np.ndarray], np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Squared impulse response of `transform` on a grid of `shape`,
    centred, normalised to sum 1."""
    ny, nx = shape
    impulse = np.zeros(shape, dtype=float)
    impulse[ny // 2, nx // 2] = 1.0
    response = np.asarray(transform(impulse), dtype=float) ** 2
    response[~np.isfinite(response)] = 0.0
    peak = float(response.max())
    if peak <= 0.0:
        out = np.zeros(shape, dtype=float)
        out[ny // 2, nx // 2] = 1.0            # a transform that returns nothing reads only its own cell
        return out
    response[response < _FOOTPRINT_FLOOR * peak] = 0.0
    return response / response.sum()


def propagate(confidence: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """`confidence` averaged under `kernel` (a centred footprint summing
    to 1). NaN cells count as 0 and stay NaN in the result."""
    covered = np.isfinite(confidence)
    filled = np.where(covered, confidence, 0.0)
    # Convolution, not correlation: c_d(i) = sum_j h(i - j)^2 c(j), and h is
    # not symmetric for every transform (a reduction to the pole leans
    # along the declination). "same" centres the result on the kernel's
    # cell (n - 1) // 2, the footprint's centre is n // 2: the same cell
    # when n is odd, so an even-sized axis gets one row of zeros appended.
    ny, nx = kernel.shape
    if ny % 2 == 0 or nx % 2 == 0:
        kernel = np.pad(kernel, ((0, 1 - ny % 2), (0, 1 - nx % 2)))
    out = fftconvolve(filled, kernel, mode="same")
    out = np.clip(out, 0.0, 1.0)
    return np.where(covered, out, np.nan)


def derived_confidence(confidence: np.ndarray, components: Sequence[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Confidence of a product built from `components` (each a linear
    transform of the grid): the smallest of their propagated confidences.
    One component = a linear product; several = a non-linear one made of
    them."""
    if not components:
        return confidence.copy()
    out = None
    for transform in components:
        c = propagate(confidence, footprint(transform, confidence.shape))
        out = c if out is None else np.fmin(out, c)
    return out


def window_mean(confidence: np.ndarray, window_cells: int) -> np.ndarray:
    """Mean confidence over a square moving window - for estimates made
    from all the cells of a window (Euler deconvolution). NaN counts as 0."""
    n = max(1, int(window_cells))
    kernel = np.zeros(confidence.shape, dtype=float)
    ny, nx = confidence.shape
    y0, x0 = ny // 2 - n // 2, nx // 2 - n // 2
    kernel[max(y0, 0):y0 + n, max(x0, 0):x0 + n] = 1.0
    kernel /= kernel.sum()
    return propagate(confidence, kernel)
