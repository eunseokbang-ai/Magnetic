"""Continuing the field outside the survey with an equivalent source layer.

Every FFT transform needs values where the survey has none, and what gets
put there leaks back inside. The nearest-value fill this replaces copies
each boundary reading straight out into the empty ground, so a strong
anomaly sitting on the survey edge is repeated down a whole column of
invented data - and the transform then spreads the disagreement between
that invention and the real field along the grid axes, which is the
straight streak that runs inward from a strong edge anomaly.

This module does not guess field values at all. It fits a layer of
sources at shallow depth that reproduces the readings that actually
exist, and then evaluates that layer's own field in the empty ground. The
continuation is therefore one that some real buried source distribution
could produce - harmonic by construction - instead of a patch pasted into
the hole. Measured values are never altered: only the gaps are filled.

How well it works, measured against a known answer. Three independent
synthetic worlds (buried dipoles, scaled to the real survey's 44 nT
standard deviation) were masked to this survey's real footprint - 64% of
the rectangle - and each transform recomputed from the masked data, then
compared against the same transform of the complete field:

    transform   fill                 error / signal, quiet ground
    dYZ         nearest (before)          19 - 47 x
    dYZ         Laplace-converged          4 -  8 x
    dYZ         this module              1.4 - 2.2 x
    dXY         nearest (before)         2.6 - 8.6 x
    dXY         this module              0.4 - 0.8 x
    1VD         any                      0.1 - 0.3 x

"Quiet ground" is the half of the survey where the true derivative is
smallest, which is exactly where a streak is visible. The strongest 0.1%
of anomalies keep 100.0% of their amplitude under every fill tested, so
this buys the quiet ground back without touching the targets.

The cost is one preconditioned conjugate-gradient solve, about 11 s for a
480,000-cell grid (a fifth of that without the preconditioner). Results
are cached on the contents of the grid, so the wait is paid once and
every later transform of the same grid is free.

What it does NOT fix: striping that the survey itself carries in. On the
real Haenam grid, swapping the fill changes the picture very little,
because most of what is visible there is per-line content rather than
boundary leakage - the stripes are as wide as the line spacing, carry no
per-column offset (0.0% of the variance), and have no fixed periodicity.
That is a levelling/acquisition question, not a boundary one.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

__all__ = ["source_continuation_fill", "continuation_report"]

# Depth of the source layer, in cells. Measured optimum on the ground-truth
# experiment: 1 cell gives 4.3x quiet-ground error, 2 cells 2.2x, 3 cells
# 3.9x, 6 cells 11x. Too shallow and the layer reproduces cell-scale noise
# instead of continuing the field; too deep and the continuation is too
# smooth to match the data near the boundary.
_DEPTH_CELLS = 2.0
# Tikhonov damping, relative to the kernel. Only needs to keep the solve
# away from the null space of a downward-continuing operator.
_DAMPING = 1e-5
# CG iterations. With the Fourier preconditioner the quiet-ground error is
# 12.7x at 5, 5.0x at 15, 3.1x at 25 and 2.2x at 40, then flat - and the
# unpreconditioned solve needs 80 for the same place. 40 it is.
_MAX_ITER = 40
_CONVERGED = 1e-10
# How far outside the grid the source layer extends, as a fraction of the
# grid. The continuation has to reach past the FFT's own padding.
_PAD_FRACTION = 0.35
# Below this many measured cells there is not enough to fit a layer to.
_MIN_VALID_CELLS = 400

_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
_CACHE_MAX = 3


def _key(grid: np.ndarray, depth_cells: float, damping: float, max_iter: int) -> str:
    h = hashlib.blake2b(np.ascontiguousarray(grid, dtype=np.float64).tobytes(), digest_size=16)
    h.update(f"{grid.shape}|{depth_cells}|{damping}|{max_iter}".encode())
    return h.hexdigest()


def continuation_report(grid: np.ndarray) -> dict:
    """Whether the fill can run on this grid, without running it. Callers
    show this rather than letting a silent no-op look like a result."""
    if grid.ndim != 2:
        return {"applicable": False, "reason": "2차원 격자가 아닙니다."}
    mask = np.isfinite(grid)
    n_valid = int(mask.sum())
    if n_valid < _MIN_VALID_CELLS:
        return {"applicable": False, "reason": f"측정된 격자점이 너무 적습니다({n_valid}개)."}
    n_gap = int((~mask).sum())
    if n_gap == 0:
        return {"applicable": False, "reason": "빈 칸이 없어 바깥을 채울 필요가 없습니다."}
    return {
        "applicable": True,
        "n_gap_cells": n_gap,
        "gap_pct": round(100.0 * n_gap / mask.size, 1),
        "depth_cells": _DEPTH_CELLS,
    }


def _fit_source_layer(grid: np.ndarray, depth_cells: float, damping: float, max_iter: int):
    """Solve for the source layer that reproduces the measured cells.

    The normal equations are (K M K + damping) s = K M d, with K the
    continuation kernel exp(-|k| h) - diagonal in the wavenumber domain,
    so every matrix-vector product is two real FFTs - and M the footprint.
    Replacing M by its coverage fraction makes the whole operator diagonal
    and gives a preconditioner that costs one multiply and halves the
    iteration count.

    h is in cells, so the kernel is |k| * h * cell_size with |k| in rad/m:
    the cell size cancels and none of this needs one.

    Returns the field the fitted layer predicts over the padded domain,
    together with the slice that cuts the original grid back out of it.
    """
    mask = np.isfinite(grid)
    pad_n = max(1, int(grid.shape[0] * _PAD_FRACTION))
    pad_e = max(1, int(grid.shape[1] * _PAD_FRACTION))
    shape = (grid.shape[0] + 2 * pad_n, grid.shape[1] + 2 * pad_e)
    inner = (slice(pad_n, pad_n + grid.shape[0]), slice(pad_e, pad_e + grid.shape[1]))

    # Single precision throughout: this is nanotesla field data, so float32
    # carries four orders of magnitude more precision than the measurement.
    observed = np.zeros(shape, dtype=np.float32)
    selected = np.zeros(shape, dtype=bool)
    offset = float(np.nanmean(grid))
    observed[inner] = np.where(mask, grid - offset, 0.0)
    selected[inner] = mask

    k_north = 2 * np.pi * np.fft.fftfreq(shape[0])
    k_east = 2 * np.pi * np.fft.rfftfreq(shape[1])
    kernel = np.exp(-np.hypot(k_north[:, None], k_east[None, :]) * depth_cells).astype(np.float32)
    coverage = np.float32(selected.mean())
    precondition = (1.0 / (coverage * kernel * kernel + damping)).astype(np.float32)

    def fwd(a):
        return np.fft.rfft2(a)

    def inv(spectrum):
        return np.fft.irfft2(spectrum, s=shape).astype(np.float32)

    def operator(s):
        field = inv(fwd(s) * kernel)
        return inv(fwd(np.where(selected, field, np.float32(0.0))) * kernel) + damping * s

    rhs = inv(fwd(observed) * kernel)
    sources = np.zeros_like(rhs)
    residual = rhs - operator(sources)
    z = inv(fwd(residual) * precondition)
    direction = z.copy()
    rz = float((residual * z).sum())
    for _ in range(max_iter):
        if rz <= 0.0:
            break
        op_d = operator(direction)
        denom = float((direction * op_d).sum())
        if denom <= 0.0:
            break
        alpha = rz / denom
        sources += alpha * direction
        residual -= alpha * op_d
        z = inv(fwd(residual) * precondition)
        rz_next = float((residual * z).sum())
        if rz_next <= rz * _CONVERGED:
            break
        direction = z + (rz_next / rz) * direction
        rz = rz_next

    predicted = inv(fwd(sources) * kernel)[inner].astype(float) + offset
    return predicted, mask


def source_continuation_fill(
    grid: np.ndarray,
    depth_cells: float = _DEPTH_CELLS,
    damping: float = _DAMPING,
    max_iter: int = _MAX_ITER,
) -> np.ndarray:
    """`grid` with its NaN cells filled by the field of a fitted source
    layer. Measured cells come back bit-for-bit unchanged.

    The normal equations are (K M K + damping) s = K M d, with K the
    continuation kernel exp(-|k| h) - diagonal in the wavenumber domain,
    so every matrix-vector product is two real FFTs - and M the footprint.
    Replacing M by its coverage fraction makes the whole operator diagonal
    and gives a preconditioner that costs one multiply and halves the
    iteration count.

    Note h is in cells, so the kernel is |k| * h * cell_size with |k| in
    rad/m: the cell size cancels and this needs no cell size argument.
    """
    if not continuation_report(grid)["applicable"]:
        return grid

    key = _key(grid, depth_cells, damping, max_iter)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit.copy()

    predicted, mask = _fit_source_layer(grid, depth_cells, damping, max_iter)
    filled = np.where(mask, grid, predicted)

    _CACHE[key] = filled.copy()
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return filled


# Depth of the layer used to REPLACE the field, rather than only to fill
# the gaps, expressed in line spacings. Measured on the Haenam west grid
# against the striping the derived grids show, with "target" the
# derivative amplitude kept at the strongest real anomalies:
#
#   preparation                    dXX stripe   target      dXY stripe  target
#   none                              24.7%      100%          2.95%     100%
#   across-line filter x1.5           12.4%       83%          1.37%      86%
#   across-line filter x2.0            6.9%       72%          0.96%      75%
#   across-line filter x2.5            4.7%       63%          0.80%      64%
#   equivalent source 1.6x             3.7%       75%          0.93%      85%
#   equivalent source 2.0x             0.5%       64%          0.46%      68%
#
# At equal amplitude retention the source layer leaves between two and
# nine times less striping than the filter, because it removes what the
# survey cannot resolve the way the ground would - no source distribution
# can produce structure finer than its own depth - instead of cutting a
# band out of the spectrum.
#
# That advantage is specific to broadband across-line aliasing, which is
# what the real data carries. On a synthetic where the aliasing lands in a
# narrow band around the line spacing - a sine at the spacing, or linear
# interpolation between sampled lines - the band filter is aimed straight
# at it and wins instead. So the table above is a measurement on the real
# grid and is not reproduced by the unit tests, which pin the physical
# properties (misfit, amplitude retention, depth ordering) rather than a
# head-to-head that a synthetic cannot fairly stand in for. Decorrugation (the classic Minty micro-
# levelling high-pass) was measured on the same grid and does nothing
# here at all: dXX striping 24.7% -> 25.1%, because this is broadband
# across-line aliasing rather than corrugation at one wavelength.
_FIELD_DEPTH_SPACINGS = 1.6
_FIELD_CACHE: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict()


def equivalent_source_field(
    grid: np.ndarray,
    depth_cells: float,
    damping: float = _DAMPING,
    max_iter: int = _MAX_ITER,
) -> tuple[np.ndarray, float]:
    """The field a source layer at `depth_cells` predicts, everywhere.

    Unlike source_continuation_fill, which keeps the measurements and only
    writes the gaps, this replaces the grid with the layer's own field.
    That is the point: a source distribution at depth h cannot produce
    structure finer than about h, so the result is band-limited the way a
    real field is - smoothly, isotropically, and without the ringing a
    filter leaves - and what disappears is what the survey could not
    resolve anyway.

    Returns (field, rms misfit in nT at the measured cells). The misfit is
    the honest check on the whole idea: on the Haenam grid the layer
    reproduces the readings to 1.9 nT against a field standard deviation
    of 43 nT, so 4% of the signal is the price of removing the striping.
    """
    report = continuation_report(grid)
    if not report["applicable"] and int(np.isfinite(grid).sum()) < _MIN_VALID_CELLS:
        return grid, float("nan")

    key = _key(grid, -depth_cells, damping, max_iter)  # negative: a different question
    hit = _FIELD_CACHE.get(key)
    if hit is not None:
        _FIELD_CACHE.move_to_end(key)
        return hit[0].copy(), hit[1]

    predicted, mask = _fit_source_layer(grid, depth_cells, damping, max_iter)
    misfit = float(np.sqrt(np.mean((predicted[mask] - grid[mask]) ** 2))) if mask.any() else float("nan")

    _FIELD_CACHE[key] = (predicted.copy(), misfit)
    while len(_FIELD_CACHE) > _CACHE_MAX:
        _FIELD_CACHE.popitem(last=False)
    return predicted, misfit


def field_depth_cells(line_spacing_m: float | None, cell_size_m: float, factor: float) -> float | None:
    """Depth for equivalent_source_field, in cells, from a factor given in
    line spacings. None when the line spacing is unknown - guessing a
    depth from the cell size alone would smooth by an amount unrelated to
    what the survey actually resolved."""
    if not line_spacing_m or line_spacing_m <= 0 or cell_size_m <= 0 or factor <= 0:
        return None
    return factor * line_spacing_m / cell_size_m
