"""FFT-based potential-field grid transforms: reduction to pole (RTP),
reduction to equator (RTE), vertical derivative (1VD) and analytic signal
(AS), following the standard wavenumber-domain filters (Blakely, 1995,
"Potential Theory in Gravity and Magnetic Applications", ch. 12).

Convention: x = northing, y = easting, z positive down. All filters take
a 2D grid (rows=northing, cols=easting) with NaN outside data coverage.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

_TAPER_FRACTION = 0.25

# Relaxation passes used to smooth the fill inside a NaN hole - see
# _fill_gaps. Measured on a synthetic grid with a ragged hole, as
# cell-scale striping (a grid with no holes gives 1.1%): filling with the
# grid mean left 21.2%, nearest-value fill alone 7.8%, and relaxation on
# top of it 3.1% at 8 passes, 2.1% at 16, 1.8% at 24 and 1.6% at 32. The
# curve is flat past ~24, and the extra passes are cheap next to the
# distance transform that precedes them (1.2s -> 1.5s on a 3M-cell grid),
# so 24 is the floor. Scaled by how deep the hole is, since relaxation
# carries information about one cell per pass.
_GAP_FILL_PASSES_PER_CELL = 6.0
_GAP_FILL_MAX_PASSES = 48
_GAP_FILL_MIN_PASSES = 24

# Derivative pre-smoothing width, as a fraction of the line spacing - see
# limit_to_line_spacing_resolution.
_RESOLUTION_SMOOTH_FACTOR = 0.4


def _wavenumbers(n_north: int, n_east: int, d_north: float, d_east: float):
    kx = 2 * np.pi * np.fft.fftfreq(n_north, d=d_north)
    ky = 2 * np.pi * np.fft.fftfreq(n_east, d=d_east)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.hypot(kx_grid, ky_grid)
    return kx_grid, ky_grid, k_mag


def _fill_gaps(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill the NaN cells of `grid` so that the result runs smoothly out of
    the surrounding data, instead of jumping to some unrelated level.

    This matters far more than it looks. Every one of these filters
    differentiates in the wavenumber domain, and a derivative of a step is
    a spike. Filling gaps with a constant - the grid mean, as this used to
    - puts a step at the edge of every masked area as tall as the distance
    from the local field to that mean, which on a real survey is hundreds
    of nT. FFT differentiation turns each into Gibbs ringing, and because
    a distance mask leaves cell-scale ragged edges around the flown lines,
    the ringing is cell-scale too: the fine hatching that shows up in AS
    and the second derivatives near masked ground.

    Two stages. Nearest-value fill first, which alone removes the step -
    the filled cell now equals the real value it borders. Then a few
    Laplace relaxation passes over the filled cells only, which smooths
    the kink in the gradient that nearest fill leaves behind, converging
    toward the harmonic (no new extrema) continuation of the surrounding
    field. Both stages leave real data untouched; only the gaps move.

    On the same synthetic grid the striping goes 21.2% -> 1.8% (a grid
    with no holes at all gives 1.1%) and the error against the known
    answer falls 1.43 -> 0.013. Costs ~1.5s on the largest grid the
    gridder will produce, most of which is the distance transform.
    """
    if not mask.any():
        return grid
    distance, indices = ndimage.distance_transform_edt(mask, return_distances=True, return_indices=True)
    filled = grid[tuple(indices)]

    depth = float(distance[mask].max())
    passes = int(np.clip(round(_GAP_FILL_PASSES_PER_CELL * depth), _GAP_FILL_MIN_PASSES, _GAP_FILL_MAX_PASSES))
    kernel = np.array([[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]])
    for _ in range(passes):
        filled = np.where(mask, ndimage.convolve(filled, kernel, mode="nearest"), filled)
    return filled


def limit_to_line_spacing_resolution(
    grid: np.ndarray,
    cell_size_m: float,
    line_spacing_m: float | None,
    factor: float = _RESOLUTION_SMOOTH_FACTOR,
) -> np.ndarray:
    """Low-pass the grid to the resolution it actually has across the
    flight lines, before anything differentiates it.

    A survey samples densely along each line and not at all between them,
    so nothing narrower than the line spacing is measured in the
    across-line direction - whatever the grid holds at that scale was
    invented by the interpolator. With the default "raw cell" (nearest)
    gridding that invention is severe: 80% of neighbouring cells across
    the lines are *exactly equal*, because each takes its value from the
    same nearest sounding, so the grid is a field of flat blocks with
    steps between them. Differentiating that measures the blocks, not the
    ground. Measured against a known field on a 50 m-line survey gridded
    at 10 m, the analytic signal came out 101% wrong and dXY 218% wrong.

    Smoothing to the real resolution first fixes it: at factor 0.4 the
    same analytic signal lands 5.2% from the truth and dXY 14.9% - better
    than re-gridding with linear interpolation (8.8% / 153%) and without
    the cost of doing so. Larger factors start erasing real signal (0.6
    takes AS back up to 7.3%).

    No-op when the line spacing is unknown, or when the cells are already
    coarse enough that the grid holds nothing finer than its resolution.
    """
    if not line_spacing_m or line_spacing_m <= 0 or cell_size_m <= 0:
        return grid
    sigma_cells = factor * line_spacing_m / cell_size_m
    if sigma_cells < 0.5:
        return grid

    # NaN-aware Gaussian: smoothing zeros through a gap would pull the
    # values beside it toward zero. Smooth the data and the validity mask
    # separately and divide, so each cell is the weighted mean of the real
    # values near it.
    valid = np.isfinite(grid)
    if not valid.any():
        return grid
    values = np.where(valid, grid, 0.0)
    weights = valid.astype(float)
    smoothed = ndimage.gaussian_filter(values, sigma_cells, mode="nearest")
    norm = ndimage.gaussian_filter(weights, sigma_cells, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(norm > 1e-6, smoothed / norm, np.nan)
    return np.where(valid, out, np.nan)


def _pad_and_fill(grid: np.ndarray):
    """Fill NaNs (see _fill_gaps) and mirror-pad the edges (tapered) to
    reduce FFT wraparound artifacts. Returns (padded, mask, pad_widths)."""
    mask = np.isnan(grid)
    fill_value = np.nanmean(grid) if not np.all(mask) else 0.0
    filled = _fill_gaps(grid, mask) if not np.all(mask) else np.full_like(grid, fill_value)

    pad_n = max(1, int(grid.shape[0] * _TAPER_FRACTION))
    pad_e = max(1, int(grid.shape[1] * _TAPER_FRACTION))
    padded = np.pad(filled, ((pad_n, pad_n), (pad_e, pad_e)), mode="reflect")

    taper_n = np.hanning(2 * pad_n) if pad_n > 1 else np.array([1.0, 1.0])
    taper_e = np.hanning(2 * pad_e) if pad_e > 1 else np.array([1.0, 1.0])
    win_n = np.ones(padded.shape[0])
    win_n[:pad_n] = taper_n[:pad_n]
    win_n[-pad_n:] = taper_n[pad_n:]
    win_e = np.ones(padded.shape[1])
    win_e[:pad_e] = taper_e[:pad_e]
    win_e[-pad_e:] = taper_e[pad_e:]
    window = np.outer(win_n, win_e)
    padded = fill_value + (padded - fill_value) * window

    return padded, mask, (pad_n, pad_e)


def _unpad_and_mask(padded: np.ndarray, mask: np.ndarray, pad_widths: tuple[int, int]) -> np.ndarray:
    pad_n, pad_e = pad_widths
    cropped = padded[pad_n : padded.shape[0] - pad_n, pad_e : padded.shape[1] - pad_e]
    return np.where(mask, np.nan, cropped)


def _apply_filter(grid: np.ndarray, cell_size_m: float, filter_fn) -> np.ndarray:
    padded, mask, pad_widths = _pad_and_fill(grid)
    kx, ky, k_mag = _wavenumbers(padded.shape[0], padded.shape[1], cell_size_m, cell_size_m)
    spectrum = np.fft.fft2(padded)
    filt = filter_fn(kx, ky, k_mag)
    result = np.real(np.fft.ifft2(spectrum * filt))
    return _unpad_and_mask(result, mask, pad_widths)


def _direction_coeffs(inclination_deg: float, declination_deg: float):
    i = np.radians(inclination_deg)
    d = np.radians(declination_deg)
    return np.cos(i) * np.cos(d), np.cos(i) * np.sin(d), np.sin(i)


# Largest amplitude gain the RTP operator is allowed to apply to any single
# wavenumber - see reduction_to_pole. 20x is well above anything the filter
# legitimately needs at mid/high magnetic latitudes (at Korea's ~53 deg
# inclination the operator's true gain peaks around 2-3x, so this never
# binds there), while still bounding the otherwise-unbounded blowup along
# the strike direction at low latitudes.
_RTP_MAX_AMPLIFICATION = 20.0

# Below this |inclination| the RTP operator is considered numerically
# unreliable and RTE is the standard alternative - the usual textbook
# threshold for "low magnetic latitude" RTP instability.
RTP_LOW_LATITUDE_INCLINATION_DEG = 30.0


def reduction_to_pole(
    grid: np.ndarray,
    cell_size_m: float,
    inclination_deg: float,
    declination_deg: float,
) -> np.ndarray:
    """Transform an induced-magnetization anomaly to its pole-reduced
    equivalent (as if measured with a vertical, 90 deg inclination field).

    The raw operator k^2 / (i*(kx*a + ky*b) + |k|*c)^2 has an unbounded
    singularity as the inclination approaches the magnetic equator: c =
    sin(I) shrinks toward 0, so for wavenumbers perpendicular to the
    declination the denominator approaches 0 and the gain diverges,
    amplifying noise along the strike direction into the classic RTP
    low-latitude streaking artefact. The operator's magnitude is therefore
    capped at _RTP_MAX_AMPLIFICATION (phase preserved, so only the runaway
    amplitudes are affected and nothing changes at mid/high latitudes
    where the cap never binds). This bounds the artefact but does not
    remove it - at low magnetic latitudes reduction_to_equator is the
    appropriate transform instead, and callers should steer users there
    (see RTP_LOW_LATITUDE_INCLINATION_DEG)."""
    a, b, c = _direction_coeffs(inclination_deg, declination_deg)

    def filt(kx, ky, k_mag):
        denom = 1j * (kx * a + ky * b) + k_mag * c
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = (k_mag**2) / (denom**2)
        theta[(k_mag == 0)] = 1.0
        theta[~np.isfinite(theta)] = 0.0
        # Clamp magnitude, keep phase: theta * min(1, cap/|theta|).
        magnitude = np.abs(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(magnitude > _RTP_MAX_AMPLIFICATION, _RTP_MAX_AMPLIFICATION / magnitude, 1.0)
        return theta * scale

    return _apply_filter(grid, cell_size_m, filt)


def reduction_to_equator(
    grid: np.ndarray,
    cell_size_m: float,
    inclination_deg: float,
    declination_deg: float,
) -> np.ndarray:
    """Transform an induced-magnetization anomaly to its equator-reduced
    equivalent (flattened to 0 deg inclination, same declination) - an
    alternative to RTP that stays numerically stable at low magnetic
    latitudes."""
    a, b, c = _direction_coeffs(inclination_deg, declination_deg)
    d = np.radians(declination_deg)
    a_t, b_t = np.cos(d), np.sin(d)  # target inclination = 0

    def filt(kx, ky, k_mag):
        numerator = (1j * (kx * a_t + ky * b_t)) ** 2
        denom = 1j * (kx * a + ky * b) + k_mag * c
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = numerator / (denom**2)
        theta[(k_mag == 0)] = 1.0
        theta[~np.isfinite(theta)] = 0.0
        return theta

    return _apply_filter(grid, cell_size_m, filt)


def vertical_derivative(grid: np.ndarray, cell_size_m: float, order: int = 1) -> np.ndarray:
    """n-th order first vertical derivative (1VD when order=1) via |k|^order."""

    def filt(kx, ky, k_mag):
        return k_mag**order

    return _apply_filter(grid, cell_size_m, filt)


def _dxyz(grid: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared (d_easting, d_northing, d_vertical) real-space derivative
    grids from a single padded FFT pass - the common basis behind
    analytic_signal, total_horizontal_derivative, tilt_angle and
    theta_map, so they all agree exactly on the same underlying gradient
    rather than being recomputed (and re-padded) independently."""
    padded, mask, pad_widths = _pad_and_fill(grid)
    kx, ky, k_mag = _wavenumbers(padded.shape[0], padded.shape[1], cell_size_m, cell_size_m)
    spectrum = np.fft.fft2(padded)

    # module convention: kx ~ northing (rows), ky ~ easting (cols) - see
    # module docstring and processing/microlevel.py's note on this axis
    # naming.
    d_east = np.real(np.fft.ifft2(spectrum * (1j * ky)))
    d_north = np.real(np.fft.ifft2(spectrum * (1j * kx)))
    d_vert = np.real(np.fft.ifft2(spectrum * k_mag))

    return (
        _unpad_and_mask(d_east, mask, pad_widths),
        _unpad_and_mask(d_north, mask, pad_widths),
        _unpad_and_mask(d_vert, mask, pad_widths),
    )


def analytic_signal(grid: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Amplitude of the analytic signal: sqrt(dF/dx^2 + dF/dy^2 + dF/dz^2)."""
    d_east, d_north, d_vert = _dxyz(grid, cell_size_m)
    return np.sqrt(d_east**2 + d_north**2 + d_vert**2)


def total_horizontal_derivative(grid: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Total horizontal derivative (THDR): sqrt(dF/dx^2 + dF/dy^2) - the
    in-plane-only counterpart of the analytic signal above (no dz term).
    Peaks over the edges of a source body regardless of magnetization
    direction, so it's a common companion/alternative to 1VD for outlining
    contacts and boundaries."""
    d_east, d_north, _d_vert = _dxyz(grid, cell_size_m)
    return np.hypot(d_east, d_north)


def tilt_angle(grid: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Tilt angle (Miller & Singh, 1994): arctan(1VD / THDR), in degrees,
    range -90..+90. Crosses zero directly over a source's edge regardless
    of its amplitude, so both shallow/strong and deep/weak sources produce
    a usable zero-contour on the same map - a common complement to the
    analytic signal when source strength varies a lot across a survey
    (e.g. compact near-surface targets alongside broader geology)."""
    d_east, d_north, d_vert = _dxyz(grid, cell_size_m)
    thdr = np.hypot(d_east, d_north)
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arctan2(d_vert, thdr))


def theta_map(grid: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Theta map (Wijns et al., 2005): arccos(THDR / analytic-signal
    amplitude), in degrees, range 0..90. Another amplitude-independent
    edge detector, normalizing the horizontal derivative by the full
    gradient magnitude instead of just the vertical derivative (as tilt
    angle does) - tends to sharpen edges slightly differently, so the two
    are commonly viewed side by side."""
    d_east, d_north, d_vert = _dxyz(grid, cell_size_m)
    thdr = np.hypot(d_east, d_north)
    asa = np.sqrt(d_east**2 + d_north**2 + d_vert**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.clip(thdr / asa, -1.0, 1.0)
        theta = np.degrees(np.arccos(ratio))
    return np.where(asa > 0, theta, np.nan)


def derivative_easting(grid: np.ndarray, cell_size_m: float) -> np.ndarray:
    """1st East-West (easting) horizontal derivative, dF/d(easting)."""

    def filt(kx, ky, k_mag):
        return 1j * ky

    return _apply_filter(grid, cell_size_m, filt)


def derivative_northing(grid: np.ndarray, cell_size_m: float) -> np.ndarray:
    """1st North-South (northing) horizontal derivative, dF/d(northing)."""

    def filt(kx, ky, k_mag):
        return 1j * kx

    return _apply_filter(grid, cell_size_m, filt)


def second_derivative_ee(grid: np.ndarray, cell_size_m: float) -> np.ndarray:
    """2nd East-West derivative, d^2F/d(easting)^2."""

    def filt(kx, ky, k_mag):
        return -(ky**2)

    return _apply_filter(grid, cell_size_m, filt)


def second_derivative_nn(grid: np.ndarray, cell_size_m: float) -> np.ndarray:
    """2nd North-South derivative, d^2F/d(northing)^2."""

    def filt(kx, ky, k_mag):
        return -(kx**2)

    return _apply_filter(grid, cell_size_m, filt)


def second_derivative_en(grid: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Mixed East-West/North-South 2nd derivative, d^2F/d(easting)d(northing)."""

    def filt(kx, ky, k_mag):
        return -(kx * ky)

    return _apply_filter(grid, cell_size_m, filt)


def second_derivative_ez(grid: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Mixed East-West/vertical 2nd derivative, d^2F/d(easting)dz."""

    def filt(kx, ky, k_mag):
        return 1j * ky * k_mag

    return _apply_filter(grid, cell_size_m, filt)


def second_derivative_nz(grid: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Mixed North-South/vertical 2nd derivative, d^2F/d(northing)dz."""

    def filt(kx, ky, k_mag):
        return 1j * kx * k_mag

    return _apply_filter(grid, cell_size_m, filt)


def upward_continuation(grid: np.ndarray, cell_size_m: float, height_m: float) -> np.ndarray:
    """Analytically continue the field upward by height_m, simulating a
    survey flown that much higher - short-wavelength (near-surface, high
    wavenumber) signal is attenuated faster than long-wavelength (deep/
    regional) signal, via the standard exp(-|k|*h) filter. Used to smooth
    out shallow noise and isolate regional trends, or as a sanity check
    of how a feature's signal degrades with altitude. height_m must be
    positive - upward continuation is a smoothing (not invertible in
    practice) operation, unlike downward continuation which amplifies
    noise and is not offered here."""
    if height_m <= 0:
        raise ValueError("상방연속 고도(height_m)는 0보다 커야 합니다.")

    def filt(kx, ky, k_mag):
        return np.exp(-k_mag * height_m)

    return _apply_filter(grid, cell_size_m, filt)
