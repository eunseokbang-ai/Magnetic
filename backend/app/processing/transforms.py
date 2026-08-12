"""FFT-based potential-field grid transforms: reduction to pole (RTP),
reduction to equator (RTE), vertical derivative (1VD) and analytic signal
(AS), following the standard wavenumber-domain filters (Blakely, 1995,
"Potential Theory in Gravity and Magnetic Applications", ch. 12).

Convention: x = northing, y = easting, z positive down. All filters take
a 2D grid (rows=northing, cols=easting) with NaN outside data coverage.
"""
from __future__ import annotations

import numpy as np

_TAPER_FRACTION = 0.25


def _wavenumbers(n_north: int, n_east: int, d_north: float, d_east: float):
    kx = 2 * np.pi * np.fft.fftfreq(n_north, d=d_north)
    ky = 2 * np.pi * np.fft.fftfreq(n_east, d=d_east)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.hypot(kx_grid, ky_grid)
    return kx_grid, ky_grid, k_mag


def _pad_and_fill(grid: np.ndarray):
    """Fill NaNs with the grid mean and mirror-pad the edges (tapered) to
    reduce FFT wraparound artifacts. Returns (padded, mask, pad_widths)."""
    mask = np.isnan(grid)
    fill_value = np.nanmean(grid) if not np.all(mask) else 0.0
    filled = np.where(mask, fill_value, grid)

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
