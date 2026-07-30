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


def reduction_to_pole(
    grid: np.ndarray,
    cell_size_m: float,
    inclination_deg: float,
    declination_deg: float,
) -> np.ndarray:
    """Transform an induced-magnetization anomaly to its pole-reduced
    equivalent (as if measured with a vertical, 90 deg inclination field)."""
    a, b, c = _direction_coeffs(inclination_deg, declination_deg)

    def filt(kx, ky, k_mag):
        denom = 1j * (kx * a + ky * b) + k_mag * c
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = (k_mag**2) / (denom**2)
        theta[(k_mag == 0)] = 1.0
        theta[~np.isfinite(theta)] = 0.0
        return theta

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


def analytic_signal(grid: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Amplitude of the analytic signal: sqrt(dF/dx^2 + dF/dy^2 + dF/dz^2)."""
    padded, mask, pad_widths = _pad_and_fill(grid)
    kx, ky, k_mag = _wavenumbers(padded.shape[0], padded.shape[1], cell_size_m, cell_size_m)
    spectrum = np.fft.fft2(padded)

    dx = np.real(np.fft.ifft2(spectrum * (1j * kx)))
    dy = np.real(np.fft.ifft2(spectrum * (1j * ky)))
    dz = np.real(np.fft.ifft2(spectrum * k_mag))

    amplitude = np.sqrt(dx**2 + dy**2 + dz**2)
    return _unpad_and_mask(amplitude, mask, pad_widths)
