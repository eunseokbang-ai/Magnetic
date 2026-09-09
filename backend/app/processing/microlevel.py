"""Micro-leveling (decorrugation filter): suppresses residual short-
wavelength "corrugation" noise that runs parallel to the flight lines -
line-to-line leveling offsets or heading-dependent drift the coarser
leveling steps didn't fully remove - via a directional filter in the
wavenumber domain, the standard approach in aeromag processing packages
(e.g. Geosoft's "decorrugation"/directional filter).

Unlike tie-line/crossover leveling (a per-line constant shift estimated
from measured crossover differences, processing/crossover_leveling.py),
heading leveling (a per-heading constant offset, processing/leveling.py)
and statistical leveling (a per-line shift from the neighbouring lines,
processing/statistical_leveling.py), this works purely on the gridded
data's own spatial spectrum: energy whose wavenumber vector points
across the flight lines - corrugation's characteristic direction - is
attenuated, while everything else, including real geological signal at
any other orientation, passes through mostly unchanged.

Two shapes are available for the wavelength part of that filter:

"decorrugation" (default) - a directional high-pass, which is the
classic Minty (1991) micro-levelling formulation: everything across-line
shorter than a cutoff (by default four line spacings) is removed. This
is the one to reach for against real striping, because a per-line level
error is not a single wavelength. A field where each line is offset by
its own amount is a square wave across the lines: it has energy at one
line spacing, at two (when the offsets alternate with flight direction),
and in a whole harmonic tail below one. A high-pass covers all of it.

"notch" - a narrow band centred on exactly one line spacing. It preserves
more across-line geological signal, but for the same reason it only
catches the part of the corrugation that happens to sit at that one
scale, and leaves the alternating and harmonic parts behind.
"""
from __future__ import annotations

import numpy as np

from .transforms import _apply_filter

# Steepness of the "decorrugation" high-pass rolloff, as a Butterworth
# order. 2 gives a gentle transition over roughly an octave around the
# cutoff, which avoids the ringing a sharp cut leaves along strong
# anomalies.
_HIGHPASS_ORDER = 2


def apply_microleveling(
    grid_values: np.ndarray,
    cell_size_m: float,
    line_azimuth_deg: float,
    line_spacing_m: float,
    strength: float = 0.8,
    angle_tolerance_deg: float = 15.0,
    wavelength_bandwidth_factor: float = 1.5,
    mode: str = "decorrugation",
    cutoff_spacing_factor: float = 4.0,
) -> np.ndarray:
    """strength: 0 (no-op) to 1 (fully null the targeted band).
    angle_tolerance_deg: half-width of the directional gate around the
    across-line direction.

    mode="decorrugation" (default): removes across-line energy at every
    wavelength shorter than cutoff_spacing_factor * line_spacing_m, with a
    Butterworth rolloff at that cutoff. wavelength_bandwidth_factor is
    unused in this mode.

    mode="notch": the narrower band-reject filter, centred on
    line_spacing_m, where wavelength_bandwidth_factor sets the half-width
    in wavelength as a multiplicative factor (e.g. 1.5 puts the 1-sigma
    points at line_spacing_m*1.5 and line_spacing_m/1.5).
    cutoff_spacing_factor is unused in this mode."""
    if not line_spacing_m or line_spacing_m <= 0:
        raise ValueError("측선 간격을 알 수 없어 micro-leveling을 적용할 수 없습니다 (측선이 2개 이상 필요).")
    if not (0.0 <= strength <= 1.0):
        raise ValueError("보정 강도(strength)는 0~1 사이여야 합니다.")
    if mode not in ("decorrugation", "notch"):
        raise ValueError(f"알 수 없는 micro-leveling 방식입니다: {mode} (decorrugation 또는 notch)")
    if cutoff_spacing_factor <= 0:
        raise ValueError("차단 파장 배수(cutoff_spacing_factor)는 0보다 커야 합니다.")

    # grid.values axes follow processing/transforms.py's convention (kx ~
    # northing/rows, ky ~ easting/cols); dominant_azimuth_deg follows
    # processing/lines.py's atan2(d_northing, d_easting) convention (0 =
    # along easting, 90 = along northing). A line's own unit direction in
    # (easting, northing) is (cos θ, sin θ); rotating 90 deg gives the
    # across-line (corrugation wavenumber) direction (-sin θ, cos θ), which
    # in the (kx=northing, ky=easting) basis is (perp_kx, perp_ky) =
    # (cos θ, -sin θ).
    theta = np.radians(line_azimuth_deg)
    perp_kx, perp_ky = np.cos(theta), -np.sin(theta)

    def filt(kx, ky, k_mag):
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_angle = (kx * perp_kx + ky * perp_ky) / np.where(k_mag == 0, 1.0, k_mag)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        # angle between this wavenumber and the across-line direction,
        # folded to [0, 90] since a direction and its opposite are the
        # same axis for a real-valued field's spectrum.
        angle_diff_deg = np.degrees(np.arccos(np.abs(cos_angle)))
        angular_gate = np.exp(-0.5 * (angle_diff_deg / angle_tolerance_deg) ** 2)

        with np.errstate(divide="ignore"):
            wavelength = np.where(k_mag > 0, 2 * np.pi / k_mag, np.inf)

        if mode == "decorrugation":
            # Butterworth high-pass on wavelength: ~1 well below the
            # cutoff wavelength (i.e. at the short wavelengths corrugation
            # lives at), falling to 0 for the long-wavelength geology
            # that must be preserved.
            cutoff = cutoff_spacing_factor * line_spacing_m
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.isfinite(wavelength), wavelength / cutoff, np.inf)
            wavelength_gate = 1.0 / (1.0 + ratio ** (2 * _HIGHPASS_ORDER))
            wavelength_gate = np.where(np.isfinite(wavelength_gate), wavelength_gate, 0.0)
        else:
            log_ratio = np.log(np.maximum(wavelength, 1e-9) / line_spacing_m)
            bandwidth = np.log(wavelength_bandwidth_factor)
            wavelength_gate = np.exp(-0.5 * (log_ratio / bandwidth) ** 2)

        attenuation = 1.0 - strength * angular_gate * wavelength_gate
        attenuation = np.where(k_mag == 0, 1.0, attenuation)
        return attenuation

    return _apply_filter(grid_values, cell_size_m, filt)
