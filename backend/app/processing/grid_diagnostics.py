"""Measuring grid corrugation instead of arguing about it.

"There is striping in the map" is not actionable, because several
unrelated faults produce stripes and they need opposite fixes. What
separates them is the *wavelength* of the corrugation and the direction
it runs, and both are measurable from the grid itself:

  ~2 cells                 the interpolator or the render is aliasing;
                           nothing in a survey varies that fast across
                           the lines, so this is fabricated detail.
  ~1 line spacing          each line sits at its own level - a leveling
                           error, or the flat Voronoi blocks that "raw
                           cell" gridding leaves between lines.
  ~2 line spacings         the two flight directions differ - a heading
                           effect, which alternates line to line.
  much longer              real geology; leave it alone.

diagnose_striping returns the peak and how strong it is, so the answer
comes from the survey rather than from whoever is looking at the map.
"""
from __future__ import annotations

import numpy as np

from .transforms import _fill_gaps

# Only wavenumbers pointing within this angle of the across-line direction
# count as corrugation. Stripes run parallel to the flight lines, so their
# wavenumber vector points across them.
_DIRECTION_TOLERANCE_DEG = 25.0

# Wavelength band searched for a corrugation peak, in cells and in line
# spacings. The short end is the 2-cell Nyquist limit (nothing shorter
# exists in a grid); the long end is where corrugation stops being
# distinguishable from geology.
_MIN_WAVELENGTH_CELLS = 2.0
_MAX_WAVELENGTH_SPACINGS = 3.0


def _across_line_unit(line_azimuth_deg: float) -> tuple[float, float]:
    """Across-line direction in the (kx=northing, ky=easting) basis the
    wavenumber grids use - the same convention as processing/microlevel.py."""
    theta = np.radians(line_azimuth_deg)
    return float(np.cos(theta)), float(-np.sin(theta))


def diagnose_striping(
    values: np.ndarray,
    cell_size_m: float,
    line_azimuth_deg: float | None = None,
    line_spacing_m: float | None = None,
) -> dict:
    """Peak corrugation wavelength running across the flight lines, and how
    much of the grid's variation sits in it.

    `amplitude_pct` is the share of the grid's total variance carried by
    across-line wavenumbers in the corrugation band. It is the number to
    watch between processing settings: a clean grid puts a few percent
    there, a visibly striped one much more.
    """
    finite = np.isfinite(values)
    if finite.sum() < 64 or values.ndim != 2 or min(values.shape) < 8:
        return {"available": False, "reason": "격자가 너무 작아 줄무늬를 분석할 수 없습니다."}

    filled = _fill_gaps(values, ~finite)
    filled = np.where(np.isfinite(filled), filled, float(np.nanmean(values)))

    # Remove the plane first: a strong regional gradient dumps power into
    # the lowest wavenumbers and would swamp the comparison.
    ny, nx = filled.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    basis = np.column_stack([np.ones(filled.size), xx.ravel(), yy.ravel()])
    coeff, *_ = np.linalg.lstsq(basis, filled.ravel(), rcond=None)
    detrended = filled - (basis @ coeff).reshape(filled.shape)

    # Window, or the grid's own edges ring across every wavenumber and
    # look like broadband corrugation.
    detrended = detrended * np.outer(np.hanning(ny), np.hanning(nx))

    power = np.abs(np.fft.fft2(detrended)) ** 2
    kx = 2 * np.pi * np.fft.fftfreq(ny, d=cell_size_m)
    ky = 2 * np.pi * np.fft.fftfreq(nx, d=cell_size_m)
    kx_g, ky_g = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.hypot(kx_g, ky_g)

    total = float(power.sum()) - float(power[0, 0])
    if total <= 0:
        return {"available": False, "reason": "격자에 변화가 없어 줄무늬를 분석할 수 없습니다."}

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = np.where(k_mag > 0, 2 * np.pi / k_mag, np.inf)

    if line_azimuth_deg is None:
        # Without a known line direction, look at both grid axes and take
        # whichever carries more - still tells the user which way the
        # stripes run.
        candidates = [("행(가로 줄무늬)", (1.0, 0.0)), ("열(세로 줄무늬)", (0.0, 1.0))]
    else:
        candidates = [("측선 직각", _across_line_unit(line_azimuth_deg))]

    max_wavelength = (
        _MAX_WAVELENGTH_SPACINGS * line_spacing_m
        if line_spacing_m and line_spacing_m > 0
        else max(ny, nx) * cell_size_m / 4.0
    )
    band = (wavelength >= _MIN_WAVELENGTH_CELLS * cell_size_m) & (wavelength <= max_wavelength)

    best = None
    for label, (ux, uy) in candidates:
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_angle = np.abs((kx_g * ux + ky_g * uy) / np.where(k_mag == 0, 1.0, k_mag))
        aligned = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0))) <= _DIRECTION_TOLERANCE_DEG
        sel = band & aligned
        if not sel.any():
            continue
        share = float(power[sel].sum()) / total
        peak_wavelength = float(wavelength[sel][np.argmax(power[sel])])
        if best is None or share > best["amplitude_pct"] / 100.0:
            best = {
                "direction": label,
                "amplitude_pct": round(100.0 * share, 2),
                "peak_wavelength_m": round(peak_wavelength, 2),
                "peak_wavelength_cells": round(peak_wavelength / cell_size_m, 2),
            }

    if best is None:
        return {"available": False, "reason": "분석할 파장 대역이 비어 있습니다."}

    out = {"available": True, "cell_size_m": cell_size_m, "line_spacing_m": line_spacing_m, **best}
    out["verdict"] = _verdict(best, cell_size_m, line_spacing_m)
    if line_spacing_m and line_spacing_m > 0:
        out["peak_wavelength_spacings"] = round(best["peak_wavelength_m"] / line_spacing_m, 2)
    return out


def _verdict(best: dict, cell_size_m: float, line_spacing_m: float | None) -> str:
    """Plain-language reading of the peak - which of the causes in the
    module docstring this grid's corrugation matches."""
    cells = best["peak_wavelength_cells"]
    # 3 cells rather than 2: a wave at exactly 2 cells is at the Nyquist
    # limit and samples to zero, so anything the grid can actually carry
    # at "cell scale" shows up at 3 cells or a little above.
    if cells <= 3.5:
        return (
            f"셀 2개 수준({cells:.1f}셀) 파장입니다 - 자료에 있는 신호가 아니라 "
            "보간/렌더 단계에서 생긴 것입니다. 셀 크기를 키우거나 보간법을 바꿔보세요."
        )
    if line_spacing_m and line_spacing_m > 0:
        ratio = best["peak_wavelength_m"] / line_spacing_m
        if 0.7 <= ratio <= 1.4:
            return (
                f"파장이 측선 간격의 {ratio:.2f}배입니다 - 측선마다 레벨이 어긋나 있거나, "
                "'원본 셀' 보간이 남기는 측선별 평평한 블록입니다. 통계적 레벨링과 "
                "미분 전 분해능 평활을 확인하세요."
            )
        if 1.6 <= ratio <= 2.5:
            return (
                f"파장이 측선 간격의 {ratio:.2f}배입니다 - 전진/후진 두 방향이 서로 다르게 "
                "읽히는 헤딩 효과의 전형적인 파장입니다. 헤딩 보정을 켜보세요."
            )
        if ratio > 2.5:
            return f"파장이 측선 간격의 {ratio:.2f}배로 충분히 길어 실제 지질 신호일 가능성이 높습니다."
        return f"파장이 측선 간격의 {ratio:.2f}배입니다 - 측선 간격보다 짧아 측정된 적 없는 성분입니다."
    return f"파장 {best['peak_wavelength_m']:.0f}m ({cells:.1f}셀). 측선 간격을 알 수 없어 원인 판정은 보류합니다."
