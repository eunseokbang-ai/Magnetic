"""The striping diagnostic has to name the cause correctly, or it is worse
than nothing - it would send someone to the wrong fix with confidence.

Each test builds a grid with striping of one known wavelength and checks
the diagnostic recovers it and reaches the right verdict.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.processing.grid_diagnostics import diagnose_striping

CELL = 5.0
SPACING = 100.0
NY = NX = 200


def _geology(ax_across: int = 1):
    """Smooth field well above the line spacing - what must NOT be flagged."""
    yy, xx = np.mgrid[0:NY, 0:NX] * CELL
    return 300 * np.sin(2 * np.pi * xx / 1500) + 200 * np.cos(2 * np.pi * yy / 1800)


def _with_stripes(wavelength_m: float, amplitude: float, across_axis: int = 1):
    """Add a corrugation of a known wavelength running across the lines.
    across_axis=1 means the wavenumber points along easting (columns), i.e.
    vertical stripes - which is what N-S flight lines produce."""
    grid = _geology()
    yy, xx = np.mgrid[0:NY, 0:NX] * CELL
    coord = xx if across_axis == 1 else yy
    return grid + amplitude * np.sin(2 * np.pi * coord / wavelength_m)


# Lines running north: azimuth 90 in this codebase's atan2(dy, dx)
# convention, so the across-line direction is easting (columns).
AZ_NORTH = 90.0


def test_it_recovers_a_known_stripe_wavelength():
    grid = _with_stripes(wavelength_m=SPACING, amplitude=40.0)

    d = diagnose_striping(grid, CELL, AZ_NORTH, SPACING)

    assert d["available"]
    assert d["peak_wavelength_m"] == pytest.approx(SPACING, rel=0.15)


def test_striping_at_one_line_spacing_reads_as_a_leveling_problem():
    grid = _with_stripes(wavelength_m=SPACING, amplitude=40.0)

    d = diagnose_striping(grid, CELL, AZ_NORTH, SPACING)

    assert d["peak_wavelength_spacings"] == pytest.approx(1.0, abs=0.3)
    assert "레벨" in d["verdict"]


def test_striping_at_two_line_spacings_reads_as_a_heading_effect():
    """A direction-dependent offset alternates line to line, which is a
    wave of twice the line spacing - a different fix from the case above,
    so the diagnostic must tell them apart."""
    grid = _with_stripes(wavelength_m=2 * SPACING, amplitude=40.0)

    d = diagnose_striping(grid, CELL, AZ_NORTH, SPACING)

    assert d["peak_wavelength_spacings"] == pytest.approx(2.0, abs=0.4)
    assert "헤딩" in d["verdict"]


def test_cell_scale_striping_reads_as_an_interpolation_artifact():
    # 3 cells, not 2: a wave at exactly the 2-cell Nyquist limit samples to
    # zero everywhere, so it is not something a grid can carry at all.
    grid = _with_stripes(wavelength_m=3 * CELL, amplitude=20.0)

    d = diagnose_striping(grid, CELL, AZ_NORTH, SPACING)

    assert d["peak_wavelength_cells"] <= 3.5
    assert "보간" in d["verdict"]


def test_a_clean_grid_carries_little_power_in_the_corrugation_band():
    clean = diagnose_striping(_geology(), CELL, AZ_NORTH, SPACING)
    striped = diagnose_striping(_with_stripes(SPACING, 40.0), CELL, AZ_NORTH, SPACING)

    assert clean["amplitude_pct"] < striped["amplitude_pct"] / 5
    assert clean["amplitude_pct"] < 10.0


def test_stronger_stripes_report_a_larger_amplitude():
    """The number has to be comparable between settings, which is the whole
    point of reporting it."""
    weak = diagnose_striping(_with_stripes(SPACING, 5.0), CELL, AZ_NORTH, SPACING)
    strong = diagnose_striping(_with_stripes(SPACING, 50.0), CELL, AZ_NORTH, SPACING)

    assert strong["amplitude_pct"] > weak["amplitude_pct"]


def test_stripes_running_the_other_way_are_not_counted_as_across_line():
    """Corrugation runs parallel to the lines. Structure perpendicular to
    that is geology and must not be blamed on the survey."""
    along = _geology() + 40.0 * np.sin(2 * np.pi * (np.mgrid[0:NY, 0:NX][0] * CELL) / SPACING)

    across_line = diagnose_striping(along, CELL, AZ_NORTH, SPACING)
    # Same grid, but told the lines run east instead - now it IS across-line.
    rotated = diagnose_striping(along, CELL, 0.0, SPACING)

    assert rotated["amplitude_pct"] > across_line["amplitude_pct"] * 3


def test_a_regional_gradient_is_not_mistaken_for_striping():
    """A strong ramp dumps power into the lowest wavenumbers; without
    detrending it would swamp the band and read as corrugation."""
    yy, xx = np.mgrid[0:NY, 0:NX] * CELL
    ramped = _geology() + 0.5 * xx

    d = diagnose_striping(ramped, CELL, AZ_NORTH, SPACING)

    assert d["amplitude_pct"] < 15.0


def test_nan_gaps_do_not_break_it():
    grid = _with_stripes(SPACING, 40.0)
    grid[80:120, 80:120] = np.nan

    d = diagnose_striping(grid, CELL, AZ_NORTH, SPACING)

    assert d["available"]
    assert d["peak_wavelength_m"] == pytest.approx(SPACING, rel=0.2)


def test_without_a_line_direction_it_still_says_which_way_the_stripes_run():
    vertical = diagnose_striping(_with_stripes(SPACING, 40.0, across_axis=1), CELL, None, SPACING)
    horizontal = diagnose_striping(_with_stripes(SPACING, 40.0, across_axis=0), CELL, None, SPACING)

    assert "세로" in vertical["direction"]
    assert "가로" in horizontal["direction"]


def test_a_grid_too_small_or_flat_declines_rather_than_reporting_noise():
    assert not diagnose_striping(np.zeros((4, 4)), CELL, AZ_NORTH, SPACING)["available"]
    assert not diagnose_striping(np.zeros((200, 200)), CELL, AZ_NORTH, SPACING)["available"]
