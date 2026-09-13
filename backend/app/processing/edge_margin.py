"""Marking (or hiding) the unreliable margin along the edge of a derived grid.

Why this exists, in the order the measurements came out:

A strong anomaly sitting at the *edge* of the data throws a straight
streak across a derived grid - along the grid rows or columns, never
diagonally - and the streak is still visible two hundred cells away. On a
synthetic grid the same anomaly placed mid-survey leaks 0.000% into the
far field; moved to two cells from the coverage boundary it leaks 0.757%.
The array edge is irrelevant; it is the *data* boundary that does it. The
cause is the gap fill: every FFT transform has to invent values where the
grid is NaN, and where the invented continuation disagrees with the real
field, the disagreement is a step in the padded array, which the FFT
spreads along its own axes.

Filling it better does not help, which is the part worth recording so
nobody spends another day on it:

    fill                      rms error in the gap     far-field streak
    nearest (what we use)             61.2 nT                0.757%
    mirror                                 -                 0.844%
    harmonic, converged                 3.3 nT               0.849%
    harmonic, decaying                     -                 0.850%
    distance-tapered                       -                 0.676%
    the true continuation                 0.0                0.000%

Only the exact answer removes it, and the streak scales with
(1 - extrapolation fidelity): a tenfold reduction needs a fill that is 90%
right, which no generic extrapolator can promise for a field it has never
seen. Tapering the data into the boundary over 20 cells gets 0.757% ->
0.556% but costs a quarter of the real signal at the edge - a bad trade
for a survey whose targets are often *on* the edge.

So the honest remedy is not a correction, it is a label: say which band of
the map is contaminated by its own boundary and let the interpreter
discount it. The band this module marks is the one the gridding
extrapolated into rather than interpolated within - the caller passes its
own extrapolation radius as `margin_m` - because that is where invented
values live and where the leakage is strongest.

Note what this does *not* claim: the streak decays slowly, so a cell just
outside the margin is not thereby trustworthy. The margin is a guide to
where to be suspicious, not a guarantee about the rest.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = [
    "edge_distance_m",
    "margin_mask",
    "apply_margin",
    "margin_outline",
]


def edge_distance_m(values: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Distance from each cell to the nearest gap, in metres.

    "Gap" means any cell without data (NaN) *and* anything past the array
    itself - the outermost row of a grid that is full to its own edge is
    still one cell from the outside world, and the FFT wraps there. Cells
    that are themselves gaps get 0.0.
    """
    if cell_size_m <= 0:
        raise ValueError(f"cell_size_m must be positive, got {cell_size_m}")
    valid = np.isfinite(values)
    # One ring of "outside" so the array border counts as a boundary; the
    # ring is stripped again below.
    padded = np.pad(valid, 1, mode="constant", constant_values=False)
    dist = ndimage.distance_transform_edt(padded, sampling=cell_size_m)
    return np.asarray(dist)[1:-1, 1:-1]


def margin_mask(values: np.ndarray, cell_size_m: float, margin_m: float) -> np.ndarray:
    """True for the cells that lie within `margin_m` of the data boundary.

    Only cells that have data are marked - a cell that is already NaN is
    not "in the margin", it is simply absent, and counting it would make
    the reported margin area depend on the shape of the bounding box.
    """
    if margin_m <= 0:
        return np.zeros(values.shape, dtype=bool)
    return (edge_distance_m(values, cell_size_m) <= margin_m) & np.isfinite(values)


def apply_margin(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """`values` with the margin cells set to NaN, as a fresh array.

    Never mutates the input: callers hand in cached grid/transform arrays
    that have to stay intact for the next request, which may well ask for
    a different margin (or none).
    """
    out = np.array(values, dtype=float, copy=True)
    out[mask] = np.nan
    return out


def margin_outline(
    values: np.ndarray,
    cell_size_m: float,
    margin_m: float,
    easting: np.ndarray,
    northing: np.ndarray,
) -> list[np.ndarray]:
    """The inner edge of the margin band as polylines in grid coordinates.

    Returns a list of (n, 2) arrays of (easting, northing) in metres - the
    `margin_m` contour of the distance field, which is exactly the line
    "everything outside me is within `margin_m` of a boundary". Drawing
    this instead of masking is the non-destructive half of the option:
    nothing is hidden, the interpreter just sees where to stop trusting
    the map.
    """
    if margin_m <= 0:
        return []
    import contourpy

    dist = edge_distance_m(values, cell_size_m)
    if float(dist.max(initial=0.0)) <= margin_m:
        # The whole survey is inside the margin - there is no inner region
        # left to draw a line around, and contouring would return nothing
        # anyway. The caller reports this case in words instead.
        return []
    east2d, north2d = np.meshgrid(easting, northing)
    cg = contourpy.contour_generator(x=east2d, y=north2d, z=dist, line_type=contourpy.LineType.Separate)
    return [np.asarray(line, dtype=float) for line in cg.lines(float(margin_m)) if len(line) >= 2]
