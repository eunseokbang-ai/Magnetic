"""Bringing every reading to one flight height before anything treats the
survey as a plane.

Everything downstream of the point stage - the FFT transforms, Euler
deconvolution, the depth estimates, the 3D inversion - takes the readings
as lying on one horizontal surface. A drone does not fly one: it follows
the ground at a set clearance, so the readings sit on a copy of the
terrain lifted by the flight height, and on hilly ground neighbouring
readings differ in height by tens of metres. A compact source's field
falls with the cube of distance, so ten metres of height on a 90 m-deep
source changes its amplitude by a third; where the height difference
follows the lines, it draws stripes along them.

This module fits an equivalent source layer to the readings at their
true three-dimensional positions and evaluates what that layer would
have produced on a level surface. The correction applied to each reading
is the layer's field at the reference height minus its field at the
reading's own height - the height change of the modelled part only,
added to the measured value. That form has two properties the plain
replacement (write the layer's field at the reference height over the
reading) does not:

* what the layer failed to fit stays in the data untouched, instead of
  being thrown away - the residual has no known height dependence, so
  nothing better can be done with it;
* a reading already at the reference height gets exactly zero
  correction, whatever the fit did, so a flat flight is left alone.

The model is a regular grid of dipoles magnetised along the inducing
field (IGRF inclination and declination), one scalar strength each,
buried below the lowest reading. Remanence is deliberately not allowed:
the job is to reproduce the readings, not to interpret them, and a fixed
direction keeps the problem well posed with a third of the unknowns.

Measured against a known answer (tests/test_height_normalization.py): a
compact source 40 m below ground (170 nT at the drone) and a geology of
1.4 km wavelength (44 nT spread), sampled on 40 lines 50 m apart flown
50 m above a hill with 60 m of relief, 0.2 nT of noise, and compared with
the true field on the median flight surface. Rms error against that
truth, in nT, for the interior (three lines and 100 m in from the edge),
the 300 m round the compact source, and the outermost line:

    layer depth below      interior       near source     outermost line
    lowest reading         before/after   before/after    before/after
    0.25 spacings (12 m)   8.25 / 2.05    20.7 / 1.5      3.9 / 9.6
    0.35 spacings (18 m)   8.25 / 4.19    20.7 / 2.5      3.9 / 7.6
    0.50 spacings (25 m)   8.25 / 1.22    20.7 / 3.0      3.9 / 8.7  <- default
    0.75 spacings (38 m)   8.25 / 2.30    20.7 / 6.3      3.9 / 6.4
    1.00 spacings (50 m)   8.25 / 2.71    20.7 / 7.5      3.9 / 3.2
    1.25 spacings (62 m)   8.25 / 3.24    20.7 / 9.0      3.9 / 5.9

Shallower is better for the compact source, until at a quarter of a line
spacing the layer starts reproducing noise between lines (the interior
turns worse again). Half a line spacing is the flat optimum, and holds
with ten times the noise (2 nT per sample: interior 8.5 -> 2.4, near
source 20.8 -> 3.6). One source per half line spacing is enough: a
quarter spacing changes the interior error by less than 0.1 nT at four
times the cost. Damping between 1e-3 and 1e-1 makes no difference; 1
under-fits (interior 1.57).

The outermost line is the one place the layer cannot be trusted: with
data on one side only, its vertical gradient there is wrong, and the
correction made that line worse in every configuration (3.9 -> 6-9 nT).
The correction is therefore tapered to zero over the last line spacing
of the coverage (_EDGE_TAPER_SPACINGS). Measured: outermost line
3.9 -> 3.1 nT with the taper, the second and third lines unchanged
(3.2 -> 2.0), survey as a whole 7.3 -> 1.7 nT. A deeper second layer for
the regional was tried and is not worth its parameter: it moves error
from the second and third lines (2.0 -> 1.5) to the interior (1.2 ->
1.5) and changes the total by nothing.

Where the reference height is matters as much as the layer. Bringing
readings *down* to it is a downward continuation, which amplifies
whatever the layer got wrong at short wavelengths: with the reference at
the median height, the readings over the hilltop come down 30 m, and the
error near the compact source (which sits under the hill) is 3.0 nT;
with the reference at the 90th percentile it is 0.5 nT, at the maximum
0.7 nT. The median is the default because it changes the least data by
the least amount; a survey whose targets sit under its high ground is
better served by a higher reference, which the caller can pass.

A flight with +/-1 m of height variation gets a median correction of
0.14 nT on the same synthetic: the 0.2 nT/m vertical gradient of its
regional really is there, and it is below the survey's own noise.

Cost, measured on a synthetic of the HaeNam block's size (969,000
readings on 105 lines 50 m apart, 6 km long): 63,000 blocks, 53,600
sources, 21 iterations, 149 s on four cores.

Cost: the kernel is dense, (blocks x sources) with tens of thousands of
each, so it is never formed. Products with it are computed on the fly in
a parallel numba loop, and the damped normal equations are solved by
conjugate gradients on the least-squares form (CGLS) with the columns
scaled to unit norm - a Jacobi preconditioner that needs no extra pass.
Readings are averaged in blocks along each track before the fit, and the
correction is evaluated at the block centres and interpolated back along
the track: the correction varies over the layer depth (a hundred metres),
so ten-metre blocks lose nothing, and the evaluation is 15x cheaper than
one per reading.

References: Dampney (1969) equivalent source technique; Cordell (1992), a
scattered equivalent-source method for interpolation and gridding of
potential-field data in three dimensions - the three-dimensional
scattered observation is this exact problem; Li & Oldenburg (2010) rapid
construction of equivalent sources; Soler & Uieda (2021) gradient-boosted
equivalent sources, for surveys with far more readings than this fits at
once.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange
from scipy import ndimage
from scipy.spatial import cKDTree

__all__ = [
    "HeightNormalization",
    "TrackBlocks",
    "block_tracks",
    "fit_height_normalization",
    "altitude_summary",
]

# Source spacing as a fraction of the line spacing. Two sources per line
# spacing (0.5) - see the table in the module docstring.
_SOURCE_SPACING_FACTOR = 0.5
# Depth of the layer below the lowest reading, in line spacings - the
# flat optimum in the docstring's table.
_DEPTH_FACTOR = 0.5
# Along-track block length. Readings 0.6 m apart carry nothing the layer
# (25 m sources, 25 m and more below the readings) could reproduce below
# ten metres.
_BLOCK_SIZE_M = 10.0
# How far past the outermost blocks the layer reaches, in multiples of
# the median height of the readings above the layer. Sources further
# than this cannot be told from zero by any reading.
_MARGIN_HEIGHTS = 2.0
# Damping on the unit-column-scaled system: lambda^2 relative to the
# diagonal of G^T G, which the scaling makes 1. See the docstring.
_DAMPING = 1e-2
# The correction fades to zero over this many line spacings in from the
# edge of the coverage - see the docstring for why the outermost line
# cannot be corrected.
_EDGE_TAPER_SPACINGS = 1.0
_MAX_ITER = 30
# The iteration stops when one step changes the fit rms by less than
# this fraction of the data spread.
_TOL = 1e-3
# Above this ratio of fit error to data spread the normalisation is
# still applied - it can only under-correct - but flagged.
_POOR_FIT_RATIO = 0.3
# Below this altitude spread there is nothing to normalise: a survey
# flown this level differs from a plane by less than its GPS noise.
_FLAT_FLIGHT_STD_M = 1.0
_CHUNK = 4096
_MIN_BLOCKS = 30


# ------------------------------------------------------------ kernel
# Total-field anomaly at (ox, oy, oz) of a dipole at (sx, sy, sz) with
# moment m along the unit field direction (fe, fn, fu), projected on the
# same direction: 100 * m * (3 (f.r)^2 - r^2) / r^5 nT per A m^2. The
# products below are what CGLS needs; the matrix itself is never built.


@njit(parallel=True, cache=True)
def _forward(ox, oy, oz, sx, sy, sz, m, fe, fn, fu):
    n_obs = ox.shape[0]
    n_src = sx.shape[0]
    out = np.empty(n_obs)
    for i in prange(n_obs):
        acc = 0.0
        xi, yi, zi = ox[i], oy[i], oz[i]
        for j in range(n_src):
            rx = xi - sx[j]
            ry = yi - sy[j]
            rz = zi - sz[j]
            r2 = rx * rx + ry * ry + rz * rz
            fr = fe * rx + fn * ry + fu * rz
            acc += m[j] * (3.0 * fr * fr - r2) / (r2 * r2 * np.sqrt(r2))
        out[i] = 100.0 * acc
    return out


@njit(parallel=True, cache=True)
def _adjoint(ox, oy, oz, sx, sy, sz, v, fe, fn, fu):
    n_obs = ox.shape[0]
    n_src = sx.shape[0]
    out = np.empty(n_src)
    for j in prange(n_src):
        acc = 0.0
        xj, yj, zj = sx[j], sy[j], sz[j]
        for i in range(n_obs):
            rx = ox[i] - xj
            ry = oy[i] - yj
            rz = oz[i] - zj
            r2 = rx * rx + ry * ry + rz * rz
            fr = fe * rx + fn * ry + fu * rz
            acc += v[i] * (3.0 * fr * fr - r2) / (r2 * r2 * np.sqrt(r2))
        out[j] = 100.0 * acc
    return out


@njit(parallel=True, cache=True)
def _column_norms_sq(ox, oy, oz, sx, sy, sz, fe, fn, fu):
    n_obs = ox.shape[0]
    n_src = sx.shape[0]
    out = np.empty(n_src)
    for j in prange(n_src):
        acc = 0.0
        xj, yj, zj = sx[j], sy[j], sz[j]
        for i in range(n_obs):
            rx = ox[i] - xj
            ry = oy[i] - yj
            rz = oz[i] - zj
            r2 = rx * rx + ry * ry + rz * rz
            fr = fe * rx + fn * ry + fu * rz
            g = (3.0 * fr * fr - r2) / (r2 * r2 * np.sqrt(r2))
            acc += g * g
        out[j] = 1e4 * acc
    return out


def _field_direction(inclination_deg: float, declination_deg: float) -> np.ndarray:
    inc, dec = np.radians(inclination_deg), np.radians(declination_deg)
    return np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)])


# ------------------------------------------------------------ blocks
@dataclass
class TrackBlocks:
    """Readings averaged in along-track blocks. `block_of_point` maps each
    reading to its block (-1 when the reading had no position or height);
    `along` is each reading's distance along its track and `along_centre`
    the block's mean of it, which is what the correction is interpolated
    on."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    group: np.ndarray
    along_centre: np.ndarray
    count: np.ndarray
    block_of_point: np.ndarray
    along: np.ndarray

    def __len__(self) -> int:
        return len(self.x)

    def mean_of(self, values: np.ndarray) -> np.ndarray:
        """Per-block mean of a per-reading quantity, NaN where no reading
        in the block had one."""
        values = np.asarray(values, dtype=float)
        ok = (self.block_of_point >= 0) & np.isfinite(values)
        total = np.bincount(self.block_of_point[ok], weights=values[ok], minlength=len(self))
        n = np.bincount(self.block_of_point[ok], minlength=len(self))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, total / np.maximum(n, 1), np.nan)

    def any_of(self, flags: np.ndarray) -> np.ndarray:
        flags = np.asarray(flags, dtype=bool) & (self.block_of_point >= 0)
        return np.bincount(self.block_of_point[flags], minlength=len(self)) > 0

    def to_points(self, per_block: np.ndarray) -> np.ndarray:
        """A per-block quantity interpolated linearly along each track to
        every reading; 0 for readings outside any block."""
        per_block = np.asarray(per_block, dtype=float)
        out = np.zeros(len(self.block_of_point))
        has = self.block_of_point >= 0
        idx = np.flatnonzero(has)
        point_group = np.full(len(out), -1, dtype=np.int64)
        point_group[has] = self.group[self.block_of_point[has]]
        for g in np.unique(self.group):
            blocks = np.flatnonzero(self.group == g)
            pts = idx[point_group[idx] == g]
            if len(pts) == 0:
                continue
            if len(blocks) == 1:
                out[pts] = per_block[blocks[0]]
                continue
            order = np.argsort(self.along_centre[blocks])
            out[pts] = np.interp(
                self.along[pts], self.along_centre[blocks][order], per_block[blocks][order]
            )
        return out


def block_tracks(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, group: np.ndarray, block_size_m: float = _BLOCK_SIZE_M
) -> TrackBlocks:
    """Cut each track (a run of consecutive readings with the same group
    id, in the order given) into blocks `block_size_m` long along the
    track and average the positions in each. Readings with no finite
    position or height belong to no block."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    group = np.asarray(group)
    n = len(x)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    # Runs of equal group id, in order: a line flown twice is two tracks.
    run = np.zeros(n, dtype=np.int64)
    if n > 1:
        run[1:] = np.cumsum(group[1:] != group[:-1])

    along = np.zeros(n)
    block_of_point = np.full(n, -1, dtype=np.int64)
    keys = []
    next_block = 0
    for r in np.unique(run):
        pts = np.flatnonzero((run == r) & ok)
        if len(pts) == 0:
            continue
        step = np.hypot(np.diff(x[pts]), np.diff(y[pts]))
        dist = np.concatenate([[0.0], np.cumsum(step)])
        along[pts] = dist
        cell = np.floor(dist / block_size_m).astype(np.int64)
        uniq, inv = np.unique(cell, return_inverse=True)
        block_of_point[pts] = next_block + inv
        keys.append((r, len(uniq)))
        next_block += len(uniq)

    n_blocks = next_block
    has = block_of_point >= 0
    count = np.bincount(block_of_point[has], minlength=n_blocks).astype(np.int64)
    safe = np.maximum(count, 1)

    def mean(v):
        return np.bincount(block_of_point[has], weights=v[has], minlength=n_blocks) / safe

    block_group = np.concatenate([np.full(k, r, dtype=np.int64) for r, k in keys]) if keys else np.zeros(0, dtype=np.int64)
    return TrackBlocks(
        x=mean(x), y=mean(y), z=mean(z), group=block_group, along_centre=mean(along),
        count=count, block_of_point=block_of_point, along=along,
    )


# ------------------------------------------------------------ result
@dataclass
class HeightNormalization:
    z_ref_m: float
    sources_xyz: np.ndarray            # (n, 3) east, north, up
    moments: np.ndarray                # (n,) A m^2 along the inducing field
    inclination_deg: float
    declination_deg: float
    line_spacing_m: float
    source_spacing_m: float
    layer_z_m: float
    block_size_m: float
    n_blocks_fitted: int
    fit_rms_nt: float
    data_rms_nt: float
    n_iterations: int
    converged: bool
    altitude: dict
    correction_nt: np.ndarray          # one per reading given to the fit, 0 where none could be made
    n_points_uncorrected: int
    n_points_edge_tapered: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def field_direction(self) -> np.ndarray:
        return _field_direction(self.inclination_deg, self.declination_deg)

    def field_at(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The layer's total-field anomaly at three-dimensional points."""
        x = np.ascontiguousarray(x, dtype=float)
        y = np.ascontiguousarray(y, dtype=float)
        z = np.ascontiguousarray(np.broadcast_to(np.asarray(z, dtype=float), x.shape))
        return _forward_chunked(x, y, z, self.sources_xyz, self.moments, self.field_direction)

    def correction_at(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """What to add to a reading taken at (x, y, z) to bring it to z_ref."""
        x = np.asarray(x, dtype=float)
        return self.field_at(x, y, np.full(x.shape, self.z_ref_m)) - self.field_at(x, y, z)

    def summary(self) -> dict:
        corr = self.correction_nt[np.isfinite(self.correction_nt)]
        corrected = corr[corr != 0.0]
        return {
            "applied": True,
            "z_ref_m": round(self.z_ref_m, 2),
            "layer_z_m": round(self.layer_z_m, 2),
            "layer_depth_below_min_m": round(float(self.altitude["min"] - self.layer_z_m), 1),
            "source_spacing_m": round(self.source_spacing_m, 2),
            "n_sources": int(len(self.sources_xyz)),
            "n_blocks_fitted": int(self.n_blocks_fitted),
            "block_size_m": self.block_size_m,
            "fit_rms_nt": round(self.fit_rms_nt, 3),
            "data_rms_nt": round(self.data_rms_nt, 3),
            "fit_ratio": round(self.fit_rms_nt / self.data_rms_nt, 3) if self.data_rms_nt > 0 else None,
            "n_iterations": int(self.n_iterations),
            "converged": bool(self.converged),
            "altitude": self.altitude,
            "correction": {
                "median_abs_nt": round(float(np.median(np.abs(corrected))), 3) if len(corrected) else 0.0,
                "max_abs_nt": round(float(np.max(np.abs(corrected))), 3) if len(corrected) else 0.0,
                "rms_nt": round(float(np.sqrt(np.mean(corrected ** 2))), 3) if len(corrected) else 0.0,
                "n_points": int(len(corrected)),
                "n_points_uncorrected": int(self.n_points_uncorrected),
                "n_points_edge_tapered": int(self.n_points_edge_tapered),
            },
            "warnings": list(self.warnings),
        }

    def to_meta(self) -> dict:
        """The scalars, for the bundle's meta.json; arrays go to to_arrays."""
        return {
            "z_ref_m": self.z_ref_m,
            "inclination_deg": self.inclination_deg,
            "declination_deg": self.declination_deg,
            "line_spacing_m": self.line_spacing_m,
            "source_spacing_m": self.source_spacing_m,
            "layer_z_m": self.layer_z_m,
            "block_size_m": self.block_size_m,
            "n_blocks_fitted": self.n_blocks_fitted,
            "fit_rms_nt": self.fit_rms_nt,
            "data_rms_nt": self.data_rms_nt,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "altitude": self.altitude,
            "n_points_uncorrected": self.n_points_uncorrected,
            "n_points_edge_tapered": self.n_points_edge_tapered,
            "warnings": list(self.warnings),
        }

    def to_arrays(self) -> dict:
        return {
            "sources_xyz": self.sources_xyz.astype(np.float64),
            "moments": self.moments.astype(np.float64),
            "correction_nt": self.correction_nt.astype(np.float32),
        }

    @classmethod
    def from_saved(cls, meta: dict, arrays: dict) -> "HeightNormalization":
        return cls(
            z_ref_m=float(meta["z_ref_m"]),
            sources_xyz=np.asarray(arrays["sources_xyz"], dtype=float),
            moments=np.asarray(arrays["moments"], dtype=float),
            inclination_deg=float(meta["inclination_deg"]),
            declination_deg=float(meta["declination_deg"]),
            line_spacing_m=float(meta["line_spacing_m"]),
            source_spacing_m=float(meta["source_spacing_m"]),
            layer_z_m=float(meta["layer_z_m"]),
            block_size_m=float(meta["block_size_m"]),
            n_blocks_fitted=int(meta["n_blocks_fitted"]),
            fit_rms_nt=float(meta["fit_rms_nt"]),
            data_rms_nt=float(meta["data_rms_nt"]),
            n_iterations=int(meta["n_iterations"]),
            converged=bool(meta["converged"]),
            altitude=dict(meta["altitude"]),
            correction_nt=np.asarray(arrays["correction_nt"], dtype=float),
            n_points_uncorrected=int(meta.get("n_points_uncorrected", 0)),
            n_points_edge_tapered=int(meta.get("n_points_edge_tapered", 0)),
            warnings=list(meta.get("warnings", [])),
        )


def altitude_summary(z: np.ndarray) -> dict:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return {"n": 0, "min": None, "max": None, "median": None, "std": None, "range": None}
    return {
        "n": int(len(z)),
        "min": round(float(z.min()), 2),
        "max": round(float(z.max()), 2),
        "median": round(float(np.median(z)), 2),
        "std": round(float(z.std()), 2),
        "range": round(float(z.max() - z.min()), 2),
    }


# ------------------------------------------------------------ fit
def _forward_chunked(x, y, z, sources, moments, f):
    out = np.empty(len(x))
    sx, sy, sz = (np.ascontiguousarray(sources[:, k]) for k in range(3))
    m = np.ascontiguousarray(moments, dtype=float)
    for i0 in range(0, len(x), _CHUNK * 8):
        sl = slice(i0, i0 + _CHUNK * 8)
        out[sl] = _forward(x[sl], y[sl], z[sl], sx, sy, sz, m, f[0], f[1], f[2])
    return out


def _source_grid(bx: np.ndarray, by: np.ndarray, spacing: float, margin: float, z: float) -> np.ndarray:
    """A regular grid of sources at height z, covering every block and a
    margin round it, and nothing further from the data than the margin."""
    x0, x1 = bx.min() - margin, bx.max() + margin
    y0, y1 = by.min() - margin, by.max() + margin
    gx = np.arange(x0, x1 + spacing, spacing)
    gy = np.arange(y0, y1 + spacing, spacing)
    gx = gx + (x0 + x1 - gx[0] - gx[-1]) / 2.0
    gy = gy + (y0 + y1 - gy[0] - gy[-1]) / 2.0
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    dist, _ = cKDTree(np.column_stack([bx, by])).query(pts, distance_upper_bound=max(margin, spacing))
    pts = pts[np.isfinite(dist)]
    return np.column_stack([pts, np.full(len(pts), z)])


def _edge_distance(bx: np.ndarray, by: np.ndarray, line_spacing_m: float) -> np.ndarray:
    """How far each block is from the edge of the coverage, in metres,
    measured on an occupancy grid of half a line spacing with the gaps
    between lines closed."""
    cell = 0.5 * float(line_spacing_m)
    ix = np.floor((bx - bx.min()) / cell).astype(int) + 2
    iy = np.floor((by - by.min()) / cell).astype(int) + 2
    occ = np.zeros((iy.max() + 3, ix.max() + 3), dtype=bool)
    occ[iy, ix] = True
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3), dtype=bool))
    occ[iy, ix] = True
    dist = ndimage.distance_transform_edt(occ) * cell
    return np.clip(dist[iy, ix] - 0.5 * cell, 0.0, None)


def _plane(bx, by, bv):
    cx, cy = bx.mean(), by.mean()
    scale = max(np.ptp(bx), np.ptp(by), 1.0)
    A = np.column_stack([np.ones_like(bx), (bx - cx) / scale, (by - cy) / scale])
    coef, *_ = np.linalg.lstsq(A, bv, rcond=None)
    return A @ coef


def _cgls(ox, oy, oz, sources, f, d, damping, max_iter, tol_nt):
    """min ||G m - d||^2 + damping ||S m||^2 with S the column norms of G,
    by CGLS on the column-scaled system. Returns m, the fit rms, the
    iteration count and whether the stop was the tolerance."""
    sx, sy, sz = (np.ascontiguousarray(sources[:, k]) for k in range(3))
    fe, fn, fu = (float(v) for v in f)
    scale = np.sqrt(_column_norms_sq(ox, oy, oz, sx, sy, sz, fe, fn, fu))
    scale[scale == 0] = 1.0

    def fwd(u):
        return _forward(ox, oy, oz, sx, sy, sz, u / scale, fe, fn, fu)

    def adj(r):
        return _adjoint(ox, oy, oz, sx, sy, sz, r, fe, fn, fu) / scale

    u = np.zeros(len(sx))
    r = d.copy()
    s = adj(r)
    p = s.copy()
    gamma = float(s @ s)
    prev_rms = float(np.sqrt(np.mean(r * r)))
    n_iter, converged = 0, False
    for n_iter in range(1, max_iter + 1):
        q = fwd(p)
        denom = float(q @ q) + damping * float(p @ p)
        if denom <= 0.0 or gamma <= 0.0:
            converged = True
            break
        alpha = gamma / denom
        u += alpha * p
        r -= alpha * q
        s = adj(r) - damping * u
        gamma_new = float(s @ s)
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new
        rms = float(np.sqrt(np.mean(r * r)))
        if abs(prev_rms - rms) < tol_nt:
            converged = True
            break
        prev_rms = rms
    rms = float(np.sqrt(np.mean(r * r)))
    return u / scale, rms, n_iter, converged


def fit_height_normalization(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    group: np.ndarray,
    line_spacing_m: float,
    inclination_deg: float,
    declination_deg: float,
    z_ref_m: float | None = None,
    fit_mask: np.ndarray | None = None,
    block_size_m: float = _BLOCK_SIZE_M,
    depth_factor: float = _DEPTH_FACTOR,
    spacing_factor: float = _SOURCE_SPACING_FACTOR,
    damping: float = _DAMPING,
    max_iter: int = _MAX_ITER,
) -> HeightNormalization:
    """Fit the layer to the readings that `fit_mask` allows (all, by
    default) and work out the correction for every reading given.

    `group` tells tracks apart (the line id; consecutive readings with
    the same value are one track). Readings whose height or value is
    missing get a zero correction and are counted.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(x)
    if not (len(y) == len(z) == len(values) == len(group) == n):
        raise ValueError("x, y, z, values and group must have the same length")
    if not line_spacing_m or line_spacing_m <= 0:
        raise ValueError("line spacing is needed to size the source layer")
    fit_mask = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)

    usable = np.isfinite(z) & np.isfinite(x) & np.isfinite(y)
    if usable.sum() < _MIN_BLOCKS:
        raise ValueError("측점 고도가 없어 비행 고도 정규화를 할 수 없습니다.")
    altitude = altitude_summary(z[fit_mask & usable] if (fit_mask & usable).any() else z[usable])
    if z_ref_m is None:
        z_ref_m = float(altitude["median"])

    blocks = block_tracks(x, y, z, group, block_size_m)
    bv = blocks.mean_of(np.where(fit_mask, values, np.nan))
    in_fit = np.isfinite(bv) & blocks.any_of(fit_mask & usable & np.isfinite(values))
    if in_fit.sum() < _MIN_BLOCKS:
        raise ValueError("정규화에 쓸 측점이 너무 적습니다.")

    bx, by, bz, d = blocks.x[in_fit], blocks.y[in_fit], blocks.z[in_fit], bv[in_fit]
    # A plane is harmonic and its own continuation, so the layer is not
    # asked to reproduce the regional gradient with edge sources.
    d = d - _plane(bx, by, d)
    data_rms = float(np.std(d))

    spacing = spacing_factor * float(line_spacing_m)
    layer_z = float(np.nanmin(bz)) - depth_factor * float(line_spacing_m)
    # The field is only meaningful above the sources; a reference near or
    # below the layer would evaluate the kernel at its singularity.
    if z_ref_m <= layer_z + spacing:
        raise ValueError(
            f"기준 고도 {z_ref_m:.1f} m가 등가층({layer_z:.1f} m)에 너무 가깝거나 그 아래입니다 - "
            f"최저 측점 고도 {altitude['min']:.1f} m 근처 이상으로 잡으세요."
        )
    margin = _MARGIN_HEIGHTS * (float(np.median(bz)) - layer_z)
    sources = _source_grid(bx, by, spacing, margin, layer_z)
    f = _field_direction(inclination_deg, declination_deg)

    ox, oy, oz = (np.ascontiguousarray(v) for v in (bx, by, bz))
    moments, fit_rms, n_iter, converged = _cgls(
        ox, oy, oz, sources, f, np.ascontiguousarray(d), damping, max_iter, _TOL * max(data_rms, 1e-6)
    )

    result = HeightNormalization(
        z_ref_m=float(z_ref_m),
        sources_xyz=sources,
        moments=moments,
        inclination_deg=float(inclination_deg),
        declination_deg=float(declination_deg),
        line_spacing_m=float(line_spacing_m),
        source_spacing_m=float(spacing),
        layer_z_m=layer_z,
        block_size_m=float(block_size_m),
        n_blocks_fitted=int(in_fit.sum()),
        fit_rms_nt=fit_rms,
        data_rms_nt=data_rms,
        n_iterations=n_iter,
        converged=converged,
        altitude=altitude,
        correction_nt=np.zeros(n),
        n_points_uncorrected=0,
    )

    # The correction at every block the readings formed - fitted or not -
    # then along each track to the readings.
    all_blocks = blocks.count > 0
    corr_block = np.zeros(len(blocks))
    corr_block[all_blocks] = result.correction_at(blocks.x[all_blocks], blocks.y[all_blocks], blocks.z[all_blocks])
    edge = _edge_distance(blocks.x[all_blocks], blocks.y[all_blocks], line_spacing_m)
    taper = np.clip(edge / (_EDGE_TAPER_SPACINGS * float(line_spacing_m)), 0.0, 1.0)
    corr_block[all_blocks] *= taper
    correction = blocks.to_points(corr_block)
    tapered = np.zeros(len(blocks))
    tapered[all_blocks] = taper < 1.0
    n_edge = int(tapered[blocks.block_of_point[blocks.block_of_point >= 0]].sum())
    correction[~usable] = 0.0
    result.correction_nt = correction
    result.n_points_uncorrected = int((~usable).sum())
    result.n_points_edge_tapered = n_edge

    if altitude["std"] is not None and altitude["std"] < _FLAT_FLIGHT_STD_M:
        result.warnings.append(
            f"측점 고도의 표준편차가 {altitude['std']:.1f} m로 GPS 잡음 수준입니다 - 이 비행은 이미 평면에 가깝고, "
            "정규화는 거의 아무것도 바꾸지 않습니다."
        )
    if data_rms > 0 and fit_rms > _POOR_FIT_RATIO * data_rms:
        result.warnings.append(
            f"등가층이 자료를 잘 재현하지 못합니다 (맞춤 오차 {fit_rms:.2f} nT, 자료 변동 {data_rms:.2f} nT의 "
            f"{100 * fit_rms / data_rms:.0f}%) - 재현되지 않은 부분은 고도 보정 없이 그대로 남습니다."
        )
    if not (altitude["min"] <= z_ref_m <= altitude["max"]):
        result.warnings.append(
            f"기준 고도 {z_ref_m:.1f} m가 측점 고도 범위({altitude['min']:.1f}~{altitude['max']:.1f} m) 밖입니다 - "
            "모든 측점이 한 방향으로 연속됩니다."
        )
    if not converged:
        result.warnings.append(
            f"켤레기울기 반복이 {max_iter}회 안에 수렴하지 않았습니다 - 맞춤 오차를 확인하세요."
        )
    if result.n_points_uncorrected:
        result.warnings.append(
            f"고도가 없는 측점 {result.n_points_uncorrected}개는 보정하지 않았습니다."
        )
    return result
