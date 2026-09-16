"""Ranking the strongest local anomalies in a survey so the operator can
step through them and decide, one at a time, which are ground structures
to be cut out and which are geology to be kept.

This is the "start editing" half of cultural-noise removal. The removal
itself is already handled at the sample level by processing/manual_smooth.py
(a polygon, or an explicit point list, is replaced by an along-line
interpolation from the background on either side). What was missing was
the step before it: on a block where cultural sources are *stronger* than
the geology being looked for, the map is dominated by the structures, and
finding them by eye - one hand-drawn polygon at a time - is the slow part.

Three design decisions are worth recording, all measured rather than
assumed (inclination 50.94 deg, the IGRF value for the 2026-09 HaeNam
block, dipole source, app's own transforms):

1. Candidates are ranked on the analytic signal by default, not on the
   anomaly itself. A compact source at this inclination produces a
   *dipolar* anomaly: a positive peak, and a negative lobe of -31% of it
   about one source-depth away. Ranked on the raw anomaly, one object
   therefore claims two of the ten slots - once as a max, once as a min -
   and neither of them sits over the object. Measured peak offsets from
   the true source position, 5 m depth:

       analytic signal  1.0 m      anomaly  2.0 m      RTP  4.1 m

   The analytic signal is also independent of magnetization direction,
   which matters here specifically: rebar, guardrails and fence posts
   carry strong *remanent* magnetization, and RTP assumes induced-only.
   RTP and residual-anomaly ranking stay available for comparison.

2. The region cut out is sized from the anomaly's own half-width, not
   from a fixed radius. For a compact source the falloff scales purely
   with depth below the sensor (measured, identical at 5/10/20/40 m, so
   these ratios hold whatever the flight clearance):

       |T| = 50% of peak out to 1.00 x depth   <- the classic half-width
       |T| = 25% of peak out to 1.40 x depth        rule, in max radius
       |T| = 10% of peak out to 1.97 x depth

   So a region has to reach about 2 x the source depth to get the anomaly
   below a tenth of its peak. Cutting a tight region around the visible
   peak instead leaves the negative lobe behind as a crater - which, at
   -31% of a cultural peak that by assumption exceeds the geology, is
   still larger than the signal being looked for.

   The depth itself is measured on the analytic signal rather than on the
   anomaly (see _source_depth_m): one positive blob centred on the source,
   whose area-equivalent half-width is a measured 0.556 x depth, and which
   needs no estimate of the local background. Measuring a half-width on
   the anomaly instead requires one, and on a block with a regional
   gradient - or where the anomaly's own negative lobe falls in the ring
   the background is taken from - that estimate is what goes wrong.

3. A candidate has to sit on ground the survey actually measured. On the
   real 2026-09 HaeNam block the three strongest peaks of the analytic
   signal - -552, -166 and +179 nT - were 132 m, 59 m and 948 m from the
   nearest sample, in the gaps between sub-blocks where the grid is the
   interpolator's invention rather than a measurement. Offering those as
   the objects to cut would have been exactly wrong: there is nothing
   there to cut. Candidates are therefore restricted to cells with a
   sample within `support_radius_m` of them.

4. Every candidate reports how many survey lines actually cross it.
   Anything narrower than the line spacing is only sampled by one line,
   and on the grid a one-line feature is indistinguishable from a
   levelling stripe - so those are flagged rather than silently offered
   for removal, and should be confirmed on the line profile.

Nothing here removes anything. It returns ranked candidates with a
suggested polygon each; the operator selects, and the existing
store.py::set_manual_smoothing does the cut.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath
from scipy import ndimage

from .transforms import analytic_signal

# ...and the radius needed to get the anomaly below a tenth of its peak
# in every direction, which is how far the cut region has to reach.
DEPTH_TO_CLEAR_RADIUS = 1.97
# Measured separation of a compact source's positive peak and its
# negative lobe, in source depths. Not used to measure anything - it is
# why a region has to cover more than the peak (see DEPTH_TO_CLEAR_RADIUS).
LOBE_SEPARATION_TO_DEPTH = 1.06
# Measured area-equivalent radius of the blob above half the analytic
# signal's peak, in source depths - how the depth is actually measured
# here. See _source_depth_m for why the analytic signal rather than the
# anomaly.
AS_HALF_WIDTH_TO_DEPTH = 0.556

# Level the region is grown down to, as a fraction of the candidate's own
# peak. 0.25 (1.40 x depth) rather than 0.10 (1.97 x depth) because the
# grown blob is afterwards buffered outward by one estimated depth, which
# carries it past the 10% contour anyway - growing all the way down to
# 10% on real, noisy data instead tends to leak into a neighbouring
# anomaly before it gets there.
DEFAULT_REGION_LEVEL_FRACTION = 0.25

# Below this many robust standard deviations above the background there is
# nothing worth offering for removal - a weaker "candidate" is ordinary
# data, and cutting it would just punch a hole in the survey.
NOISE_FLOOR_K = 4.0

# Half-width of the neighbourhood a cell has to be the maximum of before
# it can seed a candidate. 5 cells (+-2) is enough to reject a point on
# the monotonic flank of an anomaly already accepted, without demanding
# that a real peak be the maximum over a wide area it may share with a
# neighbouring source.
SEED_WINDOW_CELLS = 5

# A candidate has to be a resolved feature, not a single bright cell. A
# magnetic field measured above its sources cannot contain structure finer
# than the sensor's clearance, so anything that small is an artifact of
# the grid rather than an object on the ground.
MIN_REGION_CELLS = 4

# How many local maxima are examined before giving up. Only matters on a
# grid so disturbed that most of it looks like a peak; a normal survey
# runs out of seeds above the noise floor long before this.
MAX_SEEDS_EXAMINED = 2000

# Fraction of the line spacing a candidate's peak may be from the nearest
# sample. Half a line spacing is the most any point inside the surveyed
# area can be; 0.75 leaves room for the ends of lines and for a slightly
# uneven flight without admitting the gaps between sub-blocks.
SUPPORT_RADIUS_LINE_SPACINGS = 0.75

# A candidate with fewer samples than this inside its region has nothing
# to remove even if its peak passed the support test.
MIN_SUPPORTING_POINTS = 5

# Largest radius a source region for removal-by-modelling may take - see
# AnomalyCandidate.source_polygon_xy. Well beyond any building; short of
# swallowing a geological body.
SOURCE_MAX_RADIUS_M = 300.0


@dataclass
class AnomalyCandidate:
    rank: int
    x: float
    y: float
    ranking_value: float          # value on the field candidates were ranked on
    peak_anomaly_nt: float        # signed peak of the anomaly grid in the region
    polarity: str                 # "max" | "min", on the anomaly grid
    half_width_m: float
    depth_m: float                # source depth below the survey plane
    radius_m: float               # equivalent radius of the suggested region
    polygon_xy: list[tuple[float, float]]
    n_lines: int
    n_points: int
    at_coverage_edge: bool
    # True when the anomaly is wider than max_radius_m allows the region
    # to be: the cut will stop short of the 10% contour, and the feature
    # is broad enough to be worth asking whether it is a structure at all.
    region_capped: bool = False
    # False when the analytic signal's peak is no wider than one cell, so
    # the grid cannot resolve the source: the depth is then the grid's own
    # floor and an upper bound, not a measurement.
    depth_resolved: bool = True
    # Where the source itself sits, for removal by modelling
    # (processing/source_removal.py): the analytic-signal blob, not capped
    # at max_radius_m. That cap is about how far a *cut* may reach; a model
    # needs room for the whole source instead, and does its own reaching.
    # A 984 nT HaeNam structure fitted inside the capped 60 m region left
    # the analytic signal at 0.65 over it; inside its 220 m blob, 0.07.
    source_polygon_xy: list = None

    @property
    def single_line(self) -> bool:
        """True when only one flight line crosses this - see point 3 of
        the module docstring: on the grid alone such a feature cannot be
        told apart from a levelling stripe."""
        return self.n_lines <= 1


def _robust_sigma(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(finite - np.median(finite))))


def _grow_region(
    strength: np.ndarray,
    peak_rc: tuple[int, int],
    level: float,
    max_radius_cells: float,
) -> np.ndarray:
    """The connected patch around the peak where `strength` stays above
    `level`, clipped to `max_radius_cells` from the peak.

    The radius clip is what stops a candidate sitting on the flank of a
    broad geological high from growing across half the survey: the cut is
    meant to be local, and a region that wants to grow beyond what its own
    half-width allows is a sign the peak is not a compact source at all.
    """
    above = np.isfinite(strength) & (strength >= level)
    if not above[peak_rc]:
        above = above.copy()
        above[peak_rc] = True
    labels, _ = ndimage.label(above)
    region = labels == labels[peak_rc]

    rr, cc = np.ogrid[: strength.shape[0], : strength.shape[1]]
    within = (rr - peak_rc[0]) ** 2 + (cc - peak_rc[1]) ** 2 <= max_radius_cells**2
    return region & within


def _equivalent_radius_m(region: np.ndarray, cell_size_m: float) -> float:
    return float(np.sqrt(max(int(region.sum()), 1) * cell_size_m**2 / np.pi))


def _source_depth_m(
    signal: np.ndarray,
    peak_rc: tuple[int, int],
    cell_size_m: float,
) -> tuple[float, bool]:
    """(source depth below the survey plane, whether the grid could
    resolve it), from the width of the analytic signal over the source.

    The analytic signal is used for this rather than the anomaly itself
    for the same reason it is used for ranking: it is a single positive
    blob centred on the source, so its width can be measured without
    first estimating a local background - and a background estimate is
    exactly what goes wrong on a block with a regional gradient across
    it, or where the anomaly's own negative lobe sits in whatever ring
    the background would be taken from.

    Measured, and scale-invariant at 5/10/20/40 m depth: the
    area-equivalent radius of the blob above half the analytic signal's
    peak is 0.556 x the source depth (0.598 when the grid has only four
    cells per depth, i.e. barely resolves it).

    A blob no bigger than a cell means the grid cannot resolve the source
    at all; the depth returned is then that floor, flagged, and is an
    upper bound on how shallow the source might be rather than a
    measurement.
    """
    peak = float(signal[peak_rc])
    if not np.isfinite(peak) or peak <= 0:
        return float(cell_size_m) / AS_HALF_WIDTH_TO_DEPTH, False
    above_half = np.isfinite(signal) & (signal >= 0.5 * peak)
    labels, _ = ndimage.label(above_half)
    if labels[peak_rc] == 0:
        return float(cell_size_m) / AS_HALF_WIDTH_TO_DEPTH, False
    blob = labels == labels[peak_rc]
    half_width = _equivalent_radius_m(blob, cell_size_m)
    resolved = half_width > cell_size_m
    return max(half_width, cell_size_m) / AS_HALF_WIDTH_TO_DEPTH, resolved


def _region_polygon_xy(
    region: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    margin_m: float,
    min_radius_m: float = 0.0,
    max_radius_m: float = np.inf,
    n_vertices: int = 24,
) -> list[tuple[float, float]]:
    """A convex ring around the region's cells, pushed outward by
    `margin_m`.

    Convex rather than a traced contour: the ring is about to be used to
    select samples for removal, and a concave contour traced on a noisy
    grid can weave between neighbouring flight lines, cutting one line and
    sparing the next over the same object - which shows up afterwards as a
    stripe, exactly where a stripe is hardest to tell from a levelling
    error. Radii are taken per angular sector rather than as one bounding
    circle so an elongated source (a wall, a guardrail, a fence) keeps its
    shape instead of dragging in everything around it.

    `min_radius_m` is the floor every sector is held to. The blob is grown
    on the ranking field, and the analytic signal falls off faster than
    the anomaly it is computed from, so a region that looks generous on
    the analytic signal can still stop short of the anomaly's own negative
    lobe. The floor is what guarantees it does not.
    """
    rows, cols = np.nonzero(region)
    if rows.size == 0:
        return []
    xs, ys = easting[cols], northing[rows]
    cx, cy = float(xs.mean()), float(ys.mean())

    ang = np.arctan2(ys - cy, xs - cx)
    rad = np.hypot(xs - cx, ys - cy)
    edges = np.linspace(-np.pi, np.pi, n_vertices + 1)
    sector_half_width = (np.pi / n_vertices) * 3.0
    ring: list[tuple[float, float]] = []
    for i in range(n_vertices):
        centre = 0.5 * (edges[i] + edges[i + 1])
        # Widened sector, so every sector gets a radius even where the
        # region happens to have no cell at that exact angle.
        span = np.abs(np.angle(np.exp(1j * (ang - centre))))
        local = rad[span <= sector_half_width]
        r = float(local.max()) if local.size else float(rad.max())
        reach = min(max(r + margin_m, min_radius_m), max_radius_m)
        ring.append((cx + reach * float(np.cos(centre)), cy + reach * float(np.sin(centre))))
    return ring


def _points_in_polygon(
    polygon: list[tuple[float, float]],
    point_x: np.ndarray | None,
    point_y: np.ndarray | None,
    point_line_id: np.ndarray | None,
) -> tuple[int, int]:
    """(distinct survey lines crossing the region, samples inside it).
    (0, 0) when no point set was supplied."""
    if point_x is None or point_y is None or len(point_x) == 0 or not polygon:
        return 0, 0
    inside = MplPath(list(polygon)).contains_points(np.column_stack([point_x, point_y]))
    n_points = int(inside.sum())
    if point_line_id is None or n_points == 0:
        return 0, n_points
    lines = np.unique(np.asarray(point_line_id)[inside])
    return int(np.sum(lines >= 0)), n_points


def _support_mask(
    shape: tuple[int, int],
    easting: np.ndarray,
    northing: np.ndarray,
    point_x: np.ndarray | None,
    point_y: np.ndarray | None,
    support_radius_m: float | None,
) -> np.ndarray | None:
    """Which cells have a survey sample close enough to have measured
    them. None when there is nothing to test against - see point 3 of the
    module docstring for the real block this exists because of."""
    if support_radius_m is None or point_x is None or point_y is None or len(point_x) == 0:
        return None
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([point_x, point_y]))
    E, N = np.meshgrid(easting, northing)
    dist, _ = tree.query(np.column_stack([E.ravel(), N.ravel()]), workers=-1)
    return (dist <= support_radius_m).reshape(shape)


def find_anomaly_candidates(
    ranking: np.ndarray,
    anomaly: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    cell_size_m: float,
    n_candidates: int = 10,
    region_level_fraction: float = DEFAULT_REGION_LEVEL_FRACTION,
    region_margin_m: float | None = None,
    max_radius_m: float = 60.0,
    min_separation_m: float | None = None,
    point_x: np.ndarray | None = None,
    point_y: np.ndarray | None = None,
    point_line_id: np.ndarray | None = None,
    support_radius_m: float | None = None,
) -> list[AnomalyCandidate]:
    """The `n_candidates` strongest distinct local anomalies, each with a
    suggested removal region.

    ranking: the field peaks are ranked on (analytic signal, RTP, residual
        anomaly - see the module docstring for why the analytic signal is
        the default).
    anomaly: the anomaly grid itself, same shape - used for the reported
        peak amplitude in nT and for the half-width depth estimate, both
        of which are only meaningful on the field itself.
    region_margin_m: how far the suggested region is pushed out past the
        grown blob. None = one estimated source depth, which carries a
        region grown to the 25% contour out past the 10% one (see the
        measured falloff in the module docstring).
    support_radius_m: how far a candidate's peak may be from the nearest
        sample. None = no support test, which is only right when no point
        set is supplied at all; with one, pass 0.75 line spacings (see
        SUPPORT_RADIUS_LINE_SPACINGS and point 3 of the module docstring).
    min_separation_m: how far apart two candidates must be. None = 2.5x
        the first one's own region radius, so the negative lobe of a
        dipole (one source-depth away from its positive peak) cannot come
        back as a separate candidate from the object that produced it.

    Returns candidates strongest-first. Nothing is modified and nothing is
    removed - that is the caller's decision, via set_manual_smoothing.
    """
    if ranking.shape != anomaly.shape:
        raise ValueError("ranking and anomaly grids must have the same shape")
    # One transform for the whole scan: every candidate's depth is
    # measured from the analytic signal's width, whatever field the
    # ranking itself was done on. See _source_depth_m.
    signal = analytic_signal(anomaly, cell_size_m)

    # Rank on distance from the field's own robust centre, so one code
    # path covers a strictly-positive field (analytic signal) and a signed
    # one (residual anomaly, RTP) - and so a strong negative anomaly is as
    # findable as a strong positive one.
    finite = ranking[np.isfinite(ranking)]
    if finite.size == 0:
        return []
    centre = float(np.median(finite))
    strength = np.abs(ranking - centre)
    noise_floor = NOISE_FLOOR_K * _robust_sigma(ranking)

    max_radius_cells = max(max_radius_m / cell_size_m, 2.0)
    available = np.isfinite(strength)
    rr, cc = np.ogrid[: strength.shape[0], : strength.shape[1]]
    # "Edge" means the edge of real data: either a hole in coverage, or
    # the outer border of the grid itself. Both are where the wavenumber-
    # domain transforms have to invent what lies beyond the survey, so a
    # peak there is as likely to be the transform's own edge effect as an
    # object on the ground, and is flagged for a second look rather than
    # offered like any other candidate.
    outside_coverage = ~np.isfinite(ranking)
    outside_coverage[0, :] = outside_coverage[-1, :] = True
    outside_coverage[:, 0] = outside_coverage[:, -1] = True

    supported = _support_mask(
        strength.shape, easting, northing, point_x, point_y, support_radius_m
    )
    if supported is not None:
        available &= supported

    # Only genuine local maxima are eligible. Without this the search would
    # happily return a point part-way down the flank of the anomaly it just
    # accepted, once the peak itself had been suppressed - a second
    # "candidate" that is really the same object's tail.
    filled = np.where(np.isfinite(strength), strength, -np.inf)
    seeds = filled >= ndimage.maximum_filter(filled, size=SEED_WINDOW_CELLS, mode="nearest")
    seeds &= available & (strength > noise_floor)   # `available` already carries the support mask
    rows, cols = np.nonzero(seeds)
    order = np.argsort(-strength[rows, cols])
    seed_rc = [(int(rows[i]), int(cols[i])) for i in order]

    candidates: list[AnomalyCandidate] = []
    for peak_rc in seed_rc[:MAX_SEEDS_EXAMINED]:
        if len(candidates) >= n_candidates:
            break
        if not available[peak_rc]:
            continue
        peak_strength = float(strength[peak_rc])

        level = max(region_level_fraction * peak_strength, noise_floor)
        region = _grow_region(strength, peak_rc, level, max_radius_cells)
        if int(region.sum()) < MIN_REGION_CELLS:
            available[peak_rc] = False
            continue
        # Measure the depth at the analytic signal's own peak inside the
        # region, not at the ranking field's peak. Ranked on the residual
        # anomaly the two are a source-depth apart - the ranking peak is a
        # lobe, the analytic signal peak is over the source - and reading
        # the blob width from a point on its flank would report a source
        # several times deeper than it is.
        in_region = np.where(region & np.isfinite(signal), signal, -np.inf)
        as_rc = np.unravel_index(int(np.argmax(in_region)), signal.shape)
        if not np.isfinite(signal[as_rc]):
            as_rc = peak_rc
        depth, depth_resolved = _source_depth_m(signal, as_rc, cell_size_m)
        half_width = depth * AS_HALF_WIDTH_TO_DEPTH
        margin = region_margin_m if region_margin_m is not None else depth
        # Reach far enough that the anomaly is below a tenth of its peak
        # everywhere outside the region - otherwise the negative lobe is
        # left behind as a crater (point 2 of the module docstring).
        wanted_radius = DEPTH_TO_CLEAR_RADIUS * depth
        clear_radius = min(wanted_radius, max_radius_m)
        capped = wanted_radius > max_radius_m
        polygon = _region_polygon_xy(
            region, easting, northing, margin,
            min_radius_m=clear_radius, max_radius_m=max_radius_m,
        )
        source_cells = max(SOURCE_MAX_RADIUS_M / cell_size_m, max_radius_cells)
        source_region = _grow_region(strength, peak_rc, level, source_cells)
        source_polygon = _region_polygon_xy(
            source_region, easting, northing, cell_size_m,
            min_radius_m=min(depth, SOURCE_MAX_RADIUS_M), max_radius_m=SOURCE_MAX_RADIUS_M,
        )
        # What the polygon actually reaches - every sector of it is capped at
        # max_radius_m, so reporting the uncapped figure told the operator
        # the cut was 155 m wide when it was 60 m.
        radius = min(max(_equivalent_radius_m(region, cell_size_m) + margin, clear_radius), max_radius_m)

        in_region = anomaly[region]
        in_region = in_region[np.isfinite(in_region)]
        if in_region.size:
            lo, hi = float(in_region.min()), float(in_region.max())
            peak_value = hi if abs(hi) >= abs(lo) else lo
        else:
            peak_value = float(anomaly[peak_rc])
        polarity = "max" if peak_value >= 0 else "min"

        n_lines, n_points = _points_in_polygon(polygon, point_x, point_y, point_line_id)
        if point_x is not None and len(point_x) and n_points < MIN_SUPPORTING_POINTS:
            available[peak_rc] = False
            continue
        at_edge = bool((ndimage.binary_dilation(region, iterations=2) & outside_coverage).any())

        candidates.append(
            AnomalyCandidate(
                rank=len(candidates) + 1,
                x=float(easting[peak_rc[1]]),
                y=float(northing[peak_rc[0]]),
                ranking_value=float(ranking[peak_rc]),
                peak_anomaly_nt=peak_value,
                polarity=polarity,
                half_width_m=float(half_width),
                depth_m=float(depth),
                radius_m=float(radius),
                polygon_xy=polygon,
                source_polygon_xy=source_polygon,
                n_lines=n_lines,
                n_points=n_points,
                at_coverage_edge=at_edge,
                region_capped=capped,
                depth_resolved=depth_resolved,
            )
        )

        separation = min_separation_m if min_separation_m is not None else 2.5 * radius
        suppress_cells = max(separation, cell_size_m) / cell_size_m
        suppress = (rr - peak_rc[0]) ** 2 + (cc - peak_rc[1]) ** 2 <= suppress_cells**2
        available &= ~(suppress | region)

    return candidates
