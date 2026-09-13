"""Automatic survey-outline (display boundary) generation from the flown
lines themselves.

processing/gridding.py already caps interpolation at the convex hull of
the survey points, but a convex hull over-fills any concave footprint -
an L-shaped block's notch, a bay bitten out of one edge, the wedge
between two legs flown at an angle - and the grid then shows a smoothly
interpolated surface over ground nobody flew. The fix is a boundary that
follows the real footprint, which is what this module builds.

Method: a morphological *closing* of the flight lines.

  1. Each line becomes a LineString and is buffered by a merge radius R,
     turning it into a corridor of half-width R.
  2. Those corridors are unioned. With R larger than half the line
     spacing, neighbouring corridors overlap and fuse into one solid
     block instead of a set of parallel stripes.
  3. The union is then eroded by (R - buffer_m), shrinking it back so its
     outer edge sits exactly buffer_m outside the outermost line.

Steps 1+3 together are the classic closing operation: it bridges the
regular line-to-line spacing (which is not a data gap - it is just how a
survey is flown, and the gridder is meant to interpolate across it) while
preserving anything genuinely bigger, so a skipped line or an unflown
notch stays outside the boundary rather than being invented by the
interpolator.

R is derived from the survey's own line spacing rather than exposed as a
second knob, so the only number a user has to think about is buffer_m -
how far beyond the outermost line the map should extend.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely import unary_union
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

# Merge radius as a multiple of the estimated line spacing. 0.75 gives
# corridors of full width 1.5x the spacing, so neighbouring lines overlap
# with margin to spare (they fuse whenever 2R > spacing, i.e. anything
# above 0.5) while a gap of two or more missing lines still stays open.
_MERGE_RADIUS_LINE_SPACING_FACTOR = 0.75

# Buffer resolution: segments per quarter circle. The default (8) rounds
# line ends into 32-gons, which after the union/erosion leaves a boundary
# with far more vertices than the outline's real shape needs; 4 keeps the
# ring small enough to round-trip comfortably through the API/save file
# while staying visually smooth at survey scale.
_QUAD_SEGS = 4

# Below this many points a "line" can't define a corridor direction, so it
# is treated as a plain buffered point instead of a LineString.
_MIN_POINTS_FOR_LINESTRING = 2

# Douglas-Peucker tolerance, as a fraction of buffer_m and clamped to the
# range below. Buffering a flight line keeps one output vertex per input
# sample, so an unsimplified boundary over a real survey came out with
# ~10,000 vertices - a ring that then rides along in every process
# summary, save file and export, and gets point-in-polygon tested per
# grid cell. Simplifying at 0.05x the buffer cut a measured 17,931-point
# survey from 10,128 vertices to 103 (99% fewer) for a 0.01% area change,
# which is far below the buffer distance itself and so invisible on the
# map. Applied to the input lines as well as the final ring: the lines
# are what make the corridor complex in the first place, and simplifying
# them first is much cheaper than simplifying the union afterwards.
_SIMPLIFY_BUFFER_FRACTION = 0.05
_SIMPLIFY_MIN_M = 0.25
_SIMPLIFY_MAX_M = 2.0


def _simplify_tolerance(buffer_m: float) -> float:
    return float(np.clip(buffer_m * _SIMPLIFY_BUFFER_FRACTION, _SIMPLIFY_MIN_M, _SIMPLIFY_MAX_M))


@dataclass
class BoundaryResult:
    """rings_xy: the boundary as one or more exterior rings, each (N, 2) in
    the same local projected metres the inputs were given in, largest
    first. Usually one ring; a survey whose lines fall into blocks farther
    apart than the merge radius (a skipped line, or two separate areas in
    one project) yields one ring per block, all of them kept - dropping
    the smaller blocks would silently hide flown data from the grid."""

    rings_xy: list[np.ndarray]
    buffer_m: float
    merge_radius_m: float
    area_m2: float
    # Interior holes discarded because a ring can't carry them. Unflown
    # pockets *inside* a block: the grid's own local-gap mask
    # (processing/gridding.py::_resolve_auto_mask) already blanks those
    # cells, so dropping the hole here does not resurrect interpolated
    # values there - it only means the boundary outline doesn't trace
    # around them.
    dropped_holes: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def n_parts(self) -> int:
        return len(self.rings_xy)


def _line_geometries(x: np.ndarray, y: np.ndarray, line_id: np.ndarray, simplify_m: float) -> list:
    """One geometry per survey line, in the order the points were flown,
    thinned to simplify_m (see _SIMPLIFY_BUFFER_FRACTION)."""
    geoms = []
    for lid in np.unique(line_id):
        if lid < 0:
            continue
        sel = line_id == lid
        pts = np.column_stack([x[sel], y[sel]])
        # Consecutive duplicate positions (a hovering sample) make a
        # zero-length segment that shapely rejects.
        if len(pts) >= 2:
            keep = np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]
            pts = pts[keep]
        if len(pts) >= _MIN_POINTS_FOR_LINESTRING:
            geoms.append(LineString(pts).simplify(simplify_m))
        elif len(pts) == 1:
            geoms.append(Point(pts[0]))
    return geoms


def auto_survey_boundary(
    x: np.ndarray,
    y: np.ndarray,
    line_id: np.ndarray,
    buffer_m: float = 10.0,
    line_spacing_m: float | None = None,
) -> BoundaryResult:
    """Outline of the actually-flown area, buffer_m outside the outermost
    line - see the module docstring for the closing operation used.

    x/y are local projected metres, line_id assigns each point to a survey
    line (negative = excluded point, skipped). line_spacing_m sets the
    merge radius; without it (a single line, or an unestimatable spacing)
    the corridors are merged at buffer_m only, which for a multi-line
    survey would leave stripes - so that case is reported as a warning.
    """
    if buffer_m <= 0:
        raise ValueError("버퍼 거리(buffer_m)는 0보다 커야 합니다.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    line_id = np.asarray(line_id)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, line_id = x[finite], y[finite], line_id[finite]

    simplify_m = _simplify_tolerance(buffer_m)
    geoms = _line_geometries(x, y, line_id, simplify_m)
    if not geoms:
        raise ValueError("경계를 만들 측선 자료가 없습니다 (활성 측선이 0개입니다).")

    warnings: list[str] = []
    if line_spacing_m and line_spacing_m > 0:
        merge_radius = max(_MERGE_RADIUS_LINE_SPACING_FACTOR * line_spacing_m, buffer_m)
    else:
        merge_radius = buffer_m
        if len(geoms) > 1:
            warnings.append(
                "측선 간격을 추정하지 못해 버퍼 거리만으로 측선을 이었습니다 - "
                "측선 사이가 버퍼의 2배보다 넓으면 경계가 측선별로 갈라질 수 있습니다."
            )

    # 1+2: corridors of half-width merge_radius, fused into one block.
    merged = unary_union([g.buffer(merge_radius, quad_segs=_QUAD_SEGS) for g in geoms])
    # 3: erode back so the outer edge sits buffer_m from the lines.
    if merge_radius > buffer_m:
        merged = merged.buffer(-(merge_radius - buffer_m), quad_segs=_QUAD_SEGS)

    if merged.is_empty:
        raise ValueError("경계를 만들지 못했습니다 (버퍼 거리를 조정해 보세요).")

    parts = list(merged.geoms) if isinstance(merged, MultiPolygon) else [merged]
    parts = [p for p in parts if isinstance(p, Polygon) and not p.is_empty]
    if not parts:
        raise ValueError("경계를 만들지 못했습니다 (버퍼 거리를 조정해 보세요).")

    parts.sort(key=lambda p: p.area, reverse=True)
    # preserve_topology keeps the simplified ring valid (no self-crossing
    # where the outline doubles back on itself at a line end).
    parts = [p.simplify(simplify_m, preserve_topology=True) for p in parts]
    parts = [p for p in parts if isinstance(p, Polygon) and not p.is_empty]
    if not parts:
        raise ValueError("경계를 만들지 못했습니다 (버퍼 거리를 조정해 보세요).")
    dropped_holes = sum(len(p.interiors) for p in parts)

    if len(parts) > 1:
        warnings.append(
            f"측선이 서로 떨어진 {len(parts)}개 구역으로 나뉘어 있어 경계도 {len(parts)}개로 만들었습니다 - "
            "측선 간격보다 훨씬 넓은 빈 구간(결측 측선 등)이 있다는 뜻이니, "
            "하나로 이으려면 버퍼 거리를 늘리세요."
        )
    if dropped_holes:
        warnings.append(
            f"경계 안쪽의 빈 구역(구멍) {dropped_holes}개는 경계선에서 제외했습니다 - "
            "해당 셀은 그리딩 단계의 자동 결측 마스크가 이미 비워두므로 값이 새로 만들어지지는 않습니다."
        )

    return BoundaryResult(
        rings_xy=[np.asarray(p.exterior.coords, dtype=float) for p in parts],
        buffer_m=float(buffer_m),
        merge_radius_m=float(merge_radius),
        area_m2=float(sum(p.area for p in parts)),
        dropped_holes=dropped_holes,
        warnings=warnings,
    )
