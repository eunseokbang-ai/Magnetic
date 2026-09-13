"""Magnetic lineament extraction: vectorized structural trend lines from
ridge maxima of a derivative grid (THD/analytic signal/tilt angle/1VD),
plus rose-diagram direction statistics - the mineral-exploration
interpretation step beyond multiscale_edges.py's raw edge *point cloud*
(same underlying "ridge = local maxima above a threshold" idea, at a
single grid rather than a series of upward-continued heights, but here
neighbouring ridge points are grouped into connected lineament segments
with an along-strike direction and length, matching how a geologist reads
a THD/ASA map for fault/shear-zone/contact trends: unlabeled anomaly
pixels are not the deliverable, the strike statistics are.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

_MIN_FINITE_CELLS = 20
# How close two ridge-maxima *points* (not raster cells) need to be to
# count as the same lineament, as a multiple of cell size. Plain raster
# 8-connectivity (adjacent-pixel) grouping does not work here: a strict
# per-pixel local-maximum test almost never accepts two literally-
# adjacent cells at once (float noise breaks the tie), so a ridge crest
# that is visually one continuous line comes out as isolated single-cell
# "components" - the same reason multiscale_edges.py documents its own
# output as a scattered point cloud rather than traced ridges. Grouping by
# proximity instead (union-find over a distance threshold, the same
# primitive processing/line_grouping.py already uses to group flight
# passes) reconnects those pixel-gapped maxima into one segment.
_DEFAULT_MAX_GAP_CELLS = 2.5

# The 4 named structural-trend quadrants requested for the summary,
# centered on their compass strike (mod 180 - a lineament's strike is
# axial, "NE-SW" already implies both NE and SW ends of the same line).
_QUADRANTS = [
    ("N-S", 0.0),
    ("NE-SW", 45.0),
    ("E-W", 90.0),
    ("NW-SE", 135.0),
]


def _local_maxima_mask(grid: np.ndarray) -> np.ndarray:
    """True where a finite cell is >= all 8 finite neighbours."""
    padded = np.pad(grid, 1, mode="constant", constant_values=-np.inf)
    is_max = np.ones(grid.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor = padded[1 + dr : 1 + dr + grid.shape[0], 1 + dc : 1 + dc + grid.shape[1]]
            is_max &= grid >= neighbor
    return is_max & np.isfinite(grid)


@dataclass
class Lineament:
    points_latlon: list[tuple[float, float]]  # ordered along-strike vertices
    length_m: float
    strike_deg: float  # compass azimuth mod 180 (0=N-S, 90=E-W)
    midpoint_latlon: tuple[float, float]
    n_points: int
    peak_value: float


def _cluster_by_proximity(xs: np.ndarray, ys: np.ndarray, max_gap_m: float) -> list[np.ndarray]:
    """Union-find grouping of points that are within max_gap_m of some
    chain of other points in the same group (i.e. connected components of
    the "within max_gap_m" graph) - see _DEFAULT_MAX_GAP_CELLS for why
    this, not raster adjacency, is what actually reconnects a ridge."""
    n = len(xs)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree = cKDTree(np.column_stack([xs, ys]))
    for a, b in tree.query_pairs(r=max_gap_m):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.array(idxs) for idxs in groups.values()]


def _nearest_quadrant(strike_deg: float) -> str:
    # circular distance on a mod-180 axis
    best_name, best_dist = _QUADRANTS[0][0], 999.0
    for name, center in _QUADRANTS:
        d = abs(strike_deg - center)
        d = min(d, 180.0 - d)
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


def extract_lineaments(
    grid_values: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    cell_size_m: float,
    utm_epsg: int,
    percentile_threshold: float = 90.0,
    min_segment_points: int = 4,
    min_length_m: float = 0.0,
    rose_bin_width_deg: float = 10.0,
    max_gap_cells: float = _DEFAULT_MAX_GAP_CELLS,
) -> dict:
    """Extract ridge-maxima connected components from `grid_values` (a
    THD/analytic-signal/tilt/1VD grid the caller has already computed) as
    vectorized lineament segments, plus rose-diagram bin statistics.

    available=False (with a reason) when there isn't enough finite data or
    no ridge segment survives the min_segment_points/min_length_m
    filters - a real, if unwelcome, result (e.g. a very small or very
    smooth survey), not a bug to work around silently."""
    finite_mask = np.isfinite(grid_values)
    if finite_mask.sum() < _MIN_FINITE_CELLS:
        return {"available": False, "reason": "유효한 격자 셀이 너무 적습니다.", "lineaments": []}

    finite_vals = grid_values[finite_mask]
    threshold = float(np.percentile(finite_vals, percentile_threshold))
    ridge_mask = _local_maxima_mask(np.where(finite_mask, grid_values, -np.inf)) & (grid_values >= threshold)
    if not ridge_mask.any():
        return {"available": False, "reason": "임계값을 넘는 능선(ridge) 점이 없습니다 - 백분위수를 낮춰보세요.", "lineaments": []}

    x2d, y2d = np.meshgrid(easting, northing)
    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

    ridge_rows, ridge_cols = np.nonzero(ridge_mask)
    ridge_x, ridge_y = x2d[ridge_rows, ridge_cols], y2d[ridge_rows, ridge_cols]
    groups = _cluster_by_proximity(ridge_x, ridge_y, max_gap_cells * cell_size_m)
    if not groups:
        return {"available": False, "reason": "연결된 능선 구간을 찾지 못했습니다.", "lineaments": []}

    lineaments: list[Lineament] = []
    for group_idx in groups:
        rows, cols = ridge_rows[group_idx], ridge_cols[group_idx]
        if rows.size < min_segment_points:
            continue
        xs, ys = x2d[rows, cols], y2d[rows, cols]
        pts = np.column_stack([xs, ys])
        centered = pts - pts.mean(axis=0)

        if rows.size < 2 or np.allclose(centered, 0):
            continue
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, int(np.argmax(eigvals))]
        # Order points along the ridge by their position on the principal
        # axis - the same "PCA projection sort" already used to order
        # points along a flight line (see processing/line_grouping.py) -
        # adequate for a ridge, which is locally line-like even if it
        # curves gently over its length.
        proj = centered @ direction
        order = np.argsort(proj)
        xs_sorted, ys_sorted = xs[order], ys[order]

        length_m = float(np.sum(np.hypot(np.diff(xs_sorted), np.diff(ys_sorted))))
        if length_m < min_length_m:
            continue

        # Compass azimuth (0=N, 90=E), mod 180 since a line's strike is
        # axial - "pointing NE" and "pointing SW" describe the same trend.
        dx, dy = direction
        strike = float(np.degrees(np.arctan2(dx, dy))) % 180.0

        lons, lats = transformer.transform(xs_sorted, ys_sorted)
        mid_idx = len(xs_sorted) // 2
        values = grid_values[rows, cols]
        lineaments.append(
            Lineament(
                points_latlon=list(zip([float(v) for v in lats], [float(v) for v in lons])),
                length_m=length_m,
                strike_deg=strike,
                midpoint_latlon=(float(lats[mid_idx]), float(lons[mid_idx])),
                n_points=int(rows.size),
                peak_value=float(np.max(values)),
            )
        )

    if not lineaments:
        return {
            "available": False,
            "reason": "최소 길이/포인트 수 조건을 만족하는 리니어먼트가 없습니다 - 조건을 완화해보세요.",
            "lineaments": [],
        }

    # length-weighted, since a long throughgoing structure should count
    # for more of the survey's overall structural grain than a 2-cell
    # speck that happened to clear the ridge threshold.
    n_bins = max(1, int(round(180.0 / rose_bin_width_deg)))
    bin_edges = np.linspace(0.0, 180.0, n_bins + 1)
    bin_length = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)
    for l in lineaments:
        b = min(int(l.strike_deg // rose_bin_width_deg), n_bins - 1)
        bin_length[b] += l.length_m
        bin_count[b] += 1

    total_length = float(sum(l.length_m for l in lineaments))
    rose_bins = [
        {
            "strike_from_deg": float(bin_edges[i]),
            "strike_to_deg": float(bin_edges[i + 1]),
            "n_lineaments": int(bin_count[i]),
            "total_length_m": float(bin_length[i]),
        }
        for i in range(n_bins)
    ]

    quadrant_length: dict[str, float] = {name: 0.0 for name, _ in _QUADRANTS}
    quadrant_count: dict[str, int] = {name: 0 for name, _ in _QUADRANTS}
    for l in lineaments:
        q = _nearest_quadrant(l.strike_deg)
        quadrant_length[q] += l.length_m
        quadrant_count[q] += 1
    quadrant_summary = [
        {
            "name": name,
            "n_lineaments": quadrant_count[name],
            "total_length_m": quadrant_length[name],
            "pct_of_total_length": (100.0 * quadrant_length[name] / total_length) if total_length > 0 else 0.0,
        }
        for name, _ in _QUADRANTS
    ]

    lineaments.sort(key=lambda l: -l.length_m)
    return {
        "available": True,
        "n_lineaments": len(lineaments),
        "total_length_m": total_length,
        "threshold_value": threshold,
        "lineaments": [
            {
                "points_latlon": l.points_latlon,
                "length_m": l.length_m,
                "strike_deg": l.strike_deg,
                "midpoint_latlon": l.midpoint_latlon,
                "n_points": l.n_points,
                "peak_value": l.peak_value,
            }
            for l in lineaments
        ],
        "rose_bins": rose_bins,
        "rose_bin_width_deg": rose_bin_width_deg,
        "quadrant_summary": quadrant_summary,
    }
