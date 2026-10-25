"""Magnetic contact detection: vectorized geological-contact/rock-boundary
lines from total horizontal derivative (THDR) ridge maxima that persist
across multiple upward-continued heights ("worming" - Archibald et al.,
1999; see also multiscale_edges.py, which produces the raw per-height edge
point cloud this module clusters).

This is deliberately a different technique from lineaments.py, not a
re-skin of it: lineaments.py finds ridge maxima at a *single* grid/height
and reports structural-trend statistics (strike, rose diagram) - useful
for fault/shear-zone interpretation. A magnetic *contact* (the boundary
between two differently-magnetized rock units) is instead identified by
requiring a ridge to persist across several upward-continuation heights,
which is what separates a genuine, laterally-continuous contact from a
single-height ridge caused by shallow noise or a small discrete source
(this was Archibald et al.'s original motivation for worming). The result
is meant to be overlaid on an uploaded geological-map reference layer (see
store.py's reference-layer support) to compare the interpreted contacts
against the mapped geology.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

from .multiscale_edges import run_multiscale_edges

_MIN_EDGE_POINTS = 10
_DEFAULT_HEIGHTS_M = (0.0, 20.0, 40.0, 60.0, 80.0)
_DEFAULT_MAX_GAP_CELLS = 3.0


@dataclass
class ContactSegment:
    points_latlon: list[tuple[float, float]]
    length_m: float
    midpoint_latlon: tuple[float, float]
    n_points: int
    mean_persistence: float
    mean_thdr_value: float


def _cluster_by_proximity(xs: np.ndarray, ys: np.ndarray, max_gap_m: float) -> list[np.ndarray]:
    """Union-find grouping of points connected by a chain of neighbours
    within max_gap_m - see lineaments.py for why proximity (not raster
    adjacency) is needed to reconnect a ridge crest into one segment."""
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


def detect_magnetic_contacts(
    grid_values: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    cell_size_m: float,
    utm_epsg: int,
    heights_m: tuple[float, ...] = _DEFAULT_HEIGHTS_M,
    percentile_threshold: float = 80.0,
    min_persistence: float = 0.5,
    min_segment_points: int = 4,
    min_length_m: float = 0.0,
    max_gap_cells: float = _DEFAULT_MAX_GAP_CELLS,
) -> dict:
    """Detect persistent-ridge contact segments. available=False (with a
    reason) when the grid is too small, no ridge clears the percentile
    threshold at the base height, or nothing survives the persistence/
    length/point-count filters - a real result, not a bug to silently
    paper over (e.g. a flat survey with no sharp magnetic boundaries)."""
    heights = sorted({float(h) for h in heights_m} | {0.0})
    edge_points = run_multiscale_edges(
        grid_values, easting, northing, cell_size_m, utm_epsg, heights, percentile_threshold
    )
    if len(edge_points) < _MIN_EDGE_POINTS:
        return {
            "available": False,
            "reason": "충분한 능선(ridge)점이 없습니다 - 백분위수를 낮추거나 그리드가 충분히 큰지 확인하세요.",
            "contacts": [],
        }

    base_height = heights[0]
    base_points = [p for p in edge_points if p.height_m == base_height]
    if len(base_points) < _MIN_EDGE_POINTS:
        return {
            "available": False,
            "reason": "기준 고도(0m)에서 충분한 능선점이 없습니다.",
            "contacts": [],
        }

    to_local = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    to_latlon = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

    base_x, base_y = to_local.transform([p.lon for p in base_points], [p.lat for p in base_points])
    base_x, base_y = np.asarray(base_x), np.asarray(base_y)
    base_vals = np.array([p.thdr_value for p in base_points])

    other_heights = heights[1:]
    match_radius_m = max_gap_cells * cell_size_m
    n_present = np.zeros(len(base_points), dtype=int)
    for h in other_heights:
        h_points = [p for p in edge_points if p.height_m == h]
        if not h_points:
            continue
        hx, hy = to_local.transform([p.lon for p in h_points], [p.lat for p in h_points])
        tree = cKDTree(np.column_stack([np.asarray(hx), np.asarray(hy)]))
        dist, _ = tree.query(np.column_stack([base_x, base_y]), workers=-1)
        n_present += (dist <= match_radius_m).astype(int)

    persistence = n_present / len(other_heights) if other_heights else np.ones(len(base_points))
    persistent_mask = persistence >= min_persistence
    if persistent_mask.sum() < _MIN_EDGE_POINTS:
        return {
            "available": False,
            "reason": (
                f"{len(heights)}개 고도에서 지속적으로(영속성 {min_persistence * 100:.0f}% 이상) 나타나는 능선점이 "
                f"{int(persistent_mask.sum())}개뿐입니다 - 영속성 기준이나 백분위수를 낮춰보세요."
            ),
            "contacts": [],
        }

    px, py = base_x[persistent_mask], base_y[persistent_mask]
    pvals = base_vals[persistent_mask]
    ppersist = persistence[persistent_mask]

    groups = _cluster_by_proximity(px, py, max_gap_cells * cell_size_m)
    contacts: list[ContactSegment] = []
    for group_idx in groups:
        if group_idx.size < min_segment_points:
            continue
        xs, ys = px[group_idx], py[group_idx]
        pts = np.column_stack([xs, ys])
        centered = pts - pts.mean(axis=0)
        if np.allclose(centered, 0):
            continue
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, int(np.argmax(eigvals))]
        proj = centered @ direction
        order = np.argsort(proj)
        xs_sorted, ys_sorted = xs[order], ys[order]

        length_m = float(np.sum(np.hypot(np.diff(xs_sorted), np.diff(ys_sorted))))
        if length_m < min_length_m:
            continue

        lons, lats = to_latlon.transform(xs_sorted, ys_sorted)
        mid_idx = len(xs_sorted) // 2
        contacts.append(
            ContactSegment(
                points_latlon=list(zip([float(v) for v in lats], [float(v) for v in lons])),
                length_m=length_m,
                midpoint_latlon=(float(lats[mid_idx]), float(lons[mid_idx])),
                n_points=int(group_idx.size),
                mean_persistence=float(np.mean(ppersist[group_idx])),
                mean_thdr_value=float(np.mean(pvals[group_idx])),
            )
        )

    if not contacts:
        return {
            "available": False,
            "reason": "최소 길이/포인트 수 조건을 만족하는 접촉면 구간이 없습니다 - 조건을 완화해보세요.",
            "contacts": [],
        }

    contacts.sort(key=lambda c: -c.length_m)
    return {
        "available": True,
        "n_contacts": len(contacts),
        "total_length_m": float(sum(c.length_m for c in contacts)),
        "heights_used_m": heights,
        "contacts": [
            {
                "points_latlon": c.points_latlon,
                "length_m": c.length_m,
                "midpoint_latlon": c.midpoint_latlon,
                "n_points": c.n_points,
                "mean_persistence": c.mean_persistence,
                "mean_thdr_value": c.mean_thdr_value,
            }
            for c in contacts
        ],
    }
