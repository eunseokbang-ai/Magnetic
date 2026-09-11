"""Groups flight lines that follow the same physical ground track,
regardless of which direction they were flown - the shared primitive
behind both processing/repeatability.py (a dedicated repeatability-test
flight, analysed for system noise) and processing/duplicate_lines.py (the
main survey, checked for accidentally/deliberately re-flown lines so the
better-quality pass can be kept and the other excluded).

Each line's principal spatial axis is found via PCA and canonicalized to a
consistent sign convention, so two passes over the same track land on the
same axis regardless of flight direction; lines are then union-find
grouped by (small angle between canonical axes) AND (small perpendicular
offset between centroids projected onto that axis) - two independent
knobs callers tune to their own definition of "the same track" (a hand-
flown repeatability box allows more drift than an accidental duplicate of
an autopilot-guided survey line).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_POINTS_PER_PASS = 10


@dataclass
class LinePass:
    line_id: int
    point_ids: np.ndarray  # time-ordered, aligned with x/y/value below
    x: np.ndarray  # time-ordered
    y: np.ndarray  # time-ordered
    value: np.ndarray  # time-ordered
    centroid: np.ndarray
    # The pass's principal spatial axis, canonicalized to a consistent sign
    # convention (see _canonical_direction) - identical for two passes
    # along the same physical track regardless of which way either was
    # flown, which is exactly what's needed to group and spatially align
    # them. NOT usable on its own to tell forward from reverse - that's
    # what flight_sign is for.
    canonical_direction: np.ndarray
    # +1 if this pass moved (in time) in the same sense as
    # canonical_direction, -1 if opposite - i.e. actual flight direction,
    # independent of the arbitrary eigenvector sign PCA returns.
    flight_sign: int


def _canonical_direction(direction: np.ndarray) -> np.ndarray:
    flip = direction[0] < 0 or (direction[0] == 0 and direction[1] < 0)
    return -direction if flip else direction


def line_pass(line_id: int, group: pd.DataFrame, value_col: str = "value") -> LinePass | None:
    ordered = group.sort_values("timestamp")
    if len(ordered) < MIN_POINTS_PER_PASS:
        return None
    point_ids = ordered["point_id"].to_numpy() if "point_id" in ordered.columns else ordered.index.to_numpy()
    x, y, v = ordered["x"].to_numpy(), ordered["y"].to_numpy(), ordered[value_col].to_numpy()
    pts = np.column_stack([x, y])
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    raw_direction = eigvecs[:, int(np.argmax(eigvals))]
    canonical_direction = _canonical_direction(raw_direction)

    temporal_displacement = pts[-1] - pts[0]
    flight_sign = 1 if float(np.dot(temporal_displacement, canonical_direction)) >= 0 else -1

    return LinePass(
        line_id=line_id, point_ids=point_ids, x=x, y=y, value=v, centroid=centroid,
        canonical_direction=canonical_direction, flight_sign=flight_sign,
    )


def group_passes(passes: list[LinePass], perp_tolerance_m: float, angle_tolerance_deg: float) -> list[list[LinePass]]:
    """Union-find groups of >=2 passes judged to be the same physical
    track: within angle_tolerance_deg of each other's canonical direction,
    and within perp_tolerance_m of each other's centroid measured
    perpendicular to that shared direction."""
    n = len(passes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            # Both directions are already canonicalized, so two passes on
            # the same physical track (regardless of flight direction)
            # should point the *same* way here - no need to fold the angle.
            cos_angle = float(np.clip(np.dot(passes[i].canonical_direction, passes[j].canonical_direction), -1.0, 1.0))
            angle_deg = np.degrees(np.arccos(cos_angle))
            if angle_deg > angle_tolerance_deg:
                continue
            avg_dir = passes[i].canonical_direction + passes[j].canonical_direction
            norm = np.linalg.norm(avg_dir)
            if norm < 1e-9:
                continue
            avg_dir = avg_dir / norm
            perp = np.array([-avg_dir[1], avg_dir[0]])
            offset = float(np.dot(passes[j].centroid - passes[i].centroid, perp))
            if abs(offset) <= perp_tolerance_m:
                union(i, j)

    groups: dict[int, list[LinePass]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(passes[i])
    return [g for g in groups.values() if len(g) >= 2]
