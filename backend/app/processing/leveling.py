"""Heading (flight-direction) leveling correction.

Without a calibration (figure-8) flight, a systematic heading-dependent
bias between forward and reverse-flown lines can still be estimated: find
nearby point pairs on adjacent, opposite-heading lines, keep the pairs in
"quiet" (low local variability) areas where real geology is unlikely to
explain the difference, and take the robust central difference as the
forward/reverse offset. Half the offset is then applied to each group so
the overall field level is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .lines import cross_track_coordinate


@dataclass
class HeadingLevelingResult:
    applied: bool
    reason: str | None
    offset_nt: float | None
    n_matched_pairs: int
    n_quiet_pairs: int
    line_groups: dict = field(default_factory=dict)  # line_id -> "A" | "B"
    line_shifts: dict = field(default_factory=dict)  # line_id -> applied shift (nT)


def _line_true_heading_deg(sub: pd.DataFrame) -> float:
    """Net travel direction (0-360, atan2(dy,dx) convention) from first to
    last point of a line - unlike lines.heading_deg this is NOT folded to
    mod 180, so forward vs. reverse direction can be told apart."""
    x0, y0 = sub["x"].iloc[0], sub["y"].iloc[0]
    x1, y1 = sub["x"].iloc[-1], sub["y"].iloc[-1]
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 360.0)


def _local_variability(values: np.ndarray, k: int) -> np.ndarray:
    """Rolling (max-min) over a window of k points, used as a cheap proxy
    for "how much real signal is changing nearby" - low values = quiet."""
    n = len(values)
    half = max(1, k // 2)
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = values[lo:hi]
        out[i] = window.max() - window.min()
    return out


def compute_heading_correction(
    df: pd.DataFrame,
    value_col: str,
    dominant_azimuth_deg: float,
    line_spacing_m: float | None,
    quiet_percentile: float = 40.0,
    max_match_distance_m: float | None = None,
    k_neighbors: int = 5,
) -> HeadingLevelingResult:
    kept = df[df["line_id"] >= 0]
    line_ids = sorted(kept["line_id"].unique())
    if len(line_ids) < 2:
        return HeadingLevelingResult(False, "측선이 2개 미만이라 헤딩 보정을 계산할 수 없습니다.", None, 0, 0)

    groups: dict[int, str] = {}
    for lid in line_ids:
        sub = kept[kept["line_id"] == lid]
        heading = _line_true_heading_deg(sub)
        diff = (heading - dominant_azimuth_deg + 180.0) % 360.0 - 180.0
        groups[lid] = "A" if abs(diff) <= 90.0 else "B"

    if len(set(groups.values())) < 2:
        return HeadingLevelingResult(
            False, "모든 측선이 같은 방향으로 비행되어 헤딩 보정을 계산할 수 없습니다 (반대 방향 측선 필요).", None, 0, 0, groups
        )

    centroids = kept.groupby("line_id")[["x", "y"]].mean()
    cross = cross_track_coordinate(centroids["x"].to_numpy(), centroids["y"].to_numpy(), dominant_azimuth_deg)
    order = centroids.index.to_numpy()[np.argsort(cross)]

    # the nearest point on an adjacent line is at least ~line_spacing_m away
    # (that's the perpendicular gap between lines), so the match radius must
    # comfortably exceed it or no cross-line pairs will ever match.
    match_dist = max_match_distance_m or (1.5 * line_spacing_m if line_spacing_m else 100.0)

    diffs: list[float] = []
    quietness: list[float] = []
    n_matched = 0

    for i in range(len(order) - 1):
        lid_a, lid_b = order[i], order[i + 1]
        if groups[lid_a] == groups[lid_b]:
            continue
        sub_a = kept[kept["line_id"] == lid_a].reset_index(drop=True)
        sub_b = kept[kept["line_id"] == lid_b].reset_index(drop=True)
        if len(sub_a) < k_neighbors or len(sub_b) < k_neighbors:
            continue

        tree_b = cKDTree(sub_b[["x", "y"]].to_numpy())
        dist, idx_b = tree_b.query(sub_a[["x", "y"]].to_numpy())
        valid = dist <= match_dist
        if not valid.any():
            continue

        matched_a_idx = np.where(valid)[0]
        matched_b_idx = idx_b[valid]
        n_matched += len(matched_a_idx)

        val_a = sub_a[value_col].to_numpy()
        val_b = sub_b[value_col].to_numpy()
        quiet_a = _local_variability(val_a, k_neighbors)
        quiet_b = _local_variability(val_b, k_neighbors)

        # normalize sign so diff always means (group-A value - group-B value)
        sign = 1.0 if groups[lid_a] == "A" else -1.0
        pair_diff = (val_a[matched_a_idx] - val_b[matched_b_idx]) * sign
        pair_quiet = np.maximum(quiet_a[matched_a_idx], quiet_b[matched_b_idx])

        diffs.extend(pair_diff.tolist())
        quietness.extend(pair_quiet.tolist())

    if not diffs:
        return HeadingLevelingResult(
            False, "인접한 반대방향 측선 쌍에서 매칭되는 지점을 찾지 못해 헤딩 보정을 계산할 수 없습니다.", None, 0, 0, groups
        )

    diffs_arr = np.array(diffs)
    quiet_arr = np.array(quietness)
    threshold = np.percentile(quiet_arr, quiet_percentile)
    quiet_mask = quiet_arr <= threshold
    if not quiet_mask.any():
        quiet_mask = np.ones_like(quiet_arr, dtype=bool)

    offset = float(np.median(diffs_arr[quiet_mask]))
    shifts = {lid: (-offset / 2.0 if groups[lid] == "A" else offset / 2.0) for lid in line_ids}

    return HeadingLevelingResult(
        applied=True,
        reason=None,
        offset_nt=offset,
        n_matched_pairs=n_matched,
        n_quiet_pairs=int(quiet_mask.sum()),
        line_groups=groups,
        line_shifts=shifts,
    )


def apply_heading_correction(df: pd.DataFrame, value_col: str, result: HeadingLevelingResult) -> np.ndarray:
    values = df[value_col].to_numpy(copy=True)
    if not result.applied or not result.line_shifts:
        return values
    shift = df["line_id"].map(result.line_shifts).fillna(0.0).to_numpy()
    return values + shift
