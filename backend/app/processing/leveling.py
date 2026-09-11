"""Heading (flight-direction) leveling correction.

Without a calibration (figure-8) flight, a systematic heading-dependent
bias between forward and reverse-flown lines can still be estimated from
the survey's own data. Two methods are implemented:

"local_plane" (default) - the neighbourhood method. The survey is tiled
with overlapping circular neighbourhoods a couple of line spacings
across. Inside each one, the field is modelled as a plane plus a single
forward/reverse step: value = a + b*dx + c*dy + offset*group. The plane
absorbs whatever smooth geological gradient runs through that patch, so
the fitted step is what is left over once the local geology is accounted
for, and how well the model fits (its residual scatter) is a direct
measure of whether that patch really was quiet and uniform - the
neighbourhoods with real structure or a ground object in them fit badly
and are dropped, rather than being guessed at. The survey-wide offset is
the robust centre of the steps from the patches that survive.

Fitting the gradient and the step together is what makes this work: a
neighbourhood must contain at least three distinct lines for the two to
be separable at all (with one line per direction, "the field rises
across the gap" and "the reverse line reads high" are the same number
twice), which is checked per neighbourhood.

"nearest_pair" - the original method, kept for comparison. Matches each
point on a line to the nearest point on the adjacent opposite-heading
line and takes the robust median difference over the pairs in the
quietest areas. Its weakness is that "nearest" is a whole line spacing
away, so every pair difference carries the real cross-line geological
change at full strength, and the quiet test it screens with measures
variability *along* the lines rather than across the gap the difference
is actually taken over.

Either way, half the offset is applied to each direction group so the
overall field level is unchanged.

Note that this corrects a single offset shared by the whole survey. A
level error that differs from line to line - the usual cause of visible
striping - is not a heading effect and needs
processing/statistical_leveling.py instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .lines import cross_track_coordinate

# A neighbourhood needs at least this many distinct survey lines for the
# local gradient and the forward/reverse step to be separable - see the
# module docstring.
_MIN_LINES_PER_NEIGHBORHOOD = 3
# ...and this many points from each direction group, so neither side's
# contribution rests on a couple of samples.
_MIN_POINTS_PER_GROUP = 12


@dataclass
class HeadingLevelingResult:
    applied: bool
    reason: str | None
    offset_nt: float | None
    n_matched_pairs: int
    n_quiet_pairs: int
    line_groups: dict = field(default_factory=dict)  # line_id -> "A" | "B"
    line_shifts: dict = field(default_factory=dict)  # line_id -> applied shift (nT)
    method: str = "nearest_pair"
    # local_plane only: spread of the per-neighbourhood offsets that were
    # kept. A real heading effect is a property of the instrument, so it
    # should come out much the same everywhere; a spread comparable to the
    # offset itself means the number is not measuring a heading effect.
    offset_spread_nt: float | None = None
    # local_plane only: typical residual scatter (nT) of the fitted
    # plane+step model over the neighbourhoods that were kept.
    residual_nt: float | None = None
    warnings: list[str] = field(default_factory=list)


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


def _direction_groups(kept: pd.DataFrame, dominant_azimuth_deg: float) -> dict[int, str]:
    """Split the lines into the two flight directions: "A" for lines flown
    roughly along the dominant azimuth, "B" for the ones flown back."""
    groups: dict[int, str] = {}
    for lid in sorted(kept["line_id"].unique()):
        sub = kept[kept["line_id"] == lid]
        heading = _line_true_heading_deg(sub)
        diff = (heading - dominant_azimuth_deg + 180.0) % 360.0 - 180.0
        groups[int(lid)] = "A" if abs(diff) <= 90.0 else "B"
    return groups


def compute_heading_correction(
    df: pd.DataFrame,
    value_col: str,
    dominant_azimuth_deg: float,
    line_spacing_m: float | None,
    quiet_percentile: float = 40.0,
    max_match_distance_m: float | None = None,
    k_neighbors: int = 5,
    method: str = "local_plane",
    neighborhood_radius_factor: float = 2.0,
) -> HeadingLevelingResult:
    """method: "local_plane" (default) or "nearest_pair" - see the module
    docstring. neighborhood_radius_factor applies to "local_plane" only:
    the neighbourhood radius as a multiple of the line spacing. Bigger
    patches average over more lines (steadier) but make the "the geology
    here is a plane" assumption harder to hold."""
    if method not in ("local_plane", "nearest_pair"):
        raise ValueError(f"알 수 없는 헤딩 보정 방법입니다: {method} (local_plane 또는 nearest_pair)")
    if method == "local_plane":
        return _heading_correction_local_plane(
            df, value_col, dominant_azimuth_deg, line_spacing_m,
            quiet_percentile=quiet_percentile,
            radius_factor=neighborhood_radius_factor,
        )
    return _heading_correction_nearest_pair(
        df, value_col, dominant_azimuth_deg, line_spacing_m,
        quiet_percentile=quiet_percentile,
        max_match_distance_m=max_match_distance_m,
        k_neighbors=k_neighbors,
    )


def _heading_correction_nearest_pair(
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

    groups = _direction_groups(kept, dominant_azimuth_deg)

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
        method="nearest_pair",
    )


def _heading_correction_local_plane(
    df: pd.DataFrame,
    value_col: str,
    dominant_azimuth_deg: float,
    line_spacing_m: float | None,
    quiet_percentile: float = 40.0,
    radius_factor: float = 2.0,
) -> HeadingLevelingResult:
    """The neighbourhood method described in the module docstring: fit
    "local plane + forward/reverse step" over overlapping patches, keep
    the patches the model actually explains, and take the robust centre of
    their steps."""
    kept = df[df["line_id"] >= 0]
    line_ids = sorted(kept["line_id"].unique())
    if len(line_ids) < _MIN_LINES_PER_NEIGHBORHOOD:
        return HeadingLevelingResult(
            False,
            f"측선이 {len(line_ids)}개뿐이라 국소 평면 헤딩 보정을 계산할 수 없습니다 "
            f"(지질 경사와 헤딩 오차를 분리하려면 최소 {_MIN_LINES_PER_NEIGHBORHOOD}개 필요).",
            None, 0, 0, method="local_plane",
        )

    groups = _direction_groups(kept, dominant_azimuth_deg)
    if len(set(groups.values())) < 2:
        return HeadingLevelingResult(
            False, "모든 측선이 같은 방향으로 비행되어 헤딩 보정을 계산할 수 없습니다 (반대 방향 측선 필요).",
            None, 0, 0, groups, method="local_plane",
        )

    x = kept["x"].to_numpy(dtype=float)
    y = kept["y"].to_numpy(dtype=float)
    v = kept[value_col].to_numpy(dtype=float)
    lid = kept["line_id"].to_numpy()
    # +/-0.5 rather than 0/1 so the fitted coefficient is the full A-to-B
    # step and the plane's constant term stays the local mean level.
    g = np.where(np.vectorize(groups.get)(lid) == "A", 0.5, -0.5)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v, lid, g = x[finite], y[finite], v[finite], lid[finite], g[finite]
    if len(x) < 2 * _MIN_POINTS_PER_GROUP:
        return HeadingLevelingResult(
            False, "헤딩 보정을 계산할 유효 자료가 부족합니다.", None, 0, 0, groups, method="local_plane"
        )

    spacing = line_spacing_m if line_spacing_m and line_spacing_m > 0 else 50.0
    radius = radius_factor * spacing
    # Centres on a grid one spacing apart, so neighbourhoods overlap
    # (radius is a multiple of the spacing) and every part of the survey is
    # looked at from more than one patch.
    tree = cKDTree(np.column_stack([x, y]))
    gx = np.arange(x.min(), x.max() + spacing, spacing)
    gy = np.arange(y.min(), y.max() + spacing, spacing)
    centres = np.column_stack([m.ravel() for m in np.meshgrid(gx, gy)])

    offsets: list[float] = []
    residuals: list[float] = []
    n_considered = 0
    for cx, cy in centres:
        idx = tree.query_ball_point([cx, cy], radius)
        if len(idx) < 2 * _MIN_POINTS_PER_GROUP:
            continue
        idx = np.asarray(idx)
        gi = g[idx]
        n_a = int(np.sum(gi > 0))
        if n_a < _MIN_POINTS_PER_GROUP or len(idx) - n_a < _MIN_POINTS_PER_GROUP:
            continue
        if np.unique(lid[idx]).size < _MIN_LINES_PER_NEIGHBORHOOD:
            continue
        n_considered += 1

        design = np.column_stack([np.ones(len(idx)), x[idx] - cx, y[idx] - cy, gi])
        coeff, _res, rank, _sv = np.linalg.lstsq(design, v[idx], rcond=None)
        if rank < design.shape[1]:
            # Lines all on one side, or otherwise degenerate geometry -
            # the step isn't separable from the gradient here.
            continue
        offsets.append(float(coeff[3]))
        residuals.append(float(np.std(v[idx] - design @ coeff)))

    if not offsets:
        return HeadingLevelingResult(
            False,
            "헤딩 보정을 추정할 만한 국소 영역을 찾지 못했습니다 - 반대방향 측선이 "
            f"{_MIN_LINES_PER_NEIGHBORHOOD}개 이상 인접한 구간이 필요합니다 "
            "(영역 반경 배수를 키우거나 nearest_pair 방법을 사용해 보세요).",
            None, n_considered, 0, groups, method="local_plane",
        )

    offsets_arr = np.array(offsets)
    residual_arr = np.array(residuals)
    # The quiet test: keep the neighbourhoods the plane+step model
    # explains best. This is what "a broad area holding a steady value,
    # undisturbed by structure or ground objects" reduces to once it is
    # measured rather than eyeballed.
    threshold = np.percentile(residual_arr, quiet_percentile)
    quiet = residual_arr <= threshold
    if not quiet.any():
        quiet = np.ones_like(residual_arr, dtype=bool)

    offset = float(np.median(offsets_arr[quiet]))
    # MAD scaled to a standard deviation, so the spread is comparable to
    # the offset in the same units.
    spread = float(1.4826 * np.median(np.abs(offsets_arr[quiet] - offset)))

    warnings: list[str] = []
    if spread > abs(offset):
        warnings.append(
            f"구역별 헤딩 오프셋의 산포({spread:.2f}nT)가 추정값({offset:.2f}nT)보다 큽니다 - "
            "헤딩 오차가 아니라 측선별 레벨 오차일 가능성이 높으니 통계적 레벨링을 함께 쓰세요."
        )

    shifts = {lid_: (-offset / 2.0 if groups[lid_] == "A" else offset / 2.0) for lid_ in groups}
    return HeadingLevelingResult(
        applied=True,
        reason=None,
        offset_nt=offset,
        n_matched_pairs=n_considered,
        n_quiet_pairs=int(quiet.sum()),
        line_groups=groups,
        line_shifts=shifts,
        method="local_plane",
        offset_spread_nt=spread,
        residual_nt=float(np.median(residual_arr[quiet])),
        warnings=warnings,
    )


def apply_heading_correction(df: pd.DataFrame, value_col: str, result: HeadingLevelingResult) -> np.ndarray:
    values = df[value_col].to_numpy(copy=True)
    if not result.applied or not result.line_shifts:
        return values
    shift = df["line_id"].map(result.line_shifts).fillna(0.0).to_numpy()
    return values + shift
