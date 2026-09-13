"""Flight-line detection: project to a local metric CRS, derive heading and
speed, find the dominant survey azimuth, and flag takeoff/landing/turn
samples that do not belong to a production line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pyproj import Transformer


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


# Korea-specific projected CRS choices (GRS80-based, matching the codes
# national mapping agencies/most Korean GIS software expect), offered as
# an alternative to generic auto-UTM for surveys located in Korea.
KOREA_UTM_EPSG = 5179  # KGD2002 Unified CS (GRS80 기반 한국 통합 좌표계)


def korea2010_epsg(lon: float) -> int:
    """KGD2002 Belt 2010 (Korea2010) EPSG by longitude band."""
    if lon < 126:
        return 5185  # 서부
    if lon < 128:
        return 5186  # 중부
    if lon < 130:
        return 5187  # 동부
    return 5188  # 동해


def resolve_korea_projection_epsg(korea_projection: str, lon_center: float) -> int:
    """Resolve a "Korea Projection" selection (korea_utm/korea2010/utm) to
    a concrete EPSG code, mirroring DroneMagAdv's Coordinate options."""
    if korea_projection == "korea_utm":
        return KOREA_UTM_EPSG
    if korea_projection == "korea2010":
        return korea2010_epsg(lon_center)
    if korea_projection == "utm":
        # Korea-domestic 51N/52N split (matches the general _utm_epsg
        # result for Korea's longitude range, pinned explicitly here for
        # the two documented Korean UTM zones).
        return 32651 if lon_center < 126 else 32652
    raise ValueError(f"알 수 없는 Korea Projection 값입니다: {korea_projection}")


def project_to_local_xy(lat: np.ndarray, lon: np.ndarray, epsg_override: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Project lat/lon to a local projected CRS in meters. Auto-detects
    the UTM zone from the data's centroid unless epsg_override is given
    (e.g. to match a specific national grid, or to keep results
    consistent with a previous project that happened to sit near a UTM
    zone boundary)."""
    epsg = epsg_override or _utm_epsg(float(np.nanmean(lon)), float(np.nanmean(lat)))
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y), epsg


def _circular_diff_180(a: np.ndarray, b: float) -> np.ndarray:
    """Smallest absolute distance between angles a and b on a 180deg-periodic
    domain (i.e. heading and heading+180 are equivalent)."""
    d = (a - b + 90.0) % 180.0 - 90.0
    return np.abs(d)


def _pca_dominant_azimuth(x: np.ndarray, y: np.ndarray) -> float:
    """Dominant survey-line azimuth (degrees, mod 180) from PCA of the
    (x, y) point cloud: the direction of greatest positional variance.
    Complementary to the heading-histogram method (the default) - this
    looks at the overall spatial footprint's elongation rather than each
    sample's instantaneous heading, so it isn't biased by how much flight
    time was spent on perpendicular tie lines vs. survey lines, at the
    cost of being less informative when the survey block itself is closer
    to square than elongated."""
    pts = np.column_stack([x, y])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return 0.0
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    return float(np.degrees(np.arctan2(principal[1], principal[0])) % 180.0)


@dataclass
class LineDetectionParams:
    heading_lag_seconds: float = 1.0
    heading_tolerance_deg: float = 20.0
    min_speed_mps: float = 1.5
    min_line_length_m: float = 150.0
    turn_buffer_m: float = 15.0
    max_gap_seconds: float = 1.0
    # "heading_histogram" (default): most common instantaneous flight
    # heading. "pca": principal axis of the point cloud's spatial spread
    # (DroneMagAdv's "자동 방향 검출").
    direction_method: Literal["heading_histogram", "pca"] = "heading_histogram"
    # Bridge brief mid-line interruptions (wind gusts, a momentary GPS/IMU
    # blip) back into the same line instead of leaving a data gap or
    # splitting one physical line into two: a short excluded run between
    # two kept runs is folded back in (regardless of *why* it was
    # excluded) when it is both short (bridge_max_gap_m) and lines up with
    # the same cross-track position (bridge_max_offset_m) as the runs on
    # either side - i.e. clearly a wobble on the same line, not a turn
    # onto an adjacent one.
    bridge_gaps: bool = True
    bridge_max_gap_m: float = 100.0
    bridge_max_offset_m: float = 15.0
    # gap path length must not exceed this multiple of the straight-line
    # distance actually covered - rejects loop/U-turn maneuvers (e.g. a
    # heading-calibration turn) that circle back near their own start.
    bridge_straightness_factor: float = 2.0


def detect_lines(
    df: pd.DataFrame,
    params: LineDetectionParams | None = None,
    utm_epsg_override: int | None = None,
) -> pd.DataFrame:
    """Given a dataframe with timestamp/lat/lon columns (index-aligned,
    time-sorted), return a copy with added columns:
    x, y (local meters), heading_deg (mod 180), speed_mps,
    line_id (-1 if excluded), exclusion_reason.
    """
    params = params or LineDetectionParams()
    out = df.reset_index(drop=True).copy()

    x, y, epsg = project_to_local_xy(out["lat"].to_numpy(), out["lon"].to_numpy(), utm_epsg_override)
    out["x"] = x
    out["y"] = y

    t = out["timestamp"]
    dt_median = t.diff().dt.total_seconds().dropna()
    dt_median = dt_median[dt_median > 0].median() if not dt_median.empty else 0.1
    lag = max(1, round(params.heading_lag_seconds / dt_median))

    x_fwd = np.r_[x[lag:], [np.nan] * lag]
    y_fwd = np.r_[y[lag:], [np.nan] * lag]
    x_bwd = np.r_[[np.nan] * lag, x[:-lag]]
    y_bwd = np.r_[[np.nan] * lag, y[:-lag]]
    t_sec = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    t_fwd = np.r_[t_sec[lag:], [np.nan] * lag]
    t_bwd = np.r_[[np.nan] * lag, t_sec[:-lag]]

    dx = x_fwd - x_bwd
    dy = y_fwd - y_bwd
    d_t = t_fwd - t_bwd
    dist = np.hypot(dx, dy)

    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(d_t > 0, dist / d_t, np.nan)
        heading = np.degrees(np.arctan2(dy, dx)) % 180.0

    out["speed_mps"] = speed
    out["heading_deg"] = heading

    valid = ~np.isnan(heading) & (speed >= params.min_speed_mps)
    if params.direction_method == "pca":
        pca_source = (x[valid], y[valid]) if valid.sum() >= 10 else (x, y)
        dominant_azimuth = _pca_dominant_azimuth(*pca_source)
    elif valid.sum() < 10:
        dominant_azimuth = float(np.nanmedian(heading)) if np.isfinite(heading).any() else 0.0
    else:
        bins = np.arange(0, 180 + 2, 2.0)
        hist, edges = np.histogram(heading[valid], bins=bins)
        dominant_azimuth = float((edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2.0)

    on_azimuth = valid & (_circular_diff_180(heading, dominant_azimuth) <= params.heading_tolerance_deg)

    out["line_id"] = -1
    out["exclusion_reason"] = np.where(
        ~valid,
        "takeoff_landing_or_low_speed",
        np.where(~on_azimuth, "off_azimuth_turn", None),
    )

    line_id, reasons = _group_into_lines(out, x, y, t, on_azimuth, params, start_id=0)
    out["line_id"] = line_id
    # only on_azimuth samples pass through the grouping loop above, so only
    # their exclusion_reason should be touched (short_line/turn_buffer/kept);
    # everything else keeps its original takeoff/off-azimuth tag.
    out.loc[on_azimuth, "exclusion_reason"] = reasons[on_azimuth]

    if params.bridge_gaps:
        bridged_line_id, bridged_reason = _bridge_line_gaps(out, x, y, line_id, out["exclusion_reason"].to_numpy(), dominant_azimuth, params)
        out["line_id"] = bridged_line_id
        out["exclusion_reason"] = bridged_reason

    out["exclusion_reason"] = _tag_takeoff_landing_ramps(out, out["line_id"].to_numpy(), out["exclusion_reason"].to_numpy())

    out.attrs["utm_epsg"] = epsg
    out.attrs["dominant_azimuth_deg"] = dominant_azimuth
    return out


def _group_into_lines(
    out: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    t: pd.Series,
    mask: np.ndarray,
    params: LineDetectionParams,
    start_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Group contiguous runs of `mask` samples (splitting on state change or
    a time gap) into candidate lines, applying the min-length and
    turn-buffer trims. Returns (line_id array, exclusion_reason array) sized
    like `out`, with line_id == -1 outside the mask or trimmed away."""
    line_id_out = np.full(len(out), -1, dtype=int)
    reason_out = np.full(len(out), None, dtype=object)

    mask_s = pd.Series(mask, index=out.index)
    big_gap = t.diff().dt.total_seconds().fillna(0) > params.max_gap_seconds
    state_change = mask_s.ne(mask_s.shift()).fillna(True)
    group_id = (state_change | big_gap).cumsum()

    line_id = start_id
    for gid, idx in out.groupby(group_id).groups.items():
        idx = np.asarray(idx)
        if not mask[idx[0]]:
            continue
        seg_x, seg_y = x[idx], y[idx]
        seg_len = np.hypot(np.diff(seg_x), np.diff(seg_y)).sum()
        if seg_len < params.min_line_length_m:
            reason_out[idx] = "short_line"
            continue

        cum = np.r_[0.0, np.cumsum(np.hypot(np.diff(seg_x), np.diff(seg_y)))]
        keep = (cum >= params.turn_buffer_m) & (cum <= seg_len - params.turn_buffer_m)
        if keep.sum() == 0:
            reason_out[idx] = "short_line"
            continue

        reason_out[idx[~keep]] = "turn_buffer"
        kept_idx = idx[keep]
        line_id_out[kept_idx] = line_id
        reason_out[kept_idx] = None
        line_id += 1

    return line_id_out, reason_out


def _bridge_line_gaps(
    out: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    line_id_out: np.ndarray,
    reason_out: np.ndarray,
    dominant_azimuth_deg: float,
    params: LineDetectionParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold short excluded runs between two kept runs of the *same*
    physical line back into that line, instead of leaving a data gap or
    splitting it into two separate "lines". A run in between is bridged
    only when all three hold:
      1. short along-track (gap path length <= bridge_max_gap_m)
      2. the flanking runs sit at essentially the same cross-track
         position (bridge_max_offset_m) - not a move onto an adjacent line
      3. the gap was flown reasonably directly, not a loop/U-turn that
         circles back near its own starting point: the path length can't
         exceed bridge_straightness_factor times the straight-line
         distance actually covered. A real forward wobble travels close
         to a straight line end-to-end; a turn/calibration loop covers
         many meters of track while its net displacement stays small, so
         this ratio cleanly tells the two apart even when a loop happens
         to end up back on the same cross-track position (guard 2 alone
         can't distinguish a loop from a wobble).
    Processed independently per source file (or the whole dataset if
    there's only one), in time order, with a single forward pass: each
    bridged run is folded into the line id accumulated so far, so a chain
    of several short wobbles down one line all merge into one id."""
    line_id_out = line_id_out.copy()
    reason_out = reason_out.copy()
    cross_track = cross_track_coordinate(x, y, dominant_azimuth_deg)

    source_col = out["source_file_index"] if "source_file_index" in out.columns else pd.Series(0, index=out.index)
    for _src, idx in out.groupby(source_col).groups.items():
        idx = np.asarray(idx)
        sub_ids = line_id_out[idx]
        m = len(idx)
        current_id = None
        prev_end_pos = None  # local position (within idx) of the end of the currently-accumulated run
        i = 0
        while i < m:
            if sub_ids[i] < 0:
                i += 1
                continue
            j = i
            while j + 1 < m and sub_ids[j + 1] == sub_ids[i]:
                j += 1
            run_id = sub_ids[i]

            if current_id is not None:
                gap_local = idx[prev_end_pos + 1:i]
                span_idx = idx[prev_end_pos:i + 1]
                tail_pt, head_pt = idx[prev_end_pos], idx[i]
                gap_len_m = float(np.hypot(np.diff(x[span_idx]), np.diff(y[span_idx])).sum())
                straight_dist_m = float(np.hypot(x[head_pt] - x[tail_pt], y[head_pt] - y[tail_pt]))
                offset_diff = abs(float(cross_track[tail_pt]) - float(cross_track[head_pt]))
                direct_enough = gap_len_m <= max(straight_dist_m, 1e-6) * params.bridge_straightness_factor
                if gap_len_m <= params.bridge_max_gap_m and offset_diff <= params.bridge_max_offset_m and direct_enough:
                    line_id_out[idx[i:j + 1]] = current_id
                    line_id_out[gap_local] = current_id
                    reason_out[gap_local] = None
                    sub_ids[i:j + 1] = current_id
                else:
                    current_id = run_id
            else:
                current_id = run_id

            prev_end_pos = j
            i = j + 1

    return line_id_out, reason_out


def _tag_takeoff_landing_ramps(
    out: pd.DataFrame,
    line_id_out: np.ndarray,
    reason_out: np.ndarray,
) -> np.ndarray:
    """Relabel excluded points before the first kept survey-line point
    (per source file) as "takeoff_ramp" and points after the last kept
    point as "landing_ramp" - the transit from actual liftoff to the
    start of production-line flying, and from the end of the last line
    back to landing. These are near-certainly unusable regardless of why
    the point-level classifier excluded them, and are handled separately
    from ordinary in-survey turns/short-line trims (whose reason is left
    untouched) so the line editor can hide/protect them independently."""
    reason_out = reason_out.copy()
    source_col = out["source_file_index"] if "source_file_index" in out.columns else pd.Series(0, index=out.index)
    for _src, idx in out.groupby(source_col).groups.items():
        idx = np.asarray(idx)
        kept_mask = line_id_out[idx] >= 0
        if not kept_mask.any():
            continue
        first_kept_pos = int(np.argmax(kept_mask))
        last_kept_pos = len(kept_mask) - 1 - int(np.argmax(kept_mask[::-1]))
        pre_idx = idx[:first_kept_pos]
        post_idx = idx[last_kept_pos + 1:]
        reason_out[pre_idx[line_id_out[pre_idx] < 0]] = "takeoff_ramp"
        reason_out[post_idx[line_id_out[post_idx] < 0]] = "landing_ramp"
    return reason_out


def detect_tie_lines(
    out: pd.DataFrame,
    dominant_azimuth_deg: float,
    params: LineDetectionParams | None = None,
    tie_tolerance_deg: float = 20.0,
) -> pd.Series:
    """Given the dataframe already produced by detect_lines (with x, y,
    heading_deg, speed_mps, line_id columns), find contiguous runs flown
    roughly perpendicular to the dominant survey azimuth among the points
    detect_lines excluded as "off_azimuth_turn" - these are tie lines, not
    turns, when a real perpendicular calibration line was flown. Returns a
    tie_line_id Series (-1 where not part of a tie line)."""
    params = params or LineDetectionParams()
    x, y = out["x"].to_numpy(), out["y"].to_numpy()
    t = out["timestamp"]
    heading = out["heading_deg"].to_numpy()
    speed = out["speed_mps"].to_numpy()

    perp_azimuth = (dominant_azimuth_deg + 90.0) % 180.0
    valid = ~np.isnan(heading) & (speed >= params.min_speed_mps)
    candidate = valid & (out["line_id"].to_numpy() < 0) & (_circular_diff_180(heading, perp_azimuth) <= tie_tolerance_deg)

    tie_line_id, _reason = _group_into_lines(out, x, y, t, candidate, params, start_id=0)
    return pd.Series(tie_line_id, index=out.index, name="tie_line_id")


def _perp_vector(dominant_azimuth_deg: float) -> np.ndarray:
    """Unit vector perpendicular to the line azimuth (the cross-line axis),
    given azimuth is the atan2(dy, dx) convention used throughout this
    module (0 = along +x/easting, 90 = along +y/northing)."""
    az = np.radians(dominant_azimuth_deg)
    return np.array([-np.sin(az), np.cos(az)])


def cross_track_coordinate(x: np.ndarray, y: np.ndarray, dominant_azimuth_deg: float) -> np.ndarray:
    """Scalar coordinate of (x, y) points projected onto the cross-line
    axis - i.e. how far "sideways" a point is from other lines, used to
    order lines by adjacency."""
    perp = _perp_vector(dominant_azimuth_deg)
    return np.asarray(x) * perp[0] + np.asarray(y) * perp[1]


def group_turn_segments(df: pd.DataFrame, max_gap_seconds: float = 1.0) -> np.ndarray:
    """Groups contiguous runs of turn/off-azimuth points (already tagged
    via detect_lines' exclusion_reason column) into individual turn
    events. Each turn typically covers a small spatial footprint while
    sweeping a wide range of headings - closely matching a dedicated
    heading-effect calibration flight's clover-leaf pattern (see
    processing/heading_calibration.py's docstring; the manufacturer's own
    calibration guide notes "the best calibration data is usually located
    where the drone makes a turn"). Returns a group id array the same
    length as df (-1 for non-turn points), split on both a state change
    (turn vs. not) and a time gap exceeding max_gap_seconds so two turns
    separated by a flight-log gap aren't merged into one."""
    is_turn = df["exclusion_reason"].isin(["off_azimuth_turn", "turn_buffer"]).to_numpy()
    t = df["timestamp"]
    big_gap = t.diff().dt.total_seconds().fillna(0) > max_gap_seconds
    turn_series = pd.Series(is_turn, index=df.index)
    state_change = turn_series.ne(turn_series.shift()).fillna(True)
    group_id = (state_change | big_gap).cumsum().to_numpy()
    return np.where(is_turn, group_id, -1)


def estimate_line_spacing_m(df: pd.DataFrame, dominant_azimuth_deg: float) -> float | None:
    """Median perpendicular distance between adjacent accepted lines, used
    to size the grid interpolation search radius and line-matching distance
    for heading leveling. Returns None with fewer than 2 lines."""
    kept = df[df["line_id"] >= 0]
    if kept["line_id"].nunique() < 2:
        return None

    centroids = kept.groupby("line_id")[["x", "y"]].mean()
    cross_track = cross_track_coordinate(centroids["x"].to_numpy(), centroids["y"].to_numpy(), dominant_azimuth_deg)
    cross_track.sort()
    spacings = np.diff(cross_track)
    spacings = spacings[spacings > 0]
    if spacings.size == 0:
        return None
    return float(np.median(spacings))
