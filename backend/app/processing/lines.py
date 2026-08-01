"""Flight-line detection: project to a local metric CRS, derive heading and
speed, find the dominant survey azimuth, and flag takeoff/landing/turn
samples that do not belong to a production line.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import Transformer


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


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


@dataclass
class LineDetectionParams:
    heading_lag_seconds: float = 1.0
    heading_tolerance_deg: float = 20.0
    min_speed_mps: float = 1.5
    min_line_length_m: float = 150.0
    turn_buffer_m: float = 15.0
    max_gap_seconds: float = 1.0


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
    if valid.sum() < 10:
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
