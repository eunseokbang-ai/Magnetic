"""Repeatability-test analysis: the UAV magnetics guidelines' recommended
quick field check for "real-world system noise" (Section 9.1) - fly a
small box pattern several times (in both directions) and look at how much
each repeated pass deviates from their common mean, at a fixed position
along the shared track. Distinct from the survey's own turn-based
calibration (processing/heading_calibration.py): this is a dedicated,
short (~15-20 min) test flight, analysed on its own before or during a
survey to characterise the platform-sensor system's total noise
independent of geology.

Passes over the same physical ground track (flown repeatedly, in either
direction) are auto-grouped by proximity + parallel/anti-parallel heading;
within each group, every pass is projected onto a shared spatial axis (the
group's principal direction, which - unlike raw flight heading - is the
same for a line regardless of which way it was flown) so that forward and
reverse passes land on the same physical positions without needing any
sign flip, and the pooled deviation from the cross-pass mean at each
position gives the 1/2/3-sigma noise envelope described in the guidelines'
Figure 24. Each pass's actual flight direction (from its first/last point
in time, not from the sign-ambiguous PCA axis) is tracked separately, and
the mean difference between forward- and reverse-flown passes (when both
exist) is reported as the heading error, matching the guidelines' Figure
24 caption.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .diurnal import apply_diurnal_correction
from .lines import LineDetectionParams, detect_lines

# Two lines are treated as repeats of the same physical track when their
# centroids, projected onto the perpendicular of their shared direction,
# are closer than this - deliberately generous since a repeatability test
# box is flown by hand/autopilot with some line-to-line drift, not a
# precision survey.
_SAME_TRACK_PERP_TOLERANCE_M = 10.0
_SAME_TRACK_ANGLE_TOLERANCE_DEG = 15.0
_MIN_POINTS_PER_PASS = 10


@dataclass
class _Pass:
    line_id: int
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


def _line_pass(line_id: int, group: pd.DataFrame) -> _Pass | None:
    ordered = group.sort_values("timestamp")
    if len(ordered) < _MIN_POINTS_PER_PASS:
        return None
    x, y, v = ordered["x"].to_numpy(), ordered["y"].to_numpy(), ordered["value"].to_numpy()
    pts = np.column_stack([x, y])
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    raw_direction = eigvecs[:, int(np.argmax(eigvals))]
    canonical_direction = _canonical_direction(raw_direction)

    temporal_displacement = pts[-1] - pts[0]
    flight_sign = 1 if float(np.dot(temporal_displacement, canonical_direction)) >= 0 else -1

    return _Pass(
        line_id=line_id, x=x, y=y, value=v, centroid=centroid,
        canonical_direction=canonical_direction, flight_sign=flight_sign,
    )


def _group_passes(passes: list[_Pass]) -> list[list[_Pass]]:
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
            if angle_deg > _SAME_TRACK_ANGLE_TOLERANCE_DEG:
                continue
            avg_dir = passes[i].canonical_direction + passes[j].canonical_direction
            norm = np.linalg.norm(avg_dir)
            if norm < 1e-9:
                continue
            avg_dir = avg_dir / norm
            perp = np.array([-avg_dir[1], avg_dir[0]])
            offset = float(np.dot(passes[j].centroid - passes[i].centroid, perp))
            if abs(offset) <= _SAME_TRACK_PERP_TOLERANCE_M:
                union(i, j)

    groups: dict[int, list[_Pass]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(passes[i])
    return [g for g in groups.values() if len(g) >= 2]


def _analyze_group(group: list[_Pass]) -> dict:
    ref_direction = group[0].canonical_direction
    ref_centroid = group[0].centroid
    ref_flight_sign = group[0].flight_sign

    projections = []
    for p in group:
        # Projected onto the *shared* canonical axis (not each pass's own),
        # so forward and reverse passes over the same track land on the
        # same spatial positions with no sign flip needed.
        s = (np.column_stack([p.x, p.y]) - ref_centroid) @ ref_direction
        order = np.argsort(s)
        is_reverse = p.flight_sign != ref_flight_sign
        projections.append((s[order], p.value[order], is_reverse))

    lo = max(float(s.min()) for s, _, _ in projections)
    hi = min(float(s.max()) for s, _, _ in projections)
    if hi <= lo:
        return {"available": False}
    common_s = np.linspace(lo, hi, num=min(200, min(len(s) for s, _, _ in projections)))

    resampled = np.full((len(projections), len(common_s)), np.nan)
    is_reverse_flags = []
    for i, (s, v, is_reverse) in enumerate(projections):
        resampled[i, :] = np.interp(common_s, s, v)
        is_reverse_flags.append(is_reverse)
    is_reverse_flags = np.array(is_reverse_flags)

    # Noise is measured *within* each flight-direction group (deviation
    # from that group's own per-position mean), not against a single
    # mixed-direction mean - a real heading-dependent bias between forward
    # and reverse passes would otherwise leak into the noise estimate
    # instead of being isolated in heading_error_nt below, matching how
    # the guidelines report these as two distinct numbers (Figure 24).
    pooled_devs = []
    for flag in (False, True):
        subset = resampled[is_reverse_flags == flag, :]
        if subset.shape[0] == 0:
            continue
        group_mean = np.nanmean(subset, axis=0)
        pooled_devs.append((subset - group_mean[np.newaxis, :]).ravel())
    pooled = np.concatenate(pooled_devs) if pooled_devs else np.array([])
    pooled = pooled[np.isfinite(pooled)]
    std_nt = float(np.std(pooled)) if pooled.size else float("nan")

    heading_error_nt = None
    if is_reverse_flags.any() and (~is_reverse_flags).any():
        fwd_mean = np.nanmean(resampled[~is_reverse_flags, :], axis=0)
        rev_mean = np.nanmean(resampled[is_reverse_flags, :], axis=0)
        diff = fwd_mean - rev_mean
        diff = diff[np.isfinite(diff)]
        if diff.size:
            heading_error_nt = float(np.mean(np.abs(diff)))

    return {
        "available": True,
        "n_passes": len(group),
        "n_forward": int((~is_reverse_flags).sum()),
        "n_reverse": int(is_reverse_flags.sum()),
        "line_ids": [p.line_id for p in group],
        "track_length_m": float(hi - lo),
        "noise_1sigma_nt": std_nt,
        "noise_2sigma_nt": 2.0 * std_nt,
        "noise_3sigma_nt": 3.0 * std_nt,
        "heading_error_nt": heading_error_nt,
    }


def analyze_repeatability(
    df: pd.DataFrame,
    base_raw: pd.DataFrame,
    time_offset_seconds: float = 0.0,
    diurnal_reference: str = "mean",
    line_params: LineDetectionParams | None = None,
    utm_epsg_override: int | None = None,
) -> dict:
    """Full analysis from a raw repeatability-test-flight DataFrame (same
    schema as the main survey loader) plus the project's base station data.
    Returns available=False with a `reason` when no repeated track could be
    identified (e.g. only a single pass, or passes too far apart)."""
    diurnal = apply_diurnal_correction(
        df["timestamp"], df["mag_raw"].to_numpy(), base_raw,
        time_offset_seconds=time_offset_seconds, reference=diurnal_reference,
    )
    working = df.copy()
    working["value"] = diurnal.corrected

    detected = detect_lines(working, line_params or LineDetectionParams(), utm_epsg_override=utm_epsg_override)

    line_passes = []
    for lid, group in detected[detected["line_id"] >= 0].groupby("line_id"):
        p = _line_pass(int(lid), group)
        if p is not None:
            line_passes.append(p)

    if len(line_passes) < 2:
        return {
            "available": False,
            "reason": "반복 통과로 인식할 만한 측선이 2개 미만입니다 (같은 구간을 여러 번 왕복 비행한 자료가 필요합니다).",
        }

    groups = _group_passes(line_passes)
    if not groups:
        return {
            "available": False,
            "reason": "서로 겹치는 반복 구간을 찾지 못했습니다 - 같은 트랙을 반복 비행했는지, 자료가 충분한지 확인하세요.",
        }

    group_results = [_analyze_group(g) for g in groups]
    group_results = [g for g in group_results if g.get("available")]
    if not group_results:
        return {"available": False, "reason": "반복 구간은 찾았으나 겹치는 구간이 없어 분석할 수 없습니다."}

    all_1sigma = [g["noise_1sigma_nt"] for g in group_results if np.isfinite(g["noise_1sigma_nt"])]
    overall_1sigma = float(np.mean(all_1sigma)) if all_1sigma else None

    return {
        "available": True,
        "n_groups": len(group_results),
        "n_passes_total": sum(g["n_passes"] for g in group_results),
        "overall_noise_1sigma_nt": overall_1sigma,
        "overall_noise_2sigma_nt": overall_1sigma * 2.0 if overall_1sigma is not None else None,
        "overall_noise_3sigma_nt": overall_1sigma * 3.0 if overall_1sigma is not None else None,
        "groups": group_results,
    }
