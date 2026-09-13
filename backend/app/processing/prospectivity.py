"""Rule-based mineral prospectivity ("target score") mapping: combines
normalized (0-1) magnetic-derivative layers into one weighted score grid,
then reports the ranked, auto-explained local-maximum targets - the
"final goal" interpretation layer described for mineral exploration,
built entirely on quantities this pipeline can compute from a drone
magnetic survey alone (it does not ingest external geology/radiometric
data, so those requested layers are not available here):

  - "asa"            : analytic-signal amplitude, percentile-normalized -
                        a generic "source nearby" indicator, high over
                        the edges/tops of magnetic bodies regardless of
                        dip.
  - "thd"             : total horizontal derivative, percentile-
                        normalized - high directly over steep
                        contacts/edges.
  - "structure"       : proximity to the nearest extracted magnetic
                        lineament (lineaments.py), exponential distance
                        decay - for structurally-controlled deposit
                        styles (faults/shear zones as fluid pathways).
  - "contact"         : proximity to the nearest detected magnetic
                        contact (contacts.py), same decay - for
                        contact/skarn-style deposits localized at
                        rock-unit boundaries.
  - "susceptibility"  : near-surface susceptibility from the project's
                        most recent 3D inversion (inversion.py), if one
                        has been run - percentile-normalized average over
                        the shallowest layers.

Each purpose preset is a *default weighting* over whichever of these
layers are actually available for the current project (weights are
renormalized over the available subset, and every target's explanation
lists exactly which layers contributed) - the presets mirror the emphasis
described in standard UAV-magnetic mineral-exploration guidance
(magnetite/Fe: raw magnetic response; skarn: magnetic response at a
contact; Ni-Cu-PGE: weaker anomaly at a structural intersection), not a
literal implementation of a published deposit model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

_MIN_FINITE_CELLS = 20
_LAYER_LABELS = {
    "asa": "Analytic Signal",
    "thd": "THD (총수평미분)",
    "structure": "구조선(리니어먼트) 인접성",
    "contact": "자성 접촉면 인접성",
    "susceptibility": "천부 감수율(3차원 역산)",
}

PURPOSE_PRESETS: dict[str, dict[str, float] | None] = {
    # Magnetite/Fe: dominated by the raw magnetic response itself.
    "magnetite_fe": {"asa": 0.40, "thd": 0.30, "structure": 0.10, "contact": 0.10, "susceptibility": 0.10},
    # Skarn: magnetic high localized right at an intrusive/host-rock contact.
    "skarn": {"asa": 0.25, "thd": 0.20, "structure": 0.10, "contact": 0.35, "susceptibility": 0.10},
    # Ni-Cu-PGE: often a comparatively weak magnetic anomaly, but strongly
    # tied to structural intersections/conduits.
    "ni_cu_pge": {"asa": 0.20, "thd": 0.15, "structure": 0.35, "contact": 0.20, "susceptibility": 0.10},
    "custom": None,
}


@dataclass
class ProspectivityTarget:
    lat: float
    lon: float
    score: float
    rank: int
    breakdown: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


def _normalize_percentile(values: np.ndarray, finite_mask: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0):
    finite = values[finite_mask]
    if finite.size == 0:
        return None
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        return None
    norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return np.where(finite_mask, norm, np.nan)


def _proximity_score(x2d: np.ndarray, y2d: np.ndarray, feature_xy: np.ndarray | None, decay_length_m: float):
    if feature_xy is None or len(feature_xy) == 0:
        return None
    tree = cKDTree(feature_xy)
    dist, _ = tree.query(np.column_stack([x2d.ravel(), y2d.ravel()]), workers=-1)
    return np.exp(-dist.reshape(x2d.shape) / decay_length_m)


def _local_maxima_mask(grid: np.ndarray) -> np.ndarray:
    padded = np.pad(grid, 1, mode="constant", constant_values=-np.inf)
    is_max = np.ones(grid.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor = padded[1 + dr : 1 + dr + grid.shape[0], 1 + dc : 1 + dc + grid.shape[1]]
            is_max &= grid >= neighbor
    return is_max & np.isfinite(grid)


def compute_prospectivity(
    asa_grid: np.ndarray,
    thd_grid: np.ndarray,
    finite_mask: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    utm_epsg: int,
    lineament_points_latlon: list[tuple[float, float]] | None = None,
    contact_points_latlon: list[tuple[float, float]] | None = None,
    susceptibility_grid: np.ndarray | None = None,
    purpose: str = "custom",
    custom_weights: dict[str, float] | None = None,
    decay_length_m: float = 200.0,
    score_threshold: float = 0.6,
    max_targets: int = 20,
    min_target_separation_m: float | None = None,
) -> dict:
    """Build the weighted 0-1 target-score grid and extract ranked,
    explained local-maximum targets. available=False (with a reason) when
    the grid is too small or no cell clears score_threshold - a real
    result (a genuinely unprospective survey, or a threshold set too
    high), not a bug to silently work around."""
    if finite_mask.sum() < _MIN_FINITE_CELLS:
        return {"available": False, "reason": "유효한 격자 셀이 너무 적습니다.", "targets": []}

    weights = PURPOSE_PRESETS.get(purpose)
    if weights is None:
        if not custom_weights:
            return {"available": False, "reason": "가중치를 지정해야 합니다 (purpose=custom).", "targets": []}
        weights = custom_weights

    x2d, y2d = np.meshgrid(easting, northing)
    to_local = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)

    layers: dict[str, np.ndarray] = {}
    asa_norm = _normalize_percentile(asa_grid, finite_mask)
    if asa_norm is not None:
        layers["asa"] = asa_norm
    thd_norm = _normalize_percentile(thd_grid, finite_mask)
    if thd_norm is not None:
        layers["thd"] = thd_norm

    if lineament_points_latlon:
        lats = [p[0] for p in lineament_points_latlon]
        lons = [p[1] for p in lineament_points_latlon]
        xs, ys = to_local.transform(lons, lats)
        structure_score = _proximity_score(x2d, y2d, np.column_stack([xs, ys]), decay_length_m)
        if structure_score is not None:
            layers["structure"] = np.where(finite_mask, structure_score, np.nan)

    if contact_points_latlon:
        lats = [p[0] for p in contact_points_latlon]
        lons = [p[1] for p in contact_points_latlon]
        xs, ys = to_local.transform(lons, lats)
        contact_score = _proximity_score(x2d, y2d, np.column_stack([xs, ys]), decay_length_m)
        if contact_score is not None:
            layers["contact"] = np.where(finite_mask, contact_score, np.nan)

    if susceptibility_grid is not None:
        susc_norm = _normalize_percentile(susceptibility_grid, finite_mask & np.isfinite(susceptibility_grid))
        if susc_norm is not None:
            layers["susceptibility"] = susc_norm

    used_weights = {k: float(weights[k]) for k in layers if weights.get(k, 0) > 0}
    if not used_weights:
        return {
            "available": False,
            "reason": "선택한 탐사목적에 필요한 레이어를 하나도 계산할 수 없습니다 (예: 구조선/접촉면을 먼저 추출하세요).",
            "targets": [],
        }
    weight_sum = sum(used_weights.values())
    used_weights = {k: v / weight_sum for k, v in used_weights.items()}

    score = np.zeros_like(asa_grid, dtype=float)
    for name, w in used_weights.items():
        layer = layers[name]
        score += w * np.where(np.isfinite(layer), layer, 0.0)
    score = np.where(finite_mask, score, np.nan)

    finite_scores = score[np.isfinite(score)]
    if finite_scores.size == 0:
        return {"available": False, "reason": "점수를 계산할 수 있는 유효 셀이 없습니다.", "targets": []}

    maxima_mask = _local_maxima_mask(np.where(np.isfinite(score), score, -np.inf)) & (score >= score_threshold)
    rows, cols = np.nonzero(maxima_mask)
    if rows.size == 0:
        return {
            "available": True,
            "reason": f"점수 임계값({score_threshold:.2f}) 이상인 지점이 없습니다 - 임계값을 낮춰보세요.",
            "score_stats": {
                "min": float(np.min(finite_scores)),
                "max": float(np.max(finite_scores)),
                "mean": float(np.mean(finite_scores)),
            },
            "layers_used": list(used_weights.keys()),
            "weights": used_weights,
            "n_targets": 0,
            "targets": [],
            "score_grid_available": True,
            "score_grid": score,
        }

    cand_scores = score[rows, cols]
    cand_xy = np.column_stack([x2d[rows, cols], y2d[rows, cols]])
    order = np.argsort(-cand_scores)

    if min_target_separation_m is None:
        min_target_separation_m = decay_length_m * 0.5

    kept_idx: list[int] = []
    kept_xy: list[tuple[float, float]] = []
    for i in order:
        xy = cand_xy[i]
        too_close = any(
            np.hypot(xy[0] - kx, xy[1] - ky) < min_target_separation_m for kx, ky in kept_xy
        )
        if too_close:
            continue
        kept_idx.append(i)
        kept_xy.append((xy[0], xy[1]))
        if len(kept_idx) >= max_targets:
            break

    to_latlon = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    targets: list[ProspectivityTarget] = []
    for rank, i in enumerate(kept_idx, start=1):
        r, c = rows[i], cols[i]
        lon, lat = to_latlon.transform(x2d[r, c], y2d[r, c])
        breakdown = {}
        parts = []
        for name, w in sorted(used_weights.items(), key=lambda kv: -kv[1]):
            layer_val = float(layers[name][r, c]) if np.isfinite(layers[name][r, c]) else 0.0
            contribution = w * layer_val
            breakdown[name] = {
                "value": layer_val,
                "weight": w,
                "contribution": contribution,
            }
            pct = 100.0 * contribution / cand_scores[i] if cand_scores[i] > 0 else 0.0
            parts.append(f"{_LAYER_LABELS[name]} {layer_val:.2f} (기여 {pct:.0f}%)")
        explanation = f"종합 점수 {cand_scores[i]:.2f} = " + " + ".join(parts)
        targets.append(
            ProspectivityTarget(
                lat=float(lat), lon=float(lon), score=float(cand_scores[i]), rank=rank,
                breakdown=breakdown, explanation=explanation,
            )
        )

    return {
        "available": True,
        "score_stats": {
            "min": float(np.min(finite_scores)),
            "max": float(np.max(finite_scores)),
            "mean": float(np.mean(finite_scores)),
        },
        "layers_used": list(used_weights.keys()),
        "weights": used_weights,
        "n_targets": len(targets),
        "targets": [
            {
                "lat": t.lat, "lon": t.lon, "score": t.score, "rank": t.rank,
                "breakdown": t.breakdown, "explanation": t.explanation,
            }
            for t in targets
        ],
        "score_grid_available": True,
        "score_grid": score,
    }
