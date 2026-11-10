"""Bring one block of a survey onto the level of another, from the ground
they both covered.

Two blocks flown on different days, with different sensors, or against
different base stations come out at different levels: each is internally
consistent after its own diurnal and line levelling, and the two disagree
by tens of nT where they meet. On a map that is a step along the block
boundary, and every derivative product draws it as a structure.

`repeat_passes.py` already levels flights that re-flew the same *track*.
This is the general case: the blocks merely overlap somewhere, their
lines need not coincide. The measurement:

1. Both blocks are binned onto one square mesh. A cell that holds
   readings of both gives one difference, target median minus reference
   median. Medians, because a cell crossed by a strong anomaly holds a
   few readings far from the rest.
2. The differences are not all the level shift. Within a cell the two
   blocks sampled different spots, so the field's own gradient leaks in,
   with either sign. The estimate is therefore the median of the cell
   differences (model "constant"), or a robust plane through them (model
   "plane", for a difference that drifts across the overlap - a regional
   gradient one block's levelling removed and the other's did not).
3. The uncertainty comes from the cells themselves: the estimate is
   recomputed on resampled sets of cells (bootstrap), and the 16th-84th
   percentile spread is reported as +-1 sigma. Neighbouring cells share
   the same anomalies and are not independent, so whole groups of
   3 x 3 cells are resampled together; resampling single cells would
   call the result tighter than it is.

What this cannot tell apart from a level shift, and says so in
`warnings` instead of hiding it:

- **Different flying heights.** An anomaly reads smaller from higher up,
  so two blocks at different altitudes differ by something shaped like
  the anomaly, not by a constant. The median altitude difference over
  the shared cells is reported; beyond ALTITUDE_CAUTION_M the constant
  is a compromise and height normalisation should come first.
- **A plane carried far outside the overlap.** The gradient is measured
  on the shared cells only. `uncertainty_at` gives the +-1 sigma of the
  correction at any point; the project reports its largest value over
  the target block and warns when the far end of the block is corrected
  much less surely than the overlap is.
- **A thin overlap.** With few shared cells the bootstrap interval is
  wide, and a plane fitted to a strip only constrains the gradient along
  the strip. Below MIN_CELLS the estimate is refused; a plane whose
  gradient is within 2 sigma of zero is reported as not significant, and
  model "auto" then falls back to the constant.

A sensor offset known from a side-by-side test on the ground needs no
estimate: the project applies it directly as a constant (see
store.Project.apply_block_shift, source "manual").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_POINTS_PER_CELL = 3
MIN_CELLS = 8
N_BOOTSTRAP = 400
GROUP_CELLS = 3               # cells per side of a resampling group
ALTITUDE_CAUTION_M = 5.0
_HUBER_K = 1.345


class BlockLevelingError(ValueError):
    pass


@dataclass
class BlockShift:
    """target - reference over their shared ground. To level the target,
    subtract `value_at(x, y)` from it."""

    model: str                       # "constant" | "plane"
    constant_nt: float               # the difference at (x0, y0); the whole of it for "constant"
    constant_se_nt: float
    gradient_x_nt_per_km: float = 0.0
    gradient_y_nt_per_km: float = 0.0
    gradient_x_se: float = 0.0
    gradient_y_se: float = 0.0
    x0: float = 0.0                  # centre of the shared cells, projected metres
    y0: float = 0.0
    n_cells: int = 0
    cell_size_m: float = 0.0
    overlap_area_m2: float = 0.0
    residual_std_nt: float = 0.0     # robust scatter of the cell differences about the model
    altitude_difference_m: float | None = None     # target - reference, median over shared cells
    gradient_significant: bool = False
    warnings: list[str] = field(default_factory=list)

    def value_at(self, x, y) -> np.ndarray:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        out = np.full(x.shape, self.constant_nt, dtype=float)
        if self.model == "plane":
            out = out + self.gradient_x_nt_per_km * (x - self.x0) / 1000.0 + self.gradient_y_nt_per_km * (y - self.y0) / 1000.0
        return out

    def uncertainty_at(self, x, y) -> np.ndarray:
        """+-1 sigma of value_at. For a plane it grows with distance from
        the centre of the shared cells: the gradient was only measured
        there, and applying it a kilometre away multiplies its error by
        that kilometre. (The three terms are added in quadrature, i.e.
        treated as independent, which centring the fit on the overlap
        makes approximately true.)"""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        var = np.full(x.shape, self.constant_se_nt ** 2, dtype=float)
        if self.model == "plane":
            var = var + (self.gradient_x_se * (x - self.x0) / 1000.0) ** 2 + (self.gradient_y_se * (y - self.y0) / 1000.0) ** 2
        return np.sqrt(var)

    def to_dict(self) -> dict:
        r = lambda v, d=3: None if v is None else round(float(v), d)
        return {
            "model": self.model,
            "constant_nt": r(self.constant_nt), "constant_se_nt": r(self.constant_se_nt),
            "gradient_x_nt_per_km": r(self.gradient_x_nt_per_km), "gradient_y_nt_per_km": r(self.gradient_y_nt_per_km),
            "gradient_x_se": r(self.gradient_x_se), "gradient_y_se": r(self.gradient_y_se),
            "x0": r(self.x0, 2), "y0": r(self.y0, 2),
            "n_cells": int(self.n_cells), "cell_size_m": r(self.cell_size_m, 2), "overlap_area_m2": r(self.overlap_area_m2, 0),
            "residual_std_nt": r(self.residual_std_nt), "altitude_difference_m": r(self.altitude_difference_m, 2),
            "gradient_significant": bool(self.gradient_significant), "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BlockShift":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: d[k] for k in keys if k in d and d[k] is not None})


def _cell_medians(x, y, v, alt, x_min, y_min, cell):
    """{(ix, iy): (median value, median altitude)} for cells holding
    enough readings."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[ok], y[ok], v[ok]
    alt = alt[ok] if alt is not None else None
    ix = np.floor((x - x_min) / cell).astype(np.int64)
    iy = np.floor((y - y_min) / cell).astype(np.int64)
    key = ix * 4_000_000 + iy
    order = np.argsort(key, kind="stable")
    key, v = key[order], v[order]
    alt = alt[order] if alt is not None else None
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    out = {}
    for a, b in zip(starts, ends):
        if b - a < MIN_POINTS_PER_CELL:
            continue
        k = int(key[a])
        altitude = float(np.nanmedian(alt[a:b])) if alt is not None and np.isfinite(alt[a:b]).any() else np.nan
        out[(k // 4_000_000, k % 4_000_000)] = (float(np.median(v[a:b])), altitude)
    return out


def _mad_std(r: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(r - np.median(r)))) if len(r) else 0.0


def _fit_plane(X: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Huber-weighted least squares for d = a + bx*X1 + by*X2."""
    A = np.column_stack([np.ones(len(d)), X])
    beta, *_ = np.linalg.lstsq(A, d, rcond=None)
    for _ in range(20):
        r = d - A @ beta
        scale = max(_mad_std(r), 1e-6)
        w = np.minimum(1.0, _HUBER_K * scale / np.maximum(np.abs(r), 1e-12))
        new, *_ = np.linalg.lstsq(A * w[:, None], d * w, rcond=None)
        if np.allclose(new, beta, atol=1e-6):
            beta = new
            break
        beta = new
    return beta


def estimate_block_shift(
    ref_x, ref_y, ref_v, tgt_x, tgt_y, tgt_v,
    cell_size_m: float,
    model: str = "auto",
    ref_alt=None, tgt_alt=None,
    seed: int = 0,
) -> BlockShift:
    """target - reference from the cells both blocks cover. `model` is
    "constant", "plane" or "auto" (a plane only when its gradient is
    significant). Coordinates in projected metres."""
    if model not in ("auto", "constant", "plane"):
        raise BlockLevelingError(f"알 수 없는 모형입니다: {model}")
    ref_x, ref_y, ref_v = (np.asarray(a, dtype=float) for a in (ref_x, ref_y, ref_v))
    tgt_x, tgt_y, tgt_v = (np.asarray(a, dtype=float) for a in (tgt_x, tgt_y, tgt_v))
    if len(ref_x) == 0 or len(tgt_x) == 0:
        raise BlockLevelingError("기준 블록 또는 맞출 블록에 측점이 없습니다.")
    cell = float(cell_size_m)
    if not np.isfinite(cell) or cell <= 0:
        raise BlockLevelingError("셀 크기가 올바르지 않습니다.")
    x_min = float(min(np.nanmin(ref_x), np.nanmin(tgt_x)))
    y_min = float(min(np.nanmin(ref_y), np.nanmin(tgt_y)))
    ref_alt = None if ref_alt is None else np.asarray(ref_alt, dtype=float)
    tgt_alt = None if tgt_alt is None else np.asarray(tgt_alt, dtype=float)
    ref_cells = _cell_medians(ref_x, ref_y, ref_v, ref_alt, x_min, y_min, cell)
    tgt_cells = _cell_medians(tgt_x, tgt_y, tgt_v, tgt_alt, x_min, y_min, cell)
    shared = sorted(set(ref_cells) & set(tgt_cells))
    if len(shared) < MIN_CELLS:
        raise BlockLevelingError(
            f"두 블록이 함께 지나간 구역이 너무 작습니다 (공통 셀 {len(shared)}개, 최소 {MIN_CELLS}개, 셀 {cell:.0f} m). "
            "겹치는 범위가 있는지, 블록 선택이 맞는지 확인하세요."
        )
    ij = np.array(shared, dtype=float)
    cx = x_min + (ij[:, 0] + 0.5) * cell
    cy = y_min + (ij[:, 1] + 0.5) * cell
    d = np.array([tgt_cells[k][0] - ref_cells[k][0] for k in shared])
    dalt = np.array([tgt_cells[k][1] - ref_cells[k][1] for k in shared])
    x0, y0 = float(cx.mean()), float(cy.mean())
    X = np.column_stack([(cx - x0) / 1000.0, (cy - y0) / 1000.0])

    # resampling groups of neighbouring cells - see the module docstring
    group = (ij[:, 0] // GROUP_CELLS).astype(np.int64) * 1_000_003 + (ij[:, 1] // GROUP_CELLS).astype(np.int64)
    groups = [np.flatnonzero(group == g) for g in np.unique(group)]
    rng = np.random.default_rng(seed)

    def bootstrap(estimator) -> np.ndarray:
        out = []
        for _ in range(N_BOOTSTRAP):
            pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            try:
                out.append(estimator(pick))
            except np.linalg.LinAlgError:
                continue
        return np.array(out, dtype=float)

    def spread(samples: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(samples, [16, 84], axis=0)
        return (hi - lo) / 2.0

    constant = float(np.median(d))
    constant_se = float(spread(bootstrap(lambda idx: np.median(d[idx]))))

    # a plane needs the cells to span both directions
    span = np.ptp(X, axis=0) * 1000.0
    plane_possible = len(shared) >= 2 * MIN_CELLS and min(span) >= 2 * cell
    beta = se = None
    significant = False
    if plane_possible:
        beta = _fit_plane(X, d)
        se = spread(bootstrap(lambda idx: _fit_plane(X[idx], d[idx])))
        significant = bool(max(abs(beta[1]) / max(se[1], 1e-9), abs(beta[2]) / max(se[2], 1e-9)) >= 2.0)

    warnings: list[str] = []
    use_plane = model == "plane" or (model == "auto" and significant)
    if model == "plane" and not plane_possible:
        raise BlockLevelingError("겹치는 구역이 한 방향으로만 길어 평면을 정할 수 없습니다 - 상수 모형을 쓰세요.")
    if use_plane:
        residual = d - (beta[0] + X @ beta[1:])
        shift = BlockShift("plane", float(beta[0]), float(se[0]), float(beta[1]), float(beta[2]), float(se[1]), float(se[2]))
        if not significant:
            warnings.append("기울기가 불확실도의 2배에 못 미칩니다 - 평면보다 상수 모형이 안전합니다.")
    else:
        residual = d - constant
        shift = BlockShift("constant", constant, constant_se)
        if beta is not None:
            shift.gradient_x_nt_per_km, shift.gradient_y_nt_per_km = float(beta[1]), float(beta[2])
            shift.gradient_x_se, shift.gradient_y_se = float(se[1]), float(se[2])
            if significant:
                warnings.append("두 블록의 차이가 위치에 따라 달라집니다(기울기가 유의함) - 평면 모형을 고려하세요.")
    shift.x0, shift.y0 = x0, y0
    shift.n_cells, shift.cell_size_m, shift.overlap_area_m2 = len(shared), cell, len(shared) * cell * cell
    shift.residual_std_nt = _mad_std(residual)
    shift.gradient_significant = significant
    if np.isfinite(dalt).any():
        shift.altitude_difference_m = float(np.nanmedian(dalt))
        if abs(shift.altitude_difference_m) > ALTITUDE_CAUTION_M:
            warnings.append(
                f"겹치는 구역에서 두 블록의 비행 고도가 {abs(shift.altitude_difference_m):.1f} m 다릅니다 - 이상의 크기 자체가 달라지므로 "
                "차이가 상수가 아닙니다. 고도 정규화를 먼저 적용한 뒤 맞추는 편이 정확합니다."
            )
    if len(groups) < 5:
        warnings.append(f"겹치는 구역이 좁아(독립 구역 {len(groups)}개) 불확실도 추정이 거칩니다.")
    shift.warnings = warnings
    return shift
