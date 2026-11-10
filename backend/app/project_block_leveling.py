"""Levelling one block of files onto another from their shared ground, or
by a sensor offset known from a ground test - see
magcore/block_leveling.py. Part of store.Project.

A "block" here is a set of uploaded files (one flight day, one sensor).
The shift found for it is stored as a model (constant, or a plane), not
as corrected numbers, and is subtracted from those files' readings at
the start of the edit chain, next to the per-flight offsets from repeat
passes: both are levelling, and everything after them (structure
removal, smoothing, height normalisation) should see levelled data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from magcore.block_leveling import BlockLevelingError, BlockShift, estimate_block_shift

from .models import BlockShiftApplyRequest, BlockShiftRequest
from .project_common import ProjectError

# A plane's correction at the far end of the target block, relative to its
# uncertainty over the overlap, beyond which the user is told.
EXTRAPOLATION_CAUTION = 3.0


class BlockLevelingMixin:
    def block_files(self) -> dict:
        """The uploaded files as the things a block is made of: name, how
        many readings, when, where, and any shift applied to them."""
        if self.drone_raw is None:
            raise ProjectError("드론 자료를 먼저 업로드하세요.")
        raw = self.drone_raw
        if "source_file_index" not in raw.columns:
            return {"files": [], "shifts": self._block_shift_summaries()}
        names = list(raw.attrs.get("source_files") or [])
        shifted = {int(i): k for k, s in enumerate(self.block_shifts) for i in s["files"]}
        on_line = None
        if self.processed is not None and "source_file_index" in self.processed.columns:
            on_line = self.processed.loc[self._active_mask()].groupby("source_file_index").size().to_dict()
        files = []
        for idx, g in raw.groupby("source_file_index"):
            idx = int(idx)
            files.append({
                "index": idx,
                "name": names[idx] if idx < len(names) and names[idx] else f"파일 {idx + 1}",
                "n_points": int(len(g)),
                "n_on_lines": int(on_line.get(idx, 0)) if on_line is not None else None,
                "time_range": [g["timestamp"].min().isoformat(), g["timestamp"].max().isoformat()],
                "center": [float(g["lat"].mean()), float(g["lon"].mean())],
                "shift_index": shifted.get(idx),
            })
        return {"files": files, "shifts": self._block_shift_summaries()}

    def _block_shift_summaries(self) -> list[dict]:
        return [dict(s) for s in self.block_shifts]

    def _block_shifted(self, base: pd.DataFrame | None, skip_files: set[int] | None = None) -> pd.DataFrame | None:
        """`base` with every stored block shift subtracted from its files
        (except shifts touching `skip_files` - used when re-estimating one).
        Returns `base` itself when there is nothing to do."""
        if base is None or not self.block_shifts or "source_file_index" not in base.columns:
            return base
        file_index = base["source_file_index"].to_numpy()
        total = np.zeros(len(base), dtype=float)
        touched = False
        for s in self.block_shifts:
            if skip_files and skip_files & set(s["files"]):
                continue
            rows = np.isin(file_index, s["files"])
            if not rows.any():
                continue
            total[rows] += BlockShift.from_dict(s).value_at(base["x"].to_numpy()[rows], base["y"].to_numpy()[rows])
            touched = True
        if not touched:
            return base
        out = base.copy()
        for col in ("anomaly", "tmi"):
            out[col] = out[col].to_numpy() - total
        return out

    def _block_points(self, files: list[int], skip: set[int]) -> pd.DataFrame:
        """On-line readings of `files`, levelled by everything except the
        shifts being re-estimated."""
        base = self._block_shifted(self._repeat_levelled_base(), skip_files=skip)
        active = self._active_mask()
        rows = active.to_numpy() & np.isin(base["source_file_index"].to_numpy(), files)
        return base.loc[rows]

    def _estimate_block_shift(self, req: BlockShiftRequest) -> BlockShift:
        if self.processed is None or self.processed_base is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        if "source_file_index" not in self.processed_base.columns:
            raise ProjectError("파일 구분 정보가 없어 블록을 나눌 수 없습니다.")
        reference, target = sorted(set(req.reference_files)), sorted(set(req.target_files))
        if not reference or not target:
            raise ProjectError("기준 블록과 맞출 블록의 파일을 각각 하나 이상 고르세요.")
        if set(reference) & set(target):
            raise ProjectError("같은 파일이 기준 블록과 맞출 블록에 함께 들어 있습니다.")
        skip = set(target)
        ref, tgt = self._block_points(reference, skip), self._block_points(target, skip)
        cell = float(req.cell_size_m) if req.cell_size_m else float(self.line_spacing_m or 0.0)
        if cell <= 0:
            raise ProjectError("측선 간격을 알 수 없어 셀 크기를 정할 수 없습니다 - 셀 크기를 직접 지정하세요.")
        try:
            return estimate_block_shift(
                ref["x"], ref["y"], ref["anomaly"], tgt["x"], tgt["y"], tgt["anomaly"], cell_size_m=cell, model=req.model,
                ref_alt=ref["altitude_ellipsoidal_m"], tgt_alt=tgt["altitude_ellipsoidal_m"],
            )
        except BlockLevelingError as exc:
            raise ProjectError(str(exc)) from exc

    def estimate_block_shift(self, req: BlockShiftRequest) -> dict:
        """Measure only: what the target block reads above the reference
        over their shared ground. Nothing is changed."""
        shift = self._estimate_block_shift(req)
        target = sorted(set(req.target_files))
        return {"shift": self._with_reach(shift, target), "reference_files": sorted(set(req.reference_files)), "target_files": target}

    def _with_reach(self, shift: BlockShift, target: list[int]) -> dict:
        """The shift as a dict, plus how sure the correction still is at
        the far end of the block it will be applied to: the largest
        +-1 sigma over the target's readings. For a constant that is just
        its own uncertainty; for a plane it grows with distance from the
        overlap, and past EXTRAPOLATION_CAUTION times the overlap's own
        figure the result carries a warning."""
        out = shift.to_dict()
        base = self.processed_base
        rows = np.isin(base["source_file_index"].to_numpy(), target)
        sigma = shift.uncertainty_at(base["x"].to_numpy()[rows], base["y"].to_numpy()[rows])
        worst = float(sigma.max()) if len(sigma) else shift.constant_se_nt
        out["max_uncertainty_nt"] = round(worst, 3)
        if shift.model == "plane" and worst > max(EXTRAPOLATION_CAUTION * shift.constant_se_nt, 2.0):
            reach_km = float(np.hypot(base["x"].to_numpy()[rows] - shift.x0, base["y"].to_numpy()[rows] - shift.y0).max()) / 1000.0
            out["warnings"] = out["warnings"] + [
                f"평면을 겹치는 구역 중심에서 최대 {reach_km:.1f} km 떨어진 곳까지 연장해 적용합니다 - 그곳의 보정 불확실도는 ±{worst:.1f} nT로, "
                f"겹치는 구역(±{shift.constant_se_nt:.1f} nT)보다 큽니다."
            ]
        return out

    def apply_block_shift(self, req: BlockShiftApplyRequest) -> dict:
        with self._edit_lock:
            return self._apply_block_shift_unlocked(req)

    def _apply_block_shift_unlocked(self, req: BlockShiftApplyRequest) -> dict:
        """Subtract a block's shift from its files, replace the one it had,
        or take it off again.

        mode "apply" estimates it from the overlap (as estimate_block_shift
        does) and stores that; "manual" stores a constant the user knows
        from elsewhere - a side-by-side sensor comparison on the ground;
        "reset" removes the shifts on the given files, or all of them."""
        if self.processed_base is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        target = sorted(set(req.target_files))
        if req.mode == "reset":
            self.block_shifts = [s for s in self.block_shifts if target and not (set(s["files"]) & set(target))]
        else:
            if not target:
                raise ProjectError("맞출 블록의 파일을 고르세요.")
            if req.mode == "manual":
                if req.manual_constant_nt is None:
                    raise ProjectError("적용할 오프셋(nT)을 입력하세요.")
                entry = {**BlockShift("constant", float(req.manual_constant_nt), 0.0).to_dict(), "source": "manual", "reference_files": []}
            else:
                shift = self._estimate_block_shift(req)
                entry = {**self._with_reach(shift, target), "source": "overlap", "reference_files": sorted(set(req.reference_files))}
            entry["files"] = target
            # one shift per file: a new one replaces whatever those files had
            self.block_shifts = [s for s in self.block_shifts if not (set(s["files"]) & set(target))] + [entry]
        self._rebuild_processed()
        self.mark_changed()
        return {"shifts": self._block_shift_summaries(), "process_summary": self.process_summary()}
