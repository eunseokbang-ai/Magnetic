"""Sensor height correction: the GPS height of the sensor, anchored to the
ground it stood on before takeoff and after landing - see
magcore/sensor_height.py for the method and what it is worth.

What it changes is the altitude columns of the processed table (the
receiver's own are kept beside them), and so everything that reads a
height: the inversion's observation points, the flight-height
normalisation, the height-above-terrain report. No magnetic value is
touched.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from magcore import progress
from magcore.sensor_height import (
    GROUND_MAX_RISE_M,
    GROUND_MAX_SPEED_MPS,
    SensorHeightCorrection,
    _standing,
    build_correction,
    leave_out_validation,
)
from magcore.terrain import TerrainError, sample_dem_at

from .project_common import ProjectError, _as_job

_EXCLUDED_LABELS = {
    "no_terrain": "지형 자료가 그 지점을 덮지 않음",
    "sea_level": "지형 자료가 그 지점을 해수면(0 m)으로 봄",
    "outlier": "다른 검사와 25 m 넘게 다름 (공중에서 시작·종료된 기록으로 보임)",
}


class SensorHeightMixin:
    # ------------------------------------------------------------------ inputs
    def _sensor_height_inputs(self, df: pd.DataFrame) -> dict:
        """The receiver's own heights and what goes with them, from a
        processed table whether or not it has been corrected already."""
        def col(name):
            return df[name].to_numpy(dtype=float) if name in df.columns else np.full(len(df), np.nan)

        msl = col("altitude_gps_msl_m") if "altitude_gps_msl_m" in df.columns else col("altitude_msl_m")
        ell = col("altitude_gps_ellipsoidal_m") if "altitude_gps_ellipsoidal_m" in df.columns else col("altitude_ellipsoidal_m")
        geoid = col("geoid_separation_m")
        if not np.isfinite(msl).any() and np.isfinite(ell).any() and np.isfinite(geoid).any():
            msl = ell - geoid
        return {
            "time_s": df["timestamp"].astype("int64").to_numpy() / 1e9,
            "msl": msl,
            "ellipsoidal": ell,
            "speed": col("speed_mps"),
            "flight": df["source_file_index"].to_numpy() if "source_file_index" in df.columns else np.zeros(len(df), dtype=int),
            "lat": col("lat"),
            "lon": col("lon"),
            "geoid_separation_m": float(np.nanmedian(geoid)) if np.isfinite(geoid).any() else None,
        }

    @staticmethod
    def _standing_rows(inp: dict) -> np.ndarray:
        """Rows of every stretch a ground check could come from - the only
        places the terrain model has to be read."""
        rows = []
        for f in np.unique(inp["flight"]):
            idx = np.flatnonzero(inp["flight"] == f)
            idx = idx[np.argsort(inp["time_s"][idx], kind="stable")]
            for from_start in (True, False):
                span = _standing(inp["msl"][idx], inp["speed"][idx], inp["time_s"][idx], from_start)
                if span is not None:
                    rows.append(idx[span])
        return np.concatenate(rows) if rows else np.zeros(0, dtype=int)

    # ------------------------------------------------------------------ the table
    def _write_sensor_height(self, df: pd.DataFrame) -> pd.DataFrame:
        """df with its altitude columns corrected by self.sensor_height (the
        receiver's own kept as altitude_gps_*), or with the receiver's own
        put back when there is no correction."""
        inp = self._sensor_height_inputs(df)
        if self.sensor_height is None:
            if "altitude_gps_msl_m" in df.columns:
                df["altitude_msl_m"] = df["altitude_gps_msl_m"]
                df["altitude_ellipsoidal_m"] = df["altitude_gps_ellipsoidal_m"]
                df = df.drop(columns=["altitude_gps_msl_m", "altitude_gps_ellipsoidal_m", "altitude_sigma_m"], errors="ignore")
            return df
        error, sigma = self.sensor_height.predict(inp["time_s"])
        df["altitude_gps_msl_m"] = inp["msl"]
        df["altitude_gps_ellipsoidal_m"] = inp["ellipsoidal"]
        df["altitude_msl_m"] = inp["msl"] - error
        df["altitude_ellipsoidal_m"] = inp["ellipsoidal"] - error
        df["altitude_sigma_m"] = sigma
        return df

    def _with_sensor_height(self, df: pd.DataFrame) -> pd.DataFrame:
        """For the pipeline: a fresh processed table, carrying the
        correction already in force. The correction is a function of time
        only, so reprocessing the same flights with other settings does
        not invalidate it; new flights do (load_drone clears it)."""
        if self.sensor_height is None:
            return df
        return self._write_sensor_height(df)

    # ------------------------------------------------------------------ reports
    def _sensor_height_summary(self) -> dict:
        corr = self.sensor_height
        if corr is None:
            return {"applied": False}
        return {"applied": True, "model": corr.model.to_dict(), "resting_offset_m": corr.resting_offset_m, **corr.stats}

    def sensor_height_status(self) -> dict:
        """Whether the correction can be made, what it would rest on, and
        the state of the one in force (with its checks, flight by flight)."""
        if self.processed_base is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        inp = self._sensor_height_inputs(self.processed_base)
        has_height = bool(np.isfinite(inp["msl"]).any())
        files = (self.drone_raw.attrs.get("source_files") if self.drone_raw is not None else None) or []
        checks = []
        if self.sensor_height is not None:
            for c in self.sensor_height.checks:
                d = c.to_dict()
                d["file"] = files[c.flight] if 0 <= c.flight < len(files) else None
                d["excluded_label"] = _EXCLUDED_LABELS.get(c.excluded_reason) if c.excluded_reason else None
                checks.append(d)
        return {
            "available": has_height,
            "reason": None if has_height else "GPS 고도(해수면 기준)가 자료에 없어 센서 고도 보정을 할 수 없습니다.",
            "dem_name": self.dem_name,
            "has_dem": self.dem_bytes is not None,
            "geoid_separation_m": inp["geoid_separation_m"],
            "n_flights": int(len(np.unique(inp["flight"]))),
            **self._sensor_height_summary(),
            "checks": checks,
        }

    # ------------------------------------------------------------------ apply
    def apply_sensor_height(self, mode: str = "apply", dem_bytes: bytes | None = None, dem_name: str | None = None,
                            resting_offset_m: float = 0.0, exclude_sea_level: bool = True) -> dict:
        with self._edit_lock:
            return self._apply_sensor_height_unlocked(mode, dem_bytes, dem_name, resting_offset_m, exclude_sea_level)

    @_as_job("센서 고도 보정", rollback=True)
    def _apply_sensor_height_unlocked(self, mode, dem_bytes, dem_name, resting_offset_m, exclude_sea_level) -> dict:
        """Measure the ground checks against a terrain model (the one given
        here - a DSM that is right at the takeoff pads is the best choice -
        or the project's), fit the error model and correct every sample's
        height; or take the correction off again."""
        if self.processed_base is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        base = self.processed_base
        if mode == "reset":
            self.sensor_height = None
        else:
            terrain = dem_bytes if dem_bytes is not None else self.dem_bytes
            terrain_name = dem_name if dem_bytes is not None else self.dem_name
            if terrain is None:
                raise ProjectError("센서 고도 보정에는 지형 자료(DEM 또는 DSM GeoTIFF)가 필요합니다. 파일을 선택하거나 역산용 DEM을 먼저 올리세요.")
            inp = self._sensor_height_inputs(base)
            if not np.isfinite(inp["msl"]).any():
                raise ProjectError("GPS 고도(해수면 기준)가 자료에 없어 센서 고도 보정을 할 수 없습니다.")
            progress.report("지상 구간 찾기", 0.1)
            rows = self._standing_rows(inp)
            ground = np.full(len(base), np.nan)
            if len(rows):
                try:
                    ground[rows] = sample_dem_at(io.BytesIO(terrain), inp["lon"][rows], inp["lat"][rows])
                except TerrainError as exc:
                    raise ProjectError(str(exc)) from exc
            progress.report("오차 모델 맞춤", 0.4)
            corr = build_correction(inp["time_s"], inp["msl"], inp["speed"], inp["flight"], ground, inp["lat"], inp["lon"],
                                    resting_offset_m=resting_offset_m, exclude_sea_level=exclude_sea_level)
            used = [c for c in corr.checks if c.used]
            if not used:
                reasons = sorted({_EXCLUDED_LABELS.get(c.excluded_reason, c.excluded_reason) for c in corr.checks})
                raise ProjectError(
                    "쓸 수 있는 지상 검사가 없습니다 - "
                    + ("; ".join(reasons) if reasons else f"이륙 전·착륙 후에 센서가 멈춰 있는 구간(속도 {GROUND_MAX_SPEED_MPS} m/s 이하, 높이 변화 {GROUND_MAX_RISE_M} m 이내)을 찾지 못했습니다")
                    + ". 이착륙 지점의 표고가 들어 있는 지형 자료(가능하면 DSM)를 쓰세요."
                )
            progress.report("보정 적용", 0.7)
            error, sigma = corr.predict(inp["time_s"])
            on_line = (base["line_id"].to_numpy() >= 0) if "line_id" in base.columns else np.ones(len(base), dtype=bool)
            if not on_line.any():
                on_line = np.ones(len(base), dtype=bool)
            excluded: dict = {}
            for c in corr.checks:
                if not c.used:
                    excluded[c.excluded_reason] = excluded.get(c.excluded_reason, 0) + 1
            all_flights = [int(f) for f in np.unique(inp["flight"])]
            covered = {c.flight for c in used}
            files = (self.drone_raw.attrs.get("source_files") if self.drone_raw is not None else None) or []
            corr.stats = {
                "terrain_name": terrain_name,
                "n_checks": len(corr.checks),
                "n_checks_used": len(used),
                "n_checks_excluded": excluded,
                "excluded_labels": {k: _EXCLUDED_LABELS.get(k, k) for k in excluded},
                "n_flights": len(all_flights),
                "flights_without_checks": [files[f] if 0 <= f < len(files) else f"비행 {f + 1}" for f in all_flights if f not in covered],
                "check_error_m": {
                    "median": round(float(np.median([c.error_m for c in used])), 2),
                    "min": round(float(min(c.error_m for c in used)), 2),
                    "max": round(float(max(c.error_m for c in used)), 2),
                },
                "correction_on_lines_m": {
                    "median": round(float(np.median(error[on_line])), 2),
                    "min": round(float(error[on_line].min()), 2),
                    "max": round(float(error[on_line].max()), 2),
                },
                "sigma_on_lines_m": {
                    "median": round(float(np.median(sigma[on_line])), 2),
                    "p95": round(float(np.percentile(sigma[on_line], 95)), 2),
                    "max": round(float(sigma[on_line].max()), 2),
                },
                "uncorrected_rms_m": round(float(np.sqrt(corr.model.mean_m ** 2 + corr.model.sill_m2)), 2),
                "validation": [v for v in (leave_out_validation(corr.checks, corr.model, 0.0), leave_out_validation(corr.checks, corr.model, 5.0)) if v],
                "height_above_terrain_m": self._clearance_change(terrain, inp, error, on_line),
            }
            self.sensor_height = corr

        self.processed_base = self._write_sensor_height(base)
        # the equivalent layers were fitted at the old heights
        normalization_reset = self.height_normalization is not None
        if normalization_reset:
            self.height_normalization = None
            self._height_correction_cache = None
        self._rebuild_processed()
        self.mark_changed()
        return {
            **self.process_summary(),
            "exclusion": self.get_exclusion_state(),
            "sensor_height_status": self.sensor_height_status(),
            "height_normalization_reset": normalization_reset,
            "inversion_stale": self.inversion_result is not None,
        }

    @staticmethod
    def _clearance_change(terrain: bytes, inp: dict, error: np.ndarray, on_line: np.ndarray) -> dict | None:
        """Height of the readings on the lines above the terrain model the
        checks were made against, before and after - the number to hold
        against the height the mission was flown at."""
        rows = np.flatnonzero(on_line & np.isfinite(inp["msl"]))
        if len(rows) == 0:
            return None
        rows = rows[:: max(1, len(rows) // 20000)]
        try:
            ground = sample_dem_at(io.BytesIO(terrain), inp["lon"][rows], inp["lat"][rows])
        except TerrainError:
            return None
        ok = np.isfinite(ground)
        if ok.sum() < 10:
            return None
        before = inp["msl"][rows][ok] - ground[ok]
        after = before - error[rows][ok]
        return {
            "n": int(ok.sum()),
            "before_median": round(float(np.median(before)), 1),
            "after_median": round(float(np.median(after)), 1),
            "after_p5": round(float(np.percentile(after, 5)), 1),
            "after_p95": round(float(np.percentile(after, 95)), 1),
        }
