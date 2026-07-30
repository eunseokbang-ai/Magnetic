"""In-memory per-project state and pipeline orchestration."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from .io_.base_loader import load_base_csvs
from .io_.drone_loader import load_drone_csvs
from .models import GridRequest, ManualExcludeRequest, ProcessParams, TransformRequest
from .processing.diurnal import apply_diurnal_correction
from .processing.filters import lowpass_filter
from .processing.gridding import GridResult, grid_points
from .processing.igrf import compute_igrf_total_field, mean_inclination_declination
from .processing.leveling import HeadingLevelingResult, apply_heading_correction, compute_heading_correction
from .processing.lines import LineDetectionParams, detect_lines, estimate_line_spacing_m
from .processing.render import grid_to_png_overlay
from .processing.transforms import analytic_signal, reduction_to_equator, reduction_to_pole, vertical_derivative

DEFAULT_CMAPS = {
    "point": "viridis",
    "anomaly_grid": "RdYlBu_r",
    "tmi_grid": "viridis",
    "derivative": "RdYlBu_r",
}


class ProjectError(ValueError):
    pass


@dataclass
class Project:
    id: str
    drone_raw: pd.DataFrame | None = None
    base_raw: pd.DataFrame | None = None
    processed: pd.DataFrame | None = None
    # point_id -> True (force include, even if auto-excluded) | False (force
    # exclude, even if auto-included). Absent point_ids fall back to the
    # automatic line_id>=0 result.
    manual_overrides: dict = field(default_factory=dict)
    utm_epsg: int | None = None
    dominant_azimuth_deg: float | None = None
    line_spacing_m: float | None = None
    heading_leveling: HeadingLevelingResult | None = None
    inclination_deg: float | None = None
    declination_deg: float | None = None
    diurnal_info: dict | None = None
    last_params: ProcessParams | None = None
    grid_cache: dict = field(default_factory=dict)

    def load_drone(self, buffers: list) -> dict:
        self.drone_raw = load_drone_csvs(buffers)
        return self.drone_summary()

    def load_base(self, buffers: list) -> dict:
        self.base_raw = load_base_csvs(buffers)
        return self.base_summary()

    def drone_summary(self) -> dict:
        if self.drone_raw is None:
            return {}
        d = self.drone_raw
        return {
            "n_points": len(d),
            "time_range": [d["timestamp"].min().isoformat(), d["timestamp"].max().isoformat()],
            "lat_range": [float(d["lat"].min()), float(d["lat"].max())],
            "lon_range": [float(d["lon"].min()), float(d["lon"].max())],
            "mag_range": [float(d["mag_raw"].min()), float(d["mag_raw"].max())],
        }

    def base_summary(self) -> dict:
        if self.base_raw is None:
            return {}
        b = self.base_raw
        return {
            "n_points": len(b),
            "time_range": [b["timestamp"].min().isoformat(), b["timestamp"].max().isoformat()],
            "mag_range": [float(b["mag"].min()), float(b["mag"].max())],
        }

    def run_pipeline(self, params: ProcessParams) -> dict:
        if self.drone_raw is None:
            raise ProjectError("드론 자료를 먼저 업로드하세요.")
        if self.base_raw is None:
            raise ProjectError("베이스(일변화) 자료를 먼저 업로드하세요.")

        df = self.drone_raw.copy()
        df["mag_filtered"] = lowpass_filter(
            df["mag_raw"].to_numpy(), df["timestamp"], cutoff_hz=params.filter_cutoff_hz
        )

        line_params = LineDetectionParams(**params.line_params.model_dump())
        df = detect_lines(df, line_params)
        self.utm_epsg = df.attrs["utm_epsg"]
        self.dominant_azimuth_deg = df.attrs["dominant_azimuth_deg"]
        self.line_spacing_m = estimate_line_spacing_m(df, self.dominant_azimuth_deg)

        diurnal_result = apply_diurnal_correction(
            df["timestamp"],
            df["mag_filtered"].to_numpy(),
            self.base_raw,
            time_offset_seconds=params.diurnal_params.time_offset_seconds,
            reference=params.diurnal_params.reference,
        )
        df["mag_diurnal_corrected"] = diurnal_result.corrected
        self.diurnal_info = {
            "coverage_pct": diurnal_result.coverage_pct,
            "has_overlap": diurnal_result.has_overlap,
            "base_reference_value": diurnal_result.base_reference_value,
            "base_time_range": [str(diurnal_result.base_time_range[0]), str(diurnal_result.base_time_range[1])],
            "drone_time_range": [str(diurnal_result.drone_time_range[0]), str(diurnal_result.drone_time_range[1])],
        }

        igrf_total = compute_igrf_total_field(
            df["lat"].to_numpy(), df["lon"].to_numpy(), df["altitude_ellipsoidal_m"].to_numpy(), df["timestamp"]
        )
        df["igrf_total"] = igrf_total
        df["anomaly"] = df["mag_diurnal_corrected"] - igrf_total
        df["tmi"] = df["mag_diurnal_corrected"]

        self.inclination_deg, self.declination_deg = mean_inclination_declination(
            df["lat"].to_numpy(), df["lon"].to_numpy(), df["altitude_ellipsoidal_m"].to_numpy(), df["timestamp"]
        )

        hp = params.heading_correction
        if hp.enabled:
            self.heading_leveling = compute_heading_correction(
                df,
                "anomaly",
                self.dominant_azimuth_deg,
                self.line_spacing_m,
                quiet_percentile=hp.quiet_percentile,
                max_match_distance_m=hp.max_match_distance_m,
            )
            df["anomaly"] = apply_heading_correction(df, "anomaly", self.heading_leveling)
            df["tmi"] = apply_heading_correction(df, "tmi", self.heading_leveling)
        else:
            self.heading_leveling = HeadingLevelingResult(False, "사용자가 헤딩 보정을 비활성화했습니다.", None, 0, 0)

        self.processed = df
        self.manual_overrides = {}
        self.grid_cache = {}
        self.last_params = params
        return self.process_summary()

    def process_summary(self) -> dict:
        df = self.processed
        active = self._active_mask()
        n_lines = int(df.loc[df["line_id"] >= 0, "line_id"].nunique())
        return {
            "n_points": len(df),
            "n_lines": n_lines,
            "n_kept": int(active.sum()),
            "n_excluded_auto": int((df["line_id"] < 0).sum()),
            "n_manual_included": sum(1 for v in self.manual_overrides.values() if v),
            "n_manual_excluded": sum(1 for v in self.manual_overrides.values() if not v),
            "dominant_azimuth_deg": self.dominant_azimuth_deg,
            "line_spacing_m": self.line_spacing_m,
            "inclination_deg": self.inclination_deg,
            "declination_deg": self.declination_deg,
            "diurnal": self.diurnal_info,
            "heading_correction": _heading_correction_summary(self.heading_leveling),
            "anomaly_stats": _stats(df.loc[active, "anomaly"]),
            "tmi_stats": _stats(df.loc[active, "tmi"]),
            "lines": _line_summaries(df, self.heading_leveling),
        }

    def _active_mask(self) -> pd.Series:
        df = self.processed
        auto = df["line_id"] >= 0
        if not self.manual_overrides:
            return auto
        override = df["point_id"].map(self.manual_overrides)
        return override.where(override.notna(), auto).astype(bool)

    def get_points(self, value: str) -> list[dict]:
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed
        active = self._active_mask()
        col = "anomaly" if value == "anomaly" else "tmi"
        out = pd.DataFrame(
            {
                "point_id": df["point_id"],
                "lat": df["lat"],
                "lon": df["lon"],
                "value": df[col],
                "line_id": df["line_id"],
                "excluded": ~active,
                "timestamp": df["timestamp"].astype(str),
            }
        )
        return out.to_dict(orient="records")

    def set_manual_exclude(self, req: ManualExcludeRequest) -> dict:
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed

        if req.mode == "reset":
            self.manual_overrides = {}
            self.grid_cache = {}
            return self.process_summary()

        if req.mode == "lines":
            if not req.line_ids:
                raise ProjectError("line_ids가 필요합니다.")
            target_ids = set(df.loc[df["line_id"].isin(req.line_ids), "point_id"])
        else:
            if not req.polygon or len(req.polygon) < 3:
                raise ProjectError("polygon은 최소 3개의 [lat, lon] 좌표가 필요합니다.")
            poly_path = MplPath([(pt[1], pt[0]) for pt in req.polygon])  # (lon, lat)
            inside = poly_path.contains_points(np.column_stack([df["lon"], df["lat"]]))
            target_ids = set(df.loc[inside, "point_id"])

        # action="include" force-includes points regardless of automatic
        # line detection (e.g. restoring a turbulence segment the auto
        # detector dropped); action="exclude" force-excludes regardless.
        forced_value = req.action == "include"
        for pid in target_ids:
            self.manual_overrides[int(pid)] = forced_value
        self.grid_cache = {}
        return self.process_summary()

    def _resolve_max_distance(self, cell_size_m: float, max_distance_m: float | None) -> float:
        if max_distance_m is not None:
            return max_distance_m
        if self.line_spacing_m:
            return max(2.0 * cell_size_m, 0.6 * self.line_spacing_m)
        return 2.0 * cell_size_m

    def _grid_for(self, value: str, cell_size_m: float, method: str = "spline", max_distance_m: float | None = None) -> GridResult:
        resolved_max_distance = self._resolve_max_distance(cell_size_m, max_distance_m)
        key = (value, cell_size_m, method, resolved_max_distance)
        if key in self.grid_cache:
            return self.grid_cache[key]
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed
        active = self._active_mask()
        col = "anomaly" if value == "anomaly" else "tmi"
        result = grid_points(
            df.loc[active, "x"].to_numpy(),
            df.loc[active, "y"].to_numpy(),
            df.loc[active, col].to_numpy(),
            cell_size_m,
            method=method,
            max_distance_m=resolved_max_distance,
        )
        self.grid_cache[key] = result
        return result

    def get_grid_overlay(self, req: GridRequest) -> dict:
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        cmap = req.cmap or (DEFAULT_CMAPS["anomaly_grid"] if req.value == "anomaly" else DEFAULT_CMAPS["tmi_grid"])
        overlay = grid_to_png_overlay(
            grid.values,
            grid.easting,
            grid.northing,
            self.utm_epsg,
            cmap_name=cmap,
            symmetric=(req.value == "anomaly"),
            vmin=req.vmin,
            vmax=req.vmax,
        )
        overlay["stats"] = _stats(pd.Series(grid.values.ravel()))
        overlay["cell_size_m"] = grid.cell_size_m
        return overlay

    def get_transform_overlay(self, req: TransformRequest) -> dict:
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        if self.inclination_deg is None:
            raise ProjectError("IGRF 계산이 필요합니다 (자료 처리를 먼저 실행하세요).")

        if req.transform == "rtp":
            values = reduction_to_pole(grid.values, grid.cell_size_m, self.inclination_deg, self.declination_deg)
            symmetric = True
        elif req.transform == "rte":
            values = reduction_to_equator(grid.values, grid.cell_size_m, self.inclination_deg, self.declination_deg)
            symmetric = True
        elif req.transform == "1vd":
            values = vertical_derivative(grid.values, grid.cell_size_m, order=1)
            symmetric = True
        elif req.transform == "as":
            values = analytic_signal(grid.values, grid.cell_size_m)
            symmetric = False
        else:
            raise ProjectError(f"알 수 없는 변환입니다: {req.transform}")

        cmap = req.cmap or DEFAULT_CMAPS["derivative"]
        overlay = grid_to_png_overlay(
            values, grid.easting, grid.northing, self.utm_epsg, cmap_name=cmap, symmetric=symmetric, vmin=req.vmin, vmax=req.vmax
        )
        overlay["stats"] = _stats(pd.Series(values.ravel()))
        overlay["cell_size_m"] = grid.cell_size_m
        overlay["transform"] = req.transform
        return overlay


def _stats(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
    }


def _line_summaries(df: pd.DataFrame, heading_leveling: "HeadingLevelingResult | None" = None) -> list[dict]:
    groups = heading_leveling.line_groups if heading_leveling else {}
    shifts = heading_leveling.line_shifts if heading_leveling else {}
    lines = []
    for line_id, g in df[df["line_id"] >= 0].groupby("line_id"):
        lines.append(
            {
                "line_id": int(line_id),
                "n_points": int(len(g)),
                "start_time": g["timestamp"].min().isoformat(),
                "end_time": g["timestamp"].max().isoformat(),
                "length_m": float(np.hypot(np.diff(g["x"]), np.diff(g["y"])).sum()),
                "centroid_lat": float(g["lat"].mean()),
                "centroid_lon": float(g["lon"].mean()),
                "heading_group": groups.get(line_id),
                "heading_shift_nt": shifts.get(line_id),
            }
        )
    return lines


def _heading_correction_summary(result: "HeadingLevelingResult | None") -> dict:
    if result is None:
        return {"applied": False, "reason": None}
    return {
        "applied": result.applied,
        "reason": result.reason,
        "offset_nt": result.offset_nt,
        "n_matched_pairs": result.n_matched_pairs,
        "n_quiet_pairs": result.n_quiet_pairs,
    }


class ProjectStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._projects: dict[str, Project] = {}

    def create(self) -> Project:
        project_id = uuid.uuid4().hex[:12]
        project = Project(id=project_id)
        with self._lock:
            self._projects[project_id] = project
        return project

    def get(self, project_id: str) -> Project:
        with self._lock:
            project = self._projects.get(project_id)
        if project is None:
            raise ProjectError(f"프로젝트를 찾을 수 없습니다: {project_id}")
        return project


store = ProjectStore()
