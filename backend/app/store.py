"""In-memory per-project state and pipeline orchestration."""
from __future__ import annotations

import io
import json
import threading
import uuid
import zipfile
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import rasterio
from matplotlib.path import Path as MplPath
from pyproj import Transformer

from .io_.base_loader import load_base_csvs
from .io_.drone_loader import load_drone_csvs
from .models import (
    EulerDeconvolutionRequest,
    GridRequest,
    InversionParams,
    InversionSectionRequest,
    InversionSliceRequest,
    ManualExcludeRequest,
    ProcessParams,
    TargetDetectionRequest,
    TransformRequest,
)
from .processing.crossover_leveling import CrossoverLevelingResult, apply_crossover_leveling, compute_crossover_leveling
from .processing.despike import despike
from .processing.dipole_fit import classify_moment, detect_targets
from .processing.diurnal import apply_diurnal_correction
from .processing.filters import lowpass_filter, moving_average_filter, savgol_filter_1d
from .processing.gridding import GridResult, grid_points
from .processing.igrf import compute_igrf_total_field, mean_field_intensity_nt, mean_inclination_declination
from .processing.inversion import (
    InversionError,
    InversionMesh,
    InversionResult,
    build_mesh,
    build_sensitivity_matrix,
    horizontal_slice,
    invert,
    render_section_png,
    upsample_susceptibility,
)
from .processing.inversion import vertical_section as _inversion_vertical_section
from .processing.inversion_auto import suggest_mesh_params
from .processing.leveling import HeadingLevelingResult, apply_heading_correction, compute_heading_correction
from .processing.lines import (
    LineDetectionParams,
    detect_lines,
    detect_tie_lines,
    estimate_line_spacing_m,
    project_to_local_xy,
    resolve_korea_projection_epsg,
)
from .processing.contours import compute_contours
from .processing.euler_deconvolution import run_euler_deconvolution as _euler_deconvolution_solve
from .processing.geology_sample import GeologySampleError, sample_geotiff_at_point
from .processing.gps_lag import apply_gps_mag_lag
from .processing.overlay_image import OverlayImageError, load_geotiff_overlay
from .processing.report import generate_report_markdown
from .processing.render import (
    grid_to_geotiff_bytes,
    grid_to_geotiff_bytes_colored,
    grid_to_png_overlay,
    grid_to_surfer_grd_bytes,
    grid_to_xyz_bytes,
    polygon_to_bln_bytes,
)
from .processing.terrain import TerrainError, estimate_ground_elevation, load_dem_geotiff
from .processing.transforms import (
    analytic_signal,
    derivative_easting,
    derivative_northing,
    reduction_to_equator,
    reduction_to_pole,
    second_derivative_ee,
    second_derivative_en,
    second_derivative_ez,
    second_derivative_nn,
    second_derivative_nz,
    theta_map,
    tilt_angle,
    total_horizontal_derivative,
    upward_continuation,
    vertical_derivative,
)
from .processing.microlevel import apply_microleveling
from .processing.trend import remove_regional_trend

DEFAULT_CMAPS = {
    "point": "viridis",
    "anomaly_grid": "RdYlBu_r",
    "tmi_grid": "viridis",
    "derivative": "RdYlBu_r",
}

# Size caps for the 3D inversion mesh/observation grid. The normal
# equations are solved in "data space" (see processing/inversion.py),
# whose cost scales with n_obs^2 * n_active - these limits were picked
# from a benchmark of that solve (n_obs=3000, n_active=60000 with 6 IRLS
# iterations takes ~90s; n_obs=3500, n_active=90000 extrapolates to
# roughly 3 minutes), so a user asking for more detail still finishes in
# a bounded time rather than growing unbounded. For a large-area survey
# these caps - not just the (now fixed) depth_extent_m ceiling - can
# still force the auto-suggested cell size above the line spacing; see
# the resolution_warning field in run_inversion's response.
N_OBS_CAP = 3500
N_ACTIVE_CAP = 90000

# Target detection grids at a fine (often ~1m) cell size to resolve compact
# near-surface objects; on a large-area survey that same fine cell size
# multiplied by the full flight extent can balloon into a many-million-cell
# grid (benchmarked at ~1.5M cells / ~4-9s total on the sample survey at
# 1m/98m line spacing), so this bounds it - a user surveying a large area
# should raise cell_size_m (or narrow the area) rather than the request
# hanging or exhausting memory.
TARGET_DETECTION_GRID_CELL_CAP = 4_000_000


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
    despike_info: dict | None = None
    crossover_info: dict | None = None
    gps_lag_info: dict | None = None
    inversion_summary_cache: dict | None = None
    euler_summary_cache: dict | None = None
    target_summary_cache: dict | None = None
    reference_layers: dict = field(default_factory=dict)  # name -> raw GeoTIFF bytes
    last_params: ProcessParams | None = None
    grid_cache: dict = field(default_factory=dict)
    transform_cache: dict = field(default_factory=dict)
    dem_bytes: bytes | None = None
    dem_name: str | None = None
    inversion_result: InversionResult | None = None
    inversion_params: InversionParams | None = None
    inversion_field_intensity_nt: float | None = None
    inversion_obs_grid: GridResult | None = None
    inversion_value_field: str | None = None

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
            "n_duplicate_timestamps_removed": d.attrs.get("n_duplicate_timestamps_removed", 0),
            "n_invalid_coords_removed": d.attrs.get("n_invalid_coords_removed", 0),
            "source_formats": d.attrs.get("source_formats", []),
            "median_speed_mps": _median_speed_mps(d),
        }

    def base_summary(self) -> dict:
        if self.base_raw is None:
            return {}
        b = self.base_raw
        return {
            "n_points": len(b),
            "time_range": [b["timestamp"].min().isoformat(), b["timestamp"].max().isoformat()],
            "mag_range": [float(b["mag"].min()), float(b["mag"].max())],
            "n_duplicate_timestamps_removed": b.attrs.get("n_duplicate_timestamps_removed", 0),
        }

    def run_pipeline(self, params: ProcessParams) -> dict:
        if self.drone_raw is None:
            raise ProjectError("드론 자료를 먼저 업로드하세요.")
        if self.base_raw is None:
            raise ProjectError("베이스(일변화) 자료를 먼저 업로드하세요.")

        df = self.drone_raw.copy()

        n_before_lag = len(df)
        if params.gps_mag_lag_seconds != 0.0:
            df = apply_gps_mag_lag(df, params.gps_mag_lag_seconds)
        self.gps_lag_info = {
            "lag_seconds": params.gps_mag_lag_seconds,
            "n_points_dropped": n_before_lag - len(df),
        }

        dp = params.despike_params
        if dp.enabled:
            cleaned, spike_mask = despike(
                df["mag_raw"].to_numpy(),
                dp.window_size,
                dp.threshold_k,
                adaptive=dp.adaptive,
                adaptive_gradient_threshold=dp.adaptive_gradient_threshold,
                adaptive_expand_samples=dp.adaptive_expand_samples,
            )
            df["mag_raw"] = cleaned
            self.despike_info = {
                "enabled": True,
                "n_spikes_removed": int(spike_mask.sum()),
                "pct_spikes_removed": float(spike_mask.mean() * 100.0),
            }
        else:
            self.despike_info = {"enabled": False, "n_spikes_removed": 0, "pct_spikes_removed": 0.0}

        if params.filter_method == "savgol":
            df["mag_filtered"] = savgol_filter_1d(
                df["mag_raw"].to_numpy(), df["timestamp"], params.filter_window_seconds, params.filter_polyorder
            )
        elif params.filter_method == "moving_average":
            df["mag_filtered"] = moving_average_filter(
                df["mag_raw"].to_numpy(), df["timestamp"], params.filter_window_seconds
            )
        else:
            df["mag_filtered"] = lowpass_filter(
                df["mag_raw"].to_numpy(), df["timestamp"], cutoff_hz=params.filter_cutoff_hz
            )

        utm_epsg_override = params.utm_epsg_override
        if utm_epsg_override is None and params.korea_projection:
            utm_epsg_override = resolve_korea_projection_epsg(params.korea_projection, float(df["lon"].mean()))

        line_params = LineDetectionParams(**params.line_params.model_dump())
        df = detect_lines(df, line_params, utm_epsg_override=utm_epsg_override)
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

        # Tie-line membership is computed unconditionally (cheap) so it can
        # always be shown to the user (e.g. in the flight-path editor),
        # independent of whether the crossover-leveling correction itself
        # is enabled.
        cp = params.crossover_leveling
        tie_line_id = detect_tie_lines(df, self.dominant_azimuth_deg, line_params, tie_tolerance_deg=cp.tie_tolerance_deg)
        df["tie_line_id"] = tie_line_id
        if cp.enabled:
            crossover_result = compute_crossover_leveling(
                df, tie_line_id, "anomaly", max_crossover_distance_m=cp.max_crossover_distance_m, iterative=cp.iterative
            )
            df["anomaly"] = apply_crossover_leveling(df, "anomaly", crossover_result)
            df["tmi"] = apply_crossover_leveling(df, "tmi", crossover_result)
        else:
            crossover_result = CrossoverLevelingResult(False, "사용자가 타이라인 보정을 비활성화했습니다 (기본값 - 타이라인 비행이 없을 수 있음).")
        self.crossover_info = {
            "applied": crossover_result.applied,
            "reason": crossover_result.reason,
            "n_tie_lines": crossover_result.n_tie_lines,
            "n_crossovers": crossover_result.n_crossovers,
            "n_survey_lines_corrected": crossover_result.n_survey_lines_corrected,
            "rms_before_nt": crossover_result.rms_before_nt,
            "rms_after_nt": crossover_result.rms_after_nt,
        }

        self.processed = df
        self.manual_overrides = {}
        self.grid_cache = {}
        self.transform_cache = {}
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
            "utm_epsg": self.utm_epsg,
            "dominant_azimuth_deg": self.dominant_azimuth_deg,
            "line_spacing_m": self.line_spacing_m,
            "inclination_deg": self.inclination_deg,
            "declination_deg": self.declination_deg,
            "diurnal": self.diurnal_info,
            "despike": self.despike_info,
            "gps_mag_lag": self.gps_lag_info,
            "heading_correction": _heading_correction_summary(self.heading_leveling),
            "crossover_leveling": self.crossover_info,
            "anomaly_stats": _stats(df.loc[active, "anomaly"]),
            "tmi_stats": _stats(df.loc[active, "tmi"]),
            "lines": _line_summaries(df, self.heading_leveling),
        }

    def run_chat(self, message: str, history: list[dict]) -> dict:
        from .chat import run_chat_turn

        return run_chat_turn(self, message, history)

    def generate_report(self) -> str:
        return generate_report_markdown(
            self.drone_summary() if self.drone_raw is not None else None,
            self.base_summary() if self.base_raw is not None else None,
            self.process_summary() if self.processed is not None else None,
            self.last_params.model_dump() if self.last_params is not None else None,
            self.inversion_summary_cache,
            self.euler_summary_cache,
            self.target_summary_cache,
        )

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
                "x": df["x"],
                "y": df["y"],
                "value": df[col],
                "line_id": df["line_id"],
                "tie_line_id": df["tie_line_id"],
                "excluded": ~active,
                "timestamp": df["timestamp"].astype(str),
            }
        )
        return out.to_dict(orient="records")

    def get_line_profile(self, line_id: int, value: str) -> dict:
        """Value-vs-along-track-distance series for a single flight line -
        a QC view distinct from the map/grid overlays, for spotting
        spikes, drift, or leveling offsets directly along one pass rather
        than inferring them from the 2D color pattern."""
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed
        line_df = df[df["line_id"] == line_id].sort_values("timestamp")
        if line_df.empty:
            raise ProjectError(f"측선 {line_id}을 찾을 수 없습니다.")

        x = line_df["x"].to_numpy()
        y = line_df["y"].to_numpy()
        distance_m = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        col = "anomaly" if value == "anomaly" else "tmi"
        active = self._active_mask()

        return {
            "line_id": line_id,
            "distance_m": distance_m.tolist(),
            "value": line_df[col].tolist(),
            "lat": line_df["lat"].tolist(),
            "lon": line_df["lon"].tolist(),
            "excluded": (~active.reindex(line_df.index)).tolist(),
        }

    def get_exclusion_state(self) -> dict:
        """Just the (point_id, excluded) pairs, not the full point record
        (lat/lon/value/line_id/timestamp never change from a manual-edit
        action) - vectorized, so it stays fast even at 100k+ points where
        get_points()'s per-row to_dict(orient="records") does not (see
        set_manual_exclude, which returns this instead of making the
        frontend re-fetch the full point list after every edit)."""
        df = self.processed
        active = self._active_mask()
        return {"point_id": df["point_id"].tolist(), "excluded": (~active).tolist()}

    def set_manual_exclude(self, req: ManualExcludeRequest) -> dict:
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed

        if req.mode == "reset":
            self.manual_overrides = {}
            self.grid_cache = {}
            self.transform_cache = {}
            return {**self.process_summary(), "exclusion": self.get_exclusion_state()}

        if req.mode == "lines":
            if not req.line_ids:
                raise ProjectError("line_ids가 필요합니다.")
            target_ids = set(df.loc[df["line_id"].isin(req.line_ids), "point_id"])
        elif req.mode == "point_ids":
            if not req.point_ids:
                raise ProjectError("point_ids가 필요합니다.")
            target_ids = set(req.point_ids)
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
        self.transform_cache = {}
        return {**self.process_summary(), "exclusion": self.get_exclusion_state()}

    def _resolve_max_distance(self, cell_size_m: float, max_distance_m: float | None) -> float:
        if max_distance_m is not None:
            return max_distance_m
        if self.line_spacing_m:
            return max(2.0 * cell_size_m, 0.6 * self.line_spacing_m)
        return 2.0 * cell_size_m

    def _grid_for(
        self,
        value: str,
        cell_size_m: float,
        method: str = "spline",
        max_distance_m: float | None = None,
        along_line_smooth: bool = True,
        along_line_smooth_wavelength_m: float | None = None,
    ) -> GridResult:
        resolved_max_distance = self._resolve_max_distance(cell_size_m, max_distance_m)
        # None (auto) = the estimated cross-line spacing itself: cross-line
        # interpolation cannot resolve anything finer than that anyway, so
        # any along-line detail below it is fabricated anisotropy
        # ("corrugation") rather than real resolvable structure - see
        # processing.gridding._along_line_lowpass.
        resolved_wavelength = along_line_smooth_wavelength_m if along_line_smooth_wavelength_m is not None else self.line_spacing_m
        effective_wavelength = resolved_wavelength if along_line_smooth else None
        key = (value, cell_size_m, method, resolved_max_distance, effective_wavelength)
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
            line_id=df.loc[active, "line_id"].to_numpy(),
            along_line_smooth_wavelength_m=effective_wavelength,
        )
        self.grid_cache[key] = result
        return result

    def get_grid_overlay(self, req: GridRequest) -> dict:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        grid = self._maybe_pre_level(grid, req)
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
            hillshade=req.hillshade,
            hillshade_azimuth_deg=req.hillshade_azimuth_deg,
            hillshade_altitude_deg=req.hillshade_altitude_deg,
            hillshade_exaggeration=req.hillshade_exaggeration,
            cell_size_m=grid.cell_size_m,
            stretch=req.stretch,
        )
        overlay["stats"] = _stats(pd.Series(grid.values.ravel()))
        overlay["cell_size_m"] = grid.cell_size_m
        if req.show_contours:
            overlay["contours"] = compute_contours(
                grid.values, grid.easting, grid.northing, self.utm_epsg,
                interval=req.contour_interval_nt, n_levels=req.contour_n_levels,
            )
        return overlay

    def _transform_values(self, grid: GridResult, req: TransformRequest) -> tuple[np.ndarray, bool]:
        resolved_max_distance = self._resolve_max_distance(req.cell_size_m, req.max_distance_m)
        resolved_smooth_wavelength = req.along_line_smooth_wavelength_m if req.along_line_smooth_wavelength_m is not None else self.line_spacing_m
        effective_smooth_wavelength = resolved_smooth_wavelength if req.along_line_smooth else None
        cache_key = (
            req.value,
            req.cell_size_m,
            req.method,
            resolved_max_distance,
            effective_smooth_wavelength,
            req.transform,
            req.continuation_height_m,
            req.trend_order,
            req.microlevel_strength,
            req.microlevel_angle_tolerance_deg,
            req.microlevel_wavelength_factor,
            req.microlevel_pre_apply,
        )
        if cache_key in self.transform_cache:
            return self.transform_cache[cache_key]
        result = self._compute_transform_values(grid, req)
        self.transform_cache[cache_key] = result
        return result

    def _maybe_pre_level(self, grid: GridResult, req) -> GridResult:
        """Apply microleveling to `grid` when req.microlevel_pre_apply is
        set - shared by every GridRequest/TransformRequest call site so
        "먼저 마이크로레벨링 적용" behaves identically whether the caller
        ends up looking at the plain grid or a derived transform of it,
        instead of microleveling only being reachable as its own
        mutually-exclusive transform choice. No-op (returns grid
        unchanged) when the flag isn't set."""
        if not getattr(req, "microlevel_pre_apply", False):
            return grid
        if not self.line_spacing_m:
            raise ProjectError("측선 간격을 추정할 수 없어 micro-leveling을 적용할 수 없습니다 (측선이 2개 이상 필요).")
        leveled = apply_microleveling(
            grid.values,
            grid.cell_size_m,
            self.dominant_azimuth_deg,
            self.line_spacing_m,
            strength=req.microlevel_strength,
            angle_tolerance_deg=req.microlevel_angle_tolerance_deg,
            wavelength_bandwidth_factor=req.microlevel_wavelength_factor,
        )
        return replace(grid, values=leveled)

    def _compute_transform_values(self, grid: GridResult, req: TransformRequest) -> tuple[np.ndarray, bool]:
        transform = req.transform
        if self.inclination_deg is None:
            raise ProjectError("IGRF 계산이 필요합니다 (자료 처리를 먼저 실행하세요).")
        if transform != "microlevel":
            # matters most for derivative-based transforms (RTP/1VD/tilt/
            # etc.), which amplify whatever line-parallel corrugation is
            # still present in the input grid.
            grid = self._maybe_pre_level(grid, req)
        if transform == "rtp":
            return reduction_to_pole(grid.values, grid.cell_size_m, self.inclination_deg, self.declination_deg), True
        if transform == "rte":
            return reduction_to_equator(grid.values, grid.cell_size_m, self.inclination_deg, self.declination_deg), True
        if transform == "1vd":
            return vertical_derivative(grid.values, grid.cell_size_m, order=1), True
        if transform == "2vd":
            return vertical_derivative(grid.values, grid.cell_size_m, order=2), True
        if transform == "as":
            return analytic_signal(grid.values, grid.cell_size_m), False
        if transform == "thdr":
            return total_horizontal_derivative(grid.values, grid.cell_size_m), False
        if transform == "tilt":
            return tilt_angle(grid.values, grid.cell_size_m), True
        if transform == "theta":
            return theta_map(grid.values, grid.cell_size_m), False
        if transform == "dx":
            return derivative_easting(grid.values, grid.cell_size_m), True
        if transform == "dy":
            return derivative_northing(grid.values, grid.cell_size_m), True
        if transform == "dxx":
            return second_derivative_ee(grid.values, grid.cell_size_m), True
        if transform == "dyy":
            return second_derivative_nn(grid.values, grid.cell_size_m), True
        if transform == "dxy":
            return second_derivative_en(grid.values, grid.cell_size_m), True
        if transform == "dxz":
            return second_derivative_ez(grid.values, grid.cell_size_m), True
        if transform == "dyz":
            return second_derivative_nz(grid.values, grid.cell_size_m), True
        if transform == "upward_continuation":
            height_m = req.continuation_height_m or (2.0 * grid.cell_size_m)
            return upward_continuation(grid.values, grid.cell_size_m, height_m), True
        if transform == "detrend":
            residual, _trend = remove_regional_trend(grid.values, grid.easting, grid.northing, order=req.trend_order)
            return residual, True
        if transform == "microlevel":
            if not self.line_spacing_m:
                raise ProjectError("측선 간격을 추정할 수 없어 micro-leveling을 적용할 수 없습니다 (측선이 2개 이상 필요).")
            return (
                apply_microleveling(
                    grid.values,
                    grid.cell_size_m,
                    self.dominant_azimuth_deg,
                    self.line_spacing_m,
                    strength=req.microlevel_strength,
                    angle_tolerance_deg=req.microlevel_angle_tolerance_deg,
                    wavelength_bandwidth_factor=req.microlevel_wavelength_factor,
                ),
                True,
            )
        raise ProjectError(f"알 수 없는 변환입니다: {transform}")

    def get_transform_overlay(self, req: TransformRequest) -> dict:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        values, symmetric = self._transform_values(grid, req)

        cmap = req.cmap or DEFAULT_CMAPS["derivative"]
        overlay = grid_to_png_overlay(
            values, grid.easting, grid.northing, self.utm_epsg, cmap_name=cmap, symmetric=symmetric, vmin=req.vmin, vmax=req.vmax,
            hillshade=req.hillshade,
            hillshade_azimuth_deg=req.hillshade_azimuth_deg,
            hillshade_altitude_deg=req.hillshade_altitude_deg,
            hillshade_exaggeration=req.hillshade_exaggeration,
            cell_size_m=grid.cell_size_m,
            stretch=req.stretch,
        )
        overlay["stats"] = _stats(pd.Series(values.ravel()))
        overlay["cell_size_m"] = grid.cell_size_m
        overlay["transform"] = req.transform
        if req.show_contours:
            overlay["contours"] = compute_contours(
                values, grid.easting, grid.northing, self.utm_epsg,
                interval=req.contour_interval_nt, n_levels=req.contour_n_levels,
            )
        return overlay

    def export_grid_geotiff(self, req: GridRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        grid = self._maybe_pre_level(grid, req)
        if req.colored:
            cmap = req.cmap or (DEFAULT_CMAPS["anomaly_grid"] if req.value == "anomaly" else DEFAULT_CMAPS["tmi_grid"])
            return grid_to_geotiff_bytes_colored(
                grid.values, grid.easting, grid.northing, self.utm_epsg,
                cmap_name=cmap, symmetric=(req.value == "anomaly"), vmin=req.vmin, vmax=req.vmax,
                hillshade=req.hillshade, hillshade_azimuth_deg=req.hillshade_azimuth_deg,
                hillshade_altitude_deg=req.hillshade_altitude_deg, hillshade_exaggeration=req.hillshade_exaggeration,
                stretch=req.stretch,
            )
        return grid_to_geotiff_bytes(grid.values, grid.easting, grid.northing, self.utm_epsg)

    def export_transform_geotiff(self, req: TransformRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        values, symmetric = self._transform_values(grid, req)
        if req.colored:
            cmap = req.cmap or DEFAULT_CMAPS["derivative"]
            return grid_to_geotiff_bytes_colored(
                values, grid.easting, grid.northing, self.utm_epsg,
                cmap_name=cmap, symmetric=symmetric, vmin=req.vmin, vmax=req.vmax,
                hillshade=req.hillshade, hillshade_azimuth_deg=req.hillshade_azimuth_deg,
                hillshade_altitude_deg=req.hillshade_altitude_deg, hillshade_exaggeration=req.hillshade_exaggeration,
                stretch=req.stretch,
            )
        return grid_to_geotiff_bytes(values, grid.easting, grid.northing, self.utm_epsg)

    def export_grid_xyz(self, req: GridRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        grid = self._maybe_pre_level(grid, req)
        return grid_to_xyz_bytes(grid.values, grid.easting, grid.northing, self.utm_epsg)

    def export_transform_xyz(self, req: TransformRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        values, _symmetric = self._transform_values(grid, req)
        return grid_to_xyz_bytes(values, grid.easting, grid.northing, self.utm_epsg)

    def export_grid_surfer_grd(self, req: GridRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        grid = self._maybe_pre_level(grid, req)
        return grid_to_surfer_grd_bytes(grid.values, grid.easting, grid.northing)

    def export_transform_surfer_grd(self, req: TransformRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        values, _symmetric = self._transform_values(grid, req)
        return grid_to_surfer_grd_bytes(values, grid.easting, grid.northing)

    def export_polygon_bln(self, polygon_latlon: list[list[float]]) -> bytes:
        """Project a [[lat, lon], ...] polygon (e.g. from the map's
        include/exclude draw tool) into this project's local UTM meters
        and write it out as a Surfer Blanking File."""
        if self.utm_epsg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요 (좌표계를 알 수 없습니다).")
        if not polygon_latlon or len(polygon_latlon) < 3:
            raise ProjectError("polygon은 최소 3개의 [lat, lon] 좌표가 필요합니다.")
        lat = np.array([pt[0] for pt in polygon_latlon], dtype=float)
        lon = np.array([pt[1] for pt in polygon_latlon], dtype=float)
        x, y, _epsg = project_to_local_xy(lat, lon, epsg_override=self.utm_epsg)
        return polygon_to_bln_bytes(x, y)

    def export_points_csv(self) -> bytes:
        """Every processed point (active and manually/auto-excluded alike,
        flagged via the `excluded` column) as plain CSV - lat/lon/x/y,
        both value fields, line assignment, and timestamp - for use in
        spreadsheets or other point-based tools that don't want the
        gridded/GeoTIFF form."""
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed
        active = self._active_mask()
        out = pd.DataFrame(
            {
                "point_id": df["point_id"],
                "timestamp": df["timestamp"].astype(str),
                "lat": df["lat"],
                "lon": df["lon"],
                "x_m": df["x"],
                "y_m": df["y"],
                "anomaly_nt": df["anomaly"],
                "tmi_nt": df["tmi"],
                "line_id": df["line_id"],
                "excluded": ~active,
            }
        )
        return out.to_csv(index=False).encode("utf-8")

    def run_euler_deconvolution(self, req: EulerDeconvolutionRequest) -> dict:
        # Defaults to along-line smoothing on (see _grid_for) - Euler
        # solutions are as sensitive to corrugation-driven spurious
        # gradients as any other derivative-based analysis, and this
        # request type has no UI toggle of its own for it.
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        solutions = _euler_deconvolution_solve(
            grid.values,
            grid.easting,
            grid.northing,
            grid.cell_size_m,
            self.utm_epsg,
            structural_index=req.structural_index,
            window_size_m=req.window_size_m,
            max_depth_uncertainty_pct=req.max_depth_uncertainty_pct,
        )
        depths = [s.depth_m for s in solutions]
        summary = {
            "n_solutions": len(solutions),
            "structural_index": req.structural_index,
            "depth_stats": _stats(pd.Series(depths)) if depths else None,
            "solutions": [
                {
                    "lat": s.lat,
                    "lon": s.lon,
                    "depth_m": s.depth_m,
                    "base_level_nt": s.base_level_nt,
                    "uncertainty_m": s.uncertainty_m,
                }
                for s in solutions
            ],
        }
        self.euler_summary_cache = summary
        return summary

    def run_target_detection(self, req: TargetDetectionRequest) -> dict:
        """Near-surface compact-target detection (mines, buried ordnance,
        hidden vehicles): grids the magnetic anomaly at a fine cell size,
        picks out localized (non-regional) local-maximum anomalies, and
        fits each one to a single induced-magnetic-dipole model - see
        processing/dipole_fit.py for the physics and the caveats around
        what a dipole-moment size class can and cannot tell you."""
        if self.processed is None or self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")

        active = self._active_mask()
        df = self.processed.loc[active]
        x_span = float(df["x"].max() - df["x"].min())
        y_span = float(df["y"].max() - df["y"].min())
        est_cells = (x_span / req.cell_size_m + 1) * (y_span / req.cell_size_m + 1)
        if est_cells > TARGET_DETECTION_GRID_CELL_CAP:
            raise ProjectError(
                f"탐지 격자가 너무 촘촘합니다 (예상 셀 수 약 {int(est_cells):,}개, 상한 {TARGET_DETECTION_GRID_CELL_CAP:,}개). "
                "탐지 격자 크기(m)를 늘리거나, 폴리곤으로 관심 영역만 남기고 나머지 측선을 제외한 뒤 다시 시도하세요."
            )

        # Along-line smoothing is deliberately off here: it low-passes at a
        # wavelength matched to the (much larger) line spacing, which would
        # blur out exactly the small, compact, near-surface anomalies
        # (mines/ordnance/vehicles) this feature exists to find - unlike
        # every other _grid_for caller, this one wants full native
        # along-line resolution preserved.
        grid = self._grid_for("anomaly", req.cell_size_m, req.method, req.max_distance_m, along_line_smooth=False)

        if req.amplitude_threshold_nt is not None:
            threshold_nt = req.amplitude_threshold_nt
        else:
            finite = grid.values[np.isfinite(grid.values)]
            robust_std = float(1.4826 * np.median(np.abs(finite - np.median(finite)))) if finite.size else 0.0
            threshold_nt = max(req.threshold_k * robust_std, 1e-6)

        targets = detect_targets(
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            df["anomaly"].to_numpy(),
            grid.values,
            grid.easting,
            grid.northing,
            grid.cell_size_m,
            self.inclination_deg,
            self.declination_deg,
            threshold_nt=threshold_nt,
            min_footprint_m=req.min_footprint_m,
            max_footprint_m=req.max_footprint_m,
            fit_window_m=req.fit_window_m,
            max_depth_m=req.max_depth_m,
            min_fit_quality=req.min_fit_quality,
        )

        transformer = Transformer.from_crs(f"EPSG:{self.utm_epsg}", "EPSG:4326", always_xy=True)
        target_dicts = []
        if targets:
            lons, lats = transformer.transform([t.x for t in targets], [t.y for t in targets])
        else:
            lons, lats = [], []
        for t, lat, lon in zip(targets, lats, lons):
            target_dicts.append(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "depth_m": t.depth_m,
                    "moment_am2": t.moment_am2,
                    "size_class": classify_moment(t.moment_am2),
                    "peak_anomaly_nt": t.peak_anomaly_nt,
                    "footprint_m": t.footprint_m,
                    "fit_quality": t.fit_quality,
                    "background_nt": t.background_nt,
                }
            )
        # sort strongest/most-confident first, since these lists are most
        # useful read top-down as a triage order rather than in scan order.
        target_dicts.sort(key=lambda d: -d["fit_quality"])

        summary = {
            "n_targets": len(target_dicts),
            "amplitude_threshold_nt": threshold_nt,
            "cell_size_m": grid.cell_size_m,
            "targets": target_dicts,
        }
        self.target_summary_cache = summary
        return summary

    def load_dem(self, data: bytes, name: str) -> dict:
        try:
            with rasterio.open(io.BytesIO(data)) as src:
                if src.crs is None:
                    raise ProjectError("DEM GeoTIFF에 좌표계(CRS) 정보가 없습니다.")
                bounds = src.bounds
                width, height = src.width, src.height
        except rasterio.errors.RasterioIOError as exc:
            raise ProjectError(f"DEM 파일을 열 수 없습니다: {exc}") from exc
        self.dem_bytes = data
        self.dem_name = name
        return {"name": name, "width": width, "height": height, "bounds": list(bounds)}

    def clear_dem(self) -> dict:
        self.dem_bytes = None
        self.dem_name = None
        return {"cleared": True}

    def add_reference_layer(self, name: str, data: bytes) -> dict:
        """Keep a project-scoped copy of an uploaded reference GeoTIFF (in
        addition to the stateless preview endpoint the map layer manager
        uses) so the chat assistant can sample real pixel values at a
        point instead of only showing the layer as a picture."""
        try:
            preview = load_geotiff_overlay(io.BytesIO(data), name=name)
        except OverlayImageError as exc:
            raise ProjectError(str(exc)) from exc
        self.reference_layers[name] = data
        return preview

    def remove_reference_layer(self, name: str) -> dict:
        self.reference_layers.pop(name, None)
        return {"removed": name}

    def sample_point(self, lat: float, lon: float) -> dict:
        """Look up everything this project currently knows about a single
        lat/lon: the nearest processed survey point's field values, the
        3D inversion susceptibility profile at that column (if an
        inversion has been run), and each uploaded reference layer's
        pixel value there - the grounding tool behind the chat assistant."""
        result: dict = {"lat": lat, "lon": lon}

        if self.processed is not None and self.utm_epsg is not None:
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
            x, y = transformer.transform(lon, lat)
            active = self._active_mask()
            df = self.processed.loc[active]
            if len(df):
                dist = np.hypot(df["x"].to_numpy() - x, df["y"].to_numpy() - y)
                idx = int(np.argmin(dist))
                row = df.iloc[idx]
                result["nearest_survey_point"] = {
                    "distance_m": float(dist[idx]),
                    "anomaly_nt": float(row["anomaly"]),
                    "tmi_nt": float(row["tmi"]),
                }

            if self.inversion_result is not None:
                mesh = self.inversion_result.mesh
                col = int(np.argmin(np.abs(mesh.x_centers - x)))
                row_i = int(np.argmin(np.abs(mesh.y_centers - y)))
                col_dist_m = float(np.hypot(mesh.x_centers[col] - x, mesh.y_centers[row_i] - y))
                if col_dist_m <= 2.0 * mesh.cell_size_m:
                    chi = self.inversion_result.susceptibility[row_i, col, :]
                    result["inversion_susceptibility_profile"] = {
                        "distance_from_mesh_column_m": col_dist_m,
                        "layers": [
                            {"elevation_m": float(mesh.z_centers[k]), "susceptibility_si": float(chi[k])}
                            for k in range(len(mesh.z_centers))
                        ],
                    }
                else:
                    result["inversion_susceptibility_profile"] = {"note": "역산 메쉬 범위 밖의 좌표입니다."}

        if self.reference_layers:
            geology = {}
            for layer_name, data in self.reference_layers.items():
                try:
                    geology[layer_name] = sample_geotiff_at_point(data, lat, lon)
                except GeologySampleError as exc:
                    geology[layer_name] = {"error": str(exc)}
            result["reference_layers"] = geology

        return result

    def save_project_bundle(self) -> bytes:
        """Zip up everything needed to resume this project later without
        re-uploading raw files: the raw (already-parsed) drone/base
        dataframes, the last-used processing params, manual overrides,
        DEM, and inversion result if present. Re-loading replays
        run_pipeline(last_params) on the raw data rather than trying to
        serialize every derived column/scalar, which keeps this in sync
        with the pipeline logic for free."""
        if self.drone_raw is None:
            raise ProjectError("저장할 자료가 없습니다 (드론 자료를 먼저 업로드하세요).")

        meta = {
            "last_params": self.last_params.model_dump() if self.last_params is not None else None,
            "manual_overrides": {str(k): v for k, v in self.manual_overrides.items()},
            "dem_name": self.dem_name,
            "has_base": self.base_raw is not None,
        }

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("drone_raw.csv", self.drone_raw.to_csv(index=False))
            if self.base_raw is not None:
                zf.writestr("base_raw.csv", self.base_raw.to_csv(index=False))
            zf.writestr("meta.json", json.dumps(meta))
            if self.dem_bytes is not None:
                zf.writestr("dem.tif", self.dem_bytes)
            if self.inversion_result is not None:
                zf.writestr("inversion.npz", self.export_inversion_npz())
        return buf.getvalue()

    def load_project_bundle(self, data: bytes) -> dict:
        """Restore a project saved by save_project_bundle. Returns the
        same shape as process_summary() (plus a couple of extra flags) so
        the frontend can jump straight back to where the user left off."""
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = set(zf.namelist())
                if "drone_raw.csv" not in names or "meta.json" not in names:
                    raise ProjectError("올바른 프로젝트 저장 파일이 아닙니다.")

                drone_df = pd.read_csv(io.BytesIO(zf.read("drone_raw.csv")))
                drone_df["timestamp"] = pd.to_datetime(drone_df["timestamp"])
                self.drone_raw = drone_df

                if "base_raw.csv" in names:
                    base_df = pd.read_csv(io.BytesIO(zf.read("base_raw.csv")))
                    base_df["timestamp"] = pd.to_datetime(base_df["timestamp"])
                    self.base_raw = base_df
                else:
                    self.base_raw = None

                meta = json.loads(zf.read("meta.json").decode("utf-8"))

                if "dem.tif" in names:
                    self.dem_bytes = zf.read("dem.tif")
                    self.dem_name = meta.get("dem_name")
                else:
                    self.dem_bytes = None
                    self.dem_name = None

                processed = False
                if meta.get("last_params") is not None and self.base_raw is not None:
                    params = ProcessParams(**meta["last_params"])
                    self.run_pipeline(params)
                    overrides = meta.get("manual_overrides") or {}
                    if overrides:
                        self.manual_overrides = {int(k): v for k, v in overrides.items()}
                        self.grid_cache = {}
                        self.transform_cache = {}
                    processed = True

                inversion_summary = None
                if "inversion.npz" in names and processed:
                    inversion_summary = self.import_inversion_npz(zf.read("inversion.npz"))
        except zipfile.BadZipFile as exc:
            raise ProjectError(f"프로젝트 파일을 열 수 없습니다 (손상되었거나 zip 형식이 아닙니다): {exc}") from exc
        except KeyError as exc:
            raise ProjectError(f"프로젝트 파일 내용이 올바르지 않습니다: {exc}") from exc

        result = {
            "drone_summary": self.drone_summary(),
            "base_summary": self.base_summary(),
            "processed": processed,
            "inversion_restored": inversion_summary is not None,
            "inversion_summary": inversion_summary,
            "params": self.last_params.model_dump() if self.last_params is not None else None,
        }
        if processed:
            result["process_summary"] = self.process_summary()
        return result

    def run_inversion(self, params: InversionParams) -> dict:
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed
        active = self._active_mask()
        sub = df.loc[active]
        if len(sub) < 20:
            raise ProjectError("역산을 위한 유효 포인트가 부족합니다.")
        col = "anomaly" if params.value == "anomaly" else "tmi"

        auto_used = params.obs_cell_size_m is None or params.depth_extent_m is None or params.n_layers is None
        auto_suggestion = None
        if auto_used:
            auto_suggestion = suggest_mesh_params(
                sub["x"].to_numpy(), sub["y"].to_numpy(), sub[col].to_numpy(),
                self.line_spacing_m, N_OBS_CAP, N_ACTIVE_CAP,
            )
        obs_cell_size_m = params.obs_cell_size_m or auto_suggestion["obs_cell_size_m"]
        depth_extent_m = params.depth_extent_m or auto_suggestion["depth_extent_m"]
        n_layers = params.n_layers or auto_suggestion["n_layers"]

        max_distance = self._resolve_max_distance(obs_cell_size_m, None)
        obs_grid = grid_points(
            sub["x"].to_numpy(), sub["y"].to_numpy(), sub[col].to_numpy(),
            obs_cell_size_m, method="nearest", max_distance_m=max_distance,
        )
        alt_grid = grid_points(
            sub["x"].to_numpy(), sub["y"].to_numpy(), sub["altitude_ellipsoidal_m"].to_numpy(),
            obs_cell_size_m, method="nearest", max_distance_m=max_distance,
        )

        ny, nx = obs_grid.values.shape
        if nx * ny > N_OBS_CAP:
            raise ProjectError(
                f"관측 격자가 너무 촘촘합니다 ({nx}x{ny}={nx*ny}점, 최대 {N_OBS_CAP}점) - 격자 크기(obs_cell_size_m)를 늘려주세요."
            )

        if self.dem_bytes is not None:
            try:
                ground_elev = load_dem_geotiff(io.BytesIO(self.dem_bytes), obs_grid.easting, obs_grid.northing, self.utm_epsg)
            except TerrainError as exc:
                raise ProjectError(str(exc)) from exc
        else:
            point_ground = estimate_ground_elevation(sub["altitude_ellipsoidal_m"].to_numpy(), params.assumed_agl_m)
            ground_result = grid_points(
                sub["x"].to_numpy(), sub["y"].to_numpy(), point_ground,
                obs_cell_size_m, method="nearest", max_distance_m=max_distance,
            )
            ground_elev = ground_result.values

        try:
            mesh = build_mesh(
                obs_grid.easting, obs_grid.northing, ground_elev,
                cell_size_m=obs_cell_size_m, depth_extent_m=depth_extent_m, n_layers=n_layers,
            )
        except InversionError as exc:
            raise ProjectError(str(exc)) from exc

        n_mesh_cells = int(mesh.active.sum())
        if n_mesh_cells > N_ACTIVE_CAP:
            raise ProjectError(
                f"역산 메쉬가 너무 큽니다 ({n_mesh_cells}셀, 최대 {N_ACTIVE_CAP}셀) - 격자 크기를 늘리거나 레이어 수/심도를 줄여주세요."
            )

        easting_2d, northing_2d = np.meshgrid(obs_grid.easting, obs_grid.northing)
        obs_mask = np.isfinite(obs_grid.values) & np.isfinite(alt_grid.values)
        obs_x = easting_2d[obs_mask]
        obs_y = northing_2d[obs_mask]
        obs_z = alt_grid.values[obs_mask]
        data_nt = obs_grid.values[obs_mask]
        if len(data_nt) < 10:
            raise ProjectError("역산을 위한 유효 관측 격자점이 부족합니다.")

        field_intensity_nt = mean_field_intensity_nt(
            sub["lat"].to_numpy(), sub["lon"].to_numpy(), sub["altitude_ellipsoidal_m"].to_numpy(), sub["timestamp"]
        )
        self.inversion_field_intensity_nt = field_intensity_nt

        try:
            G, rows, cols, layers = build_sensitivity_matrix(
                obs_x, obs_y, obs_z, mesh, self.inclination_deg, self.declination_deg, field_intensity_nt
            )
            result = invert(
                G, data_nt, mesh, rows, cols, layers,
                regularization_strength=params.regularization_strength,
                n_irls_iterations=params.n_irls_iterations,
            )
        except InversionError as exc:
            raise ProjectError(str(exc)) from exc

        self.inversion_result = result
        self.inversion_params = params
        self.inversion_obs_grid = obs_grid
        self.inversion_value_field = params.value

        active_chi = result.susceptibility[result.susceptibility > 0]
        summary = {
            "n_obs": result.n_obs,
            "n_active_cells": result.n_active_cells,
            "rms_misfit_nt": result.rms_misfit_nt,
            "iterations": result.iterations,
            "n_layers": n_layers,
            "obs_cell_size_m": obs_cell_size_m,
            "depth_extent_m": depth_extent_m,
            "cell_size_m": mesh.cell_size_m,
            "layer_thickness_m": mesh.layer_thickness_m,
            "elevation_range_m": [float(mesh.z_centers.min()), float(mesh.z_centers.max())],
            "field": {
                "inclination_deg": self.inclination_deg,
                "declination_deg": self.declination_deg,
                "field_intensity_nt": field_intensity_nt,
            },
            "susceptibility_stats": _stats(pd.Series(active_chi)) if active_chi.size else _stats(pd.Series(dtype=float)),
            "used_dem": self.dem_bytes is not None,
            "auto_params": auto_used,
            "source_depth_estimate_m": auto_suggestion["source_depth_estimate_m"] if auto_suggestion else None,
            "resolution_warning": _resolution_warning(obs_cell_size_m, self.line_spacing_m, nx, ny),
        }
        self.inversion_summary_cache = summary
        return summary

    def get_inversion_horizontal_slice(self, req: InversionSliceRequest) -> dict:
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        mesh = self.inversion_result.mesh
        try:
            slice_2d = horizontal_slice(self.inversion_result, req.layer_index, req.threshold, req.threshold_max)
            overlay = grid_to_png_overlay(
                slice_2d, mesh.x_centers, mesh.y_centers, self.utm_epsg,
                cmap_name=req.cmap or "geosoft_rainbow", symmetric=False, vmin=req.vmin, vmax=req.vmax,
                hillshade=req.hillshade,
                hillshade_azimuth_deg=req.hillshade_azimuth_deg,
                hillshade_altitude_deg=req.hillshade_altitude_deg,
                hillshade_exaggeration=req.hillshade_exaggeration,
                cell_size_m=mesh.cell_size_m,
            )
        except (InversionError, ValueError) as exc:
            raise ProjectError(str(exc)) from exc
        layer_index = int(np.clip(req.layer_index, 0, mesh.active.shape[2] - 1))
        overlay["stats"] = _stats(pd.Series(slice_2d.ravel()))
        overlay["layer_index"] = layer_index
        overlay["n_layers"] = int(mesh.active.shape[2])
        overlay["elevation_m"] = float(mesh.z_centers[layer_index])
        return overlay

    def export_inversion_slice_geotiff(self, req: InversionSliceRequest) -> bytes:
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        mesh = self.inversion_result.mesh
        try:
            slice_2d = horizontal_slice(self.inversion_result, req.layer_index, req.threshold, req.threshold_max)
        except InversionError as exc:
            raise ProjectError(str(exc)) from exc
        return grid_to_geotiff_bytes(slice_2d, mesh.x_centers, mesh.y_centers, self.utm_epsg)

    def get_inversion_vertical_section(self, req: InversionSectionRequest) -> dict:
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        mesh = self.inversion_result.mesh

        if req.profile == "custom":
            if not req.path or len(req.path) < 2:
                raise ProjectError("자유선 단면을 위해서는 경로(path)가 필요합니다.")
            if self.utm_epsg is None:
                raise ProjectError("좌표계 정보가 없습니다.")
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
            lats = [pt[0] for pt in req.path]
            lons = [pt[1] for pt in req.path]
            vx, vy = transformer.transform(lons, lats)
            path_x, path_y = _densify_path(np.asarray(vx), np.asarray(vy), req.sample_spacing_m)
        else:
            if req.position_frac is None:
                raise ProjectError("동서/남북 단면을 위해서는 position_frac이 필요합니다.")
            n_samples = max(2, int((mesh.x_centers.max() - mesh.x_centers.min() if req.profile == "ew" else mesh.y_centers.max() - mesh.y_centers.min()) / req.sample_spacing_m) + 1)
            if req.profile == "ew":
                # straight line at a fixed north-south position, spanning
                # the mesh's full east-west extent - local UTM x/y is
                # close enough to true east/west, north/south at this
                # survey scale.
                fixed_y = mesh.y_centers.min() + req.position_frac * (mesh.y_centers.max() - mesh.y_centers.min())
                path_x = np.linspace(mesh.x_centers.min(), mesh.x_centers.max(), n_samples)
                path_y = np.full(n_samples, fixed_y)
            else:  # "ns"
                fixed_x = mesh.x_centers.min() + req.position_frac * (mesh.x_centers.max() - mesh.x_centers.min())
                path_y = np.linspace(mesh.y_centers.min(), mesh.y_centers.max(), n_samples)
                path_x = np.full(n_samples, fixed_x)

        try:
            section, distance = _inversion_vertical_section(self.inversion_result, path_x, path_y, req.threshold, req.threshold_max)
            png = render_section_png(
                section, distance, self.inversion_result.mesh.z_centers,
                cmap_name=req.cmap or "geosoft_rainbow", vmin=req.vmin, vmax=req.vmax,
            )
        except InversionError as exc:
            raise ProjectError(str(exc)) from exc
        png["stats"] = _stats(pd.Series(section.ravel()))
        png["profile"] = req.profile
        return png

    def get_inversion_volume(self, threshold: float | None = None, threshold_max: float | None = None) -> dict:
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        result = self.inversion_result
        mesh = result.mesh
        x0 = float(mesh.x_centers.mean())
        y0 = float(mesh.y_centers.mean())

        # Interpolate onto a much finer grid before thresholding, purely
        # so the isosurface renders as a smooth "blob" instead of a
        # blocky voxel shape - the inversion itself is unaffected, this
        # only changes how the already-solved model is displayed. See
        # processing/inversion.upsample_susceptibility.
        fine_x, fine_y, fine_z, fine_chi = upsample_susceptibility(mesh, result.susceptibility)

        chi = fine_chi
        if threshold is not None:
            chi = np.where(chi >= threshold, chi, 0.0)
        if threshold_max is not None:
            chi = np.where(chi <= threshold_max, chi, 0.0)

        Y, X, Z = np.meshgrid(fine_y - y0, fine_x - x0, fine_z, indexing="ij")
        active_chi = result.susceptibility[result.susceptibility > 0]

        # Drape the actual observed anomaly/TMI map used as inversion
        # input on the terrain-following top of the mesh, matching the
        # common "2D magnetic map on a 3D inversion box" presentation -
        # gives geographic context for where the recovered bodies sit.
        top = None
        if self.inversion_obs_grid is not None:
            top = {
                "x": (mesh.x_centers - x0).tolist(),
                "y": (mesh.y_centers - y0).tolist(),
                "z": mesh.ground_elevation.tolist(),
                "color": np.where(
                    np.isfinite(self.inversion_obs_grid.values), self.inversion_obs_grid.values, None
                ).tolist(),
                "value_field": self.inversion_value_field,
            }

        return {
            "x": X.ravel().tolist(),
            "y": Y.ravel().tolist(),
            "z": Z.ravel().tolist(),
            "value": chi.ravel().tolist(),
            "shape": list(fine_chi.shape),
            "stats": _stats(pd.Series(active_chi)) if active_chi.size else _stats(pd.Series(dtype=float)),
            "top": top,
        }

    def export_inversion_npz(self) -> bytes:
        """Serialize the mesh + solved model (and enough context to
        re-render slices/sections/volume) into a self-contained .npz so a
        later session can reload the result via import_inversion_npz
        without rerunning the inversion solve."""
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        result = self.inversion_result
        mesh = result.mesh

        payload = dict(
            x_centers=mesh.x_centers,
            y_centers=mesh.y_centers,
            z_centers=mesh.z_centers,
            cell_size_m=np.array(mesh.cell_size_m),
            layer_thickness_m=np.array(mesh.layer_thickness_m),
            ground_elevation=mesh.ground_elevation,
            active=mesh.active,
            susceptibility=result.susceptibility,
            predicted_nt=result.predicted_nt,
            observed_nt=result.observed_nt,
            rms_misfit_nt=np.array(result.rms_misfit_nt),
            n_active_cells=np.array(result.n_active_cells),
            n_obs=np.array(result.n_obs),
            iterations=np.array(result.iterations),
            utm_epsg=np.array(self.utm_epsg if self.utm_epsg is not None else -1),
            inclination_deg=np.array(self.inclination_deg if self.inclination_deg is not None else np.nan),
            declination_deg=np.array(self.declination_deg if self.declination_deg is not None else np.nan),
            field_intensity_nt=np.array(self.inversion_field_intensity_nt if self.inversion_field_intensity_nt is not None else np.nan),
            value_field=np.array(self.inversion_value_field or "anomaly"),
        )
        if self.inversion_obs_grid is not None:
            g = self.inversion_obs_grid
            payload.update(
                obs_easting=g.easting,
                obs_northing=g.northing,
                obs_values=g.values,
                obs_cell_size_m=np.array(g.cell_size_m),
                obs_region=np.array(g.region, dtype=float),
            )

        buf = io.BytesIO()
        np.savez_compressed(buf, **payload)
        return buf.getvalue()

    def import_inversion_npz(self, data: bytes) -> dict:
        """Restore a previously exported inversion result (see
        export_inversion_npz) so slice/section/volume endpoints work
        immediately - no drone/base upload or processing needed first."""
        try:
            with np.load(io.BytesIO(data), allow_pickle=False) as npz:
                mesh = InversionMesh(
                    x_centers=npz["x_centers"],
                    y_centers=npz["y_centers"],
                    z_centers=npz["z_centers"],
                    cell_size_m=float(npz["cell_size_m"]),
                    layer_thickness_m=float(npz["layer_thickness_m"]),
                    ground_elevation=npz["ground_elevation"],
                    active=npz["active"],
                )
                result = InversionResult(
                    mesh=mesh,
                    susceptibility=npz["susceptibility"],
                    predicted_nt=npz["predicted_nt"],
                    observed_nt=npz["observed_nt"],
                    rms_misfit_nt=float(npz["rms_misfit_nt"]),
                    n_active_cells=int(npz["n_active_cells"]),
                    n_obs=int(npz["n_obs"]),
                    iterations=int(npz["iterations"]),
                )
                utm_epsg = int(npz["utm_epsg"])
                inclination_deg = float(npz["inclination_deg"])
                declination_deg = float(npz["declination_deg"])
                field_intensity_nt = float(npz["field_intensity_nt"])
                value_field = str(npz["value_field"])

                obs_grid = None
                if "obs_easting" in npz:
                    obs_grid = GridResult(
                        values=npz["obs_values"],
                        easting=npz["obs_easting"],
                        northing=npz["obs_northing"],
                        cell_size_m=float(npz["obs_cell_size_m"]),
                        region=tuple(npz["obs_region"].tolist()),
                    )
        except Exception as exc:  # noqa: BLE001 - deliberately broad: validating an untrusted uploaded file
            raise ProjectError(f"역산 결과 파일을 불러올 수 없습니다 (손상되었거나 올바른 형식이 아닙니다): {exc}") from exc

        self.inversion_result = result
        self.utm_epsg = utm_epsg if utm_epsg >= 0 else None
        self.inclination_deg = None if np.isnan(inclination_deg) else inclination_deg
        self.declination_deg = None if np.isnan(declination_deg) else declination_deg
        self.inversion_field_intensity_nt = None if np.isnan(field_intensity_nt) else field_intensity_nt
        self.inversion_value_field = value_field
        self.inversion_obs_grid = obs_grid

        active_chi = result.susceptibility[result.susceptibility > 0]
        return {
            "n_obs": result.n_obs,
            "n_active_cells": result.n_active_cells,
            "rms_misfit_nt": result.rms_misfit_nt,
            "iterations": result.iterations,
            "n_layers": int(mesh.active.shape[2]),
            "obs_cell_size_m": mesh.cell_size_m,
            "depth_extent_m": mesh.layer_thickness_m * mesh.active.shape[2],
            "cell_size_m": mesh.cell_size_m,
            "layer_thickness_m": mesh.layer_thickness_m,
            "elevation_range_m": [float(mesh.z_centers.min()), float(mesh.z_centers.max())],
            "field": {
                "inclination_deg": self.inclination_deg,
                "declination_deg": self.declination_deg,
                "field_intensity_nt": self.inversion_field_intensity_nt,
            },
            "susceptibility_stats": _stats(pd.Series(active_chi)) if active_chi.size else _stats(pd.Series(dtype=float)),
            "used_dem": None,
            "auto_params": False,
            "source_depth_estimate_m": None,
            "resolution_warning": None,
            "imported": True,
        }

    def export_inversion_csv(self) -> bytes:
        """Active-cell (x, y, z, susceptibility) point cloud - a portable
        format for other 3D tools (ParaView, Voxler, GIS point layers,
        even a spreadsheet) that don't understand our mesh format."""
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        result = self.inversion_result
        mesh = result.mesh
        rows, cols, layers = np.nonzero(mesh.active)
        out = pd.DataFrame(
            {
                "easting_m": mesh.x_centers[cols],
                "northing_m": mesh.y_centers[rows],
                "elevation_m": mesh.z_centers[layers],
                "susceptibility_si": result.susceptibility[rows, cols, layers],
            }
        )
        return out.to_csv(index=False).encode("utf-8")


def _resolution_warning(obs_cell_size_m: float, line_spacing_m: float | None, nx: int, ny: int) -> str | None:
    """A cell size coarser than the flight-line spacing means the mesh is
    under-using the resolution the survey actually captured (this can
    happen for large-area surveys, where the observation/mesh size caps
    force the cell size up regardless of line spacing) - flag it rather
    than silently returning a possibly-suboptimal mesh."""
    if not line_spacing_m or obs_cell_size_m <= line_spacing_m * 1.15:
        return None
    return (
        f"선택된 격자 크기({obs_cell_size_m:.0f}m)가 측선 간격({line_spacing_m:.0f}m)보다 큽니다 - "
        f"조사 영역이 넓어 계산량 상한({nx}x{ny}점 관측 격자) 때문에 해상도가 낮아진 것으로, "
        "측선 자료가 가진 만큼의 해상도를 다 살리지 못합니다. 더 세밀한 결과가 필요하면 "
        "관심 영역만 잘라서(예: 측선 일부만 남기고 나머지를 수동 제외) 다시 처리하거나, "
        "자동 설정을 끄고 격자 크기를 직접 줄여보세요(실행 시간이 늘어납니다)."
    )


def _densify_path(x: np.ndarray, y: np.ndarray, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    seg_len = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 0:
        return x, y
    n_samples = max(2, int(total / spacing_m) + 1)
    sample_dist = np.linspace(0.0, total, n_samples)
    sample_x = np.interp(sample_dist, cum, x)
    sample_y = np.interp(sample_dist, cum, y)
    return sample_x, sample_y


def _median_speed_mps(drone_raw: pd.DataFrame) -> float | None:
    """Median ground speed (m/s) estimated from consecutive raw GPS fixes,
    used by the frontend to suggest a filter cutoff frequency from a
    desired spatial wavelength (cutoff_hz = speed_mps / wavelength_m)
    before the user has run any processing yet."""
    if len(drone_raw) < 2:
        return None
    x, y, _epsg = project_to_local_xy(drone_raw["lat"].to_numpy(), drone_raw["lon"].to_numpy())
    dt = drone_raw["timestamp"].diff().dt.total_seconds().to_numpy()[1:]
    dist = np.hypot(np.diff(x), np.diff(y))
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(dt > 0, dist / dt, np.nan)
    speed = speed[np.isfinite(speed)]
    if speed.size == 0:
        return None
    return float(np.median(speed))


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
