"""In-memory per-project state and pipeline orchestration."""
from __future__ import annotations


import io
import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import rasterio
from matplotlib.path import Path as MplPath
from pyproj import Transformer

from .io_.base_loader import load_base_csvs
from .io_.drone_loader import load_drone_csvs
from .models import (
    HeightNormalizationApplyRequest,
    RepeatPassApplyRequest,
    RepeatPassRequest,
    SourceRemovalRequest,
    InversionParams,
    ManualExcludeRequest,
    ManualSmoothRequest,
    ProcessParams,
    QcCertificateRequest,
)
from magcore import progress
from magcore.base_qc import process_base_station
from magcore.base_segments import level_base_segments
from magcore.manual_smooth import apply_manual_smoothing
from magcore.source_removal import apply_source_removals, fit_source_removal
from magcore.height_normalization import (
    HeightNormalization,
    HeightNormalizationError,
    altitude_stats,
    correction_stats,
    fit_height_normalization,
)
from magcore.crossover_leveling import CrossoverLevelingResult, apply_crossover_leveling, compute_crossover_leveling
from magcore.despike import despike
from magcore.diurnal import apply_diurnal_correction
from magcore.filters import lowpass_filter, moving_average_filter, notch_filter, savgol_filter_1d
from magcore.gridding import GridResult
from magcore.igrf import compute_igrf_total_field, mean_inclination_declination
from magcore.inversion import InversionResult
from magcore.leveling import HeadingLevelingResult, apply_heading_correction, compute_heading_correction
from magcore.line_overlap import resolve_line_overlaps
from magcore.lines import (
    detect_lines,
    detect_tie_lines,
    estimate_line_spacing_m,
    group_turn_segments,
    project_to_local_xy,
    resolve_korea_projection_epsg,
)
from magcore.gps_lag import apply_gps_mag_lag
from magcore.heading_calibration import (
    HeadingCalibrationError,
    build_turn_based_calibration,
    calibration_angular_coverage_deg,
    cross_validate_heading_effect_map,
    fit_heading_effect_map,
    magnetic_heading,
)
from magcore.qc_certificate import evaluate_qc_certificate
from magcore.report import generate_report_markdown
from magcore.sway import detect_sway
from magcore.render import polygon_to_bln_bytes
from magcore.line_resolution import assess_line_resolution
from magcore.repeat_passes import find_repeat_passes, solve_offsets
from magcore.repeat_passes import summarize as summarize_repeat_passes
from magcore.statistical_leveling import (
    StatisticalLevelingResult,
    apply_statistical_leveling,
    compute_statistical_leveling,
    measure_striping,
)
from magcore.noise_qc import compute_difference_qc
from magcore.repeatability import analyze_repeatability
from magcore.sampling_qc import compute_sampling_distance_qc
from .project_common import (
    OverlayState,
    ProjectError,
    _GRID_CACHE_MAX_ENTRIES,
    _PIPELINE_PROGRESS,
    _TRANSFORM_CACHE_MAX_ENTRIES,
    _as_job,
    _detection_params,
    _heading_correction_should_apply,
    _heading_correction_summary,
    _line_summaries,
    _median_speed_mps,
    _overlap_overrides_for,
    _statistical_leveling_summary,
    _stats,
    boundary_rings,
)
from .project_observatory import ObservatoryMixin
from .project_grids import GridMixin
from .project_interpretation import InterpretationMixin
from .project_inversion import InversionMixin
from .project_persistence import PersistenceMixin


@dataclass
class Project(ObservatoryMixin, GridMixin, InterpretationMixin, InversionMixin, PersistenceMixin):
    id: str
    drone_raw: pd.DataFrame | None = None
    base_raw: pd.DataFrame | None = None
    base_processed: pd.DataFrame | None = None  # base_raw after trim+despike QC (base_qc.py) - see run_pipeline
    base_qc_info: dict | None = None
    base_leveling = None            # BaseLevelingResult when a base log was levelled
    base_leveling_info: dict | None = None
    overlap_info: dict | None = None
    base_source: dict | None = None  # set when base_raw came from an INTERMAGNET observatory rather than a local upload
    # A parsed-but-not-yet-applied INTERMAGNET observatory fetch/upload -
    # see preview_intermagnet_text/fetch_intermagnet_preview and
    # apply_intermagnet_preview. Kept server-side so "이 자료 사용" doesn't
    # need to re-send the whole (potentially large) IAGA-2002 text.
    intermagnet_preview: object = None
    # A parsed-but-not-yet-applied multi-observatory nearest-station
    # estimate (list of IagaObservatoryData + combined IDW DataFrame) - see
    # fetch_nearest_intermagnet_preview / apply_nearest_intermagnet_preview.
    intermagnet_nearest_preview: object = None
    # Same shape as intermagnet_nearest_preview, but kept around after
    # apply_nearest_intermagnet_preview clears the preview - backs the
    # comparison chart (get_nearest_intermagnet_comparison) so the user can
    # still see the individual station data an already-applied estimate
    # was built from, not just while it's pending confirmation.
    intermagnet_nearest_last_result: object = None
    processed: pd.DataFrame | None = None
    # processed, before any manual smoothing (set_manual_smoothing) is
    # applied - the pristine base that manual smoothing is re-derived from
    # on every call, so toggling/adding smoothed points never compounds.
    processed_base: pd.DataFrame | None = None
    # point_ids flagged as affected by a localized non-geological
    # disturbance (building, fence, vehicle, ...) and interpolated across
    # - see _apply_manual_smoothing / set_manual_smoothing.
    manual_smooth_point_ids: set = field(default_factory=set)
    # Ground-structure fields fitted and subtracted as models, in the order
    # they were added - see processing/source_removal.py. Applied to
    # processed_base before the point smoothing above, so either can be
    # undone without disturbing the other.
    source_removals: list = field(default_factory=list)
    # Optional user-drawn [[lat, lon], ...] polygon that clips grid display
    # (overlay/export) to exactly that outline, on top of the automatic
    # convex-hull extrapolation cap in gridding.py - see set_display_boundary.
    display_boundary_polygon: list[list[float]] | None = None
    # Diagnostics from the last automatic boundary generation (buffer used,
    # area, dropped parts/holes, warnings) or {"failed": True, "reason":...}
    # - surfaced in process_summary so the user can see what the boundary
    # they didn't draw themselves actually did.
    auto_boundary_info: dict | None = None
    # point_id -> True (force include, even if auto-excluded) | False (force
    # exclude, even if auto-included). Absent point_ids fall back to the
    # automatic line_id>=0 result.
    manual_overrides: dict = field(default_factory=dict)
    utm_epsg: int | None = None
    dominant_azimuth_deg: float | None = None
    line_spacing_m: float | None = None
    heading_leveling: HeadingLevelingResult | None = None
    # Whether heading_leveling's offset was actually subtracted from the
    # data, as opposed to only measured - see run_pipeline.
    heading_correction_applied: bool = False
    statistical_leveling: StatisticalLevelingResult | None = None
    striping_info: dict | None = None
    line_resolution_info: dict | None = None
    inclination_deg: float | None = None
    declination_deg: float | None = None
    diurnal_info: dict | None = None
    despike_info: dict | None = None
    sway_info: dict | None = None
    calibration_raw: pd.DataFrame | None = None
    heading_calibration_info: dict | None = None
    crossover_info: dict | None = None
    gps_lag_info: dict | None = None
    noise_qc_info: dict | None = None
    sampling_qc_info: dict | None = None
    file_level_info: dict | None = None
    repeatability_raw: pd.DataFrame | None = None
    repeatability_summary_cache: dict | None = None
    multiscale_edges_summary_cache: dict | None = None
    lineament_summary_cache: dict | None = None
    tilt_depth_cache: dict | None = None
    as_depth_cache: dict | None = None
    spectral_depth_cache: dict | None = None
    contact_summary_cache: dict | None = None
    prospectivity_summary_cache: dict | None = None
    prospectivity_score_grid: np.ndarray | None = None
    prospectivity_score_easting: np.ndarray | None = None
    prospectivity_score_northing: np.ndarray | None = None
    prospectivity_score_cell_size_m: float | None = None
    inversion_summary_cache: dict | None = None
    euler_summary_cache: dict | None = None
    target_summary_cache: dict | None = None
    qc_certificate_cache: dict | None = None
    repeat_pass_summary_cache: dict | None = None
    geology_stats_cache: dict | None = None
    reference_layers: dict = field(default_factory=dict)  # name -> raw GeoTIFF bytes
    reference_vectors: dict = field(default_factory=dict)  # name -> parsed vector layer (magcore.vector_layer)
    # What the layer manager shows, in its order: [{type, name, opacity,
    # visible, path?, scheme?}]. The map's own state, mirrored here by the
    # frontend so a saved project comes back with its layers.
    reference_layer_state: list = field(default_factory=list)
    last_params: ProcessParams | None = None
    grid_cache: dict = field(default_factory=dict)
    transform_cache: dict = field(default_factory=dict)
    dem_bytes: bytes | None = None
    dem_name: str | None = None
    # User-digitized geology blocks for the inversion reference model -
    # each: {"id": int, "name": str, "susceptibility_si": float,
    # "path": [[lat, lon], ...]}. See run_inversion's geology reference
    # handling and add/update/remove_geology_unit below.
    geology_units: list = field(default_factory=list)
    geology_unit_next_id: int = 1
    inversion_result: InversionResult | None = None
    inversion_params: InversionParams | None = None
    inversion_field_intensity_nt: float | None = None
    inversion_obs_grid: GridResult | None = None
    inversion_value_field: str | None = None
    # One level offset per flight, solved on the ground the survey flew
    # twice - see processing/repeat_passes.py and _flight_levelled_base.
    repeat_pass_offsets: dict = field(default_factory=dict)
    # The equivalent layers that restate every reading at one altitude -
    # see processing/height_normalization.py and _height_normalized_base.
    # The per-point correction they imply for processed_base is cached
    # alongside (point ids, corrections), because it costs seconds on a
    # large survey and the chain is rebuilt on every edit.
    height_normalization: HeightNormalization | None = None
    _height_correction_cache: tuple | None = field(default=None, repr=False, compare=False)
    # Operator decisions about anomaly candidates: which were confirmed as
    # ground structures, which turned out to be geology, which are on
    # hold. Kept by position, not by rank - see set_candidate_note.
    candidate_notes: list = field(default_factory=list)
    # Wall-clock time of the last change worth saving, set by mark_changed
    # below and read by autosave.py. 0 means "nothing to save yet".
    changed_at: float = 0.0
    # Taken by the bundle writer while it reads the project's frames and
    # edits together, and by the methods that replace them, so an autosave
    # on its own thread never captures half of an edit.
    _edit_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    last_overlay: OverlayState | None = None

    def load_drone(self, buffers: list) -> dict:
        with self._edit_lock:
            return self._load_drone_unlocked(buffers)

    @_as_job("드론 자료 읽기", rollback=True)
    def _load_drone_unlocked(self, buffers: list) -> dict:
        self.drone_raw = load_drone_csvs(buffers)
        self.mark_changed()
        return self.drone_summary()

    def load_base(self, buffers: list, filenames: list | None = None) -> dict:
        with self._edit_lock:
            return self._load_base_unlocked(buffers, filenames)

    def _load_base_unlocked(self, buffers: list, filenames: list | None = None) -> dict:
        self.base_raw = load_base_csvs(buffers, filenames)
        self.base_source = None
        self.mark_changed()
        return self.base_summary()

    def _survey_centroid(self) -> tuple[float, float] | None:
        if self.drone_raw is None or len(self.drone_raw) == 0:
            return None
        return float(self.drone_raw["lat"].mean()), float(self.drone_raw["lon"].mean())

    def load_heading_calibration(self, buffers: list) -> dict:
        """A short calibration flight (same instrument/file format as the
        main survey) flown sweeping through many orientations over a
        magnetically quiet patch - see processing/heading_calibration.py.
        Parsed with the same multi-format loader as the main survey since
        it's the identical instrument output."""
        self.calibration_raw = load_drone_csvs(buffers)
        return self.heading_calibration_summary()

    def load_repeatability(self, buffers: list) -> dict:
        """A dedicated repeatability-test flight (same file format as the
        main survey) - see processing/repeatability.py. Not mixed into the
        main survey data; analysed on its own via run_repeatability_analysis."""
        self.repeatability_raw = load_drone_csvs(buffers)
        self.mark_changed()
        return {
            "n_points": len(self.repeatability_raw),
            "time_range": [
                self.repeatability_raw["timestamp"].min().isoformat(),
                self.repeatability_raw["timestamp"].max().isoformat(),
            ],
        }

    def run_repeatability_analysis(self) -> dict:
        if self.repeatability_raw is None:
            raise ProjectError("반복측선(Repeatability) 자료를 먼저 업로드하세요.")
        if self.base_raw is None:
            raise ProjectError("베이스(일변화) 자료를 먼저 업로드하세요.")
        diurnal_params = self.last_params.diurnal_params if self.last_params else None
        line_params = (
            _detection_params(self.last_params.line_params) if self.last_params else None
        )
        result = analyze_repeatability(
            self.repeatability_raw,
            self.base_raw,
            time_offset_seconds=diurnal_params.time_offset_seconds if diurnal_params else 0.0,
            diurnal_reference=diurnal_params.reference if diurnal_params else "mean",
            line_params=line_params,
            utm_epsg_override=self.utm_epsg,
        )
        self.repeatability_summary_cache = result
        return result

    def heading_calibration_summary(self) -> dict:
        if self.calibration_raw is None:
            return {}
        d = self.calibration_raw
        has_compass = d["compass_x"].notna().sum() >= 10
        return {
            "n_points": len(d),
            "time_range": [d["timestamp"].min().isoformat(), d["timestamp"].max().isoformat()],
            "has_compass_data": bool(has_compass),
        }

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
            # Whether the uploaded files' own "MagComp" column (a prior
            # heading/system-error compensation, e.g. from the companion
            # MagArrow-heading-error-calibration tool or the vendor's own
            # LabVIEW processor with "Keep Raw Data" on) was actually used
            # in place of the raw field - see io_/drone_loader.py
            # _parse_generic. Surfaced explicitly because it is easy to
            # upload a "-comp.csv" and assume its correction was applied
            # when, before this was tracked, there was no way to tell.
            "n_files_using_precompensated_mag": d.attrs.get("n_files_using_precompensated_mag", 0),
            "n_files_total": d.attrs.get("n_files_total", 0),
            "median_speed_mps": _median_speed_mps(d),
            "flight_dates": [dt.isoformat() for dt in self._survey_dates() or []],
        }

    def _survey_dates(self) -> list | None:
        """The distinct calendar dates the drone actually flew, straight
        from its own (GPS-time) timestamps - used to auto-target exactly
        the INTERMAGNET dates a survey needs (see
        fetch_nearest_intermagnet_preview/fetch_intermagnet_preview)
        instead of every day across the survey's full min-max span, which
        can be mostly empty padding for a survey flown on a few separate
        days weeks apart."""
        if self.drone_raw is None or len(self.drone_raw) == 0:
            return None
        return sorted(set(self.drone_raw["timestamp"].dt.date))

    def base_summary(self) -> dict:
        if self.base_raw is None:
            return {}
        b = self.base_raw
        return {
            "n_points": len(b),
            "time_range": [b["timestamp"].min().isoformat(), b["timestamp"].max().isoformat()],
            "mag_range": [float(b["mag"].min()), float(b["mag"].max())],
            "n_duplicate_timestamps_removed": b.attrs.get("n_duplicate_timestamps_removed", 0),
            "date_fallback_used": b.attrs.get("date_fallback_used", False),
            "source": self.base_source,
        }

    def get_base_timeseries(self) -> dict:
        """Raw vs QC-corrected (transient-trimmed + despiked) base-station
        log, for the user to visually confirm the installation/pickup trim
        and interior despiking did the right thing (see base_qc.py). Falls
        back to the raw series alone (base_processed=None) when the pipeline
        hasn't been run yet, so this is usable right after upload."""
        if self.base_raw is None:
            raise ProjectError("베이스(일변화) 자료를 먼저 업로드하세요.")
        raw = self.base_raw.sort_values("timestamp")
        out = {
            "raw_timestamp": raw["timestamp"].astype(str).tolist(),
            "raw_mag": raw["mag"].tolist(),
            "corrected_timestamp": None,
            "corrected_mag": None,
            "qc_info": self.base_qc_info,
        }
        if self.base_processed is not None:
            corrected = self.base_processed.sort_values("timestamp")
            out["corrected_timestamp"] = corrected["timestamp"].astype(str).tolist()
            out["corrected_mag"] = corrected["mag"].tolist()
        return out

    def run_pipeline(self, params: ProcessParams) -> dict:
        with self._edit_lock:
            return self._run_pipeline_unlocked(params)

    @_as_job("자료 처리", rollback=True)
    def _run_pipeline_unlocked(self, params: ProcessParams) -> dict:
        if self.drone_raw is None:
            raise ProjectError("드론 자료를 먼저 업로드하세요.")
        if self.base_raw is None and params.diurnal_params.mode != "assume_constant":
            raise ProjectError(
                "베이스(일변화) 자료를 먼저 업로드하세요. 베이스 자료가 없다면 "
                "일변화 보정 방식을 '베이스 자료 없음(지구자기장 일정 가정)'으로 설정하세요."
            )

        progress.report("전처리 (스파이크 제거·필터)", _PIPELINE_PROGRESS["P0"])
        df = self.drone_raw.copy()

        n_before_lag = len(df)
        if params.gps_mag_lag_seconds != 0.0:
            df = apply_gps_mag_lag(df, params.gps_mag_lag_seconds)
        self.gps_lag_info = {
            "lag_seconds": params.gps_mag_lag_seconds,
            "n_points_dropped": n_before_lag - len(df),
        }

        # Pristine copy of the sensor reading, taken before despike/notch
        # touch mag_raw in place below - lets get_line_profile show what
        # the filtering pipeline actually changed (see its docstring).
        df["mag_original"] = df["mag_raw"].to_numpy(copy=True)

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

        nf = params.notch_filter
        for freq_hz in nf.frequencies_hz:
            df["mag_raw"] = notch_filter(df["mag_raw"].to_numpy(), df["timestamp"], freq_hz, nf.quality_factor)

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

        overlap_resolution = params.line_params.overlap_resolution
        overlap_overrides = params.line_params.overlap_overrides or {}
        line_params = _detection_params(params.line_params)
        progress.report("측선 판별", _PIPELINE_PROGRESS["P1"])
        df = detect_lines(df, line_params, utm_epsg_override=utm_epsg_override)
        self.utm_epsg = df.attrs["utm_epsg"]
        self.dominant_azimuth_deg = df.attrs["dominant_azimuth_deg"]

        # A line the flight was interrupted on carries a stretch recorded
        # twice, because the drone backs up before resuming. Left in, the
        # two recordings are interleaved cell by cell at gridding time -
        # see processing/line_overlap.py.
        if overlap_resolution != "off":
            forced = {}
            if overlap_resolution == "keep_first":
                forced = {"__all__": "first"}
            elif overlap_resolution == "keep_second":
                forced = {"__all__": "second"}
            resolved = resolve_line_overlaps(
                df, "mag_filtered", self.dominant_azimuth_deg,
                df["line_id"].to_numpy(), df["exclusion_reason"].to_numpy(),
                overrides=_overlap_overrides_for(df, overlap_overrides, forced),
            )
            df["line_id"] = resolved.line_id
            df["exclusion_reason"] = resolved.exclusion_reason
            self.overlap_info = {"mode": overlap_resolution, "decisions": resolved.summary()}
        else:
            self.overlap_info = {"mode": "off", "decisions": []}

        if params.noise_qc.enabled:
            self.noise_qc_info = compute_difference_qc(df, "mag_filtered")
        else:
            self.noise_qc_info = {"available": False}
        progress.report("잡음·표본 간격 진단", _PIPELINE_PROGRESS["P2"])
        self.sampling_qc_info = compute_sampling_distance_qc(df)

        sp = params.sway_detection
        if sp.enabled:
            sway_mask, self.sway_info = detect_sway(
                df["gyro_mag"].to_numpy(), df["accel_horiz_g"].to_numpy(), threshold_k=sp.threshold_k
            )
            if self.sway_info["available"]:
                # Only downgrade points that detect_lines already accepted
                # onto a line - a swinging sample during a turn is already
                # excluded, and re-tagging it wouldn't change anything but
                # would make n_points_excluded double-count.
                flagged_active = sway_mask & (df["line_id"].to_numpy() >= 0)
                df.loc[flagged_active, "exclusion_reason"] = "high_sway"
                df.loc[flagged_active, "line_id"] = -1
                self.sway_info["n_points_excluded"] = int(flagged_active.sum())
            else:
                self.sway_info["n_points_excluded"] = 0
        else:
            self.sway_info = {"enabled": False, "available": False, "n_points_excluded": 0}

        progress.report("일변화·기준장 보정", _PIPELINE_PROGRESS["P3"])
        self.line_spacing_m = estimate_line_spacing_m(df, self.dominant_azimuth_deg)

        if params.diurnal_params.mode == "assume_constant":
            # The user explicitly chose to proceed without diurnal
            # correction (checked "베이스 자료 없이 진행"): assume the
            # Earth's field was steady over the (short) survey window, so
            # there is no diurnal (solar-driven) variation to remove - the
            # drone's own filtered reading is used as-is. This governs on
            # its own regardless of whether a base file happens to be
            # uploaded (e.g. the user uploaded one, found it unusable -
            # no time overlap with the flight - and ticked this instead of
            # removing the file) - mathematically identical to a
            # base-station correction against a perfectly constant base
            # value (the correction term collapses to zero everywhere).
            self.base_processed = None
            self.base_qc_info = None
            self.base_leveling = None
            self.base_leveling_info = None
            df["mag_diurnal_corrected"] = df["mag_filtered"]
            self.diurnal_info = {
                "mode": "assume_constant",
                "coverage_pct": None,
                "has_overlap": False,
                "base_reference_value": None,
                "base_time_range": None,
                "drone_time_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
                "note": "베이스 자료 없이 지구자기장이 일정하다고 가정했습니다 - 일변화(태양풍에 의한 시간에 따른 자기장 변화) 보정이 적용되지 않았습니다.",
            }
        else:
            bqc = params.base_qc_params
            base_qc_result = process_base_station(
                self.base_raw,
                trim_enabled=bqc.trim_enabled,
                trim_window_seconds=bqc.trim_window_seconds,
                trim_threshold_k=bqc.trim_threshold_k,
                trim_confirm_seconds=bqc.trim_confirm_seconds,
                trim_max_fraction=bqc.trim_max_fraction,
                despike_enabled=bqc.despike_enabled,
                despike_window_size=bqc.despike_window_size,
                despike_threshold_k=bqc.despike_threshold_k,
            )
            base_processed = base_qc_result.corrected
            # Level the base's own deployments onto one common level before
            # it is used as a diurnal reference. A base that was moved reads
            # a different absolute field at each spot, and that step does not
            # cancel against the single survey-wide reference the correction
            # takes - it lands in the anomaly as a constant, whole-flight
            # offset, before any levelling step runs. See
            # processing/base_segments.py.
            # Always measured and reported; what actually gets levelled is
            # bqc.segment_mode's call. See processing/base_segments.py for
            # why a gap is not treated the same as a step.
            self.base_leveling = level_base_segments(
                base_processed,
                mode=bqc.segment_mode,
                gap_minutes=bqc.segment_gap_minutes,
                step_threshold_nt=bqc.segment_step_threshold_nt,
                gap_suspicious_nt=bqc.segment_gap_suspicious_nt,
            )
            base_processed = self.base_leveling.base
            self.base_leveling_info = self.base_leveling.summary()
            self.base_processed = base_processed
            self.base_qc_info = {
                "n_points_raw": len(self.base_raw),
                "n_points_corrected": len(base_qc_result.corrected),
                "n_trimmed_start": base_qc_result.n_trimmed_start,
                "n_trimmed_end": base_qc_result.n_trimmed_end,
                "n_spikes_removed": base_qc_result.n_spikes_removed,
            }

            diurnal_result = apply_diurnal_correction(
                df["timestamp"],
                df["mag_filtered"].to_numpy(),
                self.base_processed,
                time_offset_seconds=params.diurnal_params.time_offset_seconds,
                reference=params.diurnal_params.reference,
            )
            df["mag_diurnal_corrected"] = diurnal_result.corrected
            n_extrapolated = int(diurnal_result.extrapolated_mask.sum())
            self.diurnal_info = {
                "mode": "base_station",
                "coverage_pct": diurnal_result.coverage_pct,
                "has_overlap": diurnal_result.has_overlap,
                "base_reference_value": diurnal_result.base_reference_value,
                "base_time_range": [str(diurnal_result.base_time_range[0]), str(diurnal_result.base_time_range[1])],
                "drone_time_range": [str(diurnal_result.drone_time_range[0]), str(diurnal_result.drone_time_range[1])],
                # Points whose timestamp falls outside the base station's
                # own coverage window get a boundary-clamped (not truly
                # extrapolated) correction - see DiurnalResult.extrapolated_mask.
                "n_extrapolated": n_extrapolated,
                "pct_extrapolated": float(100.0 * n_extrapolated / len(diurnal_result.extrapolated_mask))
                if len(diurnal_result.extrapolated_mask) else 0.0,
            }

        hec = params.heading_effect_calibration
        if hec.enabled:
            self.heading_calibration_info = self._apply_heading_effect_calibration(df, params)
        else:
            self.heading_calibration_info = {
                "enabled": False,
                "available": self.calibration_raw is not None,
                "applied": False,
            }

        igrf_total = compute_igrf_total_field(
            df["lat"].to_numpy(), df["lon"].to_numpy(), df["altitude_ellipsoidal_m"].to_numpy(), df["timestamp"]
        )
        if not np.isfinite(igrf_total).any():
            # compute_igrf_total_field leaves a row NaN whenever its
            # lat/lon/altitude is non-finite (see igrf.py) - if EVERY row
            # is NaN, the whole survey's altitude is unusable (e.g. the
            # SENSYS MagDrone R3 ASC format, which always reports
            # altitude_msl_m as NaN - see io_/drone_loader.py). Before this
            # check, that used to surface only much further downstream as
            # an opaque "그리딩을 위한 유효 포인트가 부족합니다" or a
            # silently blank map, with nothing pointing back at altitude as
            # the actual cause - raise here instead, where the cause is
            # still known.
            raise ProjectError(
                "고도 자료가 없어 IGRF(지구자기장) 보정을 계산할 수 없습니다 - 업로드한 파일 형식이 "
                "고도 정보를 제공하지 않는 것으로 보입니다. GPS 고도가 포함된 형식의 자료를 사용하거나, "
                "고도 열이 올바르게 인식되었는지 확인하세요."
            )
        df["igrf_total"] = igrf_total
        df["anomaly"] = df["mag_diurnal_corrected"] - igrf_total
        df["tmi"] = df["mag_diurnal_corrected"]

        self.inclination_deg, self.declination_deg = mean_inclination_declination(
            df["lat"].to_numpy(), df["lon"].to_numpy(), df["altitude_ellipsoidal_m"].to_numpy(), df["timestamp"]
        )

        hp = params.heading_correction
        # Always measured, applied only when the user asked for it.
        # "Do the forward and reverse lines read differently, and by how
        # much" is a question about the survey, not about this correction,
        # and it is the question people actually ask when they see
        # striping - they should not have to turn a correction on and
        # compare two maps by eye to get the number. Cheap enough to do
        # unconditionally: ~0.4s on a 120k-point, 40-line survey.
        progress.report("헤딩 차이 측정", _PIPELINE_PROGRESS["P4"])
        self.heading_leveling = compute_heading_correction(
            df,
            "anomaly",
            self.dominant_azimuth_deg,
            self.line_spacing_m,
            quiet_percentile=hp.quiet_percentile,
            max_match_distance_m=hp.max_match_distance_m,
            method=hp.method,
            neighborhood_radius_factor=hp.neighborhood_radius_factor,
        )
        self.heading_correction_applied = _heading_correction_should_apply(hp, self.heading_leveling)
        if self.heading_correction_applied:
            df["anomaly"] = apply_heading_correction(df, "anomaly", self.heading_leveling)
            df["tmi"] = apply_heading_correction(df, "tmi", self.heading_leveling)

        # Tie-line membership is computed unconditionally (cheap) so it can
        # always be shown to the user (e.g. in the flight-path editor),
        # independent of whether the crossover-leveling correction itself
        # is enabled.
        cp = params.crossover_leveling
        tie_line_id = detect_tie_lines(df, self.dominant_azimuth_deg, line_params, tie_tolerance_deg=cp.tie_tolerance_deg)
        df["tie_line_id"] = tie_line_id
        if cp.enabled:
            crossover_result = compute_crossover_leveling(
                df, tie_line_id, "anomaly", max_crossover_distance_m=cp.max_crossover_distance_m,
                iterative=cp.iterative, leveling_order=cp.leveling_order,
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

        # Measured before the per-line correction runs, so it describes
        # the survey as flown - the number to compare between flights,
        # settings, or aircraft. Always computed, including when the
        # correction below declines.
        progress.report("줄무늬·측선 분해능 진단", _PIPELINE_PROGRESS["P5"])
        self.striping_info = measure_striping(df, "anomaly", self.dominant_azimuth_deg, self.line_spacing_m)

        # Whether the line spacing resolves this field at all. Measured on
        # the points rather than the grid, because once gridded the
        # question is unanswerable: the gridder has already filled the gaps
        # between the lines with its own guess. See
        # processing/line_resolution.py for why this is the number that
        # explains line-parallel striping on the Haenam block, after six
        # processing-side explanations were measured and ruled out.
        #
        # Until 2026-11 this block never ran in the app: `on_lines` was an
        # undefined name, the except below turned the NameError into
        # "진단 실패", and behind it the azimuth was passed in the wrong
        # convention - this method keeps atan2(dy, dx) (from east), the
        # diagnostic wants the flight direction from north. On the Andong
        # block (N-S lines, 89 deg here) the wrong one makes every line
        # look 13 m long. tests/test_regression_andong.py pins the result.
        on_lines = df["line_id"] >= 0
        azimuth_from_north = None if self.dominant_azimuth_deg is None else (90.0 - self.dominant_azimuth_deg) % 180.0
        try:
            self.line_resolution_info = assess_line_resolution(
                df.loc[on_lines, "x"].to_numpy(),
                df.loc[on_lines, "y"].to_numpy(),
                df.loc[on_lines, "anomaly"].to_numpy(),
                df.loc[on_lines, "line_id"].to_numpy(),
                azimuth_from_north,
            )
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail a run
            self.line_resolution_info = {"available": False, "reason": f"측선 분해능 진단 실패: {exc}"}

        # Last of the three leveling steps, so it only has to explain what
        # the direction-based and tie-line corrections above could not.
        slp = params.statistical_leveling
        progress.report("통계적 레벨링", _PIPELINE_PROGRESS["P6"])
        if slp.enabled:
            self.statistical_leveling = compute_statistical_leveling(
                df, "anomaly", self.dominant_azimuth_deg, self.line_spacing_m,
                trend_window_lines=slp.trend_window_lines,
                order=slp.order,
                n_segments=slp.n_segments,
                max_shift_nt=slp.max_shift_nt,
            )
            df["anomaly"] = apply_statistical_leveling(df, "anomaly", self.statistical_leveling)
            df["tmi"] = apply_statistical_leveling(df, "tmi", self.statistical_leveling)
        else:
            self.statistical_leveling = StatisticalLevelingResult(
                False, "사용자가 통계적 레벨링을 비활성화했습니다."
            )

        self.file_level_info = self._check_file_level_offsets(df)

        self.processed = df
        self.processed_base = df.copy()
        self.manual_overrides = {}
        progress.report("경계·품질 요약 정리", _PIPELINE_PROGRESS["P7"])
        self.manual_smooth_point_ids = set()
        self.source_removals = []
        self.repeat_pass_offsets = {}
        self.height_normalization = None
        self._height_correction_cache = None
        self.grid_cache = {}
        self.transform_cache = {}
        self.last_params = params

        # Derived from the lines this run just produced, so it has to come
        # after self.processed/line_spacing_m are in place. Never fatal:
        # the boundary is a display refinement, and a survey too sparse or
        # oddly shaped to outline should still process and grid normally
        # (capped at the convex hull as before), with the reason surfaced
        # in the summary rather than failing the whole run.
        self.auto_boundary_info = None
        if params.auto_display_boundary:
            try:
                self.auto_boundary_info = self.auto_display_boundary(params.display_boundary_buffer_m)
            except ProjectError as exc:
                self.auto_boundary_info = {"failed": True, "reason": str(exc)}

        self.mark_changed()
        return self.process_summary()

    def _check_file_level_offsets(self, df: pd.DataFrame) -> dict:
        """Flags a DC level shift between separately-uploaded flight files
        (e.g. flown on different days, or with the base station moved in
        between) - the UAV magnetics guidelines' most-cited cause of
        artefacts when a survey is assembled from multiple files/tiles.
        Only an approximate diagnostic (comparing each file's own median
        final anomaly against the pooled median, not true crossover
        misties), but it is a no-op (never modifies the data) so a false
        positive on genuinely-overlapping-but-different geology just shows
        an informational warning rather than any risk."""
        if "source_file_index" not in df.columns:
            return {"available": False}
        active = df["line_id"] >= 0
        kept = df.loc[active]
        n_files = kept["source_file_index"].nunique()
        if n_files < 2:
            return {"available": False}

        overall_median = float(kept["anomaly"].median())
        # Robust point-to-point noise estimate (MAD-based std of adjacent
        # differences within each file) as the yardstick for "how big an
        # offset would actually be suspicious" - scale-appropriate for
        # this survey's own noise level rather than a fixed nT threshold.
        diffs = kept.groupby("source_file_index")["anomaly"].apply(lambda s: np.diff(s.to_numpy()))
        nonempty_diffs = [d for d in diffs if len(d)]
        # Guard on the filtered list, not on len(diffs) - a file whose group
        # has exactly one point contributes a length-0 diff array, so a
        # groupby result with several such single-point files would have
        # len(diffs) > 0 while every entry is empty, and np.concatenate([])
        # raises ValueError ("need at least one array to concatenate").
        all_diffs = np.concatenate(nonempty_diffs) if nonempty_diffs else np.array([])
        noise_std = float(1.4826 * np.median(np.abs(all_diffs - np.median(all_diffs)))) if all_diffs.size else 0.0
        flag_threshold_nt = max(3.0 * noise_std, 1.0)

        files_out = []
        for idx, group in kept.groupby("source_file_index"):
            median_anomaly = float(group["anomaly"].median())
            deviation = median_anomaly - overall_median
            files_out.append(
                {
                    "source_file_index": int(idx),
                    "n_points": int(len(group)),
                    "median_anomaly_nt": median_anomaly,
                    "deviation_nt": deviation,
                    "flagged": bool(abs(deviation) > flag_threshold_nt),
                }
            )

        return {
            "available": True,
            "n_files": int(n_files),
            "flag_threshold_nt": flag_threshold_nt,
            "flagged_any": any(f["flagged"] for f in files_out),
            "files": files_out,
        }

    def _apply_heading_effect_calibration(self, df: pd.DataFrame, params: ProcessParams) -> dict:
        """Zhang et al. (2022) heading-effect compensation (see
        processing/heading_calibration.py): fits a DeltaB(theta, phi)
        deviation surface and subtracts it from df["mag_diurnal_corrected"]
        in place, before IGRF/anomaly are derived from it. A no-op
        (returns available=False) whenever no calibration source (uploaded
        flight or, failing that, the survey's own turn segments) is usable
        - callers don't need to check beforehand.

        The calibration source is a dedicated uploaded flight
        (self.calibration_raw) if present, otherwise - when
        auto_calibrate_from_turns is enabled - the survey's own turn
        segments (see processing/lines.py:group_turn_segments), per the
        manufacturer's own guidance that turns make good calibration data
        when no dedicated flight was flown."""
        hec = params.heading_effect_calibration
        # compass_x/y/z are always present (possibly all-NaN) on anything
        # that went through io_/drone_loader.py, but a hand-built
        # DataFrame (as in some tests, or any future caller) may omit them
        # entirely - treat a missing column the same as an all-NaN one
        # rather than raising a KeyError.
        if "compass_x" not in df.columns or df["compass_x"].notna().sum() < 1:
            return {"enabled": True, "available": False, "applied": False}

        if self.calibration_raw is not None:
            cal = self.calibration_raw
            if "compass_x" not in cal.columns or cal["compass_x"].notna().sum() < 30:
                return {
                    "enabled": True,
                    "available": False,
                    "applied": False,
                    "reason": "업로드된 캘리브레이션 자료에 나침반(Compass) 데이터가 부족합니다 (지원 포맷: Geometrics MagArrow 등).",
                }
            if self.base_processed is not None:
                cal_diurnal_corrected = apply_diurnal_correction(
                    cal["timestamp"],
                    cal["mag_raw"].to_numpy(),
                    self.base_processed,
                    time_offset_seconds=params.diurnal_params.time_offset_seconds,
                    reference=params.diurnal_params.reference,
                ).corrected
            else:
                # No base station (assume_constant mode) - nothing to
                # subtract, use the calibration flight's own reading as-is.
                cal_diurnal_corrected = cal["mag_raw"].to_numpy()
            theta_cal, phi_cal = magnetic_heading(
                cal["compass_x"].to_numpy(), cal["compass_y"].to_numpy(), cal["compass_z"].to_numpy()
            )
            try:
                heading_map = fit_heading_effect_map(theta_cal, phi_cal, cal_diurnal_corrected)
            except HeadingCalibrationError as exc:
                return {"enabled": True, "available": False, "applied": False, "reason": str(exc)}
            calibration_source = "uploaded_file"
        elif hec.auto_calibrate_from_turns:
            turn_group_id = group_turn_segments(df)
            theta_turn, phi_turn = magnetic_heading(
                df["compass_x"].to_numpy(), df["compass_y"].to_numpy(), df["compass_z"].to_numpy()
            )
            try:
                heading_map = build_turn_based_calibration(
                    theta_turn, phi_turn, df["mag_diurnal_corrected"].to_numpy(), turn_group_id
                )
            except HeadingCalibrationError as exc:
                return {"enabled": True, "available": False, "applied": False, "reason": str(exc)}
            calibration_source = "auto_turns"
        else:
            return {"enabled": True, "available": False, "applied": False}

        quality = cross_validate_heading_effect_map(
            heading_map.theta_cal, heading_map.phi_cal, heading_map.deviation_cal
        )
        quality_pass = quality["available"] and quality["residual_std_nt"] <= hec.quality_threshold_nt

        coverage = calibration_angular_coverage_deg(heading_map.theta_cal, heading_map.phi_cal)

        # A calibration surface that fails its own held-out quality check
        # (see cross_validate_heading_effect_map) is fit to noise/real
        # spatial gradient rather than a clean heading effect - querying it
        # would inject that noise into every survey sample it touches
        # (point-to-point jumps as large as the underlying anomaly signal
        # itself, confirmed on real survey data), showing up as along-track
        # corrugation in the gridded output. Geometrics' own processing
        # software gates compensation the same way (Pass/Fail on the
        # calibration file before it can be used) - mirror that rather than
        # silently applying a failed calibration anyway.
        if not quality_pass:
            return {
                "enabled": True,
                "available": True,
                "applied": False,
                "calibration_source": calibration_source,
                "n_calibration_points": int(len(heading_map.theta_cal)),
                "quality_check": quality,
                "quality_pass": False,
                "quality_threshold_nt": hec.quality_threshold_nt,
                "theta_range_deg": coverage["theta_range_deg"],
                "phi_range_deg": coverage["phi_range_deg"],
                "reason": (
                    "캘리브레이션 품질검증 실패로 보정을 적용하지 않았습니다 "
                    f"(held-out 잔차 표준편차 {quality.get('residual_std_nt', float('nan')):.2f}nT > "
                    f"허용기준 {hec.quality_threshold_nt}nT). 측선 자료 자체가 원래 상태로 유지됩니다."
                ),
            }

        theta_survey, phi_survey = magnetic_heading(
            df["compass_x"].to_numpy(), df["compass_y"].to_numpy(), df["compass_z"].to_numpy()
        )
        deviation, extrapolated = heading_map.query(theta_survey, phi_survey)

        corrected_mask = np.isfinite(deviation)
        if corrected_mask.any():
            df.loc[corrected_mask, "mag_diurnal_corrected"] = (
                df.loc[corrected_mask, "mag_diurnal_corrected"] - deviation[corrected_mask]
            )

        n = len(corrected_mask)
        pct_extrapolated = float(100.0 * (extrapolated & corrected_mask).sum() / n) if n else 0.0
        return {
            "enabled": True,
            "available": True,
            "applied": bool(corrected_mask.any()),
            "calibration_source": calibration_source,
            "n_calibration_points": int(len(heading_map.theta_cal)),
            "n_survey_points_corrected": int(corrected_mask.sum()),
            "n_survey_points_extrapolated": int((extrapolated & corrected_mask).sum()),
            "pct_survey_points_corrected": float(100.0 * corrected_mask.sum() / n) if n else 0.0,
            "pct_survey_points_extrapolated": pct_extrapolated,
            "coverage_warning": (
                "캘리브레이션이 실제 비행에서 나타난 자세 범위의 일부만 포괄합니다 "
                f"(측선 포인트의 {pct_extrapolated:.1f}%가 캘리브레이션 범위 밖 - 가장 가까운 값으로 대체 보정됨)."
                if pct_extrapolated > 20.0
                else None
            ),
            "mean_abs_correction_nt": float(np.nanmean(np.abs(deviation[corrected_mask]))) if corrected_mask.any() else 0.0,
            "theta_range_deg": coverage["theta_range_deg"],
            "phi_range_deg": coverage["phi_range_deg"],
            "quality_check": quality,
            "quality_pass": quality_pass,
            "quality_threshold_nt": hec.quality_threshold_nt,
        }

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
            "source_removals": [r.summary() for r in self.source_removals],
            "repeat_pass_leveling": {
                "applied": bool(self.repeat_pass_offsets),
                "offsets": {str(k): round(float(v), 3) for k, v in self.repeat_pass_offsets.items()},
            },
            "height_normalization": self._height_normalization_summary(),
            "n_manual_excluded": sum(1 for v in self.manual_overrides.values() if not v),
            "utm_epsg": self.utm_epsg,
            "dominant_azimuth_deg": self.dominant_azimuth_deg,
            "line_spacing_m": self.line_spacing_m,
            "inclination_deg": self.inclination_deg,
            "declination_deg": self.declination_deg,
            "diurnal": self.diurnal_info,
            "base_qc": self.base_qc_info,
            "despike": self.despike_info,
            "sway_detection": self.sway_info,
            "heading_effect_calibration": self.heading_calibration_info,
            "gps_mag_lag": self.gps_lag_info,
            "heading_correction": _heading_correction_summary(self.heading_leveling, self.heading_correction_applied),
            "crossover_leveling": self.crossover_info,
            "statistical_leveling": _statistical_leveling_summary(self.statistical_leveling),
            "striping": self.striping_info,
            "line_resolution": self.line_resolution_info,
            "noise_qc": self.noise_qc_info,
            "sampling_qc": self.sampling_qc_info,
            "file_level_check": self.file_level_info,
            "anomaly_stats": _stats(df.loc[active, "anomaly"]),
            "tmi_stats": _stats(df.loc[active, "tmi"]),
            "lines": _line_summaries(df, self.heading_leveling, self.heading_correction_applied),
            "line_overlaps": self.overlap_info,
            "display_boundary_polygon": self.display_boundary_polygon,
            "auto_boundary": self.auto_boundary_info,
        }

    def run_chat(self, message: str, history: list[dict]) -> dict:
        from .chat import run_chat_turn

        return run_chat_turn(self, message, history)

    def stream_chat(self, message: str, history: list[dict]):
        from .chat import stream_chat_turn

        return stream_chat_turn(self, message, history)

    def generate_report(self) -> str:
        return generate_report_markdown(
            self.drone_summary() if self.drone_raw is not None else None,
            self.base_summary() if self.base_raw is not None else None,
            self.process_summary() if self.processed is not None else None,
            self.last_params.model_dump() if self.last_params is not None else None,
            self.inversion_summary_cache,
            self.euler_summary_cache,
            self.target_summary_cache,
            repeatability_summary=self.repeatability_summary_cache,
            multiscale_edges_summary=self.multiscale_edges_summary_cache,
            qc_certificate=self.qc_certificate_cache,
            candidate_notes=self.candidate_notes,
            repeat_pass_summary=self.repeat_pass_summary_cache,
            geology_stats=self.geology_stats_cache,
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
                "is_ramp": df["exclusion_reason"].isin(["takeoff_ramp", "landing_ramp"]),
                "timestamp": df["timestamp"].astype(str),
            }
        )
        return out.to_dict(orient="records")

    def get_line_profile(self, line_id: int, value: str) -> dict:
        """Value-vs-along-track-distance series for a single flight line -
        a QC view distinct from the map/grid overlays, for spotting
        spikes, drift, or leveling offsets directly along one pass rather
        than inferring them from the 2D color pattern.

        Also returns a raw_value trace alongside the processed value, so
        the user can compare what despike/notch/lowpass filtering removed.
        Since diurnal and IGRF correction are purely additive/subtractive
        terms computed from timestamp/position alone (not from the drone's
        own magnetometer reading), the filtering pipeline's effect on the
        final anomaly/tmi can be isolated without re-running diurnal/IGRF/
        heading-correction on a separate raw path:
            raw_value = value + (mag_original - mag_filtered)
        mag_original is the sensor reading captured before despike/notch
        mutate mag_raw in place (see run_pipeline); mag_filtered is what
        those steps plus the lowpass/Savitzky-Golay filter produced.

        raw_value's `value` term is deliberately taken from
        processed_base (the pre-manual-smoothing snapshot), not the
        possibly-smoothed self.processed - manual smoothing is a separate,
        later edit from filtering, and letting it leak into the "before
        filtering" reference trace would distort that comparison (and,
        once a stretch is smoothed, make the reference trace mirror the
        smoothing edit instead of showing the true original signal it's
        meant to show)."""
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
        filter_delta = line_df["mag_original"] - line_df["mag_filtered"]
        pre_smooth_df = self.processed_base if self.processed_base is not None else self.processed
        pre_smooth_value = pre_smooth_df.loc[line_df.index, col]

        return {
            "line_id": line_id,
            "point_id": line_df["point_id"].tolist(),
            "distance_m": distance_m.tolist(),
            "value": line_df[col].tolist(),
            "raw_value": (pre_smooth_value + filter_delta).tolist(),
            "lat": line_df["lat"].tolist(),
            "lon": line_df["lon"].tolist(),
            "excluded": (~active.reindex(line_df.index)).tolist(),
            "smoothed": line_df["point_id"].isin(self.manual_smooth_point_ids).tolist(),
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
        with self._edit_lock:
            return self._set_manual_exclude_unlocked(req)

    def _set_manual_exclude_unlocked(self, req: ManualExcludeRequest) -> dict:
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed

        if req.mode == "reset":
            self.manual_overrides = {}
            self.grid_cache = {}
            self.transform_cache = {}
            self.mark_changed()
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

        # Takeoff/landing ramp points are protected from exclude/include
        # drawing by default (see ManualExcludeRequest.include_ramp) - the
        # line editor hides them and this stops a draw action from
        # silently toggling them just because they happened to fall
        # inside the drawn shape.
        if not req.include_ramp:
            ramp_ids = set(df.loc[df["exclusion_reason"].isin(["takeoff_ramp", "landing_ramp"]), "point_id"])
            target_ids -= ramp_ids

        # action="include" force-includes points regardless of automatic
        # line detection (e.g. restoring a turbulence segment the auto
        # detector dropped); action="exclude" force-excludes regardless.
        forced_value = req.action == "include"
        for pid in target_ids:
            self.manual_overrides[int(pid)] = forced_value
        self.grid_cache = {}
        self.transform_cache = {}
        self.mark_changed()
        return {**self.process_summary(), "exclusion": self.get_exclusion_state()}

    def _flight_levelled_base(self) -> pd.DataFrame:
        """processed_base with the per-flight offsets solved on repeated
        ground subtracted (see analyze_repeat_passes). First in the chain,
        because it is a levelling correction: a structure model fitted on
        unlevelled data would absorb part of the level difference."""
        base = self.processed_base
        if not self.repeat_pass_offsets or base is None:
            return base
        if "source_file_index" not in base.columns:
            return base
        shift = base["source_file_index"].map(
            {int(k): float(v) for k, v in self.repeat_pass_offsets.items()}
        ).fillna(0.0).to_numpy()
        out = base.copy()
        for col in ("anomaly", "tmi"):
            out[col] = out[col].to_numpy() - shift
        return out

    def _height_normalized_base(self) -> pd.DataFrame:
        """The flight-levelled base with every reading restated at the
        normalisation's reference altitude. After the levelling, because
        the layers are fitted on levelled data; before the structure
        models, so those are fitted on readings at one height."""
        base = self._flight_levelled_base()
        if self.height_normalization is None or base is None:
            return base
        correction = self._height_correction_for(base)
        out = base.copy()
        for col in ("anomaly", "tmi"):
            out[col] = out[col].to_numpy() + correction
        return out

    def _height_correction_for(self, base: pd.DataFrame) -> np.ndarray:
        """Per-row correction of `base` (aligned to its rows), from the
        cache when the rows are the ones it was computed for."""
        ids = base["point_id"].to_numpy()
        cache = self._height_correction_cache
        if cache is not None and len(cache[0]) == len(ids) and np.array_equal(cache[0], ids):
            return cache[1]
        if "altitude_ellipsoidal_m" in base.columns:
            z = base["altitude_ellipsoidal_m"].to_numpy(dtype=float)
        else:
            z = np.full(len(base), np.nan)
        correction = self.height_normalization.correction(base["x"].to_numpy(), base["y"].to_numpy(), z)
        self._height_correction_cache = (ids.copy(), correction)
        return correction

    def _height_normalization_summary(self) -> dict:
        model = self.height_normalization
        return {"applied": model is not None, **(model.summary() if model is not None else {})}

    def _base_without_structures(self) -> pd.DataFrame:
        """processed_base, levelled on repeated ground and restated at one
        altitude, with every fitted structure model subtracted."""
        base = self._height_normalized_base()
        if not self.source_removals:
            return base
        out = base.copy()
        x, y = out["x"].to_numpy(), out["y"].to_numpy()
        for col in ("anomaly", "tmi"):
            out[col] = apply_source_removals(x, y, out[col].to_numpy(), self.source_removals)
        return out

    def _rebuild_processed(self, base_df: pd.DataFrame | None = None) -> None:
        if base_df is None:
            base_df = self._base_without_structures()
        self.processed = apply_manual_smoothing(base_df, self.manual_smooth_point_ids)
        self.grid_cache = {}
        self.transform_cache = {}

    def set_source_removal(self, req: SourceRemovalRequest) -> dict:
        with self._edit_lock:
            return self._set_source_removal_unlocked(req)

    def _set_source_removal_unlocked(self, req: SourceRemovalRequest) -> dict:
        """Remove ground-structure anomalies by fitting each marked region
        as a patch of dipoles and subtracting the fitted field - see
        processing/source_removal.py for why this, rather than cutting the
        region out and interpolating across it."""
        if self.processed_base is None or self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        added: list[dict] = []
        if req.mode == "reset":
            self.source_removals = []
        elif req.mode == "undo":
            if self.source_removals:
                self.source_removals.pop()
        else:
            if not req.polygons:
                raise ProjectError("polygons가 필요합니다.")
            if self.utm_epsg is None:
                raise ProjectError("좌표계 정보가 없습니다.")
            to_local = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
            active = self._active_mask().to_numpy()
            for i, polygon in enumerate(req.polygons):
                if len(polygon) < 3:
                    continue
                # Fit against the data with earlier removals already taken
                # out, so two neighbouring structures are not counted twice.
                current = self._base_without_structures()
                xs, ys = to_local.transform([pt[1] for pt in polygon], [pt[0] for pt in polygon])
                hint = None
                if req.depth_hints_m and i < len(req.depth_hints_m):
                    hint = req.depth_hints_m[i]
                try:
                    removal = fit_source_removal(
                        current["x"].to_numpy()[active],
                        current["y"].to_numpy()[active],
                        current["anomaly"].to_numpy()[active],
                        current["line_id"].to_numpy()[active],
                        list(zip(xs, ys)),
                        self.inclination_deg,
                        self.declination_deg,
                        self.line_spacing_m,
                        depth_hint_m=hint,
                    )
                except ValueError as exc:
                    raise ProjectError(f"구조물 모델을 맞출 수 없습니다: {exc}") from exc
                self.source_removals.append(removal)
                added.append(removal.summary())
        self._rebuild_processed()
        self.mark_changed()
        return {
            **self.process_summary(),
            "exclusion": self.get_exclusion_state(),
            "n_source_removals": len(self.source_removals),
            "added": added,
            "source_removals": [r.summary() for r in self.source_removals],
        }

    def set_manual_smoothing(self, req: ManualSmoothRequest) -> dict:
        with self._edit_lock:
            return self._set_manual_smoothing_unlocked(req)

    def _set_manual_smoothing_unlocked(self, req: ManualSmoothRequest) -> dict:
        """Remove a user-identified ground-structure distortion (a house,
        building etc. visibly perturbing the signal) from a stretch of a
        line, by linearly interpolating anomaly/tmi across it - see
        processing/manual_smooth.py. Always recomputed from
        self.processed_base (the pristine post-pipeline, pre-smoothing
        snapshot) rather than compounding onto the currently-displayed
        values, mirroring why manual_overrides is applied lazily via
        _active_mask() instead of mutating stored values - except here the
        edit genuinely must mutate anomaly/tmi, so the pristine snapshot is
        what makes "always fresh, never compounding" possible."""
        if self.processed_base is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        base_df = self._base_without_structures()

        if req.mode == "reset":
            self.manual_smooth_point_ids = set()
        else:
            if req.mode == "point_ids":
                if not req.point_ids:
                    raise ProjectError("point_ids가 필요합니다.")
                new_ids = set(req.point_ids)
            elif req.mode == "polygon":
                if not req.polygon or len(req.polygon) < 3:
                    raise ProjectError("polygon은 최소 3개의 [lat, lon] 좌표가 필요합니다.")
                poly_path = MplPath([(pt[1], pt[0]) for pt in req.polygon])  # (lon, lat)
                inside = poly_path.contains_points(np.column_stack([base_df["lon"], base_df["lat"]]))
                new_ids = set(base_df.loc[inside, "point_id"])
            else:  # "polygons" - a batch (e.g. every region a structure-
                # distortion auto-scan found and the user confirmed),
                # unioned into this single call/history entry instead of
                # one call per region.
                if not req.polygons:
                    raise ProjectError("polygons가 필요합니다.")
                new_ids = set()
                for polygon in req.polygons:
                    if len(polygon) < 3:
                        continue
                    poly_path = MplPath([(pt[1], pt[0]) for pt in polygon])  # (lon, lat)
                    inside = poly_path.contains_points(np.column_stack([base_df["lon"], base_df["lat"]]))
                    new_ids |= set(base_df.loc[inside, "point_id"])
            self.manual_smooth_point_ids |= new_ids

        self._rebuild_processed(base_df)
        self.mark_changed()
        return {
            **self.process_summary(),
            "exclusion": self.get_exclusion_state(),
            # Full current set (not just this call's delta) - lets the
            # frontend keep an undo history of complete states without
            # needing to separately track which points each past polygon/
            # point_ids call actually matched.
            "manual_smooth_point_ids": sorted(int(p) for p in self.manual_smooth_point_ids),
        }

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

    def generate_qc_certificate(self, req: QcCertificateRequest) -> dict:
        """Standard QC pass/fail certificate - see
        processing/qc_certificate.py for what each criterion checks and why
        a missing metric degrades to "평가 불가" instead of pass/fail."""
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        result = evaluate_qc_certificate(
            self.process_summary(),
            self.repeatability_summary_cache,
            repeat_pass_summary=self.repeat_pass_summary_cache,
            noise_threshold_multiplier=req.noise_threshold_multiplier,
            max_repeatability_1sigma_nt=req.max_repeatability_1sigma_nt,
            max_sampling_gap_pct=req.max_sampling_gap_pct,
            max_excluded_pct=req.max_excluded_pct,
        )
        self.qc_certificate_cache = result
        return result

    # ------------------------------------------------------------ repeat passes
    def _repeat_pairs(self, req) -> list:
        if self.processed is None or self.dominant_azimuth_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed.loc[self._active_mask()]
        return find_repeat_passes(
            df,
            self.dominant_azimuth_deg,
            self.line_spacing_m,
            value=getattr(req, "value", "anomaly"),
            min_overlap_m=req.min_overlap_m,
            across_tolerance_m=req.across_tolerance_m,
        )

    def analyze_repeat_passes(self, req: RepeatPassRequest) -> dict:
        """Where the survey flew the same ground twice, how far apart the
        two readings are, and what one offset per flight would do about
        it. Measures only - see apply_repeat_pass_leveling."""
        pairs = self._repeat_pairs(req)
        offsets = solve_offsets(pairs)
        self.repeat_pass_summary_cache = summarize_repeat_passes(pairs, offsets)
        return {
            "pairs": [p.to_dict() for p in pairs],
            "offsets": {str(k): round(v, 3) for k, v in offsets.items()},
            "applied": bool(self.repeat_pass_offsets),
            "applied_offsets": {str(k): round(float(v), 3) for k, v in self.repeat_pass_offsets.items()},
            **summarize_repeat_passes(pairs, offsets),
        }

    def apply_repeat_pass_leveling(self, req: RepeatPassApplyRequest) -> dict:
        with self._edit_lock:
            return self._apply_repeat_pass_leveling_unlocked(req)

    def _apply_repeat_pass_leveling_unlocked(self, req: RepeatPassApplyRequest) -> dict:
        """Subtract the solved per-flight offsets from the data, or put
        them back. Sits before structure removal and point smoothing in
        the chain, so both keep working on top of it."""
        if req.mode == "reset":
            self.repeat_pass_offsets = {}
        else:
            pairs = self._repeat_pairs(req)
            offsets = solve_offsets(pairs)
            if not offsets:
                raise ProjectError(
                    "같은 구간을 두 번 비행한 자료가 없어 비행 간 레벨을 맞출 수 없습니다."
                )
            self.repeat_pass_offsets = offsets
        self._rebuild_processed()
        self.mark_changed()
        return {
            "applied": bool(self.repeat_pass_offsets),
            "offsets": {str(k): round(float(v), 3) for k, v in self.repeat_pass_offsets.items()},
            **self.process_summary(),
        }

    # ------------------------------------------------- height normalisation
    def height_normalization_status(self) -> dict:
        """How much the sensor altitude varied over the readings in use,
        the height above the DEM when one is loaded, and the state of the
        normalisation. Measures only - see apply_height_normalization."""
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed
        active = self._active_mask().to_numpy()
        if "altitude_ellipsoidal_m" in df.columns:
            z = df["altitude_ellipsoidal_m"].to_numpy(dtype=float)
        else:
            z = np.full(len(df), np.nan)
        stats = altitude_stats(z[active])
        available = stats["n"] >= 50
        return {
            "available": available,
            "reason": None if available else (
                "측점 고도(타원체고)가 없어 비행 고도 정규화를 할 수 없습니다. "
                "자료에 GPS 고도와 지오이드 높이가 있어야 합니다."
            ),
            "altitude": stats,
            "agl": self._agl_stats(df.loc[active]),
            "z_ref_default_m": stats["median_m"],
            "line_spacing_m": self.line_spacing_m,
            **self._height_normalization_summary(),
        }

    def _agl_stats(self, sub: pd.DataFrame) -> dict | None:
        """Height of the readings above the loaded DEM: a check that the
        flight really followed the terrain. The DEM is compared with the
        MSL altitude when the data has one, because DEMs are normally
        orthometric. Best effort - None when there is no DEM or it does
        not cover the survey."""
        if self.dem_bytes is None or len(sub) == 0:
            return None
        try:
            stride = max(1, len(sub) // 20000)
            pts = sub.iloc[::stride]
            if "altitude_msl_m" in pts.columns and pts["altitude_msl_m"].notna().any():
                alt_col = "altitude_msl_m"
            else:
                alt_col = "altitude_ellipsoidal_m"
            alt = pts[alt_col].to_numpy(dtype=float)
            with rasterio.open(io.BytesIO(self.dem_bytes)) as src:
                to_dem = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                dx, dy = to_dem.transform(pts["lon"].to_numpy(dtype=float), pts["lat"].to_numpy(dtype=float))
                ground = np.array([v[0] for v in src.sample(zip(dx, dy))], dtype=float)
                if src.nodata is not None:
                    ground[ground == src.nodata] = np.nan
            agl = alt - ground
            agl = agl[np.isfinite(agl)]
            if len(agl) == 0:
                return None
            return {
                "n": int(len(agl)),
                "min_m": round(float(agl.min()), 1),
                "median_m": round(float(np.median(agl)), 1),
                "max_m": round(float(agl.max()), 1),
                "std_m": round(float(agl.std()), 1),
                "altitude_column": alt_col,
                "dem_name": self.dem_name,
            }
        except Exception:  # noqa: BLE001 - a report, never a failure
            return None

    def apply_height_normalization(self, req: HeightNormalizationApplyRequest) -> dict:
        with self._edit_lock:
            return self._apply_height_normalization_unlocked(req)

    @_as_job("비행 고도 정규화", rollback=True)
    def _apply_height_normalization_unlocked(self, req: HeightNormalizationApplyRequest) -> dict:
        """Fit the equivalent layers to the flight-levelled readings at
        their true altitudes and restate every reading at z_ref, or undo
        that. Sits after the flight levelling and before structure removal
        and point smoothing in the chain - see _height_normalized_base."""
        if self.processed_base is None or self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        if req.mode == "reset":
            self.height_normalization = None
            self._height_correction_cache = None
        else:
            base = self._flight_levelled_base()
            active = self._active_mask().to_numpy()
            if "altitude_ellipsoidal_m" in base.columns:
                z = base["altitude_ellipsoidal_m"].to_numpy(dtype=float)
            else:
                z = np.full(len(base), np.nan)
            values = base["anomaly"].to_numpy(dtype=float)
            usable = active & np.isfinite(z) & np.isfinite(values)
            if int(usable.sum()) < 50:
                raise ProjectError(
                    "측점 고도(타원체고)가 없어 비행 고도 정규화를 할 수 없습니다. "
                    "자료에 GPS 고도와 지오이드 높이가 있어야 합니다."
                )
            try:
                model = fit_height_normalization(
                    base["x"].to_numpy(dtype=float)[usable],
                    base["y"].to_numpy(dtype=float)[usable],
                    z[usable],
                    values[usable],
                    self.inclination_deg,
                    self.declination_deg,
                    self.line_spacing_m,
                    z_ref_m=req.z_ref_m,
                    cell_size_m=req.cell_size_m,
                    max_iter=req.max_iter,
                )
            except HeightNormalizationError as exc:
                raise ProjectError(f"비행 고도 정규화를 할 수 없습니다: {exc}") from exc
            self.height_normalization = model
            self._height_correction_cache = None
            correction = self._height_correction_for(base)
            model.correction_stats = correction_stats(correction[active])
        self._rebuild_processed()
        self.mark_changed()
        return {
            "applied": self.height_normalization is not None,
            **self.process_summary(),
            "exclusion": self.get_exclusion_state(),
        }

    def mark_changed(self) -> None:
        """Note that this project now holds work that is not on disk.
        autosave.py writes it out a few seconds later - see that module
        for why the write does not happen here."""
        self.changed_at = time.time()


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

    def get_or_create(self, project_id: str) -> Project:
        """The project under this id, creating an empty one if this server
        has never seen it - how a project autosaved before a restart comes
        back under the same id (see autosave.py::restore)."""
        with self._lock:
            project = self._projects.get(project_id)
            if project is None:
                project = Project(id=project_id)
                self._projects[project_id] = project
        return project

    def peek(self, project_id: str) -> Project | None:
        """The project if this server has it in memory, without raising."""
        with self._lock:
            return self._projects.get(project_id)

    def all_projects(self) -> list[Project]:
        with self._lock:
            return list(self._projects.values())


store = ProjectStore()
