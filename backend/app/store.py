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
from scipy.interpolate import RegularGridInterpolator

from .io_.base_loader import load_base_csvs
from .io_.drone_loader import load_drone_csvs
from .models import (
    AnalyticSignalDepthRequest,
    ContactDetectionRequest,
    DisplayBoundaryRequest,
    EulerDeconvolutionRequest,
    GridConfidenceRequest,
    GridRequest,
    InversionParams,
    InversionSectionRequest,
    InversionSliceRequest,
    LineamentRequest,
    ManualExcludeRequest,
    ManualSmoothRequest,
    MultiscaleEdgeRequest,
    PowerSpectrumRequest,
    ProcessParams,
    ProspectivityRequest,
    QcCertificateRequest,
    SpectralDepthRequest,
    StructureScanRequest,
    TargetDetectionRequest,
    TiltDepthRequest,
    TransformRequest,
)
from .processing.base_qc import process_base_station
from .processing.manual_smooth import apply_manual_smoothing
from .processing.intermagnet import (
    IagaParseError,
    IntermagnetFetchError,
    _bearing_deg,
    estimate_base_from_observatories,
    fetch_observatory_dates,
    haversine_km,
    parse_iaga2002,
    select_nearest_observatories,
)
from .processing.crossover_leveling import CrossoverLevelingResult, apply_crossover_leveling, compute_crossover_leveling
from .processing.despike import despike
from .processing.dipole_fit import classify_moment, detect_targets
from .processing.osm_structures import OsmFetchError, buffer_structures_to_polygons, fetch_osm_structures
from .processing.diurnal import apply_diurnal_correction
from .processing.filters import lowpass_filter, moving_average_filter, notch_filter, savgol_filter_1d
from .processing.gridding import GridResult, _nearest_axis_index, grid_confidence, grid_points
from .processing.igrf import compute_igrf_total_field, mean_field_intensity_nt, mean_inclination_declination
from .processing.inversion import (
    InversionError,
    InversionMesh,
    InversionResult,
    box_faces,
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
    group_turn_segments,
    project_to_local_xy,
    resolve_korea_projection_epsg,
)
from .processing.contours import compute_contours
from .processing.euler_deconvolution import run_euler_deconvolution as _euler_deconvolution_solve
from .processing.geology_sample import GeologySampleError, sample_geotiff_at_point
from .processing.gps_lag import apply_gps_mag_lag
from .processing.heading_calibration import (
    HeadingCalibrationError,
    build_turn_based_calibration,
    calibration_angular_coverage_deg,
    cross_validate_heading_effect_map,
    fit_heading_effect_map,
    magnetic_heading,
)
from .processing.overlay_image import OverlayImageError, load_geotiff_overlay
from .processing.qc_certificate import evaluate_qc_certificate
from .processing.report import generate_report_markdown
from .processing.sway import detect_sway
from .processing.duplicate_lines import resolve_duplicate_lines
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
from .processing.depth_estimation import (
    analytic_signal_depth_estimates,
    spectral_depth_diagnostic,
    tilt_depth_estimates,
)
from .processing.contacts import detect_magnetic_contacts
from .processing.lineaments import extract_lineaments
from .processing.microlevel import apply_microleveling
from .processing.multiscale_edges import run_multiscale_edges as _multiscale_edges_solve
from .processing.prospectivity import compute_prospectivity
from .processing.noise_qc import compute_difference_qc
from .processing.repeatability import analyze_repeatability
from .processing.sampling_qc import compute_sampling_distance_qc
from .processing.spectrum import compute_power_spectrum
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
    base_processed: pd.DataFrame | None = None  # base_raw after trim+despike QC (base_qc.py) - see run_pipeline
    base_qc_info: dict | None = None
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
    # Optional user-drawn [[lat, lon], ...] polygon that clips grid display
    # (overlay/export) to exactly that outline, on top of the automatic
    # convex-hull extrapolation cap in gridding.py - see set_display_boundary.
    display_boundary_polygon: list[list[float]] | None = None
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
    sway_info: dict | None = None
    duplicate_line_info: dict | None = None
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
    last_overlay_values: np.ndarray | None = None
    last_overlay_easting: np.ndarray | None = None
    last_overlay_northing: np.ndarray | None = None
    last_overlay_label: str | None = None

    def load_drone(self, buffers: list) -> dict:
        self.drone_raw = load_drone_csvs(buffers)
        return self.drone_summary()

    def load_base(self, buffers: list, filenames: list | None = None) -> dict:
        self.base_raw = load_base_csvs(buffers, filenames)
        self.base_source = None
        return self.base_summary()

    def _survey_centroid(self) -> tuple[float, float] | None:
        if self.drone_raw is None or len(self.drone_raw) == 0:
            return None
        return float(self.drone_raw["lat"].mean()), float(self.drone_raw["lon"].mean())

    def _preview_from_iaga(self, data, estimated_dates: list | None = None) -> dict:
        self.intermagnet_preview = data
        centroid = self._survey_centroid()
        distance_km = (
            haversine_km(centroid[0], centroid[1], data.lat, data.lon)
            if centroid is not None and data.lat is not None and data.lon is not None
            else None
        )
        return {
            "station_name": data.station_name,
            "iaga_code": data.iaga_code,
            "lat": data.lat,
            "lon": data.lon,
            "elevation_m": data.elevation_m,
            "reported": data.reported,
            "n_points": len(data.df),
            "time_range": [data.df["timestamp"].min().isoformat(), data.df["timestamp"].max().isoformat()],
            "distance_from_survey_km": distance_km,
            "estimated_dates": estimated_dates or [],
        }

    def preview_intermagnet_text(self, text: str) -> dict:
        """Parse a manually-downloaded IAGA-2002 file (from
        https://intermagnet.org or any GIN) without touching the network at
        all - the fallback path when fetch_intermagnet_preview can't reach
        the data service from wherever this server is deployed."""
        try:
            data = parse_iaga2002(text)
        except IagaParseError as exc:
            raise ProjectError(str(exc)) from exc
        return self._preview_from_iaga(data)

    def fetch_intermagnet_preview(self, iaga_code: str, dates: list) -> dict:
        """Download and parse an INTERMAGNET observatory's IAGA-2002 data
        for exactly the given calendar dates (typically the survey's own
        distinct flight dates - see _survey_dates - not a padded
        min-to-max range). Requires this server's own outbound network to
        reach the BGS GIN web service - see
        intermagnet.py::fetch_iaga2002_text for what to do when that's
        blocked (e.g. a locked-down deployment network) - use
        preview_intermagnet_text with a manually downloaded file instead.

        Any requested date that comes back entirely or mostly missing
        (definitive data for the most recent day or two often isn't
        published yet, and any specific day may simply be absent) is
        estimated from its own immediate neighbors - see
        processing/intermagnet.py::fetch_observatory_dates - and disclosed
        via estimated_dates in the returned preview."""
        try:
            data, estimated_dates = fetch_observatory_dates(iaga_code, dates)
        except IntermagnetFetchError as exc:
            raise ProjectError(str(exc)) from exc
        return self._preview_from_iaga(data, estimated_dates)

    def apply_intermagnet_preview(self) -> dict:
        """Commit the most recently previewed observatory data as this
        project's base (diurnal) station series - a separate confirm step
        so the user sees the station name/location/distance before it's
        used, rather than it being silently substituted."""
        if self.intermagnet_preview is None:
            raise ProjectError("먼저 INTERMAGNET 관측소 자료를 업로드하거나 다운로드하여 미리보기를 확인하세요.")
        data = self.intermagnet_preview
        self.base_raw = data.df
        self.base_source = {
            "type": "intermagnet",
            "station_name": data.station_name,
            "iaga_code": data.iaga_code,
            "lat": data.lat,
            "lon": data.lon,
        }
        self.intermagnet_preview = None
        return self.base_summary()

    def fetch_nearest_intermagnet_preview(
        self,
        dates: list,
        n_stations: int = 4,
        target_lat: float | None = None,
        target_lon: float | None = None,
    ) -> dict:
        """Auto-select a directionally spread set of nearby INTERMAGNET
        observatories (defaulting to the survey's own average GPS
        position when target_lat/target_lon aren't given) and combine
        their data into one substitute base station series - see
        processing/intermagnet.py::select_nearest_observatories /
        estimate_base_from_observatories. Requires this server's own
        outbound network to reach the data service.

        `dates` is typically the survey's own distinct flight dates (see
        _survey_dates), not a padded min-to-max range. Any requested date
        that comes back entirely or mostly missing for a selected station
        is estimated from its own immediate neighbors - see
        select_nearest_observatories's docstring and the per-station
        estimated_dates in the returned preview."""
        if target_lat is None or target_lon is None:
            centroid = self._survey_centroid()
            if centroid is None:
                raise ProjectError("대상 좌표가 없습니다 - 드론 자료를 먼저 업로드하거나 좌표를 직접 입력하세요.")
            target_lat, target_lon = centroid
        try:
            stations, estimated_dates_by_code = select_nearest_observatories(
                target_lat, target_lon, dates, n_stations=n_stations
            )
            combined = estimate_base_from_observatories(stations, target_lat, target_lon)
        except IntermagnetFetchError as exc:
            raise ProjectError(str(exc)) from exc

        result = {
            "target_lat": target_lat,
            "target_lon": target_lon,
            "stations": stations,
            "df": combined,
            "estimated_dates_by_code": estimated_dates_by_code,
        }
        self.intermagnet_nearest_preview = result
        self.intermagnet_nearest_last_result = result
        return {
            "target_lat": target_lat,
            "target_lon": target_lon,
            "stations": [
                {
                    "station_name": s.station_name,
                    "iaga_code": s.iaga_code,
                    "lat": s.lat,
                    "lon": s.lon,
                    "distance_km": haversine_km(target_lat, target_lon, s.lat, s.lon),
                    "bearing_deg": _bearing_deg(target_lat, target_lon, s.lat, s.lon),
                    "estimated_dates": estimated_dates_by_code.get(s.iaga_code, []),
                }
                for s in stations
            ],
            "n_points": len(combined),
            "time_range": [combined["timestamp"].min().isoformat(), combined["timestamp"].max().isoformat()],
        }

    def apply_nearest_intermagnet_preview(self) -> dict:
        """Commit the most recently previewed multi-observatory estimate
        as this project's base (diurnal) station series - a separate
        confirm step, matching apply_intermagnet_preview's UX."""
        if self.intermagnet_nearest_preview is None:
            raise ProjectError("먼저 주변 관측소 자료를 조회하여 미리보기를 확인하세요.")
        preview = self.intermagnet_nearest_preview
        self.base_raw = preview["df"]
        self.base_source = {
            "type": "intermagnet_nearest",
            "station_name": " + ".join(s.iaga_code for s in preview["stations"]),
            "iaga_code": None,
            "lat": preview["target_lat"],
            "lon": preview["target_lon"],
        }
        self.intermagnet_nearest_preview = None
        return self.base_summary()

    def get_nearest_intermagnet_comparison(self) -> dict:
        """The individual raw series of each station used in the most
        recent nearest-observatory fetch, alongside the combined (IDW)
        estimate built from them - for the user to visually sanity-check
        the estimate against the real station data (applied or still just
        previewed; see intermagnet_nearest_last_result)."""
        if self.intermagnet_nearest_last_result is None:
            raise ProjectError("먼저 주변 관측소 자료를 조회하세요.")
        result = self.intermagnet_nearest_last_result
        target_lat, target_lon = result["target_lat"], result["target_lon"]
        combined = result["df"]
        estimated_dates_by_code = result.get("estimated_dates_by_code", {})
        return {
            "combined": {
                "timestamp": combined["timestamp"].astype(str).tolist(),
                "mag": combined["mag"].tolist(),
            },
            "stations": [
                {
                    "station_name": s.station_name,
                    "iaga_code": s.iaga_code,
                    "distance_km": haversine_km(target_lat, target_lon, s.lat, s.lon),
                    "bearing_deg": _bearing_deg(target_lat, target_lon, s.lat, s.lon),
                    "timestamp": s.df["timestamp"].astype(str).tolist(),
                    "mag": s.df["mag"].tolist(),
                    "estimated_dates": estimated_dates_by_code.get(s.iaga_code, []),
                }
                for s in result["stations"]
            ],
        }

    def export_nearest_intermagnet_csv(self) -> bytes:
        """The combined multi-observatory estimate as plain CSV - for
        inspection, or reuse in another project via the IAGA-2002-style
        manual upload path (see preview_intermagnet_text) - though this
        export is plain (timestamp, mag) CSV, not IAGA-2002 fixed-width."""
        if self.intermagnet_nearest_preview is None:
            raise ProjectError("먼저 주변 관측소 자료를 조회하여 미리보기를 확인하세요.")
        df = self.intermagnet_nearest_preview["df"]
        out = df.rename(columns={"mag": "mag_nT"}).copy()
        out["timestamp"] = out["timestamp"].astype(str)
        return out.to_csv(index=False).encode("utf-8")

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
            LineDetectionParams(**self.last_params.line_params.model_dump()) if self.last_params else None
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
        if self.drone_raw is None:
            raise ProjectError("드론 자료를 먼저 업로드하세요.")
        if self.base_raw is None and params.diurnal_params.mode != "assume_constant":
            raise ProjectError(
                "베이스(일변화) 자료를 먼저 업로드하세요. 베이스 자료가 없다면 "
                "일변화 보정 방식을 '베이스 자료 없음(지구자기장 일정 가정)'으로 설정하세요."
            )

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

        line_params = LineDetectionParams(**params.line_params.model_dump())
        df = detect_lines(df, line_params, utm_epsg_override=utm_epsg_override)
        self.utm_epsg = df.attrs["utm_epsg"]
        self.dominant_azimuth_deg = df.attrs["dominant_azimuth_deg"]

        if params.noise_qc.enabled:
            self.noise_qc_info = compute_difference_qc(df, "mag_filtered")
        else:
            self.noise_qc_info = {"available": False}
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

        dlp = params.duplicate_line_params
        if dlp.enabled:
            # Run on "mag_filtered" (matches noise_qc.py's own QC channel -
            # right after flight-path cleaning, before diurnal/IGRF/leveling,
            # so which base-station diurnal reference is used can't bias the
            # quality comparison) and after sway detection (an already-
            # excluded swinging sample shouldn't count toward either pass's
            # overlap or quality). Before line_spacing_m below, so a
            # duplicate pass about to be excluded doesn't skew that
            # project-wide spacing estimate.
            self.duplicate_line_info = resolve_duplicate_lines(
                df, "mag_filtered", perp_tolerance_m=dlp.perp_tolerance_m, angle_tolerance_deg=dlp.angle_tolerance_deg
            )
            self.duplicate_line_info["enabled"] = True
            # Point ids are only needed to apply the exclusion below - kept
            # out of the stored summary (process_summary/report) so it isn't
            # carrying a potentially large raw id list around.
            excluded_ids = set(self.duplicate_line_info.pop("excluded_point_ids"))
            if excluded_ids:
                flagged_active = df["point_id"].isin(excluded_ids).to_numpy() & (df["line_id"].to_numpy() >= 0)
                df.loc[flagged_active, "exclusion_reason"] = "duplicate_repeat_flight"
                df.loc[flagged_active, "line_id"] = -1
                self.duplicate_line_info["n_points_excluded"] = int(flagged_active.sum())
        else:
            self.duplicate_line_info = {"enabled": False, "available": False, "n_groups": 0, "n_points_excluded": 0, "groups": []}

        self.line_spacing_m = estimate_line_spacing_m(df, self.dominant_azimuth_deg)

        if params.diurnal_params.mode == "assume_constant" and self.base_raw is None:
            # No base station was measured at all: assume the Earth's field
            # was steady over the (short) survey window, so there is no
            # diurnal (solar-driven) variation to remove - the drone's own
            # filtered reading is used as-is. Mathematically identical to a
            # base-station correction against a perfectly constant base
            # value (the correction term collapses to zero everywhere).
            self.base_processed = None
            self.base_qc_info = None
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
            self.base_processed = base_qc_result.corrected
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
            self.diurnal_info = {
                "mode": "base_station",
                "coverage_pct": diurnal_result.coverage_pct,
                "has_overlap": diurnal_result.has_overlap,
                "base_reference_value": diurnal_result.base_reference_value,
                "base_time_range": [str(diurnal_result.base_time_range[0]), str(diurnal_result.base_time_range[1])],
                "drone_time_range": [str(diurnal_result.drone_time_range[0]), str(diurnal_result.drone_time_range[1])],
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

        self.file_level_info = self._check_file_level_offsets(df)

        self.processed = df
        self.processed_base = df.copy()
        self.manual_overrides = {}
        self.manual_smooth_point_ids = set()
        self.grid_cache = {}
        self.transform_cache = {}
        self.last_params = params
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
        all_diffs = np.concatenate([d for d in diffs if len(d)]) if len(diffs) else np.array([])
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
            "duplicate_line_resolution": self.duplicate_line_info,
            "heading_effect_calibration": self.heading_calibration_info,
            "gps_mag_lag": self.gps_lag_info,
            "heading_correction": _heading_correction_summary(self.heading_leveling),
            "crossover_leveling": self.crossover_info,
            "noise_qc": self.noise_qc_info,
            "sampling_qc": self.sampling_qc_info,
            "file_level_check": self.file_level_info,
            "anomaly_stats": _stats(df.loc[active, "anomaly"]),
            "tmi_stats": _stats(df.loc[active, "tmi"]),
            "lines": _line_summaries(df, self.heading_leveling),
            "display_boundary_polygon": self.display_boundary_polygon,
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
            repeatability_summary=self.repeatability_summary_cache,
            multiscale_edges_summary=self.multiscale_edges_summary_cache,
            qc_certificate=self.qc_certificate_cache,
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
        return {**self.process_summary(), "exclusion": self.get_exclusion_state()}

    def set_manual_smoothing(self, req: ManualSmoothRequest) -> dict:
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
        base_df = self.processed_base

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

        self.processed = apply_manual_smoothing(base_df, self.manual_smooth_point_ids)
        self.grid_cache = {}
        self.transform_cache = {}
        return {
            **self.process_summary(),
            "exclusion": self.get_exclusion_state(),
            # Full current set (not just this call's delta) - lets the
            # frontend keep an undo history of complete states without
            # needing to separately track which points each past polygon/
            # point_ids call actually matched.
            "manual_smooth_point_ids": sorted(int(p) for p in self.manual_smooth_point_ids),
        }

    def set_display_boundary(self, req: DisplayBoundaryRequest) -> dict:
        """Set or clear the optional user-drawn boundary polygon that
        clips grid overlay/export output to exactly that outline - see
        display_boundary_polygon and _apply_display_boundary.

        Deliberately does *not* touch grid_cache/transform_cache: the
        boundary is applied as a cheap mask at render time (see
        _apply_display_boundary), not baked into the cached grid values,
        so drawing/clearing it doesn't force expensive regridding."""
        if req.polygon is not None and len(req.polygon) < 3:
            raise ProjectError("polygon은 최소 3개의 [lat, lon] 좌표가 필요합니다.")
        self.display_boundary_polygon = req.polygon
        return {"display_boundary_polygon": self.display_boundary_polygon}

    def _apply_display_boundary(self, values: np.ndarray, grid: GridResult) -> np.ndarray:
        """Mask `values` (a grid.values-shaped array) to NaN outside the
        user-drawn display_boundary_polygon, if one is set - complements
        the automatic convex-hull extrapolation cap in
        processing/gridding.py::grid_points, which still leaves the full
        convex hull filled even for a concave (e.g. L-shaped) survey
        footprint. Always returns a fresh array (never mutates `values` in
        place), since callers commonly pass in a grid_cache/transform_cache
        entry that must stay valid (boundary-free) for other callers/later
        boundary changes."""
        if not self.display_boundary_polygon or self.utm_epsg is None:
            return values
        from shapely import contains_xy
        from shapely.geometry import Polygon

        lat = [pt[0] for pt in self.display_boundary_polygon]
        lon = [pt[1] for pt in self.display_boundary_polygon]
        to_local = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
        bx, by = to_local.transform(lon, lat)
        polygon = Polygon(np.column_stack([bx, by]))
        easting_2d, northing_2d = np.meshgrid(grid.easting, grid.northing)
        inside = contains_xy(polygon, easting_2d.ravel(), northing_2d.ravel()).reshape(values.shape)
        return np.where(inside, values, np.nan)

    def _resolve_max_distance(self, cell_size_m: float, max_distance_m: float | None) -> float:
        if max_distance_m is not None:
            return max_distance_m
        if self.line_spacing_m:
            # Matches processing/gridding.py's _INTERIOR_FILL_FRACTION - see
            # its docstring for why 0.6 left too thin a margin.
            return max(2.0 * cell_size_m, 1.2 * self.line_spacing_m)
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
            # the raw (possibly None/auto) value, not resolved_max_distance -
            # None lets grid_points compute a per-cell local-line-gap-based
            # threshold instead of one project-wide constant (see its
            # docstring and _local_line_gap_m); resolved_max_distance above
            # is only used for the cache key, which just needs a stable,
            # unique-enough value, not the actual masking threshold.
            max_distance_m=max_distance_m,
            line_id=df.loc[active, "line_id"].to_numpy(),
            along_line_smooth_wavelength_m=effective_wavelength,
            typical_line_spacing_m=self.line_spacing_m,
        )
        self.grid_cache[key] = result
        return result

    def _cell_size_guideline_warning(self, cell_size_m: float) -> str | None:
        """The UAV magnetics survey guidelines' rule of thumb (Section
        11.1): grid to a cell size of 1/4 to 1/5 of the nominal line
        spacing - coarser under-resolves the data (unnecessary smoothing),
        finer risks fabricating detail the line spacing can't actually
        support (gridding artefacts). Purely informational (never blocks
        the request); a generous tolerance band around that rule since it
        is itself only a guideline, not a hard limit."""
        if not self.line_spacing_m or self.line_spacing_m <= 0:
            return None
        ratio = self.line_spacing_m / cell_size_m
        if ratio < 3.0:
            return (
                f"셀 크기({cell_size_m:.1f}m)가 측선 간격({self.line_spacing_m:.1f}m)에 비해 너무 큽니다 - "
                "가이드라인 권장 비율은 측선 간격의 1/4~1/5이며, 이보다 크면 자료가 불필요하게 뭉개질 수 있습니다."
            )
        if ratio > 8.0:
            return (
                f"셀 크기({cell_size_m:.1f}m)가 측선 간격({self.line_spacing_m:.1f}m)에 비해 너무 작습니다 - "
                "가이드라인 권장 비율(측선 간격의 1/4~1/5)보다 촘촘해 측선 사이 보간이 실제로 뒷받침하지 못하는 "
                "디테일이 만들어질 위험이 있습니다."
            )
        return None

    def get_grid_overlay(self, req: GridRequest) -> dict:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        grid = self._maybe_pre_level(grid, req)
        grid = replace(grid, values=self._apply_display_boundary(grid.values, grid))
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
        overlay["extrema"] = _extrema_locations(grid.values, grid.easting, grid.northing, self.utm_epsg)
        overlay["cell_size_m"] = grid.cell_size_m
        overlay["cell_size_guideline_warning"] = self._cell_size_guideline_warning(grid.cell_size_m)
        if req.show_contours:
            overlay["contours"] = compute_contours(
                grid.values, grid.easting, grid.northing, self.utm_epsg,
                interval=req.contour_interval_nt, n_levels=req.contour_n_levels,
            )
        self.last_overlay_values = grid.values
        self.last_overlay_easting = grid.easting
        self.last_overlay_northing = grid.northing
        self.last_overlay_label = req.value
        return overlay

    def get_grid_confidence_overlay(self, req: GridConfidenceRequest) -> dict:
        """A companion "how much should this cell be trusted" layer for
        get_grid_overlay's result - see processing/gridding.py::grid_confidence
        for what the 0-1 score means. Deliberately grids at the same cell
        size/method/max_distance the caller would use for the value layer
        itself (and shares its cache via _grid_for) so the two overlays
        line up cell-for-cell, but recomputes the confidence score fresh
        rather than trying to derive it from the already-masked GridResult
        (which only carries the final NaN/value array, not the distance
        field the score is built from)."""
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        df = self.processed
        active = self._active_mask()
        easting_2d, northing_2d = np.meshgrid(grid.easting, grid.northing)
        confidence = grid_confidence(
            df.loc[active, "x"].to_numpy(),
            df.loc[active, "y"].to_numpy(),
            easting_2d,
            northing_2d,
            grid.cell_size_m,
            max_distance_m=req.max_distance_m,
            line_id=df.loc[active, "line_id"].to_numpy(),
            typical_line_spacing_m=self.line_spacing_m,
        )
        overlay = grid_to_png_overlay(
            confidence,
            grid.easting,
            grid.northing,
            self.utm_epsg,
            cmap_name=req.colormap,
            symmetric=False,
            vmin=0.0,
            vmax=1.0,
            cell_size_m=grid.cell_size_m,
        )
        overlay["stats"] = _stats(pd.Series(confidence.ravel()))
        overlay["cell_size_m"] = grid.cell_size_m
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
        values = self._apply_display_boundary(values, grid)

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
        overlay["extrema"] = _extrema_locations(values, grid.easting, grid.northing, self.utm_epsg)
        overlay["cell_size_m"] = grid.cell_size_m
        overlay["cell_size_guideline_warning"] = self._cell_size_guideline_warning(grid.cell_size_m)
        overlay["transform"] = req.transform
        if req.show_contours:
            overlay["contours"] = compute_contours(
                values, grid.easting, grid.northing, self.utm_epsg,
                interval=req.contour_interval_nt, n_levels=req.contour_n_levels,
            )
        self.last_overlay_values = values
        self.last_overlay_easting = grid.easting
        self.last_overlay_northing = grid.northing
        self.last_overlay_label = req.transform
        return overlay

    def export_grid_geotiff(self, req: GridRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        grid = self._maybe_pre_level(grid, req)
        grid = replace(grid, values=self._apply_display_boundary(grid.values, grid))
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
        values = self._apply_display_boundary(values, grid)
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
        grid = replace(grid, values=self._apply_display_boundary(grid.values, grid))
        return grid_to_xyz_bytes(grid.values, grid.easting, grid.northing, self.utm_epsg)

    def export_transform_xyz(self, req: TransformRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        values, _symmetric = self._transform_values(grid, req)
        values = self._apply_display_boundary(values, grid)
        return grid_to_xyz_bytes(values, grid.easting, grid.northing, self.utm_epsg)

    def export_grid_surfer_grd(self, req: GridRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        grid = self._maybe_pre_level(grid, req)
        grid = replace(grid, values=self._apply_display_boundary(grid.values, grid))
        return grid_to_surfer_grd_bytes(grid.values, grid.easting, grid.northing)

    def export_transform_surfer_grd(self, req: TransformRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        values, _symmetric = self._transform_values(grid, req)
        values = self._apply_display_boundary(values, grid)
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

    def export_targets_csv(self) -> bytes:
        """The last run_target_detection() result (self.target_summary_cache)
        as a flat CSV table - lat/lon plus every dipole-fit attribute - for
        handing a detected-target list to a GIS or spreadsheet without the
        map UI. Empty (header-only) rather than an error when detection
        found nothing, since "ran clean, zero targets" is a valid result a
        caller may still want a well-formed CSV for."""
        if self.target_summary_cache is None:
            raise ProjectError("타겟 탐지를 먼저 실행하세요.")
        targets = self.target_summary_cache.get("targets", [])
        columns = ["lat", "lon", "depth_m", "moment_am2", "size_class", "peak_anomaly_nt", "footprint_m", "fit_quality", "background_nt"]
        out = pd.DataFrame(targets, columns=columns) if targets else pd.DataFrame(columns=columns)
        out.insert(0, "target_id", range(1, len(out) + 1))
        return out.to_csv(index=False).encode("utf-8")

    def export_targets_shapefile(self) -> bytes:
        """The same detected-target list as export_targets_csv, packaged as
        a zipped ESRI shapefile (.shp/.shx/.dbf/.prj point layer) for direct
        import into GIS software - the standard exchange format the
        "coverage/target handoff to GIS" use case actually wants, CSV alone
        requiring a manual re-projection/import step every time."""
        import io
        import zipfile

        import shapefile as pyshp

        if self.target_summary_cache is None:
            raise ProjectError("타겟 탐지를 먼저 실행하세요.")
        targets = self.target_summary_cache.get("targets", [])

        shp_buf, shx_buf, dbf_buf = io.BytesIO(), io.BytesIO(), io.BytesIO()
        writer = pyshp.Writer(shp=shp_buf, shx=shx_buf, dbf=dbf_buf, shapeType=pyshp.POINT)
        writer.field("target_id", "N")
        writer.field("depth_m", "F", decimal=2)
        writer.field("moment_am2", "F", decimal=4)
        writer.field("size_class", "C", size=40)
        writer.field("peak_nt", "F", decimal=2)
        # dbf field names are capped at 10 bytes - "footprint_m"/"fit_quality"
        # would silently truncate (and collide with nothing here, but it's
        # a latent footgun for any future field starting the same way).
        writer.field("footprnt_m", "F", decimal=2)
        writer.field("fit_qual", "F", decimal=3)
        writer.field("bg_nt", "F", decimal=2)
        for i, t in enumerate(targets, start=1):
            writer.point(t["lon"], t["lat"])
            writer.record(i, t["depth_m"], t["moment_am2"], t["size_class"], t["peak_anomaly_nt"], t["footprint_m"], t["fit_quality"], t["background_nt"])
        writer.close()

        # WGS84 geographic .prj - point() above was fed lon/lat, matching
        # this CRS, so the shapefile's coordinates and declared CRS agree
        # regardless of which UTM zone the project itself works in.
        prj_wkt = (
            'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("targets.shp", shp_buf.getvalue())
            zf.writestr("targets.shx", shx_buf.getvalue())
            zf.writestr("targets.dbf", dbf_buf.getvalue())
            zf.writestr("targets.prj", prj_wkt)
        return zip_buf.getvalue()

    def generate_qc_certificate(self, req: QcCertificateRequest) -> dict:
        """Standard QC pass/fail certificate - see
        processing/qc_certificate.py for what each criterion checks and why
        a missing metric degrades to "평가 불가" instead of pass/fail."""
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        result = evaluate_qc_certificate(
            self.process_summary(),
            self.repeatability_summary_cache,
            noise_threshold_multiplier=req.noise_threshold_multiplier,
            max_repeatability_1sigma_nt=req.max_repeatability_1sigma_nt,
            max_sampling_gap_pct=req.max_sampling_gap_pct,
            max_excluded_pct=req.max_excluded_pct,
        )
        self.qc_certificate_cache = result
        return result

    def run_euler_deconvolution(self, req: EulerDeconvolutionRequest) -> dict:
        # Defaults to along-line smoothing on (see _grid_for) - Euler
        # solutions are as sensitive to corrugation-driven spurious
        # gradients as any other derivative-based analysis, and this
        # request type has no UI toggle of its own for it.
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)

        altitude_grid = None
        if req.flight_agl_m is not None:
            if self.processed is None:
                raise ProjectError("자료 처리를 먼저 실행하세요.")
            df = self.processed
            active = self._active_mask()
            # Gridded with the exact same points/cell size/max_distance as
            # the anomaly grid above so the two line up cell-for-cell (same
            # pattern run_inversion uses for its obs_grid/alt_grid pair).
            altitude_grid = grid_points(
                df.loc[active, "x"].to_numpy(),
                df.loc[active, "y"].to_numpy(),
                df.loc[active, "altitude_ellipsoidal_m"].to_numpy(),
                grid.cell_size_m,
                method=req.method,
                max_distance_m=req.max_distance_m,
                line_id=df.loc[active, "line_id"].to_numpy(),
                typical_line_spacing_m=self.line_spacing_m,
            ).values

        solutions = _euler_deconvolution_solve(
            grid.values,
            grid.easting,
            grid.northing,
            grid.cell_size_m,
            self.utm_epsg,
            structural_index=req.structural_index,
            window_size_m=req.window_size_m,
            max_depth_uncertainty_pct=req.max_depth_uncertainty_pct,
            altitude_grid=altitude_grid,
            flight_agl_m=req.flight_agl_m,
        )
        depths = [s.depth_m for s in solutions]
        summary = {
            "n_solutions": len(solutions),
            "structural_index": req.structural_index,
            "depth_reference": solutions[0].depth_reference if solutions else (
                "ground_surface" if req.flight_agl_m is not None else "flat_datum"
            ),
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

    def run_multiscale_edges(self, req: MultiscaleEdgeRequest) -> dict:
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        points = _multiscale_edges_solve(
            grid.values,
            grid.easting,
            grid.northing,
            grid.cell_size_m,
            self.utm_epsg,
            heights_m=req.heights_m,
            percentile=req.percentile,
        )
        summary = {
            "n_points": len(points),
            "heights_m": req.heights_m,
            "percentile": req.percentile,
            "points": [
                {"lat": p.lat, "lon": p.lon, "height_m": p.height_m, "thdr_value": p.thdr_value} for p in points
            ],
        }
        self.multiscale_edges_summary_cache = summary
        return summary

    def run_magnetic_contact_detection(self, req: ContactDetectionRequest) -> dict:
        """Magnetic contact detection - see processing/contacts.py. Reuses
        the same multi-height THDR worming as run_multiscale_edges, but
        keeps only ridge points that persist across several heights and
        vectorizes them into contact segments (a proxy for geological
        contacts/rock-unit boundaries), meant to be compared against an
        uploaded geological-map reference layer."""
        if self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        result = detect_magnetic_contacts(
            grid.values,
            grid.easting,
            grid.northing,
            grid.cell_size_m,
            self.utm_epsg,
            heights_m=tuple(req.heights_m),
            percentile_threshold=req.percentile_threshold,
            min_persistence=req.min_persistence,
            min_segment_points=req.min_segment_points,
            min_length_m=req.min_length_m,
            max_gap_cells=req.max_gap_cells,
        )
        self.contact_summary_cache = result
        return result

    def run_lineament_extraction(self, req: LineamentRequest) -> dict:
        """Magnetic lineament extraction + rose-diagram structural
        statistics - see processing/lineaments.py. Along-line smoothing
        stays on (the default) for the underlying grid: lineament
        extraction reads a derivative of that grid, which would otherwise
        amplify any flight-line-parallel corrugation into spurious
        "lineaments" running exactly along the survey's own line
        direction."""
        if self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        source_grid = {
            "thd": lambda: total_horizontal_derivative(grid.values, grid.cell_size_m),
            "as": lambda: analytic_signal(grid.values, grid.cell_size_m),
            "tilt": lambda: tilt_angle(grid.values, grid.cell_size_m),
            "1vd": lambda: vertical_derivative(grid.values, grid.cell_size_m, order=1),
        }[req.source]()
        result = extract_lineaments(
            source_grid,
            grid.easting,
            grid.northing,
            grid.cell_size_m,
            self.utm_epsg,
            percentile_threshold=req.percentile_threshold,
            min_segment_points=req.min_segment_points,
            min_length_m=req.min_length_m,
            rose_bin_width_deg=req.rose_bin_width_deg,
            max_gap_cells=req.max_gap_cells,
        )
        result["source"] = req.source
        self.lineament_summary_cache = result
        return result

    def run_tilt_depth(self, req: TiltDepthRequest) -> dict:
        if self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        result = tilt_depth_estimates(
            grid.values, grid.easting, grid.northing, grid.cell_size_m, self.utm_epsg,
            min_depth_m=req.min_depth_m, max_depth_m=req.max_depth_m,
        )
        self.tilt_depth_cache = result
        return result

    def run_analytic_signal_depth(self, req: AnalyticSignalDepthRequest) -> dict:
        if self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        result = analytic_signal_depth_estimates(
            grid.values, grid.easting, grid.northing, grid.cell_size_m, self.utm_epsg,
            percentile_threshold=req.percentile_threshold, search_radius_cells=req.search_radius_cells,
        )
        self.as_depth_cache = result
        return result

    def run_spectral_depth(self, req: SpectralDepthRequest) -> dict:
        if self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        result = spectral_depth_diagnostic(grid.values, grid.cell_size_m)
        self.spectral_depth_cache = result
        return result

    def _near_surface_susceptibility_grid(self, easting: np.ndarray, northing: np.ndarray) -> np.ndarray | None:
        """Average susceptibility over the shallowest 2 layers of the most
        recent 3D inversion (index 0 = shallowest, see processing/
        inversion.py), resampled from the (coarse) inversion mesh onto the
        current display grid's easting/northing via linear interpolation -
        the same coarse-grid-and-interpolate pattern processing/igrf.py
        uses for IGRF, since the inversion mesh is far coarser than a
        typical display grid. Returns None (rather than raising) when no
        inversion has been run yet - susceptibility is an optional
        prospectivity layer, not a required one."""
        if self.inversion_result is None:
            return None
        mesh = self.inversion_result.mesh
        n_layers = min(2, mesh.z_centers.size)
        near_surface = np.nanmean(self.inversion_result.susceptibility[:, :, :n_layers], axis=2)
        if mesh.x_centers.size < 2 or mesh.y_centers.size < 2:
            return None
        interpolator = RegularGridInterpolator(
            (mesh.y_centers, mesh.x_centers), near_surface, bounds_error=False, fill_value=np.nan
        )
        x2d, y2d = np.meshgrid(easting, northing)
        return interpolator(np.column_stack([y2d.ravel(), x2d.ravel()])).reshape(x2d.shape)

    def run_prospectivity(self, req: ProspectivityRequest) -> dict:
        """Rule-based mineral prospectivity ("target score") mapping - see
        processing/prospectivity.py. Combines ASA/THD with (if already
        computed) structural-lineament and magnetic-contact proximity, and
        (if a 3D inversion has been run) near-surface susceptibility, into
        one weighted 0-1 score grid, then reports ranked, explained
        local-maximum targets."""
        if self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        asa_grid = analytic_signal(grid.values, grid.cell_size_m)
        thd_grid = total_horizontal_derivative(grid.values, grid.cell_size_m)
        finite_mask = np.isfinite(grid.values)

        lineament_points = None
        if self.lineament_summary_cache and self.lineament_summary_cache.get("available"):
            lineament_points = [p for l in self.lineament_summary_cache["lineaments"] for p in l["points_latlon"]]
        contact_points = None
        if self.contact_summary_cache and self.contact_summary_cache.get("available"):
            contact_points = [p for c in self.contact_summary_cache["contacts"] for p in c["points_latlon"]]
        susceptibility_grid = self._near_surface_susceptibility_grid(grid.easting, grid.northing)

        result = compute_prospectivity(
            asa_grid,
            thd_grid,
            finite_mask,
            grid.easting,
            grid.northing,
            self.utm_epsg,
            lineament_points_latlon=lineament_points,
            contact_points_latlon=contact_points,
            susceptibility_grid=susceptibility_grid,
            purpose=req.purpose,
            custom_weights=req.weights,
            decay_length_m=req.decay_length_m,
            score_threshold=req.score_threshold,
            max_targets=req.max_targets,
            min_target_separation_m=req.min_target_separation_m,
        )
        score_grid = result.pop("score_grid", None)
        if score_grid is not None:
            self.prospectivity_score_grid = score_grid
            self.prospectivity_score_easting = grid.easting
            self.prospectivity_score_northing = grid.northing
            self.prospectivity_score_cell_size_m = grid.cell_size_m
        self.prospectivity_summary_cache = result
        return result

    def get_prospectivity_overlay(self, colormap: str = "Viridis") -> dict:
        """PNG overlay of the most recent run_prospectivity() score grid -
        same rendering pattern as get_grid_confidence_overlay, since the
        0-1 score grid isn't a value the generic get_grid_overlay/
        get_transform_overlay cache keys already cover."""
        if self.prospectivity_score_grid is None:
            raise ProjectError("프로스펙티비티 분석을 먼저 실행하세요.")
        overlay = grid_to_png_overlay(
            self.prospectivity_score_grid,
            self.prospectivity_score_easting,
            self.prospectivity_score_northing,
            self.utm_epsg,
            cmap_name=colormap,
            symmetric=False,
            vmin=0.0,
            vmax=1.0,
            cell_size_m=self.prospectivity_score_cell_size_m,
        )
        overlay["stats"] = _stats(pd.Series(self.prospectivity_score_grid.ravel()))
        overlay["cell_size_m"] = self.prospectivity_score_cell_size_m
        return overlay

    def get_power_spectrum(self, req: PowerSpectrumRequest) -> dict:
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed
        line_df = df[df["line_id"] == req.line_id].sort_values("timestamp")
        if line_df.empty:
            raise ProjectError(f"측선 {req.line_id}을 찾을 수 없습니다.")
        return compute_power_spectrum(line_df[req.value].to_numpy(), line_df["timestamp"])

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

    def scan_structure_distortion(self, req: StructureScanRequest) -> dict:
        """Combine an OpenStreetMap building/road location prior with the
        same compact-anomaly signal detector used for near-surface target
        detection (run_target_detection) to auto-generate the regions the
        "지도에서 왜곡 영역 그려 스무딩" tool would otherwise need one
        hand-drawn polygon per structure for - see
        processing/osm_structures.py.

        Returns both the buffered structure polygons (regardless of
        whether a signal anomaly matched them - a subtle building without
        a peak reaching the detection threshold is still worth a visual
        check) and the compact anomalies the signal-only side found, each
        tagged with whether it falls inside a mapped structure's buffered
        region - lets the caller show "this spike lines up with a mapped
        building" separately from "no mapped structure here at all, worth
        a closer look (unmapped structure, or possibly real geology)"
        instead of blindly trusting either signal alone."""
        if self.processed is None or self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        if self.utm_epsg is None:
            raise ProjectError("좌표계 정보가 없습니다.")

        active = self._active_mask()
        df = self.processed.loc[active]
        if df.empty:
            raise ProjectError("유효한 측선 포인트가 없습니다.")

        lat_min, lat_max = float(df["lat"].min()), float(df["lat"].max())
        lon_min, lon_max = float(df["lon"].min()), float(df["lon"].max())
        margin_m = 50.0
        lat_margin = margin_m / 111_000.0
        lon_margin = margin_m / (111_000.0 * max(np.cos(np.radians((lat_min + lat_max) / 2)), 0.1))

        try:
            structures = fetch_osm_structures(
                lat_min - lat_margin, lon_min - lon_margin, lat_max + lat_margin, lon_max + lon_margin
            )
        except OsmFetchError as exc:
            raise ProjectError(str(exc)) from exc

        structure_polygons = buffer_structures_to_polygons(
            structures, self.utm_epsg, req.building_buffer_m, req.road_buffer_m
        )

        x_span = float(df["x"].max() - df["x"].min())
        y_span = float(df["y"].max() - df["y"].min())
        est_cells = (x_span / req.cell_size_m + 1) * (y_span / req.cell_size_m + 1)
        if est_cells > TARGET_DETECTION_GRID_CELL_CAP:
            raise ProjectError(
                f"탐지 격자가 너무 촘촘합니다 (예상 셀 수 약 {int(est_cells):,}개, 상한 {TARGET_DETECTION_GRID_CELL_CAP:,}개). "
                "탐지 격자 크기(m)를 늘리거나, 폴리곤으로 관심 영역만 남기고 나머지 측선을 제외한 뒤 다시 시도하세요."
            )

        # Along-line smoothing off, same reasoning as run_target_detection:
        # this hunts for compact, localized anomalies, which a low-pass
        # tuned to the (much larger) line spacing would blur away.
        grid = self._grid_for("anomaly", req.cell_size_m, req.method, req.max_distance_m, along_line_smooth=False)

        if req.amplitude_threshold_nt is not None:
            threshold_nt = req.amplitude_threshold_nt
        else:
            finite = grid.values[np.isfinite(grid.values)]
            robust_std = float(1.4826 * np.median(np.abs(finite - np.median(finite)))) if finite.size else 0.0
            threshold_nt = max(req.threshold_k * robust_std, 1e-6)

        candidates = detect_targets(
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

        structure_polys_local = []
        if structure_polygons:
            to_local = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
            for ring in structure_polygons:
                xs, ys = to_local.transform([p[1] for p in ring], [p[0] for p in ring])
                structure_polys_local.append(MplPath(list(zip(xs, ys))))

        transformer = Transformer.from_crs(f"EPSG:{self.utm_epsg}", "EPSG:4326", always_xy=True)
        anomaly_dicts = []
        if candidates:
            lons, lats = transformer.transform([c.x for c in candidates], [c.y for c in candidates])
        else:
            lons, lats = [], []
        for c, lat, lon in zip(candidates, lats, lons):
            matched = any(path.contains_point((c.x, c.y)) for path in structure_polys_local)
            anomaly_dicts.append(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "peak_anomaly_nt": c.peak_anomaly_nt,
                    "footprint_m": c.footprint_m,
                    "fit_quality": c.fit_quality,
                    "matched_structure": matched,
                }
            )
        # matched-and-confident first - the most trustworthy "this is
        # almost certainly cultural noise" candidates, read top-down.
        anomaly_dicts.sort(key=lambda d: (not d["matched_structure"], -d["fit_quality"]))

        return {
            "n_buildings": len(structures.buildings),
            "n_roads": len(structures.roads),
            "structure_polygons": structure_polygons,
            "n_structure_polygons": len(structure_polygons),
            "amplitude_threshold_nt": threshold_nt,
            "cell_size_m": grid.cell_size_m,
            "anomalies": anomaly_dicts,
            "n_anomalies": len(anomaly_dicts),
            "n_matched": sum(1 for a in anomaly_dicts if a["matched_structure"]),
        }

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

    def sample_overlay_value(self, lat: float, lon: float) -> dict:
        """Value of the nearest cell in the most recently displayed
        grid/derivative overlay to a clicked map point - matches exactly
        what the overlay's color at that pixel represents, unlike
        sample_point() which reports the nearest raw survey point.
        Powers the map's click-to-inspect tool.

        Deliberately nearest-cell, not a bilinear blend between
        neighboring cells: the PNG overlay (see
        processing/render.py::grid_to_png_overlay) renders one flat color
        per cell with hard edges (no smoothing baked into the image
        itself), so a bilinearly-interpolated read-out could return a
        value blended from a neighboring cell that visibly differs from
        (and doesn't actually match) the exact color the user clicked on -
        most noticeable right at a sharp anomaly edge, exactly where
        getting this right matters most. Nearest-cell guarantees the
        number always matches the pixel."""
        if self.last_overlay_values is None:
            raise ProjectError("먼저 그리드를 생성하세요.")
        if self.utm_epsg is None:
            raise ProjectError("좌표계 정보가 없습니다.")
        easting = self.last_overlay_easting
        northing = self.last_overlay_northing
        values = self.last_overlay_values
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
        x, y = transformer.transform(lon, lat)
        result = {"lat": lat, "lon": lon, "value_nt": None, "in_bounds": False, "label": self.last_overlay_label}
        # Half a cell of slack on each side matches the overlay image's own
        # rendered extent (see grid_to_png_overlay's "pixel is area" bounds)
        # - without it, a click inside the visually-drawn edge cell but
        # past the last cell *center* would wrongly report out-of-bounds.
        half_e = (easting[1] - easting[0]) / 2.0 if len(easting) > 1 else 0.0
        half_n = (northing[1] - northing[0]) / 2.0 if len(northing) > 1 else 0.0
        if x < easting[0] - half_e or x > easting[-1] + half_e or y < northing[0] - half_n or y > northing[-1] + half_n:
            return result
        col = int(_nearest_axis_index(easting, np.array([x]))[0])
        row = int(_nearest_axis_index(northing, np.array([y]))[0])
        value = float(values[row, col])
        if not np.isfinite(value):
            return result
        result["value_nt"] = value
        result["in_bounds"] = True
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
            "manual_smooth_point_ids": sorted(int(p) for p in self.manual_smooth_point_ids),
            "display_boundary_polygon": self.display_boundary_polygon,
            "dem_name": self.dem_name,
            "has_base": self.base_raw is not None,
        }

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("drone_raw.csv", self.drone_raw.to_csv(index=False))
            if self.base_raw is not None:
                zf.writestr("base_raw.csv", self.base_raw.to_csv(index=False))
            if self.calibration_raw is not None:
                zf.writestr("calibration_raw.csv", self.calibration_raw.to_csv(index=False))
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

                if "calibration_raw.csv" in names:
                    cal_df = pd.read_csv(io.BytesIO(zf.read("calibration_raw.csv")))
                    cal_df["timestamp"] = pd.to_datetime(cal_df["timestamp"])
                    self.calibration_raw = cal_df
                else:
                    self.calibration_raw = None

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
                    smooth_ids = meta.get("manual_smooth_point_ids") or []
                    if smooth_ids:
                        self.set_manual_smoothing(ManualSmoothRequest(mode="point_ids", point_ids=smooth_ids))
                    boundary_polygon = meta.get("display_boundary_polygon")
                    if boundary_polygon:
                        self.set_display_boundary(DisplayBoundaryRequest(polygon=boundary_polygon))
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
                assumed_noise_nt=params.assumed_noise_nt,
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
            "regularization_strength_used": result.regularization_strength_used,
            "auto_regularization": params.assumed_noise_nt is not None,
            "assumed_noise_nt": params.assumed_noise_nt,
            "layer_elevations_m": [float(z) for z in mesh.z_centers],
            "depth_resolution": result.depth_resolution,
            "depth_resolution_warning": _depth_resolution_warning(result.depth_resolution, mesh.z_centers),
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
            section, distance, ground_elev_path = _inversion_vertical_section(
                self.inversion_result, path_x, path_y, req.threshold, req.threshold_max
            )
            png = render_section_png(
                section, distance, self.inversion_result.mesh.z_centers, ground_elev_path,
                path_x, path_y, mesh.x_centers, mesh.y_centers, profile=req.profile,
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

    def get_inversion_box_faces(self, top_layer_index: int = 0) -> dict:
        """Fence-diagram style 3D view: top horizontal slice + the 4
        vertical boundary walls of the mesh, one continuous SI
        colorscale, no threshold gating - see processing/inversion.py:
        box_faces for the layout this mirrors."""
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        mesh = self.inversion_result.mesh
        x0 = float(mesh.x_centers.mean())
        y0 = float(mesh.y_centers.mean())
        faces = box_faces(self.inversion_result, top_layer_index)
        for key in ("top", "south", "north", "west", "east"):
            face = faces[key]
            face["x"] = (np.asarray(face["x"]) - x0).tolist()
            face["y"] = (np.asarray(face["y"]) - y0).tolist()
        return faces

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


def _depth_resolution_warning(depth_resolution: dict | None, z_centers: np.ndarray) -> str | None:
    """Human-readable caveat listing which depth layers the survey's own
    geometry barely constrains at all (see
    processing/inversion.py:resolution_diagnostics) - surfaced so a user
    doesn't read the deepest part of the recovered model with the same
    confidence as the well-resolved shallow part."""
    if not depth_resolution:
        return None
    poor = depth_resolution.get("poorly_resolved_layers") or []
    if not poor:
        return None
    elevations = [float(z_centers[i]) for i in poor]
    return (
        f"심도 레이어 {len(poor)}개(고도 {min(elevations):.0f}~{max(elevations):.0f}m, 대체로 더 깊은 쪽)는 "
        "이 측선 배치/고도로는 실제 민감도가 최상층 대비 5% 미만입니다 - 해당 구간의 역산 결과는 "
        "참고용으로만 보고, 얕은 구간보다 신뢰도를 낮게 두세요."
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


def _extrema_locations(values: np.ndarray, easting: np.ndarray, northing: np.ndarray, utm_epsg: int) -> dict:
    """lat/lon of the single cell holding the grid's max and min value -
    powers the "최댓값/최솟값 위치로 이동" map jump button. Exists because
    manually finding and clicking the exact extreme cell on a busy map is
    genuinely hard once cells are only a few screen pixels wide (adjacent
    cells can look nearly identical at typical zoom, especially with the
    "nearest" method's sharp cell-to-cell jumps) - jumping straight there
    sidesteps that precision problem entirely rather than just tolerating
    it."""
    finite = np.isfinite(values)
    if not finite.any():
        return {"max": None, "min": None}
    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

    def _location(idx_func):
        row, col = idx_func(np.where(finite, values, np.nan))
        lon, lat = transformer.transform(float(easting[col]), float(northing[row]))
        return {"lat": float(lat), "lon": float(lon), "value_nt": float(values[row, col])}

    return {
        "max": _location(lambda v: np.unravel_index(np.nanargmax(v), v.shape)),
        "min": _location(lambda v: np.unravel_index(np.nanargmin(v), v.shape)),
    }


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
