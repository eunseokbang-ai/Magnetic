from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LineParams(BaseModel):
    heading_lag_seconds: float = 1.0
    heading_tolerance_deg: float = Field(20.0, ge=1, le=90)
    min_speed_mps: float = Field(1.5, ge=0)
    min_line_length_m: float = Field(150.0, ge=0)
    turn_buffer_m: float = Field(15.0, ge=0)
    max_gap_seconds: float = Field(1.0, ge=0.1)
    # "heading_histogram" (default) or "pca" - see processing/lines.py.
    direction_method: Literal["heading_histogram", "pca"] = "heading_histogram"


class DiurnalParams(BaseModel):
    time_offset_seconds: float = 0.0
    reference: str = "mean"  # "mean" | "first" | numeric string


class HeadingCorrectionParams(BaseModel):
    enabled: bool = True
    quiet_percentile: float = Field(40.0, ge=1, le=100)
    max_match_distance_m: Optional[float] = None


class DespikeParams(BaseModel):
    enabled: bool = True
    window_size: int = Field(11, ge=3, le=101)  # samples
    threshold_k: float = Field(4.0, gt=0)  # robust-std multiples before a sample counts as a spike
    # Adaptive Hampel: widen the local median/MAD window on steep
    # along-track gradients so a genuine ramp isn't mistaken for a run of
    # spikes. See processing/despike.py:despike for details.
    adaptive: bool = True
    adaptive_gradient_threshold: float = Field(5.0, gt=0)  # nT/sample
    adaptive_expand_samples: int = Field(4, ge=0, le=50)


class CrossoverLevelingParams(BaseModel):
    # Off by default - tie lines aren't always flown, and this only does
    # anything useful when perpendicular calibration lines are present in
    # the raw flight (see processing/lines.py:detect_tie_lines).
    enabled: bool = False
    tie_tolerance_deg: float = Field(20.0, ge=1, le=90)
    max_crossover_distance_m: float = Field(15.0, gt=0)
    # Iterative network adjustment (survey <-> tie shifts refined together)
    # instead of treating the tie-line network as a perfect fixed
    # reference - see processing/crossover_leveling.py.
    iterative: bool = True


FilterMethod = Literal["butterworth", "savgol", "moving_average"]


class ProcessParams(BaseModel):
    filter_method: FilterMethod = "butterworth"
    filter_cutoff_hz: float = Field(1.0, gt=0)  # used when filter_method="butterworth"
    filter_window_seconds: float = Field(1.0, gt=0)  # used when filter_method="savgol" or "moving_average"
    filter_polyorder: int = Field(3, ge=1, le=7)  # used when filter_method="savgol"
    # A known constant delay (seconds) between the magnetometer and GPS
    # position streams - some loggers' internal filtering/telemetry path
    # lags the GPS fix by a fraction of a second, which shows up as a
    # small along-track position error. 0.0 = no correction (default).
    gps_mag_lag_seconds: float = 0.0
    # Target EPSG code for the local projected CRS used throughout
    # processing/export (grid, GeoTIFF, XYZ, ...). Left as None (the
    # default), the UTM zone is auto-detected from the data's centroid -
    # set this to override it, e.g. to match a national grid or to keep
    # results consistent with a survey area that straddles a UTM zone
    # boundary. Takes precedence over korea_projection when both are set.
    utm_epsg_override: Optional[int] = None
    # Convenience alternative to utm_epsg_override for Korean surveys:
    # "korea_utm" = EPSG:5179 (KGD2002 Unified CS), "korea2010" =
    # EPSG:5185-5188 (KGD2002 Belt 2010, picked by longitude band), "utm" =
    # Korea-domestic UTM 51N/52N. None (default) = generic auto UTM.
    korea_projection: Optional[Literal["korea_utm", "korea2010", "utm"]] = None
    despike_params: DespikeParams = DespikeParams()
    line_params: LineParams = LineParams()
    diurnal_params: DiurnalParams = DiurnalParams()
    heading_correction: HeadingCorrectionParams = HeadingCorrectionParams()
    crossover_leveling: CrossoverLevelingParams = CrossoverLevelingParams()


ValueField = Literal["tmi", "anomaly"]
TransformName = Literal[
    "rtp", "rte", "1vd", "2vd", "as", "thdr", "tilt", "theta",
    "dx", "dy", "dxx", "dyy", "dxy", "dxz", "dyz",
    "upward_continuation", "detrend", "microlevel",
]
GridMethod = Literal["nearest", "linear", "cubic", "spline", "minimum_curvature"]


class HillshadeParams(BaseModel):
    # Geosoft Oasis Montaj-style "color-shaded relief" - the grid values
    # are treated as a pseudo-terrain and illuminated, so subtle
    # gradients read as raised/shadowed texture instead of flat color
    # bands. Common defaults for cartographic hillshading (NW light,
    # 45 deg altitude) work well here too.
    hillshade: bool = False
    hillshade_azimuth_deg: float = Field(315.0, ge=0, le=360)
    hillshade_altitude_deg: float = Field(45.0, ge=1, le=90)
    hillshade_exaggeration: float = Field(3.0, gt=0, le=20)


StretchName = Literal["linear", "equalize"]


class ContourParams(BaseModel):
    show_contours: bool = False
    # if None, contour_n_levels evenly-spaced levels are picked from the
    # grid's own min/max range instead of a fixed nT interval.
    contour_interval_nt: Optional[float] = Field(None, gt=0)
    contour_n_levels: int = Field(10, ge=2, le=50)


class GridRequest(HillshadeParams, ContourParams):
    value: ValueField = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    stretch: StretchName = "linear"
    colored: bool = False  # GeoTIFF export only: bake in the on-screen colormap as an RGBA GeoTIFF instead of raw float values
    # Along-line low-pass applied before gridding, to prevent flight-line-parallel
    # "corrugation" ridges (along-line detail no cross-line interpolation can
    # resolve anyway, given lines are typically 5-50x farther apart than the
    # along-line sample spacing). Defaults to on, at a wavelength matched to the
    # estimated line spacing; see processing.gridding._along_line_lowpass.
    along_line_smooth: bool = True
    along_line_smooth_wavelength_m: Optional[float] = Field(None, gt=0)
    # See TransformRequest.microlevel_pre_apply - applies here too so the
    # plain "그리드(원본)" view can show the same pre-leveled surface that
    # derived transforms are computed from, instead of only being
    # reachable via the "마이크로레벨링" transform itself.
    microlevel_pre_apply: bool = False
    microlevel_strength: float = Field(0.8, ge=0, le=1)
    microlevel_angle_tolerance_deg: float = Field(15.0, gt=0, le=45)
    microlevel_wavelength_factor: float = Field(1.5, gt=1)


class TransformRequest(HillshadeParams, ContourParams):
    transform: TransformName
    value: ValueField = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    stretch: StretchName = "linear"
    colored: bool = False  # GeoTIFF export only: bake in the on-screen colormap as an RGBA GeoTIFF instead of raw float values
    # See GridRequest.along_line_smooth - equally relevant here since every
    # derived transform is computed from this same underlying grid, and
    # derivative-based ones (RTP/1VD/tilt/etc.) amplify corrugation the most.
    along_line_smooth: bool = True
    along_line_smooth_wavelength_m: Optional[float] = Field(None, gt=0)
    # transform="upward_continuation" only: how far to continue the field upward.
    continuation_height_m: Optional[float] = Field(None, gt=0)
    # transform="detrend" only: order of the polynomial regional surface removed (1=plane, 2=quadratic, 3=cubic).
    trend_order: int = Field(1, ge=1, le=3)
    # transform="microlevel" only: see processing/microlevel.py for what each controls.
    microlevel_strength: float = Field(0.8, ge=0, le=1)
    microlevel_angle_tolerance_deg: float = Field(15.0, gt=0, le=45)
    microlevel_wavelength_factor: float = Field(1.5, gt=1)
    # Applies microleveling (using the three params above) to the base grid
    # *before* computing whatever transform is requested (RTP/2VD/AS/etc.,
    # or even "none"), instead of microleveling only being its own
    # mutually-exclusive transform choice - lets line-parallel corrugation
    # get cleaned up before a derivative-based transform amplifies it,
    # rather than only after. No-op when transform="microlevel" itself.
    microlevel_pre_apply: bool = False


class PolygonExportRequest(BaseModel):
    polygon: list[list[float]] = Field(..., min_length=3)  # [[lat, lon], ...]


class ManualExcludeRequest(BaseModel):
    mode: Literal["lines", "polygon", "point_ids", "reset"]
    action: Optional[Literal["exclude", "include"]] = None
    line_ids: Optional[list[int]] = None
    polygon: Optional[list[list[float]]] = None  # [[lat, lon], ...]
    point_ids: Optional[list[int]] = None


class InversionParams(BaseModel):
    value: ValueField = "anomaly"
    # Any of these left as None (the default) is auto-estimated from the
    # survey's own line spacing/extent and anomaly spectrum - see
    # processing/inversion_auto.py.
    obs_cell_size_m: Optional[float] = Field(None, gt=0)
    depth_extent_m: Optional[float] = Field(None, gt=0)
    n_layers: Optional[int] = Field(None, ge=1, le=50)
    assumed_agl_m: float = Field(50.0, gt=0)  # used only when no DEM is uploaded
    regularization_strength: float = Field(1.0, gt=0)
    n_irls_iterations: int = Field(6, ge=1, le=30)


class InversionSliceRequest(HillshadeParams):
    layer_index: int = Field(0, ge=0)
    threshold: Optional[float] = None
    threshold_max: Optional[float] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None


SectionProfile = Literal["custom", "ew", "ns"]


class InversionSectionRequest(BaseModel):
    profile: SectionProfile = "custom"
    position_frac: Optional[float] = Field(None, ge=0, le=1)  # required for ew/ns
    path: Optional[list[list[float]]] = Field(None, min_length=2)  # [[lat, lon], ...], required for custom
    threshold: Optional[float] = None
    threshold_max: Optional[float] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    sample_spacing_m: float = Field(10.0, gt=0)


class EulerDeconvolutionRequest(BaseModel):
    value: ValueField = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    # 0 = contact/fault edge, 1 = thin dyke/sill/sheet edge, 2 = pipe/vertical
    # cylinder, 3 = sphere/point dipole - the four textbook structural indices.
    structural_index: float = Field(1.0, ge=0, le=3)
    window_size_m: float = Field(100.0, gt=0)
    max_depth_uncertainty_pct: float = Field(30.0, gt=0, le=200)


class TargetDetectionRequest(BaseModel):
    # Much finer default than the 10m used for regional geology grids -
    # near-surface compact targets (mines, ordnance, vehicles) need a
    # detection grid fine enough to resolve a footprint of a few meters.
    cell_size_m: float = Field(1.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    # If omitted, auto = threshold_k * robust std of the anomaly grid.
    amplitude_threshold_nt: Optional[float] = Field(None, gt=0)
    threshold_k: float = Field(4.0, gt=0)
    min_footprint_m: float = Field(0.5, gt=0)
    max_footprint_m: float = Field(15.0, gt=0)
    fit_window_m: float = Field(8.0, gt=0)
    max_depth_m: float = Field(5.0, gt=0)
    min_fit_quality: float = Field(0.3, ge=0, le=1)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class InversionVolumeRequest(BaseModel):
    threshold: Optional[float] = None
    threshold_max: Optional[float] = None
