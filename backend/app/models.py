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


class CrossoverLevelingParams(BaseModel):
    # Off by default - tie lines aren't always flown, and this only does
    # anything useful when perpendicular calibration lines are present in
    # the raw flight (see processing/lines.py:detect_tie_lines).
    enabled: bool = False
    tie_tolerance_deg: float = Field(20.0, ge=1, le=90)
    max_crossover_distance_m: float = Field(15.0, gt=0)


class ProcessParams(BaseModel):
    filter_cutoff_hz: float = Field(1.0, gt=0)
    # A known constant delay (seconds) between the magnetometer and GPS
    # position streams - some loggers' internal filtering/telemetry path
    # lags the GPS fix by a fraction of a second, which shows up as a
    # small along-track position error. 0.0 = no correction (default).
    gps_mag_lag_seconds: float = 0.0
    despike_params: DespikeParams = DespikeParams()
    line_params: LineParams = LineParams()
    diurnal_params: DiurnalParams = DiurnalParams()
    heading_correction: HeadingCorrectionParams = HeadingCorrectionParams()
    crossover_leveling: CrossoverLevelingParams = CrossoverLevelingParams()


ValueField = Literal["tmi", "anomaly"]
TransformName = Literal["rtp", "rte", "1vd", "as"]
GridMethod = Literal["nearest", "linear", "cubic", "spline"]


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


class ManualExcludeRequest(BaseModel):
    mode: Literal["lines", "polygon", "reset"]
    action: Optional[Literal["exclude", "include"]] = None
    line_ids: Optional[list[int]] = None
    polygon: Optional[list[list[float]]] = None  # [[lat, lon], ...]


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


class InversionVolumeRequest(BaseModel):
    threshold: Optional[float] = None
    threshold_max: Optional[float] = None
