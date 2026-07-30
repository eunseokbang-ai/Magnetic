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


class ProcessParams(BaseModel):
    filter_cutoff_hz: float = Field(1.0, gt=0)
    line_params: LineParams = LineParams()
    diurnal_params: DiurnalParams = DiurnalParams()
    heading_correction: HeadingCorrectionParams = HeadingCorrectionParams()


ValueField = Literal["tmi", "anomaly"]
TransformName = Literal["rtp", "rte", "1vd", "as"]
GridMethod = Literal["nearest", "linear", "cubic", "spline"]


class GridRequest(BaseModel):
    value: ValueField = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None


class TransformRequest(BaseModel):
    transform: TransformName
    value: ValueField = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "nearest"
    max_distance_m: Optional[float] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None


class ManualExcludeRequest(BaseModel):
    mode: Literal["lines", "polygon", "reset"]
    action: Optional[Literal["exclude", "include"]] = None
    line_ids: Optional[list[int]] = None
    polygon: Optional[list[list[float]]] = None  # [[lat, lon], ...]


class InversionParams(BaseModel):
    value: ValueField = "anomaly"
    obs_cell_size_m: float = Field(20.0, gt=0)  # observation/mesh horizontal cell size
    depth_extent_m: float = Field(100.0, gt=0)
    n_layers: int = Field(8, ge=1, le=30)
    assumed_agl_m: float = Field(50.0, gt=0)  # used only when no DEM is uploaded
    regularization_strength: float = Field(1.0, gt=0)
    n_irls_iterations: int = Field(5, ge=1, le=20)


class InversionSliceRequest(BaseModel):
    layer_index: int = Field(0, ge=0)
    threshold: Optional[float] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None


class InversionSectionRequest(BaseModel):
    path: list[list[float]] = Field(..., min_length=2)  # [[lat, lon], ...]
    threshold: Optional[float] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    sample_spacing_m: float = Field(10.0, gt=0)


class InversionVolumeRequest(BaseModel):
    threshold: Optional[float] = None
