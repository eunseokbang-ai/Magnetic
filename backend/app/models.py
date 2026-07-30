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
GridMethod = Literal["spline", "linear", "cubic"]


class GridRequest(BaseModel):
    value: ValueField = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "spline"
    max_distance_m: Optional[float] = None
    cmap: Optional[str] = None


class TransformRequest(BaseModel):
    transform: TransformName
    value: ValueField = "anomaly"
    cell_size_m: float = Field(10.0, gt=0)
    method: GridMethod = "spline"
    max_distance_m: Optional[float] = None
    cmap: Optional[str] = None


class ManualExcludeRequest(BaseModel):
    mode: Literal["lines", "polygon"]
    action: Literal["exclude", "include"]
    line_ids: Optional[list[int]] = None
    polygon: Optional[list[list[float]]] = None  # [[lat, lon], ...]
