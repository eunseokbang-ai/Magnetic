from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class GridRequest(BaseModel):
    """A grid of the total-field anomaly for the map - see
    processing/gridding.py for the defaults."""
    cell_size_m: float = Field(0.1, gt=0.01, le=5.0)
    # None = both probes.
    sensors: Optional[list[int]] = None
    # None = every file together.
    file_index: Optional[int] = None
    max_distance_m: float = Field(0.6, gt=0.05, le=20.0)
    level: Literal["median", "none"] = "median"
    cmap: str = "RdBu_r"


class RepeatRequest(BaseModel):
    """Run-to-run comparison on block means - see processing/gridding.py."""
    compare_cell_m: float = Field(0.25, gt=0.02, le=5.0)
    sensors: Optional[list[int]] = None
    min_samples: int = Field(3, ge=1, le=1000)


class QcRequest(BaseModel):
    quiet_threshold_nt: float = Field(60.0, gt=1.0, le=5000.0)


class TargetRequest(BaseModel):
    """Pick and grade compact targets - see processing/targets.py."""
    # Probe height above the ground, to turn depth below the probe into burial depth.
    sensor_height_m: float = Field(0.5, ge=0.0, le=10.0)
    # Per-file override, keyed by file index as a string ("0": 0.5, "1": 0.3).
    sensor_heights_by_file: Optional[dict[str, float]] = None
    cell_size_m: float = Field(0.1, gt=0.01, le=5.0)
    # Detection threshold = threshold_k x the run's quiet-zone noise rms (platform QC), unless threshold_nt is given.
    threshold_k: float = Field(5.0, gt=0.5, le=50.0)
    threshold_nt: Optional[float] = Field(None, gt=0)
    fit_window_m: float = Field(1.5, gt=0.2, le=20.0)
    max_depth_m: float = Field(2.0, gt=0.1, le=20.0)
    min_fit_quality: float = Field(0.3, ge=0.0, le=1.0)
    # Picks of different runs closer than this are the same target.
    match_radius_m: float = Field(0.5, gt=0.05, le=5.0)
