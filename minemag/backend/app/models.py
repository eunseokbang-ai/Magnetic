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
