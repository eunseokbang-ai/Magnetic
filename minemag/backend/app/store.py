"""In-memory projects for MineMag: the loaded probe readings and what is
derived from them. One process, one operator, so a dict keyed by project
id is enough - the same shape as DroneMag Studio's store."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from magcore.lines import project_to_local_xy
from magcore.render import grid_to_png_overlay

from .io_.magdrone_r3 import MagDroneLoadError, load_magdrone_r3_files, summarize
from .models import GridRequest, QcRequest, RepeatRequest, TargetRequest
from .processing.gridding import grid_survey, repeat_comparison
from .processing.platform_qc import platform_qc
from .processing.targets import pick_targets, targets_to_csv
from magcore.igrf import mean_inclination_declination


class ProjectError(ValueError):
    pass


@dataclass
class Project:
    id: str
    raw: pd.DataFrame | None = None       # long format, one row per probe per sample (see io_/magdrone_r3.py)
    utm_epsg: int | None = None
    summary_cache: dict | None = None
    qc_cache: dict = field(default_factory=dict)
    grid_cache: dict = field(default_factory=dict)
    repeat_cache: dict = field(default_factory=dict)
    target_cache: dict | None = None
    field_direction: tuple | None = None   # (inclination, declination) from IGRF at the site
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def _require(self) -> pd.DataFrame:
        if self.raw is None:
            raise ProjectError("자료를 먼저 올리세요.")
        return self.raw

    def load_files(self, buffers: list, filenames: list[str]) -> dict:
        with self._lock:
            try:
                df = load_magdrone_r3_files(buffers, filenames)
            except MagDroneLoadError as exc:
                raise ProjectError(str(exc)) from exc
            x, y, epsg = project_to_local_xy(df["lat"].to_numpy(), df["lon"].to_numpy())
            df["x"], df["y"] = x, y
            self.raw = df
            self.utm_epsg = int(epsg)
            self.summary_cache = None
            self.qc_cache, self.grid_cache, self.repeat_cache = {}, {}, {}
            self.target_cache = None
            self.field_direction = None
            return self.summary()

    def summary(self) -> dict:
        df = self._require()
        if self.summary_cache is None:
            self.summary_cache = {**summarize(df), "utm_epsg": self.utm_epsg}
        return self.summary_cache

    def points(self, sensor: int | None = None, max_points: int = 20000) -> dict:
        """Readings for the map, thinned to at most `max_points` rows per
        request. Lat/lon so the frontend needs no projection."""
        df = self._require()
        if sensor is not None:
            df = df[df["sensor"] == sensor]
        stride = max(1, int(np.ceil(len(df) / max_points)))
        sub = df.iloc[::stride]
        return {
            "n_total": int(len(df)),
            "stride": stride,
            "lat": np.round(sub["lat"].to_numpy(), 7).tolist(),
            "lon": np.round(sub["lon"].to_numpy(), 7).tolist(),
            "tfa": np.round(sub["tfa"].to_numpy(), 2).tolist(),
            "sensor": sub["sensor"].astype(int).tolist(),
            "track": sub["track"].astype(int).tolist(),
        }

    def qc(self, req: QcRequest) -> dict:
        df = self._require()
        key = (req.quiet_threshold_nt,)
        with self._lock:
            if key not in self.qc_cache:
                self.qc_cache[key] = platform_qc(df, quiet_threshold_nt=req.quiet_threshold_nt)
            return self.qc_cache[key]

    def grid(self, req: GridRequest) -> dict:
        """The grid as a map overlay plus its statistics."""
        df = self._require()
        key = (req.cell_size_m, tuple(req.sensors or ()), req.file_index, req.max_distance_m, req.level, req.cmap)
        with self._lock:
            if key not in self.grid_cache:
                try:
                    grid, stats = grid_survey(
                        df, cell_size_m=req.cell_size_m, sensors=req.sensors, file_index=req.file_index,
                        max_distance_m=req.max_distance_m, level=req.level,
                    )
                except ValueError as exc:
                    raise ProjectError(str(exc)) from exc
                finite = grid.values[np.isfinite(grid.values)]
                vmax = float(np.percentile(np.abs(finite), 99.5)) if len(finite) else 1.0
                vmax = max(vmax, 1.0)
                overlay = grid_to_png_overlay(
                    grid.values, grid.easting, grid.northing, self.utm_epsg,
                    cmap_name=req.cmap, symmetric=True, vmin=-vmax, vmax=vmax, cell_size_m=req.cell_size_m,
                )
                self.grid_cache[key] = {"stats": stats, "overlay": overlay}
            return self.grid_cache[key]

    def repeat(self, req: RepeatRequest) -> dict:
        df = self._require()
        key = (req.compare_cell_m, tuple(req.sensors or ()), req.min_samples)
        with self._lock:
            if key not in self.repeat_cache:
                self.repeat_cache[key] = repeat_comparison(
                    df, compare_cell_m=req.compare_cell_m, sensors=req.sensors, min_samples=req.min_samples
                )
            return self.repeat_cache[key]


    def _igrf_direction(self) -> tuple[float, float]:
        """Inclination and declination of the ambient field at the site,
        from IGRF at the readings' mean position and date. The GPS
        altitude is used as is: a few tens of metres of datum error
        change the direction by far less than a tenth of a degree."""
        if self.field_direction is None:
            df = self._require()
            alt = df["alt"].to_numpy(dtype=float)
            alt = np.where(np.isfinite(alt), alt, 50.0)
            inc, dec = mean_inclination_declination(
                df["lat"].to_numpy(), df["lon"].to_numpy(), alt, df["timestamp"]
            )
            self.field_direction = (float(inc), float(dec))
        return self.field_direction

    def targets(self, req: TargetRequest) -> dict:
        """Pick, grade and cache the targets; the CSV export reads the cache."""
        df = self._require()
        inc, dec = self._igrf_direction()
        qc = self.qc(QcRequest())
        noise_by_file = {}
        for f in qc["files"]:
            probes = [p["quiet_rms_nt"] for p in (f.get("per_probe") or {}).values() if p.get("quiet_rms_nt")]
            if probes:
                noise_by_file[int(f["file_index"])] = float(max(probes))
        with self._lock:
            try:
                result = pick_targets(
                    df, inc, dec, self.utm_epsg, noise_by_file,
                    sensor_height_m=req.sensor_height_m,
                    sensor_heights_by_file={int(k): v for k, v in (req.sensor_heights_by_file or {}).items()},
                    match_radius_m=req.match_radius_m,
                    cell_size_m=req.cell_size_m, threshold_k=req.threshold_k, threshold_nt=req.threshold_nt,
                    fit_window_m=req.fit_window_m, max_depth_m=req.max_depth_m, min_fit_quality=req.min_fit_quality,
                )
            except ValueError as exc:
                raise ProjectError(str(exc)) from exc
            result["inclination_deg"] = round(inc, 2)
            result["declination_deg"] = round(dec, 2)
            result["utm_epsg"] = self.utm_epsg
            self.target_cache = result
            return result

    def targets_csv(self) -> str:
        if self.target_cache is None:
            raise ProjectError("표적 피킹을 먼저 실행하세요.")
        return targets_to_csv(self.target_cache, self.utm_epsg)


class ProjectStore:
    def __init__(self) -> None:
        self._projects: dict[str, Project] = {}
        self._lock = threading.Lock()

    def create(self) -> Project:
        pid = uuid.uuid4().hex[:12]
        project = Project(id=pid)
        with self._lock:
            self._projects[pid] = project
        return project

    def get(self, project_id: str) -> Project:
        project = self._projects.get(project_id)
        if project is None:
            raise KeyError(project_id)
        return project


store = ProjectStore()
