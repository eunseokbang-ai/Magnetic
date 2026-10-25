"""In-memory projects for MineMag: the loaded probe readings and what is
derived from them. One process, one operator, so a dict keyed by project
id is enough - the same shape as DroneMag Studio's store, kept small
until the detection pipeline gives it something to hold."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from magcore.lines import project_to_local_xy

from .io_.magdrone_r3 import MagDroneLoadError, load_magdrone_r3_files, summarize


class ProjectError(ValueError):
    pass


@dataclass
class Project:
    id: str
    raw: pd.DataFrame | None = None       # long format, one row per probe per sample (see io_/magdrone_r3.py)
    utm_epsg: int | None = None
    summary_cache: dict | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

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
            return self.summary()

    def summary(self) -> dict:
        if self.raw is None:
            raise ProjectError("자료를 먼저 올리세요.")
        if self.summary_cache is None:
            self.summary_cache = {**summarize(self.raw), "utm_epsg": self.utm_epsg}
        return self.summary_cache

    def points(self, sensor: int | None = None, max_points: int = 20000) -> dict:
        """Readings for the map, thinned to at most `max_points` rows per
        request. Lat/lon so the frontend needs no projection."""
        if self.raw is None:
            raise ProjectError("자료를 먼저 올리세요.")
        df = self.raw if sensor is None else self.raw[self.raw["sensor"] == sensor]
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
