"""Keep the work alive across a restart.

A project lives in server memory: the parsed readings, the processing
params, the manual edits, the fitted structure models. Closing the app -
or a crash, or a Windows update at lunchtime - used to take all of it,
and the only defence was remembering to press 저장 often enough. On a
survey where every structure removal is a decision the operator made by
eye, that is real work to lose.

This writes the same bundle the 저장 button writes (Project
.save_project_bundle), on its own, to the user's data directory. Three
things make it safe to run unattended:

- **It never blocks the request that changed something.** A background
  thread notices the change and writes a few seconds later, so a click
  that edits fifty points does not wait on a multi-megabyte zip.
- **It coalesces.** A project is written at most once per interval no
  matter how many edits arrive, and only after the edits stop for a
  moment, so a burst of smoothing clicks costs one write.
- **It replaces atomically.** The bundle goes to a temporary file in the
  same directory and is renamed over the previous one, so a save
  interrupted half-way leaves the older, complete bundle intact rather
  than a truncated zip that cannot be opened.

Restoring is deliberately not automatic when the app is opened afresh.
Re-running the pipeline on a restored project takes seconds to minutes,
and the user may well have opened the app to do something else entirely,
so the saved projects are offered as a list and restored on request. The
one exception is an address that names a project (?project=<id>, which
the screen keeps in the address bar): opening that is itself the
request, so main.py's /open endpoint answers it from memory if the
project is still loaded and from the bundle here if the server was
restarted in between.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .paths import data_dir

# How often a single project may be written, and how long the edits have
# to stop before it is. The quiet period is what turns a drag of the
# smoothing brush into one save instead of thirty.
DEFAULT_INTERVAL_SECONDS = 120.0
DEFAULT_QUIET_SECONDS = 8.0
# The interval exists because writing a bundle holds the GIL and the
# screen stutters while it does; 120 s was set when a bundle took 13.9 s
# (CSV frames, HaeNam 965k rows). It now takes 2.2 s there and a fraction
# of a second on an ordinary survey, so a fixed 120 s leaves two minutes
# of edits exposed to a crash for no reason. The wait between writes is
# therefore tied to what a write costs: SAVE_COST_FACTOR times the last
# write's duration (the write then takes ~1/30 of the time, 3 %), never
# under MIN_INTERVAL_SECONDS and never over the configured interval.
# HaeNam: 30 x 2.2 s = 66 s. Andong (266,672 rows, 12.2 MB bundle): 0.36 s
# measured, 30 x 0.36 = 11 s, so the 15 s minimum applies.
SAVE_COST_FACTOR = 30.0
MIN_INTERVAL_SECONDS = 15.0
# Saved projects kept. Older ones are removed oldest-first: this is a
# safety net for the session in progress, not a project archive - the
# 저장 button is still how a project is kept deliberately.
MAX_SAVED = 5

_SUFFIX = ".magproj"


@dataclass
class SavedProject:
    project_id: str
    saved_at: float
    meta: dict
    path: Path

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "saved_at": self.saved_at,
            "size_bytes": self.path.stat().st_size if self.path.exists() else 0,
            **self.meta,
        }


def autosave_dir() -> Path:
    directory = Path(os.environ.get("MAGNETIC_AUTOSAVE_DIR") or (data_dir() / "autosave"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class AutosaveManager:
    """Watches a ProjectStore and keeps each project's bundle on disk."""

    def __init__(
        self,
        store,
        directory: Path | None = None,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        quiet_seconds: float = DEFAULT_QUIET_SECONDS,
        min_interval_seconds: float = MIN_INTERVAL_SECONDS,
    ) -> None:
        self._store = store
        self._directory = directory
        self._interval = interval_seconds
        self._quiet = quiet_seconds
        self._min_interval = min_interval_seconds
        self._last_saved: dict[str, float] = {}
        self._last_cost: dict[str, float] = {}       # seconds the last write of each project took
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------ paths
    @property
    def directory(self) -> Path:
        return self._directory if self._directory is not None else autosave_dir()

    def _bundle_path(self, project_id: str) -> Path:
        return self.directory / f"{project_id}{_SUFFIX}"

    def _meta_path(self, project_id: str) -> Path:
        return self.directory / f"{project_id}.json"

    # ------------------------------------------------------------ writing
    def save_project(self, project, now: float | None = None) -> Path | None:
        """Write one project's bundle now. Returns the path, or None when
        there is nothing worth saving (no readings loaded yet).

        Never raises: an autosave that can break the request it rides
        along with is worse than one that occasionally misses a cycle.
        """
        if getattr(project, "drone_raw", None) is None:
            return None
        started = time.perf_counter()
        try:
            data = project.save_project_bundle()
        except Exception:
            return None
        directory = self.directory
        target = self._bundle_path(project.id)
        tmp = directory / f".{project.id}.tmp"
        try:
            tmp.write_bytes(data)
            os.replace(tmp, target)
            self._meta_path(project.id).write_text(
                json.dumps({"saved_at": time.time(), **self._describe(project)}),
                encoding="utf-8",
            )
        except Exception:
            tmp.unlink(missing_ok=True)
            return None
        with self._lock:
            self._last_saved[project.id] = time.time() if now is None else now
            self._last_cost[project.id] = time.perf_counter() - started
        self._prune()
        return target

    def interval_for(self, project_id: str) -> float:
        """How long to leave between two writes of this project - see
        SAVE_COST_FACTOR. Until a write has been timed, the minimum: that
        is the case right after a restore or a reopen from disk, where
        waiting the full interval left the first two minutes of edits
        unsaved (measured in the browser: 115 s to the first write). One
        early write of a large project is the price of knowing its cost."""
        with self._lock:
            cost = self._last_cost.get(project_id)
        if cost is None:
            return min(self._interval, self._min_interval)
        return min(self._interval, max(self._min_interval, SAVE_COST_FACTOR * cost))

    def has_saved(self, project_id: str) -> bool:
        return self._bundle_path(project_id).exists()

    @staticmethod
    def _describe(project) -> dict:
        """What the restore list shows, so the user can tell one saved
        project from another without loading it."""
        info: dict = {"n_points": 0, "processed": project.processed is not None}
        df = project.drone_raw
        try:
            info["n_points"] = int(len(df))
            if "timestamp" in df.columns and len(df):
                info["time_range"] = [str(df["timestamp"].min()), str(df["timestamp"].max())]
            if "lat" in df.columns and len(df):
                info["centroid"] = [float(df["lat"].mean()), float(df["lon"].mean())]
        except Exception:
            pass
        info["n_source_removals"] = len(getattr(project, "source_removals", []) or [])
        info["n_smoothed_points"] = len(getattr(project, "manual_smooth_point_ids", []) or [])
        return info

    def _prune(self) -> None:
        saved = self.list_saved()
        for old in saved[MAX_SAVED:]:
            self.delete(old.project_id)

    # ------------------------------------------------------------ reading
    def list_saved(self) -> list[SavedProject]:
        """Newest first. A bundle whose metadata is missing or unreadable
        is still listed - the bundle is the part that matters."""
        out: list[SavedProject] = []
        for path in self.directory.glob(f"*{_SUFFIX}"):
            project_id = path.stem
            meta: dict = {}
            meta_path = self._meta_path(project_id)
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            saved_at = float(meta.pop("saved_at", 0.0) or path.stat().st_mtime)
            out.append(SavedProject(project_id, saved_at, meta, path))
        out.sort(key=lambda s: s.saved_at, reverse=True)
        return out

    def restore(self, project_id: str):
        """Load a saved bundle back into the store under its own id, so
        links and ids from before the restart keep working. Returns
        (project, summary)."""
        path = self._bundle_path(project_id)
        if not path.exists():
            raise FileNotFoundError(project_id)
        project = self._store.get_or_create(project_id)
        summary = project.load_project_bundle(path.read_bytes())
        with self._lock:
            self._last_saved[project_id] = time.time()
        return project, summary

    def delete(self, project_id: str) -> None:
        self._bundle_path(project_id).unlink(missing_ok=True)
        self._meta_path(project_id).unlink(missing_ok=True)
        with self._lock:
            self._last_saved.pop(project_id, None)
            self._last_cost.pop(project_id, None)

    # ------------------------------------------------------------ thread
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="autosave", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(1.0):
            try:
                self.save_due()
            except Exception:
                pass

    def save_due(self, now: float | None = None) -> list[str]:
        """Save every project whose edits have settled and whose interval
        has elapsed. Returns the ids written - the test hook, and what the
        loop above calls once a second."""
        now = time.time() if now is None else now
        written: list[str] = []
        for project in self._store.all_projects():
            changed_at = getattr(project, "changed_at", 0.0) or 0.0
            if changed_at <= 0.0:
                continue
            with self._lock:
                last = self._last_saved.get(project.id, 0.0)
            if changed_at <= last:
                continue                       # nothing new since the last write
            if now - changed_at < self._quiet:
                continue                       # still being edited
            if now - last < self.interval_for(project.id):
                continue                       # written recently enough
            if self.save_project(project, now=now) is not None:
                written.append(project.id)
        return written
