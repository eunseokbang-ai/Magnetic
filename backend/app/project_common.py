"""Shared pieces of the project model: the error type, cache limits and
small helpers used by store.Project and its mixin modules
(project_observatory, project_grids, project_interpretation,
project_inversion, project_persistence). Split out of store.py in
2026-11 so those modules can import them without importing store itself
(store imports the mixins, so the other direction would be circular).
"""
from __future__ import annotations

import functools
import numpy as np
import pandas as pd
from dataclasses import dataclass
from magcore import progress
from magcore.lines import LineDetectionParams, project_to_local_xy
from magcore.leveling import HeadingLevelingResult
from magcore.statistical_leveling import StatisticalLevelingResult
from .models import HeadingCorrectionParams


# eq=False: the generated __eq__ would compare the numpy array fields with
# ==, producing an elementwise array rather than a bool and raising
# "truth value of an array is ambiguous". Nothing compares or hashes these
# today, so this is purely a guard against a future caller doing so and
# hitting a confusing error far from the cause. (frozen=True with
# eq=False also leaves __hash__ as the default identity hash, which is the
# right semantics here anyway - two snapshots are the same only if they
# are literally the same object.)
@dataclass(frozen=True, eq=False)
class OverlayState:
    """Snapshot of the most recently rendered grid/derivative overlay,
    used by sample_overlay_value's click-to-inspect tool. Bundled into
    one immutable object (rather than four separate Project attributes)
    so a request handler publishes it with a single attribute
    assignment - CPython's GIL makes that single assignment atomic,
    unlike four separate assignments, which a concurrent request handler
    (FastAPI's sync endpoints run in a threadpool, so real thread
    interleaving is possible) could observe half-updated: e.g. new
    values paired with the previous request's easting/northing axes,
    silently reporting a value from the wrong location."""
    values: np.ndarray
    easting: np.ndarray
    northing: np.ndarray
    label: str


def azimuth_from_north(azimuth_deg: float | None) -> float | None:
    """The survey's line direction as a compass azimuth (clockwise from
    north, 0-180).

    Project.dominant_azimuth_deg is atan2(dy, dx): measured from east,
    counter-clockwise - the convention of lines.py, leveling,
    microlevelling and the grid diagnostics. Two modules describe the
    flight direction the other way, from north: line_resolution.py and
    repeat_passes.py. Passing them the project's number unconverted makes
    a north-south survey look east-west to them, and both then quietly
    find nothing: the line-resolution diagnostic reported "진단 실패", and
    the repeat-pass comparison found no repeated ground on any survey
    flown north-south or east-west (found 2026-11, when the
    reproducibility map was first tested through the API; the modules'
    own tests call them with the right convention and passed all along).
    Everything that calls either module goes through here."""
    return None if azimuth_deg is None else (90.0 - float(azimuth_deg)) % 180.0


class ProjectError(ValueError):
    pass


def boundary_rings(polygon) -> list[list[list[float]]]:
    """Normalize a display boundary to a list of [[lat, lon], ...] rings.

    The stored value is polymorphic for backward compatibility: a boundary
    the user drew (or any project saved before multi-part support) is a
    single flat ring [[lat, lon], ...], while an auto-generated boundary
    over a survey that splits into separate blocks is a list of such
    rings. The two are told apart unambiguously by looking one level in -
    a single ring's first element is a [lat, lon] pair of numbers, a
    multi-ring's first element is itself a ring. Every consumer
    (masking, export, save, the API response) goes through this so
    neither shape has to be special-cased anywhere else."""
    if not polygon:
        return []
    first = polygon[0]
    if first and isinstance(first[0], (list, tuple)):
        return [list(ring) for ring in polygon]
    return [list(polygon)]


_GRID_CACHE_MAX_ENTRIES = 4


_TRANSFORM_CACHE_MAX_ENTRIES = 8


# Where each pipeline stage starts on the progress bar: the share of the
# run's wall time that had passed when it began. Measured on the Andong
# 2026-10-28 survey (266,672 samples, 41 lines, 1.58 s in all; two runs
# agreed to 0.01): 0.00, 0.15, 0.30, 0.33, 0.49, 0.71, 0.77, 0.86. A
# different survey shifts them a little, which only makes the bar move
# at a slightly uneven speed.
_PIPELINE_PROGRESS = {"P0": 0.01, "P1": 0.15, "P2": 0.30, "P3": 0.33, "P4": 0.49, "P5": 0.71, "P6": 0.77, "P7": 0.86}


def _as_job(name: str, rollback: bool = False):
    """Run a Project method as a visible, cancellable job (magcore.progress).

    rollback=True is for methods that change the project as they go: if
    the user cancels halfway, every attribute is put back to what it was
    when the method started, so a cancelled run leaves the previous
    result on screen rather than half of a new one. It is a shallow
    restore - these methods replace their results (new frames, new
    dicts) instead of editing the old ones in place, which is what makes
    that sufficient. Such a method must already hold the edit lock when
    it is entered (decorate the *_unlocked variant), or an edit made
    between the snapshot and the lock would be undone with it."""

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            snapshot = dict(self.__dict__) if rollback else None
            try:
                with progress.job(name, self.id):
                    return fn(self, *args, **kwargs)
            except progress.JobCancelled:
                if snapshot is not None:
                    self.__dict__.clear()
                    self.__dict__.update(snapshot)
                raise

        return wrapper

    return decorate


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


def _detection_params(line_params) -> LineDetectionParams:
    """LineDetectionParams from the request model, minus the settings that
    belong to the step *after* detection (overlap resolution) - they travel
    together in LineParams for the user's sake, not because detection wants
    them."""
    dump = dict(line_params.model_dump())
    for key in ("overlap_resolution", "overlap_overrides"):
        dump.pop(key, None)
    return LineDetectionParams(**dump)


# What kind of stretch a point is in, for point_run_ids: a line, a ferry
# leg, a ramp, or the rest (turns and the pieces trimmed around them).
_RUN_KINDS = {"transit": 1, "takeoff_ramp": 2, "landing_ramp": 3}


def point_run_ids(df: pd.DataFrame) -> np.ndarray:
    """A number per point for the stretch of flight it belongs to: one
    pass along a line, one ferry leg, one turn, one ramp.

    This is what the map selects when a point is clicked. A line id will
    not do for that - a line resumed in a second flight is one id and two
    passes, and nothing that was excluded has an id at all - so a stretch
    is defined by what is next to what in the record: consecutive points
    of the same flight with the same line id and the same kind of
    exclusion."""
    line = df["line_id"].to_numpy()
    kind = np.where(line >= 0, 0, df["exclusion_reason"].map(_RUN_KINDS).fillna(4).to_numpy()).astype(int)
    source = df["source_file_index"].to_numpy() if "source_file_index" in df.columns else np.zeros(len(df), dtype=int)
    if len(df) == 0:
        return np.zeros(0, dtype=int)
    change = np.r_[True, (line[1:] != line[:-1]) | (kind[1:] != kind[:-1]) | (source[1:] != source[:-1])]
    return np.cumsum(change) - 1


def transit_summary(runs: list[dict] | None, df: pd.DataFrame, enabled: bool, source_files: list | None = None) -> dict:
    """The ferry legs detect_lines took out (and the runs it kept but was
    not sure of), as the line editor lists them: each with the stretch of
    the map it is (run_id, see point_run_ids) and where to look."""
    runs = runs or []
    if not runs:
        return {"enabled": enabled, "n_removed": 0, "n_points_removed": 0, "length_removed_m": 0.0, "n_suspect": 0, "runs": []}
    run_id = point_run_ids(df)
    row_of = pd.Series(np.arange(len(df)), index=df["point_id"].to_numpy())
    files = source_files or []
    out = []
    for run in runs:
        first, last = int(row_of.get(run["first_point_id"], -1)), int(row_of.get(run["last_point_id"], -1))
        if first < 0 or last < 0:
            continue
        mid = (first + last) // 2
        src = run["source_file_index"]
        out.append({
            **{k: run[k] for k in ("verdict", "why", "where", "source_file_index", "n_points", "length_m", "speed_mps",
                                   "line_speed_mps", "azimuth_deviation_deg", "line_id")},
            "source_file": files[src] if 0 <= src < len(files) else None,
            "run_id": int(run_id[mid]),
            "lat": float(df["lat"].iloc[mid]),
            "lon": float(df["lon"].iloc[mid]),
        })
    removed = [r for r in out if r["verdict"] == "transit"]
    return {
        "enabled": enabled,
        "n_removed": len(removed),
        "n_points_removed": int(sum(r["n_points"] for r in removed)),
        "length_removed_m": round(float(sum(r["length_m"] for r in removed)), 1),
        "n_suspect": len(out) - len(removed),
        "runs": out,
    }


def _overlap_overrides_for(df, per_line: dict, forced: dict) -> dict:
    """Merge the global keep_first/keep_second choice with any per-line
    overrides, the per-line ones winning."""
    if "__all__" in forced:
        out = {int(l): forced["__all__"] for l in df["line_id"].unique() if l >= 0}
    else:
        out = {}
    for k, v in (per_line or {}).items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def _line_summaries(
    df: pd.DataFrame,
    heading_leveling: "HeadingLevelingResult | None" = None,
    heading_applied: bool = False,
) -> list[dict]:
    # The A/B direction group is worth showing whenever it was worked out,
    # but the per-line shift only when it was really subtracted - the
    # offset is now measured even with the correction off, and listing a
    # shift the data never received would misreport what is on screen.
    groups = heading_leveling.line_groups if heading_leveling else {}
    shifts = heading_leveling.line_shifts if (heading_leveling and heading_applied) else {}
    # A line the drone flew in one go is one continuous run of samples. One
    # that was interrupted - a pause, or a flight stopped and resumed,
    # possibly as a separate file - is now still a single line (see
    # processing/lines.py::merge_continued_lines) but arrives as several
    # runs with a break between them. Those breaks are exactly where a line
    # needs looking at by hand, so they are reported rather than left for
    # the user to find by scrolling the profile.
    _JOIN_GAP_SECONDS = 5.0

    lines = []
    for line_id, g in df[df["line_id"] >= 0].groupby("line_id"):
        g = g.sort_values("timestamp")
        dt = g["timestamp"].diff().dt.total_seconds().to_numpy()
        break_at = np.flatnonzero(dt > _JOIN_GAP_SECONDS)
        joins = []
        for pos in break_at:
            before = g.iloc[pos - 1]
            after = g.iloc[pos]
            joins.append({
                "gap_seconds": float(dt[pos]),
                "gap_m": float(np.hypot(after["x"] - before["x"], after["y"] - before["y"])),
                "before_time": before["timestamp"].isoformat(),
                "after_time": after["timestamp"].isoformat(),
                "lat": float((before["lat"] + after["lat"]) / 2.0),
                "lon": float((before["lon"] + after["lon"]) / 2.0),
            })
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
                # 1 for a line flown in one go; more when it was resumed.
                "n_segments": int(len(joins) + 1),
                "joins": joins,
            }
        )
    return lines


def _heading_correction_should_apply(hp: "HeadingCorrectionParams", result: "HeadingLevelingResult") -> bool:
    """Only actually remove the offset when the user turned the correction
    on AND the estimate passed its own reliability check (empty
    `warnings`). `result.applied` alone is not enough to gate on - it only
    means "a number could be computed", not "the number is a real
    direction-dependent effect". A self-diagnosed unreliable estimate
    (`offset_spread_nt` >= `offset_nt`, see processing/leveling.py) is
    dominated by per-line noise rather than a heading effect, so applying
    it anyway shifts each line by noise instead of removing anything real.

    Measured on the 2026-09 HaeNam M400 block: with the gate missing,
    turning heading_correction on visibly added wrinkles even though
    compute_heading_correction correctly flagged the estimate unreliable
    (offset 1.26 nT < spread 3.03 nT) - the aggregate stripe_ratio metric
    barely moved (0.95% either way), which is what made this look safe
    before a user noticed the actual grid looked worse. This gate is what
    was missing to make the "self-declining" design described in
    HeadingCorrectionParams' docstring actually hold end to end - compare
    statistical_leveling, which is always-on by design because its
    per-line corrections already decline individually rather than as one
    global step that can only be all-or-nothing.
    """
    return bool(hp.enabled and result.applied and not result.warnings)


def _heading_correction_summary(result: "HeadingLevelingResult | None", applied: bool = False) -> dict:
    """`measured` = the forward/reverse offset could be estimated;
    `applied` = it was actually removed from the data. They differ
    whenever the user leaves the correction off, which is the default -
    the offset is still reported so the size of any direction-dependent
    step is visible without having to enable anything."""
    if result is None:
        return {"measured": False, "applied": False, "reason": None}
    return {
        "measured": result.applied,
        "applied": applied,
        "reason": result.reason,
        "offset_nt": result.offset_nt,
        "n_matched_pairs": result.n_matched_pairs,
        "n_quiet_pairs": result.n_quiet_pairs,
        "method": result.method,
        "offset_spread_nt": result.offset_spread_nt,
        "residual_nt": result.residual_nt,
        "warnings": result.warnings,
    }


def _statistical_leveling_summary(result: "StatisticalLevelingResult | None") -> dict:
    if result is None:
        return {"applied": False, "reason": None}
    return {
        "applied": result.applied,
        "reason": result.reason,
        "n_lines": result.n_lines,
        "n_pairs": result.n_pairs,
        "trend_window_lines": result.trend_window_lines,
        "max_shift_nt": result.max_shift_nt,
        "rms_shift_nt": result.rms_shift_nt,
        "roughness_before_nt": result.roughness_before_nt,
        "roughness_after_nt": result.roughness_after_nt,
        "warnings": result.warnings,
    }
