"""Along-track point-to-point sampling distance QC: the UAV magnetics
guidelines list a "histogram of sampling distance / confirm maximum
distance is within contract" as a standard QA/QC deliverable, used to
catch GPS dropouts and data gaps (the guidelines recommend a 20 m
tolerance for breaks in survey lines due to dead-zones/GPS dropouts) that
a simple point count wouldn't reveal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Guideline-cited default tolerance for GPS dropouts/dead-zone breaks.
_DEFAULT_GAP_TOLERANCE_M = 20.0
_N_BINS = 20


def compute_sampling_distance_qc(
    df: pd.DataFrame, line_id_col: str = "line_id", gap_tolerance_m: float = _DEFAULT_GAP_TOLERANCE_M
) -> dict:
    """Histogram of consecutive-point distances within each flight line
    (time-ordered), plus a count of gaps exceeding gap_tolerance_m.
    available=False when there aren't enough active points to say
    anything meaningful."""
    active = df[df[line_id_col] >= 0]
    if len(active) < 4:
        return {"available": False}

    distances = []
    for _lid, group in active.groupby(line_id_col):
        ordered = group.sort_values("timestamp")
        x = ordered["x"].to_numpy()
        y = ordered["y"].to_numpy()
        if len(x) < 2:
            continue
        distances.append(np.hypot(np.diff(x), np.diff(y)))

    if not distances:
        return {"available": False}

    all_distances = np.concatenate(distances)
    if all_distances.size == 0:
        return {"available": False}

    counts, bin_edges = np.histogram(all_distances, bins=_N_BINS)
    n_gaps = int(np.sum(all_distances > gap_tolerance_m))

    return {
        "available": True,
        "gap_tolerance_m": gap_tolerance_m,
        "n_samples": int(all_distances.size),
        "median_distance_m": float(np.median(all_distances)),
        "max_distance_m": float(np.max(all_distances)),
        "n_gaps_exceeding_tolerance": n_gaps,
        "pct_gaps_exceeding_tolerance": float(100.0 * n_gaps / all_distances.size),
        "histogram_counts": [int(c) for c in counts],
        "histogram_bin_edges": [float(e) for e in bin_edges],
    }
