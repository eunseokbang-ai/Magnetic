"""Unit tests for processing/duplicate_lines.py's core grouping/quality
logic, independent of the full upload/process API (see
test_duplicate_lines_integration.py for the end-to-end path)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.processing.duplicate_lines import resolve_duplicate_lines


def _survey(rng, noisy_duplicate=True):
    n = 200
    x = np.linspace(0, 500, n)

    y0 = np.zeros(n)
    v0 = 100 + 5 * np.sin(x / 50) + rng.normal(0, 0.5, n)  # clean pass

    y1 = np.full(n, 1.5)  # near-duplicate track (1.5m offset, well inside tolerance)
    noise = 8.0 if noisy_duplicate else 0.5
    v1 = 100 + 5 * np.sin(x / 50) + rng.normal(0, noise, n)

    y2 = np.full(n, 50.0)  # a normal, distinctly-spaced line - must never be grouped
    v2 = 90 + 3 * np.sin(x / 60) + rng.normal(0, 0.5, n)

    return pd.DataFrame({
        "point_id": np.arange(3 * n),
        "x": np.concatenate([x, x, x]),
        "y": np.concatenate([y0, y1, y2]),
        "timestamp": pd.to_datetime(np.tile(np.arange(n), 3), unit="s"),
        "value": np.concatenate([v0, v1, v2]),
        "line_id": np.concatenate([np.zeros(n), np.ones(n), np.full(n, 2)]).astype(int),
    })


def test_detects_near_duplicate_track_and_excludes_the_noisier_pass():
    rng = np.random.default_rng(0)
    df = _survey(rng)
    n = 200

    result = resolve_duplicate_lines(df, value_col="value")
    assert result["available"] is True
    assert result["n_groups"] == 1
    assert result["n_points_excluded"] == n

    group = result["groups"][0]
    assert sorted(group["line_ids"]) == [0, 1]
    assert group["best_line_id"] == 0  # the clean pass

    excluded = set(result["excluded_point_ids"])
    assert not any(pid < n for pid in excluded), "the clean line (0) must not lose any points"
    assert all(n <= pid < 2 * n for pid in excluded), "only the noisy duplicate (1) should be excluded"
    assert not any(pid >= 2 * n for pid in excluded), "the normally-spaced line (2) must be untouched"


def test_normally_spaced_lines_are_never_grouped():
    """Two lines 50m apart (a completely ordinary survey line spacing)
    must never be mistaken for a repeat-flown duplicate, even though the
    same grouping machinery is shared with a much looser tolerance in
    processing/repeatability.py."""
    rng = np.random.default_rng(1)
    n = 200
    x = np.linspace(0, 500, n)
    df = pd.DataFrame({
        "point_id": np.arange(2 * n),
        "x": np.concatenate([x, x]),
        "y": np.concatenate([np.zeros(n), np.full(n, 50.0)]),
        "timestamp": pd.to_datetime(np.tile(np.arange(n), 2), unit="s"),
        "value": np.concatenate([
            100 + rng.normal(0, 0.5, n),
            90 + rng.normal(0, 0.5, n),
        ]),
        "line_id": np.concatenate([np.zeros(n), np.ones(n)]).astype(int),
    })

    result = resolve_duplicate_lines(df, value_col="value")
    assert result["available"] is True
    assert result["n_groups"] == 0
    assert result["n_points_excluded"] == 0
    assert result["excluded_point_ids"] == []


def test_unavailable_gracefully_with_fewer_than_two_lines():
    df = pd.DataFrame({
        "point_id": np.arange(50),
        "x": np.linspace(0, 100, 50),
        "y": np.zeros(50),
        "timestamp": pd.to_datetime(np.arange(50), unit="s"),
        "value": np.full(50, 100.0),
        "line_id": np.zeros(50, dtype=int),
    })
    result = resolve_duplicate_lines(df, value_col="value")
    assert result["available"] is True
    assert result["n_groups"] == 0
    assert result["excluded_point_ids"] == []


def test_tolerance_parameters_are_respected():
    """A track offset just outside the default 8m perpendicular tolerance
    must not be grouped; the same data with a widened tolerance must be."""
    rng = np.random.default_rng(2)
    n = 200
    x = np.linspace(0, 500, n)
    df = pd.DataFrame({
        "point_id": np.arange(2 * n),
        "x": np.concatenate([x, x]),
        "y": np.concatenate([np.zeros(n), np.full(n, 9.0)]),  # 9m offset
        "timestamp": pd.to_datetime(np.tile(np.arange(n), 2), unit="s"),
        "value": np.concatenate([
            100 + rng.normal(0, 0.5, n),
            100 + rng.normal(0, 5.0, n),
        ]),
        "line_id": np.concatenate([np.zeros(n), np.ones(n)]).astype(int),
    })

    default_result = resolve_duplicate_lines(df, value_col="value")
    assert default_result["n_groups"] == 0

    widened_result = resolve_duplicate_lines(df, value_col="value", perp_tolerance_m=10.0)
    assert widened_result["n_groups"] == 1
