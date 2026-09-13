"""The flight-line points are drawn client-side on top of the server-
rendered grid image. Whatever colours the two use have to agree, because
a point that renders a different colour from the grid cell holding the
same value looks exactly like a leveling error in the data - a stripe
down every flight line - and is easily mistaken for one.

They used to disagree badly: the points were coloured by a plain linear
ramp over the raw point min/max, while the grid image used whichever
stretch was selected over its own (often symmetric, percentile-clipped)
range. These tests pin the fix - `color_stops`, the value->colour mapping
the image was actually rendered with, handed to the frontend to
interpolate through (see colormap.js:makeColorScale).
"""
from __future__ import annotations

import numpy as np
import matplotlib.colors
import pytest

from app.processing.render import (
    _COLOR_STOP_FRACTIONS,
    _EqualizeNorm,
    _NormalNorm,
    _stretch_stops,
    robust_center_scale,
)

# One entry of a 256-colour map. Agreement closer than this means the
# point and the grid cell are literally the same colour on screen.
ONE_COLOR_ENTRY = 1.0 / 256


def _long_tailed_grid() -> np.ndarray:
    """A grid shaped like real anomaly data: a broad background plus a
    small population of strong anomalies, which is what pulls a raw
    min/max range so far away from the displayed range."""
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(0.0, 30.0, 20000), rng.normal(400.0, 80.0, 400)])


def _server_norm(values: np.ndarray, stretch: str):
    """The same normalization _render_rgba builds for each stretch."""
    if stretch == "equalize":
        return _EqualizeNorm(np.sort(values)), float(values.min()), float(values.max())
    if stretch == "normal":
        norm = _NormalNorm(*robust_center_scale(values))
        return norm, norm.vmin, norm.vmax
    vmin, vmax = float(np.percentile(values, 2)), float(np.percentile(values, 98))
    m = max(abs(vmin), abs(vmax))  # anomaly grids render symmetric about zero
    return matplotlib.colors.Normalize(vmin=-m, vmax=m), -m, m


def _client_position(value: float, stops: list[float]) -> float:
    """Mirror of makeColorScale's interpolation in frontend colormap.js -
    kept in step with it by the tests below."""
    last = len(stops) - 1
    if value <= stops[0]:
        return 0.0
    if value >= stops[last]:
        return 1.0
    lo, hi = 0, last
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if stops[mid] <= value:
            lo = mid
        else:
            hi = mid
    width = stops[hi] - stops[lo]
    return (lo + ((value - stops[lo]) / width if width > 0 else 0.0)) / last


@pytest.mark.parametrize("stretch", ["linear", "equalize", "normal"])
def test_point_colors_match_the_rendered_grid_for_every_stretch(stretch):
    values = _long_tailed_grid()
    norm, vmin, vmax = _server_norm(values, stretch)
    stops = _stretch_stops(values, stretch, vmin, vmax, _COLOR_STOP_FRACTIONS)

    probes = np.percentile(values, np.linspace(0.5, 99.5, 60))
    server = np.clip(np.asarray(norm(probes), dtype=float), 0.0, 1.0)
    client = np.array([_client_position(float(v), stops) for v in probes])

    assert np.max(np.abs(server - client)) < ONE_COLOR_ENTRY


@pytest.mark.parametrize("stretch", ["linear", "equalize", "normal"])
def test_the_old_linear_ramp_really_was_far_off(stretch):
    """Guards the premise: without color_stops the mismatch is not a
    subtle shade difference but most of the colour bar, which is why every
    flight line read as a stripe."""
    values = _long_tailed_grid()
    norm, _vmin, _vmax = _server_norm(values, stretch)
    probes = np.percentile(values, np.linspace(0.5, 99.5, 60))

    server = np.clip(np.asarray(norm(probes), dtype=float), 0.0, 1.0)
    old = np.clip((probes - values.min()) / (values.max() - values.min()), 0.0, 1.0)

    assert np.max(np.abs(server - old)) > 0.5


def test_color_stops_are_monotone_so_the_client_can_search_them():
    """makeColorScale binary-searches the stops; a non-monotone sequence
    would silently return the wrong colour."""
    values = _long_tailed_grid()
    for stretch in ("linear", "equalize", "normal"):
        norm, vmin, vmax = _server_norm(values, stretch)
        stops = np.array(_stretch_stops(values, stretch, vmin, vmax, _COLOR_STOP_FRACTIONS))
        assert np.all(np.diff(stops) >= 0), f"{stretch} stops are not monotone"
        assert len(stops) == len(_COLOR_STOP_FRACTIONS)


def test_overlay_response_carries_the_stops():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    for endpoint, path in (
        ("drone", "tests/fixtures/sample_drone_survey.csv"),
        ("base", "tests/fixtures/sample_base_station.csv"),
    ):
        with open(path, "rb") as f:
            client.post(f"/api/projects/{pid}/upload/{endpoint}", files={"files": ("f.csv", f, "text/csv")})
    client.post(f"/api/projects/{pid}/process", json={"line_params": {}, "diurnal_params": {}, "heading_correction": {}})

    for stretch in ("linear", "equalize", "normal"):
        overlay = client.post(
            f"/api/projects/{pid}/grid",
            json={"value": "anomaly", "cell_size_m": 20.0, "method": "nearest", "stretch": stretch},
        ).json()
        stops = overlay["color_stops"]
        assert len(stops) == len(_COLOR_STOP_FRACTIONS)
        assert np.all(np.diff(np.array(stops)) >= 0)
        # The stops span the range the image was rendered over, which is
        # the whole point - the points used to span something else.
        assert stops[0] == pytest.approx(overlay["vmin"], abs=1e-6)
        assert stops[-1] == pytest.approx(overlay["vmax"], abs=1e-6)
