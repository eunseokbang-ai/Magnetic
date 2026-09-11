"""Equivalent-source continuation of the field outside the survey.

The claim this makes is measurable, so these tests measure it: masked
synthetic data, transformed, compared against the same transform of the
complete field. The rest is about what must never change - the readings
themselves, the survey footprint, and the answer when there is nothing to
continue into.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import TransformRequest
from app.processing.continuation import continuation_report, source_continuation_fill
from app.processing.transforms import second_derivative_nz
from app.store import store as project_store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"
CELL = 10.0


def _dipole_world(shape=(180, 220), cell=CELL, n=90, seed=3, inc_deg=52.0, dec_deg=-8.0):
    """A real potential field: the total-field anomaly of buried dipoles,
    known everywhere. Nothing any fill could reconstruct for free."""
    ny, nx = shape
    rng = np.random.default_rng(seed)
    i, d = np.radians(inc_deg), np.radians(dec_deg)
    fx, fy, fz = np.cos(i) * np.cos(d), np.cos(i) * np.sin(d), np.sin(i)
    yy, xx = np.meshgrid(np.arange(ny) * cell, np.arange(nx) * cell, indexing="ij")
    field = np.zeros(shape)
    for _ in range(n):
        x0 = rng.uniform(-0.1, 1.1) * ny * cell
        y0 = rng.uniform(-0.1, 1.1) * nx * cell
        depth = rng.uniform(5.0, 100.0)
        moment = rng.normal(0, 1.0) * depth ** 3 * rng.uniform(0.5, 4.0)
        rx, ry = xx - x0, yy - y0
        r2 = rx * rx + ry * ry + depth * depth
        r = np.sqrt(r2)
        dot = (rx * fx + ry * fy + depth * fz) / r
        field += moment * (3.0 * dot * dot - 1.0) / (r2 * r)
    return field * (44.0 / field.std())


def _footprint(shape):
    """An L-shaped survey with a ragged edge - a rectangle would let a
    fill get away with things a real outline does not."""
    ny, nx = shape
    keep = np.zeros(shape, dtype=bool)
    keep[12:ny - 12, 15:nx - 15] = True
    keep[: ny // 2, : nx // 2] = False
    rng = np.random.default_rng(11)
    for j in range(nx):
        keep[: 12 + rng.integers(0, 6), j] = False
        keep[ny - 12 - rng.integers(0, 6):, j] = False
    return keep


@pytest.fixture(scope="module")
def world():
    field = _dipole_world()
    keep = _footprint(field.shape)
    return field, keep, np.where(keep, field, np.nan)


def _deep(keep, cells=12):
    from scipy import ndimage

    dist = ndimage.distance_transform_edt(np.pad(keep, 1, constant_values=False))[1:-1, 1:-1]
    return keep & (dist >= cells)


# ---------------------------------------------------------------------------
# what it must not touch
# ---------------------------------------------------------------------------


def test_measured_values_come_back_bit_for_bit(world):
    field, keep, masked = world

    filled = source_continuation_fill(masked)

    assert np.array_equal(filled[keep], masked[keep]), "a fill may only write where there is no data"


def test_every_gap_is_filled_and_nothing_is_left_nan(world):
    _field, keep, masked = world

    filled = source_continuation_fill(masked)

    assert np.isfinite(filled).all()
    assert not np.isfinite(masked[~keep]).any(), "the fixture really does have gaps"


def test_a_grid_with_no_gaps_is_returned_unchanged():
    field = _dipole_world(shape=(64, 64), n=20)

    assert source_continuation_fill(field) is field
    assert continuation_report(field)["applicable"] is False


def test_too_few_readings_declines_instead_of_fitting_noise():
    grid = np.full((64, 64), np.nan)
    grid[:8, :8] = 1.0

    report = continuation_report(grid)

    assert report["applicable"] is False and "적습니다" in report["reason"]
    assert np.array_equal(np.isnan(source_continuation_fill(grid)), np.isnan(grid))


# ---------------------------------------------------------------------------
# what it is for
# ---------------------------------------------------------------------------


def _quiet_error_ratio(out, truth, deep):
    """Error relative to signal in the quiet half of the survey - where the
    true derivative is small is exactly where a streak is visible."""
    quiet = deep & (np.abs(truth) < np.nanquantile(np.abs(truth[deep]), 0.5))
    return float(np.sqrt(np.mean((out - truth)[quiet] ** 2)) / np.sqrt(np.mean(truth[quiet] ** 2)))


def test_it_beats_the_nearest_value_fill_against_a_known_answer(world):
    field, keep, masked = world
    truth = second_derivative_nz(field, CELL)
    deep = _deep(keep)

    before = second_derivative_nz(masked, CELL)
    # Transform the continued field, then show only the ground that was
    # actually flown - masking it back BEFORE the transform would hand
    # the FFT the same hole again and measure nothing.
    after = np.where(keep, second_derivative_nz(source_continuation_fill(masked), CELL), np.nan)

    assert _quiet_error_ratio(after, truth, deep) < 0.5 * _quiet_error_ratio(before, truth, deep)


def test_it_does_not_cost_the_strong_anomalies_any_amplitude(world):
    """The whole point is buying back quiet ground without paying in
    targets - a filter that shaved the peaks would be a worse trade than
    the streak it removes."""
    field, keep, masked = world
    truth = second_derivative_nz(field, CELL)
    deep = _deep(keep)
    peaks = deep & (np.abs(truth) >= np.nanquantile(np.abs(truth[deep]), 0.99))

    after = np.where(keep, second_derivative_nz(source_continuation_fill(masked), CELL), np.nan)

    kept = np.abs(after[peaks]).mean() / np.abs(truth[peaks]).mean()
    assert 0.95 <= kept <= 1.05, f"peak amplitude changed by {100 * (kept - 1):.1f}%"


def test_the_continuation_outside_is_a_field_not_a_copy_of_the_edge(world):
    """Nearest fill repeats a boundary reading out into the empty ground,
    which is the column of invented data the streak comes from. A fitted
    source layer has no reason to hold a value constant like that."""
    _field, keep, masked = world
    filled = source_continuation_fill(masked)

    gap = ~keep
    # Along each column, how often does a filled cell exactly repeat the
    # one below it (to within a hair)? Nearest fill does this constantly.
    from scipy import ndimage

    nearest = masked[tuple(ndimage.distance_transform_edt(gap, return_distances=False, return_indices=True))]
    def repeat_rate(a):
        d = np.abs(np.diff(np.where(gap, a, np.nan), axis=0))
        return float(np.nanmean(d < 1e-6))

    assert repeat_rate(filled) < 0.1 * repeat_rate(nearest) + 1e-9


def test_the_result_is_cached_on_the_grid_contents(world):
    _field, _keep, masked = world
    import time

    source_continuation_fill(masked)
    t0 = time.perf_counter()
    again = source_continuation_fill(masked)
    assert time.perf_counter() - t0 < 0.5, "a repeat call must come from the cache"
    assert np.array_equal(again, source_continuation_fill(masked))


def test_the_cache_is_not_shared_between_different_grids(world):
    _field, keep, masked = world
    other = np.where(keep, _dipole_world(seed=99), np.nan)

    a = source_continuation_fill(masked)
    b = source_continuation_fill(other)

    assert not np.allclose(a[~keep], b[~keep])


# ---------------------------------------------------------------------------
# through the pipeline
# ---------------------------------------------------------------------------


def _processed_project():
    client = TestClient(app)
    project_id = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(
        f"/api/projects/{project_id}/process",
        json={"line_params": {}, "diurnal_params": {}, "heading_correction": {},
              "auto_display_boundary": False},
    )
    assert r.status_code == 200, r.text
    return project_store.get(project_id)


def _req(**over):
    # max_distance_m leaves ground unflown inside the grid rectangle; the
    # sample survey is a single dense strip that otherwise fills its own
    # bounding box, and with no gaps there is nothing to continue into.
    kwargs = {"transform": "dyz", "value": "anomaly", "cell_size_m": 5.0,
              "method": "linear", "max_distance_m": 8.0}
    kwargs.update(over)
    return TransformRequest(**kwargs)


def test_it_is_on_by_default_and_says_so():
    project = _processed_project()

    overlay = project.get_transform_overlay(_req())

    report = overlay["boundary_continuation"]
    assert report["applies"] and report["applied"] is True
    assert report["gap_pct"] > 0


def test_a_grid_with_no_gaps_says_there_is_nothing_to_continue_into():
    """Declining is the right answer here, and it has to be visible: a
    fully covered grid has no boundary to inherit."""
    project = _processed_project()

    report = project.get_transform_overlay(_req(max_distance_m=None))["boundary_continuation"]

    assert report["applied"] is False and "빈 칸이 없어" in report["reason"]


def test_turning_it_off_is_reported_rather_than_silent():
    project = _processed_project()

    report = project.get_transform_overlay(_req(boundary_continuation=False))["boundary_continuation"]

    assert report["applied"] is False and "껐습니다" in report["reason"]


def test_the_survey_footprint_is_unchanged_by_the_continuation():
    """It fills the gap to compute the transform; it must not then display
    ground that was never flown."""
    project = _processed_project()

    project.get_transform_overlay(_req(boundary_continuation=False))
    without = np.isnan(project.last_overlay.values)
    project.get_transform_overlay(_req(boundary_continuation=True))
    with_it = np.isnan(project.last_overlay.values)

    assert np.array_equal(without, with_it)


def test_it_changes_the_derivative_it_is_supposed_to_change():
    project = _processed_project()

    project.get_transform_overlay(_req(boundary_continuation=False))
    before = project.last_overlay.values.copy()
    project.get_transform_overlay(_req(boundary_continuation=True))
    after = project.last_overlay.values

    ok = np.isfinite(before) & np.isfinite(after)
    assert not np.allclose(before[ok], after[ok]), "the option has to actually do something"


def test_a_transform_that_does_not_use_the_fft_is_left_alone():
    project = _processed_project()

    report = project.get_transform_overlay(_req(transform="detrend"))["boundary_continuation"]

    assert report == {"applies": False}
