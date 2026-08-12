"""Project.grid_cache/transform_cache used to grow without bound until the
next full reprocess - and a single grid_cache entry can weigh up to
~100MB (GridResult carries values + tree_dist + max_distance_grid, at the
~3M-cell gridding cap), so a user experimenting with cell sizes and
interpolation methods could accumulate gigabytes. Both caches are now
LRU-capped (store._GRID_CACHE_MAX_ENTRIES / _TRANSFORM_CACHE_MAX_ENTRIES):
oldest-USED entry evicted first, a cache hit refreshes recency.

Also covers the transform-cache key carrying the RAW max_distance_m
alongside the resolved one - the same collision _grid_for's key was fixed
for (auto/None vs an explicit value equal to what auto resolves to use
genuinely different masking algorithms, so their grids - and therefore
their transforms - differ), one level up."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import app.store as store_module
from app.main import app
from app.models import TransformRequest
from app.store import store as project_store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def _make_processed_project():
    client = TestClient(app)
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{project_id}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    r = client.post(f"/api/projects/{project_id}/process", json={"line_params": {}, "diurnal_params": {}, "heading_correction": {}})
    assert r.status_code == 200, r.text
    return project_store.get(project_id)


def _grid_key_for_cell_size(project, cell_size):
    for key in project.grid_cache:
        if key[1] == cell_size:
            return key
    return None


def test_grid_cache_evicts_least_recently_used_beyond_cap():
    project = _make_processed_project()
    cap = store_module._GRID_CACHE_MAX_ENTRIES

    cell_sizes = [20.0 + 5.0 * i for i in range(cap + 2)]  # cap+2 distinct entries
    for cs in cell_sizes:
        project._grid_for("anomaly", cs, method="nearest")
        # re-touch the FIRST grid every round so it stays the most recently
        # used - under LRU it must survive even though it's the oldest
        # INSERTED, which is exactly what distinguishes LRU from FIFO
        if cs != cell_sizes[0]:
            project._grid_for("anomaly", cell_sizes[0], method="nearest")

    assert len(project.grid_cache) <= cap
    assert _grid_key_for_cell_size(project, cell_sizes[0]) is not None  # refreshed -> kept
    assert _grid_key_for_cell_size(project, cell_sizes[1]) is None  # oldest-used -> evicted
    assert _grid_key_for_cell_size(project, cell_sizes[-1]) is not None  # newest -> kept


def test_transform_cache_no_collision_between_auto_and_explicit_max_distance():
    project = _make_processed_project()
    cell_size = 20.0
    resolved = project._resolve_max_distance(cell_size, None)

    req_auto = TransformRequest(value="anomaly", cell_size_m=cell_size, method="nearest", transform="thdr", max_distance_m=None)
    req_explicit = TransformRequest(value="anomaly", cell_size_m=cell_size, method="nearest", transform="thdr", max_distance_m=resolved)

    grid_auto = project._grid_for("anomaly", cell_size, method="nearest", max_distance_m=None)
    grid_explicit = project._grid_for("anomaly", cell_size, method="nearest", max_distance_m=resolved)
    project._transform_values(grid_auto, req_auto)
    project._transform_values(grid_explicit, req_explicit)

    # Two distinct requests (auto vs explicit-equal max_distance_m, which
    # sit on genuinely different grids) must occupy two cache entries -
    # a collision would return one grid's THDR for the other's request.
    assert len(project.transform_cache) == 2


def test_transform_cache_respects_entry_cap(monkeypatch):
    project = _make_processed_project()
    monkeypatch.setattr(store_module, "_TRANSFORM_CACHE_MAX_ENTRIES", 2)

    grid = project._grid_for("anomaly", 20.0, method="nearest")
    for transform in ["thdr", "1vd", "as"]:
        req = TransformRequest(value="anomaly", cell_size_m=20.0, method="nearest", transform=transform)
        project._transform_values(grid, req)

    assert len(project.transform_cache) == 2
    remaining_transforms = {key[6] for key in project.transform_cache}  # key[6] = req.transform
    assert remaining_transforms == {"1vd", "as"}


if __name__ == "__main__":
    print("This test uses pytest fixtures/monkeypatch - run via pytest.")
