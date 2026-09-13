"""Project._grid_for's cache key used to be built from resolved_max_distance
alone (value, cell_size_m, method, resolved_max_distance, wavelength), while
the actual grid_points() call underneath uses the raw max_distance_m
(store.py's own comment: None triggers a per-cell local-line-gap adaptive
threshold, an explicit float triggers one uniform project-wide threshold -
genuinely different algorithms, not just different numbers). Since
_resolve_max_distance(cell_size_m, max_distance_m) returns max_distance_m
unchanged whenever it isn't None, an explicit call whose max_distance_m
happens to equal what auto-resolution would have produced collided in the
cache with the auto (None) call and incorrectly reused its result.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app
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
    return project_id


def test_explicit_max_distance_matching_auto_value_does_not_collide_in_cache():
    project_id = _make_processed_project()
    project = project_store.get(project_id)
    cell_size_m = 20.0

    project._grid_for("anomaly", cell_size_m, method="nearest", max_distance_m=None)
    resolved = project._resolve_max_distance(cell_size_m, None)
    project._grid_for("anomaly", cell_size_m, method="nearest", max_distance_m=resolved)

    # Two distinct calls (auto vs an explicit value that happens to equal
    # what auto resolved to) must occupy two distinct cache entries, not
    # collide into one - a collision would mean the explicit call silently
    # got back the auto call's result, or vice versa.
    assert len(project.grid_cache) == 2


if __name__ == "__main__":
    test_explicit_max_distance_matching_auto_value_does_not_collide_in_cache()
    print("ALL CHECKS PASSED")
