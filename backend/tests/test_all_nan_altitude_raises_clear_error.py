"""compute_igrf_total_field (processing/igrf.py) NaN-safely leaves a row
NaN whenever its own lat/lon/altitude is non-finite - a genuine fix for a
prior bug where even one NaN altitude poisoned an entire day's coarse
interpolation grid. But some device formats (SENSYS MagDrone R3 ASC, see
io_/drone_loader.py:_parse_sensys_r3_asc) always report altitude as NaN
for the whole survey, which used to crash loudly downstream (a grid
"points must be strictly ascending" ValueError from processing/igrf.py's
internals) and now, thanks to the NaN-safe fix, silently produces an
all-NaN anomaly instead - a real, blank-map failure with nothing pointing
back at "altitude data is missing" as the cause. run_pipeline must detect
the all-NaN case itself and raise a clear, actionable error."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.store import store as project_store

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def client():
    return TestClient(app)


def test_all_nan_altitude_raises_clear_error_instead_of_blank_result(client):
    r = client.post("/api/projects")
    project_id = r.json()["project_id"]
    drone_path = FIXTURES / "sample_drone_survey.csv"
    with open(drone_path, "rb") as f:
        r = client.post(f"/api/projects/{project_id}/upload/drone", files={"files": (drone_path.name, f, "text/csv")})
    assert r.status_code == 200, r.text

    project = project_store.get(project_id)
    project.drone_raw["altitude_ellipsoidal_m"] = np.nan

    r = client.post(
        f"/api/projects/{project_id}/process",
        json={"diurnal_params": {"mode": "assume_constant"}},
    )
    assert r.status_code == 400, r.text
    assert "고도" in r.json()["detail"]


if __name__ == "__main__":
    print("This test uses pytest's client fixture - run via pytest.")
