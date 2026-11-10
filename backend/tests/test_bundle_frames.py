"""Project bundles carry their frames as npz arrays, never as pickles:
round trip, legacy pickles through the allowlist, and a crafted pickle
refused before it can run anything."""
import io
import json
import os
import pathlib
import pickle
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.bundle_frames import BundleFrameError, frame_from_bytes, frame_from_legacy_pickle, frame_to_bytes
from app.main import app
from app.store import Project, ProjectError, store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def _frame(n=500):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-28 05:13:13.3", periods=n, freq="100ms"),
        "lat": 36.5 + np.linspace(0, 0.01, n),
        "mag": np.r_[np.nan, np.random.default_rng(0).normal(50000, 30, n - 1)],
        "point_id": np.arange(n, dtype="int64"),
        "small": np.arange(n, dtype="int16"),
        "used_precompensated_mag": np.arange(n) % 2 == 0,
        "note": ["a", None, "한글"] * (n // 3) + ["z"] * (n % 3),
    })
    df.loc[3, "timestamp"] = pd.NaT
    df.attrs = {"source_format": "magarrow", "n_files_total": 2, "source_formats": ["magarrow", "generic"], "flag": True,
                "numpy_int": np.int64(7)}
    return df


def test_round_trip_keeps_values_dtypes_nulls_and_attrs():
    df = _frame()
    back = frame_from_bytes(frame_to_bytes(df))
    assert list(back.columns) == list(df.columns) and len(back) == len(df)
    for c in ("timestamp", "lat", "mag", "point_id", "small", "used_precompensated_mag"):
        assert back[c].dtype == df[c].dtype, c
    pd.testing.assert_frame_equal(back.drop(columns="note"), df.drop(columns="note"))
    assert back["note"].tolist() == df["note"].tolist()                 # strings and their Nones
    assert pd.isna(back.loc[3, "timestamp"]) and np.isnan(back.loc[0, "mag"])
    assert back.attrs == {"source_format": "magarrow", "n_files_total": 2, "source_formats": ["magarrow", "generic"], "flag": True, "numpy_int": 7}


def test_what_cannot_be_read_back_safely_is_refused_at_save_time():
    with pytest.raises(BundleFrameError, match="문자열이 아닌 객체"):
        frame_to_bytes(pd.DataFrame({"o": [object(), object()]}))
    with pytest.raises(BundleFrameError, match="속성"):
        df = pd.DataFrame({"a": [1.0]})
        df.attrs = {"x": object()}
        frame_to_bytes(df)
    with pytest.raises(BundleFrameError, match="색인"):
        frame_to_bytes(_frame().iloc[::2])


def test_the_npz_member_holds_no_pickle():
    data = frame_to_bytes(_frame())
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:          # would raise on an object array
        assert all(npz[k].dtype != object for k in npz.files)
    with pytest.raises(BundleFrameError):
        frame_from_bytes(b"not an npz")


def test_legacy_pickled_frame_still_loads():
    df = _frame().drop(columns="note")
    for proto in (4, 5):
        back = frame_from_legacy_pickle(pickle.dumps(df, protocol=proto))
        pd.testing.assert_frame_equal(back, df)
        assert back.attrs["source_format"] == "magarrow"


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned > bundle_frames_pwned.txt",))


@pytest.mark.parametrize("payload", [
    pickle.dumps(_Evil()),
    pickle.dumps({"df": _Evil()}),
    b"cbuiltins\neval\n(S'__import__(\"os\").getcwd()'\ntR.",
    b"csubprocess\nPopen\n(S'whoami'\ntR.",
])
def test_a_crafted_pickle_is_refused_and_nothing_runs(payload, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(BundleFrameError, match="허용되지 않은"):
        frame_from_legacy_pickle(payload)
    assert not (tmp_path / "bundle_frames_pwned.txt").exists()


def _bundle(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_project_bundle_round_trip_and_legacy_and_crafted_bundles(tmp_path, monkeypatch):
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files=[("files", ("d.csv", f.read(), "text/csv"))])
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files=[("files", ("b.csv", f.read(), "text/csv"))])
    project = store.get(pid)
    data = project.save_project_bundle()
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert "drone_raw.npz" in names and "base_raw.npz" in names and not [n for n in names if n.endswith(".pkl")]

    fresh = Project(id="fresh")
    fresh.load_project_bundle(data)
    pd.testing.assert_frame_equal(fresh.drone_raw, project.drone_raw)
    pd.testing.assert_frame_equal(fresh.base_raw, project.base_raw)
    assert fresh.drone_raw.attrs == project.drone_raw.attrs

    # a bundle the previous version wrote: pickled frames
    legacy = _bundle({"drone_raw.pkl": pickle.dumps(project.drone_raw, protocol=pickle.HIGHEST_PROTOCOL),
                      "base_raw.pkl": pickle.dumps(project.base_raw, protocol=pickle.HIGHEST_PROTOCOL),
                      "meta.json": json.dumps({"last_params": None, "has_base": True})})
    old = Project(id="old")
    old.load_project_bundle(legacy)
    pd.testing.assert_frame_equal(old.drone_raw, project.drone_raw)

    # a bundle somebody crafted
    monkeypatch.chdir(tmp_path)
    crafted = _bundle({"drone_raw.pkl": pickle.dumps(_Evil()), "meta.json": json.dumps({"last_params": None})})
    with pytest.raises(ProjectError, match="허용되지 않은"):
        Project(id="x").load_project_bundle(crafted)
    assert not (tmp_path / "bundle_frames_pwned.txt").exists()
    r = client.post(f"/api/projects/{pid}/load", files={"file": ("evil.zip", crafted, "application/zip")})
    assert r.status_code == 400 and "허용되지 않은" in r.json()["detail"]
    assert not (tmp_path / "bundle_frames_pwned.txt").exists()
