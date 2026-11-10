"""Progress reporting and cancellation of long operations
(magcore/progress.py and the jobs that use it)."""
import pathlib
import sys
import threading
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

import app.store as store_module
from app.main import app
from app.store import store
from magcore import progress, tile_cache

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


# ------------------------------------------------------------- the module
def test_reporting_outside_a_job_does_nothing():
    progress.report("아무 일", 0.5)
    progress.check()
    assert progress.current() is None and progress.active_jobs() == []


def test_a_job_is_visible_while_it_runs_and_gone_afterwards():
    with progress.job("자료 처리", "p1") as job:
        progress.report("측선 판별", 0.25)
        (seen,) = progress.active_jobs()
        assert (seen["id"], seen["name"], seen["project_id"], seen["stage"], seen["fraction"]) == (job.id, "자료 처리", "p1", "측선 판별", 0.25)
        progress.report("격자 보간", None)                     # a stage that cannot say how far along it is
        assert progress.active_jobs()[0]["fraction"] is None
    assert progress.active_jobs() == []


def test_a_nested_job_is_a_stretch_of_the_outer_one():
    with progress.job("프로젝트 불러오기") as outer:
        with progress.span(0.1, 0.8):
            with progress.job("자료 처리") as inner:                 # e.g. the pipeline re-run inside a restore
                assert inner is outer and len(progress.active_jobs()) == 1
                progress.report("측선 판별", 0.5)
                assert outer.fraction == pytest.approx(0.45)      # halfway through 0.1..0.8
                with progress.span(0.5, 1.0):
                    progress.report(None, 0.5)
                    assert outer.fraction == pytest.approx(0.1 + 0.7 * 0.75)
        progress.report("마무리", 0.9)
        assert outer.fraction == pytest.approx(0.9)                # the span is over


def test_cancel_takes_effect_at_the_next_report():
    with pytest.raises(progress.JobCancelled, match="취소"):
        with progress.job("3차원 역산") as job:
            progress.report("메시 구성", 0.1)
            assert progress.cancel(job.id) is True
            assert progress.active_jobs()[0]["cancel_requested"] is True
            progress.report("감도 행렬", 0.2)                      # raises here, in the working thread
    assert progress.active_jobs() == []
    assert progress.cancel("no-such-job") is False


def test_jobs_in_different_threads_do_not_see_each_other():
    seen = {}

    def work(name):
        with progress.job(name) as job:
            time.sleep(0.05)
            seen[name] = progress.current() is job

    threads = [threading.Thread(target=work, args=(n,)) for n in ("a", "b")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen == {"a": True, "b": True}


# ------------------------------------------------------- through the API
def _project(client):
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files=[("files", ("d.csv", f.read(), "text/csv"))])
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files=[("files", ("b.csv", f.read(), "text/csv"))])
    return pid


def _in_background(fn):
    box = {}
    thread = threading.Thread(target=lambda: box.update(result=fn()))
    thread.start()
    return thread, box


def _wait_for_job(client, stage, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        jobs = client.get("/api/jobs").json()["jobs"]
        if jobs and jobs[0]["stage"] == stage:
            return jobs[0]
        time.sleep(0.02)
    raise AssertionError(f"no job reached stage {stage!r}: {client.get('/api/jobs').json()}")


def test_processing_reports_its_stages_in_order_and_finishes_clean():
    client = TestClient(app)
    pid = _project(client)
    marks = []
    original = progress.Job.report

    def recording(self, stage, fraction):
        marks.append((stage, fraction))
        return original(self, stage, fraction)

    with patch.object(progress.Job, "report", recording):
        assert client.post(f"/api/projects/{pid}/process", json={}).status_code == 200
    stages = [s for s, _ in marks if s]
    assert stages[0] == "전처리 (스파이크 제거·필터)" and "측선 판별" in stages and stages[-1] == "경계·품질 요약 정리"
    fractions = [f for _, f in marks if f is not None]
    assert fractions == sorted(fractions) and 0.0 < fractions[0] < fractions[-1] < 1.0
    assert client.get("/api/jobs").json() == {"jobs": []}


def test_cancelling_a_run_answers_409_and_leaves_the_previous_result(monkeypatch):
    client = TestClient(app)
    pid = _project(client)
    assert client.post(f"/api/projects/{pid}/process", json={}).status_code == 200
    project = store.get(pid)
    before_frame, before_summary = project.processed, project.process_summary()
    changed_before = project.changed_at

    # hold the second run inside line detection until the test has cancelled it
    release = threading.Event()
    real_detect_lines = store_module.detect_lines

    def slow_detect_lines(*args, **kwargs):
        release.wait(20)
        return real_detect_lines(*args, **kwargs)

    monkeypatch.setattr(store_module, "detect_lines", slow_detect_lines)
    thread, box = _in_background(lambda: client.post(f"/api/projects/{pid}/process", json={"filter_cutoff_hz": 0.5, "despike_params": {"enabled": False}}))
    job = _wait_for_job(client, "측선 판별")
    assert job["name"] == "자료 처리" and job["project_id"] == pid and job["fraction"] == pytest.approx(0.15)
    assert client.post(f"/api/jobs/{job['id']}/cancel").json() == {"cancel_requested": True}
    release.set()
    thread.join(30)

    r = box["result"]
    assert r.status_code == 409 and r.json()["cancelled"] is True and "취소" in r.json()["detail"]
    # nothing of the half-finished run is left: the same frame, the same summary, the same settings
    assert project.processed is before_frame
    assert project.process_summary() == before_summary
    assert project.last_params.filter_cutoff_hz != 0.5 and project.changed_at == changed_before
    assert client.get("/api/jobs").json() == {"jobs": []}
    # and the project still works
    assert client.post(f"/api/projects/{pid}/grid", json={"value": "anomaly", "cell_size_m": 10.0}).status_code == 200


def test_inversion_progress_moves_forward_through_the_iterations():
    client = TestClient(app)
    pid = _project(client)
    assert client.post(f"/api/projects/{pid}/process", json={}).status_code == 200
    marks = []
    original = progress.Job.report

    def recording(self, stage, fraction):
        original(self, stage, fraction)
        marks.append((self.stage, self.fraction))

    with patch.object(progress.Job, "report", recording):
        assert client.post(f"/api/projects/{pid}/inversion", json={"regularization_strength": 1.0}).status_code == 200
    stages = [s for s, _ in marks]
    assert "감도 행렬 계산" in stages and "반복 역산 (IRLS)" in stages
    fractions = [f for _, f in marks if f is not None]
    assert fractions == sorted(fractions) and fractions[-1] >= 0.9   # whole-bar fractions, mapped through the span


def test_cancelling_a_tile_download_stops_requests_and_keeps_what_arrived(tmp_path, monkeypatch):
    monkeypatch.setattr(tile_cache, "CACHE_ROOT", tmp_path / "tile_cache")
    client = TestClient(app)
    fetched = []
    hold = threading.Event()

    def fake_get(url, **kwargs):
        fetched.append(url)
        if len(fetched) == 3:
            hold.wait(20)                                         # the third tile is in flight when the cancel arrives
        return MagicMock(status_code=200, content=b"tile")

    bbox = {"source": "esri", "min_lat": 36.50, "min_lon": 128.45, "max_lat": 36.56, "max_lon": 128.55, "min_zoom": 14, "max_zoom": 14}
    total = client.post("/api/tiles/estimate", json=bbox).json()["n_tiles"]
    assert total > 10
    with patch("magcore.tile_cache.requests.get", side_effect=fake_get):
        thread, box = _in_background(lambda: client.post("/api/tiles/download", json=bbox))
        t0 = time.time()
        while len(fetched) < 3 and time.time() - t0 < 20:
            time.sleep(0.02)
        (job,) = client.get("/api/jobs").json()["jobs"]
        assert job["name"] == "오프라인 지도 받기" and job["fraction"] == pytest.approx(2 / total, abs=1e-3)
        client.post(f"/api/jobs/{job['id']}/cancel")
        hold.set()
        thread.join(30)
    assert box["result"].status_code == 409
    assert len(fetched) == 3                                       # no request after the cancel
    assert client.get("/api/tiles/status").json()["esri"]["n_tiles"] == 3
