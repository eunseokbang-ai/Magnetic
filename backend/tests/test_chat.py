"""Tests for the tool-calling chat assistant: verifies the tool-execution
functions return correct data grounded in a real processed project (no
network calls), the store-level API endpoints for reference layers and
the chat error path, and - via a mocked Anthropic client - that the
manual tool-use loop in chat.py drives the conversation correctly.
Does not require ANTHROPIC_API_KEY or network access.
"""
import io
import json
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.crs import CRS
from rasterio.transform import from_origin

from app.chat import _execute_tool, run_chat_turn
from app.main import app
from app.models import EulerDeconvolutionRequest, InversionParams, ProcessParams
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
    return client, project_id


def test_tool_execution_functions_grounded_in_real_project():
    client, project_id = _make_processed_project()
    project = project_store.get(project_id)

    out = json.loads(_execute_tool(project, "get_survey_summary", {}))
    assert "lines" not in out  # stripped to keep tool output compact
    assert out["n_lines"] == 4
    assert out["anomaly_stats"]["min"] is not None

    # not yet run
    out = json.loads(_execute_tool(project, "get_inversion_summary", {}))
    assert "error" in out
    out = json.loads(_execute_tool(project, "get_euler_solutions_summary", {}))
    assert "error" in out
    out = json.loads(_execute_tool(project, "list_reference_layers", {}))
    assert out["layers"] == []

    # sample_point before inversion: should still return nearest survey point
    pts = project.processed[project._active_mask()]
    lat, lon = float(pts["lat"].iloc[0]), float(pts["lon"].iloc[0])
    out = json.loads(_execute_tool(project, "sample_point", {"lat": lat, "lon": lon}))
    assert out["nearest_survey_point"]["distance_m"] < 5.0
    assert "inversion_susceptibility_profile" not in out

    # run inversion + euler, then re-check tool outputs
    r = client.post(f"/api/projects/{project_id}/inversion", json=InversionParams(regularization_strength=1.0).model_dump())
    assert r.status_code == 200, r.text
    r = client.post(
        f"/api/projects/{project_id}/euler-deconvolution",
        json=EulerDeconvolutionRequest(cell_size_m=10.0, structural_index=1.0, window_size_m=100.0).model_dump(),
    )
    assert r.status_code == 200, r.text

    out = json.loads(_execute_tool(project, "get_inversion_summary", {}))
    assert "error" not in out
    assert out["susceptibility_stats"]["min"] is not None

    out = json.loads(_execute_tool(project, "get_euler_solutions_summary", {}))
    assert "error" not in out
    assert out["n_solutions"] > 0
    assert "solutions" not in out  # summary strips the raw solution list

    out = json.loads(_execute_tool(project, "sample_point", {"lat": lat, "lon": lon}))
    assert "inversion_susceptibility_profile" in out
    profile = out["inversion_susceptibility_profile"]
    if "layers" in profile:
        assert len(profile["layers"]) > 0
        assert all("susceptibility_si" in layer for layer in profile["layers"])

    print("ALL TOOL EXECUTION CHECKS PASSED")


def test_reference_layer_upload_list_sample_remove():
    client, project_id = _make_processed_project()

    # build a tiny synthetic geology GeoTIFF covering the survey area, with an indexed colormap
    path = "/tmp/_chat_test_geology.tif"
    transform = from_origin(590000, 5155000, 10, 10)
    crs = CRS.from_epsg(32648)
    data = np.ones((50, 50), dtype="uint8")
    data[:25, :] = 2  # two rock-unit classes
    with rasterio.open(path, "w", driver="GTiff", height=50, width=50, count=1, dtype="uint8", crs=crs, transform=transform) as dst:
        dst.write(data, 1)
        dst.write_colormap(1, {1: (200, 200, 0, 255), 2: (0, 100, 200, 255)})

    with open(path, "rb") as f:
        r = client.post(f"/api/projects/{project_id}/reference-layers", files={"file": ("geology.tif", f, "image/tiff")})
    assert r.status_code == 200, r.text
    resp = r.json()
    assert resp["name"] == "geology.tif"
    assert resp["image_data_url"].startswith("data:image/png;base64,")

    project = project_store.get(project_id)
    out = json.loads(_execute_tool(project, "list_reference_layers", {}))
    assert out["layers"] == ["geology.tif"]

    # sample a point inside the raster bounds
    lat_c = (resp["bounds"][0][0] + resp["bounds"][1][0]) / 2
    lon_c = (resp["bounds"][0][1] + resp["bounds"][1][1]) / 2
    sample = project.sample_point(lat_c, lon_c)
    geo = sample["reference_layers"]["geology.tif"]
    assert geo["in_bounds"] is True
    assert geo["value"] in (1.0, 2.0)
    assert "color_rgba" in geo

    r = client.delete(f"/api/projects/{project_id}/reference-layers/geology.tif")
    assert r.status_code == 200, r.text
    out = json.loads(_execute_tool(project, "list_reference_layers", {}))
    assert out["layers"] == []
    print("ALL REFERENCE LAYER CHECKS PASSED")


def test_chat_endpoint_without_api_key_returns_400():
    client, project_id = _make_processed_project()
    with patch.dict("os.environ", {}, clear=False):
        import os

        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            r = client.post(f"/api/projects/{project_id}/chat", json={"message": "안녕하세요", "history": []})
            assert r.status_code == 400, r.text
            assert "ANTHROPIC_API_KEY" in r.json()["detail"]
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved
    print("CHAT MISSING-KEY ERROR PATH OK")


class _FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, id_, name, input_):
        self.id = id_
        self.name = name
        self.input = input_


class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


def test_run_chat_turn_tool_loop_with_mocked_client():
    """Verifies the manual agentic loop in chat.py: on a tool_use response
    it executes the real tool against a real project and feeds the result
    back, then returns the model's final text - without any network call."""
    client, project_id = _make_processed_project()
    project = project_store.get(project_id)

    call_log = []

    def fake_create(**kwargs):
        call_log.append(kwargs)
        if len(call_log) == 1:
            assert kwargs["messages"][-1]["content"] == "이 측선탐사는 측선이 몇 개인가요?"
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[_FakeToolUseBlock("toolu_1", "get_survey_summary", {})],
            )
        # second call: tool result should be present in the message history
        last_msg = kwargs["messages"][-1]
        assert last_msg["role"] == "user"
        assert last_msg["content"][0]["type"] == "tool_result"
        assert last_msg["content"][0]["tool_use_id"] == "toolu_1"
        payload = json.loads(last_msg["content"][0]["content"])
        assert payload["n_lines"] == 4
        return SimpleNamespace(stop_reason="end_turn", content=[_FakeTextBlock("측선은 4개입니다.")])

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("app.chat.anthropic.Anthropic", return_value=fake_client):
            result = run_chat_turn(project, "이 측선탐사는 측선이 몇 개인가요?", [])

    assert result["reply"] == "측선은 4개입니다."
    assert result["tool_calls"] == [{"name": "get_survey_summary", "input": {}}]
    assert len(call_log) == 2
    print("MOCKED TOOL-USE LOOP CHECKS PASSED")


if __name__ == "__main__":
    test_tool_execution_functions_grounded_in_real_project()
    test_reference_layer_upload_list_sample_remove()
    test_chat_endpoint_without_api_key_returns_400()
    test_run_chat_turn_tool_loop_with_mocked_client()
    print("\nALL CHECKS PASSED")
