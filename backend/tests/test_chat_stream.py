"""The chat assistant's request shape and its handling of how a turn can
end - with a mocked Anthropic client (no key, no network).

What is pinned here is what the API documentation asks for and the old
code did not do: the current model, a streamed request, a cache
breakpoint on the unchanging prefix, no thinking budget, tools run only
on a clean tool_use stop, and every other stop reason turned into
something the user can read.
"""
import json
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from app import chat
from app.chat import ChatError, run_chat_turn, stream_chat_turn
from app.main import app
from app.store import store

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"


def _text(text):
    return SimpleNamespace(type="text", text=text)


def _tool(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def _thinking():
    return SimpleNamespace(type="thinking", thinking="", signature="sig")


def _message(stop_reason, content, usage=None, stop_details=None):
    usage = usage or SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(stop_reason=stop_reason, content=content, usage=usage, stop_details=stop_details)


class _Stream:
    """Stands in for client.messages.stream(...): a context manager with
    text_stream and get_final_message()."""

    def __init__(self, message, chunk=4):
        self._message = message
        text = "".join(b.text for b in message.content if b.type == "text")
        self.text_stream = [text[i:i + chunk] for i in range(0, len(text), chunk)]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class _Client:
    def __init__(self, replies):
        self.calls = []
        self._replies = list(replies)
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        # the loop mutates its message list afterwards: keep what was sent
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Stream(reply)


@pytest.fixture(scope="module")
def project():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files={"files": ("d.csv", f, "text/csv")})
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files={"files": ("b.csv", f, "text/csv")})
    assert client.post(f"/api/projects/{pid}/process", json={}).status_code == 200
    return store.get(pid)


def _run(project, replies, message="측선이 몇 개인가요?", history=()):
    fake = _Client(replies)
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), patch("app.chat.anthropic.Anthropic", return_value=fake):
        events = list(stream_chat_turn(project, message, list(history)))
    return fake, events


def test_the_request_is_shaped_as_the_api_documents(project, monkeypatch):
    monkeypatch.delenv("MAGNETIC_CHAT_MODEL", raising=False)
    fake, events = _run(project, [_message("end_turn", [_text("안녕하세요.")])])
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 16000
    assert "thinking" not in call                                   # adaptive on its own; a budget is a 400
    # the unchanging prefix (tools + system) behind one breakpoint, the conversation behind a moving one
    assert call["system"] == [{"type": "text", "text": chat.SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    assert call["cache_control"] == {"type": "ephemeral"}
    assert call["tools"] is chat.TOOLS
    assert events[-1]["type"] == "done" and events[-1]["stop_reason"] == "end_turn"

    monkeypatch.setenv("MAGNETIC_CHAT_MODEL", "claude-sonnet-5-5")
    fake, _ = _run(project, [_message("end_turn", [_text("네.")])])
    assert fake.calls[0]["model"] == "claude-sonnet-5-5"


def test_text_streams_as_deltas_and_adds_up_to_the_reply(project):
    _, events = _run(project, [_message("end_turn", [_text("측선은 여섯 개입니다.")])])
    deltas = [e["delta"] for e in events if e["type"] == "text"]
    assert len(deltas) > 1 and "".join(deltas) == "측선은 여섯 개입니다."
    assert events[-1]["reply"] == "측선은 여섯 개입니다."


def test_tool_loop_runs_the_real_lookup_and_keeps_thinking_with_its_tool_call(project):
    first = _message("tool_use", [_thinking(), _text("조회하겠습니다."), _tool("toolu_1", "get_survey_summary", {})])
    second = _message("end_turn", [_text("측선은 6개입니다.")],
                      usage=SimpleNamespace(input_tokens=30, output_tokens=10, cache_read_input_tokens=900, cache_creation_input_tokens=0))
    fake, events = _run(project, [first, second])

    assert [e["type"] for e in events if e["type"] != "text"] == ["tool", "done"]
    assert events[-1]["tool_calls"] == [{"name": "get_survey_summary", "input": {}}]
    assert events[-1]["reply"] == "조회하겠습니다.\n\n측선은 6개입니다."
    assert events[-1]["usage"] == {"input_tokens": 130, "output_tokens": 30, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0}

    sent = fake.calls[1]["messages"]
    # the assistant turn goes back whole - thinking block included - followed by the tool result
    assert sent[-2] == {"role": "assistant", "content": first.content}
    result = sent[-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_1"
    assert json.loads(result["content"])["n_lines"] == 6             # the real project's number


def test_a_turn_cut_off_by_max_tokens_runs_no_tool_and_says_so(project):
    cut = _message("max_tokens", [_text("분석 결과는"), _tool("toolu_x", "sample_point", {"lat": 46.5})])   # half-written call
    fake, events = _run(project, [cut])
    assert len(fake.calls) == 1 and not [e for e in events if e["type"] == "tool"]
    assert events[-1]["stop_reason"] == "max_tokens" and events[-1]["tool_calls"] == []
    assert events[-1]["reply"].startswith("분석 결과는") and "끊겼습니다" in events[-1]["reply"]


def test_a_refusal_is_reported_with_its_explanation(project):
    refused = _message("refusal", [], stop_details=SimpleNamespace(category=None, explanation="policy"))
    _, events = _run(project, [refused])
    assert events[-1]["stop_reason"] == "refusal"
    assert "답할 수 없다" in events[-1]["reply"] and "policy" in events[-1]["reply"]


def test_a_paused_turn_is_sent_back_to_continue(project):
    paused = _message("pause_turn", [_text("계속 ")])
    fake, events = _run(project, [paused, _message("end_turn", [_text("이어서 답합니다.")])])
    assert len(fake.calls) == 2 and fake.calls[1]["messages"][-1] == {"role": "assistant", "content": paused.content}
    assert events[-1]["reply"] == "계속 이어서 답합니다."


def test_a_loop_that_never_settles_stops_at_the_limit(project):
    loop = [_message("tool_use", [_tool(f"t{i}", "list_reference_layers", {})]) for i in range(chat.MAX_TOOL_ITERATIONS)]
    fake, events = _run(project, loop)
    assert len(fake.calls) == chat.MAX_TOOL_ITERATIONS
    assert "횟수 제한" in events[-1]["reply"]


def test_history_is_sent_before_the_new_message(project):
    history = [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "안녕하세요."}]
    fake, _ = _run(project, [_message("end_turn", [_text("네.")])], message="측선 수는?", history=history)
    assert fake.calls[0]["messages"] == history + [{"role": "user", "content": "측선 수는?"}]


def _status_error(cls, status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return cls("nope", response=response, body=None)


@pytest.mark.parametrize("exc, expected", [
    (_status_error(anthropic.AuthenticationError, 401), "API 키"),
    (_status_error(anthropic.RateLimitError, 429), "요청 한도"),
    (_status_error(anthropic.NotFoundError, 404), "모델을 찾을 수 없습니다"),
    (_status_error(anthropic.InternalServerError, 529), "일시적으로"),
    (anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")), "연결하지 못했습니다"),
])
def test_api_failures_become_messages_a_user_can_act_on(project, exc, expected):
    fake = _Client([exc])
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), patch("app.chat.anthropic.Anthropic", return_value=fake):
        with pytest.raises(ChatError, match=expected):
            run_chat_turn(project, "안녕", [])


def test_the_assistant_can_see_vector_reference_layers(project):
    project.reference_vectors["drill"] = {"name": "drill", "n_points": 2, "n_lines": 0, "n_polygons": 0, "features": []}
    try:
        out = json.loads(chat._execute_tool(project, "list_reference_layers", {}))
    finally:
        project.reference_vectors.pop("drill")
    assert out["vector_layers"] == [{"name": "drill", "n_points": 2, "n_lines": 0, "n_polygons": 0}]


def test_stream_endpoint_sends_ndjson_events_and_errors_in_band(project):
    client = TestClient(app)
    replies = [_message("tool_use", [_tool("toolu_1", "get_survey_summary", {})]), _message("end_turn", [_text("측선은 6개입니다.")])]
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), patch("app.chat.anthropic.Anthropic", return_value=_Client(replies)):
        r = client.post(f"/api/projects/{project.id}/chat/stream", json={"message": "측선 수?", "history": []})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    # not gzipped, although the client accepts it: a compressor would hold the small chunks back
    assert "gzip" in r.request.headers["accept-encoding"] and r.headers["content-encoding"] == "identity"
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    assert events[0] == {"type": "tool", "name": "get_survey_summary", "input": {}}
    assert events[-1]["type"] == "done" and events[-1]["reply"] == "측선은 6개입니다."

    # a failure after the stream has started cannot change the status: it arrives as an event
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
            patch("app.chat.anthropic.Anthropic", return_value=_Client([_status_error(anthropic.RateLimitError, 429)])):
        r = client.post(f"/api/projects/{project.id}/chat/stream", json={"message": "측선 수?", "history": []})
    assert r.status_code == 200
    assert json.loads(r.text.splitlines()[-1]) == {"type": "error", "message": "요청 한도에 걸렸습니다. 잠시 뒤 다시 시도하세요."}

    # no key at all is still an ordinary 400 before anything is streamed
    with patch.dict("os.environ", {}, clear=False) as env:
        env.pop("ANTHROPIC_API_KEY", None)
        r = client.post(f"/api/projects/{project.id}/chat/stream", json={"message": "안녕", "history": []})
    assert r.status_code == 400 and "ANTHROPIC_API_KEY" in r.json()["detail"]
