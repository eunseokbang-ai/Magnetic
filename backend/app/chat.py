"""Tool-calling interpretation assistant for mineral-exploration and
near-surface compact-target (mine/UXO/hidden-vehicle) use cases.

Wraps the Claude API with a handful of read-only tools grounded in this
project's actual survey data (processed anomaly stats, 3D inversion
susceptibility, Euler deconvolution solutions, dipole-fit target
detections, uploaded reference/geology layers) so answers cite real
numbers pulled from the project instead of guessing from a map
screenshot. See store.py:Project for the data these tools read - nothing
here mutates project state.
"""
from __future__ import annotations

import json
import os
from typing import Iterator

import anthropic

# The model is read when a turn starts, so MAGNETIC_CHAT_MODEL can point an
# installation at another one without a code change.
DEFAULT_MODEL = "claude-opus-5-5"
# Thinking is always on with this model and counts toward max_tokens, so
# the old 2048 could end a turn before any answer text was written. 16000
# with streaming is the documented default shape: the stream keeps a long
# answer from running into an HTTP timeout.
MAX_TOKENS = 16000
MAX_TOOL_ITERATIONS = 8


def chat_model() -> str:
    return os.environ.get("MAGNETIC_CHAT_MODEL") or DEFAULT_MODEL

SYSTEM_PROMPT = """당신은 드론 자력탐사 자료 해석을 돕는 보조자입니다. 두 가지 용도로 쓰입니다: (1) 지질 정보와 종합한 광물자원탐사,
(2) 지뢰·불발탄·은닉 차량 등 근지표 금속 표적탐지. 사용자가 처리한 프로젝트의 실제 자료(측선 통계, 자력 이상값, 3차원 역산 자화율,
오일러 디컨볼루션 위치·심도 추정, 쌍극자 피팅 표적탐지 결과, 업로드된 지질도 등 참조 레이어)에 접근할 수 있는 도구가 주어집니다.

규칙:
- 수치를 절대 지어내지 마세요. 실제 값이 필요하면 반드시 먼저 도구를 호출해 확인한 뒤 답하세요.
- 광물자원탐사 맥락: 자력탐사만으로는 암종이나 광종을 특정할 수 없습니다. 자화율이 높다는 것은 자성광물(자철석 등)의 존재 가능성을
  시사할 뿐이며, 최종 해석에는 지질도·시추 등 다른 자료와의 종합, 그리고 현장 검증이 필요하다는 점을 항상 분명히 하세요.
- 표적탐지 맥락: get_target_detection_summary가 주는 쌍극자모멘트·크기등급은 상대적인 철질량 크기 추정치일 뿐이며, 자력탐사만으로
  표적의 정확한 종류(지뢰/포탄/전차/단순 고철 등)를 식별할 수 없습니다. 최소금속(low/minimal-metal) 지뢰는 자력 신호가 거의 없어
  탐지되지 않을 수 있다는 점, 그리고 실제 위치 확인·접근·처리는 반드시 EOD(폭발물처리반) 등 전문 인력이 현장에서 수행해야 한다는 점을
  항상 분명히 하세요 - 이 도구의 결과만으로 안전을 단정하거나 접근을 권하지 마세요.
- 좌표나 특정 지점에 대한 질문에는 sample_point 도구로 그 지점의 자력 이상값, (역산을 실행했다면) 추정 자화율 깊이별 분포, (참조
  레이어가 있다면) 지질도 등의 값을 함께 조회해 답하세요.
- 아직 실행되지 않은 단계(예: 역산 미실행)에 대한 질문에는 도구 결과의 안내를 그대로 사용자에게 전달하고, 먼저 해당 기능을 실행해보라고 안내하세요.
- 답변은 간결하게, 실제 조회한 수치를 인용하며 작성하세요."""

TOOLS = [
    {
        "name": "get_survey_summary",
        "description": (
            "이 프로젝트의 자료 처리 요약을 조회합니다: 측선 수, 유효 포인트 수, 자력 이상/TMI 통계"
            "(min/max/mean/std), 적용된 보정(스파이크 제거, 헤딩 보정, 타이라인 보정 등)."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_inversion_summary",
        "description": (
            "가장 최근에 실행한 3차원 자력 역산 결과 요약을 조회합니다: 자화율(SI) 통계, RMS 오차, "
            "격자 크기·심도·레이어 수 등 파라미터. 아직 역산을 실행하지 않았으면 그 사실을 알립니다."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_euler_solutions_summary",
        "description": (
            "가장 최근에 실행한 오일러 디컨볼루션 결과 요약을 조회합니다: 추정된 이상체 해의 개수, "
            "심도 범위 통계. 아직 실행하지 않았으면 그 사실을 알립니다."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_target_detection_summary",
        "description": (
            "가장 최근에 실행한 근지표 표적탐지(쌍극자 피팅) 결과를 조회합니다: 탐지된 표적 후보 목록"
            "(위경도, 심도, 쌍극자모멘트, 상대적 크기등급, 첨두 이상값, 적합도). 아직 실행하지 않았으면 그 사실을 알립니다."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_reference_layers",
        "description": (
            "이 프로젝트에 업로드된 참조 레이어 목록을 조회합니다: 지질도 등 GeoTIFF 레이어의 이름(layers)과 "
            "시추 위치·조사 경계 같은 벡터 레이어의 이름·피처 수(vector_layers)."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "sample_point",
        "description": (
            "특정 위경도 좌표에서의 실제 자료값을 조회합니다: 가장 가까운 측점의 자력 이상/TMI 값, "
            "(역산을 실행했다면) 해당 위치의 깊이별 추정 자화율, (참조 레이어가 있다면) 각 레이어의 값."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "위도 (decimal degrees)"},
                "lon": {"type": "number", "description": "경도 (decimal degrees)"},
            },
            "required": ["lat", "lon"],
        },
    },
]


class ChatError(RuntimeError):
    pass


def _execute_tool(project, name: str, tool_input: dict) -> str:
    if name == "get_survey_summary":
        if project.processed is None:
            return json.dumps({"error": "자료 처리가 아직 실행되지 않았습니다."}, ensure_ascii=False)
        summary = project.process_summary()
        summary.pop("lines", None)
        return json.dumps(summary, ensure_ascii=False, default=str)

    if name == "get_inversion_summary":
        if not project.inversion_summary_cache:
            return json.dumps({"error": "아직 3차원 역산을 실행하지 않았습니다."}, ensure_ascii=False)
        return json.dumps(project.inversion_summary_cache, ensure_ascii=False, default=str)

    if name == "get_euler_solutions_summary":
        if not project.euler_summary_cache:
            return json.dumps({"error": "아직 오일러 디컨볼루션을 실행하지 않았습니다."}, ensure_ascii=False)
        summary = {k: v for k, v in project.euler_summary_cache.items() if k != "solutions"}
        return json.dumps(summary, ensure_ascii=False, default=str)

    if name == "get_target_detection_summary":
        if not project.target_summary_cache:
            return json.dumps({"error": "아직 근지표 표적탐지를 실행하지 않았습니다."}, ensure_ascii=False)
        summary = dict(project.target_summary_cache)
        # cap the list so a very cluttered detection doesn't blow the tool
        # result budget - callers can re-run with a tighter min_fit_quality
        # to shrink it further if they need the full list.
        targets = summary.get("targets") or []
        if len(targets) > 30:
            summary["targets"] = targets[:30]
            summary["note"] = f"적합도 상위 30개만 표시됨 (전체 {len(targets)}개)."
        return json.dumps(summary, ensure_ascii=False, default=str)

    if name == "list_reference_layers":
        vectors = [{"name": name, "n_points": layer.get("n_points", 0), "n_lines": layer.get("n_lines", 0),
                    "n_polygons": layer.get("n_polygons", 0)} for name, layer in project.reference_vectors.items()]
        return json.dumps({"layers": list(project.reference_layers.keys()), "vector_layers": vectors}, ensure_ascii=False)

    if name == "sample_point":
        try:
            result = project.sample_point(float(tool_input["lat"]), float(tool_input["lon"]))
        except Exception as exc:  # keep the loop going with an error the model can react to
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False, default=str)

    return json.dumps({"error": f"알 수 없는 도구입니다: {name}"}, ensure_ascii=False)


def _api_error_message(exc: Exception) -> str:
    """An SDK error as something the person at the screen can act on."""
    if isinstance(exc, anthropic.AuthenticationError):
        return "ANTHROPIC_API_KEY가 올바르지 않습니다. 백엔드 실행 환경의 API 키를 확인하세요."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return f"이 API 키로는 모델({chat_model()})을 쓸 수 없습니다."
    if isinstance(exc, anthropic.NotFoundError):
        return f"모델을 찾을 수 없습니다: {chat_model()} (MAGNETIC_CHAT_MODEL 설정을 확인하세요)."
    if isinstance(exc, anthropic.RateLimitError):
        return "요청 한도에 걸렸습니다. 잠시 뒤 다시 시도하세요."
    if isinstance(exc, anthropic.APIConnectionError):          # includes timeouts
        return "Claude API에 연결하지 못했습니다. 인터넷 연결을 확인하세요."
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code >= 500:
            return f"Claude API가 일시적으로 응답하지 못했습니다 ({exc.status_code}). 잠시 뒤 다시 시도하세요."
        return f"Claude API가 요청을 거절했습니다 ({exc.status_code}): {exc.message}"
    return f"Claude API 호출에 실패했습니다: {exc}"


def require_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ChatError(
            "ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다. 백엔드를 실행하는 환경에 API 키를 설정하세요."
        )
    return api_key


def stream_chat_turn(project, message: str, history: list[dict]) -> Iterator[dict]:
    """One user turn as a stream of events, through a manual tool-use loop:

        {"type": "text", "delta": str}          answer text as it is written
        {"type": "tool", "name", "input"}       a project lookup is being made
        {"type": "done", "reply", "tool_calls", "stop_reason", "usage"}

    history is [{"role", "content": str}] from earlier turns, kept by the
    client (the API is stateless). Raises ChatError for anything the user
    has to hear about (no key, API failure).

    What the request carries, and why:
    - The system prompt and the tool list never change, so they sit behind
      a cache breakpoint; the top-level cache_control moves a second one
      along the growing conversation, which is what makes the second and
      later requests of a tool loop cheap. (A prefix below the model's
      minimum simply is not cached - no error.)
    - No `thinking` argument: the model thinks adaptively on its own, and a
      thinking budget is rejected. The assistant turn is sent back whole
      (response.content) so thinking blocks stay attached to the tool
      calls they led to, which the API requires.
    - The loop looks at stop_reason before running a tool: a turn cut off
      by max_tokens may hold a half-written tool call, and a refusal is
      not a request to continue.
    """
    client = anthropic.Anthropic(api_key=require_api_key())

    messages: list[dict] = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})
    system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

    tool_calls_log: list[dict] = []
    text_parts: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    stop_reason = None
    response = None
    note = None

    for _ in range(MAX_TOOL_ITERATIONS):
        turn_text: list[str] = []
        try:
            with client.messages.stream(
                model=chat_model(),
                max_tokens=MAX_TOKENS,
                system=system,
                tools=TOOLS,
                messages=messages,
                cache_control={"type": "ephemeral"},
            ) as stream:
                for delta in stream.text_stream:
                    turn_text.append(delta)
                    yield {"type": "text", "delta": delta}
                response = stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise ChatError(_api_error_message(exc)) from exc

        if turn_text:
            text_parts.append("".join(turn_text))
        if getattr(response, "usage", None) is not None:
            for key in usage:
                usage[key] += int(getattr(response.usage, key, 0) or 0)
        stop_reason = response.stop_reason

        if stop_reason == "pause_turn":            # the API paused a long turn: send it back to continue
            messages.append({"role": "assistant", "content": response.content})
            continue
        if stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            yield {"type": "tool", "name": block.name, "input": block.input}
            result_text = _execute_tool(project, block.name, block.input)
            tool_calls_log.append({"name": block.name, "input": block.input})
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})
        messages.append({"role": "user", "content": tool_results})
        if turn_text:
            # what the model said before a lookup and what it says after are separate paragraphs
            text_parts.append("\n\n")
            yield {"type": "text", "delta": "\n\n"}
    else:
        note = "도구 호출 횟수 제한에 도달했습니다. 질문을 더 구체적으로 나눠서 다시 시도해주세요."

    if stop_reason == "max_tokens":
        note = '답변이 길어 중간에서 끊겼습니다. "이어서"라고 보내면 계속합니다.'
    elif stop_reason == "refusal":
        explanation = getattr(getattr(response, "stop_details", None), "explanation", None)
        note = "이 요청에는 답할 수 없다는 응답을 받았습니다." + (f" ({explanation})" if explanation else "")

    if note:
        delta = ("\n\n" if "".join(text_parts).strip() else "") + note
        text_parts.append(delta)
        yield {"type": "text", "delta": delta}

    yield {"type": "done", "reply": "".join(text_parts).strip(), "tool_calls": tool_calls_log,
           "stop_reason": stop_reason, "usage": usage}


def run_chat_turn(project, message: str, history: list[dict]) -> dict:
    """The same turn without the streaming: {"reply", "tool_calls",
    "stop_reason", "usage"}."""
    final = None
    for event in stream_chat_turn(project, message, history):
        if event["type"] == "done":
            final = event
    if final is None:
        raise ChatError("응답을 생성하지 못했습니다.")
    return {k: v for k, v in final.items() if k != "type"}
