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

import anthropic

MODEL = "claude-opus-5"
MAX_TOOL_ITERATIONS = 6

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
        "description": "이 프로젝트에 업로드된 지질도 등 참조 레이어(GeoTIFF)의 이름 목록을 조회합니다.",
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
        return json.dumps({"layers": list(project.reference_layers.keys())}, ensure_ascii=False)

    if name == "sample_point":
        try:
            result = project.sample_point(float(tool_input["lat"]), float(tool_input["lon"]))
        except Exception as exc:  # keep the loop going with an error the model can react to
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False, default=str)

    return json.dumps({"error": f"알 수 없는 도구입니다: {name}"}, ensure_ascii=False)


def run_chat_turn(project, message: str, history: list[dict]) -> dict:
    """Runs one user turn through a manual tool-use loop against the
    Claude API and returns {"reply": str, "tool_calls": [...]}. history is
    a list of {"role": "user"|"assistant", "content": str} from prior
    turns in this conversation (kept client-side - the API is stateless)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ChatError(
            "ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다. 백엔드를 실행하는 환경에 API 키를 설정하세요."
        )
    client = anthropic.Anthropic(api_key=api_key)

    messages: list[dict] = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    tool_calls_log: list[dict] = []
    response = None
    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        if response.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result_text = _execute_tool(project, block.name, block.input)
            tool_calls_log.append({"name": block.name, "input": block.input})
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})
        messages.append({"role": "user", "content": tool_results})

    if response is None:
        raise ChatError("응답을 생성하지 못했습니다.")

    reply = "".join(block.text for block in response.content if block.type == "text")
    if not reply and response.stop_reason == "tool_use":
        reply = "도구 호출 횟수 제한에 도달했습니다. 질문을 더 구체적으로 나눠서 다시 시도해주세요."
    return {"reply": reply, "tool_calls": tool_calls_log}
