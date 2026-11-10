import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "./api";

// fetch is the only thing these functions touch, so it is the only thing replaced.
function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

// A streamed body cut at arbitrary byte positions, as the network delivers it.
function streamedResponse(text, cuts) {
  const bytes = new TextEncoder().encode(text);
  const chunks = [];
  let at = 0;
  for (const cut of [...cuts, bytes.length]) {
    chunks.push(bytes.slice(at, cut));
    at = cut;
  }
  const body = new ReadableStream({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(c));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "application/x-ndjson" } });
}

afterEach(() => vi.unstubAllGlobals());

describe("request errors", () => {
  it("carry the backend's own explanation", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ detail: "자료 처리를 먼저 실행하세요." }, 400)));
    await expect(api.listJobs()).rejects.toThrow("자료 처리를 먼저 실행하세요.");
  });

  it("flatten FastAPI validation errors into one line", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ detail: [{ msg: "Input should be 'esri'" }, { msg: "Field required" }] }, 422)));
    await expect(api.cancelJob("j1")).rejects.toThrow("Input should be 'esri'; Field required");
  });
});

describe("the busy signal that drives the progress bar", () => {
  it("is on only while a state-changing request is in flight", async () => {
    const seen = [];
    const unsubscribe = api.subscribeBusy((busy) => seen.push(busy));
    let release;
    vi.stubGlobal("fetch", vi.fn(() => new Promise((resolve) => (release = () => resolve(jsonResponse({ ok: true }))))));

    const pending = api.openProject("p1"); // POST
    await Promise.resolve();
    expect(seen).toEqual([false, true]);
    release();
    await pending;
    expect(seen).toEqual([false, true, false]);
    unsubscribe();
  });

  it("ignores reads, the cancel request itself and the background layer sync", async () => {
    const seen = [];
    const unsubscribe = api.subscribeBusy((busy) => seen.push(busy));
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ jobs: [] })));
    await api.listJobs();
    await api.cancelJob("j1");
    await api.setReferenceLayerState("p1", []);
    expect(seen).toEqual([false]);
    unsubscribe();
  });

  it("is released when the request fails", async () => {
    const seen = [];
    const unsubscribe = api.subscribeBusy((busy) => seen.push(busy));
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ detail: "취소했습니다." }, 409)));
    await expect(api.openProject("p1")).rejects.toThrow();
    expect(seen).toEqual([false, true, false]);
    unsubscribe();
  });
});

describe("the chat stream", () => {
  const lines = [
    { type: "text", delta: "처리 요약을 조회하겠습니다." },
    { type: "tool", name: "get_survey_summary", input: {} },
    { type: "text", delta: "측선은 6개입니다." },
    { type: "done", reply: "처리 요약을 조회하겠습니다.\n\n측선은 6개입니다.", tool_calls: [{ name: "get_survey_summary", input: {} }], stop_reason: "end_turn" },
  ];
  const text = lines.map((l) => JSON.stringify(l)).join("\n") + "\n";

  it("delivers every event in order however the bytes are split - even inside a Korean character", async () => {
    // cut points chosen to land mid-line and mid-character (UTF-8 Hangul is three bytes)
    for (const cuts of [[], [5], [17, 18, 19, 60], Array.from({ length: 40 }, (_, i) => 3 + i * 7)]) {
      vi.stubGlobal("fetch", vi.fn(async () => streamedResponse(text, cuts)));
      const events = [];
      const done = await api.streamChatMessage("p1", "측선 수?", [], (e) => events.push(e));
      expect(events).toEqual(lines);
      expect(done).toEqual(lines[3]);
    }
  });

  it("sends the question and history as JSON", async () => {
    const fetchMock = vi.fn(async () => streamedResponse(text, []));
    vi.stubGlobal("fetch", fetchMock);
    await api.streamChatMessage("p1", "측선 수?", [{ role: "user", content: "안녕" }]);
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/projects/p1/chat/stream");
    expect(JSON.parse(options.body)).toEqual({ message: "측선 수?", history: [{ role: "user", content: "안녕" }] });
  });

  it("an error event rejects with its message", async () => {
    const body = JSON.stringify({ type: "text", delta: "조회" }) + "\n" + JSON.stringify({ type: "error", message: "요청 한도에 걸렸습니다." }) + "\n";
    vi.stubGlobal("fetch", vi.fn(async () => streamedResponse(body, [10])));
    await expect(api.streamChatMessage("p1", "q", [])).rejects.toThrow("요청 한도에 걸렸습니다.");
  });

  it("a stream that ends without a done event is reported as cut off", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => streamedResponse(JSON.stringify({ type: "text", delta: "조회" }) + "\n", [])));
    await expect(api.streamChatMessage("p1", "q", [])).rejects.toThrow("끊겼습니다");
  });

  it("a refusal before the stream starts (no API key) is a plain error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ detail: "ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다." }, 400)));
    await expect(api.streamChatMessage("p1", "q", [])).rejects.toThrow("ANTHROPIC_API_KEY");
  });
});
