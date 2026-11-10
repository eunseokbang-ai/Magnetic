import { describe, expect, it } from "vitest";
import { afterFailure, applyStreamEvent, historyFor, withAnswer, withQuestion } from "./chatMessages";

const asked = withQuestion([], "측선이 몇 개야?");

describe("a turn from question to answer", () => {
  it("starts with the question and an empty pending answer", () => {
    expect(asked).toEqual([
      { role: "user", content: "측선이 몇 개야?" },
      { role: "assistant", content: "", toolCalls: [], pending: true },
    ]);
  });

  it("text deltas add up in the pending answer", () => {
    let messages = applyStreamEvent(asked, { type: "text", delta: "측선은 " });
    messages = applyStreamEvent(messages, { type: "text", delta: "6개입니다." });
    expect(messages[1].content).toBe("측선은 6개입니다.");
    expect(messages[0]).toBe(asked[0]); // earlier messages are left as they are
  });

  it("a lookup stays active through the blank line after it, until real text arrives", () => {
    let messages = applyStreamEvent(asked, { type: "tool", name: "get_survey_summary", input: {} });
    expect(messages[1]).toMatchObject({ activeTool: "get_survey_summary", toolCalls: [{ name: "get_survey_summary", input: {} }] });
    messages = applyStreamEvent(messages, { type: "text", delta: "\n\n" });
    expect(messages[1].activeTool).toBe("get_survey_summary");
    messages = applyStreamEvent(messages, { type: "text", delta: "측선은" });
    expect(messages[1].activeTool).toBeNull();
  });

  it("the done event settles the answer", () => {
    const streamed = applyStreamEvent(asked, { type: "text", delta: "측선은 6" });
    const done = withAnswer(streamed, { reply: "측선은 6개입니다.", tool_calls: [{ name: "get_survey_summary", input: {} }], stop_reason: "end_turn" });
    expect(done[1]).toMatchObject({ content: "측선은 6개입니다.", pending: false, activeTool: null, stopReason: "end_turn" });
    expect(done[1].toolCalls).toHaveLength(1);
  });

  it("unknown event types are ignored", () => {
    expect(applyStreamEvent(asked, { type: "ping" })).toBe(asked);
  });
});

describe("a turn that fails", () => {
  it("drops an answer that never started", () => {
    expect(afterFailure(asked)).toEqual([{ role: "user", content: "측선이 몇 개야?" }]);
  });

  it("keeps what was written and stops showing it as in progress", () => {
    const partial = applyStreamEvent(applyStreamEvent(asked, { type: "tool", name: "sample_point", input: {} }), { type: "text", delta: "이 지점은" });
    const after = afterFailure(partial);
    expect(after).toHaveLength(2);
    expect(after[1]).toMatchObject({ content: "이 지점은", pending: false, activeTool: null });
  });
});

describe("history sent with the next question", () => {
  it("is the finished turns as role and text only", () => {
    const done = withAnswer(asked, { reply: "6개입니다.", tool_calls: [], stop_reason: "end_turn" });
    expect(historyFor(done)).toEqual([
      { role: "user", content: "측선이 몇 개야?" },
      { role: "assistant", content: "6개입니다." },
    ]);
  });

  it("leaves out an answer still being written and empty ones", () => {
    const streaming = applyStreamEvent(asked, { type: "text", delta: "측선" });
    expect(historyFor(streaming)).toEqual([{ role: "user", content: "측선이 몇 개야?" }]);
    expect(historyFor([{ role: "assistant", content: "" }])).toEqual([]);
  });
});
