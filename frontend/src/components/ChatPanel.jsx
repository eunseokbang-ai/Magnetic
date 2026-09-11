import { useEffect, useRef, useState } from "react";

const bubbleStyle = (role) => ({
  alignSelf: role === "user" ? "flex-end" : "flex-start",
  background: role === "user" ? "#a9631f" : "#e6dac0",
  color: role === "user" ? "white" : "#2c2418",
  borderRadius: 10,
  padding: "6px 10px",
  fontSize: 12,
  maxWidth: "92%",
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
});

export default function ChatPanel({ ready, messages, onSend, sending, error }) {
  const [input, setInput] = useState("");
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, sending]);

  const submit = () => {
    const text = input.trim();
    if (!text || sending) return;
    onSend(text);
    setInput("");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ fontSize: 11, color: "#8a7a5c" }}>
        측선 통계, 3차원 역산 자화율, 오일러 디컨볼루션 결과, 업로드한 참조 레이어(지질도 등)의 실제 값을 조회해 답합니다. 자력탐사만으로 암종·광종을
        단정할 수 없으니 참고용으로만 활용하고 현장 검증을 거치세요.
      </div>
      <div
        ref={scrollRef}
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 6,
          height: 260,
          overflowY: "auto",
          border: "1px solid #e6dac0",
          borderRadius: 8,
          padding: 8,
          background: "white",
        }}
      >
        {messages.length === 0 && (
          <div style={{ fontSize: 12, color: "#ab9a78" }}>
            {ready
              ? '예: "측선 통계 요약해줘", "위도 46.501, 경도 106.275 지점은 어때?"'
              : "먼저 자료를 처리하면 실제 수치를 조회해 더 정확히 답할 수 있습니다."}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={bubbleStyle(m.role)}>
            {m.content}
            {m.toolCalls && m.toolCalls.length > 0 && (
              <div style={{ marginTop: 4, fontSize: 10, opacity: 0.75 }}>조회: {m.toolCalls.map((t) => t.name).join(", ")}</div>
            )}
          </div>
        ))}
        {sending && <div style={{ ...bubbleStyle("assistant"), opacity: 0.6 }}>답변 생성 중...</div>}
      </div>
      {error && <div style={{ fontSize: 11, color: "#dc2626" }}>{error}</div>}
      <div style={{ display: "flex", gap: 6 }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder="질문을 입력하세요..."
          rows={2}
          style={{ flex: 1, fontSize: 12, padding: 6, borderRadius: 6, border: "1px solid #ddd0b2", resize: "none" }}
        />
        <button
          onClick={submit}
          disabled={sending || !input.trim()}
          style={{
            padding: "0 12px",
            borderRadius: 6,
            border: "1px solid #a9631f",
            background: sending || !input.trim() ? "#dcb37a" : "#a9631f",
            color: "white",
            cursor: sending || !input.trim() ? "default" : "pointer",
            fontSize: 12,
          }}
        >
          전송
        </button>
      </div>
    </div>
  );
}
