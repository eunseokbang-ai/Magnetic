import { useState } from "react";

export default function LineEditor({ lines, onToggleLines, drawMode, onToggleDrawMode, onClearPolygonExclusions, nExcludedManual }) {
  const [uncheckedLines, setUncheckedLines] = useState(new Set());

  if (!lines || lines.length === 0) {
    return <div style={{ fontSize: 12, color: "#9ca3af" }}>자료 처리를 먼저 실행하세요.</div>;
  }

  const handleToggle = (lineId, checked) => {
    const next = new Set(uncheckedLines);
    if (checked) next.delete(lineId);
    else next.add(lineId);
    setUncheckedLines(next);
    onToggleLines([lineId], checked ? "include" : "exclude");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ fontSize: 12, color: "#374151" }}>
        측선 체크를 해제하면 지도/그리딩에서 제외됩니다. 수동 제외된 포인트: <b>{nExcludedManual ?? 0}</b>
      </div>

      <div style={{ maxHeight: 220, overflowY: "auto", border: "1px solid #e5e7eb", borderRadius: 6 }}>
        {lines.map((l) => (
          <label
            key={l.line_id}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "4px 8px",
              fontSize: 12,
              borderBottom: "1px solid #f3f4f6",
            }}
          >
            <input
              type="checkbox"
              checked={!uncheckedLines.has(l.line_id)}
              onChange={(e) => handleToggle(l.line_id, e.target.checked)}
            />
            <span style={{ flex: 1 }}>측선 #{l.line_id}</span>
            <span style={{ color: "#6b7280" }}>
              {l.n_points}pt / {l.length_m.toFixed(0)}m
            </span>
          </label>
        ))}
      </div>

      <button
        onClick={onToggleDrawMode}
        style={{
          padding: "6px 10px",
          fontSize: 12,
          borderRadius: 6,
          border: drawMode ? "1px solid #dc2626" : "1px solid #d1d5db",
          background: drawMode ? "#fef2f2" : "white",
          color: drawMode ? "#dc2626" : "#374151",
          cursor: "pointer",
        }}
      >
        {drawMode ? "영역 선택 종료" : "지도에서 영역 그려서 제외"}
      </button>
      <button
        onClick={onClearPolygonExclusions}
        style={{ padding: "6px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #d1d5db", background: "white", cursor: "pointer" }}
      >
        수동 제외 모두 되돌리기
      </button>
    </div>
  );
}
