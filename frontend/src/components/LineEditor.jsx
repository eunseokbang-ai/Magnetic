import { useState } from "react";

export default function LineEditor({
  lines,
  onToggleLines,
  drawMode,
  drawAction,
  onSetDrawAction,
  onToggleDrawMode,
  onResetManual,
  nManualIncluded,
  nManualExcluded,
  showPointsOverGrid,
  onToggleShowPointsOverGrid,
  showLineLabels,
  onToggleShowLineLabels,
}) {
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
        측선 체크를 해제하면 지도/그리딩에서 제외됩니다. 수동 포함: <b>{nManualIncluded ?? 0}</b> / 수동 제외: <b>{nManualExcluded ?? 0}</b>
      </div>

      <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
        <input type="checkbox" checked={showLineLabels} onChange={(e) => onToggleShowLineLabels(e.target.checked)} />
        <span>지도에 측선 번호 표시</span>
      </label>
      <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
        <input type="checkbox" checked={showPointsOverGrid} onChange={(e) => onToggleShowPointsOverGrid(e.target.checked)} />
        <span>그리드 위에 측선 점 표시</span>
      </label>

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
            <span style={{ flex: 1 }}>
              측선 #{l.line_id}
              {l.heading_group && (
                <span
                  title={l.heading_group === "A" ? "순방향" : "역방향"}
                  style={{
                    marginLeft: 6,
                    padding: "0 5px",
                    borderRadius: 4,
                    fontSize: 10,
                    background: l.heading_group === "A" ? "#dbeafe" : "#fde68a",
                    color: l.heading_group === "A" ? "#1e40af" : "#92400e",
                  }}
                >
                  {l.heading_group}
                </span>
              )}
            </span>
            <span style={{ color: "#6b7280" }}>
              {l.n_points}pt / {l.length_m.toFixed(0)}m
              {l.heading_shift_nt != null && ` / ${l.heading_shift_nt >= 0 ? "+" : ""}${l.heading_shift_nt.toFixed(1)}nT`}
            </span>
          </label>
        ))}
      </div>

      <div style={{ fontSize: 12, color: "#374151" }}>
        지도에서 영역을 그려 아래 모드대로 처리합니다 (자동 제외된 흔들림 구간도 회색 점으로 표시되니 그 위에 영역을 그리면 됩니다).
      </div>
      <div style={{ display: "flex", gap: 6 }}>
        <button
          onClick={() => onSetDrawAction("exclude")}
          style={{
            flex: 1,
            padding: "6px 8px",
            fontSize: 12,
            borderRadius: 6,
            border: drawAction === "exclude" ? "1px solid #dc2626" : "1px solid #d1d5db",
            background: drawAction === "exclude" ? "#fef2f2" : "white",
            color: drawAction === "exclude" ? "#dc2626" : "#374151",
            cursor: "pointer",
          }}
        >
          제외 모드
        </button>
        <button
          onClick={() => onSetDrawAction("include")}
          style={{
            flex: 1,
            padding: "6px 8px",
            fontSize: 12,
            borderRadius: 6,
            border: drawAction === "include" ? "1px solid #16a34a" : "1px solid #d1d5db",
            background: drawAction === "include" ? "#f0fdf4" : "white",
            color: drawAction === "include" ? "#16a34a" : "#374151",
            cursor: "pointer",
          }}
        >
          포함(복원) 모드
        </button>
      </div>
      <button
        onClick={onToggleDrawMode}
        style={{
          padding: "6px 10px",
          fontSize: 12,
          borderRadius: 6,
          border: drawMode ? "1px solid #2563eb" : "1px solid #d1d5db",
          background: drawMode ? "#eff6ff" : "white",
          color: drawMode ? "#2563eb" : "#374151",
          cursor: "pointer",
        }}
      >
        {drawMode ? `영역 선택 종료 (${drawAction === "include" ? "포함" : "제외"} 모드로 그리는 중)` : "지도에서 영역 그리기 시작"}
      </button>
      <button
        onClick={onResetManual}
        style={{ padding: "6px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #d1d5db", background: "white", cursor: "pointer" }}
      >
        수동 편집 모두 되돌리기 (완전 자동으로)
      </button>
    </div>
  );
}
