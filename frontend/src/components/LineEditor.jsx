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
  onShowLineProfile,
  lineProfileLoadingId,
  onOpenFlightPathEditor,
  onExportBln,
  exportingBln,
  canExportBln,
  showRampPoints,
  onToggleShowRampPoints,
  smoothDrawMode,
  onToggleSmoothDrawMode,
  smoothing,
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
      {onToggleShowRampPoints && (
        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
          <input type="checkbox" checked={showRampPoints} onChange={(e) => onToggleShowRampPoints(e.target.checked)} />
          <span>이착륙 구간(이륙→측선시작, 측선종료→착륙) 표시 — 기본적으로 숨기고 제외/복원 대상에서도 제외됩니다</span>
        </label>
      )}

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
            {onShowLineProfile && (
              <button
                type="button"
                title="측선 프로파일 보기 (거리-값 그래프)"
                onClick={(e) => {
                  e.preventDefault();
                  onShowLineProfile(l.line_id);
                }}
                disabled={lineProfileLoadingId === l.line_id}
                style={{ border: "none", background: "none", cursor: "pointer", fontSize: 13, padding: "0 2px" }}
              >
                {lineProfileLoadingId === l.line_id ? "…" : "📈"}
              </button>
            )}
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

      {onToggleSmoothDrawMode && (
        <>
          <hr style={{ border: "none", borderTop: "1px solid #e5e7eb", margin: "4px 0" }} />
          <div style={{ fontSize: 12, color: "#374151" }}>
            지상 구조물(집, 건물 등)로 인한 자력값 왜곡을 지도에서 직접 제거합니다. 위 "12. 참조 레이어"에 드론
            정사영상(orthophoto)을 GeoTIFF로 올려두면 구조물 위치를 눈으로 확인하며 그 위에 영역을 그릴 수 있습니다.
            선택한 영역의 이상/TMI 값이 주변 값으로 보간되어 왜곡이 제거됩니다.
          </div>
          <button
            onClick={onToggleSmoothDrawMode}
            disabled={smoothing}
            style={{
              padding: "6px 10px",
              fontSize: 12,
              borderRadius: 6,
              border: smoothDrawMode ? "1px solid #16a34a" : "1px solid #d1d5db",
              background: smoothDrawMode ? "#f0fdf4" : "white",
              color: smoothDrawMode ? "#16a34a" : "#374151",
              cursor: "pointer",
              opacity: smoothing ? 0.6 : 1,
            }}
          >
            {smoothing ? "스무딩 적용 중..." : smoothDrawMode ? "지도에서 왜곡 영역 그리기 (종료하려면 다시 클릭)" : "지도에서 왜곡 영역 그려 스무딩"}
          </button>
        </>
      )}

      {onOpenFlightPathEditor && (
        <>
          <hr style={{ border: "none", borderTop: "1px solid #e5e7eb", margin: "4px 0" }} />
          <div style={{ fontSize: 12, color: "#374151" }}>
            비행 경로를 투영 좌표(미터) 평면에서 확대해 보며 자르기/복원할 수 있는 전용 편집기입니다 (DroneMagAdv 방식).
          </div>
          <button
            onClick={onOpenFlightPathEditor}
            style={{ padding: "6px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #2563eb", background: "#eff6ff", color: "#2563eb", cursor: "pointer" }}
          >
            비행 경로 편집기 열기
          </button>
        </>
      )}

      {onExportBln && (
        <>
          <button
            onClick={onExportBln}
            disabled={exportingBln || !canExportBln}
            style={{ padding: "6px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #d1d5db", background: "white", color: "#374151", cursor: "pointer" }}
          >
            {exportingBln ? "저장 중..." : "마지막으로 그린 영역을 BLN(Surfer Blanking)으로 저장"}
          </button>
          {!canExportBln && <div style={{ fontSize: 11, color: "#9ca3af" }}>먼저 위 "지도에서 영역 그리기"로 폴리곤/사각형을 그리세요.</div>}
        </>
      )}
    </div>
  );
}
