import { useState } from "react";
import { RemovalMethodControls } from "./AnomalyCandidatePanel";
import { transitReason, transitWhere } from "../lib/runSelection";

// The ferry legs line detection took out, and the runs it kept but was
// not sure of: each one a click away from being shown on the map, where
// the selection card offers to put it back (or take it out).
function TransitRuns({ transit, onSelectRun, selectedRunId }) {
  if (!transit) return null;
  const removed = transit.runs.filter((r) => r.verdict === "transit");
  const suspects = transit.runs.filter((r) => r.verdict !== "transit");
  if (!transit.enabled) {
    return <div style={{ fontSize: 12, color: "#8a7a5c" }}>이동 구간 자동 제외가 꺼져 있습니다 (2. 자료 처리 → 측선 판별 설정).</div>;
  }
  const row = (r) => (
    <button
      key={r.run_id}
      type="button"
      onClick={() => onSelectRun(r.run_id)}
      title={`${transitReason(r)} — 클릭하면 지도에서 이 구간을 보여 줍니다`}
      style={{
        display: "flex", gap: 6, alignItems: "baseline", width: "100%", textAlign: "left", padding: "3px 8px", fontSize: 11.5,
        border: "none", borderBottom: "1px solid #f3ecd9", cursor: "pointer",
        background: selectedRunId === r.run_id ? "#fef9c3" : "white", color: "#4a3d28",
      }}
    >
      <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {r.line_id != null ? `측선 #${r.line_id} · ` : ""}
        {(r.source_file || `비행 ${r.source_file_index + 1}`).replace(/\.csv$/i, "")} · {transitWhere(r)}
      </span>
      <span style={{ color: "#8a7a5c", whiteSpace: "nowrap" }}>
        {r.length_m.toFixed(0)}m · {r.speed_mps.toFixed(1)}m/s · {r.azimuth_deviation_deg >= 0 ? "+" : ""}{r.azimuth_deviation_deg.toFixed(1)}°
      </span>
    </button>
  );
  return (
    <div data-testid="transit-runs" style={{ fontSize: 12, color: "#4a3d28" }}>
      <div>
        이동 구간 자동 제외: <b>{transit.n_removed}</b>개 · {(transit.length_removed_m / 1000).toFixed(2)} km · {transit.n_points_removed.toLocaleString()}점
        {transit.n_suspect > 0 && <span style={{ color: "#92400e" }}> / 확인 필요 <b>{transit.n_suspect}</b>개</span>}
      </div>
      {transit.runs.length > 0 && (
        <details style={{ marginTop: 4 }}>
          <summary style={{ cursor: "pointer", color: "#6b5d43" }}>목록 보기 — 누르면 지도에 표시되고, 거기서 다시 포함하거나 제외할 수 있습니다</summary>
          <div style={{ maxHeight: 180, overflowY: "auto", border: "1px solid #e6dac0", borderRadius: 6, marginTop: 4 }}>
            {suspects.length > 0 && <div style={{ padding: "3px 8px", fontSize: 11, background: "#fef3c7", color: "#92400e" }}>확인 필요 (제외하지 않음)</div>}
            {suspects.map(row)}
            {removed.length > 0 && <div style={{ padding: "3px 8px", fontSize: 11, background: "#f3ecd9", color: "#6b5d43" }}>자동 제외됨 (이륙 지점↔측선 사이 이동)</div>}
            {removed.map(row)}
          </div>
        </details>
      )}
    </div>
  );
}

export default function LineEditor({
  lines,
  onToggleLines,
  overlapFor,
  onToggleOverlap,
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
  nSmoothedActions,
  onUndoSmoothing,
  removalMethod,
  setRemovalMethod,
  nSourceRemovals,
  onUndoSourceRemoval,
  onResetSourceRemovals,
  onResetAllSmoothing,
  runSelectEnabled,
  onToggleRunSelect,
  transit,
  onSelectRun,
  selectedRunId,
}) {
  const [uncheckedLines, setUncheckedLines] = useState(new Set());

  if (!lines || lines.length === 0) {
    return <div style={{ fontSize: 12, color: "#ab9a78" }}>자료 처리를 먼저 실행하세요.</div>;
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
      <div style={{ fontSize: 12, color: "#4a3d28" }}>
        측선 체크를 해제하면 지도/그리딩에서 제외됩니다. 수동 포함: <b>{nManualIncluded ?? 0}</b> / 수동 제외: <b>{nManualExcluded ?? 0}</b>
      </div>

      <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
        <input type="checkbox" checked={showLineLabels} onChange={(e) => onToggleShowLineLabels(e.target.checked)} />
        <span>지도에 측선 번호 표시</span>
      </label>
      <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
        <input type="checkbox" checked={showPointsOverGrid} onChange={(e) => onToggleShowPointsOverGrid(e.target.checked)} />
        <span title="측선 점은 그리드 이미지 '위에' 그려집니다. 그리드는 배경지도 위에 반투명하게 깔리는데 점은 그렇지 않기 때문에, 값이 똑같아도 점이 항상 더 진하게 보입니다. 즉 자료에 아무 문제가 없어도 측선마다 색 띠가 생깁니다 - 이걸 줄무늬(레벨 오차)로 오해하지 마세요. 자력값을 읽을 때는 이 옵션을 끄고 그리드만 보세요. 이 표시는 '어디를 실제로 비행했는지' 확인용입니다.">
          그리드 위에 측선 점 표시 — 비행 경로 확인용 (켜면 측선이 실제보다 진하게 보입니다) ⓘ
        </span>
      </label>
      {onToggleShowRampPoints && (
        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
          <input type="checkbox" checked={showRampPoints} onChange={(e) => onToggleShowRampPoints(e.target.checked)} />
          <span>이착륙 구간(이륙→측선시작, 측선종료→착륙) 표시 — 기본적으로 숨기고 제외/복원 대상에서도 제외됩니다</span>
        </label>
      )}

      {onToggleRunSelect && (
        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
          <input type="checkbox" checked={!!runSelectEnabled} onChange={(e) => onToggleRunSelect(e.target.checked)} />
          <span title="지도에서 측점 하나를 클릭하면 그 점과 이어진 구간(한 번의 측선 통과, 이동 구간, 선회)이 노랗게 선택되고, 제외할지 다시 포함할지 묻는 창이 뜹니다. 영역 그리기·값 읽기 등 다른 지도 도구를 쓰는 동안에는 잠시 꺼집니다.">
            지도에서 측점을 클릭해 이어진 구간 선택 → 제외/복원 ⓘ
          </span>
        </label>
      )}
      <TransitRuns transit={transit} onSelectRun={onSelectRun} selectedRunId={selectedRunId} />

      <div style={{ maxHeight: 220, overflowY: "auto", border: "1px solid #e6dac0", borderRadius: 6 }}>
        {lines.map((l) => (
          <label
            key={l.line_id}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "4px 8px",
              fontSize: 12,
              borderBottom: "1px solid #f3ecd9",
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
            <span style={{ color: "#8a7a5c" }}>
              {l.n_points}pt / {l.length_m.toFixed(0)}m
              {l.heading_shift_nt != null && ` / ${l.heading_shift_nt >= 0 ? "+" : ""}${l.heading_shift_nt.toFixed(1)}nT`}
            </span>
            {overlapFor && overlapFor(l.line_id) && (
              <span
                title={(() => {
                  const o = overlapFor(l.line_id);
                  const nf = o.noise_first_nt, ns = o.noise_second_nt;
                  return (
                    `이 측선은 ${o.overlap_m.toFixed(0)}m가 두 번 기록됐습니다 (멈췄다 후진 후 재개). ` +
                    `현재 ${o.kept === "first" ? "멈추기 전" : "재개 후"} 자료를 사용 중입니다` +
                    (o.reason === "noise" ? " (자동: 잡음이 더 적은 쪽)" : o.reason === "override" ? " (수동 지정)" : "") +
                    `. 잡음 비교 — 멈추기 전 ${nf != null ? nf.toFixed(3) : "?"}nT / 재개 후 ${ns != null ? ns.toFixed(3) : "?"}nT. ` +
                    `클릭하면 반대쪽 자료로 바꿉니다.`
                  );
                })()}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  if (onToggleOverlap) onToggleOverlap(l.line_id, overlapFor(l.line_id).kept === "first" ? "second" : "first");
                }}
                style={{
                  padding: "0 5px", borderRadius: 3, fontSize: 10, cursor: onToggleOverlap ? "pointer" : "default",
                  background: "#e0f2fe", color: "#075985", border: "1px solid #bae6fd",
                }}
              >
                겹침 {overlapFor(l.line_id).kept === "first" ? "앞" : "뒤"} {overlapFor(l.line_id).overlap_m.toFixed(0)}m
              </span>
            )}
            {l.n_segments > 1 && (
              <span
                title={
                  `이 측선은 ${l.n_segments}개 구간이 이어진 것입니다 (비행 중단 후 재개 등). ` +
                  (l.joins || [])
                    .map((j, i) => `이음 ${i + 1}: ${j.gap_seconds.toFixed(0)}초 끊김, 거리 ${j.gap_m.toFixed(0)}m`)
                    .join(" / ") +
                  ". 프로파일에서 이음 부분을 확인하고 필요하면 수동으로 편집하세요."
                }
                style={{
                  padding: "0 5px", borderRadius: 3, fontSize: 10,
                  background: "#fef3c7", color: "#92400e", border: "1px solid #fde68a",
                }}
              >
                이음 {l.n_segments - 1}
              </span>
            )}
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

      <div style={{ fontSize: 12, color: "#4a3d28" }}>
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
            border: drawAction === "exclude" ? "1px solid #dc2626" : "1px solid #ddd0b2",
            background: drawAction === "exclude" ? "#fef2f2" : "white",
            color: drawAction === "exclude" ? "#dc2626" : "#4a3d28",
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
            border: drawAction === "include" ? "1px solid #16a34a" : "1px solid #ddd0b2",
            background: drawAction === "include" ? "#f0fdf4" : "white",
            color: drawAction === "include" ? "#16a34a" : "#4a3d28",
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
          border: drawMode ? "1px solid #a9631f" : "1px solid #ddd0b2",
          background: drawMode ? "#faf0e2" : "white",
          color: drawMode ? "#a9631f" : "#4a3d28",
          cursor: "pointer",
        }}
      >
        {drawMode ? `영역 선택 종료 (${drawAction === "include" ? "포함" : "제외"} 모드로 그리는 중)` : "지도에서 영역 그리기 시작"}
      </button>
      <button
        onClick={onResetManual}
        style={{ padding: "6px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #ddd0b2", background: "white", cursor: "pointer" }}
      >
        수동 편집 모두 되돌리기 (완전 자동으로)
      </button>

      {onToggleSmoothDrawMode && (
        <>
          <hr style={{ border: "none", borderTop: "1px solid #e6dac0", margin: "4px 0" }} />
          <div style={{ fontSize: 12, color: "#4a3d28" }}>
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
              border: smoothDrawMode ? "1px solid #16a34a" : "1px solid #ddd0b2",
              background: smoothDrawMode ? "#f0fdf4" : "white",
              color: smoothDrawMode ? "#16a34a" : "#4a3d28",
              cursor: "pointer",
              opacity: smoothing ? 0.6 : 1,
            }}
          >
            {smoothing
              ? removalMethod === "model"
                ? "구조물 모델 맞추는 중..."
                : "스무딩 적용 중..."
              : smoothDrawMode
                ? "지도에서 왜곡 영역 그리기 (종료하려면 다시 클릭)"
                : removalMethod === "model"
                  ? "지도에서 구조물 영역 그려 모델 차감"
                  : "지도에서 왜곡 영역 그려 스무딩"}
          </button>
          {setRemovalMethod && (
            <RemovalMethodControls
              removalMethod={removalMethod}
              setRemovalMethod={setRemovalMethod}
              nSourceRemovals={nSourceRemovals || 0}
              onUndoSourceRemoval={onUndoSourceRemoval}
              onResetSourceRemovals={onResetSourceRemovals}
              busy={smoothing}
            />
          )}
          {(onUndoSmoothing || onResetAllSmoothing) && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <span style={{ color: "#8a7a5c" }}>
                {nSmoothedActions > 0 ? `스무딩 적용 ${nSmoothedActions}회` : "적용된 스무딩 없음"}
              </span>
              <button
                onClick={onUndoSmoothing}
                disabled={smoothing || !nSmoothedActions}
                style={{
                  marginLeft: "auto",
                  padding: "4px 8px",
                  fontSize: 12,
                  borderRadius: 6,
                  border: "1px solid #ddd0b2",
                  background: "white",
                  cursor: nSmoothedActions ? "pointer" : "default",
                  opacity: nSmoothedActions && !smoothing ? 1 : 0.5,
                }}
                title="가장 최근에 적용한 스무딩 1건만 취소합니다"
              >
                ↩ 실행취소
              </button>
              <button
                onClick={onResetAllSmoothing}
                disabled={smoothing || !nSmoothedActions}
                style={{
                  padding: "4px 8px",
                  fontSize: 12,
                  borderRadius: 6,
                  border: "1px solid #ddd0b2",
                  background: "white",
                  cursor: nSmoothedActions ? "pointer" : "default",
                  opacity: nSmoothedActions && !smoothing ? 1 : 0.5,
                }}
                title="지금까지 적용한 스무딩을 모두 취소하고 원본 값으로 되돌립니다"
              >
                전체 되돌리기
              </button>
            </div>
          )}
        </>
      )}

      {onOpenFlightPathEditor && (
        <>
          <hr style={{ border: "none", borderTop: "1px solid #e6dac0", margin: "4px 0" }} />
          <div style={{ fontSize: 12, color: "#4a3d28" }}>
            비행 경로를 투영 좌표(미터) 평면에서 확대해 보며 자르기/복원할 수 있는 전용 편집기입니다 (DroneMagAdv 방식).
          </div>
          <button
            onClick={onOpenFlightPathEditor}
            style={{ padding: "6px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #a9631f", background: "#faf0e2", color: "#a9631f", cursor: "pointer" }}
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
            style={{ padding: "6px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #ddd0b2", background: "white", color: "#4a3d28", cursor: "pointer" }}
          >
            {exportingBln ? "저장 중..." : "마지막으로 그린 영역을 BLN(Surfer Blanking)으로 저장"}
          </button>
          {!canExportBln && <div style={{ fontSize: 11, color: "#ab9a78" }}>먼저 위 "지도에서 영역 그리기"로 폴리곤/사각형을 그리세요.</div>}
        </>
      )}
    </div>
  );
}
