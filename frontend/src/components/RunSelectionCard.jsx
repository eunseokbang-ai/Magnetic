import { clock, runActions, runTitle, transitReason, transitWhere } from "../lib/runSelection";

// The card that opens over the map when a survey point is clicked: what
// stretch of flight was selected (it is highlighted on the map) and what
// can be done with it - exclude it, exclude the whole line it is a pass
// of, or put it back. One click on a button is the confirmation; the
// selection closes afterwards, and the same stretch can be clicked again
// to undo.
export default function RunSelectionCard({ run, busy, onAction, onShowProfile, onClose }) {
  if (!run) return null;
  const actions = runActions(run);
  const nIncluded = run.nPoints - run.nExcluded;
  const state =
    run.nExcluded === 0 ? "사용 중" : nIncluded === 0 ? "제외됨" : `사용 ${nIncluded.toLocaleString()}점 / 제외 ${run.nExcluded.toLocaleString()}점`;

  return (
    <div
      data-testid="run-selection-card"
      style={{
        position: "absolute", top: 60, left: "50%", transform: "translateX(-50%)", zIndex: 1300,
        width: 340, maxWidth: "calc(100% - 120px)", background: "#fffdf7", border: "1px solid #c9b88f",
        borderRadius: 8, boxShadow: "0 4px 18px rgba(40,30,10,0.28)", padding: "10px 12px", fontSize: 12, color: "#3a2f1c",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <span style={{ width: 22, height: 8, borderRadius: 4, background: "#fde047", border: "1px solid #111827", flexShrink: 0 }} />
        <b style={{ fontSize: 13, flex: 1 }}>{runTitle(run)}</b>
        <button type="button" onClick={onClose} title="선택 해제" style={{ border: "none", background: "none", cursor: "pointer", fontSize: 16, lineHeight: 1, color: "#6b5d43" }}>
          ×
        </button>
      </div>
      <div style={{ lineHeight: 1.6 }}>
        {run.nPoints.toLocaleString()}점 · {run.lengthM >= 1000 ? `${(run.lengthM / 1000).toFixed(2)} km` : `${run.lengthM.toFixed(0)} m`}
        {clock(run.timeStart) && ` · ${clock(run.timeStart)}–${clock(run.timeEnd)}`}
        <br />
        현재: <b style={{ color: run.nExcluded === 0 ? "#166534" : nIncluded === 0 ? "#991b1b" : "#92400e" }}>{state}</b>
        {run.kind === "line" && run.nLinePoints > run.nPoints && (
          <span style={{ color: "#6b5d43" }}> · 측선 #{run.lineId}은 이 구간 말고도 {(run.nLinePoints - run.nPoints).toLocaleString()}점이 더 있습니다</span>
        )}
      </div>
      {run.transit && (
        <div style={{ marginTop: 6, padding: "5px 7px", background: "#fef3c7", border: "1px solid #fde68a", borderRadius: 5, lineHeight: 1.5 }}>
          {run.transit.verdict === "transit" ? "자동 제외 사유" : "확인 필요"} ({transitWhere(run.transit)}): {transitReason(run.transit)}
          {run.transit.source_file && <span style={{ color: "#6b5d43" }}> · {run.transit.source_file}</span>}
        </div>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 5, marginTop: 9 }}>
        {actions.map((a) => (
          <button
            key={a.key}
            type="button"
            data-testid={`run-action-${a.key}`}
            disabled={busy}
            onClick={() => onAction(a.action, a.scope)}
            style={{
              padding: "6px 8px", fontSize: 12, borderRadius: 6, cursor: busy ? "wait" : "pointer", opacity: busy ? 0.6 : 1,
              border: a.danger ? "1px solid #dc2626" : "1px solid #16a34a",
              background: a.danger ? "#fef2f2" : "#f0fdf4",
              color: a.danger ? "#b91c1c" : "#166534",
              fontWeight: a.key === actions[0].key ? 600 : 400,
            }}
          >
            {a.label}
          </button>
        ))}
        <div style={{ display: "flex", gap: 5 }}>
          {run.kind === "line" && onShowProfile && (
            <button type="button" onClick={() => onShowProfile(run.lineId)} style={{ flex: 1, padding: "5px 8px", fontSize: 12, borderRadius: 6, border: "1px solid #ddd0b2", background: "white", cursor: "pointer" }}>
              📈 측선 프로파일
            </button>
          )}
          <button type="button" onClick={onClose} style={{ flex: 1, padding: "5px 8px", fontSize: 12, borderRadius: 6, border: "1px solid #ddd0b2", background: "white", cursor: "pointer" }}>
            취소
          </button>
        </div>
      </div>
    </div>
  );
}
