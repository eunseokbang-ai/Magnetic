import { useState } from "react";
import * as api from "../api";

const sectionStyle = { border: "1px solid #e6dac0", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const buttonStyle = {
  padding: "7px 10px",
  fontSize: 12,
  borderRadius: 6,
  border: "1px solid #a9631f",
  background: "white",
  color: "#a9631f",
  cursor: "pointer",
};

/**
 * Ground the survey flew twice, what the two passes disagree by, and one
 * level per flight solved from it. This is the only handle the app has on
 * flight-to-flight level differences when there are no tie lines - and
 * when there are no repeats either, the honest answer is a survey-design
 * one, which is what the backend returns instead of a correction.
 */
export default function RepeatPassPanel({ projectId, processSummary, onApplied, onError }) {
  const [open, setOpen] = useState(false);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);

  const ready = !!processSummary;

  const run = async () => {
    try {
      setBusy(true);
      setResult(await api.analyzeRepeatPasses(projectId, {}));
    } catch (e) {
      onError && onError(e);
    } finally {
      setBusy(false);
    }
  };

  const apply = async (mode) => {
    try {
      setBusy(true);
      const resp = await api.applyRepeatPassLeveling(projectId, { mode });
      onApplied && (await onApplied(resp));
      setResult(await api.analyzeRepeatPasses(projectId, {}));
    } catch (e) {
      onError && onError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={sectionStyle}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          width: "100%",
          textAlign: "left",
          padding: "8px 12px",
          border: "none",
          background: "none",
          fontSize: 12,
          fontWeight: 700,
          color: "#4a3d28",
          cursor: "pointer",
        }}
      >
        {open ? "▾" : "▸"} 재비행 구간 비교 (비행 간 레벨·반복성)
      </button>
      {open && (
        <div style={bodyStyle}>
          <div style={{ color: "#8a7a5c", fontSize: 11 }}>
            같은 구간을 두 번 이상 비행했다면, 두 기록의 차이는 자기장이 아니라 탐사 자신의 오차입니다(잔여 일변화,
            센서 드리프트, 고도차, 자세에 따른 헤딩 효과). 이 차이로 비행마다 레벨 하나를 풀어 맞출 수 있습니다 —
            타이라인이 없을 때 비행 간 단차를 잡는 유일한 방법입니다.
          </div>
          <button style={{ ...buttonStyle, opacity: ready && !busy ? 1 : 0.5 }} disabled={!ready || busy} onClick={run}>
            {busy ? "분석 중..." : "재비행 구간 찾기"}
          </button>

          {result && result.n_pairs === 0 && (
            <div style={{ color: "#b45309", lineHeight: 1.45 }}>{result.advice}</div>
          )}

          {result && result.n_pairs > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <div>
                재비행 구간 <b>{result.n_pairs}쌍</b> · 총 {(result.total_overlap_m / 1000).toFixed(2)} km · 간격 중앙값{" "}
                {result.median_hours_apart}시간
              </div>
              <div>
                레벨 차이 중앙값 <b>{result.median_offset_nt} nT</b> (최대 {result.max_offset_nt} nT)
                {result.median_offset_after_nt !== null && ` → 보정 후 ${result.median_offset_after_nt} nT`}
              </div>
              <div title="상수 보정으로는 설명되지 않는 성분입니다. 이 탐사의 실제 반복도로 보고하세요.">
                상수 보정 후 남는 차이(반복도): <b>{result.median_residual_rms_nt} nT rms</b>
              </div>
              <div style={{ maxHeight: 140, overflow: "auto", border: "1px solid #eee4cd", borderRadius: 6 }}>
                {result.pairs.map((p, i) => (
                  <div key={i} style={{ padding: "3px 6px", borderBottom: "1px solid #f3ecdc", fontSize: 11 }}>
                    측선 {p.line_a} ↔ {p.line_b} · {(p.overlap_m / 1000).toFixed(2)} km · 경로차 {p.across_track_m} m ·{" "}
                    <b>{p.offset_nt} nT</b> · 잔차 {p.residual_rms_nt} nT · {p.hours_apart}시간 간격
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                <button style={{ ...buttonStyle, opacity: busy ? 0.5 : 1 }} disabled={busy} onClick={() => apply("apply")}>
                  비행별 레벨 보정 적용 ({Object.keys(result.offsets).length}개 비행)
                </button>
                {result.applied && (
                  <button
                    style={{ ...buttonStyle, opacity: busy ? 0.5 : 1, borderColor: "#ddd0b2", color: "#6b5c42" }}
                    disabled={busy}
                    onClick={() => apply("reset")}
                  >
                    되돌리기
                  </button>
                )}
              </div>
              {result.applied && (
                <div style={{ color: "#0f766e", fontSize: 11 }}>
                  적용됨: {Object.entries(result.applied_offsets).map(([k, v]) => `비행 ${k} ${v > 0 ? "+" : ""}${v} nT`).join(" · ")}
                </div>
              )}
              <div style={{ color: "#8a7a5c", fontSize: 11 }}>
                보정은 측점 단계에 적용되어 구조물 모델 차감·수동 스무딩보다 먼저 들어가고, 되돌리기는 따로 동작합니다.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
