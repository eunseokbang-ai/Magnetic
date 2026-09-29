import { useEffect, useState } from "react";
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
const inputStyle = { width: 90, padding: "4px 6px", fontSize: 12, border: "1px solid #ddd0b2", borderRadius: 4 };

const fmt = (v, d = 1) => (v === null || v === undefined ? "-" : Number(v).toFixed(d));

/**
 * The readings restated at one altitude. A terrain-following flight reads
 * each point at a different height, and every later step (grids, their
 * derivatives, Euler, the inversion) takes them to be on one plane. The
 * backend fits equivalent dipole layers to the readings at their true 3D
 * positions and shifts each reading by the layers' difference between its
 * altitude and the reference one - see processing/height_normalization.py.
 */
export default function HeightNormalizationPanel({ projectId, processSummary, onApplied, onError }) {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState(null);
  const [zRef, setZRef] = useState("");
  const [busy, setBusy] = useState(false);

  const ready = !!processSummary;

  const refresh = async () => {
    if (!projectId || !ready) return;
    try {
      const s = await api.getHeightNormalization(projectId);
      setStatus(s);
      setZRef((prev) => (prev === "" && s.z_ref_default_m != null ? String(s.z_ref_default_m) : prev));
    } catch (e) {
      onError && onError(e);
    }
  };

  // The altitude statistics are cheap, so they load as soon as the panel
  // is opened on processed data; re-processing resets them.
  useEffect(() => {
    if (open) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, projectId, processSummary?.n_points, processSummary?.height_normalization?.applied]);

  const apply = async (mode) => {
    try {
      setBusy(true);
      const req = { mode };
      if (mode === "apply" && zRef !== "" && Number.isFinite(Number(zRef))) req.z_ref_m = Number(zRef);
      const resp = await api.applyHeightNormalization(projectId, req);
      onApplied && (await onApplied(resp));
      await refresh();
    } catch (e) {
      onError && onError(e);
    } finally {
      setBusy(false);
    }
  };

  const alt = status?.altitude;
  const applied = !!status?.applied;
  const cs = status?.correction_stats;
  const warnings = status?.warnings || [];

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
        {open ? "▾" : "▸"} 5-1. 비행 고도 정규화 (등가층으로 한 고도면의 값으로 통일)
        {applied && <span style={{ color: "#0f766e", fontWeight: 400 }}> · 적용됨</span>}
      </button>
      {open && (
        <div style={bodyStyle}>
          <div style={{ color: "#8a7a5c", fontSize: 11, lineHeight: 1.45 }}>
            드론은 지형을 따라 날므로 측점마다 고도가 다르지만, 격자·미분·오일러·역산은 모두 측점이 한 평면 위에
            있다고 봅니다. 측점의 실제 3차원 위치에 등가 쌍극자층을 맞춰 기준 고도에서의 값으로 바꿉니다. 얕은 이상은
            고도 10 m 차이에 진폭이 30% 이상 달라집니다.
          </div>

          {!ready && <div style={{ color: "#8a7a5c" }}>자료 처리를 먼저 실행하세요.</div>}

          {ready && status && !status.available && (
            <div style={{ color: "#b45309", lineHeight: 1.45 }}>{status.reason}</div>
          )}

          {ready && status && status.available && alt && (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <div>
                측점 고도(타원체고): <b>{fmt(alt.min_m)} ~ {fmt(alt.max_m)} m</b> · 중앙값 {fmt(alt.median_m)} m · 표준편차{" "}
                {fmt(alt.std_m)} m ({alt.n?.toLocaleString()}점)
              </div>
              {status.agl && (
                <div title={`DEM(${status.agl.dem_name || ""})과 ${status.agl.altitude_column === "altitude_msl_m" ? "해수면 고도" : "타원체고"}의 차`}>
                  DEM 기준 대지 고도(AGL): 중앙값 <b>{fmt(status.agl.median_m)} m</b> ({fmt(status.agl.min_m)} ~{" "}
                  {fmt(status.agl.max_m)} m, 표준편차 {fmt(status.agl.std_m)} m)
                </div>
              )}
              {alt.range_m !== null && alt.range_m < 2 && (
                <div style={{ color: "#8a7a5c" }}>고도 변화가 2 m 미만이라 정규화 효과가 거의 없습니다.</div>
              )}

              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span>기준 고도 z_ref (m)</span>
                <input
                  type="number"
                  step="1"
                  value={zRef}
                  onChange={(e) => setZRef(e.target.value)}
                  style={inputStyle}
                  disabled={busy}
                />
                <button
                  style={{ ...buttonStyle, padding: "3px 6px", fontSize: 11, borderColor: "#ddd0b2", color: "#6b5c42" }}
                  disabled={busy}
                  onClick={() => setZRef(String(status.z_ref_default_m ?? ""))}
                  title="측점 고도의 중앙값"
                >
                  중앙값
                </button>
              </div>

              <div style={{ display: "flex", gap: 6 }}>
                <button style={{ ...buttonStyle, opacity: busy ? 0.5 : 1 }} disabled={busy} onClick={() => apply("apply")}>
                  {busy ? "계산 중... (자료가 크면 수 분)" : applied ? "다시 적용" : "정규화 적용"}
                </button>
                {applied && (
                  <button
                    style={{ ...buttonStyle, opacity: busy ? 0.5 : 1, borderColor: "#ddd0b2", color: "#6b5c42" }}
                    disabled={busy}
                    onClick={() => apply("reset")}
                  >
                    되돌리기
                  </button>
                )}
              </div>

              {applied && (
                <div style={{ display: "flex", flexDirection: "column", gap: 4, color: "#0f766e", fontSize: 11 }}>
                  <div>
                    적용됨: 기준 고도 <b>{fmt(status.z_ref_m)} m</b> · 등가층 쌍극자 {status.n_sources?.toLocaleString()}개 (
                    {(status.layers || []).map((l) => `${fmt(l.depth_below_lowest_m, 0)} m 깊이/${fmt(l.spacing_m, 0)} m 간격`).join(", ")})
                  </div>
                  <div>
                    맞춤 오차 <b>{fmt(status.fit_rms_nt, 2)} nT rms</b> (자료 변동 {fmt(status.data_std_nt, 2)} nT의{" "}
                    {status.misfit_ratio !== null ? fmt(100 * status.misfit_ratio, 0) : "-"}%) · 반복 {status.n_iter}회 ·{" "}
                    {status.n_blocks?.toLocaleString()}개 블록
                  </div>
                  {cs && (
                    <div>
                      보정량: 중앙값 <b>{fmt(cs.median_abs_nt, 2)} nT</b> · 95% {fmt(cs.p95_abs_nt, 2)} nT · 최대{" "}
                      {fmt(cs.max_abs_nt, 2)} nT
                    </div>
                  )}
                  {warnings.map((w, i) => (
                    <div key={i} style={{ color: "#b45309" }}>
                      ⚠ {w}
                    </div>
                  ))}
                </div>
              )}

              <div style={{ color: "#8a7a5c", fontSize: 11 }}>
                보정은 측점 단계에서 비행 간 레벨 보정 뒤, 구조물 모델 차감·수동 스무딩 앞에 들어가며, 되돌리기는 따로
                동작합니다. 층이 맞추지 못한 세부(측선 간격보다 짧은 성분·잡음)는 그대로 남고, 평탄한 비행은 거의 바뀌지
                않습니다. 맞춤 오차가 자료 변동의 30%를 넘으면 경고합니다.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
