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

const fmt = (v, d = 1) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));

/**
 * The readings brought to one flight height. A terrain-following drone
 * reads every anomaly at a different distance, and everything after the
 * point stage (the derived grids, Euler, the inversion) takes the survey
 * as one level surface. The backend fits an equivalent source layer at
 * the readings' true 3D positions and adds the height change of the
 * modelled field to each reading - see processing/height_normalization.py.
 */
export default function HeightNormalizationPanel({ projectId, processSummary, onApplied, onError }) {
  const [open, setOpen] = useState(true);
  const [info, setInfo] = useState(null);
  const [zRef, setZRef] = useState("");
  const [busy, setBusy] = useState(false);

  const ready = !!processSummary;
  const applied = processSummary?.height_normalization?.applied ? processSummary.height_normalization : null;

  const analyze = async () => {
    try {
      setBusy(true);
      const resp = await api.analyzeHeightNormalization(projectId);
      setInfo(resp);
      if (zRef === "" && resp.suggested_z_ref_m !== null && resp.suggested_z_ref_m !== undefined) {
        setZRef(String(resp.suggested_z_ref_m));
      }
    } catch (e) {
      if (onError) onError(e);
    } finally {
      setBusy(false);
    }
  };

  // The altitude statistics are cheap and describe the flight itself,
  // so they are shown as soon as there is a processed survey.
  useEffect(() => {
    setInfo(null);
    setZRef("");
    if (projectId && ready) analyze();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, ready, processSummary?.n_points]);

  const apply = async (mode) => {
    try {
      setBusy(true);
      const body = { mode };
      if (mode === "apply" && zRef !== "" && Number.isFinite(Number(zRef))) body.z_ref_m = Number(zRef);
      const resp = await api.applyHeightNormalization(projectId, body);
      if (onApplied) await onApplied(resp);
      setInfo(await api.analyzeHeightNormalization(projectId));
    } catch (e) {
      if (onError) onError(e);
    } finally {
      setBusy(false);
    }
  };

  const alt = info?.altitude;
  const agl = info?.agl;
  const corr = applied?.correction;
  const poorFit = applied && applied.fit_ratio !== null && applied.fit_ratio > 0.3;

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
        {open ? "▾" : "▸"} 비행 고도 정규화 (등가 소스층)
        {applied && <span style={{ color: "#0f766e", fontWeight: 400 }}> · 적용됨</span>}
      </button>
      {open && (
        <div style={bodyStyle}>
          <div style={{ color: "#8a7a5c", fontSize: 11, lineHeight: 1.45 }}>
            드론은 지형을 따라 날므로 측점마다 고도가 다르고, 얕은 이상은 고도 10 m 차이에 진폭이 30% 넘게
            달라집니다. 파생 그리드·오일러·역산은 모두 한 높이에서 잰 값을 전제하므로, 측점의 실제 3차원 위치에
            등가 소스층을 맞춰 <b>기준 고도면에서의 값</b>으로 바꿉니다. 층이 재현하지 못한 성분은 그대로 남기고,
            측정된 값은 모델의 고도 변화량만큼만 움직입니다.
          </div>

          {!ready && <div style={{ color: "#8a7a5c" }}>자료 처리를 먼저 실행하세요.</div>}

          {info && !info.available && <div style={{ color: "#b45309", lineHeight: 1.45 }}>{info.reason}</div>}

          {alt && alt.n > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <div>
                측점 고도(타원체고) <b>{fmt(alt.min)}~{fmt(alt.max)} m</b> · 중앙값 {fmt(alt.median)} m · 표준편차{" "}
                <b>{fmt(alt.std)} m</b>
              </div>
              {agl && agl.n > 0 && (
                <div title={agl.note}>
                  DEM 위 높이(AGL) {fmt(agl.min, 0)}~{fmt(agl.max, 0)} m · 중앙값 {fmt(agl.median, 0)} m · 표준편차{" "}
                  {fmt(agl.std, 1)} m{agl.datum === "ellipsoidal" ? " (지오이드 미보정)" : ""}
                </div>
              )}
              {alt.std < 1.0 && (
                <div style={{ color: "#8a7a5c", fontSize: 11 }}>
                  고도 변화가 GPS 잡음 수준입니다 — 이 비행은 이미 평면에 가깝고, 정규화는 거의 아무것도 바꾸지 않습니다.
                </div>
              )}
            </div>
          )}

          {info?.available && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
              <label htmlFor="height-z-ref">기준 고도 z_ref (m)</label>
              <input
                id="height-z-ref"
                type="number"
                step="1"
                value={zRef}
                onChange={(e) => setZRef(e.target.value)}
                style={inputStyle}
                disabled={busy}
              />
              <button
                type="button"
                style={{ ...buttonStyle, padding: "3px 6px", fontSize: 11, borderColor: "#ddd0b2", color: "#6b5c42" }}
                disabled={busy || !alt}
                onClick={() => setZRef(String(alt.median))}
                title="측점 고도의 중앙값. 가장 적은 자료를 가장 적게 움직입니다."
              >
                중앙값
              </button>
              <button
                type="button"
                style={{ ...buttonStyle, padding: "3px 6px", fontSize: 11, borderColor: "#ddd0b2", color: "#6b5c42" }}
                disabled={busy || !alt}
                onClick={() => setZRef(String(alt.max))}
                title="측점 최고 고도. 모든 측점을 위로만 옮기므로 안정적이지만, 세부가 그만큼 부드러워집니다."
              >
                최고
              </button>
            </div>
          )}

          {info?.available && (
            <div style={{ color: "#8a7a5c", fontSize: 11, lineHeight: 1.45 }}>
              기준면보다 높은 측점은 <b>하향연속</b>이 되어 층이 놓친 짧은 파장 오차가 커집니다. 관심 이상체가 높은 지형
              아래에 있으면 기준 고도를 높게 잡으세요. 테두리 한 측선간격 구간은 층 바깥에 자료가 없어 보정을 0으로
              줄입니다.
            </div>
          )}

          <div style={{ display: "flex", gap: 6 }}>
            <button
              style={{ ...buttonStyle, opacity: ready && info?.available && !busy ? 1 : 0.5 }}
              disabled={!ready || !info?.available || busy}
              onClick={() => apply("apply")}
            >
              {busy ? "계산 중..." : applied ? "다시 정규화" : "정규화 적용"}
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
          {busy && (
            <div style={{ color: "#8a7a5c", fontSize: 11 }}>
              측점 수만 개와 소스 수만 개의 켤레기울기 풀이입니다 — 해남 규모(96만 점)에서 2~3분 걸립니다.
            </div>
          )}

          {applied && (
            <div style={{ display: "flex", flexDirection: "column", gap: 4, color: "#0f766e" }}>
              <div>
                적용됨: 기준 고도 <b>{fmt(applied.z_ref_m)} m</b> · 등가층 {applied.n_sources}개 소스 (간격{" "}
                {fmt(applied.source_spacing_m, 0)} m, 최저 측점 아래 {fmt(applied.layer_depth_below_min_m, 0)} m)
              </div>
              <div>
                맞춤 오차 <b>{fmt(applied.fit_rms_nt, 2)} nT</b> / 자료 변동 {fmt(applied.data_rms_nt, 1)} nT (
                {applied.fit_ratio !== null ? `${(applied.fit_ratio * 100).toFixed(1)}%` : "–"}) · 반복{" "}
                {applied.n_iterations}회{applied.converged ? "" : " (미수렴)"}
              </div>
              {corr && (
                <div>
                  보정량 중앙값 <b>{fmt(corr.median_abs_nt, 2)} nT</b> · 최대 {fmt(corr.max_abs_nt, 1)} nT · rms{" "}
                  {fmt(corr.rms_nt, 2)} nT
                  {corr.n_points_edge_tapered ? ` · 테두리 테이퍼 ${corr.n_points_edge_tapered.toLocaleString()}점` : ""}
                  {corr.n_points_uncorrected ? ` · 고도 없음 ${corr.n_points_uncorrected.toLocaleString()}점` : ""}
                </div>
              )}
              {(applied.warnings || []).map((w, i) => (
                <div key={i} style={{ color: poorFit || !applied.converged ? "#b45309" : "#8a7a5c", lineHeight: 1.45 }}>
                  ⚠ {w}
                </div>
              ))}
              <div style={{ color: "#8a7a5c", fontSize: 11 }}>
                보정은 비행 간 레벨 보정 다음, 구조물 모델 차감·수동 스무딩 앞에 들어가고, 되돌리기는 따로 동작합니다.
                파이프라인을 다시 실행하면 해제됩니다.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
