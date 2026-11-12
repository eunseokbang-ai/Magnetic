import { useEffect, useState } from "react";
import * as api from "../api";

const sectionStyle = { border: "1px solid #e6dac0", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const buttonStyle = {
  padding: "7px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #a9631f", background: "white", color: "#a9631f", cursor: "pointer",
};
const inputStyle = { width: 70, padding: "4px 6px", fontSize: 12, border: "1px solid #ddd0b2", borderRadius: 4 };

const fmt = (v, d = 1) => (v === null || v === undefined ? "-" : Number(v).toFixed(d));
const signed = (v, d = 1) => (v === null || v === undefined ? "-" : `${v >= 0 ? "+" : ""}${Number(v).toFixed(d)}`);
const clock = (seconds) => new Date(seconds * 1000).toISOString().slice(5, 16).replace("T", " ");

/**
 * The sensor's GPS height, anchored to the ground it stood on. Before
 * takeoff and after landing the sensor lies on the ground, where a terrain
 * model says how high that is; the difference is the GPS error at that
 * moment. The backend finds those moments, fits how the error wanders in
 * time and corrects every sample - see magcore/sensor_height.py. Changes
 * heights only, never a magnetic value.
 */
export default function SensorHeightPanel({ projectId, processSummary, onApplied, onError }) {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState(null);
  const [file, setFile] = useState(null);
  const [offset, setOffset] = useState("0");
  const [excludeSea, setExcludeSea] = useState(true);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);

  const ready = !!processSummary;
  const applied = !!processSummary?.sensor_height?.applied;

  const refresh = async () => {
    if (!projectId || !ready) return;
    try {
      setStatus(await api.getSensorHeight(projectId));
    } catch (e) {
      onError && onError(e);
    }
  };

  useEffect(() => {
    if (open) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, projectId, processSummary?.n_points, applied]);

  const run = async (mode) => {
    try {
      setBusy(true);
      setNotice(null);
      const resp = await api.applySensorHeight(projectId, {
        mode, file: mode === "apply" ? file : null, resting_offset_m: Number(offset) || 0, exclude_sea_level: excludeSea,
      });
      setStatus(resp.sensor_height_status);
      const notes = [];
      if (resp.height_normalization_reset) notes.push("비행 고도 정규화가 이전 고도로 맞춰져 있어 해제했습니다. 다시 적용하세요.");
      if (resp.inversion_stale) notes.push("화면의 역산 결과는 이전 고도로 계산된 것입니다. 다시 실행하세요.");
      setNotice(notes.join(" ") || null);
      onApplied && (await onApplied(resp));
    } catch (e) {
      onError && onError(e);
    } finally {
      setBusy(false);
    }
  };

  const s = status;
  const validation = s?.validation || [];
  const far = validation.find((v) => v.exclude_within_min > 0);
  const near = validation.find((v) => v.exclude_within_min === 0);
  const excluded = s?.n_checks_excluded || {};

  return (
    <div style={sectionStyle} data-testid="sensor-height-panel">
      <button
        onClick={() => setOpen((v) => !v)}
        style={{ width: "100%", textAlign: "left", padding: "8px 12px", border: "none", background: "none", fontSize: 12, fontWeight: 700, color: "#4a3d28", cursor: "pointer" }}
      >
        {open ? "▾" : "▸"} 5-0. 센서 고도 보정 (이착륙 때 지면 높이로 GPS 고도를 맞춤)
        {applied && <span style={{ color: "#0f766e", fontWeight: 400 }}> · 적용됨</span>}
      </button>
      {open && (
        <div style={bodyStyle}>
          <div style={{ color: "#8a7a5c", fontSize: 11, lineHeight: 1.45 }}>
            견인 센서의 GPS 고도는 수 m에서 10 m 넘게 틀리고, 그 오차가 수십 분에 걸쳐 천천히 변합니다(가사도: 평균 +9.7 m,
            ±9 m). 이륙 전·착륙 후 센서가 지면에 놓여 있을 때는 지형 자료로 참값을 알 수 있으므로, 그 순간들의 오차로
            비행 전체의 고도를 보정합니다. 자력값은 바꾸지 않고 고도만 바꿉니다 — 역산의 관측점 높이와 비행 고도 정규화가
            보정된 고도를 씁니다.
          </div>

          {!ready && <div style={{ color: "#8a7a5c" }}>자료 처리를 먼저 실행하세요.</div>}
          {ready && s && !s.available && <div style={{ color: "#b45309", lineHeight: 1.45 }}>{s.reason}</div>}

          {ready && s && s.available && (
            <>
              <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                <div title="이착륙 지점의 표고가 정확해야 합니다. 5 m급 DEM은 해안의 이착륙장을 해수면(0 m)으로 지워 놓은 경우가 많아, 가능하면 정사영상과 함께 만든 DSM을 쓰세요. 보정을 적용한 뒤에는 이 파일이 없어도 됩니다(검사값이 프로젝트에 저장됨).">
                  지형 자료 (DEM/DSM GeoTIFF) ⓘ
                  <input
                    type="file"
                    accept=".tif,.tiff"
                    data-testid="sensor-height-file"
                    style={{ display: "block", marginTop: 3, fontSize: 11 }}
                    onChange={(e) => setFile(e.target.files[0] || null)}
                  />
                  <div style={{ fontSize: 11, color: "#8a7a5c", marginTop: 2 }}>
                    {file ? `선택: ${file.name}` : s.has_dem ? `선택하지 않으면 역산용 DEM(${s.dem_name})을 씁니다.` : "역산용 DEM이 없으므로 파일을 선택해야 합니다."}
                  </div>
                </div>
                <label style={{ display: "flex", alignItems: "center", gap: 6 }} title="지면에 놓였을 때 센서(GPS 안테나)가 지면에서 떠 있는 높이. 바닥에 그대로 놓으면 0.">
                  지면에 놓였을 때 센서 높이 (m) ⓘ
                  <input type="number" step="0.1" style={inputStyle} value={offset} onChange={(e) => setOffset(e.target.value)} />
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 6 }} title="지형 자료가 이착륙 지점을 정확히 0 m로 보고 있으면 해안을 바다로 지운 것일 가능성이 높아, 그 검사는 쓰지 않습니다.">
                  <input type="checkbox" checked={excludeSea} onChange={(e) => setExcludeSea(e.target.checked)} />
                  지형 자료가 해수면(0 m)으로 보는 지점의 검사는 제외 ⓘ
                </label>
              </div>

              <div style={{ display: "flex", gap: 6 }}>
                <button data-testid="sensor-height-apply" style={{ ...buttonStyle, flex: 1, opacity: busy ? 0.6 : 1 }} disabled={busy || (!file && !s.has_dem)} onClick={() => run("apply")}>
                  {busy ? "계산 중..." : applied ? "다시 적용" : "센서 고도 보정 적용"}
                </button>
                {applied && (
                  <button data-testid="sensor-height-reset" style={{ ...buttonStyle, borderColor: "#ddd0b2", color: "#4a3d28", opacity: busy ? 0.6 : 1 }} disabled={busy} onClick={() => run("reset")}>
                    해제
                  </button>
                )}
              </div>

              {notice && <div style={{ color: "#92400e", background: "#fffbeb", padding: 6, borderRadius: 4, lineHeight: 1.45 }}>⚠ {notice}</div>}

              {s.applied && (
                <div data-testid="sensor-height-result" style={{ display: "flex", flexDirection: "column", gap: 4, borderTop: "1px dashed #ddd0b2", paddingTop: 6, lineHeight: 1.5 }}>
                  <div>
                    지상 검사 <b>{s.n_checks_used}</b>개 사용 ({s.n_flights}개 비행, {s.terrain_name})
                    {Object.keys(excluded).length > 0 && (
                      <span style={{ color: "#92400e" }}>
                        {" "}· 제외 {Object.entries(excluded).map(([k, n]) => `${n}개(${s.excluded_labels?.[k] || k})`).join(", ")}
                      </span>
                    )}
                  </div>
                  <div>
                    지면에서 본 GPS 오차: 중앙값 <b>{signed(s.check_error_m?.median)} m</b> ({signed(s.check_error_m?.min)} ~ {signed(s.check_error_m?.max)} m)
                  </div>
                  <div>
                    측선 위 보정량: 중앙값 <b>{signed(-s.correction_on_lines_m?.median)} m</b> ({signed(-s.correction_on_lines_m?.max)} ~ {signed(-s.correction_on_lines_m?.min)} m)
                  </div>
                  {s.height_above_terrain_m && (
                    <div title="검사에 쓴 지형 자료 기준. 임무에서 설정한 대지 고도(센서 기준)와 비교해 보세요.">
                      측선 위 대지 고도 중앙값: {fmt(s.height_above_terrain_m.before_median)} m → <b>{fmt(s.height_above_terrain_m.after_median)} m</b>{" "}
                      (보정 후 5–95%: {fmt(s.height_above_terrain_m.after_p5)} ~ {fmt(s.height_above_terrain_m.after_p95)} m) ⓘ
                    </div>
                  )}
                  <div title="보정 뒤에도 모르는 채로 남는 고도 오차(1σ). 이착륙 근처에서 작고 비행 중간에서 큽니다.">
                    남는 불확실성(1σ): 측선 위 중앙값 <b>{fmt(s.sigma_on_lines_m?.median)} m</b>, 95% {fmt(s.sigma_on_lines_m?.p95)} m (보정 전 약 {fmt(s.uncorrected_rms_m)} m) ⓘ
                  </div>
                  <div style={{ color: "#6b5c42" }}>
                    오차 모델: 평균 {signed(s.model?.mean_m)} m, 변동 ±{fmt(Math.sqrt(s.model?.sill_m2 || 0))} m, 상관 시간 {fmt(s.model?.tau_min, 0)}분
                    {s.model?.fitted ? " (이 탐사의 검사로 맞춤)" : ` (검사가 ${s.model?.n_checks}개뿐이라 변동·상관 시간은 기본값)`}
                  </div>
                  {near && far && (
                    <div style={{ color: "#6b5c42" }} title="검사 하나를 빼고 나머지로 그 값을 맞혀 본 오차(rms). '5분 이상 떨어진 검사만'이 비행 중간에 가까운 조건입니다.">
                      검증(검사를 하나씩 빼고 예측): 보정 안 함 {fmt(near.rms_uncorrected_m)} m · 평균만 {fmt(near.rms_mean_only_m)} m · 이 보정 {fmt(near.rms_kriged_m)} m
                      (5분 이상 떨어진 검사만 쓰면 {fmt(far.rms_kriged_m)} m) ⓘ
                    </div>
                  )}
                  {s.flights_without_checks?.length > 0 && (
                    <div style={{ color: "#92400e" }}>
                      검사가 없는 비행 {s.flights_without_checks.length}개는 앞뒤 비행의 검사와 평균으로만 보정됩니다: {s.flights_without_checks.slice(0, 6).join(", ")}
                      {s.flights_without_checks.length > 6 ? " ..." : ""}
                    </div>
                  )}
                  {s.geoid_separation_m != null && (
                    <div style={{ color: "#6b5c42" }}>
                      역산에서 DEM을 쓸 때 "DEM 지오이드 보정"에는 <b>{fmt(s.geoid_separation_m, 2)} m</b>(이 GPS의 지오이드 높이)를 넣으세요.
                    </div>
                  )}
                  {s.checks?.length > 0 && (
                    <details>
                      <summary style={{ cursor: "pointer", color: "#6b5c42" }}>지상 검사 {s.checks.length}개 보기</summary>
                      <div style={{ maxHeight: 200, overflowY: "auto", border: "1px solid #e6dac0", borderRadius: 6, marginTop: 4 }}>
                        <table style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
                          <thead>
                            <tr style={{ background: "#f7f2e6", textAlign: "right" }}>
                              <th style={{ textAlign: "left", padding: "2px 6px" }}>비행</th>
                              <th style={{ padding: "2px 6px" }}>시각(UTC)</th>
                              <th style={{ padding: "2px 6px" }}>GPS</th>
                              <th style={{ padding: "2px 6px" }}>지면</th>
                              <th style={{ padding: "2px 6px" }}>오차</th>
                            </tr>
                          </thead>
                          <tbody>
                            {s.checks.map((c, i) => (
                              <tr key={i} style={{ textAlign: "right", color: c.used ? "#4a3d28" : "#b0a58c", borderTop: "1px solid #f3ecd9" }} title={c.excluded_label || ""}>
                                <td style={{ textAlign: "left", padding: "2px 6px", whiteSpace: "nowrap" }}>
                                  {(c.file || `비행 ${c.flight + 1}`).replace(/^SRVY0-/, "").replace(/-10Hz.*$/, "")} {c.kind === "takeoff" ? "이륙" : "착륙"}
                                </td>
                                <td style={{ padding: "2px 6px" }}>{clock(c.time_s)}</td>
                                <td style={{ padding: "2px 6px" }}>{fmt(c.gps_height_m)}</td>
                                <td style={{ padding: "2px 6px" }}>{fmt(c.ground_m)}</td>
                                <td style={{ padding: "2px 6px" }}>{c.used ? signed(c.error_m) : "제외"}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </details>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
