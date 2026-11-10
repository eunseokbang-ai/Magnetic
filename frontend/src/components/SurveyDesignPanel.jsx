import { useState } from "react";
import * as api from "../api";

const bodyStyle = { display: "flex", flexDirection: "column", gap: 8, fontSize: 12, padding: "8px 0" };
const inputStyle = { fontSize: 12, padding: "4px 6px", borderRadius: 6, border: "1px solid #ddd0b2", width: "100%", boxSizing: "border-box" };
const buttonStyle = { padding: "7px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #a9631f", background: "white", color: "#a9631f", cursor: "pointer" };
const disabledStyle = { ...buttonStyle, borderColor: "#ddd0b2", color: "#ab9a78", cursor: "default" };
const gridStyle = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 };
const boxStyle = { padding: "6px 8px", borderRadius: 6, background: "#f7f2e6", border: "1px solid #e6dac0" };

const num = (v) => (v === "" || v == null ? null : Number(v));

function NumberField({ label, value, onChange, step = "any", placeholder }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ color: "#6b5c42" }}>{label}</span>
      <input type="number" step={step} style={inputStyle} value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}

/**
 * Before the flight: how far apart to fly the lines for the product that
 * is wanted, whether the target will show at that height, and a flight
 * plan (survey lines, tie lines, a repeated line) over an area, as a map
 * preview and a KML for the pilot's planning software.
 * See magcore/survey_design.py for where the numbers come from.
 */
export default function SurveyDesignPanel({ mapBounds, boundaryPolygon, onPreview }) {
  const [depth, setDepth] = useState("20");
  const [height, setHeight] = useState("30");
  const [product, setProduct] = useState("first_derivative");
  const [radius, setRadius] = useState("");
  const [chi, setChi] = useState("");
  const [noise, setNoise] = useState("0.5");
  const [speed, setSpeed] = useState("5");
  const [advice, setAdvice] = useState(null);

  const [areaSource, setAreaSource] = useState(boundaryPolygon ? "boundary" : "view");
  const [spacing, setSpacing] = useState("");
  const [azimuth, setAzimuth] = useState("");
  const [endurance, setEndurance] = useState("25");
  const [repeatLine, setRepeatLine] = useState(true);
  const [plan, setPlan] = useState(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const run = async (fn) => {
    try {
      setError(null);
      setBusy(true);
      await fn();
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  const recommend = () =>
    run(async () => {
      const resp = await api.recommendSurveyDesign({
        target_depth_m: num(depth),
        flight_height_m: num(height),
        product,
        noise_nt: num(noise) ?? 0.5,
        speed_mps: num(speed) ?? 5,
        target_radius_m: num(radius),
        susceptibility_si: num(chi),
      });
      setAdvice(resp);
      setSpacing(String(resp.line_spacing_m));
    });

  const polygon = () => {
    if (areaSource === "boundary" && boundaryPolygon?.length >= 3) return boundaryPolygon;
    if (!mapBounds) return null;
    const [s, w, n, e] = [mapBounds.getSouth(), mapBounds.getWest(), mapBounds.getNorth(), mapBounds.getEast()];
    return [[s, w], [s, e], [n, e], [n, w]];
  };

  const planRequest = () => ({
    polygon: polygon(),
    line_spacing_m: num(spacing),
    azimuth_deg: num(azimuth),
    speed_mps: num(speed) ?? 5,
    endurance_min: num(endurance) ?? 25,
    repeat_line: repeatLine,
  });

  const makePlan = () =>
    run(async () => {
      const resp = await api.planFlight(planRequest());
      setPlan(resp);
      onPreview && onPreview(resp.layer);
    });

  const canPlan = !busy && num(spacing) > 0 && !!polygon();
  const d = advice?.detectability;

  return (
    <div style={bodyStyle}>
      <div style={{ color: "#8a7a5c" }}>
        측선이 너무 성기거나 타이라인·재비행이 없는 것은 탐사가 끝난 뒤에는 처리로 고칠 수 없습니다. 비행 전에 여기서 정합니다.
      </div>

      <b>① 측선 간격</b>
      <div style={gridStyle}>
        <NumberField label="가장 얕은 목표의 심도 (m, 지표 아래)" value={depth} onChange={setDepth} />
        <NumberField label="비행 고도 (m, 지표 위)" value={height} onChange={setHeight} />
      </div>
      <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <span style={{ color: "#6b5c42" }}>만들려는 산출물</span>
        <select style={inputStyle} value={product} onChange={(e) => setProduct(e.target.value)}>
          <option value="field">자력 이상도만 (TMI · RTP)</option>
          <option value="first_derivative">1차 미분까지 (구배 · 해석신호 · 틸트)</option>
          <option value="second_derivative">2차 미분까지 (2차 수직미분 · 오일러)</option>
        </select>
      </label>
      <details>
        <summary style={{ cursor: "pointer", color: "#6b5c42" }}>표적이 보일지도 확인 (선택: 크기와 자화율)</summary>
        <div style={{ ...gridStyle, marginTop: 6 }}>
          <NumberField label="표적 반경 (m)" value={radius} onChange={setRadius} placeholder="예: 5" />
          <NumberField label="자화율 대비 (SI)" value={chi} onChange={setChi} placeholder="예: 0.1" />
          <NumberField label="자료 잡음 (nT)" value={noise} onChange={setNoise} />
          <NumberField label="비행 속도 (m/s)" value={speed} onChange={setSpeed} />
        </div>
      </details>
      <button style={busy ? disabledStyle : buttonStyle} disabled={busy} onClick={recommend}>
        권장 간격 계산
      </button>
      {advice && (
        <div style={boxStyle}>
          <div>
            <b>측선 간격 {advice.line_spacing_m} m 이하</b> (센서–소스 거리 {advice.source_distance_m} m의 {advice.spacing_over_distance}배, 에일리어싱 {advice.alias_tolerance * 100}% 기준)
          </div>
          <div style={{ color: "#6b5c42" }}>
            산출물별: 이상도 {advice.spacing_by_product.field} m · 1차 미분 {advice.spacing_by_product.first_derivative} m · 2차 미분 {advice.spacing_by_product.second_derivative} m
          </div>
          <div style={{ color: "#6b5c42" }}>
            타이라인 간격 약 {advice.tie_line_spacing_m} m · 이상 폭 약 {advice.anomaly_width_m} m · 측선 방향 샘플 {advice.along_line_sample_m} m
          </div>
          {d && (
            <div style={{ marginTop: 3, color: d.detectable ? "#166534" : "#b45309" }}>
              예상 이상 {d.peak_anomaly_nt} nT (잡음의 {d.snr}배) — {d.detectable ? "보입니다" : "보이지 않을 수 있습니다"}. 비행 고도 {d.max_flight_height_m} m까지 잡음의 {d.min_snr}배를 넘습니다.
            </div>
          )}
          {advice.warnings.map((w, i) => (
            <div key={i} style={{ color: "#b45309", marginTop: 3 }}>⚠ {w}</div>
          ))}
        </div>
      )}

      <b style={{ marginTop: 4 }}>② 비행 계획</b>
      <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <span style={{ color: "#6b5c42" }}>조사 구역</span>
        <select style={inputStyle} value={areaSource} onChange={(e) => setAreaSource(e.target.value)}>
          <option value="view">지금 지도에 보이는 범위</option>
          <option value="boundary" disabled={!(boundaryPolygon?.length >= 3)}>
            이 프로젝트의 표시 경계{boundaryPolygon?.length >= 3 ? "" : " (없음)"}
          </option>
        </select>
      </label>
      <div style={gridStyle}>
        <NumberField label="측선 간격 (m)" value={spacing} onChange={setSpacing} placeholder="①에서 계산" />
        <NumberField label="측선 방위 (°, 비우면 긴 변 방향)" value={azimuth} onChange={setAzimuth} placeholder="자동" />
        <NumberField label="1회 비행 시간 (분)" value={endurance} onChange={setEndurance} />
        <label style={{ display: "flex", alignItems: "center", gap: 6, alignSelf: "end", paddingBottom: 4 }}>
          <input type="checkbox" checked={repeatLine} onChange={(e) => setRepeatLine(e.target.checked)} />
          첫 측선 재비행 포함
        </label>
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        <button style={canPlan ? buttonStyle : disabledStyle} disabled={!canPlan} onClick={makePlan}>
          {busy ? "계산 중..." : "비행 계획 만들기"}
        </button>
        <button style={canPlan && plan ? buttonStyle : disabledStyle} disabled={!canPlan || !plan} onClick={() => run(() => api.downloadFlightPlanKml(planRequest(), "flight_plan.kml"))}>
          KML 내려받기
        </button>
      </div>
      {plan && (
        <div style={boxStyle}>
          <div>
            <b>측선 {plan.n_survey_lines}개</b> (방위 {plan.azimuth_deg}°, 간격 {plan.line_spacing_m} m) · 타이라인 {plan.n_tie_lines}개 ({plan.tie_spacing_m} m)
            {plan.has_repeat_line && " · 재비행 1개"}
          </div>
          <div style={{ color: "#6b5c42" }}>
            총 {(plan.length_m.total / 1000).toFixed(1)} km · 약 {plan.flight_time_min}분 · 비행 {plan.n_sorties}회 · 구역 {plan.area_km2} km²
          </div>
          <div style={{ color: "#8a7a5c" }}>지도에 "비행 계획" 참조 레이어로 추가했습니다. 지형 추종 고도와 장애물 회피는 비행 계획 프로그램에서 정하세요.</div>
          {plan.warnings.map((w, i) => (
            <div key={i} style={{ color: "#b45309", marginTop: 3 }}>⚠ {w}</div>
          ))}
        </div>
      )}
      {error && <div style={{ color: "#dc2626" }}>{error}</div>}
    </div>
  );
}
