import { useState } from "react";

const sectionStyle = { border: "1px solid #e5e7eb", borderRadius: 8, marginBottom: 10, background: "white" };
const summaryStyle = { padding: "10px 12px", fontWeight: 600, fontSize: 13, cursor: "pointer" };
const bodyStyle = { padding: "0 12px 12px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #d1d5db" };
const buttonStyle = {
  padding: "7px 10px",
  fontSize: 12,
  fontWeight: 600,
  borderRadius: 6,
  border: "1px solid #2563eb",
  background: "#2563eb",
  color: "white",
  cursor: "pointer",
};

function Field({ label, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ color: "#4b5563" }}>{label}</span>
      {children}
    </label>
  );
}

export default function WorkflowSteps({
  onUploadDrone,
  droneSummary,
  onUploadBase,
  baseSummary,
  processParams,
  setProcessParams,
  onProcess,
  processSummary,
  processing,
  gridCellSize,
  setGridCellSize,
  gridMethod,
  setGridMethod,
  gridMaxDistance,
  setGridMaxDistance,
  onGrid,
  gridding,
  activeTransform,
  onTransform,
  transformLoading,
  valueField,
  setValueField,
  error,
}) {
  const [droneFileNames, setDroneFileNames] = useState([]);
  const [baseFileNames, setBaseFileNames] = useState([]);

  const lp = processParams.line_params;
  const dp = processParams.diurnal_params;
  const hc = processParams.heading_correction;

  const updateLine = (key, val) => setProcessParams((p) => ({ ...p, line_params: { ...p.line_params, [key]: val } }));
  const updateDiurnal = (key, val) => setProcessParams((p) => ({ ...p, diurnal_params: { ...p.diurnal_params, [key]: val } }));
  const updateHeading = (key, val) => setProcessParams((p) => ({ ...p, heading_correction: { ...p.heading_correction, [key]: val } }));

  return (
    <div>
      {error && (
        <div style={{ ...sectionStyle, padding: 10, background: "#fef2f2", color: "#dc2626", fontSize: 12 }}>{error}</div>
      )}

      <details style={sectionStyle} open>
        <summary style={summaryStyle}>1. 드론 자력자료 업로드</summary>
        <div style={bodyStyle}>
          <input
            type="file"
            accept=".csv"
            multiple
            style={inputStyle}
            onChange={(e) => {
              const files = Array.from(e.target.files);
              if (files.length > 0) {
                setDroneFileNames(files.map((f) => f.name));
                onUploadDrone(files);
              }
            }}
          />
          <div style={{ color: "#6b7280" }}>여러 비행 파일을 함께 선택하면 하나로 합쳐 처리합니다.</div>
          {droneFileNames.length > 0 && <div style={{ color: "#6b7280" }}>{droneFileNames.join(", ")}</div>}
          {droneSummary && (
            <div style={{ color: "#374151" }}>
              포인트 수: {droneSummary.n_points}
              <br />
              시간범위: {droneSummary.time_range?.[0]} ~ {droneSummary.time_range?.[1]}
              <br />
              Mag 범위: {droneSummary.mag_range?.[0]?.toFixed(1)} ~ {droneSummary.mag_range?.[1]?.toFixed(1)} nT
            </div>
          )}
        </div>
      </details>

      <details style={sectionStyle} open>
        <summary style={summaryStyle}>2. 베이스(일변화) 자료 업로드</summary>
        <div style={bodyStyle}>
          <input
            type="file"
            accept=".csv"
            multiple
            style={inputStyle}
            onChange={(e) => {
              const files = Array.from(e.target.files);
              if (files.length > 0) {
                setBaseFileNames(files.map((f) => f.name));
                onUploadBase(files);
              }
            }}
          />
          <div style={{ color: "#6b7280" }}>여러 베이스 로그 파일을 함께 선택하면 하나로 합쳐 처리합니다.</div>
          {baseFileNames.length > 0 && <div style={{ color: "#6b7280" }}>{baseFileNames.join(", ")}</div>}
          {baseSummary && (
            <div style={{ color: "#374151" }}>
              포인트 수: {baseSummary.n_points}
              <br />
              시간범위: {baseSummary.time_range?.[0]} ~ {baseSummary.time_range?.[1]}
              <br />
              Mag 범위: {baseSummary.mag_range?.[0]?.toFixed(1)} ~ {baseSummary.mag_range?.[1]?.toFixed(1)} nT
            </div>
          )}
        </div>
      </details>

      <details style={sectionStyle} open>
        <summary style={summaryStyle}>3~5. 필터 · 측선판별 · 보정</summary>
        <div style={bodyStyle}>
          <Field label="저주파 통과 필터 차단주파수 (Hz)">
            <input
              type="number"
              step="0.1"
              style={inputStyle}
              value={processParams.filter_cutoff_hz}
              onChange={(e) => setProcessParams((p) => ({ ...p, filter_cutoff_hz: parseFloat(e.target.value) }))}
            />
          </Field>

          <Field label="측선 방향 허용오차 (deg)">
            <input type="number" style={inputStyle} value={lp.heading_tolerance_deg} onChange={(e) => updateLine("heading_tolerance_deg", parseFloat(e.target.value))} />
          </Field>
          <Field label="최소 측선 속도 (m/s) — 이보다 느리면 이착륙/정지로 간주">
            <input type="number" step="0.1" style={inputStyle} value={lp.min_speed_mps} onChange={(e) => updateLine("min_speed_mps", parseFloat(e.target.value))} />
          </Field>
          <Field label="최소 측선 길이 (m)">
            <input type="number" style={inputStyle} value={lp.min_line_length_m} onChange={(e) => updateLine("min_line_length_m", parseFloat(e.target.value))} />
          </Field>
          <Field label="터닝 여분 구간 (m) — 측선 양끝에서 추가로 자를 거리">
            <input type="number" style={inputStyle} value={lp.turn_buffer_m} onChange={(e) => updateLine("turn_buffer_m", parseFloat(e.target.value))} />
          </Field>

          <Field label="베이스 시간 오프셋 (초) — 베이스 로거 시계가 GPS와 안맞을 때 보정">
            <input type="number" style={inputStyle} value={dp.time_offset_seconds} onChange={(e) => updateDiurnal("time_offset_seconds", parseFloat(e.target.value))} />
          </Field>
          <Field label="일변화 기준값">
            <select style={inputStyle} value={dp.reference} onChange={(e) => updateDiurnal("reference", e.target.value)}>
              <option value="mean">비행시간 동안 베이스 평균</option>
              <option value="first">베이스 첫 값</option>
            </select>
          </Field>

          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={hc.enabled} onChange={(e) => updateHeading("enabled", e.target.checked)} />
            <span>헤딩(비행방향) 보정 사용 — 캘리브레이션 비행 없이 인접 반대방향 측선의 조용한 구간으로 추정</span>
          </label>
          {hc.enabled && (
            <Field label="조용한 구간 기준 백분위 (%) — 낮을수록 더 엄격하게 조용한 지점만 사용">
              <input
                type="number"
                style={inputStyle}
                value={hc.quiet_percentile}
                onChange={(e) => updateHeading("quiet_percentile", parseFloat(e.target.value))}
              />
            </Field>
          )}

          <button style={buttonStyle} disabled={processing || !droneSummary || !baseSummary} onClick={onProcess}>
            {processing ? "처리 중..." : "자료 처리 실행 (필터+측선판별+보정+IGRF)"}
          </button>

          {processSummary && (
            <div style={{ color: "#374151", marginTop: 4 }}>
              측선 {processSummary.n_lines}개 검출 / 유효 {processSummary.n_kept} / 자동제외 {processSummary.n_excluded_auto}
              <br />
              복각(Inclination): {processSummary.inclination_deg?.toFixed(2)}°, 편각(Declination): {processSummary.declination_deg?.toFixed(2)}°
              <br />
              {processSummary.heading_correction?.applied ? (
                <span>
                  헤딩 보정 오프셋: {processSummary.heading_correction.offset_nt?.toFixed(2)} nT (매칭 {processSummary.heading_correction.n_matched_pairs}쌍 중 조용한{" "}
                  {processSummary.heading_correction.n_quiet_pairs}쌍 사용)
                </span>
              ) : (
                processSummary.heading_correction?.reason && <span style={{ color: "#b45309" }}>헤딩 보정: {processSummary.heading_correction.reason}</span>
              )}
              <br />
              {processSummary.diurnal && !processSummary.diurnal.has_overlap && (
                <div style={{ color: "#dc2626", marginTop: 4 }}>
                  ⚠ 베이스 자료와 드론 비행시간이 겹치지 않습니다. 베이스 시간범위: {processSummary.diurnal.base_time_range?.[0]} ~{" "}
                  {processSummary.diurnal.base_time_range?.[1]}. 시간 오프셋을 조정하세요.
                </div>
              )}
            </div>
          )}
        </div>
      </details>

      <details style={sectionStyle} open>
        <summary style={summaryStyle}>6, 8. 지도 표시 값</summary>
        <div style={bodyStyle}>
          <Field label="컬러로 표시할 값">
            <select style={inputStyle} value={valueField} onChange={(e) => setValueField(e.target.value)}>
              <option value="tmi">TMI (보정된 전자력)</option>
              <option value="anomaly">자력 이상 (TMI - IGRF)</option>
            </select>
          </Field>
        </div>
      </details>

      <details style={sectionStyle}>
        <summary style={summaryStyle}>9. 그리딩</summary>
        <div style={bodyStyle}>
          <Field label="셀 크기 (m)">
            <input type="number" style={inputStyle} value={gridCellSize} onChange={(e) => setGridCellSize(parseFloat(e.target.value))} />
          </Field>
          <Field label="보간 방법">
            <select style={inputStyle} value={gridMethod} onChange={(e) => setGridMethod(e.target.value)}>
              <option value="nearest">원본 셀 (가장 빠름, 기본값)</option>
              <option value="linear">선형(Linear) - 보통 속도</option>
              <option value="cubic">큐빅(Cubic) - 느림</option>
              <option value="spline">스플라인 (가장 부드러움, 가장 느림)</option>
            </select>
          </Field>
          <Field label="보간 반경 (m) — 비워두면 측선 간격 기반 자동 계산">
            <input
              type="number"
              style={inputStyle}
              placeholder={processSummary?.line_spacing_m ? `자동: ${Math.round(Math.max(2 * gridCellSize, 0.6 * processSummary.line_spacing_m))}` : "자동"}
              value={gridMaxDistance ?? ""}
              onChange={(e) => setGridMaxDistance(e.target.value === "" ? null : parseFloat(e.target.value))}
            />
          </Field>
          {processSummary?.line_spacing_m && (
            <div style={{ color: "#6b7280" }}>추정 측선 간격: {processSummary.line_spacing_m.toFixed(1)}m</div>
          )}
          <button style={buttonStyle} disabled={gridding || !processSummary} onClick={() => onGrid()}>
            {gridding ? "그리딩 중..." : "그리드 생성"}
          </button>
        </div>
      </details>

      <details style={sectionStyle}>
        <summary style={summaryStyle}>10. 파생 그리드 (RTP / RTE / AS / 1VD)</summary>
        <div style={bodyStyle}>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {[
              ["none", "그리드(원본)"],
              ["rtp", "RTP"],
              ["rte", "RTE"],
              ["1vd", "1VD"],
              ["as", "AS"],
            ].map(([key, label]) => (
              <button
                key={key}
                onClick={() => onTransform(key)}
                disabled={transformLoading || !processSummary}
                style={{
                  ...buttonStyle,
                  background: activeTransform === key ? "#2563eb" : "white",
                  color: activeTransform === key ? "white" : "#2563eb",
                }}
              >
                {label}
              </button>
            ))}
          </div>
          {transformLoading && <div style={{ color: "#6b7280" }}>계산 중...</div>}
        </div>
      </details>

      <details style={sectionStyle}>
        <summary style={summaryStyle}>11. 배경지도</summary>
        <div style={bodyStyle}>지도 우측 상단 레이어 컨트롤에서 OSM / Esri 위성 / Google 위성을 전환할 수 있습니다.</div>
      </details>
    </div>
  );
}
