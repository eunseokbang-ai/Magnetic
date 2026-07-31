import { useState } from "react";
import InfoIcon from "./InfoIcon";

const TRANSFORM_INFO = {
  none: "그리딩된 원본 자력 이상값을 그대로 표시합니다.",
  rtp: "RTP(Reduction to Pole, 자극환산) — 비스듬한 지자기장 방향 때문에 이상체 바로 위가 아닌 옆으로 치우쳐 보이는 이상대를, 자극(수직 자기장)에서 측정한 것처럼 보정해 이상체 바로 위에 오도록 만듭니다.",
  rte: "RTE(Reduction to Equator, 자기적도환산) — 저위도(적도 부근)처럼 복각이 낮아 RTP가 불안정한 지역에서 대신 사용하는 유사한 보정 기법입니다.",
  "1vd": "1VD(수직 1차 미분) — 값 자체가 아니라 수직 방향 변화율을 보여줘, 얕고 경계가 뚜렷한 이상체를 더 선명하게 강조합니다.",
  as: "AS(Analytic Signal, 해석 신호 진폭) — 자화 방향에 무관하게 이상체 바로 위에서 극대값을 갖는 값으로, 자화 방향을 모를 때도 이상체 위치를 판단하기 좋습니다.",
};

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

function ProgressBar({ fraction }) {
  if (fraction == null) return null;
  const pct = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, height: 6, borderRadius: 3, background: "#e5e7eb", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: "#2563eb", transition: "width 0.15s" }} />
      </div>
      <span style={{ color: "#6b7280", fontSize: 11, minWidth: 32, textAlign: "right" }}>{pct}%</span>
    </div>
  );
}

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
  droneUploadProgress,
  onUploadBase,
  baseSummary,
  baseUploadProgress,
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
  gridOpacity,
  setGridOpacity,
  hillshade,
  setHillshade,
  hillshadeAzimuth,
  setHillshadeAzimuth,
  hillshadeAltitude,
  setHillshadeAltitude,
  hillshadeExaggeration,
  setHillshadeExaggeration,
  stretch,
  setStretch,
  showContours,
  setShowContours,
  contourInterval,
  setContourInterval,
  contourNLevels,
  setContourNLevels,
  onGrid,
  gridding,
  activeTransform,
  onTransform,
  transformLoading,
  valueField,
  setValueField,
  onExportGeotiff,
  exportingGeotiff,
  exportGeotiffColored,
  setExportGeotiffColored,
  error,
}) {
  const [droneFileNames, setDroneFileNames] = useState([]);
  const [baseFileNames, setBaseFileNames] = useState([]);

  const lp = processParams.line_params;
  const dsp = processParams.despike_params;
  const updateDespike = (key, val) => setProcessParams((p) => ({ ...p, despike_params: { ...p.despike_params, [key]: val } }));
  const cl = processParams.crossover_leveling;
  const updateCrossover = (key, val) => setProcessParams((p) => ({ ...p, crossover_leveling: { ...p.crossover_leveling, [key]: val } }));
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
          <ProgressBar fraction={droneUploadProgress} />
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
          <ProgressBar fraction={baseUploadProgress} />
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

          <Field label="GPS-자력계 시간 오프셋 (초) — 자력계 내부 지연으로 위치가 실제와 어긋날 때 보정. 0=보정 없음">
            <input
              type="number"
              step="0.1"
              style={inputStyle}
              value={processParams.gps_mag_lag_seconds}
              onChange={(e) => setProcessParams((p) => ({ ...p, gps_mag_lag_seconds: parseFloat(e.target.value) }))}
            />
          </Field>

          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={dsp.enabled} onChange={(e) => updateDespike("enabled", e.target.checked)} />
            <span>스파이크(이상치) 제거 사용 — 필터 전에 울타리·차량 등 순간 잡음을 이동중앙값으로 대체</span>
          </label>
          {dsp.enabled && (
            <>
              <Field label="이동창 크기 (샘플 수, 홀수) — 클수록 넓은 구간의 중앙값과 비교">
                <input type="number" step="2" min="3" style={inputStyle} value={dsp.window_size} onChange={(e) => updateDespike("window_size", parseInt(e.target.value, 10))} />
              </Field>
              <Field label="탐지 민감도 (표준편차 배수) — 작을수록 더 많이 스파이크로 판정">
                <input type="number" step="0.5" min="0.1" style={inputStyle} value={dsp.threshold_k} onChange={(e) => updateDespike("threshold_k", parseFloat(e.target.value))} />
              </Field>
            </>
          )}

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

          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={cl.enabled} onChange={(e) => updateCrossover("enabled", e.target.checked)} />
            <span>타이라인(검교정 측선) 보정 사용 — 측선과 수직으로 비행한 타이라인이 있을 때만 유효 (선택사항)</span>
          </label>
          {cl.enabled && (
            <>
              <Field label="타이라인 방향 허용오차 (deg) — 주 측선과 수직 방향 기준">
                <input type="number" style={inputStyle} value={cl.tie_tolerance_deg} onChange={(e) => updateCrossover("tie_tolerance_deg", parseFloat(e.target.value))} />
              </Field>
              <Field label="교차점 매칭 거리 (m) — 측선과 타이라인이 이 거리 이내로 지나가면 교차점으로 인정">
                <input
                  type="number"
                  style={inputStyle}
                  value={cl.max_crossover_distance_m}
                  onChange={(e) => updateCrossover("max_crossover_distance_m", parseFloat(e.target.value))}
                />
              </Field>
            </>
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
              {processSummary.gps_mag_lag?.lag_seconds !== 0 && (
                <>
                  GPS-자력계 시간 오프셋 보정: {processSummary.gps_mag_lag.lag_seconds}초 적용 (경계 구간 {processSummary.gps_mag_lag.n_points_dropped}개 포인트 제외)
                  <br />
                </>
              )}
              {processSummary.despike?.enabled && (
                <>
                  스파이크 제거: {processSummary.despike.n_spikes_removed}개 ({processSummary.despike.pct_spikes_removed?.toFixed(2)}%)
                  <br />
                </>
              )}
              {cl.enabled && processSummary.crossover_leveling && (
                <>
                  {processSummary.crossover_leveling.applied ? (
                    <span>
                      타이라인 보정: 측선 {processSummary.crossover_leveling.n_survey_lines_corrected}개 보정 (교차점 {processSummary.crossover_leveling.n_crossovers}개), RMS{" "}
                      {processSummary.crossover_leveling.rms_before_nt?.toFixed(2)} → {processSummary.crossover_leveling.rms_after_nt?.toFixed(2)} nT
                    </span>
                  ) : (
                    <span style={{ color: "#b45309" }}>타이라인 보정: {processSummary.crossover_leveling.reason}</span>
                  )}
                  <br />
                </>
              )}
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
          <Field
            label={
              <>
                컬러로 표시할 값
                <InfoIcon text="TMI(Total Magnetic Intensity, 전자력) = 측정된 자기장 세기(일변화 보정 적용). IGRF(International Geomagnetic Reference Field) = 지구 자체의 배경 자기장 모델값. 자력 이상 = TMI - IGRF로, 지구 배경 자기장을 뺀 뒤 남는, 지하 자성체에 의한 자기장 변화만을 의미합니다 — 보통 이 값을 봅니다." />
              </>
            }
          >
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
          <Field label={`그리드 불투명도: ${Math.round(gridOpacity * 100)}%`}>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={gridOpacity}
              onChange={(e) => setGridOpacity(parseFloat(e.target.value))}
            />
          </Field>
          <Field label="색상 스트레치(대비) 방식">
            <select style={inputStyle} value={stretch} onChange={(e) => setStretch(e.target.value)}>
              <option value="linear">선형 (기본값)</option>
              <option value="equalize">히스토그램 균등화 — 값 분포에 맞춰 대비 자동 강조 (극단값에 덜 묻힘)</option>
            </select>
          </Field>
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={hillshade} onChange={(e) => setHillshade(e.target.checked)} />
            <span style={{ color: "#4b5563" }}>힐쉐이드 효과 (음영기복 - 값 변화를 입체감 있게 강조, Geosoft 스타일)</span>
          </label>
          {hillshade && (
            <>
              <Field label={`광원 방위각: ${hillshadeAzimuth}°`}>
                <input type="range" min="0" max="360" step="5" value={hillshadeAzimuth} onChange={(e) => setHillshadeAzimuth(parseFloat(e.target.value))} />
              </Field>
              <Field label={`광원 고도각: ${hillshadeAltitude}°`}>
                <input type="range" min="5" max="90" step="5" value={hillshadeAltitude} onChange={(e) => setHillshadeAltitude(parseFloat(e.target.value))} />
              </Field>
              <Field label={`강조 정도: ${hillshadeExaggeration}`}>
                <input type="range" min="0.5" max="20" step="0.5" value={hillshadeExaggeration} onChange={(e) => setHillshadeExaggeration(parseFloat(e.target.value))} />
              </Field>
            </>
          )}
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={showContours} onChange={(e) => setShowContours(e.target.checked)} />
            <span style={{ color: "#4b5563" }}>등고선(등자력선) 표시</span>
          </label>
          {showContours && (
            <>
              <Field label="등고선 간격 (nT) — 비워두면 아래 개수로 자동 분할">
                <input
                  type="number"
                  style={inputStyle}
                  value={contourInterval ?? ""}
                  placeholder="자동"
                  onChange={(e) => setContourInterval(e.target.value === "" ? null : parseFloat(e.target.value))}
                />
              </Field>
              <Field label="자동 등고선 개수 (간격을 비워둔 경우)">
                <input type="number" style={inputStyle} value={contourNLevels} onChange={(e) => setContourNLevels(parseInt(e.target.value, 10))} />
              </Field>
            </>
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
                title={TRANSFORM_INFO[key]}
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
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={exportGeotiffColored} onChange={(e) => setExportGeotiffColored(e.target.checked)} />
            <span>컬러 GeoTIFF로 저장 (화면에 보이는 색상 그대로, 값 아님)</span>
          </label>
          <button
            style={{ ...buttonStyle, background: "white", color: "#2563eb" }}
            disabled={exportingGeotiff || !processSummary}
            onClick={() => onExportGeotiff()}
          >
            {exportingGeotiff
              ? "내보내는 중..."
              : `현재 결과(${activeTransform === "none" ? "그리드" : activeTransform.toUpperCase()})를 GeoTIFF로 저장`}
          </button>
          <div style={{ color: "#9ca3af" }}>
            {exportGeotiffColored
              ? "화면에 보이는 컬러맵/힐쉐이드가 그대로 이미지로 저장됩니다 (Google Earth 등에서 바로 볼 때 적합, 값 재분석 불가)."
              : "실제 값(nT 등)이 그대로 저장되어 Oasis Montaj/QGIS/ArcGIS 등에서 다시 열 수 있습니다."}
          </div>
        </div>
      </details>

      <details style={sectionStyle}>
        <summary style={summaryStyle}>11. 배경지도</summary>
        <div style={bodyStyle}>지도 우측 상단 레이어 컨트롤에서 OSM / Esri 위성 / Google 위성을 전환할 수 있습니다.</div>
      </details>
    </div>
  );
}
