import { useState } from "react";
import InfoIcon from "./InfoIcon";

const TRANSFORM_INFO = {
  none: "그리딩된 원본 자력 이상값을 그대로 표시합니다.",
  rtp: "RTP(Reduction to Pole, 자극환산) — 비스듬한 지자기장 방향 때문에 이상체 바로 위가 아닌 옆으로 치우쳐 보이는 이상대를, 자극(수직 자기장)에서 측정한 것처럼 보정해 이상체 바로 위에 오도록 만듭니다.",
  rte: "RTE(Reduction to Equator, 자기적도환산) — 저위도(적도 부근)처럼 복각이 낮아 RTP가 불안정한 지역에서 대신 사용하는 유사한 보정 기법입니다.",
  "1vd": "1VD(수직 1차 미분) — 값 자체가 아니라 수직 방향 변화율을 보여줘, 얕고 경계가 뚜렷한 이상체를 더 선명하게 강조합니다.",
  as: "AS(Analytic Signal, 해석 신호 진폭) — 자화 방향에 무관하게 이상체 바로 위에서 극대값을 갖는 값으로, 자화 방향을 모를 때도 이상체 위치를 판단하기 좋습니다.",
  thdr: "THDR(총수평미분) — 수평 방향 변화율의 크기로, 자화 방향에 무관하게 이상체 경계에서 뚜렷하게 나타나 접촉면·경계 파악에 유용합니다.",
  upward_continuation: "상방연속 — 지정한 높이만큼 더 높은 고도에서 측정한 것처럼 계산을 재구성합니다. 얕고 짧은 파장 잡음이 깊고 넓은 이상보다 훨씬 빠르게 약해져, 광역 추세만 강조하거나 잡음을 완화할 때 사용합니다.",
  detrend: "추세면 제거 — 완만하게 변화하는 광역 배경(다항식 추세면)을 최소자승으로 맞춰 뺀 나머지(잔차)만 표시합니다. 얕은 국지 이상을 광역 배경과 분리해 볼 때 유용합니다.",
  microlevel: "마이크로레벨링(디코러게이션) — 측선과 나란한 방향의 짧은 파장(측선 간격 규모) 줄무늬 잡음만 골라 완화합니다. 헤딩/타이라인 보정 후에도 남는 줄무늬가 있을 때 사용합니다.",
  "2vd": "2VD(수직 2차 미분) — 1VD보다 한 단계 더 미분해, 서로 가까이 붙은 여러 이상체를 분리해 보는 데 유용합니다 (잡음도 그만큼 더 증폭됩니다).",
  tilt: "틸트 각(Tilt Angle) — 1VD/THDR의 비율을 각도(-90°~90°)로 표현해, 이상체 진폭 크기와 무관하게 경계에서 0을 지나갑니다. 강한 이상체와 약한 이상체가 섞여 있어도 같은 기준으로 경계를 볼 수 있습니다.",
  theta: "세타 맵(Theta Map) — THDR을 해석신호 진폭(AS)으로 정규화한 각도(0°~90°)로, 틸트 각과 마찬가지로 진폭에 무관한 경계 탐지 보조 지표입니다.",
  dx: "1차 동서방향 미분(dX) — 동서(easting) 방향 변화율.",
  dy: "1차 남북방향 미분(dY) — 남북(northing) 방향 변화율.",
  dxx: "2차 동서방향 미분(dXX).",
  dyy: "2차 남북방향 미분(dYY).",
  dxy: "동서-남북 혼합 2차 미분(dXY).",
  dxz: "동서-수직 혼합 2차 미분(dXZ).",
  dyz: "남북-수직 혼합 2차 미분(dYZ).",
};

const KOREA_PROJECTION_INFO = {
  "": "비워두면 측선 중심 좌표로 UTM 존을 자동 감지합니다 (국내외 공통).",
  korea_utm: "KoreaUTM (EPSG:5179) — GRS80 기반 한국 통합 좌표계 (KGD2002 Unified CS). 국내 조사에 권장.",
  korea2010: "Korea2010 (EPSG:5185~5188) — 한반도 권역별(서부/중부/동부/동해) 띠 좌표계 (KGD2002 Belt 2010).",
  utm: "UTM (EPSG:32651/32652) — 국내 지역 51N/52N을 경도 기준으로 자동 선택하는 표준 UTM.",
};

const FILTER_METHOD_INFO = {
  butterworth: "Butterworth 저역통과 — 지정한 차단주파수 이상을 뚜렷하게 잘라내는 표준적인 영위상(zero-phase) 필터입니다.",
  savgol: "Savitzky-Golay — 이동창 안에서 다항식을 맞춰 평활화합니다. 이동평균보다 이상체의 봉우리 높이·폭을 더 잘 보존합니다.",
  moving_average: "이동평균 — 가장 단순한 평활화 방식으로, 뾰족한 이상체를 다소 무디게 만들 수 있습니다.",
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
  alongLineSmooth,
  setAlongLineSmooth,
  alongLineSmoothWavelength,
  setAlongLineSmoothWavelength,
  lineSpacingM,
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
  onExportXyz,
  exportingXyz,
  onExportGrd,
  exportingGrd,
  onExportPointsCsv,
  exportingPointsCsv,
  transformExtraParams,
  setTransformExtraParams,
  error,
}) {
  const [droneFileNames, setDroneFileNames] = useState([]);
  const [baseFileNames, setBaseFileNames] = useState([]);
  const [targetWavelengthM, setTargetWavelengthM] = useState(5.0);

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
              {droneSummary.source_formats?.length > 0 && (
                <>
                  <br />
                  인식된 파일 형식: {droneSummary.source_formats.join(", ")}
                </>
              )}
              {(droneSummary.n_duplicate_timestamps_removed > 0 || droneSummary.n_invalid_coords_removed > 0) && (
                <>
                  <br />
                  <span style={{ color: "#b45309" }}>
                    품질검사로 제외됨: 중복 시각 {droneSummary.n_duplicate_timestamps_removed}개, 비정상 좌표 {droneSummary.n_invalid_coords_removed}개
                  </span>
                </>
              )}
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
              {baseSummary.n_duplicate_timestamps_removed > 0 && (
                <>
                  <br />
                  <span style={{ color: "#b45309" }}>품질검사로 제외됨: 중복 시각 {baseSummary.n_duplicate_timestamps_removed}개</span>
                </>
              )}
            </div>
          )}
        </div>
      </details>

      <details style={sectionStyle} open>
        <summary style={summaryStyle}>3~5. 필터 · 측선판별 · 보정</summary>
        <div style={bodyStyle}>
          <Field label="필터 종류">
            <select
              style={inputStyle}
              value={processParams.filter_method || "butterworth"}
              onChange={(e) => setProcessParams((p) => ({ ...p, filter_method: e.target.value }))}
              title={FILTER_METHOD_INFO[processParams.filter_method || "butterworth"]}
            >
              <option value="butterworth">Butterworth 저역통과 (기본값)</option>
              <option value="savgol">Savitzky-Golay (봉우리 보존)</option>
              <option value="moving_average">이동평균 (가장 단순)</option>
            </select>
          </Field>
          <div style={{ color: "#9ca3af" }}>{FILTER_METHOD_INFO[processParams.filter_method || "butterworth"]}</div>

          {(processParams.filter_method || "butterworth") === "butterworth" ? (
            <>
              <Field label="저주파 통과 필터 차단주파수 (Hz)">
                <input
                  type="number"
                  step="0.1"
                  style={inputStyle}
                  value={processParams.filter_cutoff_hz}
                  onChange={(e) => setProcessParams((p) => ({ ...p, filter_cutoff_hz: parseFloat(e.target.value) }))}
                />
              </Field>
              {droneSummary?.median_speed_mps > 0 && (
                <div style={{ display: "flex", alignItems: "flex-end", gap: 6, color: "#6b7280" }}>
                  <Field label={`평균 비행속도 ${droneSummary.median_speed_mps.toFixed(1)} m/s 기준, 보존할 최소 파장(m)`}>
                    <input
                      type="number"
                      step="0.5"
                      min="0.1"
                      style={inputStyle}
                      value={targetWavelengthM}
                      onChange={(e) => setTargetWavelengthM(parseFloat(e.target.value))}
                    />
                  </Field>
                  <button
                    type="button"
                    style={{ ...buttonStyle, background: "white", color: "#2563eb", padding: "4px 8px", whiteSpace: "nowrap" }}
                    disabled={!targetWavelengthM || targetWavelengthM <= 0}
                    onClick={() => {
                      const suggested = droneSummary.median_speed_mps / targetWavelengthM;
                      setProcessParams((p) => ({ ...p, filter_cutoff_hz: Math.round(suggested * 1000) / 1000 }));
                    }}
                  >
                    권장값 적용 ({(droneSummary.median_speed_mps / (targetWavelengthM || 1)).toFixed(3)} Hz)
                  </button>
                </div>
              )}
            </>
          ) : (
            <>
              <Field label="필터 창 크기 (초)">
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  style={inputStyle}
                  value={processParams.filter_window_seconds}
                  onChange={(e) => setProcessParams((p) => ({ ...p, filter_window_seconds: parseFloat(e.target.value) }))}
                />
              </Field>
              {processParams.filter_method === "savgol" && (
                <Field label="다항식 차수 (커질수록 봉우리 보존↑, 잡음 제거↓)">
                  <input
                    type="number"
                    min="1"
                    max="7"
                    style={inputStyle}
                    value={processParams.filter_polyorder}
                    onChange={(e) => setProcessParams((p) => ({ ...p, filter_polyorder: parseInt(e.target.value, 10) }))}
                  />
                </Field>
              )}
            </>
          )}

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
              <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={dsp.adaptive} onChange={(e) => updateDespike("adaptive", e.target.checked)} />
                <span>Adaptive Hampel — 급격한 기울기 구간(진짜 경사 신호)은 이동창을 넓혀 스파이크 오판을 줄임</span>
              </label>
              {dsp.adaptive && (
                <Field label="급변 판정 기울기 (nT/샘플) — 이 값을 초과하면 창을 확장">
                  <input
                    type="number"
                    step="0.5"
                    min="0.1"
                    style={inputStyle}
                    value={dsp.adaptive_gradient_threshold}
                    onChange={(e) => updateDespike("adaptive_gradient_threshold", parseFloat(e.target.value))}
                  />
                </Field>
              )}
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
          <Field label="주 측선 방향 검출 방식">
            <select style={inputStyle} value={lp.direction_method || "heading_histogram"} onChange={(e) => updateLine("direction_method", e.target.value)}>
              <option value="heading_histogram">진행방향 히스토그램 (기본값) — 순간 비행 방향 최빈값</option>
              <option value="pca">PCA 자동 방향 검출 — 측점 분포의 주축(길게 늘어진 방향) 기준</option>
            </select>
          </Field>

          <Field label="좌표계 (Korea Projection) — 국내 조사 시 표준 좌표계 선택, 비워두면 UTM 자동 감지">
            <select
              style={inputStyle}
              value={processParams.korea_projection ?? ""}
              onChange={(e) => setProcessParams((p) => ({ ...p, korea_projection: e.target.value === "" ? null : e.target.value }))}
              title={KOREA_PROJECTION_INFO[processParams.korea_projection ?? ""]}
            >
              <option value="">자동 (UTM 자동 감지)</option>
              <option value="korea_utm">KoreaUTM (EPSG:5179)</option>
              <option value="korea2010">Korea2010 (EPSG:5185~5188)</option>
              <option value="utm">UTM (국내 51N/52N)</option>
            </select>
          </Field>
          <div style={{ color: "#9ca3af" }}>{KOREA_PROJECTION_INFO[processParams.korea_projection ?? ""]}</div>

          <Field label="좌표계 수동 지정 (EPSG 코드) — 지정 시 위 Korea Projection 선택보다 우선 적용. 예: UTM 48N = 32648">
            <input
              type="number"
              style={inputStyle}
              placeholder="자동 감지"
              value={processParams.utm_epsg_override ?? ""}
              onChange={(e) => setProcessParams((p) => ({ ...p, utm_epsg_override: e.target.value === "" ? null : parseInt(e.target.value, 10) }))}
            />
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
              <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={cl.iterative} onChange={(e) => updateCrossover("iterative", e.target.checked)} />
                <span>반복(iterative) 네트워크 보정 — 타이라인도 함께 보정해 오차를 양쪽에 고르게 분산 (해제 시 타이라인을 고정 기준으로 취급)</span>
              </label>
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
              좌표계: EPSG:{processSummary.utm_epsg}
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
              <option value="minimum_curvature">최소곡률(Minimum Curvature) — Surfer 기본 격자화 방식과 동일한 알고리즘</option>
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
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={alongLineSmooth} onChange={(e) => setAlongLineSmooth(e.target.checked)} />
            <span
              title="측선 방향 자료 간격(수 m)이 측선 사이 간격(보통 수십~수백 m)보다 훨씬 촘촘해서, 옆 측선과는 비교할 수 없는 미세한 변화까지 그리드에 그대로 반영되면 측선과 나란한 방향의 물결/주름 무늬(코러게이션)가 생깁니다. 측선 방향으로 이 정도 파장 이하의 세부 변화를 미리 눌러주면, 어떤 보간 방법을 쓰든 주름이 크게 줄어듭니다."
            >
              측선방향 평활화 — 측선/횡측선 간격 불일치로 인한 주름(코러게이션) 억제 (기본 켜짐)
            </span>
          </label>
          {alongLineSmooth && (
            <Field
              label={
                lineSpacingM
                  ? `평활화 파장 (m) — 비워두면 추정 측선 간격(${lineSpacingM.toFixed(1)}m) 사용`
                  : "평활화 파장 (m) — 비워두면 측선 간격 자동 추정값 사용"
              }
            >
              <input
                type="number"
                style={inputStyle}
                placeholder={lineSpacingM ? `자동: ${lineSpacingM.toFixed(1)}` : "자동"}
                value={alongLineSmoothWavelength ?? ""}
                onChange={(e) => setAlongLineSmoothWavelength(e.target.value === "" ? null : parseFloat(e.target.value))}
              />
            </Field>
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
        <summary style={summaryStyle}>10. 파생 그리드 (RTP / RTE / AS / 1VD 외)</summary>
        <div style={bodyStyle}>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {[
              ["none", "그리드(원본)"],
              ["rtp", "RTP"],
              ["rte", "RTE"],
              ["1vd", "1VD"],
              ["2vd", "2VD"],
              ["as", "AS"],
              ["thdr", "THDR"],
              ["tilt", "틸트각"],
              ["theta", "세타맵"],
              ["dx", "dX"],
              ["dy", "dY"],
              ["dxx", "dXX"],
              ["dyy", "dYY"],
              ["dxy", "dXY"],
              ["dxz", "dXZ"],
              ["dyz", "dYZ"],
              ["upward_continuation", "상방연속"],
              ["detrend", "추세면제거"],
              ["microlevel", "마이크로레벨링"],
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
            <input
              type="checkbox"
              checked={transformExtraParams.microlevel_pre_apply}
              onChange={(e) => setTransformExtraParams((p) => ({ ...p, microlevel_pre_apply: e.target.checked }))}
            />
            <span
              title="지금 마이크로레벨링 버튼을 눌러야만 적용되는 게 아니라, 이 체크박스를 켜두면 위 그리드(원본)를 포함한 다른 모든 파생그리드 계산 전에 먼저 마이크로레벨링(측선방향 줄무늬 완화)을 적용한 뒤 그 결과에 RTP/1VD/AS 등을 계산합니다. 미분 기반 변환일수록 줄무늬가 미리 지워진 상태에서 계산되어 결과가 더 매끈해집니다."
            >
              선택한 파생그리드 계산 전에 마이크로레벨링 먼저 적용 (원본 그리드 보기에도 적용)
            </span>
          </label>
          {activeTransform === "upward_continuation" && (
            <Field label="상방연속 고도 (m)">
              <input
                type="number"
                style={inputStyle}
                value={transformExtraParams.continuation_height_m}
                onChange={(e) => setTransformExtraParams((p) => ({ ...p, continuation_height_m: parseFloat(e.target.value) }))}
              />
            </Field>
          )}
          {activeTransform === "detrend" && (
            <Field label="추세면 차수 (1=평면, 2=2차, 3=3차 - 높을수록 더 복잡한 배경을 뺌)">
              <select
                style={inputStyle}
                value={transformExtraParams.trend_order}
                onChange={(e) => setTransformExtraParams((p) => ({ ...p, trend_order: parseInt(e.target.value, 10) }))}
              >
                <option value={1}>1차 (평면)</option>
                <option value={2}>2차</option>
                <option value={3}>3차</option>
              </select>
            </Field>
          )}
          {(activeTransform === "microlevel" || transformExtraParams.microlevel_pre_apply) && (
            <>
              <Field label={`보정 강도: ${transformExtraParams.microlevel_strength}`}>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={transformExtraParams.microlevel_strength}
                  onChange={(e) => setTransformExtraParams((p) => ({ ...p, microlevel_strength: parseFloat(e.target.value) }))}
                />
              </Field>
              <Field label="방향 허용오차 (deg) — 측선 직각방향 기준">
                <input
                  type="number"
                  style={inputStyle}
                  value={transformExtraParams.microlevel_angle_tolerance_deg}
                  onChange={(e) => setTransformExtraParams((p) => ({ ...p, microlevel_angle_tolerance_deg: parseFloat(e.target.value) }))}
                />
              </Field>
              <Field label="파장 대역폭 배수 (측선 간격 기준, 클수록 더 넓은 파장대를 완화)">
                <input
                  type="number"
                  step="0.1"
                  min="1.01"
                  style={inputStyle}
                  value={transformExtraParams.microlevel_wavelength_factor}
                  onChange={(e) => setTransformExtraParams((p) => ({ ...p, microlevel_wavelength_factor: parseFloat(e.target.value) }))}
                />
              </Field>
            </>
          )}
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
          <button style={{ ...buttonStyle, background: "white", color: "#2563eb" }} disabled={exportingXyz || !processSummary} onClick={() => onExportXyz()}>
            {exportingXyz ? "내보내는 중..." : `현재 결과(${activeTransform === "none" ? "그리드" : activeTransform.toUpperCase()})를 XYZ(텍스트)로 저장`}
          </button>
          <div style={{ color: "#9ca3af" }}>경도·위도·값 3열의 공백 구분 텍스트 파일 — GeoTIFF를 지원하지 않는 다른 프로그램에서도 열람 가능.</div>
          <button style={{ ...buttonStyle, background: "white", color: "#2563eb" }} disabled={exportingGrd || !processSummary} onClick={() => onExportGrd()}>
            {exportingGrd ? "내보내는 중..." : `현재 결과(${activeTransform === "none" ? "그리드" : activeTransform.toUpperCase()})를 Surfer GRD로 저장`}
          </button>
          <div style={{ color: "#9ca3af" }}>Surfer 6 Binary Grid(.grd, DSBB) 형식 — Golden Software Surfer에서 바로 열람 가능.</div>
          <button style={{ ...buttonStyle, background: "white", color: "#2563eb" }} disabled={exportingPointsCsv || !processSummary} onClick={() => onExportPointsCsv()}>
            {exportingPointsCsv ? "내보내는 중..." : "처리된 포인트 전체를 CSV로 저장"}
          </button>
        </div>
      </details>

      <details style={sectionStyle}>
        <summary style={summaryStyle}>11. 배경지도</summary>
        <div style={bodyStyle}>지도 우측 상단 레이어 컨트롤에서 OSM / Esri 위성 / Google 위성을 전환할 수 있습니다.</div>
      </details>
    </div>
  );
}
