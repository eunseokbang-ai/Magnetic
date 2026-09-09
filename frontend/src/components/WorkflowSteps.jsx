import { useState } from "react";
import InfoIcon from "./InfoIcon";
import IntermagnetPanel from "./IntermagnetPanel";

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

const sectionStyle = { border: "1px solid #e6dac0", borderRadius: 8, marginBottom: 10, background: "white" };
const summaryStyle = { padding: "10px 12px", fontWeight: 600, fontSize: 13, cursor: "pointer" };
const bodyStyle = { padding: "0 12px 12px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #ddd0b2" };
const buttonStyle = {
  padding: "7px 10px",
  fontSize: 12,
  fontWeight: 600,
  borderRadius: 6,
  border: "1px solid #a9631f",
  background: "#a9631f",
  color: "white",
  cursor: "pointer",
};

function ProgressBar({ fraction }) {
  if (fraction == null) return null;
  const pct = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, height: 6, borderRadius: 3, background: "#e6dac0", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: "#a9631f", transition: "width 0.15s" }} />
      </div>
      <span style={{ color: "#8a7a5c", fontSize: 11, minWidth: 32, textAlign: "right" }}>{pct}%</span>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ color: "#6b5c42" }}>{label}</span>
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
  onShowBaseTimeseries,
  baseTimeseriesLoading,
  onUploadIaga2002,
  onFetchIntermagnet,
  onApplyIntermagnet,
  onCancelIntermagnetPreview,
  intermagnetPreview,
  intermagnetLoading,
  intermagnetApplying,
  intermagnetError,
  onFetchNearestIntermagnet,
  onApplyNearestIntermagnet,
  onCancelNearestIntermagnetPreview,
  onExportNearestIntermagnetCsv,
  nearestIntermagnetCsvFilename,
  setNearestIntermagnetCsvFilename,
  onShowNearestIntermagnetComparison,
  nearestIntermagnetPreview,
  nearestIntermagnetLoading,
  nearestIntermagnetApplying,
  nearestIntermagnetError,
  nearestIntermagnetHasResult,
  nearestIntermagnetComparisonLoading,
  onUploadHeadingCalibration,
  headingCalibrationSummary,
  headingCalibrationUploadProgress,
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
  overlay,
  onGridConfidence,
  confidenceLoading,
  confidenceOverlay,
  showConfidenceOverlay,
  setShowConfidenceOverlay,
  boundaryMode,
  onToggleBoundaryMode,
  boundaryPolygon,
  onClearBoundary,
  onExportBoundary,
  boundaryFilename,
  setBoundaryFilename,
  onImportBoundaryFile,
  boundaryBufferM,
  setBoundaryBufferM,
  onAutoBoundary,
  autoBoundaryBusy,
  autoBoundaryInfo,
  onExportBoundaryGis,
  inspectMode,
  onToggleInspectMode,
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
  const [headingCalFileNames, setHeadingCalFileNames] = useState([]);
  const [targetWavelengthM, setTargetWavelengthM] = useState(5.0);

  const lp = processParams.line_params;
  const dsp = processParams.despike_params;
  const updateDespike = (key, val) => setProcessParams((p) => ({ ...p, despike_params: { ...p.despike_params, [key]: val } }));
  const swd = processParams.sway_detection;
  const updateSway = (key, val) => setProcessParams((p) => ({ ...p, sway_detection: { ...p.sway_detection, [key]: val } }));
  const dlp = processParams.duplicate_line_params;
  const updateDuplicateLine = (key, val) =>
    setProcessParams((p) => ({ ...p, duplicate_line_params: { ...p.duplicate_line_params, [key]: val } }));
  const hec = processParams.heading_effect_calibration;
  const updateHeadingCal = (key, val) =>
    setProcessParams((p) => ({ ...p, heading_effect_calibration: { ...p.heading_effect_calibration, [key]: val } }));
  const cl = processParams.crossover_leveling;
  const updateCrossover = (key, val) => setProcessParams((p) => ({ ...p, crossover_leveling: { ...p.crossover_leveling, [key]: val } }));
  const dp = processParams.diurnal_params;
  const hc = processParams.heading_correction;
  const sl = processParams.statistical_leveling;
  const updateStatLevel = (key, val) =>
    setProcessParams((p) => ({ ...p, statistical_leveling: { ...p.statistical_leveling, [key]: val } }));

  const updateLine = (key, val) => setProcessParams((p) => ({ ...p, line_params: { ...p.line_params, [key]: val } }));
  const updateDiurnal = (key, val) => setProcessParams((p) => ({ ...p, diurnal_params: { ...p.diurnal_params, [key]: val } }));
  const updateHeading = (key, val) => setProcessParams((p) => ({ ...p, heading_correction: { ...p.heading_correction, [key]: val } }));

  return (
    <div>
      {error && (
        <div style={{ ...sectionStyle, padding: 10, background: "#fef2f2", color: "#dc2626", fontSize: 12 }}>{error}</div>
      )}

      <details id="wf-section-drone" style={sectionStyle} open>
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
          <div style={{ color: "#8a7a5c" }}>여러 비행 파일을 함께 선택하면 하나로 합쳐 처리합니다.</div>
          <ProgressBar fraction={droneUploadProgress} />
          {droneFileNames.length > 0 && <div style={{ color: "#8a7a5c" }}>{droneFileNames.join(", ")}</div>}
          {droneSummary && (
            <div style={{ color: "#4a3d28" }}>
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

      <details id="wf-section-base" style={sectionStyle} open>
        <summary style={summaryStyle}>2. 베이스(일변화) 자료 업로드</summary>
        <div style={bodyStyle}>
          <label style={{ display: "flex", alignItems: "flex-start", gap: 6, background: "#faf6ec", border: "1px solid #e6dac0", borderRadius: 6, padding: "6px 8px" }}>
            <input
              type="checkbox"
              checked={dp.mode === "assume_constant"}
              onChange={(e) => updateDiurnal("mode", e.target.checked ? "assume_constant" : "base_station")}
            />
            <span>베이스 자료 없이 진행 — 지구자기장이 일정하다고 가정 (일변화 보정 생략, 측정값을 그대로 사용)</span>
          </label>
          {dp.mode === "assume_constant" && (
            <div style={{ color: "#b45309" }}>
              ⚠ 태양풍 등에 의한 시간에 따른 자기장 변화(일변화)가 보정되지 않습니다. 짧은 조사(하루 이내)라면 영향이 작지만,
              가능하면 아래 "INTERMAGNET 관측소 자료 사용"으로 인근 관측소 자료를 대신 활용하는 것을 권장합니다.
            </div>
          )}
          <input
            type="file"
            accept=".csv,.txt"
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
          <div style={{ color: "#8a7a5c" }}>
            여러 베이스 로그 파일을 함께 선택하면 하나로 합쳐 처리합니다. 쉼표구분 CSV(오전/오후 시각) 형식과, 파일명에
            날짜(YYYYMMDD)가 포함된 "시 분 초 X Y Z F" 공백구분 1초간격 텍스트(예: cyg202607151s.txt) 형식을 모두
            자동으로 인식합니다.
          </div>
          <ProgressBar fraction={baseUploadProgress} />
          {baseFileNames.length > 0 && <div style={{ color: "#8a7a5c" }}>{baseFileNames.join(", ")}</div>}
          {baseSummary && (
            <div style={{ color: "#4a3d28" }}>
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
              {baseSummary.date_fallback_used && (
                <>
                  <br />
                  <span style={{ color: "#dc2626" }}>
                    ⚠ 파일명에서 날짜(YYYYMMDD)를 찾지 못해 오늘 날짜를 임시로 사용했습니다 - 실제 측정일과 다르면
                    드론 자료와 시간이 맞지 않아 일변화 보정이 실패합니다. 파일명에 날짜를 포함해 다시 올려주세요.
                  </span>
                </>
              )}
              {baseSummary.source?.type === "intermagnet" && (
                <>
                  <br />
                  <span style={{ color: "#a9631f" }}>
                    출처: INTERMAGNET 관측소 {baseSummary.source.station_name} ({baseSummary.source.iaga_code})
                  </span>
                </>
              )}
              {baseSummary.source?.type === "intermagnet_nearest" && (
                <>
                  <br />
                  <span style={{ color: "#a9631f" }}>
                    출처: 주변 INTERMAGNET 관측소 {baseSummary.source.station_name} 자료의 거리가중평균(IDW) 추정값
                  </span>
                </>
              )}
            </div>
          )}

          {onUploadIaga2002 && (
            <details style={{ marginTop: 6 }}>
              <summary style={{ cursor: "pointer", color: "#4a3d28" }}>
                INTERMAGNET 관측소 자료 사용 (베이스 자료 대신 - 로컬 일변화는 반영 못하지만 태양풍에 의한 지역/전지구적 변화는 반영)
              </summary>
              <IntermagnetPanel
                onUploadIaga2002={onUploadIaga2002}
                onFetchIntermagnet={onFetchIntermagnet}
                onApplyIntermagnet={onApplyIntermagnet}
                onCancelPreview={onCancelIntermagnetPreview}
                preview={intermagnetPreview}
                loading={intermagnetLoading}
                applying={intermagnetApplying}
                error={intermagnetError}
                onFetchNearestIntermagnet={onFetchNearestIntermagnet}
                onApplyNearestIntermagnet={onApplyNearestIntermagnet}
                onCancelNearestPreview={onCancelNearestIntermagnetPreview}
                onExportNearestCsv={onExportNearestIntermagnetCsv}
                nearestCsvFilename={nearestIntermagnetCsvFilename}
                onNearestCsvFilenameChange={setNearestIntermagnetCsvFilename}
                onShowNearestComparison={onShowNearestIntermagnetComparison}
                nearestPreview={nearestIntermagnetPreview}
                nearestLoading={nearestIntermagnetLoading}
                nearestApplying={nearestIntermagnetApplying}
                nearestError={nearestIntermagnetError}
                nearestHasResult={nearestIntermagnetHasResult}
                nearestComparisonLoading={nearestIntermagnetComparisonLoading}
                flightDates={droneSummary?.flight_dates}
              />
            </details>
          )}

          {onShowBaseTimeseries && (
            <button
              type="button"
              onClick={onShowBaseTimeseries}
              disabled={!baseSummary || baseTimeseriesLoading}
              style={{ ...buttonStyle, opacity: baseSummary ? 1 : 0.5 }}
            >
              {baseTimeseriesLoading ? "불러오는 중..." : "📈 베이스 원본/보정 자료 그래프 보기"}
            </button>
          )}
          <details style={{ marginTop: 6 }}>
            <summary style={{ cursor: "pointer", color: "#4a3d28" }}>베이스 자료 QC 옵션 (설치/회수 노이즈 트림 · 스파이크 제거)</summary>
            <div style={{ ...bodyStyle, paddingTop: 8 }}>
              <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input
                  type="checkbox"
                  checked={processParams.base_qc_params.trim_enabled}
                  onChange={(e) =>
                    setProcessParams((p) => ({ ...p, base_qc_params: { ...p.base_qc_params, trim_enabled: e.target.checked } }))
                  }
                />
                <span>설치/회수 구간 트림 사용 (베이스 로거를 내려놓거나 집어들 때의 큰 노이즈 제거)</span>
              </label>
              {processParams.base_qc_params.trim_enabled && (
                <Field label="트림 민감도 (임계값 배수 - 낮을수록 더 많이 트림)">
                  <input
                    type="number"
                    step="0.5"
                    style={inputStyle}
                    value={processParams.base_qc_params.trim_threshold_k}
                    onChange={(e) =>
                      setProcessParams((p) => ({
                        ...p,
                        base_qc_params: { ...p.base_qc_params, trim_threshold_k: parseFloat(e.target.value) || 6.0 },
                      }))
                    }
                  />
                </Field>
              )}
              <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input
                  type="checkbox"
                  checked={processParams.base_qc_params.despike_enabled}
                  onChange={(e) =>
                    setProcessParams((p) => ({ ...p, base_qc_params: { ...p.base_qc_params, despike_enabled: e.target.checked } }))
                  }
                />
                <span>중간 구간 스파이크 제거 사용</span>
              </label>
              {processParams.base_qc_params.despike_enabled && (
                <Field label="스파이크 민감도 (임계값 배수 - 낮을수록 더 많이 제거)">
                  <input
                    type="number"
                    step="0.5"
                    style={inputStyle}
                    value={processParams.base_qc_params.despike_threshold_k}
                    onChange={(e) =>
                      setProcessParams((p) => ({
                        ...p,
                        base_qc_params: { ...p.base_qc_params, despike_threshold_k: parseFloat(e.target.value) || 5.0 },
                      }))
                    }
                  />
                </Field>
              )}
              {processSummary?.base_qc && (
                <div style={{ color: "#4a3d28" }}>
                  설치구간 트림: {processSummary.base_qc.n_trimmed_start}건 · 회수구간 트림: {processSummary.base_qc.n_trimmed_end}건 · 스파이크
                  제거: {processSummary.base_qc.n_spikes_removed}건
                </div>
              )}
            </div>
          </details>
        </div>
      </details>

      <details style={sectionStyle}>
        <summary style={summaryStyle}>2-1. 헤딩효과 캘리브레이션 비행 (선택사항)</summary>
        <div style={bodyStyle}>
          <div style={{ color: "#8a7a5c" }}>
            Geometrics MagArrow 등 나침반(Compass) 데이터가 있는 장비에서, 자기 경사도가 낮은(1nT/m 미만) 좁은 구역(10x10m
            내외, 가능한 한 높은 고도)을 최소 1~2바퀴 회전 후 클로버잎 또는 실제 측선과 같은 방향의 패턴으로, 실제 조사와
            비슷한 속도로 여러 자세를 스쳐 지나가며 짧게 비행한 자료입니다. 업로드하면 헤딩(자세)에 따른 판독 오차를
            모델링해 본 측선 자료에서 제거합니다 (Zhang et al. 2022, The Leading Edge — MagArrow 개발사 논문 기법). 별도
            캘리브레이션 비행 자료가 없으면 아래 "턴 구간 자동 활용" 옵션으로 측선 자체의 턴(방향전환) 구간을 대신
            사용할 수 있습니다 — 제조사 지침에도 "턴 구간이 보통 가장 좋은 캘리브레이션 자료"라고 명시되어 있습니다.
          </div>
          <input
            type="file"
            accept=".csv"
            multiple
            style={inputStyle}
            onChange={(e) => {
              const files = Array.from(e.target.files);
              if (files.length > 0) {
                setHeadingCalFileNames(files.map((f) => f.name));
                onUploadHeadingCalibration(files);
              }
            }}
          />
          <ProgressBar fraction={headingCalibrationUploadProgress} />
          {headingCalFileNames.length > 0 && <div style={{ color: "#8a7a5c" }}>{headingCalFileNames.join(", ")}</div>}
          {headingCalibrationSummary && (
            <div style={{ color: "#4a3d28" }}>
              포인트 수: {headingCalibrationSummary.n_points}
              <br />
              시간범위: {headingCalibrationSummary.time_range?.[0]} ~ {headingCalibrationSummary.time_range?.[1]}
              <br />
              {headingCalibrationSummary.has_compass_data ? (
                <span>나침반(Compass) 데이터 확인됨</span>
              ) : (
                <span style={{ color: "#b45309" }}>⚠ 나침반(Compass) 컬럼을 찾지 못했습니다 — 지원 포맷(MagArrow 등)인지 확인하세요.</span>
              )}
            </div>
          )}
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={hec.enabled} onChange={(e) => updateHeadingCal("enabled", e.target.checked)} />
            <span>헤딩효과 보정 사용</span>
          </label>
          {hec.enabled && (
            <>
              <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input
                  type="checkbox"
                  checked={hec.auto_calibrate_from_turns}
                  onChange={(e) => updateHeadingCal("auto_calibrate_from_turns", e.target.checked)}
                />
                <span title="위 캘리브레이션 비행을 업로드하지 않은 경우, 측선 자료 자체의 턴(방향전환) 구간을 캘리브레이션 자료로 자동 추출해 사용합니다. 업로드된 캘리브레이션 파일이 있으면 그쪽이 항상 우선 사용됩니다.">
                  턴 구간 자동 활용 — 캘리브레이션 비행 업로드가 없으면 측선의 턴 구간에서 자동 추출 (기본 켜짐)
                </span>
              </label>
              <Field label="캘리브레이션 품질 기준 (교차검증 잔차 표준편차, nT) — 이보다 크면 신뢰도 낮음(FAIL)으로 표시">
                <input
                  type="number"
                  step="0.5"
                  min="0.1"
                  style={inputStyle}
                  value={hec.quality_threshold_nt}
                  onChange={(e) => updateHeadingCal("quality_threshold_nt", parseFloat(e.target.value))}
                />
              </Field>
            </>
          )}
        </div>
      </details>

      <details id="wf-section-process" style={sectionStyle} open>
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
          <div style={{ color: "#ab9a78" }}>{FILTER_METHOD_INFO[processParams.filter_method || "butterworth"]}</div>

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
                <div style={{ display: "flex", alignItems: "flex-end", gap: 6, color: "#8a7a5c" }}>
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
                    style={{ ...buttonStyle, background: "white", color: "#a9631f", padding: "4px 8px", whiteSpace: "nowrap" }}
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

          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={swd.enabled} onChange={(e) => updateSway("enabled", e.target.checked)} />
            <span
              title="드론에 매달린 자력계가 흔들리거나 회전하면 헤딩 오차가 생겨 측선 줄무늬(코러게이션)의 주원인이 됩니다. 원본 파일에 자이로/가속도 컬럼이 있을 때만 동작하며, 없으면 자동으로 건너뜁니다."
            >
              IMU 흔들림(스웨이) 검출 — 자이로/가속도 데이터로 센서가 흔들린 구간을 자동 제외 (지원 포맷에서만 동작)
            </span>
          </label>
          {swd.enabled && (
            <Field label="탐지 민감도 (robust z-score 배수) — 작을수록 더 많이 흔들림으로 판정">
              <input
                type="number"
                step="0.5"
                min="0.1"
                style={inputStyle}
                value={swd.threshold_k}
                onChange={(e) => updateSway("threshold_k", parseFloat(e.target.value))}
              />
            </Field>
          )}

          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input
              type="checkbox"
              checked={dlp.enabled}
              onChange={(e) => updateDuplicateLine("enabled", e.target.checked)}
            />
            <span title="같은 구간을 실수로(또는 재비행으로) 2번 이상 겹치게 비행한 경우, 겹치는 구간에서 자력계 노이즈(4th difference)가 더 낮은(품질이 더 좋은) 측선만 남기고 나머지는 자동으로 제외합니다. 정상적인 측선 간격(보통 수십 m 이상)은 건드리지 않고, 거의 같은 경로(기본 8m 이내)를 다시 비행한 경우만 잡아냅니다.">
              반복비행 중복 측선 자동 선택 — 같은 경로를 2번 이상 비행했을 때 더 나은 품질의 측선만 사용
            </span>
          </label>
          {dlp.enabled && (
            <>
              <Field label="같은 측선으로 판단할 최대 수직 거리 (m) — 이보다 가까운 두 측선만 중복으로 판단">
                <input
                  type="number"
                  step="0.5"
                  min="0.1"
                  style={inputStyle}
                  value={dlp.perp_tolerance_m}
                  onChange={(e) => updateDuplicateLine("perp_tolerance_m", parseFloat(e.target.value))}
                />
              </Field>
              <Field label="같은 측선으로 판단할 최대 방향 차이 (deg)">
                <input
                  type="number"
                  step="1"
                  min="1"
                  max="45"
                  style={inputStyle}
                  value={dlp.angle_tolerance_deg}
                  onChange={(e) => updateDuplicateLine("angle_tolerance_deg", parseFloat(e.target.value))}
                />
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
          <div style={{ color: "#ab9a78" }}>{KOREA_PROJECTION_INFO[processParams.korea_projection ?? ""]}</div>

          <Field label="좌표계 수동 지정 (EPSG 코드) — 지정 시 위 Korea Projection 선택보다 우선 적용. 예: UTM 48N = 32648">
            <input
              type="number"
              style={inputStyle}
              placeholder="자동 감지"
              value={processParams.utm_epsg_override ?? ""}
              onChange={(e) => setProcessParams((p) => ({ ...p, utm_epsg_override: e.target.value === "" ? null : parseInt(e.target.value, 10) }))}
            />
          </Field>

          <Field label="베이스 시간 오프셋 — 베이스 로거 시계가 GPS와 안맞을 때 보정 (기본값 0시간 0분 0초)">
            {(() => {
              const total = dp.time_offset_seconds || 0;
              const sign = total < 0 ? -1 : 1;
              const absTotal = Math.abs(total);
              const h = Math.trunc(absTotal / 3600);
              const m = Math.trunc((absTotal - h * 3600) / 60);
              const s = Math.round(absTotal - h * 3600 - m * 60);
              const setPart = (part, value) => {
                const v = value === "" || Number.isNaN(value) ? 0 : Math.abs(value);
                const next = { h, m, s, [part]: v };
                updateDiurnal("time_offset_seconds", sign * (next.h * 3600 + next.m * 60 + next.s));
              };
              return (
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <button
                    type="button"
                    title="부호 전환 (베이스 시계가 GPS보다 빠르면 +, 느리면 -)"
                    style={{ width: 28, padding: "3px 0", fontSize: 12, borderRadius: 4, border: "1px solid #ddd0b2", background: "white", cursor: "pointer" }}
                    onClick={() => updateDiurnal("time_offset_seconds", -total)}
                  >
                    {sign < 0 ? "−" : "+"}
                  </button>
                  <input type="number" min="0" style={{ ...inputStyle, width: 60 }} value={h} title="시" onChange={(e) => setPart("h", parseInt(e.target.value, 10))} />
                  <span>시</span>
                  <input type="number" min="0" style={{ ...inputStyle, width: 60 }} value={m} title="분" onChange={(e) => setPart("m", parseInt(e.target.value, 10))} />
                  <span>분</span>
                  <input type="number" min="0" style={{ ...inputStyle, width: 60 }} value={s} title="초" onChange={(e) => setPart("s", parseInt(e.target.value, 10))} />
                  <span>초</span>
                </div>
              );
            })()}
          </Field>
          <Field label="일변화 기준값">
            <select style={inputStyle} value={dp.reference} onChange={(e) => updateDiurnal("reference", e.target.value)}>
              <option value="mean">비행시간 동안 베이스 평균</option>
              <option value="first">베이스 첫 값</option>
            </select>
          </Field>

          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={hc.enabled} onChange={(e) => updateHeading("enabled", e.target.checked)} />
            <span>헤딩(비행방향) 보정 사용 — 캘리브레이션 비행 없이 반대방향 측선의 조용한 구간으로 추정</span>
          </label>
          {hc.enabled && (
            <>
              <Field
                label={
                  <span title="국소 평면 방식은 측선 2개 간격 정도의 구역마다 '평면(지질 경사) + 방향별 단차'를 동시에 맞춰, 측선 사이의 실제 지질 변화가 헤딩 오차로 잘못 잡히는 것을 막습니다. 모델이 잘 맞는(=넓고 값이 일정한) 구역만 골라 쓰므로, 조용한 구역을 눈으로 고를 필요가 없습니다. 최근접 점쌍은 기존 방식으로, 측선 간격만큼 떨어진 두 점을 직접 빼기 때문에 지질 경사가 그대로 섞여 들어갑니다.">
                    헤딩 보정 방법 ⓘ
                  </span>
                }
              >
                <select style={inputStyle} value={hc.method} onChange={(e) => updateHeading("method", e.target.value)}>
                  <option value="local_plane">국소 평면 + 방향 단차 (권장)</option>
                  <option value="nearest_pair">최근접 점쌍 (기존 방식)</option>
                </select>
              </Field>
              <Field
                label={
                  hc.method === "local_plane"
                    ? "조용한 구역 기준 백분위 (%) — 모델 잔차가 작은(=값이 일정한) 구역만 이 비율만큼 사용"
                    : "조용한 구간 기준 백분위 (%) — 낮을수록 더 엄격하게 조용한 지점만 사용"
                }
              >
                <input
                  type="number"
                  style={inputStyle}
                  value={hc.quiet_percentile}
                  onChange={(e) => updateHeading("quiet_percentile", parseFloat(e.target.value))}
                />
              </Field>
              {hc.method === "local_plane" && (
                <Field label="구역 반경 (측선 간격 배수) — 클수록 안정적이지만 '지질이 평면'이라는 가정이 약해집니다">
                  <input
                    type="number"
                    step="0.5"
                    min="1.5"
                    max="6"
                    style={inputStyle}
                    value={hc.neighborhood_radius_factor}
                    onChange={(e) => updateHeading("neighborhood_radius_factor", parseFloat(e.target.value))}
                  />
                </Field>
              )}
            </>
          )}

          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={sl.enabled} onChange={(e) => updateStatLevel("enabled", e.target.checked)} />
            <span title="지도에 세로 줄무늬(스트립)가 남아 있을 때 쓰는 보정입니다. 타이라인도 캘리브레이션 비행도 필요 없이, 각 측선을 좌우 이웃 측선과 비교해 측선마다 다른 레벨 오차를 찾아 없앱니다. 헤딩 보정은 전진/후진 두 방향에 공통된 오프셋 하나만 잡을 수 있어서, 측선마다 제각각인 오차는 이 보정으로만 제거됩니다.">
              통계적 레벨링 사용 — 이웃 측선 비교로 측선별 레벨 오차 제거 (줄무늬 완화의 핵심, 기본 켜짐) ⓘ
            </span>
          </label>
          {sl.enabled && (
            <>
              <Field
                label={
                  <span title="측선 몇 개 범위로 추세(지질)를 맞출지 정합니다. '클수록 안전' 같은 방향성이 없고 적정값이 따로 있습니다. 너무 좁으면 추세가 오차 자체를 따라가 보정이 거의 안 되고, 너무 넓으면 추세가 지질 변화를 못 따라가 그 맞춤 오차를 레벨 오차인 양 빼버립니다(오차가 전혀 없는 자료에서도 수십 nT를 만들어냄). 기본값 9는 적정 구간에서 약간 보수적인 쪽입니다. 너무 넓게 잡으면 프로그램이 스스로 감지해 경고합니다.">
                    추세 창 (측선 수) — 기본값 9 근처를 권장 ⓘ
                  </span>
                }
              >
                <input
                  type="number"
                  min="7"
                  max="51"
                  style={inputStyle}
                  value={sl.trend_window_lines}
                  onChange={(e) => updateStatLevel("trend_window_lines", parseInt(e.target.value, 10) || 9)}
                />
              </Field>
              <Field label="보정 형태">
                <select
                  style={inputStyle}
                  value={sl.order}
                  onChange={(e) => updateStatLevel("order", parseInt(e.target.value, 10))}
                >
                  <option value={0}>측선당 상수 (0차)</option>
                  <option value={1}>측선 방향 1차 — 한 측선 안의 드리프트까지 보정</option>
                </select>
              </Field>
              <Field label="측선당 보정 상한 (nT) — 비워두면 제한 없음. 실제 이상대 위를 지난 측선이 깎이는 것을 막습니다">
                <input
                  type="number"
                  placeholder="제한 없음"
                  style={inputStyle}
                  value={sl.max_shift_nt ?? ""}
                  onChange={(e) =>
                    updateStatLevel("max_shift_nt", e.target.value === "" ? null : parseFloat(e.target.value))
                  }
                />
              </Field>
            </>
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
              <Field
                label={
                  <span title="0차(상수): 측선 하나당 일정한 오프셋 하나만 보정합니다. 1차(폴리노미얼): 교차점 위치에 따라 보정값이 측선을 따라 선형으로 변하도록 fit합니다 - 타이라인이 2개 이상 있어 한 측선에 교차점이 여러 개 있을 때 더 정확합니다. 1차 선택 시 반복 네트워크 보정 여부와 무관하게 항상 단일 패스로 계산됩니다.">
                    보정 차수 ⓘ
                  </span>
                }
              >
                <select style={inputStyle} value={cl.leveling_order} onChange={(e) => updateCrossover("leveling_order", parseInt(e.target.value, 10))}>
                  <option value={0}>0차 (상수 오프셋, 기본값)</option>
                  <option value={1}>1차 (폴리노미얼/선형 - 타이라인 2개 이상 권장)</option>
                </select>
              </Field>
            </>
          )}

          <button
            style={buttonStyle}
            disabled={processing || !droneSummary || (!baseSummary && dp.mode !== "assume_constant")}
            onClick={onProcess}
          >
            {processing ? "처리 중..." : "자료 처리 실행 (필터+측선판별+보정+IGRF)"}
          </button>

          {processSummary && (
            <div style={{ color: "#4a3d28", marginTop: 4 }}>
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
              {processSummary.heading_effect_calibration?.applied && (
                <>
                  헤딩효과 캘리브레이션 보정 (
                  {processSummary.heading_effect_calibration.calibration_source === "uploaded_file" ? "업로드된 비행" : "측선 턴 구간 자동 추출"}
                  ): 측선 {processSummary.heading_effect_calibration.n_survey_points_corrected}개 포인트 보정 (
                  {processSummary.heading_effect_calibration.pct_survey_points_corrected?.toFixed(1)}%, 평균{" "}
                  {processSummary.heading_effect_calibration.mean_abs_correction_nt?.toFixed(2)} nT)
                  {processSummary.heading_effect_calibration.quality_check?.available && (
                    <>
                      {" — 품질검증: "}
                      <span style={{ color: processSummary.heading_effect_calibration.quality_pass ? "#15803d" : "#b91c1c", fontWeight: 600 }}>
                        {processSummary.heading_effect_calibration.quality_pass ? "PASS" : "FAIL"}
                      </span>
                      {` (잔차 표준편차 ${processSummary.heading_effect_calibration.quality_check.residual_std_nt?.toFixed(2)} nT)`}
                    </>
                  )}
                  <br />
                  {processSummary.heading_effect_calibration.coverage_warning && (
                    <>
                      <span style={{ color: "#b45309" }}>⚠ {processSummary.heading_effect_calibration.coverage_warning}</span>
                      <br />
                    </>
                  )}
                </>
              )}
              {processSummary.heading_effect_calibration?.enabled &&
                !processSummary.heading_effect_calibration?.applied &&
                processSummary.heading_effect_calibration?.reason && (
                  <>
                    <span style={{ color: "#ab9a78" }}>헤딩효과 캘리브레이션 보정: {processSummary.heading_effect_calibration.reason}</span>
                    <br />
                  </>
                )}
              {processSummary.sway_detection?.enabled && processSummary.sway_detection?.available && (
                <>
                  IMU 흔들림 검출: {processSummary.sway_detection.n_points_excluded}개 포인트 제외 ({processSummary.sway_detection.pct_points_flagged?.toFixed(2)}%, 신호:{" "}
                  {processSummary.sway_detection.signal_used})
                  <br />
                </>
              )}
              {processSummary.sway_detection?.enabled && processSummary.sway_detection?.available === false && (
                <>
                  <span style={{ color: "#ab9a78" }}>IMU 흔들림 검출: 원본 파일에 자이로/가속도 데이터 없음 (건너뜀)</span>
                  <br />
                </>
              )}
              {processSummary.duplicate_line_resolution?.enabled && processSummary.duplicate_line_resolution?.n_groups > 0 && (
                <>
                  반복비행 중복 측선 자동 선택: {processSummary.duplicate_line_resolution.n_groups}개 그룹 발견,{" "}
                  {processSummary.duplicate_line_resolution.n_points_excluded}개 포인트 제외 (품질이 더 낮은 측선)
                  <br />
                </>
              )}
              {processSummary.duplicate_line_resolution?.enabled && processSummary.duplicate_line_resolution?.n_groups === 0 && (
                <>
                  <span style={{ color: "#ab9a78" }}>반복비행 중복 측선 자동 선택: 중복으로 판단되는 측선 없음</span>
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
              {processSummary.noise_qc?.available && (
                <>
                  노이즈 QC (정규화 4th/8th difference): 전체 RMS {processSummary.noise_qc.overall_rms_4th_diff_nt?.toFixed(3)} /{" "}
                  {processSummary.noise_qc.overall_rms_8th_diff_nt?.toFixed(3)} nT
                  {processSummary.noise_qc.n_lines_flagged > 0 && (
                    <span style={{ color: "#b45309" }}> — 이상 측선 {processSummary.noise_qc.n_lines_flagged}개 감지됨</span>
                  )}
                  <br />
                </>
              )}
              {processSummary.sampling_qc?.available && (
                <>
                  샘플링 거리 QC: 중앙값 {processSummary.sampling_qc.median_distance_m?.toFixed(2)}m, 최대{" "}
                  {processSummary.sampling_qc.max_distance_m?.toFixed(2)}m
                  {processSummary.sampling_qc.n_gaps_exceeding_tolerance > 0 && (
                    <span style={{ color: "#b45309" }}>
                      {" "}
                      — 허용기준({processSummary.sampling_qc.gap_tolerance_m}m) 초과 구간 {processSummary.sampling_qc.n_gaps_exceeding_tolerance}개 (
                      {processSummary.sampling_qc.pct_gaps_exceeding_tolerance?.toFixed(2)}%, GPS 순간 끊김 가능성)
                    </span>
                  )}
                  <br />
                </>
              )}
              {processSummary.file_level_check?.available && processSummary.file_level_check.flagged_any && (
                <div style={{ color: "#dc2626", marginTop: 4 }}>
                  ⚠ 파일(타일) 간 레벨 불일치 감지: 베이스 위치 변경 등을 확인하세요 (
                  {processSummary.file_level_check.files
                    .filter((f) => f.flagged)
                    .map((f) => `파일#${f.source_file_index} ${f.deviation_nt?.toFixed(1)}nT`)
                    .join(", ")}
                  )
                </div>
              )}
              {processSummary.heading_correction?.applied ? (
                <span>
                  헤딩 보정 오프셋: {processSummary.heading_correction.offset_nt?.toFixed(2)} nT
                  {processSummary.heading_correction.method === "local_plane" ? (
                    <>
                      {" "}
                      (국소 평면 방식 — 구역 {processSummary.heading_correction.n_matched_pairs}개 중 조용한{" "}
                      {processSummary.heading_correction.n_quiet_pairs}개 사용, 구역별 산포 ±
                      {processSummary.heading_correction.offset_spread_nt?.toFixed(2)} nT)
                    </>
                  ) : (
                    <>
                      {" "}
                      (매칭 {processSummary.heading_correction.n_matched_pairs}쌍 중 조용한{" "}
                      {processSummary.heading_correction.n_quiet_pairs}쌍 사용)
                    </>
                  )}
                </span>
              ) : (
                processSummary.heading_correction?.reason && <span style={{ color: "#b45309" }}>헤딩 보정: {processSummary.heading_correction.reason}</span>
              )}
              {(processSummary.heading_correction?.warnings || []).map((w, i) => (
                <div key={i} style={{ color: "#b45309", marginTop: 2 }}>⚠ {w}</div>
              ))}
              <br />
              {processSummary.striping?.available && (
                <div
                  style={{ marginTop: 2 }}
                  title="측선이 좌우 이웃과 얼마나 어긋나 있는지를 nT로 잰 값입니다(줄무늬의 절대 크기). 괄호 안은 그 값을 이 탐사 자체의 자기이상 범위로 나눈 비율로, 서로 다른 탐사·시스템·기체를 비교할 때는 이 비율을 보세요 - 같은 2nT라도 자성이 강한 지역에서는 안 보이고 조용한 지역에서는 지배적입니다. 보정 전 값이라 설정을 바꿔가며 비교할 수 있습니다."
                >
                  줄무늬 세기(보정 전): 측선간 어긋남 {processSummary.striping.line_level_jitter_nt?.toFixed(2)} nT
                  {processSummary.striping.stripe_ratio_pct != null && (
                    <> (이상 범위의 {processSummary.striping.stripe_ratio_pct}%)</>
                  )}{" "}
                  ⓘ
                </div>
              )}
              {processSummary.statistical_leveling?.applied ? (
                <div style={{ marginTop: 2 }}>
                  통계적 레벨링: 측선 {processSummary.statistical_leveling.n_lines}개 보정 (인접쌍{" "}
                  {processSummary.statistical_leveling.n_pairs}개), 보정량 RMS{" "}
                  {processSummary.statistical_leveling.rms_shift_nt?.toFixed(2)} / 최대{" "}
                  {processSummary.statistical_leveling.max_shift_nt?.toFixed(2)} nT, 측선간 요철{" "}
                  {processSummary.statistical_leveling.roughness_before_nt?.toFixed(2)} →{" "}
                  {processSummary.statistical_leveling.roughness_after_nt?.toFixed(2)} nT
                </div>
              ) : (
                processSummary.statistical_leveling?.reason && (
                  <div style={{ color: "#8a7a5c", marginTop: 2 }}>통계적 레벨링: {processSummary.statistical_leveling.reason}</div>
                )
              )}
              {(processSummary.statistical_leveling?.warnings || []).map((w, i) => (
                <div key={i} style={{ color: "#b45309", marginTop: 2 }}>⚠ {w}</div>
              ))}
              {processSummary.diurnal?.mode === "assume_constant" ? (
                <div style={{ color: "#b45309", marginTop: 4 }}>⚠ {processSummary.diurnal.note}</div>
              ) : (
                processSummary.diurnal && !processSummary.diurnal.has_overlap && (
                  <div style={{ color: "#dc2626", marginTop: 4 }}>
                    ⚠ 베이스 자료와 드론 비행시간이 겹치지 않습니다. 베이스 시간범위: {processSummary.diurnal.base_time_range?.[0]} ~{" "}
                    {processSummary.diurnal.base_time_range?.[1]}. 시간 오프셋을 조정하세요.
                  </div>
                )
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

      <details id="wf-section-grid" style={sectionStyle}>
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
              <option value="boxing">Boxing — 빈 셀은 보간하지 않고, 실측값이 있는 셀만 그 평균으로 채움 (자료 없는 구간을 있는 그대로 표시)</option>
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
            <div style={{ color: "#8a7a5c" }}>추정 측선 간격: {processSummary.line_spacing_m.toFixed(1)}m</div>
          )}
          <Field
            label={
              <span title="자료 처리를 실행하면 실제 비행한 측선을 기준으로 경계가 자동 생성되어, 측선이 없는 곳까지 내삽/외삽으로 채워지는 것을 막습니다. 버퍼는 가장 바깥 측선에서 경계까지의 거리입니다. 측선 사이의 정상 간격은 자동으로 이어 붙이므로 안쪽은 빈틈 없이 채워지고, 측선 간격보다 훨씬 넓은 빈 구간(결측 측선 등)만 경계 밖으로 빠집니다.">
                표시 경계 — 측선 기준 자동 생성 (버퍼 조절 가능) ⓘ
              </span>
            }
          >
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center", marginBottom: 6 }}>
              <span style={{ color: "#6b5c42" }}>버퍼</span>
              <input
                type="number"
                min="0.1"
                step="1"
                value={boundaryBufferM}
                // Blank is a real value here - it means "auto = 측선 간격" -
                // so keep it instead of coercing to a number.
                onChange={(e) => {
                  const raw = e.target.value;
                  setBoundaryBufferM(raw === "" ? "" : parseFloat(raw) || 0);
                }}
                placeholder="자동"
                style={{ ...inputStyle, width: 70 }}
                title="가장 바깥 측선에서 경계선까지의 거리(m). 비워두면 측선 간격으로 자동 설정됩니다."
              />
              <span style={{ color: "#6b5c42" }}>m</span>
              <button style={buttonStyle} onClick={() => onAutoBoundary()} disabled={autoBoundaryBusy}>
                {autoBoundaryBusy ? "생성 중..." : boundaryBufferM === "" ? "자동 버퍼로 경계 생성" : "이 버퍼로 경계 다시 생성"}
              </button>
              {boundaryBufferM !== "" && (
                <button
                  style={buttonStyle}
                  onClick={() => setBoundaryBufferM("")}
                  disabled={autoBoundaryBusy}
                  title="버퍼를 비워 측선 간격 자동값으로 되돌립니다."
                >
                  자동값으로
                </button>
              )}
            </div>
            {autoBoundaryInfo && !autoBoundaryInfo.failed && (
              <div style={{ fontSize: 11, color: "#6b5c42", marginBottom: 6 }}>
                자동 경계 적용됨: 버퍼 {autoBoundaryInfo.buffer_m}m
                {autoBoundaryInfo.buffer_auto ? " (측선 간격 자동)" : ""}, 면적 {autoBoundaryInfo.area_km2}km²
                {autoBoundaryInfo.n_parts > 1 ? `, 구역 ${autoBoundaryInfo.n_parts}개` : ""}
                {(autoBoundaryInfo.warnings || []).map((w, i) => (
                  <div key={i} style={{ color: "#b45309", marginTop: 2 }}>
                    ⚠ {w}
                  </div>
                ))}
              </div>
            )}
            {autoBoundaryInfo?.failed && (
              <div style={{ fontSize: 11, color: "#b45309", marginBottom: 6 }}>
                ⚠ 자동 경계를 만들지 못했습니다: {autoBoundaryInfo.reason} (그리드는 기존 방식대로 표시됩니다)
              </div>
            )}
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              <button
                style={{
                  ...buttonStyle,
                  background: boundaryMode ? "#ea580c" : "white",
                  color: boundaryMode ? "white" : "#ea580c",
                  borderColor: "#ea580c",
                }}
                onClick={() => onToggleBoundaryMode()}
              >
                {boundaryMode ? "경계 그리는 중 (다각형 완성하면 자동 적용)" : "경계 그리기"}
              </button>
              {boundaryPolygon && (
                <button style={buttonStyle} onClick={() => onClearBoundary()}>
                  경계 지우기
                </button>
              )}
              {boundaryPolygon && (
                <>
                  <input
                    type="text"
                    value={boundaryFilename}
                    onChange={(e) => setBoundaryFilename(e.target.value)}
                    placeholder="저장할 파일명"
                    title="다른 프로젝트에서도 같은 경계를 재사용하려면 알아보기 쉬운 이름으로 바꿀 수 있습니다"
                    style={{ ...inputStyle, width: 160 }}
                  />
                  <button
                    style={buttonStyle}
                    onClick={() => onExportBoundary()}
                    title="이 앱에서 다시 불러오기 위한 형식 (아래 '경계 파일 불러오기'로 재사용)"
                  >
                    JSON 저장
                  </button>
                  <button
                    style={buttonStyle}
                    onClick={() => onExportBoundaryGis("shp")}
                    title="ESRI 셰이프파일(.shp/.shx/.dbf/.prj)을 zip으로 내려받습니다 - QGIS/ArcGIS에서 바로 열립니다 (WGS84 경위도)"
                  >
                    SHP 저장
                  </button>
                  <button
                    style={buttonStyle}
                    onClick={() => onExportBoundaryGis("kml")}
                    title="구글어스 등에서 열 수 있는 KML로 내려받습니다 (WGS84 경위도)"
                  >
                    KML 저장
                  </button>
                </>
              )}
              <label style={{ ...buttonStyle, display: "inline-flex", alignItems: "center", cursor: "pointer" }}>
                경계 파일 불러오기
                <input
                  type="file"
                  accept=".json"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    const file = e.target.files[0];
                    if (file) onImportBoundaryFile(file);
                    e.target.value = "";
                  }}
                />
              </label>
            </div>
          </Field>
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
                <span title="UAV 자력탐사 가이드라인의 디코러게이션 규칙: 저역통과 파장은 타이라인 간격의 2배, 고역통과 파장은 측선 간격의 2배가 일반적입니다. 여기 자동값(측선 간격 1배)은 우리 프로그램의 그리딩 전 앤티앨리어싱 목적에 맞춘 값으로, 가이드라인 수치와 반드시 같을 필요는 없지만 참고용으로 함께 조정해볼 수 있습니다.">
                  {lineSpacingM
                    ? `평활화 파장 (m) — 비워두면 추정 측선 간격(${lineSpacingM.toFixed(1)}m) 사용 ⓘ`
                    : "평활화 파장 (m) — 비워두면 측선 간격 자동 추정값 사용 ⓘ"}
                </span>
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
              <option value="normal">정규분포 — 평균·표준편차 기준으로 색상 배분 (값이 정규분포에 가까울 때 적합)</option>
            </select>
          </Field>
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={hillshade} onChange={(e) => setHillshade(e.target.checked)} />
            <span style={{ color: "#6b5c42" }}>힐쉐이드 효과 (음영기복 - 값 변화를 입체감 있게 강조, Geosoft 스타일)</span>
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
            <span style={{ color: "#6b5c42" }}>등고선(등자력선) 표시</span>
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
          <button
            style={{
              ...buttonStyle,
              background: inspectMode ? "#a9631f" : "white",
              color: inspectMode ? "white" : "#a9631f",
            }}
            disabled={!overlay}
            onClick={() => onToggleInspectMode()}
            title="켜면 지도를 클릭할 때마다 그 지점의 값(nT)이 지도 위에 표시됩니다. 여러 지점을 계속 클릭해 동시에 비교할 수 있고, 이 버튼을 다시 누르면 전부 지워집니다."
          >
            {inspectMode ? "지점값 확인 중 (클릭하면 종료)" : "지점값 확인"}
          </button>
          <button
            style={{ ...buttonStyle, background: "white", color: "#4a3d28", border: "1px solid #ddd0b2" }}
            disabled={!processSummary || confidenceLoading}
            onClick={() => onGridConfidence()}
            title="자료점으로부터의 거리(같은 그리드 빈 공간 채우기 기준)를 바탕으로, 각 셀이 실제 자료에 얼마나 가까이 구속되어 있는지(1=자료점 바로 위, 0=보간 한계 지점)를 색으로 보여줍니다. 붉은/노란 구간은 보간에 크게 의존한 값이니 해석 시 주의하세요."
          >
            {confidenceLoading ? "신뢰도 계산 중..." : "격자 신뢰도 레이어 계산"}
          </button>
          {confidenceOverlay && (
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input type="checkbox" checked={showConfidenceOverlay} onChange={(e) => setShowConfidenceOverlay(e.target.checked)} />
              <span style={{ color: "#6b5c42" }}>지도에 신뢰도 레이어 표시 (초록=신뢰 높음, 빨강=보간 의존)</span>
            </label>
          )}
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
                  background: activeTransform === key ? "#a9631f" : "white",
                  color: activeTransform === key ? "white" : "#a9631f",
                }}
              >
                {label}
              </button>
            ))}
          </div>
          {transformLoading && <div style={{ color: "#8a7a5c" }}>계산 중...</div>}
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
              <Field
                label={
                  <span title="측선별 레벨 오차가 만드는 줄무늬는 파장 하나가 아니라 여러 파장에 걸쳐 있습니다. 측선마다 값이 다르면 측선 간격 파장에, 전진/후진이 번갈아 어긋나면 측선 간격의 2배 파장에, 그 아래로도 배음이 계속 생깁니다. 디코러게이션(고역통과)은 차단 파장보다 짧은 측선직각 성분을 한꺼번에 제거하므로 이 전부를 잡습니다. 노치는 측선 간격 한 파장만 좁게 제거하므로 지질 신호는 더 잘 보존하지만 줄무늬는 일부만 사라집니다.">
                    필터 방식 ⓘ
                  </span>
                }
              >
                <select
                  style={inputStyle}
                  value={transformExtraParams.microlevel_mode}
                  onChange={(e) => setTransformExtraParams((p) => ({ ...p, microlevel_mode: e.target.value }))}
                >
                  <option value="decorrugation">디코러게이션 (고역통과, 권장)</option>
                  <option value="notch">노치 (측선 간격 한 파장만)</option>
                </select>
              </Field>
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
              {transformExtraParams.microlevel_mode === "decorrugation" ? (
                <Field
                  label={
                    <span title="측선 간격의 몇 배보다 짧은 측선직각 성분을 제거할지 정합니다. UAV 자력탐사 가이드라인의 통상값은 측선 간격의 4배입니다. 키우면 더 긴 파장까지 제거되어 줄무늬는 더 잘 사라지지만, 측선과 나란한 실제 지질 신호도 함께 깎입니다.">
                      차단 파장 (측선 간격 배수) — 이보다 짧은 측선직각 성분 제거 ⓘ
                    </span>
                  }
                >
                  <input
                    type="number"
                    step="0.5"
                    min="1.5"
                    max="20"
                    style={inputStyle}
                    value={transformExtraParams.microlevel_cutoff_factor}
                    onChange={(e) => setTransformExtraParams((p) => ({ ...p, microlevel_cutoff_factor: parseFloat(e.target.value) }))}
                  />
                </Field>
              ) : (
                <Field
                  label={
                    <span title="노치의 폭입니다. 측선 간격 파장을 중심으로 이 배수만큼 위아래 대역을 완화합니다.">
                      파장 대역폭 배수 (측선 간격 기준, 클수록 더 넓은 파장대를 완화) ⓘ
                    </span>
                  }
                >
                  <input
                    type="number"
                    step="0.1"
                    min="1.01"
                    style={inputStyle}
                    value={transformExtraParams.microlevel_wavelength_factor}
                    onChange={(e) => setTransformExtraParams((p) => ({ ...p, microlevel_wavelength_factor: parseFloat(e.target.value) }))}
                  />
                </Field>
              )}
            </>
          )}
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={exportGeotiffColored} onChange={(e) => setExportGeotiffColored(e.target.checked)} />
            <span>컬러 GeoTIFF로 저장 (화면에 보이는 색상 그대로, 값 아님)</span>
          </label>
          <button
            style={{ ...buttonStyle, background: "white", color: "#a9631f" }}
            disabled={exportingGeotiff || !processSummary}
            onClick={() => onExportGeotiff()}
          >
            {exportingGeotiff
              ? "내보내는 중..."
              : `현재 결과(${activeTransform === "none" ? "그리드" : activeTransform.toUpperCase()})를 GeoTIFF로 저장`}
          </button>
          <div style={{ color: "#ab9a78" }}>
            {exportGeotiffColored
              ? "화면에 보이는 컬러맵/힐쉐이드가 그대로 이미지로 저장됩니다 (Google Earth 등에서 바로 볼 때 적합, 값 재분석 불가)."
              : "실제 값(nT 등)이 그대로 저장되어 Oasis Montaj/QGIS/ArcGIS 등에서 다시 열 수 있습니다."}
          </div>
          <button style={{ ...buttonStyle, background: "white", color: "#a9631f" }} disabled={exportingXyz || !processSummary} onClick={() => onExportXyz()}>
            {exportingXyz ? "내보내는 중..." : `현재 결과(${activeTransform === "none" ? "그리드" : activeTransform.toUpperCase()})를 XYZ(텍스트)로 저장`}
          </button>
          <div style={{ color: "#ab9a78" }}>경도·위도·값 3열의 공백 구분 텍스트 파일 — GeoTIFF를 지원하지 않는 다른 프로그램에서도 열람 가능.</div>
          <button style={{ ...buttonStyle, background: "white", color: "#a9631f" }} disabled={exportingGrd || !processSummary} onClick={() => onExportGrd()}>
            {exportingGrd ? "내보내는 중..." : `현재 결과(${activeTransform === "none" ? "그리드" : activeTransform.toUpperCase()})를 Surfer GRD로 저장`}
          </button>
          <div style={{ color: "#ab9a78" }}>Surfer 6 Binary Grid(.grd, DSBB) 형식 — Golden Software Surfer에서 바로 열람 가능.</div>
          <button style={{ ...buttonStyle, background: "white", color: "#a9631f" }} disabled={exportingPointsCsv || !processSummary} onClick={() => onExportPointsCsv()}>
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
