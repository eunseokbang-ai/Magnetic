import { useState } from "react";
import { useApp } from "../../AppContext";
import { Field, ProgressBar, bodyStyle, inputStyle, sectionStyle, summaryStyle } from "./shared";

// Workflow step 2-1: the optional heading-effect calibration flight.
// One section of components/WorkflowSteps.jsx; it reads App's state with useApp().
export default function HeadingCalibrationSection() {
  const {
    handleUploadHeadingCalibration, headingCalibrationSummary, headingCalibrationUploadProgress,
    processParams, setProcessParams,
  } = useApp();

  const [headingCalFileNames, setHeadingCalFileNames] = useState([]);
  const hec = processParams.heading_effect_calibration;
  const updateHeadingCal = (key, val) =>
    setProcessParams((p) => ({ ...p, heading_effect_calibration: { ...p.heading_effect_calibration, [key]: val } }));

  return (
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
              handleUploadHeadingCalibration(files);
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
  );
}
