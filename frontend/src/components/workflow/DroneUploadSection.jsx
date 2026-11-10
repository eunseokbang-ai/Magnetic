import { useState } from "react";
import { useApp } from "../../AppContext";
import { ProgressBar, bodyStyle, inputStyle, sectionStyle, summaryStyle } from "./shared";

// Workflow step 1: uploading the drone magnetometer files, and what was read from them.
// One section of components/WorkflowSteps.jsx; it reads App's state with useApp().
export default function DroneUploadSection() {
  const { droneSummary, droneUploadProgress, handleUploadDrone } = useApp();

  const [droneFileNames, setDroneFileNames] = useState([]);

  return (
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
              handleUploadDrone(files);
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
            {droneSummary.sensor_labels && (
              <>
                <br />
                <span style={{ color: "#166534" }}>
                  센서 2개({droneSummary.sensor_labels.join(" · ")})가 함께 기록된 자료입니다 - 처리에 쓸 센서는 3번 자료 처리에서 고릅니다.
                </span>
              </>
            )}
            {droneSummary.n_files_placeholder_date > 0 && (
              <>
                <br />
                <span style={{ color: "#b45309" }}>
                  {droneSummary.n_files_placeholder_date}개 파일은 이름에 날짜가 없어 오늘 날짜로 시각을 매겼습니다 - 베이스 자료와 시각이 맞지
                  않으니, 원래 파일명(Log_Drone_YYYYMMDD_HHMMSS.csv)으로 다시 올리세요.
                </span>
              </>
            )}
            {droneSummary.n_files_total > 0 && !droneSummary.sensor_labels && (
              <>
                <br />
                {droneSummary.n_files_using_precompensated_mag > 0 ? (
                  <span style={{ color: "#166534" }}>
                    사전 보정값(MagComp) 사용: {droneSummary.n_files_total}개 중{" "}
                    {droneSummary.n_files_using_precompensated_mag}개 파일
                    {droneSummary.n_files_using_precompensated_mag < droneSummary.n_files_total &&
                      " (나머지는 MagComp 컬럼이 없어 원본 Mag 사용)"}
                  </span>
                ) : (
                  <span style={{ color: "#8a7a5c" }}>
                    MagComp 컬럼이 없어 원본 Mag 값을 사용합니다 - 헤딩/시스템 오차 보정이 이미 된
                    "-comp.csv"를 올렸다면, 그 파일에 "Keep Raw Data"로 만든 MagComp 컬럼이 있는지 확인하세요.
                  </span>
                )}
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
  );
}
