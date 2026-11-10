import { useState } from "react";
import { useApp } from "../../AppContext";
import IntermagnetPanel from "../IntermagnetPanel";
import { Field, ProgressBar, bodyStyle, buttonStyle, inputStyle, sectionStyle, summaryStyle } from "./shared";

// Workflow step 2: the base station (diurnal) record - an uploaded log, an
// INTERMAGNET observatory, or none - and its QC options.
// One section of components/WorkflowSteps.jsx; it reads App's state with useApp().
export default function BaseStationSection() {
  const {
    baseSummary, baseTimeseriesLoading, baseUploadProgress, droneSummary, handleApplyIntermagnet,
    handleApplyNearestIntermagnet, handleCancelIntermagnetPreview, handleCancelNearestIntermagnetPreview,
    handleExportNearestIntermagnetCsv, handleFetchIntermagnet, handleFetchNearestIntermagnet,
    handleShowBaseTimeseries, handleShowNearestIntermagnetComparison, handleUploadBase, handleUploadIaga2002,
    intermagnetApplying, intermagnetError, intermagnetLoading, intermagnetPreview,
    nearestIntermagnetApplying, nearestIntermagnetComparisonLoading, nearestIntermagnetCsvFilename,
    nearestIntermagnetError, nearestIntermagnetHasResult, nearestIntermagnetLoading,
    nearestIntermagnetPreview, processParams, processSummary, setNearestIntermagnetCsvFilename,
    setProcessParams,
  } = useApp();

  const [baseFileNames, setBaseFileNames] = useState([]);
  const dp = processParams.diurnal_params;
  const updateDiurnal = (key, val) => setProcessParams((p) => ({ ...p, diurnal_params: { ...p.diurnal_params, [key]: val } }));

  return (
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
              handleUploadBase(files);
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

        {handleUploadIaga2002 && (
          <details style={{ marginTop: 6 }}>
            <summary style={{ cursor: "pointer", color: "#4a3d28" }}>
              INTERMAGNET 관측소 자료 사용 (베이스 자료 대신 - 로컬 일변화는 반영 못하지만 태양풍에 의한 지역/전지구적 변화는 반영)
            </summary>
            <IntermagnetPanel
              onUploadIaga2002={handleUploadIaga2002}
              onFetchIntermagnet={handleFetchIntermagnet}
              onApplyIntermagnet={handleApplyIntermagnet}
              onCancelPreview={handleCancelIntermagnetPreview}
              preview={intermagnetPreview}
              loading={intermagnetLoading}
              applying={intermagnetApplying}
              error={intermagnetError}
              onFetchNearestIntermagnet={handleFetchNearestIntermagnet}
              onApplyNearestIntermagnet={handleApplyNearestIntermagnet}
              onCancelNearestPreview={handleCancelNearestIntermagnetPreview}
              onExportNearestCsv={handleExportNearestIntermagnetCsv}
              nearestCsvFilename={nearestIntermagnetCsvFilename}
              onNearestCsvFilenameChange={setNearestIntermagnetCsvFilename}
              onShowNearestComparison={handleShowNearestIntermagnetComparison}
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

        {handleShowBaseTimeseries && (
          <button
            type="button"
            onClick={handleShowBaseTimeseries}
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

            <Field label="기준국 구간 레벨 보정 — 기준국을 옮겼는지 여부에 따라 다르게 처리합니다">
              <select
                style={inputStyle}
                value={processParams.base_qc_params.segment_mode}
                onChange={(e) =>
                  setProcessParams((p) => ({
                    ...p,
                    base_qc_params: { ...p.base_qc_params, segment_mode: e.target.value },
                  }))
                }
              >
                <option value="steps">기록 중간 불연속만 보정 (기본, 기준국 고정)</option>
                <option value="all">기준국을 옮김 — 기록이 끊긴 구간도 레벨 맞춤</option>
                <option value="off">보정 안 함 (측정·표시만)</option>
              </select>
            </Field>
            <div style={{ fontSize: 11, color: "#7b6f5a", marginTop: -4, marginBottom: 6 }}>
              위치가 고정된 기준국의 날짜 간 레벨 차이(보통 20nT 이내)는 실제 지자기 변화라
              드론 자료에 전달되어야 합니다. 기준국을 옮겨 생긴 단차(수십~수백 nT)만 제거 대상입니다.
            </div>
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
  );
}
