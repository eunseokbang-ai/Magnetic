import { useApp } from "../../AppContext";
import Legend from "../Legend";
import { SensorPairBox } from "./shared";

// Right column, last part: what describes the map as it is drawn now - the
// measuring tool, the legend, and the notes and warnings attached to the grid
// on screen.
// A region of App's screen; it reads App's state with useApp().
export default function MapReadout() {
  const {
    cmapName, handleClearMeasurements, handleJumpToExtremum, handleRemoveMeasurement, hoverPoint,
    legendLabel, legendRange, legendStats, manualRange, measureMode, measurements, overlay, processSummary,
    setCmapName, setManualRange, toggleMeasureMode,
  } = useApp();

  return (
    <>
      <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>17. 거리/면적 측정</h2>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 10 }}>
        <div style={{ display: "flex", gap: 6 }}>
          <button
            onClick={() => toggleMeasureMode("distance")}
            style={{
              flex: 1,
              padding: "6px 8px",
              fontSize: 12,
              borderRadius: 6,
              border: `1px solid ${measureMode === "distance" ? "#dc2626" : "#ddd0b2"}`,
              background: measureMode === "distance" ? "#dc2626" : "white",
              color: measureMode === "distance" ? "white" : "#4a3d28",
              cursor: "pointer",
            }}
            title="지도를 클릭해 선을 그리면(더블클릭으로 종료) 그 경로의 총 거리를 계산해 표시합니다. 이상대 길이 등을 잴 때 사용하세요. 계속 여러 선을 그릴 수 있고, 버튼을 다시 누르면 그리기 모드만 꺼집니다(측정 결과는 남습니다)."
          >
            {measureMode === "distance" ? "거리 측정 중 (다시 누르면 종료)" : "거리 측정"}
          </button>
          <button
            onClick={() => toggleMeasureMode("area")}
            style={{
              flex: 1,
              padding: "6px 8px",
              fontSize: 12,
              borderRadius: 6,
              border: `1px solid ${measureMode === "area" ? "#7c3aed" : "#ddd0b2"}`,
              background: measureMode === "area" ? "#7c3aed" : "white",
              color: measureMode === "area" ? "white" : "#4a3d28",
              cursor: "pointer",
            }}
            title="지도를 클릭해 다각형을 그리면(더블클릭 또는 시작점 클릭으로 종료) 그 영역의 면적을 계산해 표시합니다. 이상대 면적 등을 잴 때 사용하세요. 계속 여러 영역을 그릴 수 있고, 버튼을 다시 누르면 그리기 모드만 꺼집니다(측정 결과는 남습니다)."
          >
            {measureMode === "area" ? "면적 측정 중 (다시 누르면 종료)" : "면적 측정"}
          </button>
        </div>
        {measurements.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {measurements.map((m) => (
              <div
                key={m.id}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  fontSize: 12,
                  background: "white",
                  border: "1px solid #e6dac0",
                  borderRadius: 6,
                  padding: "4px 8px",
                }}
              >
                <span>
                  {m.label}: {m.valueText}
                </span>
                <button
                  onClick={() => handleRemoveMeasurement(m.id)}
                  style={{ border: "none", background: "transparent", color: "#ab9a78", cursor: "pointer", fontSize: 13, lineHeight: 1 }}
                  title="이 측정 삭제"
                >
                  ✕
                </button>
              </div>
            ))}
            <button
              onClick={handleClearMeasurements}
              style={{ alignSelf: "flex-start", fontSize: 11, color: "#8a7a5c", background: "transparent", border: "none", cursor: "pointer", textDecoration: "underline", padding: 0 }}
            >
              전체 지우기
            </button>
          </div>
        )}
      </div>

      <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>범례</h2>
      <Legend
        label={legendLabel}
        unit="nT"
        vmin={legendRange.vmin}
        vmax={legendRange.vmax}
        ticks={overlay?.legend_ticks}
        stretch={overlay?.stretch}
        cmapName={cmapName}
        onCmapChange={setCmapName}
        manualRange={manualRange}
        onManualRangeChange={setManualRange}
        stats={legendStats}
        hoverPoint={hoverPoint}
        extrema={overlay?.extrema}
        onJumpToExtremum={handleJumpToExtremum}
      />
      {overlay?.cell_size_guideline_warning && (
        <div style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, color: "#b45309", background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 6 }}>
          ⚠ {overlay.cell_size_guideline_warning}
        </div>
      )}
      {overlay?.altitude_warning && (
        <div style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, color: "#b45309", background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 6 }}>
          ⚠ {overlay.altitude_warning}
        </div>
      )}
      {(overlay?.regional_removal?.method && overlay.regional_removal.method !== "none") || overlay?.derived_from_rtp?.requested ? (
        // What the grid on the map was made from, when it is not the
        // anomaly as it comes: a number on it means something else then.
        <div data-testid="base-grid-readout" style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, color: "#4a3d28", background: "#f7f2e6", border: "1px solid #e6dac0", borderRadius: 6 }}>
          {overlay.regional_removal?.method && overlay.regional_removal.method !== "none" && (
            <div>
              기준 격자: 광역 배경 제거 —{" "}
              {overlay.regional_removal.method === "level"
                ? `수준 ${overlay.regional_removal.level_nt >= 0 ? "+" : ""}${overlay.regional_removal.level_nt.toFixed(1)} nT`
                : `${overlay.regional_removal.method === "plane" ? "1차" : "2차"} 추세면 ${overlay.regional_removal.min_nt.toFixed(0)} ~ ${overlay.regional_removal.max_nt.toFixed(0)} nT`}
              {overlay.regional_removal.method !== "level" && ` (제거 전 중앙값 ${overlay.regional_removal.data_median_before_nt.toFixed(1)} nT)`}
            </div>
          )}
          {overlay.derived_from_rtp?.applied && <div>RTP 기준 파생: 기준 격자를 RTP한 것에서 계산했습니다.</div>}
          {overlay.derived_from_rtp?.requested && !overlay.derived_from_rtp.applied && overlay.derived_from_rtp.reason && (
            <div style={{ color: "#8a7a5c" }}>RTP 기준 파생 미적용: {overlay.derived_from_rtp.reason}</div>
          )}
        </div>
      ) : null}
      {overlay?.rtp_latitude_warning && (
        <div style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, color: "#b45309", background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 6 }}>
          ⚠ {overlay.rtp_latitude_warning}
        </div>
      )}
      {overlay?.raw_cell_derivative_warning && (
        <div style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, color: "#b45309", background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 6 }}>
          ⚠ {overlay.raw_cell_derivative_warning}
        </div>
      )}
      {overlay?.derivative_presmooth?.applies && (
        // Says whether the across-line filter actually ran. It can
        // decline for reasons invisible on the map, and a filter that
        // quietly does nothing looks exactly like a broken one.
        <div style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, borderRadius: 6,
                      color: overlay.derivative_presmooth.applied ? "#4a3d28" : "#b45309",
                      background: overlay.derivative_presmooth.applied ? "#f7f2e6" : "#fffbeb",
                      border: `1px solid ${overlay.derivative_presmooth.applied ? "#e6dac0" : "#fde68a"}` }}>
          {overlay.derivative_presmooth.applied ? (
            <>
              측선직각 평활 적용됨 — 차단 파장 {overlay.derivative_presmooth.cutoff_m}m (
              {overlay.derivative_presmooth.cutoff_cells}셀
              {overlay.derivative_presmooth.from_line_spacing
                ? `, 측선 간격 ${overlay.derivative_presmooth.line_spacing_m?.toFixed(0)}m 기준`
                : ", 측선 간격을 몰라 셀 크기 기준 기본값"}
              )
            </>
          ) : (
            <>⚠ 측선직각 평활이 적용되지 않았습니다: {overlay.derivative_presmooth.reason}</>
          )}
        </div>
      )}
      {overlay?.boundary_continuation?.applies && (
        // It declines on a grid with no gaps, which is correct but
        // invisible - and if it is off, the derivative is back to
        // inheriting whatever sits on the survey boundary.
        <div style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, borderRadius: 6,
                      color: overlay.boundary_continuation.applied ? "#4a3d28" : "#8a7a5c",
                      background: overlay.boundary_continuation.applied ? "#f7f2e6" : "#faf7f0",
                      border: `1px solid ${overlay.boundary_continuation.applied ? "#e6dac0" : "#e9e2d2"}` }}>
          {overlay.boundary_continuation.applied ? (
            <>
              탐사 바깥 등가 소스층 연속 적용됨 — 빈 땅 {overlay.boundary_continuation.gap_pct}% (
              {overlay.boundary_continuation.n_gap_cells?.toLocaleString()}셀)을 측정값 대신 소스 모델의 자기장으로 채웠습니다
            </>
          ) : (
            <>탐사 바깥 등가 소스층 연속: 적용 안 함 — {overlay.boundary_continuation.reason}</>
          )}
        </div>
      )}
      {overlay?.equivalent_source?.applies && (
        // The misfit is the number that makes a smoothed deliverable
        // defensible: it says in nanotesla what was given up.
        <div style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, borderRadius: 6,
                      color: overlay.equivalent_source.applied ? "#4a3d28" : "#b45309",
                      background: overlay.equivalent_source.applied ? "#f7f2e6" : "#fffbeb",
                      border: `1px solid ${overlay.equivalent_source.applied ? "#e6dac0" : "#fde68a"}` }}>
          {overlay.equivalent_source.applied ? (
            <>
              <b>등가 소스층 적용</b> — 깊이 {overlay.equivalent_source.depth_m}m (측선 간격의{" "}
              {overlay.equivalent_source.factor}배), 실측 재현 오차{" "}
              <b>{overlay.equivalent_source.misfit_nt} nT</b>
              {overlay.equivalent_source.field_std_nt != null && (
                <> (자기장 표준편차 {overlay.equivalent_source.field_std_nt} nT의 {overlay.equivalent_source.misfit_pct}%)</>
              )}
              <div style={{ marginTop: 3, color: "#8a7a5c" }}>
                이 수치가 "무엇을 얼마나 매끈하게 했는지"의 근거입니다 - 보고서에 그대로 쓰실 수 있습니다.
              </div>
            </>
          ) : (
            <>⚠ 등가 소스층이 적용되지 않았습니다: {overlay.equivalent_source.reason}</>
          )}
        </div>
      )}
      {overlay?.boundary_margin && overlay.boundary_margin.mode !== "off" && (
        // How much of the map the margin covers, and - just as important
        // - that the streak it marks fades slowly, so being outside the
        // margin is not a clean bill of health.
        <div style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, borderRadius: 6,
                      color: overlay.boundary_margin.applied ? "#4a3d28" : "#b45309",
                      background: overlay.boundary_margin.applied ? "#f7f2e6" : "#fffbeb",
                      border: `1px solid ${overlay.boundary_margin.applied ? "#e6dac0" : "#fde68a"}` }}>
          {overlay.boundary_margin.applied ? (
            <>
              <b>경계 여백 {overlay.boundary_margin.mode === "mask" ? "가림" : "표시"}</b> — 폭{" "}
              {overlay.boundary_margin.margin_m}m ({overlay.boundary_margin.margin_cells}셀
              {overlay.boundary_margin.from_extrapolation_radius ? ", 격자 외삽 거리 기준" : ""}), 그리드의{" "}
              {overlay.boundary_margin.pct_of_grid}%
              {overlay.boundary_margin.reason && <div style={{ marginTop: 3 }}>⚠ {overlay.boundary_margin.reason}</div>}
              <div style={{ marginTop: 3, color: "#8a7a5c" }}>{overlay.boundary_margin.note}</div>
            </>
          ) : (
            <>⚠ 경계 여백이 적용되지 않았습니다: {overlay.boundary_margin.reason}</>
          )}
        </div>
      )}
      {overlay?.striping?.available && (
        <div
          style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, color: "#4a3d28", background: "#f7f2e6", border: "1px solid #e6dac0", borderRadius: 6 }}
          title="지금 보고 있는 그리드의 줄무늬를 파장으로 측정한 값입니다. 줄무늬는 원인마다 파장이 다르고 처방도 반대라서, 눈으로 보는 대신 이 숫자로 원인을 가릅니다. 세기(%)는 그리드 전체 변동 중 줄무늬 대역이 차지하는 비율이라, 설정을 바꿔가며 비교하는 데 쓰면 됩니다."
        >
          <b>줄무늬 진단</b> — 파장 {overlay.striping.peak_wavelength_m}m (
          {overlay.striping.peak_wavelength_cells}셀
          {overlay.striping.peak_wavelength_spacings != null && `, 측선 간격의 ${overlay.striping.peak_wavelength_spacings}배`}
          ), 세기 {overlay.striping.amplitude_pct}% ⓘ
          <div style={{ marginTop: 3 }}>{overlay.striping.verdict}</div>
          {overlay.striping_base_grid?.available && (
            // Which of the two carries it says whether the transform made
            // the striping or only revealed what the base grid already had.
            <div style={{ marginTop: 3, color: "#8a7a5c" }}>
              원본 그리드 자체: 파장 {overlay.striping_base_grid.peak_wavelength_m}m, 세기{" "}
              {overlay.striping_base_grid.amplitude_pct}%
            </div>
          )}
        </div>
      )}
      {processSummary?.line_resolution?.available && (
        // Whether the striping above is something processing can act on
        // at all. Measured on the points, before gridding: once gridded,
        // the gap between the lines has already been filled with a guess
        // and the question cannot be asked any more.
        <div
          style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, borderRadius: 6,
                   color: processSummary.line_resolution.unresolved_pct >= 50 ? "#b45309" : "#4a3d28",
                   background: processSummary.line_resolution.unresolved_pct >= 50 ? "#fffbeb" : "#f7f2e6",
                   border: `1px solid ${processSummary.line_resolution.unresolved_pct >= 50 ? "#fde68a" : "#e6dac0"}` }}
          title="각 측선의 값을 양옆 두 측선의 평균과 비교합니다. 측선 간격이 자기장 변화를 따라잡고 있다면 가운데 측선은 양옆의 평균으로 거의 예측되고, 예측되지 않는 부분이 곧 '측선 사이에서는 측정된 적이 없는' 성분입니다. 격자는 그 부분을 보간으로 지어낼 수밖에 없고, 미분 계열 파생그리드에서 측선방향 줄무늬로 드러납니다. 측선방향 평활을 아무리 키워도 이 비율은 거의 줄지 않습니다 - 측선 사이의 정보는 애초에 없기 때문입니다."
        >
          <b>측선 간격 분해능</b> — 이웃 측선이 못 보는 성분{" "}
          {processSummary.line_resolution.unresolved_pct}% (측선 간격{" "}
          {processSummary.line_resolution.line_spacing_m}m, 비교 {processSummary.line_resolution.n_triples}쌍) ⓘ
          <div style={{ marginTop: 3 }}>{processSummary.line_resolution.verdict}</div>
        </div>
      )}
      {processSummary?.sensor_pair?.available && <SensorPairBox pair={processSummary.sensor_pair} />}
    </>
  );
}
