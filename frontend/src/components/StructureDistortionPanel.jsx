const sectionStyle = { border: "1px solid #e5e7eb", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #d1d5db" };
const buttonStyle = {
  padding: "7px 10px",
  fontSize: 12,
  fontWeight: 600,
  borderRadius: 6,
  border: "1px solid #0f766e",
  background: "#0f766e",
  color: "white",
  cursor: "pointer",
};
const applyButtonStyle = { ...buttonStyle, border: "1px solid #2563eb", background: "#2563eb" };
const listStyle = { maxHeight: 240, overflowY: "auto", border: "1px solid #f3f4f6", borderRadius: 4, padding: "4px 6px" };
const rowStyle = { display: "flex", alignItems: "center", gap: 6, padding: "3px 0", borderBottom: "1px solid #f9fafb" };

function Field({ label, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ color: "#4b5563" }}>{label}</span>
      {children}
    </label>
  );
}

export default function StructureDistortionPanel({
  ready,
  params,
  setParams,
  onRun,
  running,
  result,
  error,
  showOnMap,
  setShowOnMap,
  selectedPolygonIndices,
  onTogglePolygon,
  selectedAnomalyIndices,
  onToggleAnomaly,
  onApplySelected,
  applying,
}) {
  const update = (key, val) => setParams((p) => ({ ...p, [key]: val }));
  const nSelected = (selectedPolygonIndices?.size || 0) + (selectedAnomalyIndices?.size || 0);

  return (
    <div style={sectionStyle}>
      <div style={bodyStyle}>
        <div style={{ color: "#6b7280" }}>
          OpenStreetMap의 건물·도로 위치 정보와, 근지표 표적탐지와 같은 원리의 국지 이상 자동탐지를 함께 사용해 지상구조물(건물,
          도로 등)로 인한 자력 왜곡 후보를 찾습니다. 건물/도로 근처에서 실제로 뾰족한 이상이 함께 발견되면 "구조물 매칭"으로,
          매칭되는 지도상 구조물 없이 이상만 발견되면 "미매칭"(지질이상일 수 있으므로 직접 확인 필요)으로 구분해 보여줍니다.
          결과를 검토한 뒤 원하는 항목만 선택해 기존 "왜곡 영역 스무딩"을 한번에 일괄 적용할 수 있습니다.
        </div>
        <Field label="건물 버퍼 반경 (m) — 건물 외곽에서 이 거리까지 왜곡 영향권으로 간주">
          <input type="number" style={inputStyle} value={params.building_buffer_m} onChange={(e) => update("building_buffer_m", parseFloat(e.target.value))} />
        </Field>
        <Field label="도로 버퍼 반경 (m)">
          <input type="number" style={inputStyle} value={params.road_buffer_m} onChange={(e) => update("road_buffer_m", parseFloat(e.target.value))} />
        </Field>
        <Field label="탐지 격자 크기 (m)">
          <input type="number" style={inputStyle} value={params.cell_size_m} onChange={(e) => update("cell_size_m", parseFloat(e.target.value))} />
        </Field>
        <Field label="임계값 배수 (배경 표준편차의 몇 배부터 이상으로 볼지)">
          <input type="number" style={inputStyle} value={params.threshold_k} onChange={(e) => update("threshold_k", parseFloat(e.target.value))} />
        </Field>
        <Field label="최대 이상 크기 (m) — 이보다 넓게 퍼진 이상은 지질 배경으로 간주해 제외">
          <input type="number" style={inputStyle} value={params.max_footprint_m} onChange={(e) => update("max_footprint_m", parseFloat(e.target.value))} />
        </Field>
        <Field label="최소 적합도 (0~1) — 낮을수록 더 많은(불확실한) 후보 포함">
          <input
            type="number"
            step="0.05"
            min="0"
            max="1"
            style={inputStyle}
            value={params.min_fit_quality}
            onChange={(e) => update("min_fit_quality", parseFloat(e.target.value))}
          />
        </Field>
        <button style={buttonStyle} disabled={!ready || running} onClick={onRun}>
          {running ? "탐지 중... (OpenStreetMap 조회 포함, 시간이 걸릴 수 있습니다)" : "지상구조물 왜곡 자동탐지 실행"}
        </button>
        {error && <div style={{ color: "#dc2626" }}>{error}</div>}
        {result && (
          <div style={{ color: "#374151", display: "flex", flexDirection: "column", gap: 8 }}>
            <div>
              건물 {result.n_buildings}개 / 도로 {result.n_roads}개 (버퍼 병합 후 영역 {result.n_structure_polygons}개) ·
              신호기반 이상후보 {result.n_anomalies}개 (그 중 구조물 매칭 {result.n_matched}개, 미매칭 {result.n_anomalies - result.n_matched}개)
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input type="checkbox" checked={showOnMap} onChange={(e) => setShowOnMap(e.target.checked)} />
              <span>지도에 결과 표시</span>
            </label>

            {result.structure_polygons.length > 0 && (
              <div>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>구조물 영역 (건물/도로 버퍼)</div>
                <div style={listStyle}>
                  {result.structure_polygons.map((_poly, i) => (
                    <label key={i} style={rowStyle}>
                      <input type="checkbox" checked={selectedPolygonIndices.has(i)} onChange={() => onTogglePolygon(i)} />
                      <span>영역 #{i + 1}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}

            {result.anomalies.length > 0 && (
              <div>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>신호기반 이상후보</div>
                <div style={listStyle}>
                  {result.anomalies.map((a, i) => (
                    <label key={i} style={rowStyle}>
                      <input type="checkbox" checked={selectedAnomalyIndices.has(i)} onChange={() => onToggleAnomaly(i)} />
                      <span style={{ color: a.matched_structure ? "#0f766e" : "#b45309" }}>
                        {a.matched_structure ? "매칭" : "미매칭"}
                      </span>
                      <span>
                        {a.peak_anomaly_nt.toFixed(1)} nT · {a.footprint_m.toFixed(1)}m · 적합도 {a.fit_quality.toFixed(2)}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            )}

            <button style={applyButtonStyle} disabled={applying || nSelected === 0} onClick={onApplySelected}>
              {applying ? "적용 중..." : `선택한 ${nSelected}개 영역 왜곡 스무딩 일괄 적용`}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
