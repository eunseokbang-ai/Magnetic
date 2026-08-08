const sectionStyle = { border: "1px solid #e5e7eb", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #d1d5db" };
const buttonStyle = {
  padding: "7px 10px",
  fontSize: 12,
  fontWeight: 600,
  borderRadius: 6,
  border: "1px solid #b91c1c",
  background: "#b91c1c",
  color: "white",
  cursor: "pointer",
};
const tableStyle = { width: "100%", borderCollapse: "collapse", fontSize: 11 };
const thStyle = { textAlign: "left", padding: "3px 4px", borderBottom: "1px solid #e5e7eb", color: "#6b7280" };
const tdStyle = { padding: "3px 4px", borderBottom: "1px solid #f3f4f6" };

function Field({ label, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ color: "#4b5563" }}>{label}</span>
      {children}
    </label>
  );
}

export default function TargetDetectionPanel({ ready, params, setParams, onRun, running, result, error, showTargets, setShowTargets, projectId, exportTargetsCsv, exportTargetsShapefile }) {
  const update = (key, val) => setParams((p) => ({ ...p, [key]: val }));

  return (
    <div style={sectionStyle}>
      <div style={bodyStyle}>
        <div style={{ color: "#6b7280" }}>
          지뢰·불발탄·은닉 차량 등 작고 국지적인 강자성 표적을 자동으로 골라내 위치·심도·상대적 철질량 크기를 추정합니다(단일 자기
          쌍극자 모델 피팅). 넓은 지역의 완만한 지질체를 가정하는 3차원 역산/오일러 디컨볼루션과 달리, 작고 뾰족한 국지 이상만 표적
          후보로 다룹니다.
        </div>
        <div style={{ color: "#b91c1c", fontWeight: 600 }}>
          ⚠ 자력탐사만으로 표적의 정확한 종류(지뢰/포탄/전차/고철 등)를 식별할 수 없습니다. 최소금속 물체는 탐지되지 않을 수 있습니다.
          실제 위치 확인·접근·처리는 반드시 EOD(폭발물처리반) 등 전문 인력이 수행해야 합니다.
        </div>
        <Field label="탐지 격자 크기 (m) — 표적 크기에 맞춰 촘촘하게">
          <input type="number" style={inputStyle} value={params.cell_size_m} onChange={(e) => update("cell_size_m", parseFloat(e.target.value))} />
        </Field>
        <Field label="진폭 임계값 직접 지정 (nT, 비워두면 배경 표준편차 기반 자동)">
          <input
            type="number"
            style={inputStyle}
            value={params.amplitude_threshold_nt ?? ""}
            onChange={(e) => update("amplitude_threshold_nt", e.target.value === "" ? null : parseFloat(e.target.value))}
          />
        </Field>
        <Field label="임계값 배수 (자동 임계값일 때, 배경 표준편차의 몇 배)">
          <input type="number" style={inputStyle} value={params.threshold_k} onChange={(e) => update("threshold_k", parseFloat(e.target.value))} />
        </Field>
        <Field label="최소 표적 크기 (m)">
          <input type="number" style={inputStyle} value={params.min_footprint_m} onChange={(e) => update("min_footprint_m", parseFloat(e.target.value))} />
        </Field>
        <Field label="최대 표적 크기 (m) — 이보다 넓게 퍼진 이상은 지질 배경으로 간주해 제외">
          <input type="number" style={inputStyle} value={params.max_footprint_m} onChange={(e) => update("max_footprint_m", parseFloat(e.target.value))} />
        </Field>
        <Field label="최대 탐색 심도 (m) — 표적탐지는 근지표용이므로 얕게 제한">
          <input type="number" style={inputStyle} value={params.max_depth_m} onChange={(e) => update("max_depth_m", parseFloat(e.target.value))} />
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
          {running ? "탐지 중..." : "표적탐지 실행"}
        </button>
        {error && <div style={{ color: "#dc2626" }}>{error}</div>}
        {result && (
          <div style={{ color: "#374151" }}>
            표적 후보 {result.n_targets}개 (임계값 {result.amplitude_threshold_nt?.toFixed(1)} nT, 격자 {result.cell_size_m} m)
            <label style={{ display: "flex", alignItems: "center", gap: 6, margin: "6px 0" }}>
              <input type="checkbox" checked={showTargets} onChange={(e) => setShowTargets(e.target.checked)} />
              <span>지도에 결과 표시</span>
            </label>
            {result.targets.length > 0 && (
              <div style={{ display: "flex", gap: 6 }}>
                <button
                  style={{ ...buttonStyle, background: "white", color: "#374151", border: "1px solid #d1d5db", flex: 1 }}
                  onClick={() => exportTargetsCsv(projectId, "targets.csv")}
                >
                  CSV로 내보내기
                </button>
                <button
                  style={{ ...buttonStyle, background: "white", color: "#374151", border: "1px solid #d1d5db", flex: 1 }}
                  onClick={() => exportTargetsShapefile(projectId, "targets_shapefile.zip")}
                >
                  Shapefile로 내보내기
                </button>
              </div>
            )}
            {result.targets.length > 0 && (
              <div style={{ maxHeight: 220, overflowY: "auto", border: "1px solid #f3f4f6", borderRadius: 4 }}>
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={thStyle}>#</th>
                      <th style={thStyle}>심도(m)</th>
                      <th style={thStyle}>모멘트(A·m²)</th>
                      <th style={thStyle}>크기등급</th>
                      <th style={thStyle}>적합도</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.targets.map((t, i) => (
                      <tr key={i}>
                        <td style={tdStyle}>{i + 1}</td>
                        <td style={tdStyle}>{t.depth_m.toFixed(2)}</td>
                        <td style={tdStyle}>{t.moment_am2.toFixed(2)}</td>
                        <td style={tdStyle}>{t.size_class}</td>
                        <td style={tdStyle}>{t.fit_quality.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
