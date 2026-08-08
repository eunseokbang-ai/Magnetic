const sectionStyle = { border: "1px solid #e5e7eb", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #d1d5db" };
const buttonStyle = {
  padding: "7px 10px",
  fontSize: 12,
  fontWeight: 600,
  borderRadius: 6,
  border: "1px solid #a16207",
  background: "#a16207",
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

export default function ContactPanel({ ready, params, setParams, onRun, running, result, error, showOnMap, setShowOnMap }) {
  const update = (key, val) => setParams((p) => ({ ...p, [key]: val }));

  return (
    <div style={sectionStyle}>
      <div style={bodyStyle}>
        <div style={{ color: "#6b7280" }}>
          THD 능선(ridge)이 여러 상향연속(upward continuation) 고도에서 지속적으로 나타나는 지점만 골라 자성 암체 경계(지질
          접촉면) 후보를 추출합니다(Archibald et al. 1999 "worming" 기법) — 리니어먼트 추출과 달리 단일 고도의 능선이 아니라
          여러 고도에서의 지속성(persistence)을 기준으로 삼아, 얕은 노이즈나 국소 소스로 인한 가짜 능선을 걸러냅니다. 지질도
          GeoTIFF 참조 레이어(12번)를 함께 켜서 실제 지질 경계와 비교해 보세요.
        </div>
        <Field label="셀 크기 (m)">
          <input type="number" style={inputStyle} value={params.cell_size_m} onChange={(e) => update("cell_size_m", parseFloat(e.target.value))} />
        </Field>
        <Field label="능선 판정 백분위수 (%) — 낮을수록 더 많은 후보 포함">
          <input
            type="number"
            style={inputStyle}
            value={params.percentile_threshold}
            onChange={(e) => update("percentile_threshold", parseFloat(e.target.value))}
          />
        </Field>
        <Field label="영속성 기준 (0~1) — 여러 고도 중 이 비율 이상에서 나타나야 인정">
          <input
            type="number"
            step="0.1"
            min="0.1"
            max="1"
            style={inputStyle}
            value={params.min_persistence}
            onChange={(e) => update("min_persistence", parseFloat(e.target.value))}
          />
        </Field>
        <Field label="최소 구성 포인트 수">
          <input
            type="number"
            style={inputStyle}
            value={params.min_segment_points}
            onChange={(e) => update("min_segment_points", parseInt(e.target.value, 10))}
          />
        </Field>
        <button style={buttonStyle} disabled={!ready || running} onClick={onRun}>
          {running ? "탐지 중..." : "자성 접촉면 탐지 실행"}
        </button>
        {error && <div style={{ color: "#dc2626" }}>{error}</div>}
        {result && !result.available && <div style={{ color: "#9ca3af" }}>{result.reason}</div>}
        {result && result.available && (
          <div style={{ color: "#374151" }}>
            접촉면 {result.n_contacts}개, 총 길이 {(result.total_length_m / 1000).toFixed(2)} km
            <label style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6 }}>
              <input type="checkbox" checked={showOnMap} onChange={(e) => setShowOnMap(e.target.checked)} />
              <span>지도에 결과 표시 (색상=영속성, 노랑=낮음 ~ 보라=높음)</span>
            </label>
          </div>
        )}
      </div>
    </div>
  );
}
