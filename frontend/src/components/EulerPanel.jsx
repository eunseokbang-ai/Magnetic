const sectionStyle = { border: "1px solid #e5e7eb", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
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

function Field({ label, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ color: "#4b5563" }}>{label}</span>
      {children}
    </label>
  );
}

export default function EulerPanel({
  ready,
  structuralIndex,
  setStructuralIndex,
  windowSize,
  setWindowSize,
  maxUncertaintyPct,
  setMaxUncertaintyPct,
  onRun,
  running,
  result,
  error,
  showSolutions,
  setShowSolutions,
}) {
  return (
    <div style={sectionStyle}>
      <div style={bodyStyle}>
        <div style={{ color: "#6b7280" }}>
          그리드된 이상값의 수평/수직 미분을 이용해 이상체의 위치와 깊이를 자동으로 추정합니다 (Euler's homogeneity equation). 3차원
          역산보다 훨씬 빠르지만 개략적인 결과입니다.
        </div>
        <Field label="구조지수 (Structural Index)">
          <select style={inputStyle} value={structuralIndex} onChange={(e) => setStructuralIndex(parseFloat(e.target.value))}>
            <option value={0}>0 - 접촉면/단층 경계 (Contact/Fault)</option>
            <option value={1}>1 - 얇은 암맥/실 경계 (Dyke/Sill edge)</option>
            <option value={2}>2 - 파이프형 (수직 원통, Pipe)</option>
            <option value={3}>3 - 구형/점자기쌍극자 (Sphere)</option>
          </select>
        </Field>
        <Field label="탐색창 크기 (m)">
          <input type="number" style={inputStyle} value={windowSize} onChange={(e) => setWindowSize(parseFloat(e.target.value))} />
        </Field>
        <Field label="깊이 불확실도 허용 한계 (%) — 낮을수록 더 신뢰도 높은 결과만 표시">
          <input
            type="number"
            style={inputStyle}
            value={maxUncertaintyPct}
            onChange={(e) => setMaxUncertaintyPct(parseFloat(e.target.value))}
          />
        </Field>
        <button style={buttonStyle} disabled={!ready || running} onClick={onRun}>
          {running ? "계산 중..." : "오일러 디컨볼루션 실행"}
        </button>
        {error && <div style={{ color: "#dc2626" }}>{error}</div>}
        {result && (
          <div style={{ color: "#374151" }}>
            해 {result.n_solutions}개
            {result.depth_stats && (
              <>
                <br />
                깊이 범위: {result.depth_stats.min?.toFixed(1)} ~ {result.depth_stats.max?.toFixed(1)} m (평균{" "}
                {result.depth_stats.mean?.toFixed(1)} m)
              </>
            )}
            <label style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6 }}>
              <input type="checkbox" checked={showSolutions} onChange={(e) => setShowSolutions(e.target.checked)} />
              <span>지도에 결과 표시</span>
            </label>
          </div>
        )}
      </div>
    </div>
  );
}
