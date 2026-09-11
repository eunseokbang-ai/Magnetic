const sectionStyle = { border: "1px solid #e6dac0", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #ddd0b2" };
const buttonStyle = {
  padding: "6px 10px",
  fontSize: 12,
  fontWeight: 600,
  borderRadius: 6,
  border: "1px solid #b45309",
  background: "#b45309",
  color: "white",
  cursor: "pointer",
};
const subHeaderStyle = { fontWeight: 600, color: "#4a3d28", marginTop: 4 };
const hrStyle = { border: "none", borderTop: "1px solid #e6dac0", margin: "4px 0" };

function Field({ label, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ color: "#6b5c42" }}>{label}</span>
      {children}
    </label>
  );
}

function DepthStatsLine({ result }) {
  if (!result) return null;
  if (!result.available) return <div style={{ color: "#ab9a78" }}>{result.reason}</div>;
  const s = result.depth_stats;
  return (
    <div style={{ color: "#4a3d28" }}>
      {result.n_points}개 지점 — 심도 최소 {s.min.toFixed(1)}m / 중앙값 {s.median.toFixed(1)}m / 최대 {s.max.toFixed(1)}m
    </div>
  );
}

export default function DepthEstimationPanel({
  ready,
  cellSizeM,
  setCellSizeM,
  tilt,
  analyticSignal,
  spectral,
}) {
  return (
    <div style={sectionStyle}>
      <div style={bodyStyle}>
        <div style={{ color: "#8a7a5c" }}>
          3차원 역산/오일러 디컨볼루션보다 훨씬 빠른 개략 심도추정 도구 3종입니다. 단순한 2차원 접촉면/얇은 판 모델을 가정하므로
          정밀한 형상 추정에는 3차원 역산을, 여러 소스의 구조지수 기반 정밀 추정에는 오일러 디컨볼루션을 사용하세요.
        </div>
        <Field label="공통 그리딩 셀 크기 (m)">
          <input type="number" style={inputStyle} value={cellSizeM} onChange={(e) => setCellSizeM(parseFloat(e.target.value))} />
        </Field>

        <hr style={hrStyle} />
        <div style={subHeaderStyle}>틸트각 심도추정 (Tilt-depth, Salem et al. 2007)</div>
        <div style={{ color: "#8a7a5c" }}>
          틸트각이 0°를 지나는 모든 지점에서, 그 지점의 틸트각 기울기로부터 곧바로 심도를 계산합니다(등고선 추적 불필요). 수직에
          가까운 접촉면/얇은 판 모델에 정확합니다.
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <Field label="최소 심도 (m)">
            <input type="number" style={inputStyle} value={tilt.params.min_depth_m} onChange={(e) => tilt.setParams((p) => ({ ...p, min_depth_m: parseFloat(e.target.value) }))} />
          </Field>
          <Field label="최대 심도 (m)">
            <input type="number" style={inputStyle} value={tilt.params.max_depth_m} onChange={(e) => tilt.setParams((p) => ({ ...p, max_depth_m: parseFloat(e.target.value) }))} />
          </Field>
        </div>
        <button style={buttonStyle} disabled={!ready || tilt.running} onClick={tilt.onRun}>
          {tilt.running ? "계산 중..." : "틸트각 심도추정 실행"}
        </button>
        {tilt.error && <div style={{ color: "#dc2626" }}>{tilt.error}</div>}
        <DepthStatsLine result={tilt.result} />
        {tilt.result?.available && (
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={tilt.showOnMap} onChange={(e) => tilt.setShowOnMap(e.target.checked)} />
            <span>지도에 표시 (파랑=얕음 ~ 빨강=깊음)</span>
          </label>
        )}

        <hr style={hrStyle} />
        <div style={subHeaderStyle}>Analytic Signal 심도추정 (반치폭법, Roest et al. 1992)</div>
        <div style={{ color: "#8a7a5c" }}>
          Analytic signal 진폭의 각 극댓값(피크)에서, 피크 진폭의 절반이 되는 지점까지의 거리(반치폭)를 심도로 추정합니다.
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <Field label="극댓값 임계 백분위수 (%)">
            <input
              type="number"
              style={inputStyle}
              value={analyticSignal.params.percentile_threshold}
              onChange={(e) => analyticSignal.setParams((p) => ({ ...p, percentile_threshold: parseFloat(e.target.value) }))}
            />
          </Field>
          <Field label="탐색 반경 (셀)">
            <input
              type="number"
              style={inputStyle}
              value={analyticSignal.params.search_radius_cells}
              onChange={(e) => analyticSignal.setParams((p) => ({ ...p, search_radius_cells: parseInt(e.target.value, 10) }))}
            />
          </Field>
        </div>
        <button style={buttonStyle} disabled={!ready || analyticSignal.running} onClick={analyticSignal.onRun}>
          {analyticSignal.running ? "계산 중..." : "AS 심도추정 실행"}
        </button>
        {analyticSignal.error && <div style={{ color: "#dc2626" }}>{analyticSignal.error}</div>}
        <DepthStatsLine result={analyticSignal.result} />
        {analyticSignal.result?.available && (
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={analyticSignal.showOnMap} onChange={(e) => analyticSignal.setShowOnMap(e.target.checked)} />
            <span>지도에 표시 (파랑=얕음 ~ 빨강=깊음)</span>
          </label>
        )}

        <hr style={hrStyle} />
        <div style={subHeaderStyle}>스펙트럼 심도추정 (Spector &amp; Grant 1970)</div>
        <div style={{ color: "#8a7a5c" }}>
          그리드 전체의 방사평균 파워스펙트럼 기울기로부터, 조사구역 전체를 대표하는 단일 평균 소스 심도를 추정합니다(개별 위치가
          아닌 앙상블 평균값 - 3차원 역산 메쉬 자동설정에도 쓰이는 것과 같은 계산입니다).
        </div>
        <button style={buttonStyle} disabled={!ready || spectral.running} onClick={spectral.onRun}>
          {spectral.running ? "계산 중..." : "스펙트럼 심도추정 실행"}
        </button>
        {spectral.error && <div style={{ color: "#dc2626" }}>{spectral.error}</div>}
        {spectral.result && !spectral.result.available && <div style={{ color: "#ab9a78" }}>{spectral.result.reason}</div>}
        {spectral.result?.available && (
          <div style={{ color: "#4a3d28" }}>평균 소스 심도: {spectral.result.depth_m.toFixed(1)} m</div>
        )}
      </div>
    </div>
  );
}
