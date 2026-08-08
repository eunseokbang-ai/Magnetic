const sectionStyle = { border: "1px solid #e5e7eb", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #d1d5db" };
const buttonStyle = {
  padding: "7px 10px",
  fontSize: 12,
  fontWeight: 600,
  borderRadius: 6,
  border: "1px solid #0891b2",
  background: "#0891b2",
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

const SOURCE_LABELS = {
  thd: "THD (총수평미분)",
  as: "AS (analytic signal)",
  tilt: "틸트각",
  "1vd": "1VD (수직미분)",
};

// Lightweight inline-SVG rose diagram (a Plotly polar chart would pull in
// the ~4.4MB plotly chunk just for one small radial histogram - see
// App.jsx's note on why GuidelinePanel's Plotly import is lazy-loaded
// behind a details toggle; this panel avoids that dependency entirely).
// Mirrored across 180deg since a lineament's strike is axial (a line
// striking 30deg is the same line as one striking 210deg).
function RoseDiagram({ bins, binWidthDeg }) {
  const size = 220;
  const center = size / 2;
  const maxRadius = size / 2 - 24;
  const maxLength = Math.max(1, ...bins.map((b) => b.total_length_m));

  const wedges = [];
  bins.forEach((b, i) => {
    const r = (b.total_length_m / maxLength) * maxRadius;
    if (r <= 0) return;
    [b.strike_from_deg, b.strike_from_deg + 180].forEach((fromDeg, mirror) => {
      const toDeg = fromDeg + binWidthDeg;
      // compass azimuth (0=N, up) to SVG angle (0=right, clockwise) : svgAngle = azimuth - 90
      const a0 = ((fromDeg - 90) * Math.PI) / 180;
      const a1 = ((toDeg - 90) * Math.PI) / 180;
      const x0 = center + r * Math.cos(a0);
      const y0 = center + r * Math.sin(a0);
      const x1 = center + r * Math.cos(a1);
      const y1 = center + r * Math.sin(a1);
      wedges.push(
        <path
          key={`${i}-${mirror}`}
          d={`M ${center} ${center} L ${x0.toFixed(1)} ${y0.toFixed(1)} A ${r.toFixed(1)} ${r.toFixed(1)} 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)} Z`}
          fill={`hsl(${(b.strike_from_deg / 180) * 360}, 70%, 55%)`}
          stroke="white"
          strokeWidth="0.5"
        />
      );
    });
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {[0.25, 0.5, 0.75, 1.0].map((f) => (
        <circle key={f} cx={center} cy={center} r={maxRadius * f} fill="none" stroke="#e5e7eb" strokeWidth="0.5" />
      ))}
      <line x1={center} y1={center - maxRadius - 6} x2={center} y2={center + maxRadius + 6} stroke="#9ca3af" strokeWidth="0.5" />
      <line x1={center - maxRadius - 6} y1={center} x2={center + maxRadius + 6} y2={center} stroke="#9ca3af" strokeWidth="0.5" />
      {wedges}
      <text x={center} y={14} textAnchor="middle" fontSize="10" fill="#374151">N</text>
      <text x={size - 6} y={center + 4} textAnchor="end" fontSize="10" fill="#374151">E</text>
    </svg>
  );
}

export default function LineamentPanel({ ready, params, setParams, onRun, running, result, error, showOnMap, setShowOnMap }) {
  const update = (key, val) => setParams((p) => ({ ...p, [key]: val }));

  return (
    <div style={sectionStyle}>
      <div style={bodyStyle}>
        <div style={{ color: "#6b7280" }}>
          THD/analytic signal/틸트각/1VD 격자에서 능선(ridge) 극댓값을 찾아 서로 가까운 점들을 하나의 리니어먼트(단층·전단대·접촉면
          추정 구조선)로 묶고, 주향(strike) 방향 통계를 로즈다이어그램으로 보여줍니다. 광체 자체보다 구조통제형 광상의 구조 파악에
          유용합니다.
        </div>
        <Field label="추출 기준 격자">
          <select style={inputStyle} value={params.source} onChange={(e) => update("source", e.target.value)}>
            {Object.entries(SOURCE_LABELS).map(([k, label]) => (
              <option key={k} value={k}>{label}</option>
            ))}
          </select>
        </Field>
        <Field label="셀 크기 (m)">
          <input type="number" style={inputStyle} value={params.cell_size_m} onChange={(e) => update("cell_size_m", parseFloat(e.target.value))} />
        </Field>
        <Field label="능선 판정 백분위수 (%) — 낮을수록 더 많은 리니어먼트 포함">
          <input
            type="number"
            style={inputStyle}
            value={params.percentile_threshold}
            onChange={(e) => update("percentile_threshold", parseFloat(e.target.value))}
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
        <Field label="최소 길이 (m)">
          <input type="number" style={inputStyle} value={params.min_length_m} onChange={(e) => update("min_length_m", parseFloat(e.target.value))} />
        </Field>
        <button style={buttonStyle} disabled={!ready || running} onClick={onRun}>
          {running ? "추출 중..." : "리니어먼트 추출 실행"}
        </button>
        {error && <div style={{ color: "#dc2626" }}>{error}</div>}
        {result && !result.available && <div style={{ color: "#9ca3af" }}>{result.reason}</div>}
        {result && result.available && (
          <div style={{ color: "#374151" }}>
            리니어먼트 {result.n_lineaments}개, 총 길이 {(result.total_length_m / 1000).toFixed(2)} km
            <label style={{ display: "flex", alignItems: "center", gap: 6, margin: "6px 0" }}>
              <input type="checkbox" checked={showOnMap} onChange={(e) => setShowOnMap(e.target.checked)} />
              <span>지도에 결과 표시 (색상=주향)</span>
            </label>
            <div style={{ display: "flex", justifyContent: "center", margin: "6px 0" }}>
              <RoseDiagram bins={result.rose_bins} binWidthDeg={result.rose_bin_width_deg} />
            </div>
            <div style={{ maxHeight: 160, overflowY: "auto", border: "1px solid #f3f4f6", borderRadius: 4 }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>구조계</th>
                    <th style={thStyle}>개수</th>
                    <th style={thStyle}>총 길이(m)</th>
                    <th style={thStyle}>비율(%)</th>
                  </tr>
                </thead>
                <tbody>
                  {result.quadrant_summary.map((q) => (
                    <tr key={q.name}>
                      <td style={tdStyle}>{q.name}</td>
                      <td style={tdStyle}>{q.n_lineaments}</td>
                      <td style={tdStyle}>{q.total_length_m.toFixed(0)}</td>
                      <td style={tdStyle}>{q.pct_of_total_length.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
