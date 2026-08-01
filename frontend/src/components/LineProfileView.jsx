// Plain-SVG distance-vs-value chart for one flight line's QC profile - no
// charting library needed for a single polyline + a handful of markers.
// Mirrors InversionSectionView's floating-panel layout.
export default function LineProfileView({ data, valueLabel, onClose }) {
  if (!data) return null;
  const { distance_m, value, excluded, line_id } = data;

  const width = 900;
  const height = 320;
  const padding = { left: 60, right: 20, top: 16, bottom: 36 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const xMax = Math.max(...distance_m, 1);
  const finiteValues = value.filter((v) => v != null && Number.isFinite(v));
  const yMinRaw = finiteValues.length ? Math.min(...finiteValues) : 0;
  const yMaxRaw = finiteValues.length ? Math.max(...finiteValues) : 1;
  const yPad = (yMaxRaw - yMinRaw) * 0.08 || 1;
  const y0 = yMinRaw - yPad;
  const y1 = yMaxRaw + yPad;

  const xScale = (d) => padding.left + (d / xMax) * plotW;
  const yScale = (v) => padding.top + (1 - (v - y0) / (y1 - y0)) * plotH;

  const pathD = distance_m.map((d, i) => `${i === 0 ? "M" : "L"} ${xScale(d).toFixed(1)} ${yScale(value[i]).toFixed(1)}`).join(" ");
  const nExcluded = excluded.filter(Boolean).length;

  return (
    <div
      style={{
        position: "absolute",
        top: 16,
        right: 16,
        bottom: 16,
        left: 16,
        background: "white",
        border: "1px solid #d1d5db",
        borderRadius: 8,
        boxShadow: "0 4px 20px rgba(0,0,0,0.25)",
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 12px", borderBottom: "1px solid #e5e7eb" }}>
        <div style={{ fontSize: 13, fontWeight: 600 }}>측선 프로파일 (측선 #{line_id})</div>
        <button onClick={onClose} style={{ border: "none", background: "none", cursor: "pointer", fontSize: 14 }}>
          ✕ 닫기
        </button>
      </div>
      <div style={{ flex: 1, padding: 16, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", overflow: "auto" }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto", maxHeight: "70vh", background: "white" }}>
          <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#9ca3af" strokeWidth="1" />
          <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#9ca3af" strokeWidth="1" />
          <text x={padding.left - 6} y={yScale(y1) + 4} fontSize="11" textAnchor="end" fill="#6b7280">
            {y1.toFixed(0)}
          </text>
          <text x={padding.left - 6} y={yScale(y0) + 4} fontSize="11" textAnchor="end" fill="#6b7280">
            {y0.toFixed(0)}
          </text>
          <text x={padding.left} y={height - padding.bottom + 18} fontSize="11" fill="#6b7280">
            0 m
          </text>
          <text x={width - padding.right} y={height - padding.bottom + 18} fontSize="11" textAnchor="end" fill="#6b7280">
            {xMax.toFixed(0)} m
          </text>
          <path d={pathD} fill="none" stroke="#2563eb" strokeWidth="1.5" />
          {distance_m.map((d, i) =>
            excluded[i] ? <circle key={i} cx={xScale(d)} cy={yScale(value[i])} r={2.5} fill="#dc2626" /> : null
          )}
        </svg>
        <div style={{ fontSize: 12, color: "#6b7280", marginTop: 8, display: "flex", gap: 20 }}>
          <span>거리: 0 ~ {xMax.toFixed(0)} m</span>
          <span>
            {valueLabel}: {yMinRaw.toFixed(1)} ~ {yMaxRaw.toFixed(1)}
          </span>
          <span style={{ color: nExcluded > 0 ? "#dc2626" : "#6b7280" }}>● 빨간 점 = 제외된 포인트 ({nExcluded}개)</span>
        </div>
      </div>
    </div>
  );
}
