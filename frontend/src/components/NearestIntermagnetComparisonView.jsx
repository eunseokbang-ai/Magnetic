// Plain-SVG time-vs-value chart comparing the combined (IDW-estimated)
// base station series against each real nearby INTERMAGNET observatory's
// own raw data it was built from - lets the user visually sanity-check
// the estimate against real station behavior. Mirrors BaseStationView's
// floating-panel layout and plain-SVG approach (no chart library needed
// for a handful of lines).
const STATION_COLORS = ["#f59e0b", "#10b981", "#8b5cf6", "#ef4444", "#0ea5e9", "#ec4899", "#84cc16", "#f97316"];

export default function NearestIntermagnetComparisonView({ data, onClose }) {
  if (!data) return null;
  const { combined, stations } = data;
  const hasCombined = Array.isArray(combined?.timestamp) && combined.timestamp.length > 0;

  const width = 900;
  const height = 380;
  const padding = { left: 70, right: 20, top: 16, bottom: 36 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const allTimestamps = [
    ...(hasCombined ? combined.timestamp : []),
    ...stations.flatMap((s) => s.timestamp),
  ];
  if (allTimestamps.length === 0) return null;
  const t0 = Math.min(...allTimestamps.map((t) => new Date(t).getTime()));

  const toSeconds = (timestamps) => timestamps.map((t) => (new Date(t).getTime() - t0) / 1000);
  const combinedSeconds = hasCombined ? toSeconds(combined.timestamp) : [];
  const stationSeconds = stations.map((s) => toSeconds(s.timestamp));

  const xMax = Math.max(...combinedSeconds, ...stationSeconds.flat(), 1);
  const allValues = [...(hasCombined ? combined.mag : []), ...stations.flatMap((s) => s.mag)].filter(
    (v) => v != null && Number.isFinite(v)
  );
  const yMinRaw = allValues.length ? Math.min(...allValues) : 0;
  const yMaxRaw = allValues.length ? Math.max(...allValues) : 1;
  const yPad = (yMaxRaw - yMinRaw) * 0.08 || 1;
  const y0 = yMinRaw - yPad;
  const y1 = yMaxRaw + yPad;

  const xScale = (s) => padding.left + (s / xMax) * plotW;
  const yScale = (v) => padding.top + (1 - (v - y0) / (y1 - y0)) * plotH;

  const pathFor = (seconds, values) =>
    seconds.map((s, i) => `${i === 0 ? "M" : "L"} ${xScale(s).toFixed(1)} ${yScale(values[i]).toFixed(1)}`).join(" ");

  const combinedPathD = hasCombined ? pathFor(combinedSeconds, combined.mag) : "";
  const stationPaths = stations.map((s, i) => pathFor(stationSeconds[i], s.mag));

  const fmtSeconds = (s) => {
    if (s < 120) return `${s.toFixed(0)}s`;
    if (s < 7200) return `${(s / 60).toFixed(1)}min`;
    return `${(s / 3600).toFixed(1)}h`;
  };

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
        <div style={{ fontSize: 13, fontWeight: 600 }}>주변 관측소 자료 vs 결합 추정 베이스 자료 비교</div>
        <button onClick={onClose} style={{ border: "none", background: "none", cursor: "pointer", fontSize: 14 }}>
          ✕ 닫기
        </button>
      </div>
      <div style={{ flex: 1, padding: 16, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", overflow: "auto" }}>
        <div style={{ width: "100%", maxWidth: width, display: "flex", alignItems: "center", gap: 16, marginBottom: 6, fontSize: 12, flexWrap: "wrap" }}>
          {hasCombined && <span style={{ color: "#1f2937", fontWeight: 600 }}>━━ 결합 추정(사용할 베이스 자료)</span>}
          {stations.map((s, i) => (
            <span key={s.iaga_code} style={{ color: STATION_COLORS[i % STATION_COLORS.length] }}>
              ━━ {s.station_name} ({s.iaga_code}) — {s.distance_km?.toFixed(0)}km
            </span>
          ))}
        </div>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto", maxHeight: "65vh", background: "white" }}>
          <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#9ca3af" strokeWidth="1" />
          <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#9ca3af" strokeWidth="1" />
          <text x={padding.left - 6} y={yScale(y1) + 4} fontSize="11" textAnchor="end" fill="#6b7280">
            {y1.toFixed(1)}
          </text>
          <text x={padding.left - 6} y={yScale(y0) + 4} fontSize="11" textAnchor="end" fill="#6b7280">
            {y0.toFixed(1)}
          </text>
          <text x={padding.left} y={height - padding.bottom + 18} fontSize="11" fill="#6b7280">
            0s
          </text>
          <text x={width - padding.right} y={height - padding.bottom + 18} fontSize="11" textAnchor="end" fill="#6b7280">
            {fmtSeconds(xMax)}
          </text>
          {stationPaths.map((d, i) => (
            <path key={stations[i].iaga_code} d={d} fill="none" stroke={STATION_COLORS[i % STATION_COLORS.length]} strokeWidth="1" opacity={0.8} />
          ))}
          {hasCombined && <path d={combinedPathD} fill="none" stroke="#1f2937" strokeWidth="2" />}
        </svg>
        <div style={{ fontSize: 12, color: "#6b7280", marginTop: 8 }}>
          nT 범위: {yMinRaw.toFixed(1)} ~ {yMaxRaw.toFixed(1)} — 굵은 검은 선이 실제 사용/적용된(또는 적용 예정인) 결합 추정 자료이고, 가는 색선들이 그 계산에 사용된 각 관측소의 실제 원본 자료입니다.
        </div>
      </div>
    </div>
  );
}
