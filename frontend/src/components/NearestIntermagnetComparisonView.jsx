// Plain-SVG time-vs-value chart comparing the combined (IDW-estimated)
// base station series against each real nearby INTERMAGNET observatory's
// own raw data it was built from - lets the user visually sanity-check
// the estimate against real station behavior. Each series gets its own
// panel with its own y-axis scale, because each station's absolute field
// level differs by location enough that a shared y-axis would flatten the
// (much smaller) diurnal variation we actually want to compare.
const STATION_COLORS = ["#f59e0b", "#10b981", "#8b5cf6", "#ef4444", "#0ea5e9", "#ec4899", "#84cc16", "#f97316"];

function Panel({ title, color, seconds, values, xMax, fmtSeconds }) {
  const width = 420;
  const height = 190;
  const padding = { left: 62, right: 14, top: 10, bottom: 30 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const finiteValues = values.filter((v) => v != null && Number.isFinite(v));
  const yMinRaw = finiteValues.length ? Math.min(...finiteValues) : 0;
  const yMaxRaw = finiteValues.length ? Math.max(...finiteValues) : 1;
  const yPad = (yMaxRaw - yMinRaw) * 0.08 || 1;
  const y0 = yMinRaw - yPad;
  const y1 = yMaxRaw + yPad;

  const xScale = (s) => padding.left + (s / xMax) * plotW;
  const yScale = (v) => padding.top + (1 - (v - y0) / (y1 - y0)) * plotH;

  const pathD = seconds.map((s, i) => `${i === 0 ? "M" : "L"} ${xScale(s).toFixed(1)} ${yScale(values[i]).toFixed(1)}`).join(" ");

  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: 6, padding: "8px 10px" }}>
      <div style={{ fontSize: 12, fontWeight: 600, color, marginBottom: 4 }}>{title}</div>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto", background: "white" }}>
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#9ca3af" strokeWidth="1" />
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#9ca3af" strokeWidth="1" />
        <text x={padding.left - 6} y={yScale(y1) + 4} fontSize="10" textAnchor="end" fill="#6b7280">
          {y1.toFixed(1)}
        </text>
        <text x={padding.left - 6} y={yScale(y0) + 4} fontSize="10" textAnchor="end" fill="#6b7280">
          {y0.toFixed(1)}
        </text>
        <text x={padding.left} y={height - padding.bottom + 16} fontSize="10" fill="#6b7280">
          0s
        </text>
        <text x={width - padding.right} y={height - padding.bottom + 16} fontSize="10" textAnchor="end" fill="#6b7280">
          {fmtSeconds(xMax)}
        </text>
        {seconds.length > 0 && <path d={pathD} fill="none" stroke={color} strokeWidth="1.5" />}
      </svg>
    </div>
  );
}

export default function NearestIntermagnetComparisonView({ data, onClose }) {
  if (!data) return null;
  const { combined, stations } = data;
  const hasCombined = Array.isArray(combined?.timestamp) && combined.timestamp.length > 0;

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
        <div style={{ fontSize: 13, fontWeight: 600 }}>주변 관측소 자료 vs 결합 추정 베이스 자료 비교 (창별 개별 스케일)</div>
        <button onClick={onClose} style={{ border: "none", background: "none", cursor: "pointer", fontSize: 14 }}>
          ✕ 닫기
        </button>
      </div>
      <div style={{ flex: 1, padding: 16, overflow: "auto" }}>
        <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 10 }}>
          각 그래프는 자체 nT 범위로 스케일되어 있어 일변화 형태를 서로 비교하기 쉽습니다 (절대값 수준은 그래프마다 다름). 굵은 검은 테두리 그래프가 실제 사용/적용 예정인 결합 추정 자료입니다.
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))",
            gap: 12,
          }}
        >
          {stations.map((s, i) => (
            <Panel
              key={s.iaga_code}
              title={`${s.station_name} (${s.iaga_code}) — ${s.distance_km?.toFixed(0)}km${
                s.estimated_dates?.length > 0 ? ` (추정: ${s.estimated_dates.join(", ")})` : ""
              }`}
              color={STATION_COLORS[i % STATION_COLORS.length]}
              seconds={stationSeconds[i]}
              values={s.mag}
              xMax={xMax}
              fmtSeconds={fmtSeconds}
            />
          ))}
          {hasCombined && (
            <Panel
              title="결합 추정(사용할 베이스 자료)"
              color="#1f2937"
              seconds={combinedSeconds}
              values={combined.mag}
              xMax={xMax}
              fmtSeconds={fmtSeconds}
            />
          )}
        </div>
      </div>
    </div>
  );
}
