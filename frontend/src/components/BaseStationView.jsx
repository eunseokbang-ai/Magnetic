// Plain-SVG time-vs-value chart for the base (diurnal) station log: raw as
// recorded vs. QC-corrected (installation/pickup transient trimmed,
// interior spikes despiked - see backend processing/base_qc.py). Lets the
// user visually confirm the trim/despike behaved sensibly before it's used
// for diurnal correction. Mirrors LineProfileView's floating-panel layout.
export default function BaseStationView({ data, onClose }) {
  if (!data) return null;
  const { raw_timestamp, raw_mag, corrected_timestamp, corrected_mag, qc_info } = data;
  const hasCorrected = Array.isArray(corrected_timestamp) && corrected_timestamp.length > 0;

  const width = 900;
  const height = 340;
  const padding = { left: 70, right: 20, top: 16, bottom: 36 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const t0 = new Date(raw_timestamp[0]).getTime();
  const rawSeconds = raw_timestamp.map((t) => (new Date(t).getTime() - t0) / 1000);
  const correctedSeconds = hasCorrected ? corrected_timestamp.map((t) => (new Date(t).getTime() - t0) / 1000) : [];

  const xMax = Math.max(...rawSeconds, 1);
  const allValues = hasCorrected ? [...raw_mag, ...corrected_mag] : raw_mag;
  const finiteValues = allValues.filter((v) => v != null && Number.isFinite(v));
  const yMinRaw = finiteValues.length ? Math.min(...finiteValues) : 0;
  const yMaxRaw = finiteValues.length ? Math.max(...finiteValues) : 1;
  const yPad = (yMaxRaw - yMinRaw) * 0.08 || 1;
  const y0 = yMinRaw - yPad;
  const y1 = yMaxRaw + yPad;

  const xScale = (s) => padding.left + (s / xMax) * plotW;
  const yScale = (v) => padding.top + (1 - (v - y0) / (y1 - y0)) * plotH;

  const rawPathD = rawSeconds.map((s, i) => `${i === 0 ? "M" : "L"} ${xScale(s).toFixed(1)} ${yScale(raw_mag[i]).toFixed(1)}`).join(" ");
  const correctedPathD = hasCorrected
    ? correctedSeconds.map((s, i) => `${i === 0 ? "M" : "L"} ${xScale(s).toFixed(1)} ${yScale(corrected_mag[i]).toFixed(1)}`).join(" ")
    : "";

  const fmtSeconds = (s) => {
    if (s < 120) return `${s.toFixed(0)}s`;
    return `${(s / 60).toFixed(1)}min`;
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
        <div style={{ fontSize: 13, fontWeight: 600 }}>베이스(일변화) 자료 - 원본 vs 보정</div>
        <button onClick={onClose} style={{ border: "none", background: "none", cursor: "pointer", fontSize: 14 }}>
          ✕ 닫기
        </button>
      </div>
      <div style={{ flex: 1, padding: 16, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", overflow: "auto" }}>
        <div style={{ width: "100%", maxWidth: width, display: "flex", alignItems: "center", gap: 16, marginBottom: 6, fontSize: 12 }}>
          <span style={{ color: "#f59e0b" }}>━━ 원본 (설치/회수 노이즈 포함)</span>
          {hasCorrected && <span style={{ color: "#2563eb" }}>━━ 보정 후 (트림 + 스파이크 제거)</span>}
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
          <path d={rawPathD} fill="none" stroke="#f59e0b" strokeWidth="1" opacity={0.85} />
          {hasCorrected && <path d={correctedPathD} fill="none" stroke="#2563eb" strokeWidth="1.5" />}
        </svg>
        <div style={{ fontSize: 12, color: "#6b7280", marginTop: 8, display: "flex", gap: 20, flexWrap: "wrap" }}>
          <span>
            원본 nT: {yMinRaw.toFixed(1)} ~ {yMaxRaw.toFixed(1)} ({raw_mag.length}포인트)
          </span>
          {hasCorrected && <span>보정 후: {corrected_mag.length}포인트</span>}
        </div>
        {qc_info && (
          <div style={{ marginTop: 10, fontSize: 12, color: "#374151", background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 6, padding: "8px 12px" }}>
            설치구간 트림: <b>{qc_info.n_trimmed_start}</b>건 · 회수구간 트림: <b>{qc_info.n_trimmed_end}</b>건 · 중간 스파이크 제거: <b>{qc_info.n_spikes_removed}</b>건
          </div>
        )}
        {!hasCorrected && (
          <div style={{ marginTop: 10, fontSize: 12, color: "#b45309" }}>
            ⚠ 아직 자료 처리를 실행하지 않아 보정된 자료가 없습니다. "자료 처리 실행" 후 다시 확인하세요.
          </div>
        )}
      </div>
    </div>
  );
}
