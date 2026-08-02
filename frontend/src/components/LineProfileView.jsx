import { useState, useRef } from "react";

// Plain-SVG distance-vs-value chart for one flight line's QC profile - no
// charting library needed for a single polyline + a handful of markers.
// Mirrors InversionSectionView's floating-panel layout.
//
// Also draws the raw_value trace (before despike/notch/lowpass filtering)
// alongside the processed value, so the user can visually confirm what the
// pipeline removed. Dragging across the chart selects a distance range;
// "선택 구간 스무딩 적용" sends those point_ids to the manual-smoothing
// endpoint, which linearly interpolates across a ground-structure
// distortion (house, building, etc.) between the good values on either side.
export default function LineProfileView({ data, valueLabel, onClose, onApplySmoothing, smoothing }) {
  const [showRaw, setShowRaw] = useState(true);
  const [dragStart, setDragStart] = useState(null);
  const [dragEnd, setDragEnd] = useState(null);
  const svgRef = useRef(null);

  if (!data) return null;
  const { distance_m, value, raw_value, excluded, smoothed, point_id, line_id } = data;
  const hasRaw = Array.isArray(raw_value);
  const hasSmoothed = Array.isArray(smoothed);

  const width = 900;
  const height = 340;
  const padding = { left: 60, right: 20, top: 16, bottom: 36 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const xMax = Math.max(...distance_m, 1);
  const allSeries = hasRaw && showRaw ? [...value, ...raw_value] : value;
  const finiteValues = allSeries.filter((v) => v != null && Number.isFinite(v));
  const yMinRaw = finiteValues.length ? Math.min(...finiteValues) : 0;
  const yMaxRaw = finiteValues.length ? Math.max(...finiteValues) : 1;
  const yPad = (yMaxRaw - yMinRaw) * 0.08 || 1;
  const y0 = yMinRaw - yPad;
  const y1 = yMaxRaw + yPad;

  const xScale = (d) => padding.left + (d / xMax) * plotW;
  const yScale = (v) => padding.top + (1 - (v - y0) / (y1 - y0)) * plotH;
  const xInvert = (px) => Math.max(0, Math.min(xMax, ((px - padding.left) / plotW) * xMax));

  const pathD = distance_m.map((d, i) => `${i === 0 ? "M" : "L"} ${xScale(d).toFixed(1)} ${yScale(value[i]).toFixed(1)}`).join(" ");
  const rawPathD = hasRaw
    ? distance_m.map((d, i) => `${i === 0 ? "M" : "L"} ${xScale(d).toFixed(1)} ${yScale(raw_value[i]).toFixed(1)}`).join(" ")
    : "";
  const nExcluded = excluded.filter(Boolean).length;

  const selRange = dragStart != null && dragEnd != null ? [Math.min(dragStart, dragEnd), Math.max(dragStart, dragEnd)] : null;
  const selectedPointIds =
    selRange && point_id
      ? point_id.filter((_, i) => distance_m[i] >= selRange[0] && distance_m[i] <= selRange[1])
      : [];

  const svgPointFromEvent = (e) => {
    const rect = svgRef.current.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    return xInvert(px);
  };

  const handleMouseDown = (e) => {
    const d = svgPointFromEvent(e);
    setDragStart(d);
    setDragEnd(d);
  };
  const handleMouseMove = (e) => {
    if (dragStart == null) return;
    setDragEnd(svgPointFromEvent(e));
  };
  const handleMouseUp = () => {
    if (dragStart != null && dragEnd != null && Math.abs(dragEnd - dragStart) < xMax * 0.003) {
      // treat as a stray click, not a real selection
      setDragStart(null);
      setDragEnd(null);
    }
  };
  const clearSelection = () => {
    setDragStart(null);
    setDragEnd(null);
  };

  const handleApply = () => {
    if (!selectedPointIds.length || !onApplySmoothing) return;
    onApplySmoothing(selectedPointIds);
    clearSelection();
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
        <div style={{ fontSize: 13, fontWeight: 600 }}>측선 프로파일 (측선 #{line_id})</div>
        <button onClick={onClose} style={{ border: "none", background: "none", cursor: "pointer", fontSize: 14 }}>
          ✕ 닫기
        </button>
      </div>
      <div style={{ flex: 1, padding: 16, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", overflow: "auto" }}>
        {hasRaw && (
          <div style={{ width: "100%", maxWidth: width, display: "flex", alignItems: "center", gap: 16, marginBottom: 6, fontSize: 12 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input type="checkbox" checked={showRaw} onChange={(e) => setShowRaw(e.target.checked)} />
              <span style={{ color: "#f59e0b" }}>━━ 원본(필터 전) 신호 표시</span>
            </label>
            <span style={{ color: "#2563eb" }}>━━ 처리 후 신호</span>
            <span style={{ color: "#6b7280", marginLeft: "auto" }}>
              차트를 드래그해 구간을 선택하면 왜곡(집/구조물 등) 구간을 주변 값으로 스무딩할 수 있습니다.
            </span>
          </div>
        )}
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          style={{ width: "100%", height: "auto", maxHeight: "65vh", background: "white", cursor: onApplySmoothing ? "crosshair" : "default", userSelect: "none" }}
          onMouseDown={onApplySmoothing ? handleMouseDown : undefined}
          onMouseMove={onApplySmoothing ? handleMouseMove : undefined}
          onMouseUp={onApplySmoothing ? handleMouseUp : undefined}
          onMouseLeave={onApplySmoothing ? handleMouseUp : undefined}
        >
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
          {hasRaw && showRaw && (
            <path d={rawPathD} fill="none" stroke="#f59e0b" strokeWidth="1" strokeDasharray="3,2" opacity={0.85} />
          )}
          <path d={pathD} fill="none" stroke="#2563eb" strokeWidth="1.5" />
          {distance_m.map((d, i) =>
            excluded[i] ? <circle key={`ex-${i}`} cx={xScale(d)} cy={yScale(value[i])} r={2.5} fill="#dc2626" /> : null
          )}
          {hasSmoothed &&
            distance_m.map((d, i) =>
              smoothed[i] ? <circle key={`sm-${i}`} cx={xScale(d)} cy={yScale(value[i])} r={2.5} fill="#16a34a" /> : null
            )}
          {selRange && (
            <rect
              x={xScale(selRange[0])}
              y={padding.top}
              width={Math.max(1, xScale(selRange[1]) - xScale(selRange[0]))}
              height={plotH}
              fill="#2563eb"
              opacity={0.12}
            />
          )}
        </svg>
        <div style={{ fontSize: 12, color: "#6b7280", marginTop: 8, display: "flex", gap: 20, flexWrap: "wrap" }}>
          <span>거리: 0 ~ {xMax.toFixed(0)} m</span>
          <span>
            {valueLabel}: {yMinRaw.toFixed(1)} ~ {yMaxRaw.toFixed(1)}
          </span>
          <span style={{ color: nExcluded > 0 ? "#dc2626" : "#6b7280" }}>● 빨간 점 = 제외된 포인트 ({nExcluded}개)</span>
          {hasSmoothed && <span style={{ color: "#16a34a" }}>● 초록 점 = 스무딩 적용된 포인트 ({smoothed.filter(Boolean).length}개)</span>}
        </div>
        {onApplySmoothing && (
          <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 10 }}>
            <button
              onClick={handleApply}
              disabled={!selectedPointIds.length || smoothing}
              style={{
                padding: "6px 12px",
                fontSize: 12,
                borderRadius: 6,
                border: "1px solid #16a34a",
                background: selectedPointIds.length ? "#f0fdf4" : "white",
                color: "#16a34a",
                cursor: selectedPointIds.length ? "pointer" : "default",
                opacity: selectedPointIds.length && !smoothing ? 1 : 0.5,
              }}
            >
              {smoothing ? "적용 중..." : `선택 구간(${selectedPointIds.length}개) 스무딩 적용`}
            </button>
            {selRange && (
              <button onClick={clearSelection} style={{ padding: "6px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #d1d5db", background: "white", cursor: "pointer" }}>
                선택 취소
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
