import { getColorFn, COLORMAP_OPTIONS } from "../colormap";

const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #d1d5db" };
const jumpButtonStyle = {
  marginLeft: 6,
  padding: "1px 6px",
  fontSize: 10,
  borderRadius: 4,
  border: "1px solid #93c5fd",
  background: "#eff6ff",
  color: "#1d4ed8",
  cursor: "pointer",
};

function Gradient({ cmapName }) {
  const fn = getColorFn(cmapName);
  const stops = Array.from({ length: 20 }, (_, i) => fn(i / 19));
  return (
    <div
      style={{
        height: 14,
        borderRadius: 4,
        background: `linear-gradient(to right, ${stops.join(",")})`,
        border: "1px solid #d1d5db",
      }}
    />
  );
}

export default function Legend({
  label,
  unit,
  vmin,
  vmax,
  cmapName,
  onCmapChange,
  manualRange,
  onManualRangeChange,
  stats,
  hoverPoint,
  extrema,
  onJumpToExtremum,
}) {
  const updateManual = (key, value) => onManualRangeChange({ ...manualRange, [key]: value });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
          {label} {unit ? `(${unit})` : ""}
        </div>
        <Gradient cmapName={cmapName} />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#6b7280", marginTop: 2 }}>
          <span>{vmin != null ? vmin.toFixed(1) : "-"}</span>
          <span>{vmax != null ? vmax.toFixed(1) : "-"}</span>
        </div>
      </div>

      <div style={{ fontSize: 12 }}>
        <label style={{ display: "flex", flexDirection: "column", gap: 2, marginBottom: 8 }}>
          <span style={{ color: "#4b5563" }}>컬러맵</span>
          <select style={inputStyle} value={cmapName} onChange={(e) => onCmapChange(e.target.value)}>
            {COLORMAP_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
          <input type="checkbox" checked={manualRange.enabled} onChange={(e) => updateManual("enabled", e.target.checked)} />
          <span style={{ color: "#4b5563" }}>표시 범위 수동 지정</span>
        </label>
        {manualRange.enabled && (
          <div style={{ display: "flex", gap: 6 }}>
            <input
              type="number"
              style={inputStyle}
              placeholder="최소"
              value={manualRange.vmin ?? ""}
              onChange={(e) => updateManual("vmin", e.target.value === "" ? null : parseFloat(e.target.value))}
            />
            <input
              type="number"
              style={inputStyle}
              placeholder="최대"
              value={manualRange.vmax ?? ""}
              onChange={(e) => updateManual("vmax", e.target.value === "" ? null : parseFloat(e.target.value))}
            />
          </div>
        )}
      </div>

      {stats && (
        <div style={{ fontSize: 12, color: "#374151" }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>통계</div>
          <table style={{ width: "100%" }}>
            <tbody>
              <tr>
                <td>Min</td>
                <td style={{ textAlign: "right" }}>
                  {fmt(stats.min)}
                  {extrema?.min && onJumpToExtremum && (
                    <button
                      onClick={() => onJumpToExtremum("min")}
                      title="지도에서 최솟값 위치로 이동"
                      style={jumpButtonStyle}
                    >
                      📍이동
                    </button>
                  )}
                </td>
              </tr>
              <tr>
                <td>Max</td>
                <td style={{ textAlign: "right" }}>
                  {fmt(stats.max)}
                  {extrema?.max && onJumpToExtremum && (
                    <button
                      onClick={() => onJumpToExtremum("max")}
                      title="지도에서 최댓값 위치로 이동"
                      style={jumpButtonStyle}
                    >
                      📍이동
                    </button>
                  )}
                </td>
              </tr>
              <tr>
                <td>Mean</td>
                <td style={{ textAlign: "right" }}>{fmt(stats.mean)}</td>
              </tr>
              <tr>
                <td>Std</td>
                <td style={{ textAlign: "right" }}>{fmt(stats.std)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {hoverPoint && (
        <div style={{ fontSize: 12, color: "#374151", borderTop: "1px solid #e5e7eb", paddingTop: 10 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>선택 포인트 (TMI)</div>
          <div>위도: {hoverPoint.lat.toFixed(6)}</div>
          <div>경도: {hoverPoint.lon.toFixed(6)}</div>
          <div>
            값: {hoverPoint.value.toFixed(2)} {unit}
          </div>
          <div>측선 ID: {hoverPoint.line_id >= 0 ? hoverPoint.line_id : "제외됨"}</div>
          <div>시간: {hoverPoint.timestamp}</div>
        </div>
      )}
    </div>
  );
}

function fmt(v) {
  return v == null ? "-" : v.toFixed(2);
}
