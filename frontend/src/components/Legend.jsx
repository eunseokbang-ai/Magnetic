import { rdylbuReversed, viridis } from "../colormap";

function Gradient({ cmapKind }) {
  const fn = cmapKind === "anomaly" ? rdylbuReversed : viridis;
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

export default function Legend({ label, unit, vmin, vmax, cmapKind, stats, hoverPoint }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
          {label} {unit ? `(${unit})` : ""}
        </div>
        <Gradient cmapKind={cmapKind} />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#6b7280", marginTop: 2 }}>
          <span>{vmin != null ? vmin.toFixed(1) : "-"}</span>
          <span>{vmax != null ? vmax.toFixed(1) : "-"}</span>
        </div>
      </div>

      {stats && (
        <div style={{ fontSize: 12, color: "#374151" }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>통계</div>
          <table style={{ width: "100%" }}>
            <tbody>
              <tr>
                <td>Min</td>
                <td style={{ textAlign: "right" }}>{fmt(stats.min)}</td>
              </tr>
              <tr>
                <td>Max</td>
                <td style={{ textAlign: "right" }}>{fmt(stats.max)}</td>
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
