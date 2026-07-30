import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

// Full plotly.js is ~6Mb; plotly.js-dist-min + the factory API keeps the
// bundle to just what the `volume` trace needs, self-contained (no CDN).
export default function InversionVolumeView({ data, onClose }) {
  if (!data) return null;

  const nonZero = (data.value || []).filter((v) => v > 0);
  const isomin = nonZero.length ? nonZero.sort((a, b) => a - b)[Math.floor(nonZero.length * 0.5)] : 0;

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
        <div style={{ fontSize: 13, fontWeight: 600 }}>3차원 자화율 이상대 (SI, 임계값 이상만 표시)</div>
        <button onClick={onClose} style={{ border: "none", background: "none", cursor: "pointer", fontSize: 14 }}>
          ✕ 닫기
        </button>
      </div>
      <div style={{ flex: 1 }}>
        <Plot
          data={[
            {
              type: "volume",
              x: data.x,
              y: data.y,
              z: data.z,
              value: data.value,
              isomin: data.threshold ?? isomin,
              isomax: data.stats?.max ?? undefined,
              opacity: 0.15,
              opacityscale: [
                [0, 0],
                [0.4, 0.35],
                [1, 0.9],
              ],
              surface: { count: 14 },
              colorscale: "Turbo",
              caps: { x: { show: false }, y: { show: false }, z: { show: false } },
              colorbar: { title: { text: "SI" } },
            },
          ]}
          layout={{
            autosize: true,
            margin: { l: 0, r: 0, t: 0, b: 0 },
            scene: {
              xaxis: { title: { text: "동서 (m)" } },
              yaxis: { title: { text: "남북 (m)" } },
              zaxis: { title: { text: "고도 (m)" } },
              aspectmode: "data",
            },
          }}
          useResizeHandler
          style={{ width: "100%", height: "100%" }}
          config={{ displaylogo: false }}
        />
      </div>
    </div>
  );
}
