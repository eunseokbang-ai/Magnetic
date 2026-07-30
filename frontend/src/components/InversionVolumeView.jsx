import { useState } from "react";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

// Full plotly.js is ~6Mb; plotly.js-dist-min + the factory API keeps the
// bundle to just what the isosurface/surface traces need, self-contained
// (no CDN).
export default function InversionVolumeView({ data, onClose }) {
  const [showTop, setShowTop] = useState(true);
  if (!data) return null;

  const nonZero = (data.value || []).filter((v) => v > 0).sort((a, b) => a - b);
  // Default to a high percentile (not the median) when the user hasn't
  // set an explicit lower bound: at the median, roughly half of all
  // nonzero cells qualify, which renders as one huge pale sheet rather
  // than the isolated, well-defined "anomaly blobs" this view is for.
  const defaultIsomin = nonZero.length ? nonZero[Math.floor(nonZero.length * 0.85)] : 0;
  const top = data.top;

  const traces = [
    {
      type: "isosurface",
      x: data.x,
      y: data.y,
      z: data.z,
      value: data.value,
      isomin: data.threshold ?? defaultIsomin,
      isomax: data.thresholdMax ?? data.stats?.max ?? undefined,
      // A single closed shell (not several partially-transparent ones,
      // and not the diffuse `volume` trace) is what makes the recovered
      // bodies read as solid, smoothly-bounded "blobs" the way
      // UBC/Geosoft-style inversion figures show them.
      surface: { count: 1, fill: 1 },
      opacity: 1,
      colorscale: "Turbo",
      caps: { x: { show: false }, y: { show: false }, z: { show: false } },
      colorbar: { title: { text: "SI" }, x: 1.02, len: 0.4, y: 0.78 },
      lighting: { ambient: 0.55, diffuse: 0.8, specular: 0.3, roughness: 0.6 },
    },
  ];

  if (top && showTop) {
    // top.z / top.color are already 2D (ny x nx) - the terrain-following
    // elevation and the observed anomaly/TMI value at each mesh column,
    // draped as a "lid" on the inversion box for geographic context.
    traces.push({
      type: "surface",
      x: top.x,
      y: top.y,
      z: top.z,
      surfacecolor: top.color,
      showscale: true,
      colorscale: "Jet",
      colorbar: { title: { text: top.value_field === "tmi" ? "TMI (nT)" : "이상 (nT)" }, x: 1.02, len: 0.4, y: 0.22 },
      lighting: { ambient: 0.9, diffuse: 0.3 },
      hoverinfo: "skip",
    });
  }

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
        <div style={{ fontSize: 13, fontWeight: 600 }}>3차원 자화율 이상대 (SI, 지정한 범위만 표시)</div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {top && (
            <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
              <input type="checkbox" checked={showTop} onChange={(e) => setShowTop(e.target.checked)} />
              상단에 관측 자력이상 지도 표시
            </label>
          )}
          <button onClick={onClose} style={{ border: "none", background: "none", cursor: "pointer", fontSize: 14 }}>
            ✕ 닫기
          </button>
        </div>
      </div>
      <div style={{ flex: 1 }}>
        <Plot
          data={traces}
          layout={{
            autosize: true,
            margin: { l: 0, r: 80, t: 0, b: 0 },
            scene: {
              xaxis: { title: { text: "동서 (m)" }, showbackground: true, backgroundcolor: "#f3f4f6", gridcolor: "#d1d5db" },
              yaxis: { title: { text: "남북 (m)" }, showbackground: true, backgroundcolor: "#f3f4f6", gridcolor: "#d1d5db" },
              zaxis: { title: { text: "고도 (m)" }, showbackground: true, backgroundcolor: "#e5e7eb", gridcolor: "#d1d5db" },
              aspectmode: "data",
              camera: { eye: { x: 1.4, y: -1.6, z: 1.1 } },
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
