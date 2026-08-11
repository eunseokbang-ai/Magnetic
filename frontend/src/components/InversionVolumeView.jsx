import { useState } from "react";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

// Full plotly.js is ~6Mb; plotly.js-dist-min + the factory API keeps the
// bundle to just what the isosurface/surface traces need, self-contained
// (no CDN).
// One interior section plane (동서/남북/자유선), rendered as its own
// Plotly surface trace at its true 3D position - added on top of the
// isosurface "blob" trace so the combined scene shows the thresholded
// anomaly volume and one or more full-context cross-sections at once,
// the way mining-industry 3D modeling packages (GOCAD/Leapfrog-style)
// present a volume together with several simultaneous section planes.
function sliceToTrace(face, name, showscale) {
  if (!face) return null;
  return {
    type: "surface",
    name,
    x: face.x,
    y: face.y,
    z: face.z,
    surfacecolor: face.value,
    cmin: face.vmin ?? 0,
    cmax: face.vmax ?? 1,
    colorscale: "Turbo",
    showscale,
    colorbar: showscale ? { title: { text: "자화율 (SI)" }, x: 1.02, len: 0.4, y: 0.5 } : undefined,
    connectgaps: false,
    opacity: 0.97,
    lighting: { ambient: 0.85, diffuse: 0.5, specular: 0.1 },
    hovertemplate: `${name}<br>동서: %{x:.0f} m<br>남북: %{y:.0f} m<br>고도: %{z:.0f} m<br>SI: %{surfacecolor:.4f}<extra></extra>`,
  };
}

const SLICE_ORIENTATION_LABELS = { ew: "동서", ns: "남북", azimuth: "기울기", custom: "자유선" };

// One row in the active-slice list: a small inline control (position or
// azimuth) plus a remove button, shown for each plane currently added
// to the combined 3D scene.
function SliceRow({ slice, onUpdate, onRemove }) {
  const label = SLICE_ORIENTATION_LABELS[slice.orientation] || slice.orientation;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, background: "white", border: "1px solid #e6dac0", borderRadius: 5, padding: "2px 6px" }}>
      <span>{label} 단면</span>
      {(slice.orientation === "ew" || slice.orientation === "ns") && (
        <input
          type="range"
          min="0"
          max="1"
          step="0.02"
          value={slice.positionFrac}
          onChange={(e) => onUpdate(slice.id, { positionFrac: parseFloat(e.target.value) })}
          style={{ width: 70 }}
        />
      )}
      {slice.orientation === "azimuth" && (
        <input
          type="range"
          min="0"
          max="179"
          step="1"
          value={slice.azimuthDeg}
          title={`${slice.azimuthDeg}°`}
          onChange={(e) => onUpdate(slice.id, { azimuthDeg: parseFloat(e.target.value) })}
          style={{ width: 70 }}
        />
      )}
      {slice.loading && <span style={{ color: "#ab9a78" }}>...</span>}
      <button
        onClick={() => onRemove(slice.id)}
        style={{ border: "none", background: "none", cursor: "pointer", color: "#ab9a78", fontSize: 13, padding: 0, lineHeight: 1 }}
        title="이 단면 제거"
      >
        ✕
      </button>
    </div>
  );
}

export default function InversionVolumeView({
  data,
  onClose,
  boxData,
  boxTopLayerIndex,
  onBoxTopLayerIndexChange,
  boxLoading,
  nLayers,
  slices,
  onAddSlice,
  onRemoveSlice,
  onUpdateSlice,
  customSliceAvailable,
  opacity,
  onOpacityChange,
}) {
  const [showTop, setShowTop] = useState(true);
  const [mode, setMode] = useState("blob");
  if (!data) return null;

  if (mode === "box") {
    return (
      <BoxFacesView
        data={boxData}
        loading={boxLoading}
        topLayerIndex={boxTopLayerIndex}
        onTopLayerIndexChange={onBoxTopLayerIndexChange}
        nLayers={nLayers}
        onClose={onClose}
        onSwitchToBlob={() => setMode("blob")}
      />
    );
  }

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
      opacity: opacity ?? 1,
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

  // Any number of interior 동서/남북/기울기/자유선 section planes shown
  // together with the isosurface volume in this same scene (see
  // App.jsx's volumeSlices state) - one shared colorbar since they all
  // use the same continuous SI scale (0 to the model's own max, no
  // threshold gating), matching the multi-panel "volume + several
  // simultaneous section planes" presentation this view is modeled on.
  const activeSlices = (slices || []).filter((s) => s.data);
  activeSlices.forEach((slice, i) => {
    const name = `${SLICE_ORIENTATION_LABELS[slice.orientation] || slice.orientation} 단면`;
    const trace = sliceToTrace(slice.data, name, i === 0);
    if (trace) traces.push(trace);
  });

  return (
    <div
      style={{
        position: "absolute",
        top: 16,
        right: 16,
        bottom: 16,
        left: 16,
        background: "white",
        border: "1px solid #ddd0b2",
        borderRadius: 8,
        boxShadow: "0 4px 20px rgba(0,0,0,0.25)",
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 12px", borderBottom: "1px solid #e6dac0" }}>
        <div style={{ fontSize: 13, fontWeight: 600 }}>3차원 자화율 이상대 (SI, 지정한 범위만 표시)</div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {onOpacityChange && (
            <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 4 }} title="이상대를 반투명하게 하면 내부 단면이 함께 보입니다">
              투명도
              <input
                type="range"
                min="0.15"
                max="1"
                step="0.05"
                value={opacity ?? 1}
                onChange={(e) => onOpacityChange(parseFloat(e.target.value))}
                style={{ width: 70 }}
              />
            </label>
          )}
          {top && (
            <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
              <input type="checkbox" checked={showTop} onChange={(e) => setShowTop(e.target.checked)} />
              상단에 관측 자력이상 지도 표시
            </label>
          )}
          <button
            onClick={() => {
              setMode("box");
              if (!boxData) onBoxTopLayerIndexChange?.(boxTopLayerIndex ?? 0);
            }}
            style={{ border: "1px solid #a9631f", background: "white", color: "#a9631f", borderRadius: 6, padding: "3px 8px", fontSize: 12, cursor: "pointer" }}
          >
            SI 연속 컬러링 박스 단면으로 보기
          </button>
          <button onClick={onClose} style={{ border: "none", background: "none", cursor: "pointer", fontSize: 14 }}>
            ✕ 닫기
          </button>
        </div>
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "6px 12px",
          borderBottom: "1px solid #e6dac0",
          background: "#faf6ec",
          flexWrap: "wrap",
        }}
      >
        <span style={{ fontSize: 11, color: "#8a7a5c" }}>단면 추가:</span>
        <button onClick={() => onAddSlice?.("ew")} style={{ border: "1px solid #ddd0b2", background: "white", borderRadius: 5, padding: "2px 8px", fontSize: 12, cursor: "pointer" }}>
          + 동서
        </button>
        <button onClick={() => onAddSlice?.("ns")} style={{ border: "1px solid #ddd0b2", background: "white", borderRadius: 5, padding: "2px 8px", fontSize: 12, cursor: "pointer" }}>
          + 남북
        </button>
        <button onClick={() => onAddSlice?.("azimuth")} style={{ border: "1px solid #ddd0b2", background: "white", borderRadius: 5, padding: "2px 8px", fontSize: 12, cursor: "pointer" }}>
          + 기울기(자유방향)
        </button>
        <button
          onClick={() => onAddSlice?.("custom")}
          disabled={!customSliceAvailable}
          title={customSliceAvailable ? undefined : "먼저 '수직 섹션 뷰'에서 자유선을 그려주세요"}
          style={{
            border: "1px solid #ddd0b2",
            background: "white",
            borderRadius: 5,
            padding: "2px 8px",
            fontSize: 12,
            cursor: customSliceAvailable ? "pointer" : "not-allowed",
            color: customSliceAvailable ? "inherit" : "#ab9a78",
          }}
        >
          + 자유선(마지막으로 그린 선)
        </button>
        {(slices || []).length > 0 && <span style={{ width: 1, alignSelf: "stretch", background: "#e6dac0" }} />}
        {(slices || []).map((slice) => (
          <SliceRow key={slice.id} slice={slice} onUpdate={onUpdateSlice} onRemove={onRemoveSlice} />
        ))}
      </div>
      <div style={{ flex: 1 }}>
        <Plot
          data={traces}
          layout={{
            autosize: true,
            margin: { l: 0, r: 80, t: 0, b: 0 },
            scene: {
              xaxis: { title: { text: "동서 (m)" }, showbackground: true, backgroundcolor: "#f3ecd9", gridcolor: "#ddd0b2" },
              yaxis: { title: { text: "남북 (m)" }, showbackground: true, backgroundcolor: "#f3ecd9", gridcolor: "#ddd0b2" },
              zaxis: { title: { text: "고도 (m)" }, showbackground: true, backgroundcolor: "#e6dac0", gridcolor: "#ddd0b2" },
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

// "Fence diagram" style box: top horizontal slice + the 4 vertical
// boundary walls of the mesh, all sharing one continuous SI colorscale
// (no threshold gating) - mirrors the reference-style figure of a
// colored box with a distinct top surface and colored side faces.
function BoxFacesView({ data, loading, topLayerIndex, onTopLayerIndexChange, nLayers, onClose, onSwitchToBlob }) {
  const faceTraces = [];
  if (data) {
    const vmin = data.vmin ?? 0;
    const vmax = data.vmax ?? 1;
    const faceDefs = [
      { key: "top", name: "상단면" },
      { key: "south", name: "남쪽 벽면" },
      { key: "north", name: "북쪽 벽면" },
      { key: "west", name: "서쪽 벽면" },
      { key: "east", name: "동쪽 벽면" },
    ];
    faceDefs.forEach(({ key, name }, i) => {
      const face = data[key];
      if (!face) return;
      faceTraces.push({
        type: "surface",
        name,
        x: face.x,
        y: face.y,
        z: face.z,
        surfacecolor: face.value,
        cmin: vmin,
        cmax: vmax,
        colorscale: "Turbo",
        showscale: i === 0,
        colorbar: i === 0 ? { title: { text: "자화율 (SI)" }, x: 1.02 } : undefined,
        connectgaps: false,
        lighting: { ambient: 0.85, diffuse: 0.5, specular: 0.1 },
        hovertemplate: "동서: %{x:.0f} m<br>남북: %{y:.0f} m<br>고도: %{z:.0f} m<br>SI: %{surfacecolor:.4f}<extra></extra>",
      });
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
        border: "1px solid #ddd0b2",
        borderRadius: 8,
        boxShadow: "0 4px 20px rgba(0,0,0,0.25)",
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 12px", borderBottom: "1px solid #e6dac0" }}>
        <div style={{ fontSize: 13, fontWeight: 600 }}>SI 연속 컬러링 박스 단면 (상단면 + 4개 측벽, 임계값 없이 전체 표시)</div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}>
            상단면 깊이 레이어: {topLayerIndex ?? 0} / {Math.max(0, (nLayers ?? 1) - 1)}
            {data?.top_elevation_m != null && <span style={{ color: "#8a7a5c" }}>(고도 약 {data.top_elevation_m.toFixed(0)} m)</span>}
            <input
              type="range"
              min="0"
              max={Math.max(0, (nLayers ?? 1) - 1)}
              step="1"
              value={topLayerIndex ?? 0}
              onChange={(e) => onTopLayerIndexChange?.(parseInt(e.target.value, 10))}
              style={{ width: 120 }}
            />
          </label>
          <button onClick={onSwitchToBlob} style={{ border: "1px solid #a9631f", background: "white", color: "#a9631f", borderRadius: 6, padding: "3px 8px", fontSize: 12, cursor: "pointer" }}>
            이상대 블롭 뷰로 돌아가기
          </button>
          <button onClick={onClose} style={{ border: "none", background: "none", cursor: "pointer", fontSize: 14 }}>
            ✕ 닫기
          </button>
        </div>
      </div>
      <div style={{ flex: 1, position: "relative" }}>
        {loading && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "#8a7a5c", fontSize: 13, zIndex: 1 }}>
            불러오는 중...
          </div>
        )}
        {!loading && faceTraces.length > 0 && (
          <Plot
            data={faceTraces}
            layout={{
              autosize: true,
              margin: { l: 0, r: 80, t: 0, b: 0 },
              scene: {
                xaxis: { title: { text: "동서 (m)" }, showbackground: true, backgroundcolor: "#f3ecd9", gridcolor: "#ddd0b2" },
                yaxis: { title: { text: "남북 (m)" }, showbackground: true, backgroundcolor: "#f3ecd9", gridcolor: "#ddd0b2" },
                zaxis: { title: { text: "고도 (m)" }, showbackground: true, backgroundcolor: "#e6dac0", gridcolor: "#ddd0b2" },
                aspectmode: "data",
                camera: { eye: { x: 1.5, y: -1.7, z: 1.0 } },
              },
            }}
            useResizeHandler
            style={{ width: "100%", height: "100%" }}
            config={{ displaylogo: false }}
          />
        )}
      </div>
    </div>
  );
}
