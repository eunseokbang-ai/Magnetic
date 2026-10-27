import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, ImageOverlay, CircleMarker, Polygon, Polyline, Tooltip, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet-imageoverlay-rotated";
import * as api from "./api";

const panel = { width: 420, padding: 14, borderRight: "1px solid #d9e2d6", overflowY: "auto", background: "#f4f7f2", fontSize: 13 };
const h2 = { fontSize: 13, margin: "16px 0 6px", color: "#2f3e2a" };
const row = { display: "flex", justifyContent: "space-between", padding: "2px 0", borderBottom: "1px dotted #d9e2d6" };
const button = { padding: "6px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #4f7a3a", background: "white", color: "#2f5a22", cursor: "pointer" };
const ghost = { ...button, borderColor: "#c9d3c4", color: "#5a6a55" };
const note = { color: "#6b7a66", fontSize: 11, lineHeight: 1.45 };
const warn = { color: "#8a4b08", fontSize: 12, lineHeight: 1.45 };
const box = { marginTop: 8, padding: 8, background: "white", border: "1px solid #d9e2d6", borderRadius: 6 };
const inputW = (w) => ({ width: w });

const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(d));
const gradeColor = (g) => (g === "A" ? "#b42318" : g === "B" ? "#b26a00" : "#6b7a66");

/** Blue–white–red for the total-field anomaly, clipped at +-vmax. */
function color(v, vmax) {
  const t = Math.max(-1, Math.min(1, v / vmax));
  const r = t > 0 ? 255 : Math.round(255 * (1 + t));
  const b = t < 0 ? 255 : Math.round(255 * (1 - t));
  const g = Math.round(255 * (1 - Math.abs(t)));
  return `rgb(${r},${g},${b})`;
}

function FitBounds({ lat, lon }) {
  const map = useMap();
  useEffect(() => {
    if (lat.length) map.fitBounds([[Math.min(...lat), Math.min(...lon)], [Math.max(...lat), Math.max(...lon)]], { padding: [20, 20] });
  }, [lat, lon, map]);
  return null;
}

/** The grid PNG placed by its three corners (UTM grid north is a degree
 * off true north here, enough to shift a 0.5 m target by its own size on
 * a 30 m site if the image were stretched into an axis-aligned box). */
function GridOverlay({ overlay, opacity }) {
  const map = useMap();
  useEffect(() => {
    if (!overlay) return undefined;
    const layer = L.imageOverlay.rotated(overlay.image_data_url, overlay.topleft, overlay.topright, overlay.bottomleft, { opacity });
    layer.addTo(map);
    return () => layer.remove();
  }, [overlay, opacity, map]);
  return null;
}

/** Map clicks go to the backend: value of the shown grid there and the nearest reading. */
/** Live basemaps, Google imagery first (the default). Esri goes through
 * this app's tile cache (api.tileUrl) so what was seen online is still
 * there offline; Google and OSM load straight into the browser and are
 * never cached - OSM's tile policy forbids bulk use and it blocked the
 * cache's fetches with 403, Google's terms do not allow it either. */
const BASEMAPS = {
  google: { label: "Google 위성 (비공식 타일, 약관 주의)", url: "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", maxNative: 20, attribution: "&copy; Google" },
  esri: { label: "Esri 위성영상 (오프라인 캐시)", url: api.tileUrl("esri"), maxNative: 19, attribution: "Tiles &copy; Esri" },
  osm: { label: "OpenStreetMap", url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", maxNative: 19, attribution: "&copy; OpenStreetMap contributors" },
  none: { label: "없음" },
};

function BoundsWatcher({ onChange }) {
  const map = useMapEvents({ moveend: () => onChange(map.getBounds()), zoomend: () => onChange(map.getBounds()) });
  useEffect(() => { onChange(map.getBounds()); }, [map]);   // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

/** Pre-download Esri imagery tiles of a small area for a site with no
 * internet. Fields: bbox + zoom range, filled from the map view or from the
 * loaded survey (plus a margin). */
function OfflinePanel({ mapBounds, points }) {
  const source = "esri";
  const [bbox, setBbox] = useState({ min_lat: "", min_lon: "", max_lat: "", max_lon: "" });
  const [zoom, setZoom] = useState({ min: 14, max: 19 });
  const [estimate, setEstimate] = useState(null);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const refresh = () => api.getOfflineTileStatus().then(setStatus).catch(() => {});
  useEffect(() => { refresh(); }, []);

  const useView = () => mapBounds && setBbox({
    min_lat: mapBounds.getSouth().toFixed(5), min_lon: mapBounds.getWest().toFixed(5),
    max_lat: mapBounds.getNorth().toFixed(5), max_lon: mapBounds.getEast().toFixed(5),
  });
  const useSurvey = () => {
    if (!points) return;
    const la = points.lat, lo = points.lon;
    const mLat = 200 / 110540, mLon = 200 / (111320 * Math.cos((Math.min(...la) * Math.PI) / 180));   // 200 m margin
    setBbox({ min_lat: (Math.min(...la) - mLat).toFixed(5), min_lon: (Math.min(...lo) - mLon).toFixed(5),
      max_lat: (Math.max(...la) + mLat).toFixed(5), max_lon: (Math.max(...lo) + mLon).toFixed(5) });
  };
  const req = () => ({ source, min_lat: Number(bbox.min_lat), min_lon: Number(bbox.min_lon), max_lat: Number(bbox.max_lat), max_lon: Number(bbox.max_lon), min_zoom: Number(zoom.min), max_zoom: Number(zoom.max) });
  const valid = Object.values(bbox).every((v) => v !== "" && !Number.isNaN(Number(v)));
  const run = async (fn, set) => {
    setBusy(true); setErr(null);
    try { set(await fn(req())); if (fn === api.downloadOfflineTiles) refresh(); } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };
  const field = (k, label) => (
    <label style={{ fontSize: 11 }}>{label} <input type="number" step="0.00001" value={bbox[k]} onChange={(e) => setBbox({ ...bbox, [k]: e.target.value })} style={{ width: 90 }} /></label>
  );
  return (
    <details style={{ marginTop: 6 }}>
      <summary style={{ fontSize: 12, cursor: "pointer" }}>오프라인 지도 미리 받기 {status && `(Esri ${status.esri?.n_tiles ?? 0}장 저장됨)`}</summary>
      <div style={note}>
        인터넷이 없는 현장에서도 배경지도가 보이도록 조사 지역의 Esri 위성영상 타일을 미리 받아 둡니다. 받아 둔 타일은 위의 "Esri 위성영상 (오프라인 캐시)" 선택에서 그대로 쓰입니다
        (온라인에서 한 번 본 곳도 자동으로 저장됩니다). OpenStreetMap과 Google은 타일 정책상 미리 받을 수 없습니다. 좁은 범위와 필요한 확대단계만 받으세요. 한 번에 5,000장까지입니다.
      </div>
      <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", marginTop: 4 }}>
        <button style={ghost} onClick={useView} disabled={!mapBounds}>현재 화면 범위</button>
        <button style={ghost} onClick={useSurvey} disabled={!points}>자료 범위 +200 m</button>
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 4 }}>
        {field("min_lat", "남")}{field("max_lat", "북")}{field("min_lon", "서")}{field("max_lon", "동")}
        <label style={{ fontSize: 11 }}>확대 <input type="number" min="0" max="19" value={zoom.min} onChange={(e) => setZoom({ ...zoom, min: e.target.value })} style={{ width: 40 }} />~<input type="number" min="0" max="19" value={zoom.max} onChange={(e) => setZoom({ ...zoom, max: e.target.value })} style={{ width: 40 }} /></label>
      </div>
      <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", marginTop: 4 }}>
        <button style={ghost} disabled={!valid || busy} onClick={() => run(api.estimateOfflineTiles, setEstimate)}>타일 수 확인</button>
        <button style={button} disabled={!valid || busy || (estimate && estimate.exceeds_max)} onClick={() => run(api.downloadOfflineTiles, setResult)}>{busy ? "받는 중..." : "다운로드"}</button>
        {estimate && <span style={{ fontSize: 11, color: estimate.exceeds_max ? "#b42318" : undefined }}>{estimate.n_tiles.toLocaleString()}장 · 약 {estimate.estimated_mb} MB{estimate.exceeds_max && " · 최대치 초과, 범위나 확대단계를 줄이세요"}</span>}
      </div>
      {result && <div style={{ fontSize: 11, marginTop: 4 }}>완료: 새로 {result.n_downloaded.toLocaleString()}장, 이미 있던 {result.n_already_cached.toLocaleString()}장{result.n_failed > 0 && <span style={{ color: "#b42318" }}>, 실패 {result.n_failed}장</span>}</div>}
      {err && <div style={warn}>{err}</div>}
      {status && ["esri"].map((k) => status[k] && (
        <div key={k} style={{ fontSize: 11 }}>{status[k].label}: {status[k].n_tiles.toLocaleString()}장 ({status[k].size_mb} MB){status[k].zoom_levels.length > 0 && ` · 확대 ${status[k].zoom_levels.join(", ")}`}</div>
      ))}
    </details>
  );
}

function ClickInspector({ onClick }) {
  useMapEvents({ click: (e) => onClick(e.latlng.lat, e.latlng.lng) });
  return null;
}

/** Colour bar of the shown grid with the backend's tick values. */
function Legend({ overlay, unit }) {
  if (!overlay) return null;
  const ticks = Array.isArray(overlay.legend_ticks) && overlay.legend_ticks.length >= 2 ? overlay.legend_ticks : [overlay.vmin, overlay.vmax];
  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ height: 12, borderRadius: 3, border: "1px solid #c9d3c4", backgroundImage: `url(${overlay.colorbar_data_url})`, backgroundSize: "100% 100%" }} />
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#4f5e4a" }}>
        {ticks.map((v, i) => <span key={i}>{fmt(v, Math.abs(v) < 10 ? 1 : 0)}</span>)}
      </div>
      <div style={{ ...note, textAlign: "right" }}>{unit}</div>
    </div>
  );
}

/** Readings along one line, raw and processed, against distance. */
function Profile({ data, onClose }) {
  if (!data) return null;
  const W = 820, H = 220, padL = 50, padR = 10, padT = 10, padB = 26;
  const all = data.series.flatMap((s) => [...s.value, ...(s.raw_value || [])]).filter(Number.isFinite);
  const dmax = Math.max(...data.series.flatMap((s) => s.distance_m), 1);
  const lo = Math.min(...all), hi = Math.max(...all);
  const X = (d) => padL + ((W - padL - padR) * d) / dmax;
  const Y = (v) => padT + (H - padT - padB) * (1 - (v - lo) / Math.max(1e-9, hi - lo));
  const path = (ds, vs) => ds.map((d, i) => (Number.isFinite(vs[i]) ? `${i ? "L" : "M"}${X(d).toFixed(1)},${Y(vs[i]).toFixed(1)}` : "")).join(" ");
  const colors = { 1: "#2f5a22", 2: "#1d4ed8" };
  return (
    <div style={{ position: "absolute", left: 10, right: 10, bottom: 10, zIndex: 1000, background: "white", border: "1px solid #c9d3c4", borderRadius: 8, padding: 8, boxShadow: "0 2px 8px rgba(0,0,0,0.15)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
        <span>측선 프로파일 — 파일 [{data.file_index}] 트랙 {data.line_id}{data.processed ? " (회색: 전처리 전, 색: 전처리 후)" : " (원 자료)"} · 탐침 {data.series.map((s) => s.sensor).join(", ")}</span>
        <button style={{ ...ghost, padding: "1px 8px" }} onClick={onClose}>닫기</button>
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
        {data.series.map((s) => s.raw_value && <path key={`r${s.sensor}`} d={path(s.distance_m, s.raw_value)} fill="none" stroke="#bbb" strokeWidth={0.8} />)}
        {data.series.map((s) => <path key={`v${s.sensor}`} d={path(s.distance_m, s.value)} fill="none" stroke={colors[s.sensor] || "#b42318"} strokeWidth={1.2} />)}
        {[lo, (lo + hi) / 2, hi].map((v, i) => <text key={i} x={padL - 4} y={Y(v) + 3} fontSize={10} textAnchor="end" fill="#4f5e4a">{fmt(v, 0)}</text>)}
        {[0, 0.25, 0.5, 0.75, 1].map((f) => <text key={f} x={X(f * dmax)} y={H - 8} fontSize={10} textAnchor="middle" fill="#4f5e4a">{fmt(f * dmax, 0)} m</text>)}
      </svg>
    </div>
  );
}

/** A small inline amplitude spectrum: log amplitude vs frequency. */
function Spectrum({ spectrum, gaitBand }) {
  if (!spectrum || !spectrum.hz.length) return <div style={note}>조용한 구간이 짧아 스펙트럼을 내지 못했습니다.</div>;
  const W = 380, H = 110, padL = 34, padB = 18;
  const fmax = Math.max(...spectrum.hz);
  const logs = spectrum.amplitude.map((a) => Math.log10(Math.max(a, 1e-4)));
  const lo = Math.min(...logs), hi = Math.max(...logs);
  const X = (f) => padL + ((W - padL - 4) * f) / fmax;
  const Y = (l) => 4 + (H - padB - 4) * (1 - (l - lo) / Math.max(1e-9, hi - lo));
  const path = spectrum.hz.map((f, i) => `${i ? "L" : "M"}${X(f).toFixed(1)},${Y(logs[i]).toFixed(1)}`).join(" ");
  return (
    <svg width={W} height={H} style={{ display: "block", background: "white", border: "1px solid #d9e2d6", borderRadius: 4 }}>
      <rect x={X(gaitBand[0])} y={4} width={X(gaitBand[1]) - X(gaitBand[0])} height={H - padB - 4} fill="#f6e3c9" />
      <path d={path} fill="none" stroke="#2f5a22" strokeWidth={1.2} />
      {spectrum.peaks.map((p, i) => (
        <g key={i}>
          <line x1={X(p.hz)} x2={X(p.hz)} y1={4} y2={H - padB} stroke="#b42318" strokeDasharray="3 2" />
          <text x={X(p.hz) + 3} y={14 + 11 * i} fontSize={10} fill="#b42318">{p.hz} Hz{p.spatial_period_m ? ` (${p.spatial_period_m} m)` : ""}</text>
        </g>
      ))}
      {[0, 5, 10, 15, 20].filter((f) => f <= fmax).map((f) => (
        <text key={f} x={X(f)} y={H - 5} fontSize={10} fill="#6b7a66" textAnchor="middle">{f}</text>
      ))}
      <text x={2} y={12} fontSize={10} fill="#6b7a66">nT/√Hz (log)</text>
      <text x={W - 4} y={H - 5} fontSize={10} fill="#6b7a66" textAnchor="end">Hz</text>
    </svg>
  );
}

export default function App() {
  const [version, setVersion] = useState(null);
  const [projectId, setProjectId] = useState(null);
  const [summary, setSummary] = useState(null);
  const [points, setPoints] = useState(null);
  const [sensor, setSensor] = useState(0);
  const [showProcessed, setShowProcessed] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const [qc, setQc] = useState(null);
  const [qcBusy, setQcBusy] = useState(false);

  const [pre, setPre] = useState(null);
  const [preBusy, setPreBusy] = useState(false);
  const [lpf, setLpf] = useState(2);
  const [lpfByFile, setLpfByFile] = useState({});
  const [samplesPerM, setSamplesPerM] = useState(10);
  const [hampelK, setHampelK] = useState(3);
  const [headingOn, setHeadingOn] = useState(true);

  const [cell, setCell] = useState(0.125);
  const [maxDist, setMaxDist] = useState("");          // "" = auto from the line spacing
  const [gridMethod, setGridMethod] = useState("minimum_curvature");
  const [detrend, setDetrend] = useState(1);
  const [kind, setKind] = useState("anomaly");
  const [gridFile, setGridFile] = useState("");
  const [gridSensors, setGridSensors] = useState("");
  const [grid, setGrid] = useState(null);
  const [gridBusy, setGridBusy] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [showPoints, setShowPoints] = useState(true);
  const [opacity, setOpacity] = useState(0.85);

  const [repeat, setRepeat] = useState(null);
  const [repeatBusy, setRepeatBusy] = useState(false);
  const [compareCell, setCompareCell] = useState(0.25);

  const [sensorHeight, setSensorHeight] = useState(0.5);
  const [heightsByFile, setHeightsByFile] = useState({});
  const [tMethod, setTMethod] = useState("thd");
  const [derivative, setDerivative] = useState("thd");
  const [peakK, setPeakK] = useState(20);
  const [thresholdK, setThresholdK] = useState(5);
  const [targets, setTargets] = useState(null);
  const [targetsBusy, setTargetsBusy] = useState(false);
  const [maxTargets, setMaxTargets] = useState("");

  const [sample, setSample] = useState(null);
  const [area, setArea] = useState(null);              // {polygon, n_inside, n_total}
  const [drawMode, setDrawMode] = useState(null);      // null | "polygon" | "rect"
  const [drawPts, setDrawPts] = useState([]);
  const [profile, setProfile] = useState(null);
  const [profileFile, setProfileFile] = useState(0);
  const [profileLine, setProfileLine] = useState("");

  const [basemaps, setBasemaps] = useState([]);        // GeoTIFF overlays with image_data_url
  const [basemapOpacity, setBasemapOpacity] = useState(1);
  const [basemap, setBasemap] = useState(() => { try { return localStorage.getItem("minemag.basemap.v2") || "google"; } catch { return "google"; } });
  const [mapBounds, setMapBounds] = useState(null);
  useEffect(() => { try { localStorage.setItem("minemag.basemap.v2", basemap); } catch { /* private window */ } }, [basemap]);
  const [tileLayers, setTileLayers] = useState([]);    // registered local tile folders
  const [tilePath, setTilePath] = useState("");
  const [mapBusy, setMapBusy] = useState(false);

  useEffect(() => {
    api.getVersion().then(setVersion).catch(() => {});
    const existing = new URLSearchParams(window.location.search).get("project");
    if (existing) {
      setProjectId(existing);
      api.getSummary(existing).then(setSummary).catch((e) => setError(e.message));
      api.getArea(existing).then((a) => setArea(a.polygon ? a : null)).catch(() => {});
      api.listBasemaps(existing)
        .then((r) => Promise.all(r.basemaps.map((b) => api.getBasemap(existing, b.name))))
        .then(setBasemaps)
        .catch(() => {});
    }
  }, []);

  const resetDerived = () => {
    setQc(null);
    setPre(null);
    setGrid(null);
    setShowPoints(true);                                  // the grid is gone: nothing else shows the data
    setRepeat(null);
    setTargets(null);
  };

  const guard = async (setFlag, fn) => {
    setFlag(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e.message);
    } finally {
      setFlag(false);
    }
  };

  const upload = (files) => {
    if (!files.length) return;
    guard(setBusy, async () => {
      const pid = projectId || (await api.createProject()).project_id;
      setProjectId(pid);
      const s = await api.uploadFiles(pid, files);
      resetDerived();
      setSummary(s);
    });
  };

  useEffect(() => {
    if (!projectId || !summary) return;
    api.getPoints(projectId, sensor || undefined, 8000, showProcessed && !!pre).then(setPoints).catch((e) => setError(e.message));
  }, [sensor, projectId, summary, pre, showProcessed]);

  const files = summary?.files || [];
  const gridReq = () => ({
    cell_size_m: Number(cell),
    max_distance_m: maxDist === "" ? null : Number(maxDist),
    method: gridMethod,
    detrend_order: Number(detrend),
    kind,
    file_index: gridFile === "" ? null : Number(gridFile),
    sensors: gridSensors === "" ? null : [Number(gridSensors)],
  });

  const runQc = () => guard(setQcBusy, async () => setQc(await api.runQc(projectId, {})));
  const runPre = () => guard(setPreBusy, async () => {
    const byFile = {};
    for (const [k, v] of Object.entries(lpfByFile)) if (v !== "" && Number.isFinite(Number(v))) byFile[k] = Number(v);
    const r = await api.runPreprocess(projectId, {
      samples_per_m: Number(samplesPerM), hampel_k: Number(hampelK), lpf_hz: Number(lpf), lpf_hz_by_file: byFile, heading_correction: headingOn,
    });
    setPre(r);
    setGrid(null);
    setRepeat(null);
    setTargets(null);
  });
  const undoPre = () => guard(setPreBusy, async () => {
    await api.resetPreprocess(projectId);
    setPre(null);
    setGrid(null);
    setRepeat(null);
    setTargets(null);
  });
  const runGrid = () => guard(setGridBusy, async () => {
    setGrid(await api.makeGrid(projectId, gridReq()));
    setShowGrid(true);
  });
  const exportGrid = (fmt) => guard(setGridBusy, () => api.downloadGrid(projectId, gridReq(), fmt));
  const runRepeat = () => guard(setRepeatBusy, async () => setRepeat(await api.compareRepeats(projectId, { compare_cell_m: Number(compareCell) })));
  const addBasemap = (file) => guard(setMapBusy, async () => {
    if (!file) return;
    const pid = projectId || (await api.createProject()).project_id;
    setProjectId(pid);
    const r = await api.uploadBasemap(pid, file);
    const full = await api.getBasemap(pid, file.name);
    setBasemaps((prev) => [...prev.filter((b) => b.name !== full.name), full]);
    return r;
  });
  const dropBasemap = (name) => guard(setMapBusy, async () => {
    await api.deleteBasemap(projectId, name);
    setBasemaps((prev) => prev.filter((b) => b.name !== name));
  });
  const pickTiles = () => guard(setMapBusy, async () => {
    const r = await api.pickLocalTileFolder();
    if (r.path) setTilePath(r.path);
  });
  const addTiles = () => guard(setMapBusy, async () => {
    if (!tilePath) return;
    const layer = await api.registerLocalTiles({ path: tilePath });
    setTileLayers((prev) => [...prev, layer]);
    setTilePath("");
  });
  const dropTiles = (id) => guard(setMapBusy, async () => {
    await api.unregisterLocalTiles(id);
    setTileLayers((prev) => prev.filter((l) => l.id !== id));
  });

  const applyArea = async (polygon) => {
    try {
      const a = await api.setArea(projectId, polygon);
      setArea(a);
      setQc(null);
      setGrid(null);
      setShowPoints(true);
      setRepeat(null);
      setTargets(null);
      if (pre) setPre(null);                              // the pre-processing was on the whole set: redo it inside the area
      setDrawMode(null);
      setDrawPts([]);
    } catch (e) {
      setError(e.message);
    }
  };
  /** The area as a GeoJSON Polygon (WGS84, [lon, lat]) so the same
   * boundary can be reloaded next time and opened in QGIS. */
  const saveArea = () => {
    if (!area || !area.polygon) return;
    const ring = area.polygon.map(([la, lo]) => [lo, la]);
    ring.push(ring[0]);
    const gj = {
      type: "Feature",
      properties: { name: "MineMag 영역", saved: new Date().toISOString(), files: summary ? summary.files.map((f) => f.filename) : [], n_inside: area.n_inside, n_total: area.n_total },
      geometry: { type: "Polygon", coordinates: [ring] },
    };
    const blob = new Blob([JSON.stringify(gj, null, 2)], { type: "application/geo+json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const stem = summary && summary.files.length ? summary.files[0].filename.replace(/\.[^.]+$/, "") : "minemag";
    a.download = `${stem}_영역.geojson`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  /** Accepts the file saveArea wrote, any GeoJSON Feature / FeatureCollection /
   * Polygon in WGS84 (first ring of the first polygon), or {"polygon": [[lat, lon], ...]}. */
  const loadAreaFile = async (file) => {
    if (!file) return;
    try {
      const obj = JSON.parse(await file.text());
      let geom = obj;
      if (obj.type === "FeatureCollection") geom = obj.features && obj.features[0] && obj.features[0].geometry;
      else if (obj.type === "Feature") geom = obj.geometry;
      let polygon;
      if (Array.isArray(obj.polygon)) polygon = obj.polygon;
      else if (geom && geom.type === "Polygon") polygon = geom.coordinates[0].map(([lo, la]) => [la, lo]);
      else if (geom && geom.type === "MultiPolygon") polygon = geom.coordinates[0][0].map(([lo, la]) => [la, lo]);
      if (!polygon || polygon.length < 3) throw new Error("다각형을 찾지 못했습니다 (GeoJSON Polygon 또는 {polygon: [[lat, lon], ...]} 형식).");
      if (polygon.length > 3 && polygon[0][0] === polygon[polygon.length - 1][0] && polygon[0][1] === polygon[polygon.length - 1][1]) polygon = polygon.slice(0, -1);
      if (!polygon.every(([la, lo]) => Math.abs(la) <= 90 && Math.abs(lo) <= 180)) throw new Error("위경도(WGS84) 좌표가 아닙니다. QGIS에서 EPSG:4326으로 저장하세요.");
      await applyArea(polygon);
    } catch (e) {
      setError(`영역 파일: ${e.message}`);
    }
  };
  const dropArea = async () => {
    try {
      await api.clearArea(projectId);
      setArea(null);
      setQc(null);
      setGrid(null);
      setShowPoints(true);
      setRepeat(null);
      setTargets(null);
      if (pre) setPre(null);
    } catch (e) {
      setError(e.message);
    }
  };
  const mapClick = (lat, lon) => {
    if (drawMode === "polygon") {
      setDrawPts((p) => [...p, [lat, lon]]);
      return;
    }
    if (drawMode === "rect") {
      if (drawPts.length === 0) {
        setDrawPts([[lat, lon]]);
      } else {
        const [a, b] = [drawPts[0], [lat, lon]];
        applyArea([[a[0], a[1]], [a[0], b[1]], [b[0], b[1]], [b[0], a[1]]]);
      }
      return;
    }
    inspect(lat, lon);
  };
  const inspect = async (lat, lon) => {
    if (!projectId || !summary) return;
    try {
      setSample(await api.sampleAt(projectId, lat, lon));
    } catch (e) {
      setError(e.message);
    }
  };
  const showProfile = async (fileIndex, lineId) => {
    if (!projectId || lineId === "" || lineId === null || lineId === undefined) return;
    try {
      setProfile(await api.lineProfile(projectId, Number(fileIndex), Number(lineId)));
      setProfileFile(Number(fileIndex));
      setProfileLine(Number(lineId));
    } catch (e) {
      setError(e.message);
    }
  };
  const stepProfile = (dir) => {
    if (!profile) return;
    const i = profile.tracks.indexOf(profile.line_id);
    const next = profile.tracks[i + dir];
    if (next !== undefined) showProfile(profile.file_index, next);
  };

  const runTargets = () => guard(setTargetsBusy, async () => {
    const byFile = {};
    for (const [k, v] of Object.entries(heightsByFile)) if (v !== "" && Number.isFinite(Number(v))) byFile[k] = Number(v);
    setTargets(await api.pickTargets(projectId, {
      method: tMethod, derivative, peak_k: Number(peakK), threshold_k: Number(thresholdK),
      sensor_height_m: Number(sensorHeight), sensor_heights_by_file: byFile,
      cell_size_m: Number(cell), grid_method: gridMethod, detrend_order: Number(detrend),
      max_targets: maxTargets === "" ? null : Number(maxTargets),
    }));
  });

  const vmax = useMemo(() => {
    if (!points) return 200;
    const a = points.tfa.map(Math.abs).sort((x, y) => x - y);
    return Math.max(50, a[Math.floor(a.length * 0.98)] || 200);
  }, [points]);

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <div style={panel}>
        <div style={{ fontSize: 18, fontWeight: 700 }}>MineMag</div>
        <div style={note}>드론·사족보행로봇 자력탐사 지뢰·불발탄 탐지 {version?.commit && `· ${version.commit}`}</div>

        <h2 style={h2}>0. 배경지도</h2>
        <div style={note}>
          정사영상 GeoTIFF를 올리면 지도 위에 겹칩니다(2000픽셀로 줄여서 표시). 수 GB짜리 큰 정사영상은 QGIS "XYZ 타일 생성"이나 gdal2tiles.py로
          한 번 타일로 만든 뒤 그 폴더를 지정하면 원본 크기 그대로, 파일을 올리지 않고 바로 봅니다.
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", marginTop: 6 }}>
          <label style={{ fontSize: 12 }}>GeoTIFF <input type="file" accept=".tif,.tiff" disabled={mapBusy} onChange={(e) => addBasemap(e.target.files[0])} /></label>
          <label style={{ fontSize: 12 }}>바탕 <select value={basemap} onChange={(e) => setBasemap(e.target.value)}>
            {Object.entries(BASEMAPS).map(([k, b]) => <option key={k} value={k}>{b.label}</option>)}
          </select></label>
          {basemaps.length > 0 && <label style={{ fontSize: 12 }}>불투명도 <input type="range" min="0.2" max="1" step="0.05" value={basemapOpacity} onChange={(e) => setBasemapOpacity(Number(e.target.value))} /></label>}
        </div>
        {basemaps.map((b) => (
          <div key={b.name} style={{ ...row, fontSize: 11 }}><span>{b.name}</span><button style={{ ...ghost, padding: "1px 6px" }} onClick={() => dropBasemap(b.name)}>지우기</button></div>
        ))}
        <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", marginTop: 6 }}>
          <input type="text" placeholder="타일 폴더 경로 ({z}/{x}/{y}.png)" value={tilePath} onChange={(e) => setTilePath(e.target.value)} style={{ flex: 1, minWidth: 180, fontSize: 11 }} />
          <button style={ghost} disabled={mapBusy} onClick={pickTiles}>폴더 선택</button>
          <button style={button} disabled={mapBusy || !tilePath} onClick={addTiles}>타일 추가</button>
        </div>
        {tileLayers.map((l) => (
          <div key={l.id} style={{ ...row, fontSize: 11 }}><span>{l.label} (z {l.min_zoom}~{l.max_zoom}, {l.scheme})</span><button style={{ ...ghost, padding: "1px 6px" }} onClick={() => dropTiles(l.id)}>지우기</button></div>
        ))}
        <OfflinePanel mapBounds={mapBounds} points={points} />

        <h2 style={h2}>1. 자료 올리기 (MagDrone R3 .asc / Log_Drone .csv)</h2>
        <input type="file" multiple accept=".asc,.txt,.csv" disabled={busy} onChange={(e) => upload(Array.from(e.target.files))} />
        <div style={{ ...note, marginTop: 4 }}>
          MagDrone DataTool이 내보낸 ASC(플럭스게이트 2탐침, 200 Hz) 또는 광펌핑 스칼라 자력계의 Log_Drone CSV(J1·J2, 50 Hz). 형식은 첫 줄로 자동 판별하고 섞어 올려도 됩니다. 여러 비행·반복 측정 파일을 함께 선택하면 시간순으로 합쳐집니다. Log_Drone은 총자기장에서 IGRF를 빼 이상으로 바꾸고, 트랙 번호가 없으므로 비행 경로의 직선 구간을 트랙으로 만듭니다.
        </div>
        {busy && <div style={{ marginTop: 6 }}>읽는 중...</div>}
        {error && <div style={{ marginTop: 6, color: "#b42318" }}>{error}</div>}

        {summary && (
          <>
            <h2 style={h2}>적재 요약</h2>
            <div style={row}><span>파일 / 행</span><b>{summary.n_files} / {summary.n_rows.toLocaleString()}</b></div>
            <div style={row}><span>탐침</span><b>{summary.sensors.join(", ")} {Object.keys(summary.sensor_separation_m).length ? `(간격 ${Object.values(summary.sensor_separation_m).map((v) => fmt(v, 2)).join(", ")} m)` : "(같은 위치 기록, 간격 미상)"}</b></div>
            <div style={row}><span>샘플 간격 / 트랙 / 시간</span><b>{fmt(summary.sample_interval_ms, 0)} ms / {summary.n_tracks} / {fmt(summary.duration_min)} 분</b></div>
            <div style={row}><span>속도 / GPS 고도</span><b>{fmt(summary.speed_m_s?.median, 2)} m/s / {fmt(summary.altitude_m?.min)} ~ {fmt(summary.altitude_m?.max)} m</b></div>
            <div style={row}><span>총자기장 이상</span><b>{fmt(summary.tfa_nt?.min, 0)} ~ {fmt(summary.tfa_nt?.max, 0)} nT (σ {fmt(summary.tfa_nt?.std, 0)})</b></div>
            {files.map((f, i) => (
              <div key={i} style={{ ...note, padding: "2px 0" }}>
                [{i}] {f.filename} — {f.format === "log_drone_csv" ? "광펌핑 J1·J2" : "MagDrone R3"} · {f.start_date || "날짜 미상"} · {f.n_rows.toLocaleString()}행 · {f.settings.join(" ")}
              </div>
            ))}

            <h2 style={h2}>1-1. 영역 지정</h2>
            <div style={note}>
              지도에 사각형이나 다각형을 그리면 그 안의 측점만 QC·전처리·격자·비교·피킹에 씁니다. 전처리가 못 걸러낸 선회·이착륙 구간은 이렇게 바깥으로 빼면 됩니다.
              사각형은 모서리 두 점, 다각형은 꼭짓점을 차례로 찍고 "완료".
            </div>
            <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", marginTop: 6 }}>
              <button style={drawMode === "rect" ? button : ghost} onClick={() => { setDrawMode(drawMode === "rect" ? null : "rect"); setDrawPts([]); }}>사각형</button>
              <button style={drawMode === "polygon" ? button : ghost} onClick={() => { setDrawMode(drawMode === "polygon" ? null : "polygon"); setDrawPts([]); }}>다각형</button>
              {drawMode === "polygon" && <button style={button} disabled={drawPts.length < 3} onClick={() => applyArea(drawPts)}>완료 ({drawPts.length}점)</button>}
              {drawMode && <button style={ghost} onClick={() => { setDrawMode(null); setDrawPts([]); }}>취소</button>}
              {area && <button style={ghost} onClick={dropArea}>영역 지우기</button>}
              {area && <button style={ghost} onClick={saveArea} title="GeoJSON(WGS84)으로 저장 - 다음 해석 때 불러오거나 QGIS에서 엽니다">영역 저장</button>}
              <label style={{ ...ghost, display: "inline-block" }} title="저장해 둔 영역 GeoJSON을 불러와 같은 영역으로 해석합니다">
                영역 불러오기<input type="file" accept=".geojson,.json" style={{ display: "none" }} disabled={!projectId || !summary} onChange={(e) => { loadAreaFile(e.target.files[0]); e.target.value = ""; }} />
              </label>
            </div>
            {drawMode && <div style={warn}>지도를 클릭해 {drawMode === "rect" ? "모서리 두 점을" : "꼭짓점을"} 찍으세요.</div>}
            {area && (
              <div style={row}><span>영역 안 측점</span><b>{area.n_inside?.toLocaleString()} / {area.n_total?.toLocaleString()} ({fmt((100 * area.n_inside) / Math.max(1, area.n_total), 0)}%)</b></div>
            )}

            <h2 style={h2}>2. 플랫폼 잡음 QC</h2>
            <div style={note}>
              원 자료의 조용한 구간에서 배경 잡음, 두 탐침의 상관, 잡음 스펙트럼을 잽니다. 보행·회전익 주기가 보이면 공간 주기를 알려 주고,
              스펙트럼의 잡음 대역이 전처리의 차단 주파수를 정하는 근거가 됩니다.
            </div>
            <button style={{ ...button, marginTop: 6 }} disabled={qcBusy} onClick={runQc}>{qcBusy ? "계산 중..." : "QC 실행"}</button>
            {qc && qc.files.map((f) => (
              <div key={f.file_index} style={box}>
                <div style={{ fontWeight: 700, fontSize: 12 }}>[{f.file_index}] {f.filename}</div>
                {f.error ? <div style={warn}>{f.error}</div> : (
                  <>
                    <div style={row}><span>표본율 / 속도 / 조용한 비율</span><b>{fmt(f.sample_rate_hz, 0)} Hz / {fmt(f.speed_m_s, 2)} m/s / {fmt(100 * f.quiet_fraction, 0)}%</b></div>
                    {Object.entries(f.per_probe).map(([k, p]) => (
                      <div key={k} style={row}>
                        <span>탐침 {k} 잡음 rms (전대역 / &gt;1 Hz / &gt;5 Hz)</span>
                        <b>{fmt(p.quiet_rms_nt)} / {fmt(p.quiet_rms_above_1hz_nt)} / {fmt(p.quiet_rms_above_5hz_nt)} nT</b>
                      </div>
                    ))}
                    {f.gradient && (
                      <div style={row}><span>두 탐침 상관 / 차의 rms</span><b>{fmt(f.gradient.probe_correlation_above_1hz, 2)} / {fmt(f.gradient.difference_rms_nt)} nT ({fmt(f.gradient.cancellation_pct, 0)}% 감소)</b></div>
                    )}
                    <div style={{ marginTop: 6 }}><Spectrum spectrum={f.spectrum} gaitBand={qc.gait_band_hz} /></div>
                    {f.verdict.map((line, i) => <div key={i} style={{ ...warn, marginTop: 4 }}>• {line}</div>)}
                    {f.sensor_pair && !f.sensor_pair.error && (
                      <div style={{ marginTop: 8 }}>
                        <div style={{ fontWeight: 700, fontSize: 12 }}>센서 쌍 진단 (센서 {f.sensor_pair.sensors.join(" · ")})</div>
                        <div style={row}><span>오프셋 / 차의 산포 (정지 / 비행)</span><b>{fmt(f.sensor_pair.offset_nt, 1)} / {fmt(f.sensor_pair.difference_std_still_nt)} / {fmt(f.sensor_pair.difference_std_flight_nt)} nT</b></div>
                        {f.sensor_pair.direction && (
                          <div style={row}><span>방향 반전 효과 (왕복 평균 차)</span><b>{Object.entries(f.sensor_pair.direction.forward_minus_backward_nt).map(([s, v]) => `센서 ${s} ${v >= 0 ? "+" : ""}${fmt(v, 1)}`).join(" · ")} nT</b></div>
                        )}
                        {f.sensor_pair.background_std_nt && (
                          <div style={row}><span>배경 산포 (표적 제외)</span><b>{Object.entries(f.sensor_pair.background_std_nt).map(([s, v]) => `센서 ${s} ${fmt(v)}`).join(" · ")} nT</b></div>
                        )}
                        {f.sensor_pair.attitude && (
                          <div style={row}><span>자세 회귀 뒤 배경 산포</span><b>{Object.entries(f.sensor_pair.attitude.per_sensor).map(([s, p]) => `센서 ${s} ${fmt(p.std_before_nt)}→${fmt(p.std_after_nt)}`).join(" · ")} nT</b></div>
                        )}
                        {f.sensor_pair.verdict.map((line, i) => <div key={`sp${i}`} style={{ ...note, marginTop: 3 }}>• {line}</div>)}
                      </div>
                    )}
                  </>
                )}
              </div>
            ))}

            <h2 style={h2}>3. 전처리 (논문 2.6절)</h2>
            <div style={note}>
              리샘플링(속도 × m당 표본 수) → 적응형 Hampel(k × MAD, 급경사 구간은 창 확장) → 영위상 가우시안 저역통과(로봇 2 Hz, 드론 5 Hz) →
              트랙 안의 선회·대기 구간 제외 → 왕복 헤딩 보정(신뢰될 때만 적용). 이후의 격자·비교·피킹은 전처리된 자료를 씁니다.
            </div>
            <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", marginTop: 6 }}>
              <label>m당 표본 <input type="number" step="1" min="2" value={samplesPerM} onChange={(e) => setSamplesPerM(e.target.value)} style={inputW(50)} /></label>
              <label>Hampel k <input type="number" step="0.5" min="1" value={hampelK} onChange={(e) => setHampelK(e.target.value)} style={inputW(45)} /></label>
              <label>LPF fc(Hz) <input type="number" step="0.5" min="0.2" value={lpf} onChange={(e) => setLpf(e.target.value)} style={inputW(50)} /></label>
              {files.length > 1 && files.map((f, i) => (
                <label key={i} title={f.filename}>[{i}] <input type="number" step="0.5" min="0.2" placeholder={lpf} value={lpfByFile[i] ?? ""} onChange={(e) => setLpfByFile({ ...lpfByFile, [i]: e.target.value })} style={inputW(45)} /></label>
              ))}
              <label><input type="checkbox" checked={headingOn} onChange={(e) => setHeadingOn(e.target.checked)} /> 헤딩 보정</label>
              <button style={button} disabled={preBusy} onClick={runPre}>{preBusy ? "계산 중..." : pre ? "다시 적용" : "전처리 적용"}</button>
              {pre && <button style={ghost} disabled={preBusy} onClick={undoPre}>되돌리기</button>}
            </div>
            {pre && pre.files.map((f) => (
              <div key={f.file_index} style={box}>
                <div style={{ fontWeight: 700, fontSize: 12 }}>[{f.file_index}] {f.filename}</div>
                <div style={row}><span>속도 / 리샘플링</span><b>{fmt(f.speed_m_s, 2)} m/s / {fmt(f.resample_hz, 1)} Hz ({f.n_in.toLocaleString()} → {f.n_out.toLocaleString()}행)</b></div>
                <div style={row}><span>스파이크 / LPF</span><b>{f.n_spikes}개 / {fmt(f.lpf_hz, 1)} Hz</b></div>
                <div style={row}><span>잡음(4차 차분) 전 → 후</span><b>{fmt(f.noise_before_nt, 2)} → {fmt(f.noise_after_nt, 2)} nT</b></div>
                <div style={row}><span>측선 / 제외 비율</span><b>{f.n_lines} / {fmt(100 * f.turn_fraction, 0)}%</b></div>
                <div style={row}>
                  <span>헤딩 보정</span>
                  <b>{f.heading_correction.per_sensor && Object.keys(f.heading_correction.per_sensor).length
                    ? Object.entries(f.heading_correction.per_sensor).map(([s, e]) => `센서 ${s}: ${e.applied ? `적용 ${fmt(e.offset_nt)} nT` : `미적용${e.offset_nt !== null && e.offset_nt !== undefined ? ` (${fmt(e.offset_nt)} ± ${fmt(e.offset_spread_nt)})` : ""}`}`).join(" · ")
                    : f.heading_correction.applied ? `적용 ${fmt(f.heading_correction.offset_nt)} nT` : `미적용${f.heading_correction.offset_nt !== null && f.heading_correction.offset_nt !== undefined ? ` (측정 ${fmt(f.heading_correction.offset_nt)} ± ${fmt(f.heading_correction.offset_spread_nt)} nT)` : ""}`}</b>
                </div>
                {f.heading_correction.reason && <div style={warn}>• {f.heading_correction.reason}</div>}
              </div>
            ))}

            <h2 style={h2}>4. 격자와 파생 격자</h2>
            <div style={note}>
              블록평균 뒤 장력 최소곡률(논문) 또는 선형 보간, 셀 0.125 m. 가장 가까운 측점에서 설정 거리보다 먼 셀은 비웁니다.
              추세면(평면)을 빼서 잔차 이상을 만들고, 그 위에서 THD(총수평미분)·ASA(해석신호)를 계산합니다. THD는 자화 방향과 무관하게 이상체 직상부에서 최대가 됩니다.
            </div>
            <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", marginTop: 6 }}>
              <label>셀(m) <input type="number" step="0.025" min="0.02" value={cell} onChange={(e) => setCell(e.target.value)} style={inputW(60)} /></label>
              <label title="비워 두면 측선 간격의 0.75배(0.6~5 m)로 자동">최대 거리(m) <input type="number" step="0.1" min="0.1" placeholder="자동" value={maxDist} onChange={(e) => setMaxDist(e.target.value)} style={inputW(55)} /></label>
              <select value={gridMethod} onChange={(e) => setGridMethod(e.target.value)}>
                <option value="minimum_curvature">최소곡률(장력)</option>
                <option value="linear">선형</option>
              </select>
              <select value={detrend} onChange={(e) => setDetrend(e.target.value)}>
                <option value={0}>추세 제거 없음</option>
                <option value={1}>평면 추세 제거</option>
                <option value={2}>2차 추세 제거</option>
              </select>
              <select value={kind} onChange={(e) => setKind(e.target.value)}>
                <option value="anomaly">잔차 이상</option>
                <option value="rtp">극자화 변환(RTP)</option>
                <option value="thd">THD</option>
                <option value="asa">ASA</option>
                <option value="tilt">틸트각</option>
                <option value="vd1">1차 수직미분</option>
              </select>
              <select value={gridFile} onChange={(e) => setGridFile(e.target.value)}>
                <option value="">모든 파일</option>
                {files.map((f, i) => <option key={i} value={i}>[{i}] {f.filename.slice(0, 28)}</option>)}
              </select>
              <select value={gridSensors} onChange={(e) => setGridSensors(e.target.value)}>
                <option value="">두 탐침</option>
                {summary.sensors.map((s) => <option key={s} value={s}>탐침 {s}</option>)}
              </select>
              <button style={button} disabled={gridBusy} onClick={runGrid}>{gridBusy ? "계산 중..." : "격자 만들기"}</button>
            </div>
            {grid && (
              <div style={{ marginTop: 6 }}>
                <div style={row}><span>자료 / 방법</span><b>{grid.stats.processed ? "전처리됨" : "원 자료"} / {grid.stats.method === "minimum_curvature" ? "최소곡률" : "선형"}{grid.stats.detrend_order ? `, ${grid.stats.detrend_order}차 추세 제거(σ ${fmt(grid.stats.trend_std_nt)} nT)` : ""}</b></div>
                <div style={row}><span>덮임 / 셀 수</span><b>{fmt(grid.stats.coverage_pct)}% / {grid.stats.n_cells.toLocaleString()}</b></div>
                <div style={row}><span>최대 거리</span><b>{fmt(grid.stats.max_distance_m)} m{grid.stats.max_distance_auto ? ` (자동${grid.stats.line_spacing_m ? `, 측선 간격 ${fmt(grid.stats.line_spacing_m)} m` : ""})` : ""}</b></div>
                <div style={row}><span>{({ anomaly: "이상", rtp: "RTP", thd: "THD", asa: "ASA", tilt: "틸트각", vd1: "1차 수직미분" })[grid.stats.kind]} 범위 / rms</span><b>{fmt(grid.stats.min, 1)} ~ {fmt(grid.stats.max, 1)} / {fmt(grid.stats.rms, 1)} {({ anomaly: "nT", rtp: "nT", thd: "nT/m", asa: "nT/m", tilt: "°", vd1: "nT/m" })[grid.stats.kind]}</b></div>
                <div style={row}><span>색 범위</span><b>{["anomaly", "rtp", "tilt", "vd1"].includes(grid.stats.kind) ? `±${fmt(grid.overlay.vmax, 0)}` : `0 ~ ${fmt(grid.overlay.vmax, 0)}`}</b></div>
                <Legend overlay={grid.overlay} unit={({ anomaly: "nT", rtp: "nT", thd: "nT/m", asa: "nT/m", tilt: "°", vd1: "nT/m" })[grid.stats.kind]} />
                <div style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 4, fontSize: 12, flexWrap: "wrap" }}>
                  <label><input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} /> 격자</label>
                  <label>불투명도 <input type="range" min="0.2" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} /></label>
                  <button style={ghost} disabled={gridBusy} onClick={() => exportGrid("grd")}>GRD</button>
                  <button style={ghost} disabled={gridBusy} onClick={() => exportGrid("tif")}>GeoTIFF</button>
                </div>
              </div>
            )}

            {files.length >= 2 && (
              <>
                <h2 style={h2}>5. 반복 측정 비교</h2>
                <div style={note}>
                  파일 쌍마다 같은 셀 안에 두 측정이 모두 넣은 값의 평균을 비교합니다(보간 없음). 표적은 반복되고 배경은 반복되지 않으면 플랫폼 잡음입니다. 셀이 클수록 잡음이 평균되어 상관이 오르므로, 셀 크기는 결과의 일부입니다.
                </div>
                <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 6 }}>
                  <label>비교 셀(m) <input type="number" step="0.05" min="0.05" value={compareCell} onChange={(e) => setCompareCell(e.target.value)} style={inputW(60)} /></label>
                  <button style={button} disabled={repeatBusy} onClick={runRepeat}>{repeatBusy ? "계산 중..." : "비교 실행"}</button>
                </div>
                {repeat && (
                  <div style={{ marginTop: 6 }}>
                    {repeat.pairs.map((p, i) => (
                      <div key={i} style={{ ...box, marginTop: 4, padding: 6 }}>
                        <div style={note}>[{p.file_a}] ↔ [{p.file_b}] · 공통 {p.n_cells.toLocaleString()}셀 ({fmt(p.common_area_m2, 0)} m², 셀 {repeat.compare_cell_m} m)</div>
                        <div style={row}><span>차이 rms / 평균</span><b>{fmt(p.diff_rms_nt)} / {fmt(p.diff_mean_nt)} nT</b></div>
                        <div style={row}><span>신호 rms (A / B) / 상관</span><b>{fmt(p.signal_rms_a_nt)} / {fmt(p.signal_rms_b_nt)} nT / {fmt(p.correlation, 2)}</b></div>
                      </div>
                    ))}
                    {repeat.note && <div style={warn}>• {repeat.note}</div>}
                  </div>
                )}
              </>
            )}

            <h2 style={h2}>6. 표적 피킹</h2>
            <div style={note}>
              논문 방식(기본): 측정마다 잔차 격자의 THD(또는 ASA)에서 국소 봉우리를 찾아 prominence ≥ K × MAD인 것만 남기고, 봉우리 주변 원 측점에 유도 쌍극자를 맞춰 깊이·모멘트를 더합니다.
              논문의 기준값(탐침 0.5 m): 대전차지뢰 K ≈ 65~71, 대인지뢰 K ≈ 19~20 — K는 현장마다 보정합니다. 피크-투-피크 진폭으로 유형(대전차/대인/소형)을 가늠합니다.
              측정이 여럿이면 {targets ? targets.match_radius_m : 0.5} m 안에서 맞춰 "그 자리를 덮은 측정 n회 중 k회에서 보임"을 셉니다(드론만 간 곳은 드론만 분모). 등급 A = 덮은 측정 모두에서 보이고 K ≥ 20, B = K ≥ 10, C = 나머지. ⚠는 덮임 가장자리의 봉우리.
            </div>
            <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", marginTop: 6 }}>
              <select value={tMethod} onChange={(e) => setTMethod(e.target.value)}>
                <option value="thd">파생 격자 피크 (K × MAD)</option>
                <option value="amplitude">진폭 문턱 (k × 잡음)</option>
              </select>
              {tMethod === "thd" ? (
                <>
                  <select value={derivative} onChange={(e) => setDerivative(e.target.value)}>
                    <option value="thd">THD</option>
                    <option value="asa">ASA</option>
                    <option value="rtp">RTP</option>
                  </select>
                  <label>K <input type="number" step="1" min="1" value={peakK} onChange={(e) => setPeakK(e.target.value)} style={inputW(50)} /></label>
                </>
              ) : (
                <label>문턱 k <input type="number" step="0.5" min="1" value={thresholdK} onChange={(e) => setThresholdK(e.target.value)} style={inputW(50)} /></label>
              )}
              <label>탐침 지상고(m) <input type="number" step="0.1" min="0" value={sensorHeight} onChange={(e) => setSensorHeight(e.target.value)} style={inputW(55)} /></label>
              {files.length > 1 && files.map((f, i) => (
                <label key={i} title={f.filename}>[{i}] <input type="number" step="0.1" min="0" placeholder={sensorHeight} value={heightsByFile[i] ?? ""} onChange={(e) => setHeightsByFile({ ...heightsByFile, [i]: e.target.value })} style={inputW(45)} /></label>
              ))}
              <label>표시 개수(큰 것부터) <input type="number" step="1" min="1" placeholder="전부" value={maxTargets} onChange={(e) => setMaxTargets(e.target.value)} style={inputW(55)} /></label>
              <button style={button} disabled={targetsBusy} onClick={runTargets}>{targetsBusy ? "계산 중..." : "표적 피킹"}</button>
            </div>
            {targets && (
              <div style={{ marginTop: 6 }}>
                <div style={row}><span>표적 (A / B / C){targets.max_targets ? ` · 상위 ${targets.max_targets}개 / 전체 ${targets.n_before_cut}` : ""}</span><b>{targets.n_targets} ({targets.n_by_grade.A} / {targets.n_by_grade.B} / {targets.n_by_grade.C})</b></div>
                <div style={row}><span>자료 / 복각 / 편각</span><b>{targets.processed ? "전처리됨" : "원 자료"} / {fmt(targets.inclination_deg)}° / {fmt(targets.declination_deg)}°</b></div>
                {Object.entries(targets.runs).map(([k, v]) => (
                  <div key={k} style={row}>
                    <span>[{k}] {v.method === "amplitude" ? "잡음 rms / 문턱" : `${v.method.toUpperCase()} MAD / 문턱(K=${v.peak_k})`} / 피킹</span>
                    <b>{v.method === "amplitude" ? `${fmt(v.noise_rms_nt)} / ${fmt(v.threshold_nt, 0)} nT` : `${fmt(v.mad, 2)} / ${fmt(v.prominence_threshold, 1)}`} / {v.n_picks}</b>
                  </div>
                ))}
                <div style={{ maxHeight: 280, overflow: "auto", marginTop: 6, border: "1px solid #d9e2d6", borderRadius: 6, background: "white" }}>
                  <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 11 }}>
                    <thead>
                      <tr style={{ background: "#eef3ea" }}>
                        {["#", "등급", "p2p(nT)", "K", "유형", "폭(m)", "매설(m)", "모멘트", "맞춤", "자화", "보임"].map((h) => <th key={h} style={{ padding: "3px 4px", textAlign: "right" }}>{h}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {targets.targets.map((t) => (
                        <tr key={t.id} style={{ borderTop: "1px solid #eef3ea" }}>
                          <td style={{ padding: "2px 4px", textAlign: "right" }}>{t.id}</td>
                          <td style={{ padding: "2px 4px", textAlign: "right", fontWeight: 700, color: gradeColor(t.grade) }}>{t.grade}</td>
                          <td style={{ padding: "2px 4px", textAlign: "right" }}>{fmt(t.p2p_nt ?? t.peak_nt, 0)}</td>
                          <td style={{ padding: "2px 4px", textAlign: "right" }}>{t.k_needed !== null && t.k_needed !== undefined ? fmt(t.k_needed, 0) : fmt(t.snr, 0)}</td>
                          <td style={{ padding: "2px 4px", textAlign: "right" }}>{t.type_class ? t.type_class.replace("지뢰급", "") : "-"}</td>
                          <td style={{ padding: "2px 4px", textAlign: "right" }}>{fmt(t.extent_m ?? t.footprint_m, 1)}</td>
                          <td style={{ padding: "2px 4px", textAlign: "right" }}>{fmt(t.burial_depth_m, 2)}</td>
                          <td style={{ padding: "2px 4px", textAlign: "right" }} title={t.size_class || ""}>{t.moment_am2 !== null ? fmt(t.moment_am2, 2) : "-"}</td>
                          <td style={{ padding: "2px 4px", textAlign: "right" }}>{fmt(t.fit_quality, 2)}</td>
                          <td style={{ padding: "2px 4px", textAlign: "right" }} title={t.magnetization ? `유도 방향과 ${fmt(t.magnetization.angle_from_induced_deg, 0)}°, 자유 방향의 개선도 ${fmt(t.magnetization.improvement, 2)}` : ""}>
                            {t.magnetization ? ({ induced: "유도", remanent: "잔류", unclear: "보류" })[t.magnetization.verdict] : "-"}
                          </td>
                          <td style={{ padding: "2px 4px", textAlign: "right" }} title={t.at_edge ? "덮임 가장자리의 봉우리: 이상이 잘렸을 수 있음" : ""}>{t.runs_seen.length}/{t.n_runs_covering ?? t.n_runs}{t.at_edge ? " ⚠" : ""}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div style={{ display: "flex", gap: 10, marginTop: 6, fontSize: 12 }}>
                  {targets.n_targets > 0 && <a href={api.targetsCsvUrl(projectId)}>CSV</a>}
                  {targets.n_targets > 0 && <a href={api.targetsGeojsonUrl(projectId)} download="minemag_targets.geojson">GeoJSON</a>}
                </div>
                <div style={{ ...note, marginTop: 4 }}>
                  유형은 탐침 0.5 m 기준 피크-투-피크 진폭의 참고 구분이고, 모멘트 크기는 자릿수 분류일 뿐입니다. "자화"는 자유 방향 쌍극자가 유도 방향 쌍극자보다 얼마나 더 잘 맞는지로 판정한 것으로,
                  강철 지뢰는 유도자화에 가깝고 잔류자화가 강한 물체(현무암, 소성된 철물)는 다르게 나옵니다 — 증거일 뿐 단독 판별 기준은 아닙니다. 자력만으로는 지뢰와 같은 금속량의 파편을 구분하지 못합니다.
                </div>
              </div>
            )}

            <h2 style={h2}>지점값 확인</h2>
            <div style={note}>지도를 클릭하면 그 자리의 격자 값(마지막에 만든 격자)과 가장 가까운 측점의 값을 보여 줍니다.</div>
            {sample && (
              <div style={{ ...box, marginTop: 6 }}>
                <div style={row}><span>위치</span><b>{fmt(sample.lat, 6)}, {fmt(sample.lon, 6)}</b></div>
                <div style={row}><span>격자 값{sample.grid_kind ? ` (${({ anomaly: "잔차 이상", rtp: "RTP", thd: "THD", asa: "ASA", tilt: "틸트각", vd1: "1VD" })[sample.grid_kind]})` : ""}</span><b>{sample.in_grid ? fmt(sample.grid_value, 2) : "격자 밖"}</b></div>
                {sample.nearest && (
                  <>
                    <div style={row}><span>가장 가까운 측점</span><b>{fmt(sample.nearest.distance_m, 2)} m · 파일 [{sample.nearest.file_index}] 트랙 {sample.nearest.track} 탐침 {sample.nearest.sensor}</b></div>
                    <div style={row}><span>측점 값{sample.nearest.tfa_raw_nt !== null ? " (전처리 후 / 전)" : ""}</span><b>{fmt(sample.nearest.tfa_nt, 1)}{sample.nearest.tfa_raw_nt !== null ? ` / ${fmt(sample.nearest.tfa_raw_nt, 1)}` : ""} nT</b></div>
                    <button style={{ ...ghost, marginTop: 4 }} onClick={() => showProfile(sample.nearest.file_index, sample.nearest.track)}>이 측선 프로파일 보기</button>
                  </>
                )}
              </div>
            )}

            <h2 style={h2}>측선 프로파일</h2>
            <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
              <select value={profileFile} onChange={(e) => setProfileFile(Number(e.target.value))}>
                {files.map((f, i) => <option key={i} value={i}>[{i}] {f.filename.slice(0, 24)}</option>)}
              </select>
              <input type="number" placeholder="트랙 번호" value={profileLine} onChange={(e) => setProfileLine(e.target.value)} style={inputW(80)} />
              <button style={button} onClick={() => showProfile(profileFile, profileLine)}>보기</button>
              {profile && <button style={ghost} onClick={() => stepProfile(-1)}>◀ 이전</button>}
              {profile && <button style={ghost} onClick={() => stepProfile(1)}>다음 ▶</button>}
            </div>

            <h2 style={h2}>지도 표시</h2>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <select value={sensor} onChange={(e) => setSensor(Number(e.target.value))}>
                <option value={0}>두 탐침 모두</option>
                {summary.sensors.map((s) => <option key={s} value={s}>탐침 {s}</option>)}
              </select>
              <label style={{ fontSize: 12 }}><input type="checkbox" checked={showPoints} onChange={(e) => setShowPoints(e.target.checked)} /> 측점 표시</label>
              {pre && <label style={{ fontSize: 12 }}><input type="checkbox" checked={showProcessed} onChange={(e) => setShowProcessed(e.target.checked)} /> 전처리된 값</label>}
              {points && <span style={note}>{points.lat.length.toLocaleString()}점 (1/{points.stride}){points.processed ? " · 전처리" : ""}</span>}
            </div>
          </>
        )}
      </div>
      <div style={{ flex: 1, position: "relative" }}>
        <Profile data={profile} onClose={() => setProfile(null)} />
        <MapContainer center={[36.3789, 127.361]} zoom={18} maxZoom={24} preferCanvas style={{ height: "100%" }}>
          <ClickInspector onClick={mapClick} />
          {area && area.polygon && <Polygon positions={area.polygon} pathOptions={{ color: "#2f5a22", weight: 2, fillOpacity: 0.04, dashArray: "6 4" }} />}
          {drawPts.length > 0 && (drawMode === "polygon"
            ? <Polyline positions={drawPts} pathOptions={{ color: "#b42318", weight: 2, dashArray: "4 3" }} />
            : <CircleMarker center={drawPts[0]} radius={5} pathOptions={{ color: "#b42318" }} />)}
          {sample && <CircleMarker center={[sample.lat, sample.lon]} radius={6} weight={2} pathOptions={{ color: "#111", fillOpacity: 0 }} />}
          <BoundsWatcher onChange={setMapBounds} />
          {BASEMAPS[basemap].url && <TileLayer key={basemap} url={BASEMAPS[basemap].url} maxNativeZoom={BASEMAPS[basemap].maxNative} maxZoom={24} attribution={BASEMAPS[basemap].attribution} />}
          {tileLayers.map((l) => (
            <TileLayer key={l.id} url={api.localTileUrl(l.id)} minNativeZoom={l.min_zoom} maxNativeZoom={l.max_zoom} maxZoom={24} bounds={l.bounds} />
          ))}
          {basemaps.map((b) => (
            <ImageOverlay key={b.name} url={b.image_data_url} bounds={b.bounds} opacity={basemapOpacity} />
          ))}
          {points && <FitBounds lat={points.lat} lon={points.lon} />}
          {grid && showGrid && <GridOverlay overlay={grid.overlay} opacity={opacity} />}
          {points && showPoints && points.lat.map((la, i) => (
            <CircleMarker key={i} center={[la, points.lon[i]]} radius={grid && showGrid ? 1 : 2} stroke={false} fillOpacity={0.9}
              pathOptions={{ fillColor: (points.in_area && !points.in_area[i]) || (points.line_id && points.line_id[i] < 0) ? "#bbb" : grid && showGrid ? "#333" : color(points.tfa[i], vmax) }} />
          ))}
          {targets && targets.targets.map((t) => (
            <CircleMarker key={`t${t.id}`} center={[t.lat, t.lon]} radius={9} weight={2}
              pathOptions={{ color: gradeColor(t.grade), fillOpacity: 0.1 }}>
              <Tooltip permanent direction="right" offset={[8, 0]}>{t.id}{t.grade}</Tooltip>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </div>
  );
}
