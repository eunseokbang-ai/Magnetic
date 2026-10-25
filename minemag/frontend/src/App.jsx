import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, useMap } from "react-leaflet";
import * as api from "./api";

const panel = { width: 360, padding: 14, borderRight: "1px solid #d9e2d6", overflowY: "auto", background: "#f4f7f2", fontSize: 13 };
const h2 = { fontSize: 13, margin: "14px 0 6px" };
const row = { display: "flex", justifyContent: "space-between", padding: "2px 0", borderBottom: "1px dotted #d9e2d6" };

const fmt = (v, d = 1) => (v === null || v === undefined ? "-" : Number(v).toFixed(d));

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

export default function App() {
  const [version, setVersion] = useState(null);
  const [projectId, setProjectId] = useState(null);
  const [summary, setSummary] = useState(null);
  const [points, setPoints] = useState(null);
  const [sensor, setSensor] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getVersion().then(setVersion).catch(() => {});
    // ?project=<id> reopens a project this server already holds (another
    // tab, or files pushed through the API).
    const existing = new URLSearchParams(window.location.search).get("project");
    if (existing) {
      setProjectId(existing);
      api.getSummary(existing).then(setSummary).catch((e) => setError(e.message));
    }
  }, []);

  const upload = async (files) => {
    if (!files.length) return;
    setBusy(true);
    setError(null);
    try {
      const pid = projectId || (await api.createProject()).project_id;
      setProjectId(pid);
      const s = await api.uploadFiles(pid, files);
      setSummary(s);
      setPoints(await api.getPoints(pid, sensor || undefined));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (!projectId || !summary) return;
    api.getPoints(projectId, sensor || undefined).then(setPoints).catch((e) => setError(e.message));
  }, [sensor, projectId, summary]);

  const vmax = useMemo(() => {
    if (!points) return 200;
    const a = points.tfa.map(Math.abs).sort((x, y) => x - y);
    return Math.max(50, a[Math.floor(a.length * 0.98)] || 200);
  }, [points]);

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <div style={panel}>
        <div style={{ fontSize: 18, fontWeight: 700 }}>MineMag</div>
        <div style={{ color: "#6b7a66", fontSize: 11 }}>
          드론 자력탐사 지뢰·불발탄 탐지 {version?.commit && `· ${version.commit}`}
        </div>

        <h2 style={h2}>1. MagDrone R3 자료 올리기 (.asc)</h2>
        <input type="file" multiple accept=".asc,.txt,.csv" disabled={busy} onChange={(e) => upload(Array.from(e.target.files))} />
        <div style={{ color: "#6b7a66", fontSize: 11, marginTop: 4 }}>
          여러 비행·반복 측정 파일을 함께 선택하면 시간순으로 합쳐집니다. 두 탐침, 트랙, 3축 성분, 고도를 모두 읽습니다.
        </div>
        {busy && <div style={{ marginTop: 6 }}>읽는 중...</div>}
        {error && <div style={{ marginTop: 6, color: "#b42318" }}>{error}</div>}

        {summary && (
          <>
            <h2 style={h2}>적재 요약</h2>
            <div style={row}><span>파일 / 행</span><b>{summary.n_files} / {summary.n_rows.toLocaleString()}</b></div>
            <div style={row}><span>탐침</span><b>{summary.sensors.join(", ")} (간격 {Object.values(summary.sensor_separation_m).map((v) => fmt(v, 2)).join(", ")} m)</b></div>
            <div style={row}><span>샘플 간격</span><b>{fmt(summary.sample_interval_ms, 0)} ms</b></div>
            <div style={row}><span>트랙 / 시간</span><b>{summary.n_tracks} / {fmt(summary.duration_min)} 분</b></div>
            <div style={row}><span>공백(&gt;1 s)</span><b>{summary.n_gaps}회, 최대 {fmt(summary.longest_gap_s)} s</b></div>
            <div style={row}><span>속도</span><b>{fmt(summary.speed_m_s?.median, 2)} m/s</b></div>
            <div style={row}><span>GPS 고도</span><b>{fmt(summary.altitude_m?.min)} ~ {fmt(summary.altitude_m?.max)} m</b></div>
            <div style={row}><span>총자기장 이상</span><b>{fmt(summary.tfa_nt?.min, 0)} ~ {fmt(summary.tfa_nt?.max, 0)} nT (σ {fmt(summary.tfa_nt?.std, 0)})</b></div>
            <div style={row}><span>3축 성분</span><b>{summary.has_vector ? "있음" : "없음"}</b></div>
            {Object.entries(summary.per_sensor).map(([k, v]) => (
              <div key={k} style={row}>
                <span>탐침 {k} 잡음(4차 차분)</span>
                <b>{fmt(v.noise_4th_diff_nt, 2)} nT{summary.sensor_offset_nt[k] !== undefined && ` · 오프셋 ${fmt(summary.sensor_offset_nt[k], 1)} nT`}</b>
              </div>
            ))}
            <h2 style={h2}>파일</h2>
            {summary.files.map((f, i) => (
              <div key={i} style={{ fontSize: 11, padding: "2px 0" }}>
                {f.filename} — {f.start_date || "날짜 미상"} · {f.n_rows.toLocaleString()}행 · {f.settings.join(" ")}
              </div>
            ))}
            <h2 style={h2}>지도 표시</h2>
            <select value={sensor} onChange={(e) => setSensor(Number(e.target.value))}>
              <option value={0}>두 탐침 모두</option>
              {summary.sensors.map((s) => <option key={s} value={s}>탐침 {s}</option>)}
            </select>
            {points && <span style={{ marginLeft: 8, fontSize: 11, color: "#6b7a66" }}>{points.lat.length.toLocaleString()}점 표시 (1/{points.stride})</span>}
          </>
        )}
      </div>
      <div style={{ flex: 1 }}>
        <MapContainer center={[36.3789, 127.361]} zoom={18} maxZoom={23} preferCanvas style={{ height: "100%" }}>
          <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" maxNativeZoom={19} maxZoom={23} attribution="&copy; OpenStreetMap" />
          {points && <FitBounds lat={points.lat} lon={points.lon} />}
          {points && points.lat.map((la, i) => (
            <CircleMarker key={i} center={[la, points.lon[i]]} radius={2} stroke={false} fillOpacity={0.9} pathOptions={{ fillColor: color(points.tfa[i], vmax) }} />
          ))}
        </MapContainer>
      </div>
    </div>
  );
}
