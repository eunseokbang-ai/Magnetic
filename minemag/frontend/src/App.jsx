import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet-imageoverlay-rotated";
import * as api from "./api";

const panel = { width: 400, padding: 14, borderRight: "1px solid #d9e2d6", overflowY: "auto", background: "#f4f7f2", fontSize: 13 };
const h2 = { fontSize: 13, margin: "16px 0 6px", color: "#2f3e2a" };
const row = { display: "flex", justifyContent: "space-between", padding: "2px 0", borderBottom: "1px dotted #d9e2d6" };
const button = { padding: "6px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #4f7a3a", background: "white", color: "#2f5a22", cursor: "pointer" };
const note = { color: "#6b7a66", fontSize: 11, lineHeight: 1.45 };
const warn = { color: "#8a4b08", fontSize: 12, lineHeight: 1.45 };

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

/** A small inline amplitude spectrum: log amplitude vs frequency. */
function Spectrum({ spectrum, gaitBand }) {
  if (!spectrum || !spectrum.hz.length) return <div style={note}>조용한 구간이 짧아 스펙트럼을 내지 못했습니다.</div>;
  const W = 360, H = 110, padL = 34, padB = 18;
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
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const [qc, setQc] = useState(null);
  const [qcBusy, setQcBusy] = useState(false);

  const [cell, setCell] = useState(0.1);
  const [maxDist, setMaxDist] = useState(0.6);
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

  const resetDerived = () => {
    setQc(null);
    setGrid(null);
    setRepeat(null);
  };

  const upload = async (files) => {
    if (!files.length) return;
    setBusy(true);
    setError(null);
    try {
      const pid = projectId || (await api.createProject()).project_id;
      setProjectId(pid);
      const s = await api.uploadFiles(pid, files);
      resetDerived();
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

  const runQc = async () => {
    setQcBusy(true);
    setError(null);
    try {
      setQc(await api.runQc(projectId, {}));
    } catch (e) {
      setError(e.message);
    } finally {
      setQcBusy(false);
    }
  };

  const gridReq = () => ({
    cell_size_m: Number(cell),
    max_distance_m: Number(maxDist),
    file_index: gridFile === "" ? null : Number(gridFile),
    sensors: gridSensors === "" ? null : [Number(gridSensors)],
  });

  const runGrid = async () => {
    setGridBusy(true);
    setError(null);
    try {
      setGrid(await api.makeGrid(projectId, gridReq()));
      setShowGrid(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setGridBusy(false);
    }
  };

  const runRepeat = async () => {
    setRepeatBusy(true);
    setError(null);
    try {
      setRepeat(await api.compareRepeats(projectId, { compare_cell_m: Number(compareCell) }));
    } catch (e) {
      setError(e.message);
    } finally {
      setRepeatBusy(false);
    }
  };

  const vmax = useMemo(() => {
    if (!points) return 200;
    const a = points.tfa.map(Math.abs).sort((x, y) => x - y);
    return Math.max(50, a[Math.floor(a.length * 0.98)] || 200);
  }, [points]);

  const files = summary?.files || [];

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <div style={panel}>
        <div style={{ fontSize: 18, fontWeight: 700 }}>MineMag</div>
        <div style={note}>드론 자력탐사 지뢰·불발탄 탐지 {version?.commit && `· ${version.commit}`}</div>

        <h2 style={h2}>1. MagDrone R3 자료 올리기 (.asc)</h2>
        <input type="file" multiple accept=".asc,.txt,.csv" disabled={busy} onChange={(e) => upload(Array.from(e.target.files))} />
        <div style={{ ...note, marginTop: 4 }}>
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
            {Object.entries(summary.per_sensor).map(([k, v]) => (
              <div key={k} style={row}>
                <span>탐침 {k} 잡음(4차 차분)</span>
                <b>{fmt(v.noise_4th_diff_nt, 2)} nT{summary.sensor_offset_nt[k] !== undefined && ` · 오프셋 ${fmt(summary.sensor_offset_nt[k], 1)} nT`}</b>
              </div>
            ))}
            {files.map((f, i) => (
              <div key={i} style={{ ...note, padding: "2px 0" }}>
                [{i}] {f.filename} — {f.start_date || "날짜 미상"} · {f.n_rows.toLocaleString()}행 · {f.settings.join(" ")}
              </div>
            ))}

            <h2 style={h2}>2. 플랫폼 잡음 QC</h2>
            <div style={note}>
              표적이 없는 조용한 구간에서 배경 잡음, 두 탐침의 상관, 잡음 스펙트럼을 잽니다. 보행·회전익 주기가 보이면 그 공간 주기를,
              두 탐침 상관이 낮으면 구배로 지울 수 없다는 판정을 냅니다.
            </div>
            <button style={{ ...button, marginTop: 6 }} disabled={qcBusy} onClick={runQc}>{qcBusy ? "계산 중..." : "QC 실행"}</button>
            {qc && qc.files.map((f) => (
              <div key={f.file_index} style={{ marginTop: 8, padding: 8, background: "white", border: "1px solid #d9e2d6", borderRadius: 6 }}>
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
                      <div style={row}>
                        <span>두 탐침 상관 / 차의 rms</span>
                        <b>{fmt(f.gradient.probe_correlation_above_1hz, 2)} / {fmt(f.gradient.difference_rms_nt)} nT ({fmt(f.gradient.cancellation_pct, 0)}% 감소)</b>
                      </div>
                    )}
                    <div style={{ marginTop: 6 }}><Spectrum spectrum={f.spectrum} gaitBand={qc.gait_band_hz} /></div>
                    {f.verdict.map((line, i) => <div key={i} style={{ ...warn, marginTop: 4 }}>• {line}</div>)}
                  </>
                )}
              </div>
            ))}

            <h2 style={h2}>3. 격자</h2>
            <div style={note}>
              블록평균 뒤 선형 보간, 셀 0.1 m 기본. 가장 가까운 측점에서 설정 거리보다 먼 셀은 비워 두어(덮임 공백) 측정하지 않은 곳을 그대로 보여 줍니다.
              탐침·파일별 중앙값을 빼서 탐침 오프셋이 격자의 단차로 남지 않게 합니다.
            </div>
            <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", marginTop: 6 }}>
              <label>셀(m) <input type="number" step="0.05" min="0.02" value={cell} onChange={(e) => setCell(e.target.value)} style={{ width: 60 }} /></label>
              <label>최대 거리(m) <input type="number" step="0.1" min="0.1" value={maxDist} onChange={(e) => setMaxDist(e.target.value)} style={{ width: 60 }} /></label>
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
                <div style={row}><span>덮임 / 셀 수</span><b>{fmt(grid.stats.coverage_pct)}% / {grid.stats.n_cells.toLocaleString()}</b></div>
                <div style={row}><span>값 범위 / rms</span><b>{fmt(grid.stats.min_nt, 0)} ~ {fmt(grid.stats.max_nt, 0)} nT / {fmt(grid.stats.rms_nt)} nT</b></div>
                <div style={row}><span>색 범위</span><b>±{fmt(grid.overlay.vmax, 0)} nT</b></div>
                <div style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 4, fontSize: 12 }}>
                  <label><input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} /> 격자</label>
                  <label><input type="checkbox" checked={showPoints} onChange={(e) => setShowPoints(e.target.checked)} /> 측점</label>
                  <label>불투명도 <input type="range" min="0.2" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} /></label>
                </div>
              </div>
            )}

            {files.length >= 2 && (
              <>
                <h2 style={h2}>4. 반복 측정 비교</h2>
                <div style={note}>
                  파일 쌍마다 같은 셀 안에 두 측정이 모두 넣은 값의 평균을 비교합니다(보간 없음 — 보간 격자로 비교하면 좌표계만 바꿔도 상관이 0.25~0.53으로 흔들렸습니다).
                  표적은 반복되고 배경은 반복되지 않으면 플랫폼 잡음입니다. 셀이 클수록 잡음이 평균되어 상관이 오르므로, 셀 크기는 결과의 일부입니다.
                </div>
                <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 6 }}>
                  <label>비교 셀(m) <input type="number" step="0.05" min="0.05" value={compareCell} onChange={(e) => setCompareCell(e.target.value)} style={{ width: 60 }} /></label>
                  <button style={button} disabled={repeatBusy} onClick={runRepeat}>{repeatBusy ? "계산 중..." : "비교 실행"}</button>
                </div>
                {repeat && (
                  <div style={{ marginTop: 6 }}>
                    {repeat.pairs.map((p, i) => (
                      <div key={i} style={{ padding: 6, background: "white", border: "1px solid #d9e2d6", borderRadius: 6, marginBottom: 4 }}>
                        <div style={{ fontSize: 11, color: "#6b7a66" }}>[{p.file_a}] ↔ [{p.file_b}] · 공통 {p.n_cells.toLocaleString()}셀 ({fmt(p.common_area_m2, 0)} m², 셀 {repeat.compare_cell_m} m)</div>
                        <div style={row}><span>차이 rms / 평균</span><b>{fmt(p.diff_rms_nt)} / {fmt(p.diff_mean_nt)} nT</b></div>
                        <div style={row}><span>신호 rms (A / B)</span><b>{fmt(p.signal_rms_a_nt)} / {fmt(p.signal_rms_b_nt)} nT</b></div>
                        <div style={row}><span>상관</span><b>{fmt(p.correlation, 2)}</b></div>
                      </div>
                    ))}
                    {repeat.note && <div style={warn}>• {repeat.note}</div>}
                  </div>
                )}
              </>
            )}

            <h2 style={h2}>지도 표시</h2>
            <select value={sensor} onChange={(e) => setSensor(Number(e.target.value))}>
              <option value={0}>두 탐침 모두</option>
              {summary.sensors.map((s) => <option key={s} value={s}>탐침 {s}</option>)}
            </select>
            {points && <span style={{ marginLeft: 8, ...note }}>{points.lat.length.toLocaleString()}점 표시 (1/{points.stride})</span>}
          </>
        )}
      </div>
      <div style={{ flex: 1 }}>
        <MapContainer center={[36.3789, 127.361]} zoom={18} maxZoom={24} preferCanvas style={{ height: "100%" }}>
          <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" maxNativeZoom={19} maxZoom={24} attribution="&copy; OpenStreetMap" />
          {points && <FitBounds lat={points.lat} lon={points.lon} />}
          {grid && showGrid && <GridOverlay overlay={grid.overlay} opacity={opacity} />}
          {points && showPoints && points.lat.map((la, i) => (
            <CircleMarker key={i} center={[la, points.lon[i]]} radius={grid && showGrid ? 1 : 2} stroke={false} fillOpacity={0.9} pathOptions={{ fillColor: grid && showGrid ? "#333" : color(points.tfa[i], vmax) }} />
          ))}
        </MapContainer>
      </div>
    </div>
  );
}
