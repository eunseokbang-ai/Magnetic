import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } from "react-leaflet";
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
  const [maxDist, setMaxDist] = useState(0.6);
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

  useEffect(() => {
    api.getVersion().then(setVersion).catch(() => {});
    const existing = new URLSearchParams(window.location.search).get("project");
    if (existing) {
      setProjectId(existing);
      api.getSummary(existing).then(setSummary).catch((e) => setError(e.message));
    }
  }, []);

  const resetDerived = () => {
    setQc(null);
    setPre(null);
    setGrid(null);
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
    max_distance_m: Number(maxDist),
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
  const runTargets = () => guard(setTargetsBusy, async () => {
    const byFile = {};
    for (const [k, v] of Object.entries(heightsByFile)) if (v !== "" && Number.isFinite(Number(v))) byFile[k] = Number(v);
    setTargets(await api.pickTargets(projectId, {
      method: tMethod, derivative, peak_k: Number(peakK), threshold_k: Number(thresholdK),
      sensor_height_m: Number(sensorHeight), sensor_heights_by_file: byFile,
      cell_size_m: Number(cell), grid_method: gridMethod, detrend_order: Number(detrend),
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

        <h2 style={h2}>1. MagDrone R3 자료 올리기 (.asc)</h2>
        <input type="file" multiple accept=".asc,.txt,.csv" disabled={busy} onChange={(e) => upload(Array.from(e.target.files))} />
        <div style={{ ...note, marginTop: 4 }}>
          MagDrone DataTool이 내보낸 ASC. 여러 비행·반복 측정 파일을 함께 선택하면 시간순으로 합쳐집니다. 두 탐침, 트랙, 3축 성분, 고도를 모두 읽습니다.
        </div>
        {busy && <div style={{ marginTop: 6 }}>읽는 중...</div>}
        {error && <div style={{ marginTop: 6, color: "#b42318" }}>{error}</div>}

        {summary && (
          <>
            <h2 style={h2}>적재 요약</h2>
            <div style={row}><span>파일 / 행</span><b>{summary.n_files} / {summary.n_rows.toLocaleString()}</b></div>
            <div style={row}><span>탐침</span><b>{summary.sensors.join(", ")} (간격 {Object.values(summary.sensor_separation_m).map((v) => fmt(v, 2)).join(", ")} m)</b></div>
            <div style={row}><span>샘플 간격 / 트랙 / 시간</span><b>{fmt(summary.sample_interval_ms, 0)} ms / {summary.n_tracks} / {fmt(summary.duration_min)} 분</b></div>
            <div style={row}><span>속도 / GPS 고도</span><b>{fmt(summary.speed_m_s?.median, 2)} m/s / {fmt(summary.altitude_m?.min)} ~ {fmt(summary.altitude_m?.max)} m</b></div>
            <div style={row}><span>총자기장 이상</span><b>{fmt(summary.tfa_nt?.min, 0)} ~ {fmt(summary.tfa_nt?.max, 0)} nT (σ {fmt(summary.tfa_nt?.std, 0)})</b></div>
            {files.map((f, i) => (
              <div key={i} style={{ ...note, padding: "2px 0" }}>
                [{i}] {f.filename} — {f.start_date || "날짜 미상"} · {f.n_rows.toLocaleString()}행 · {f.settings.join(" ")}
              </div>
            ))}

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
                  <b>{f.heading_correction.applied ? `적용 ${fmt(f.heading_correction.offset_nt)} nT` : `미적용${f.heading_correction.offset_nt !== null && f.heading_correction.offset_nt !== undefined ? ` (측정 ${fmt(f.heading_correction.offset_nt)} ± ${fmt(f.heading_correction.offset_spread_nt)} nT)` : ""}`}</b>
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
              <label>최대 거리(m) <input type="number" step="0.1" min="0.1" value={maxDist} onChange={(e) => setMaxDist(e.target.value)} style={inputW(55)} /></label>
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
                <option value="thd">THD</option>
                <option value="asa">ASA</option>
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
                <div style={row}><span>{grid.stats.kind === "anomaly" ? "이상 범위 / rms" : `${grid.stats.kind.toUpperCase()} 범위 / rms`}</span><b>{fmt(grid.stats.min, 1)} ~ {fmt(grid.stats.max, 1)} / {fmt(grid.stats.rms, 1)}{grid.stats.kind === "anomaly" ? " nT" : " nT/m"}</b></div>
                <div style={row}><span>색 범위</span><b>{grid.stats.kind === "anomaly" ? `±${fmt(grid.overlay.vmax, 0)} nT` : `0 ~ ${fmt(grid.overlay.vmax, 0)} nT/m`}</b></div>
                <div style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 4, fontSize: 12, flexWrap: "wrap" }}>
                  <label><input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} /> 격자</label>
                  <label><input type="checkbox" checked={showPoints} onChange={(e) => setShowPoints(e.target.checked)} /> 측점</label>
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
              <button style={button} disabled={targetsBusy} onClick={runTargets}>{targetsBusy ? "계산 중..." : "표적 피킹"}</button>
            </div>
            {targets && (
              <div style={{ marginTop: 6 }}>
                <div style={row}><span>표적 (A / B / C)</span><b>{targets.n_targets} ({targets.n_by_grade.A} / {targets.n_by_grade.B} / {targets.n_by_grade.C})</b></div>
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
                        {["#", "등급", "p2p(nT)", "K", "유형", "폭(m)", "매설(m)", "모멘트", "맞춤", "보임"].map((h) => <th key={h} style={{ padding: "3px 4px", textAlign: "right" }}>{h}</th>)}
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
                  유형은 탐침 0.5 m 기준 피크-투-피크 진폭의 참고 구분이고, 모멘트 크기는 자릿수 분류일 뿐입니다. 자력만으로는 지뢰와 같은 금속량의 파편을 구분하지 못하며, 금속이 거의 없는 지뢰는 이상이 없을 수 있습니다.
                </div>
              </div>
            )}

            <h2 style={h2}>지도 표시</h2>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <select value={sensor} onChange={(e) => setSensor(Number(e.target.value))}>
                <option value={0}>두 탐침 모두</option>
                {summary.sensors.map((s) => <option key={s} value={s}>탐침 {s}</option>)}
              </select>
              {pre && <label style={{ fontSize: 12 }}><input type="checkbox" checked={showProcessed} onChange={(e) => setShowProcessed(e.target.checked)} /> 전처리된 값</label>}
              {points && <span style={note}>{points.lat.length.toLocaleString()}점 (1/{points.stride}){points.processed ? " · 전처리" : ""}</span>}
            </div>
          </>
        )}
      </div>
      <div style={{ flex: 1 }}>
        <MapContainer center={[36.3789, 127.361]} zoom={18} maxZoom={24} preferCanvas style={{ height: "100%" }}>
          <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" maxNativeZoom={19} maxZoom={24} attribution="&copy; OpenStreetMap" />
          {points && <FitBounds lat={points.lat} lon={points.lon} />}
          {grid && showGrid && <GridOverlay overlay={grid.overlay} opacity={opacity} />}
          {points && showPoints && points.lat.map((la, i) => (
            <CircleMarker key={i} center={[la, points.lon[i]]} radius={grid && showGrid ? 1 : 2} stroke={false} fillOpacity={0.9}
              pathOptions={{ fillColor: points.line_id && points.line_id[i] < 0 ? "#aaa" : grid && showGrid ? "#333" : color(points.tfa[i], vmax) }} />
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
