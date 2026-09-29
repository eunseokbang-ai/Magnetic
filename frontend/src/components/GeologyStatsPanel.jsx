const sectionStyle = { border: "1px solid #e6dac0", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #ddd0b2" };
const buttonStyle = {
  padding: "7px 10px",
  fontSize: 12,
  fontWeight: 600,
  borderRadius: 6,
  border: "1px solid #a16207",
  background: "#a16207",
  color: "white",
  cursor: "pointer",
};
const smallButton = { ...inputStyle, cursor: "pointer", width: "auto" };

/**
 * The readings joined to the national geology map, one row per rock
 * unit the survey crossed. This is what an interpretation report states
 * about the geology - and it is computed on the readings, not the grid,
 * because the grid between lines is the interpolator's invention.
 */
export default function GeologyStatsPanel({
  ready,
  params,
  setParams,
  onRun,
  running,
  result,
  error,
  showOnMap,
  setShowOnMap,
  onExportCsv,
}) {
  const update = (key, val) => setParams((p) => ({ ...p, [key]: val }));
  const units = result?.units || [];
  const report = result?.report || {};

  return (
    <div style={sectionStyle}>
      <div style={bodyStyle}>
        <div style={{ color: "#8a7a5c" }}>
          KIGAM 지질도의 암상 폴리곤을 받아 <b>각 측점이 어느 지질 단위 위에 있는지</b> 판정하고, 단위별로 자기이상의
          평균·산포와 해석신호를 표로 냅니다. 격자가 아니라 <b>실제 측점</b>으로 계산하므로 측선 사이의 보간값이
          섞이지 않습니다. 인터넷이 필요하며, 한 번 받은 폴리곤은 저장되어 다음부터는 바로 계산됩니다.
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
          <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <span style={{ color: "#6b5c42" }}>지질도 축척</span>
            <select style={inputStyle} value={params.scale} onChange={(e) => update("scale", e.target.value)}>
              <option value="50k">1:5만 (상세, 일부 섬 미포함)</option>
              <option value="250k">1:25만</option>
              <option value="1m">1:100만</option>
            </select>
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <span style={{ color: "#6b5c42" }}>집계 값</span>
            <select style={inputStyle} value={params.value} onChange={(e) => update("value", e.target.value)}>
              <option value="anomaly">자기이상</option>
              <option value="tmi">TMI</option>
            </select>
          </label>
        </div>
        <button style={{ ...buttonStyle, opacity: ready && !running ? 1 : 0.5 }} disabled={!ready || running} onClick={onRun}>
          {running ? "지질도 받는 중..." : "지질 단위별 통계 계산"}
        </button>
        {error && <div style={{ color: "#b91c1c" }}>{error}</div>}

        {result && (
          <>
            <div style={{ color: "#4a3d28" }}>
              폴리곤 {report.n_polygons}개 · 지질 단위 {report.n_units}개 · 측점 {report.n_points?.toLocaleString()}개
              {report.n_outside ? ` (지질도 밖 ${report.n_outside.toLocaleString()}개)` : ""}
              {report.sheets?.length ? ` · 도폭: ${report.sheets.join(", ")}` : ""}
            </div>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={showOnMap} onChange={(e) => setShowOnMap(e.target.checked)} />
                <span>지도에 지질 단위 경계 표시</span>
              </label>
              <button style={{ ...smallButton, marginLeft: "auto" }} onClick={onExportCsv}>
                표 내보내기 (CSV)
              </button>
            </div>
            <div style={{ maxHeight: 260, overflow: "auto", border: "1px solid #eee4cd", borderRadius: 6 }}>
              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 11 }}>
                <thead>
                  <tr style={{ background: "#faf6ec", position: "sticky", top: 0 }}>
                    {["기호", "암상", "측점", "평균", "표준편차", "P10~P90", "AS"].map((h) => (
                      <th key={h} style={{ textAlign: "left", padding: "3px 5px", borderBottom: "1px solid #e6dac0" }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {units.map((u) => (
                    <tr key={u.symbol} title={`${u.age || ""} · 측선 ${u.n_lines}개 · 측점 비율 ${u.share_pct}%`}>
                      <td style={{ padding: "3px 5px", fontWeight: 700 }}>{u.symbol}</td>
                      <td style={{ padding: "3px 5px" }}>{u.name}</td>
                      <td style={{ padding: "3px 5px" }}>{u.n_points.toLocaleString()}</td>
                      <td style={{ padding: "3px 5px" }}>{u.mean_nt.toFixed(1)}</td>
                      <td style={{ padding: "3px 5px" }}>{u.std_nt.toFixed(1)}</td>
                      <td style={{ padding: "3px 5px" }}>
                        {u.p10_nt.toFixed(0)}~{u.p90_nt.toFixed(0)}
                      </td>
                      <td style={{ padding: "3px 5px" }}>{u.mean_signal != null ? u.mean_signal.toFixed(3) : "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ color: "#8a7a5c", fontSize: 11 }}>
              평균·표준편차·P10~P90은 {params.value === "tmi" ? "TMI" : "자기이상"}(nT), AS는 해석신호(nT/m) 평균입니다.
              표준편차가 큰 단위는 그 안에 자성 차이가 있는 암상이 섞여 있거나 구조물 이상이 남아 있다는 뜻입니다.
              {result.attribution ? ` ${result.attribution}.` : ""}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
