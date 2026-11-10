import { useEffect, useState } from "react";
import * as api from "../api";

const sectionStyle = { border: "1px solid #e6dac0", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const buttonStyle = { padding: "7px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #a9631f", background: "white", color: "#a9631f", cursor: "pointer" };
const disabledStyle = { ...buttonStyle, borderColor: "#ddd0b2", color: "#ab9a78", cursor: "default" };
const inputStyle = { fontSize: 12, padding: "4px 6px", borderRadius: 6, border: "1px solid #ddd0b2" };

const pm = (v) => (v == null ? "—" : `±${v}`);
const signed = (v) => (v == null ? "—" : `${v > 0 ? "+" : ""}${v}`);

/**
 * Bring one block of files (a flight day, a sensor) onto the level of
 * another from the ground both covered, or by an offset known from a
 * ground comparison. The backend measures the difference with its
 * uncertainty and says what it cannot tell apart from a level shift
 * (different flying heights, a thin overlap, a plane carried far outside
 * the overlap) - see magcore/block_leveling.py.
 */
export default function BlockLevelingPanel({ projectId, processSummary, onApplied, onError }) {
  const [open, setOpen] = useState(false);
  const [files, setFiles] = useState([]);
  const [shifts, setShifts] = useState([]);
  const [role, setRole] = useState({}); // file index -> "ref" | "tgt"
  const [model, setModel] = useState("auto");
  const [estimate, setEstimate] = useState(null);
  const [manual, setManual] = useState("");
  const [busy, setBusy] = useState(false);

  const ready = !!processSummary;
  const reference = files.filter((f) => role[f.index] === "ref").map((f) => f.index);
  const target = files.filter((f) => role[f.index] === "tgt").map((f) => f.index);

  const load = async () => {
    try {
      const resp = await api.getBlockFiles(projectId);
      setFiles(resp.files);
      setShifts(resp.shifts);
    } catch (e) {
      onError && onError(e);
    }
  };

  // the list follows the project: a new upload or a re-run changes it
  useEffect(() => {
    if (open && ready && projectId) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, ready, projectId, processSummary]);

  const run = async (fn) => {
    try {
      setBusy(true);
      await fn();
    } catch (e) {
      onError && onError(e);
    } finally {
      setBusy(false);
    }
  };

  const measure = () =>
    run(async () => {
      setEstimate(await api.estimateBlockShift(projectId, { reference_files: reference, target_files: target, model }));
    });

  const apply = (req) =>
    run(async () => {
      const resp = await api.applyBlockShift(projectId, req);
      setShifts(resp.shifts);
      setEstimate(null);
      onApplied && (await onApplied(resp.process_summary));
    });

  const setFileRole = (index, value) => {
    setEstimate(null);
    setRole((prev) => ({ ...prev, [index]: prev[index] === value ? undefined : value }));
  };

  const name = (index) => files.find((f) => f.index === index)?.name ?? `파일 ${index + 1}`;
  const canMeasure = ready && !busy && reference.length > 0 && target.length > 0;
  const manualValue = manual.trim() === "" ? null : Number(manual);
  const canManual = ready && !busy && target.length > 0 && manualValue != null && Number.isFinite(manualValue);
  const shift = estimate?.shift;

  return (
    <div style={sectionStyle}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{ width: "100%", textAlign: "left", padding: "8px 12px", border: "none", background: "none", cursor: "pointer", fontSize: 13, fontWeight: 600, color: "#2c2418" }}
      >
        {open ? "▾" : "▸"} 5-0. 블록·센서 간 레벨 맞춤 (겹치는 구역의 차이로)
        {shifts.length > 0 && <span style={{ marginLeft: 6, fontWeight: 400, color: "#166534" }}>적용됨 {shifts.length}건</span>}
      </button>
      {open && (
        <div style={bodyStyle}>
          <div style={{ color: "#8a7a5c" }}>
            비행 날짜나 센서가 다른 블록은 각자 보정을 마친 뒤에도 수십 nT씩 어긋나, 블록 경계에 계단이 생깁니다. 두 블록이 함께 지나간
            구역에서 그 차이를 재고, 맞출 블록에서 뺍니다. 파일마다 <b>기준</b> 또는 <b>맞출 대상</b>을 고르세요.
          </div>
          {processSummary?.statistical_leveling?.applied && (
            <div style={{ color: "#8a7a5c" }}>
              지금은 통계적 레벨링이 켜져 있어, 측선이 서로 섞이는 블록이라면 측선 단위 보정이 블록 차이를 이미 상당 부분 없앴을 수
              있습니다. 여기서 재는 값은 그 뒤에 <b>남은</b> 차이입니다.
            </div>
          )}
          {!ready && <div style={{ color: "#ab9a78" }}>먼저 자료 처리를 실행하세요.</div>}
          {ready && files.length < 2 && <div style={{ color: "#ab9a78" }}>파일이 둘 이상일 때 쓸 수 있습니다.</div>}
          {ready && files.length >= 2 && (
            <div style={{ maxHeight: 220, overflowY: "auto", border: "1px solid #eadfc8", borderRadius: 6 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead>
                  <tr style={{ background: "#f7f2e6", position: "sticky", top: 0 }}>
                    <th style={{ textAlign: "left", padding: "4px 6px" }}>파일</th>
                    <th style={{ padding: "4px 6px" }}>측선 위 측점</th>
                    <th style={{ padding: "4px 6px" }}>기준</th>
                    <th style={{ padding: "4px 6px" }}>맞출 대상</th>
                  </tr>
                </thead>
                <tbody>
                  {files.map((f) => (
                    <tr key={f.index} style={{ borderTop: "1px solid #f0e8d6" }}>
                      <td style={{ padding: "3px 6px", maxWidth: 150, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={`${f.name} · ${f.time_range[0]} ~ ${f.time_range[1]}`}>
                        {f.name}
                        {f.shift_index != null && <span style={{ color: "#166534" }}> ✓</span>}
                      </td>
                      <td style={{ padding: "3px 6px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{f.n_on_lines?.toLocaleString() ?? "—"}</td>
                      <td style={{ textAlign: "center" }}>
                        <input type="checkbox" aria-label={`${f.name} 기준 블록`} checked={role[f.index] === "ref"} onChange={() => setFileRole(f.index, "ref")} />
                      </td>
                      <td style={{ textAlign: "center" }}>
                        <input type="checkbox" aria-label={`${f.name} 맞출 대상`} checked={role[f.index] === "tgt"} onChange={() => setFileRole(f.index, "tgt")} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {ready && files.length >= 2 && (
            <>
              <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ whiteSpace: "nowrap" }}>차이의 모양</span>
                <select style={{ ...inputStyle, flex: 1, minWidth: 0 }} value={model} onChange={(e) => { setModel(e.target.value); setEstimate(null); }}>
                  <option value="auto">자동 (기울기가 뚜렷할 때만 평면)</option>
                  <option value="constant">상수</option>
                  <option value="plane">평면 (위치에 따라 달라지는 차이)</option>
                </select>
              </label>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                <button style={canMeasure ? buttonStyle : disabledStyle} disabled={!canMeasure} onClick={measure}>
                  {busy ? "계산 중..." : "차이 재기"}
                </button>
                <button
                  style={canMeasure && shift ? { ...buttonStyle, background: "#a9631f", color: "white" } : disabledStyle}
                  disabled={!canMeasure || !shift}
                  onClick={() => apply({ mode: "apply", reference_files: reference, target_files: target, model: shift.model })}
                >
                  이 차이를 빼서 맞추기
                </button>
              </div>
              {shift && (
                <div style={{ padding: "6px 8px", borderRadius: 6, background: "#f7f2e6", border: "1px solid #e6dac0" }}>
                  <div>
                    <b>맞출 대상 − 기준 = {signed(shift.constant_nt)} nT</b> ({pm(shift.constant_se_nt)} nT, {shift.model === "plane" ? "평면" : "상수"})
                  </div>
                  {shift.model === "plane" && (
                    <div>
                      기울기: 동쪽으로 {signed(shift.gradient_x_nt_per_km)} ({pm(shift.gradient_x_se)}), 북쪽으로 {signed(shift.gradient_y_nt_per_km)} ({pm(shift.gradient_y_se)}) nT/km ·
                      블록 끝의 보정 불확실도 ±{shift.max_uncertainty_nt} nT
                    </div>
                  )}
                  <div style={{ color: "#6b5c42" }}>
                    공통 셀 {shift.n_cells}개 ({shift.cell_size_m} m, {(shift.overlap_area_m2 / 1e4).toFixed(1)} ha) · 모형을 뺀 뒤 셀 간 산포 {shift.residual_std_nt} nT
                    {shift.altitude_difference_m != null && <> · 고도 차 {signed(shift.altitude_difference_m)} m</>}
                  </div>
                  {shift.warnings.map((w, i) => (
                    <div key={i} style={{ color: "#b45309", marginTop: 3 }}>⚠ {w}</div>
                  ))}
                </div>
              )}
              <div style={{ borderTop: "1px dashed #e6dac0", paddingTop: 8 }}>
                <div style={{ color: "#8a7a5c", marginBottom: 4 }}>
                  지상에서 두 센서를 나란히 놓고 잰 차이처럼 이미 아는 값이 있으면 직접 넣습니다 (맞출 대상이 기준보다 높게 읽는 만큼, nT).
                </div>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <input type="number" step="0.1" aria-label="알고 있는 오프셋 (nT)" placeholder="예: 21.4" style={{ ...inputStyle, width: 90 }} value={manual} onChange={(e) => setManual(e.target.value)} />
                  <button style={canManual ? buttonStyle : disabledStyle} disabled={!canManual} onClick={() => apply({ mode: "manual", target_files: target, manual_constant_nt: manualValue })}>
                    이 값으로 맞추기
                  </button>
                </div>
              </div>
            </>
          )}
          {shifts.length > 0 && (
            <div>
              <b>적용된 맞춤</b>
              {shifts.map((s, i) => (
                <div key={i} style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 4 }}>
                  <span style={{ flex: 1 }}>
                    {s.files.map(name).join(", ")}: {signed(s.constant_nt)} nT{s.model === "plane" ? " + 평면" : ""} {s.source === "manual" ? "(직접 입력)" : `(${pm(s.constant_se_nt)}, 기준 ${s.reference_files.map(name).join(", ")})`}
                  </span>
                  <button style={busy ? disabledStyle : buttonStyle} disabled={busy} onClick={() => apply({ mode: "reset", target_files: s.files })}>
                    되돌리기
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
