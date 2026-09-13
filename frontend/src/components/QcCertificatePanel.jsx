const sectionStyle = { border: "1px solid #e6dac0", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #ddd0b2" };
const buttonStyle = {
  padding: "7px 10px",
  fontSize: 12,
  fontWeight: 600,
  borderRadius: 6,
  border: "1px solid #7c3aed",
  background: "#7c3aed",
  color: "white",
  cursor: "pointer",
};
const tableStyle = { width: "100%", borderCollapse: "collapse", fontSize: 11 };
const thStyle = { textAlign: "left", padding: "3px 4px", borderBottom: "1px solid #e6dac0", color: "#8a7a5c" };
const tdStyle = { padding: "3px 4px", borderBottom: "1px solid #f3ecd9", verticalAlign: "top" };

function Field({ label, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ color: "#6b5c42" }}>{label}</span>
      {children}
    </label>
  );
}

const STATUS_LABEL = { pass: "PASS", fail: "FAIL", not_evaluated: "평가 불가" };
const STATUS_COLOR = { pass: "#15803d", fail: "#b91c1c", not_evaluated: "#ab9a78" };

export default function QcCertificatePanel({ ready, params, setParams, onRun, running, result, error }) {
  const update = (key, val) => setParams((p) => ({ ...p, [key]: val }));

  return (
    <div style={sectionStyle}>
      <div style={bodyStyle}>
        <div style={{ color: "#8a7a5c" }}>
          이미 계산된 노이즈 QC, 샘플링 간격 QC, 파일간 레벨 오프셋, 헤딩효과 캘리브레이션 교차검증, 반복측선 재현성, 자동 제외 비율
          지표를 아래 허용기준과 비교해 항목별 PASS/FAIL과 종합 결과를 보여줍니다. 반복측선 재현성은 "반복측선 분석"을 먼저 실행해야
          평가됩니다. 기준값은 발주처 규격/자체 품질기준에 맞게 조정하세요.
        </div>
        <Field label="노이즈 허용 배수 (샘플링 주파수 기준 참고 임계값의 몇 배까지 허용)">
          <input
            type="number"
            step="0.1"
            min="0.1"
            style={inputStyle}
            value={params.noise_threshold_multiplier}
            onChange={(e) => update("noise_threshold_multiplier", parseFloat(e.target.value))}
          />
        </Field>
        <Field label="반복측선 재현성 허용 1-sigma (nT)">
          <input
            type="number"
            step="0.5"
            min="0.1"
            style={inputStyle}
            value={params.max_repeatability_1sigma_nt}
            onChange={(e) => update("max_repeatability_1sigma_nt", parseFloat(e.target.value))}
          />
        </Field>
        <Field label="샘플링 간격 계약기준 초과 허용 비율 (%)">
          <input
            type="number"
            step="0.5"
            min="0"
            style={inputStyle}
            value={params.max_sampling_gap_pct}
            onChange={(e) => update("max_sampling_gap_pct", parseFloat(e.target.value))}
          />
        </Field>
        <Field label="자동 제외 자료 허용 비율 (%)">
          <input
            type="number"
            step="1"
            min="0"
            style={inputStyle}
            value={params.max_excluded_pct}
            onChange={(e) => update("max_excluded_pct", parseFloat(e.target.value))}
          />
        </Field>
        <button style={buttonStyle} disabled={!ready || running} onClick={onRun}>
          {running ? "평가 중..." : "QC 인증서 생성"}
        </button>
        {error && <div style={{ color: "#dc2626" }}>{error}</div>}
        {result && (
          <div style={{ color: "#4a3d28" }}>
            <div style={{ fontWeight: 700, color: STATUS_COLOR[result.overall_status], marginBottom: 6 }}>
              종합 결과: {STATUS_LABEL[result.overall_status]} ({result.n_pass} PASS / {result.n_fail} FAIL / {result.n_not_evaluated} 평가 불가)
            </div>
            <div style={{ maxHeight: 260, overflowY: "auto", border: "1px solid #f3ecd9", borderRadius: 4 }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>항목</th>
                    <th style={thStyle}>측정값</th>
                    <th style={thStyle}>허용기준</th>
                    <th style={thStyle}>결과</th>
                  </tr>
                </thead>
                <tbody>
                  {result.criteria.map((c, i) => (
                    <tr key={i} title={c.detail}>
                      <td style={tdStyle}>{c.name}</td>
                      <td style={tdStyle}>{c.value === null ? "N/A" : typeof c.value === "number" ? `${c.value.toFixed(2)}${c.unit}` : `${c.value}${c.unit}`}</td>
                      <td style={tdStyle}>{c.threshold === null ? "-" : typeof c.threshold === "number" ? `${c.threshold.toFixed(2)}${c.unit}` : String(c.threshold)}</td>
                      <td style={{ ...tdStyle, color: STATUS_COLOR[c.status], fontWeight: 600 }}>{STATUS_LABEL[c.status]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
