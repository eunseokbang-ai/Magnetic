const sectionStyle = { border: "1px solid #e5e7eb", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #d1d5db" };
const buttonStyle = {
  padding: "8px 10px",
  fontSize: 12,
  fontWeight: 700,
  borderRadius: 6,
  border: "1px solid #c2410c",
  background: "#c2410c",
  color: "white",
  cursor: "pointer",
};
const tableStyle = { width: "100%", borderCollapse: "collapse", fontSize: 11 };
const thStyle = { textAlign: "left", padding: "3px 4px", borderBottom: "1px solid #e5e7eb", color: "#6b7280" };
const tdStyle = { padding: "3px 4px", borderBottom: "1px solid #f3f4f6", verticalAlign: "top" };

function Field({ label, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ color: "#4b5563" }}>{label}</span>
      {children}
    </label>
  );
}

const PURPOSE_LABELS = {
  magnetite_fe: "자철석/철광 (Magnetite/Fe) — 자력 반응 자체 위주",
  skarn: "스카른 (Skarn) — 접촉면 인접 자력 이상 위주",
  ni_cu_pge: "Ni-Cu-PGE — 구조교차부 인접성 위주",
  custom: "사용자 지정 가중치",
};

const LAYER_LABELS = {
  asa: "ASA",
  thd: "THD",
  structure: "구조선 인접",
  contact: "접촉면 인접",
  susceptibility: "천부 감수율",
};

export default function ProspectivityPanel({ ready, params, setParams, onRun, running, result, error, showTargetsOnMap, setShowTargetsOnMap, showOverlay, setShowOverlay }) {
  const update = (key, val) => setParams((p) => ({ ...p, [key]: val }));
  const updateWeight = (key, val) => setParams((p) => ({ ...p, weights: { ...p.weights, [key]: val } }));

  return (
    <div style={sectionStyle}>
      <div style={bodyStyle}>
        <div style={{ color: "#6b7280" }}>
          이 프로그램이 실제로 계산할 수 있는 레이어(ASA, THD, 구조선/접촉면 인접성, 그리고 3차원 역산을 실행했다면 천부
          감수율)만을 0~1로 정규화해 가중합산한 점수를 계산합니다. 지질도·방사능 등 이 프로그램이 취득하지 않는 외부 자료
          기반 레이어는 포함되어 있지 않습니다 — 아래 탐사목적 프리셋은 각 레이어에 대한 가중치 배분일 뿐, 발표된 광상모델을
          그대로 구현한 것은 아닙니다. 구조선(14-3)과 접촉면(14-5)을 먼저 추출해 두면 해당 레이어가 함께 반영됩니다.
        </div>
        <Field label="탐사 목적">
          <select style={inputStyle} value={params.purpose} onChange={(e) => update("purpose", e.target.value)}>
            {Object.entries(PURPOSE_LABELS).map(([k, label]) => (
              <option key={k} value={k}>{label}</option>
            ))}
          </select>
        </Field>
        {params.purpose === "custom" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 4, border: "1px solid #f3f4f6", borderRadius: 4, padding: 8 }}>
            <span style={{ color: "#4b5563" }}>레이어별 가중치 (자동으로 합이 1이 되도록 정규화됩니다)</span>
            {Object.entries(LAYER_LABELS).map(([k, label]) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 70, color: "#6b7280" }}>{label}</span>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  style={inputStyle}
                  value={params.weights[k] ?? 0}
                  onChange={(e) => updateWeight(k, parseFloat(e.target.value))}
                />
              </div>
            ))}
          </div>
        )}
        <Field label="셀 크기 (m)">
          <input type="number" style={inputStyle} value={params.cell_size_m} onChange={(e) => update("cell_size_m", parseFloat(e.target.value))} />
        </Field>
        <Field label="구조/접촉면 인접성 감쇠거리 (m) — 클수록 멀리 있어도 점수 유지">
          <input type="number" style={inputStyle} value={params.decay_length_m} onChange={(e) => update("decay_length_m", parseFloat(e.target.value))} />
        </Field>
        <Field label="타깃 판정 점수 임계값 (0~1)">
          <input
            type="number"
            step="0.05"
            min="0"
            max="1"
            style={inputStyle}
            value={params.score_threshold}
            onChange={(e) => update("score_threshold", parseFloat(e.target.value))}
          />
        </Field>
        <Field label="최대 타깃 개수">
          <input type="number" style={inputStyle} value={params.max_targets} onChange={(e) => update("max_targets", parseInt(e.target.value, 10))} />
        </Field>
        <button style={buttonStyle} disabled={!ready || running} onClick={onRun}>
          {running ? "분석 중..." : "프로스펙티비티 분석 실행"}
        </button>
        {error && <div style={{ color: "#dc2626" }}>{error}</div>}
        {result && !result.available && <div style={{ color: "#9ca3af" }}>{result.reason}</div>}
        {result && result.available && (
          <div style={{ color: "#374151" }}>
            {result.n_targets === 0 ? (
              <div style={{ color: "#9ca3af" }}>{result.reason}</div>
            ) : (
              <>사용된 레이어: {result.layers_used.map((k) => LAYER_LABELS[k]).join(", ")} — 타깃 {result.n_targets}개</>
            )}
            <div style={{ display: "flex", gap: 12, margin: "6px 0" }}>
              <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={showTargetsOnMap} onChange={(e) => setShowTargetsOnMap(e.target.checked)} />
                <span>타깃 지도 표시</span>
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} />
                <span>점수 레이어 지도 표시</span>
              </label>
            </div>
            {result.targets && result.targets.length > 0 && (
              <div style={{ maxHeight: 220, overflowY: "auto", border: "1px solid #f3f4f6", borderRadius: 4 }}>
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={thStyle}>순위</th>
                      <th style={thStyle}>점수</th>
                      <th style={thStyle}>설명</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.targets.map((t) => (
                      <tr key={t.rank}>
                        <td style={tdStyle}>TARGET {String(t.rank).padStart(2, "0")}</td>
                        <td style={tdStyle}>{t.score.toFixed(2)}</td>
                        <td style={tdStyle}>{t.explanation}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
