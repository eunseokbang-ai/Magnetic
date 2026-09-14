const sectionStyle = { border: "1px solid #e6dac0", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #ddd0b2" };
const buttonStyle = {
  padding: "7px 10px",
  fontSize: 12,
  fontWeight: 600,
  borderRadius: 6,
  border: "1px solid #0f766e",
  background: "#0f766e",
  color: "white",
  cursor: "pointer",
};
const applyButtonStyle = { ...buttonStyle, border: "1px solid #a9631f", background: "#a9631f" };
const listStyle = { maxHeight: 320, overflowY: "auto", border: "1px solid #f3ecd9", borderRadius: 4, padding: "4px 6px" };
const rowStyle = { display: "flex", alignItems: "flex-start", gap: 6, padding: "4px 0", borderBottom: "1px solid #faf6ec" };
const badgeStyle = {
  minWidth: 20,
  height: 20,
  borderRadius: 10,
  background: "#2c2418",
  color: "white",
  fontSize: 11,
  fontWeight: 700,
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  flexShrink: 0,
};
const warnStyle = { color: "#b45309", fontSize: 11 };

const FIELD_LABELS = {
  as: "해석신호 (AS) — 권장",
  rtp: "극자기화 변환 (RTP)",
  residual: "잔여이상 (광역추세 제거)",
  anomaly: "자기이상 (원본)",
};

function Field({ label, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ color: "#6b5c42" }}>{label}</span>
      {children}
    </label>
  );
}

export default function AnomalyCandidatePanel({
  ready,
  params,
  setParams,
  onRun,
  running,
  result,
  error,
  showOnMap,
  setShowOnMap,
  selectedIndices,
  onToggle,
  onSelectAll,
  onClearSelection,
  onFocusCandidate,
  onApplySelected,
  applying,
}) {
  const update = (key, val) => setParams((p) => ({ ...p, [key]: val }));
  const nSelected = selectedIndices?.size || 0;

  return (
    <div style={sectionStyle}>
      <div style={bodyStyle}>
        <div style={{ color: "#8a7a5c" }}>
          과업지역에서 가장 강한 국지 이상 {params.n_candidates}곳을 순위대로 찾아 지도에 번호로 표시합니다. 지질이상보다 지상
          인공구조물의 이상이 더 큰 지역에서, 구조물 이상을 하나씩 확인하며 제거하기 위한 <b>편집 시작점</b>입니다. 이 단계는
          아무것도 지우지 않습니다 — 목록에서 인공구조물이라고 판단한 것만 선택해 제거하세요.
        </div>
        <Field label="순위 기준 그리드">
          <select style={inputStyle} value={params.field} onChange={(e) => update("field", e.target.value)}>
            {Object.entries(FIELD_LABELS).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </Field>
        <div style={{ color: "#8a7a5c", fontSize: 11, marginTop: -4 }}>
          기본값이 해석신호인 이유: 자기이상 원본에서는 하나의 구조물이 (+)첨두와 그 북쪽 (−)로브 두 개로 나타나 한 물체가 후보
          두 자리를 차지하고, 두 지점 모두 물체 바로 위가 아닙니다. 해석신호는 자화 방향과 무관해 물체마다 첨두가 하나이고 위치도
          가장 정확합니다 (실측 위치오차 AS 1.0 m / 원본 2.0 m / RTP 4.1 m).
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <div style={{ flex: 1 }}>
            <Field label="표시할 후보 개수">
              <input
                type="number"
                min="1"
                max="100"
                style={inputStyle}
                value={params.n_candidates}
                onChange={(e) => update("n_candidates", parseInt(e.target.value, 10))}
              />
            </Field>
          </div>
          <div style={{ flex: 1 }}>
            <Field label="탐지 격자 크기 (m)">
              <input
                type="number"
                style={inputStyle}
                value={params.cell_size_m}
                onChange={(e) => update("cell_size_m", parseFloat(e.target.value))}
              />
            </Field>
          </div>
        </div>
        <Field label="제거 영역 최대 반경 (m) — 이보다 넓게 퍼진 이상은 국지 구조물이 아님">
          <input
            type="number"
            style={inputStyle}
            value={params.max_radius_m}
            onChange={(e) => update("max_radius_m", parseFloat(e.target.value))}
          />
        </Field>
        <button style={buttonStyle} disabled={!ready || running} onClick={onRun}>
          {running ? "탐색 중..." : "편집 모드 시작 — 이상 후보 표시"}
        </button>
        {error && <div style={{ color: "#dc2626" }}>{error}</div>}

        {result && (
          <div style={{ color: "#4a3d28", display: "flex", flexDirection: "column", gap: 8 }}>
            <div>
              후보 {result.n_candidates}개 · 기준 {FIELD_LABELS[result.field] || result.field} · 격자 {result.cell_size_m} m
              {result.line_spacing_m ? ` · 측선간격 ${result.line_spacing_m.toFixed(1)} m` : ""}
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input type="checkbox" checked={showOnMap} onChange={(e) => setShowOnMap(e.target.checked)} />
              <span>지도에 후보 표시</span>
            </label>

            <div style={{ display: "flex", gap: 6 }}>
              <button style={{ ...inputStyle, cursor: "pointer", width: "auto" }} onClick={onSelectAll}>
                전체 선택
              </button>
              <button style={{ ...inputStyle, cursor: "pointer", width: "auto" }} onClick={onClearSelection}>
                선택 해제
              </button>
            </div>

            <div style={listStyle}>
              {result.candidates.map((c, i) => (
                <div key={i} style={rowStyle}>
                  <input
                    type="checkbox"
                    checked={selectedIndices.has(i)}
                    onChange={() => onToggle(i)}
                    style={{ marginTop: 3 }}
                  />
                  <span style={badgeStyle}>{c.rank}</span>
                  <div
                    style={{ flex: 1, cursor: "pointer" }}
                    onClick={() => onFocusCandidate && onFocusCandidate(c)}
                    title="지도에서 이 후보로 이동"
                  >
                    <div>
                      <b>{c.peak_anomaly_nt.toFixed(1)} nT</b> ({c.polarity === "max" ? "최대" : "최소"}) · 깊이 약{" "}
                      {c.depth_m.toFixed(1)} m · 반경 {c.radius_m.toFixed(0)} m
                    </div>
                    <div style={{ color: "#8a7a5c", fontSize: 11 }}>
                      측선 {c.n_lines}개 · 자료점 {c.n_points}개
                    </div>
                    {c.single_line && (
                      <div style={warnStyle}>
                        단일 측선 — 격자만으로는 레벨링 줄무늬와 구분 불가. 측선 프로파일에서 확인 후 제거하세요.
                      </div>
                    )}
                    {c.region_capped && (
                      <div style={warnStyle}>
                        영역이 상한({params.max_radius_m} m)에 걸림 — 이상이 더 넓게 퍼져 있어 잘라도 일부가 남습니다. 지상
                        구조물이 맞는지 먼저 확인하세요.
                      </div>
                    )}
                    {c.depth_resolved === false && (
                      <div style={warnStyle}>
                        격자가 소스를 분해하지 못함 — 심도는 측정값이 아니라 상한값입니다. 격자 크기를 줄여 다시 확인하세요.
                      </div>
                    )}
                    {c.at_coverage_edge && (
                      <div style={warnStyle}>자료 경계 — 변환이 만들어낸 가장자리 효과일 수 있습니다.</div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            <div style={{ color: "#8a7a5c", fontSize: 11 }}>
              깊이는 해석신호 첨두의 폭에서 측정하고(반치폭 = 깊이 × 0.556, 실측), 제거 영역은 그 깊이의 약 2배 반경까지
              잡습니다. 첨두만 좁게 잘라내면 첨두의 −31% 크기인 (−)로브가 그대로 남아 오히려 구덩이가 생기기 때문입니다.
            </div>
            <button style={applyButtonStyle} disabled={applying || nSelected === 0} onClick={onApplySelected}>
              {applying ? "적용 중..." : `선택한 ${nSelected}개 영역 제거 (측선방향 보간)`}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
