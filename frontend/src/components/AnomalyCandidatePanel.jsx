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

export function RemovalMethodControls({
  removalMethod,
  setRemovalMethod,
  nSourceRemovals,
  onUndoSourceRemoval,
  onResetSourceRemovals,
  busy,
}) {
  const small = {
    padding: "4px 8px",
    fontSize: 12,
    borderRadius: 6,
    border: "1px solid #ddd0b2",
    background: "white",
    cursor: nSourceRemovals ? "pointer" : "default",
    opacity: nSourceRemovals && !busy ? 1 : 0.5,
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12 }}>
      <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <span style={{ color: "#6b5c42" }}>제거 방식</span>
        <select
          value={removalMethod}
          onChange={(e) => setRemovalMethod(e.target.value)}
          style={{ padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #ddd0b2" }}
        >
          <option value="model">구조물 모델 차감 — 권장</option>
          <option value="interpolate">영역 잘라내고 측선방향 보간</option>
        </select>
      </label>
      <div style={{ color: "#8a7a5c", fontSize: 11 }}>
        {removalMethod === "model"
          ? "표시한 영역 안에 쌍극자들을 놓아 구조물의 자기장을 맞춘 뒤, 그 계산된 자기장만 뺍니다. 영역 밖으로 뻗은 꼬리와 (−)로브까지 함께 빠지고, 구멍을 메우지 않으므로 경계 테두리나 측선 방향 줄무늬가 생기지 않습니다. 영역은 구조물(해석신호 덩어리)을 감싸면 충분합니다. 영역 안의 작은 지질 굴곡은 함께 빠질 수 있습니다."
          : "영역 안 측점을 각 측선의 앞뒤 배경값으로 직선 보간합니다. 측선마다 따로 채우므로 해석신호·미분 그리드에 측선 방향 줄무늬나 테두리가 남을 수 있고, 영역 밖으로 뻗은 (−)로브는 그대로 남습니다. 측선 프로파일의 짧은 구간 제거에 적합합니다."}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ color: "#8a7a5c" }}>
          {nSourceRemovals > 0 ? `구조물 모델 차감 ${nSourceRemovals}건` : "차감된 구조물 모델 없음"}
        </span>
        <button
          style={{ ...small, marginLeft: "auto" }}
          disabled={busy || !nSourceRemovals}
          onClick={onUndoSourceRemoval}
          title="가장 최근에 차감한 구조물 모델 1건을 되돌립니다"
        >
          ↩ 실행취소
        </button>
        <button
          style={small}
          disabled={busy || !nSourceRemovals}
          onClick={onResetSourceRemovals}
          title="차감한 구조물 모델을 모두 되돌립니다"
        >
          전체 되돌리기
        </button>
      </div>
    </div>
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
  onSelectClean,
  onClearSelection,
  onFocusCandidate,
  onApplySelected,
  applying,
  removalMethod,
  setRemovalMethod,
  sourceRemovals,
  lastRemovalAdded,
  onUndoSourceRemoval,
  onResetSourceRemovals,
  removalWorking,
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
              <button
                style={{ ...inputStyle, cursor: "pointer", width: "auto", color: "#0f766e", borderColor: "#0f766e" }}
                onClick={onSelectClean}
                title="여러 측선에서 보이고 자료 경계에 닿지 않은 후보(지도의 청록 번호)만 선택합니다"
              >
                청록 후보만 선택
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
                  <span style={{ ...badgeStyle, background: c.single_line || c.at_coverage_edge ? "#b45309" : "#0f766e" }}>{c.rank}</span>
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

            <RemovalMethodControls
              removalMethod={removalMethod}
              setRemovalMethod={setRemovalMethod}
              nSourceRemovals={sourceRemovals?.length || 0}
              onUndoSourceRemoval={onUndoSourceRemoval}
              onResetSourceRemovals={onResetSourceRemovals}
              busy={applying || removalWorking}
            />
            <button style={applyButtonStyle} disabled={applying || nSelected === 0} onClick={onApplySelected}>
              {applying
                ? removalMethod === "model"
                  ? "구조물 모델 맞추는 중..."
                  : "적용 중..."
                : removalMethod === "model"
                  ? `선택한 ${nSelected}개 구조물 모델 차감`
                  : `선택한 ${nSelected}개 영역 제거 (측선방향 보간)`}
            </button>
            {lastRemovalAdded?.length > 0 && (
              <div style={{ fontSize: 11, color: "#4a3d28" }}>
                {lastRemovalAdded.map((r, i) => (
                  <div key={i}>
                    모델 {i + 1}: 쌍극자 {r.n_sources}개, 깊이 {r.depth_m.toFixed(0)} m, 맞춤 오차 {r.fit_rms_nt.toFixed(1)} nT
                    (주변 자료 변동 {r.data_rms_nt.toFixed(0)} nT), 최대 {r.peak_model_nt.toFixed(0)} nT 차감
                    {r.warnings?.map((w, j) => (
                      <div key={j} style={warnStyle}>
                        {w}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
