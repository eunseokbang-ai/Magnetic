import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

const sectionStyle = { border: "1px solid #e6dac0", borderRadius: 8, marginBottom: 10, background: "white" };
const bodyStyle = { padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };
const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #ddd0b2" };
const buttonStyle = {
  padding: "7px 10px",
  fontSize: 12,
  fontWeight: 600,
  borderRadius: 6,
  border: "1px solid #a9631f",
  background: "#a9631f",
  color: "white",
  cursor: "pointer",
};
const subHeaderStyle = { fontWeight: 600, color: "#4a3d28", marginTop: 4 };
const hrStyle = { border: "none", borderTop: "1px solid #e6dac0", margin: "4px 0" };

function Field({ label, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ color: "#6b5c42" }}>{label}</span>
      {children}
    </label>
  );
}

// --- 반복측선(Repeatability) 분석 --------------------------------------

function RepeatabilitySection({
  ready,
  onUploadFiles,
  uploading,
  uploadInfo,
  onAnalyze,
  analyzing,
  result,
  error,
}) {
  return (
    <div>
      <div style={subHeaderStyle}>반복측선(Repeatability) 분석</div>
      <div style={{ color: "#8a7a5c" }}>
        같은 구간을 여러 번(왕복 포함) 비행한 별도의 테스트 비행 자료를 업로드하면, 통과별 편차로부터 실제 시스템 노이즈
        봉투(1/2/3-시그마)와 헤딩오차를 계산합니다. UAV 자력탐사 가이드라인 권장 QC 절차입니다.
      </div>
      <input
        type="file"
        accept=".csv,.asc,.txt"
        multiple
        disabled={!ready || uploading}
        onChange={(e) => e.target.files.length && onUploadFiles(Array.from(e.target.files))}
      />
      {uploading && <div style={{ color: "#8a7a5c" }}>업로드 중...</div>}
      {uploadInfo && <div style={{ color: "#4a3d28" }}>업로드됨: {uploadInfo.n_points}개 포인트</div>}
      <button style={buttonStyle} disabled={!ready || !uploadInfo || analyzing} onClick={onAnalyze}>
        {analyzing ? "분석 중..." : "반복측선 분석 실행"}
      </button>
      {error && <div style={{ color: "#dc2626" }}>{error}</div>}
      {result && !result.available && <div style={{ color: "#b45309" }}>{result.reason}</div>}
      {result && result.available && (
        <div style={{ color: "#4a3d28" }}>
          반복 그룹 {result.n_groups}개, 총 통과 {result.n_passes_total}회
          <br />
          전체 노이즈 봉투: 1σ {result.overall_noise_1sigma_nt?.toFixed(2)} nT / 2σ{" "}
          {result.overall_noise_2sigma_nt?.toFixed(2)} nT / 3σ {result.overall_noise_3sigma_nt?.toFixed(2)} nT
          {result.groups?.map((g, i) => (
            <div key={i} style={{ marginTop: 4, paddingLeft: 8, borderLeft: "2px solid #e6dac0" }}>
              그룹 {i + 1}: 통과 {g.n_passes}회 (정방향 {g.n_forward} / 역방향 {g.n_reverse}), 1σ{" "}
              {g.noise_1sigma_nt?.toFixed(2)} nT
              {g.heading_error_nt != null && <> , 헤딩오차 {g.heading_error_nt.toFixed(2)} nT</>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// --- 파워스펙트럼 진단 + 노치필터 --------------------------------------

function SpectrumSection({
  ready,
  lineOptions,
  selectedLineId,
  setSelectedLineId,
  spectrumValue,
  setSpectrumValue,
  onView,
  loading,
  result,
  error,
  notchFrequencies,
  onAddNotch,
  onRemoveNotch,
}) {
  return (
    <div>
      <div style={subHeaderStyle}>파워스펙트럼 진단 / 노치필터</div>
      <div style={{ color: "#8a7a5c" }}>
        측선 하나를 선택해 원시 자력 신호의 파워스펙트럼을 확인합니다. 드론 모터 회전 주파수(~45-60Hz)나 흔들림 주파수
        (~0.1-0.6Hz) 같은 특정 간섭 주파수가 뚜렷한 피크로 보이면, 아래에서 노치필터로 등록해 다음 처리 실행 시 제거할 수
        있습니다.
      </div>
      <Field label="측선 선택">
        <select
          style={inputStyle}
          value={selectedLineId ?? ""}
          onChange={(e) => setSelectedLineId(e.target.value === "" ? null : parseInt(e.target.value, 10))}
        >
          <option value="">측선 선택...</option>
          {(lineOptions || []).map((id) => (
            <option key={id} value={id}>
              측선 #{id}
            </option>
          ))}
        </select>
      </Field>
      <Field label="신호">
        <select style={inputStyle} value={spectrumValue} onChange={(e) => setSpectrumValue(e.target.value)}>
          <option value="mag_raw">원시 자력값 (mag_raw)</option>
          <option value="mag_filtered">필터링된 자력값 (mag_filtered)</option>
        </select>
      </Field>
      <button style={buttonStyle} disabled={!ready || selectedLineId == null || loading} onClick={onView}>
        {loading ? "계산 중..." : "스펙트럼 보기"}
      </button>
      {error && <div style={{ color: "#dc2626" }}>{error}</div>}
      {result && result.available && (
        <>
          <Plot
            data={[{ x: result.freqs_hz, y: result.psd, type: "scatter", mode: "lines", line: { color: "#a9631f" } }]}
            layout={{
              width: 280,
              height: 180,
              margin: { l: 45, r: 10, t: 10, b: 35 },
              xaxis: { title: "Hz" },
              yaxis: { title: "PSD", type: "log" },
            }}
            config={{ displayModeBar: false }}
          />
          {result.peak_frequencies_hz?.length > 0 && (
            <div>
              감지된 후보 피크:
              {result.peak_frequencies_hz.map((f) => (
                <button
                  key={f}
                  style={{ ...buttonStyle, background: "white", color: "#a9631f", padding: "3px 6px", marginLeft: 4, marginTop: 4 }}
                  onClick={() => onAddNotch(f)}
                  disabled={notchFrequencies.some((nf) => Math.abs(nf - f) < 0.01)}
                >
                  {f.toFixed(2)}Hz 노치필터에 추가
                </button>
              ))}
            </div>
          )}
        </>
      )}
      {result && !result.available && <div style={{ color: "#b45309" }}>이 측선의 자료가 부족합니다.</div>}
      {notchFrequencies.length > 0 && (
        <div>
          등록된 노치필터 주파수 (다음 "자료 처리 실행" 시 적용):
          {notchFrequencies.map((f) => (
            <span
              key={f}
              style={{
                display: "inline-block", margin: "4px 4px 0 0", padding: "2px 6px",
                background: "#faf0e2", border: "1px solid #ecd0a3", borderRadius: 4,
              }}
            >
              {f.toFixed(2)}Hz{" "}
              <button style={{ border: "none", background: "none", color: "#dc2626", cursor: "pointer" }} onClick={() => onRemoveNotch(f)}>
                ×
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// --- 멀티스케일 엣지 검출(Worming) ---------------------------------------

function MultiscaleEdgesSection({ ready, heightsText, setHeightsText, percentile, setPercentile, onRun, running, result, error, showLayer, setShowLayer }) {
  return (
    <div>
      <div style={subHeaderStyle}>멀티스케일 엣지 검출 (Worming)</div>
      <div style={{ color: "#8a7a5c" }}>
        여러 상향연속 고도에서 총수평도함수(THDR) 능선을 추출해 지질 경계/구조선을 개략적으로 매핑합니다. 오일러
        디컨볼루션보다 가벼운 빠른 개관용 도구입니다.
      </div>
      <Field label="상향연속 고도 목록 (m, 쉼표로 구분)">
        <input style={inputStyle} value={heightsText} onChange={(e) => setHeightsText(e.target.value)} />
      </Field>
      <Field label="능선 임계 백분위수(%)">
        <input
          type="number"
          style={inputStyle}
          value={percentile}
          onChange={(e) => setPercentile(parseFloat(e.target.value))}
        />
      </Field>
      <button style={buttonStyle} disabled={!ready || running} onClick={onRun}>
        {running ? "계산 중..." : "멀티스케일 엣지 검출 실행"}
      </button>
      {error && <div style={{ color: "#dc2626" }}>{error}</div>}
      {result && (
        <div style={{ color: "#4a3d28" }}>
          검출된 엣지 포인트: {result.n_points}개
          <label style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6 }}>
            <input type="checkbox" checked={showLayer} onChange={(e) => setShowLayer(e.target.checked)} />
            <span>지도에 결과 표시</span>
          </label>
        </div>
      )}
    </div>
  );
}

export default function GuidelinePanel(props) {
  return (
    <div style={sectionStyle}>
      <div style={bodyStyle}>
        <RepeatabilitySection {...props.repeatability} ready={props.ready} />
        <hr style={hrStyle} />
        <SpectrumSection {...props.spectrum} ready={props.ready} />
        <hr style={hrStyle} />
        <MultiscaleEdgesSection {...props.multiscaleEdges} ready={props.ready} />
      </div>
    </div>
  );
}
