import { useState } from "react";
import { useApp } from "../../AppContext";
import { FILTER_METHOD_INFO, Field, KOREA_PROJECTION_INFO, bodyStyle, buttonStyle, inputStyle, sectionStyle, summaryStyle } from "./shared";

// Workflow steps 3 to 5: filtering, line detection and the corrections, with
// every processing option and the button that runs the pipeline.
// One section of components/WorkflowSteps.jsx; it reads App's state with useApp().
export default function ProcessingSection() {
  const {
    baseSummary, droneSummary, handleProcess, processParams, processSummary, processing, setProcessParams,
  } = useApp();

  const [targetWavelengthM, setTargetWavelengthM] = useState(5.0);
  const lp = processParams.line_params;
  const dsp = processParams.despike_params;
  const updateDespike = (key, val) => setProcessParams((p) => ({ ...p, despike_params: { ...p.despike_params, [key]: val } }));
  const swd = processParams.sway_detection;
  const updateSway = (key, val) => setProcessParams((p) => ({ ...p, sway_detection: { ...p.sway_detection, [key]: val } }));
  const cl = processParams.crossover_leveling;
  const updateCrossover = (key, val) => setProcessParams((p) => ({ ...p, crossover_leveling: { ...p.crossover_leveling, [key]: val } }));
  const dp = processParams.diurnal_params;
  const hc = processParams.heading_correction;
  const sl = processParams.statistical_leveling;
  const updateStatLevel = (key, val) =>
    setProcessParams((p) => ({ ...p, statistical_leveling: { ...p.statistical_leveling, [key]: val } }));
  const updateLine = (key, val) => setProcessParams((p) => ({ ...p, line_params: { ...p.line_params, [key]: val } }));
  const updateDiurnal = (key, val) => setProcessParams((p) => ({ ...p, diurnal_params: { ...p.diurnal_params, [key]: val } }));
  const updateHeading = (key, val) => setProcessParams((p) => ({ ...p, heading_correction: { ...p.heading_correction, [key]: val } }));

  return (
    <details id="wf-section-process" style={sectionStyle} open>
      <summary style={summaryStyle}>3~5. 필터 · 측선판별 · 보정</summary>
      <div style={bodyStyle}>
        <Field label="필터 종류">
          <select
            style={inputStyle}
            value={processParams.filter_method || "butterworth"}
            onChange={(e) => setProcessParams((p) => ({ ...p, filter_method: e.target.value }))}
            title={FILTER_METHOD_INFO[processParams.filter_method || "butterworth"]}
          >
            <option value="butterworth">Butterworth 저역통과 (기본값)</option>
            <option value="savgol">Savitzky-Golay (봉우리 보존)</option>
            <option value="moving_average">이동평균 (가장 단순)</option>
          </select>
        </Field>
        <div style={{ color: "#ab9a78" }}>{FILTER_METHOD_INFO[processParams.filter_method || "butterworth"]}</div>

        {droneSummary?.sensor_labels && (
          <Field label="처리에 쓸 센서 (센서 2개 기록 자료)">
            <select
              style={inputStyle}
              value={processParams.sensor_channel || "1"}
              onChange={(e) => setProcessParams((p) => ({ ...p, sensor_channel: e.target.value }))}
            >
              <option value="1">{droneSummary.sensor_labels[0]} (기본)</option>
              <option value="2">{droneSummary.sensor_labels[1]}</option>
              <option value="mean">두 센서 평균</option>
            </select>
            <div style={{ color: "#ab9a78" }}>
              어느 쪽이 나은지는 처리 후 QC의 "센서 쌍 진단"에 나옵니다(센서별 왕복 방향 효과와 배경 산포).
            </div>
          </Field>
        )}

        {(processParams.filter_method || "butterworth") === "butterworth" ? (
          <>
            <Field label="저주파 통과 필터 차단주파수 (Hz)">
              <input
                type="number"
                step="0.1"
                style={inputStyle}
                value={processParams.filter_cutoff_hz}
                onChange={(e) => setProcessParams((p) => ({ ...p, filter_cutoff_hz: parseFloat(e.target.value) }))}
              />
            </Field>
            {droneSummary?.median_speed_mps > 0 && (
              <div style={{ display: "flex", alignItems: "flex-end", gap: 6, color: "#8a7a5c" }}>
                <Field label={`평균 비행속도 ${droneSummary.median_speed_mps.toFixed(1)} m/s 기준, 보존할 최소 파장(m)`}>
                  <input
                    type="number"
                    step="0.5"
                    min="0.1"
                    style={inputStyle}
                    value={targetWavelengthM}
                    onChange={(e) => setTargetWavelengthM(parseFloat(e.target.value))}
                  />
                </Field>
                <button
                  type="button"
                  style={{ ...buttonStyle, background: "white", color: "#a9631f", padding: "4px 8px", whiteSpace: "nowrap" }}
                  disabled={!targetWavelengthM || targetWavelengthM <= 0}
                  onClick={() => {
                    const suggested = droneSummary.median_speed_mps / targetWavelengthM;
                    setProcessParams((p) => ({ ...p, filter_cutoff_hz: Math.round(suggested * 1000) / 1000 }));
                  }}
                >
                  권장값 적용 ({(droneSummary.median_speed_mps / (targetWavelengthM || 1)).toFixed(3)} Hz)
                </button>
              </div>
            )}
          </>
        ) : (
          <>
            <Field label="필터 창 크기 (초)">
              <input
                type="number"
                step="0.1"
                min="0.1"
                style={inputStyle}
                value={processParams.filter_window_seconds}
                onChange={(e) => setProcessParams((p) => ({ ...p, filter_window_seconds: parseFloat(e.target.value) }))}
              />
            </Field>
            {processParams.filter_method === "savgol" && (
              <Field label="다항식 차수 (커질수록 봉우리 보존↑, 잡음 제거↓)">
                <input
                  type="number"
                  min="1"
                  max="7"
                  style={inputStyle}
                  value={processParams.filter_polyorder}
                  onChange={(e) => setProcessParams((p) => ({ ...p, filter_polyorder: parseInt(e.target.value, 10) }))}
                />
              </Field>
            )}
          </>
        )}

        <Field label="GPS-자력계 시간 오프셋 (초) — 자력계 내부 지연으로 위치가 실제와 어긋날 때 보정. 0=보정 없음">
          <input
            type="number"
            step="0.1"
            style={inputStyle}
            value={processParams.gps_mag_lag_seconds}
            onChange={(e) => setProcessParams((p) => ({ ...p, gps_mag_lag_seconds: parseFloat(e.target.value) }))}
          />
        </Field>

        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={dsp.enabled} onChange={(e) => updateDespike("enabled", e.target.checked)} />
          <span>스파이크(이상치) 제거 사용 — 필터 전에 울타리·차량 등 순간 잡음을 이동중앙값으로 대체</span>
        </label>
        {dsp.enabled && (
          <>
            <Field label="이동창 크기 (샘플 수, 홀수) — 클수록 넓은 구간의 중앙값과 비교">
              <input type="number" step="2" min="3" style={inputStyle} value={dsp.window_size} onChange={(e) => updateDespike("window_size", parseInt(e.target.value, 10))} />
            </Field>
            <Field label="탐지 민감도 (표준편차 배수) — 작을수록 더 많이 스파이크로 판정">
              <input type="number" step="0.5" min="0.1" style={inputStyle} value={dsp.threshold_k} onChange={(e) => updateDespike("threshold_k", parseFloat(e.target.value))} />
            </Field>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input type="checkbox" checked={dsp.adaptive} onChange={(e) => updateDespike("adaptive", e.target.checked)} />
              <span>Adaptive Hampel — 급격한 기울기 구간(진짜 경사 신호)은 이동창을 넓혀 스파이크 오판을 줄임</span>
            </label>
            {dsp.adaptive && (
              <Field label="급변 판정 기울기 (nT/샘플) — 이 값을 초과하면 창을 확장">
                <input
                  type="number"
                  step="0.5"
                  min="0.1"
                  style={inputStyle}
                  value={dsp.adaptive_gradient_threshold}
                  onChange={(e) => updateDespike("adaptive_gradient_threshold", parseFloat(e.target.value))}
                />
              </Field>
            )}
          </>
        )}

        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={swd.enabled} onChange={(e) => updateSway("enabled", e.target.checked)} />
          <span
            title="드론에 매달린 자력계가 흔들리거나 회전하면 헤딩 오차가 생겨 측선 줄무늬(코러게이션)의 주원인이 됩니다. 원본 파일에 자이로/가속도 컬럼이 있을 때만 동작하며, 없으면 자동으로 건너뜁니다."
          >
            IMU 흔들림(스웨이) 검출 — 자이로/가속도 데이터로 센서가 흔들린 구간을 자동 제외 (지원 포맷에서만 동작)
          </span>
        </label>
        {swd.enabled && (
          <Field label="탐지 민감도 (robust z-score 배수) — 작을수록 더 많이 흔들림으로 판정">
            <input
              type="number"
              step="0.5"
              min="0.1"
              style={inputStyle}
              value={swd.threshold_k}
              onChange={(e) => updateSway("threshold_k", parseFloat(e.target.value))}
            />
          </Field>
        )}

        <Field label="측선 방향 허용오차 (deg)">
          <input type="number" style={inputStyle} value={lp.heading_tolerance_deg} onChange={(e) => updateLine("heading_tolerance_deg", parseFloat(e.target.value))} />
        </Field>
        <Field label="최소 측선 속도 (m/s) — 이보다 느리면 이착륙/정지로 간주">
          <input type="number" step="0.1" style={inputStyle} value={lp.min_speed_mps} onChange={(e) => updateLine("min_speed_mps", parseFloat(e.target.value))} />
        </Field>
        <Field label="최소 측선 길이 (m)">
          <input type="number" style={inputStyle} value={lp.min_line_length_m} onChange={(e) => updateLine("min_line_length_m", parseFloat(e.target.value))} />
        </Field>
        <Field label="터닝 여분 구간 (m) — 측선 양끝에서 추가로 자를 거리">
          <input type="number" style={inputStyle} value={lp.turn_buffer_m} onChange={(e) => updateLine("turn_buffer_m", parseFloat(e.target.value))} />
        </Field>
        <Field label="주 측선 방향 검출 방식">
          <select style={inputStyle} value={lp.direction_method || "heading_histogram"} onChange={(e) => updateLine("direction_method", e.target.value)}>
            <option value="heading_histogram">진행방향 히스토그램 (기본값) — 순간 비행 방향 최빈값</option>
            <option value="pca">PCA 자동 방향 검출 — 측점 분포의 주축(길게 늘어진 방향) 기준</option>
          </select>
        </Field>

        <Field label="좌표계 (Korea Projection) — 국내 조사 시 표준 좌표계 선택, 비워두면 UTM 자동 감지">
          <select
            style={inputStyle}
            value={processParams.korea_projection ?? ""}
            onChange={(e) => setProcessParams((p) => ({ ...p, korea_projection: e.target.value === "" ? null : e.target.value }))}
            title={KOREA_PROJECTION_INFO[processParams.korea_projection ?? ""]}
          >
            <option value="">자동 (UTM 자동 감지)</option>
            <option value="korea_utm">KoreaUTM (EPSG:5179)</option>
            <option value="korea2010">Korea2010 (EPSG:5185~5188)</option>
            <option value="utm">UTM (국내 51N/52N)</option>
          </select>
        </Field>
        <div style={{ color: "#ab9a78" }}>{KOREA_PROJECTION_INFO[processParams.korea_projection ?? ""]}</div>

        <Field label="좌표계 수동 지정 (EPSG 코드) — 지정 시 위 Korea Projection 선택보다 우선 적용. 예: UTM 48N = 32648">
          <input
            type="number"
            style={inputStyle}
            placeholder="자동 감지"
            value={processParams.utm_epsg_override ?? ""}
            onChange={(e) => setProcessParams((p) => ({ ...p, utm_epsg_override: e.target.value === "" ? null : parseInt(e.target.value, 10) }))}
          />
        </Field>

        <Field label="베이스 시간 오프셋 — 베이스 로거 시계가 GPS와 안맞을 때 보정 (기본값 0시간 0분 0초)">
          {(() => {
            const total = dp.time_offset_seconds || 0;
            const sign = total < 0 ? -1 : 1;
            const absTotal = Math.abs(total);
            const h = Math.trunc(absTotal / 3600);
            const m = Math.trunc((absTotal - h * 3600) / 60);
            const s = Math.round(absTotal - h * 3600 - m * 60);
            const setPart = (part, value) => {
              const v = value === "" || Number.isNaN(value) ? 0 : Math.abs(value);
              const next = { h, m, s, [part]: v };
              updateDiurnal("time_offset_seconds", sign * (next.h * 3600 + next.m * 60 + next.s));
            };
            return (
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <button
                  type="button"
                  title="부호 전환 (베이스 시계가 GPS보다 빠르면 +, 느리면 -)"
                  style={{ width: 28, padding: "3px 0", fontSize: 12, borderRadius: 4, border: "1px solid #ddd0b2", background: "white", cursor: "pointer" }}
                  onClick={() => updateDiurnal("time_offset_seconds", -total)}
                >
                  {sign < 0 ? "−" : "+"}
                </button>
                <input type="number" min="0" style={{ ...inputStyle, width: 60 }} value={h} title="시" onChange={(e) => setPart("h", parseInt(e.target.value, 10))} />
                <span>시</span>
                <input type="number" min="0" style={{ ...inputStyle, width: 60 }} value={m} title="분" onChange={(e) => setPart("m", parseInt(e.target.value, 10))} />
                <span>분</span>
                <input type="number" min="0" style={{ ...inputStyle, width: 60 }} value={s} title="초" onChange={(e) => setPart("s", parseInt(e.target.value, 10))} />
                <span>초</span>
              </div>
            );
          })()}
        </Field>
        <Field label="일변화 기준값">
          <select style={inputStyle} value={dp.reference} onChange={(e) => updateDiurnal("reference", e.target.value)}>
            <option value="mean">비행시간 동안 베이스 평균</option>
            <option value="first">베이스 첫 값</option>
          </select>
        </Field>

        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={hc.enabled} onChange={(e) => updateHeading("enabled", e.target.checked)} />
          <span>헤딩(비행방향) 보정 사용 — 캘리브레이션 비행 없이 반대방향 측선의 조용한 구간으로 추정</span>
        </label>
        {hc.enabled && (
          <>
            <Field
              label={
                <span title="국소 평면 방식은 측선 2개 간격 정도의 구역마다 '평면(지질 경사) + 방향별 단차'를 동시에 맞춰, 측선 사이의 실제 지질 변화가 헤딩 오차로 잘못 잡히는 것을 막습니다. 모델이 잘 맞는(=넓고 값이 일정한) 구역만 골라 쓰므로, 조용한 구역을 눈으로 고를 필요가 없습니다. 최근접 점쌍은 기존 방식으로, 측선 간격만큼 떨어진 두 점을 직접 빼기 때문에 지질 경사가 그대로 섞여 들어갑니다.">
                  헤딩 보정 방법 ⓘ
                </span>
              }
            >
              <select style={inputStyle} value={hc.method} onChange={(e) => updateHeading("method", e.target.value)}>
                <option value="local_plane">국소 평면 + 방향 단차 (권장)</option>
                <option value="nearest_pair">최근접 점쌍 (기존 방식)</option>
              </select>
            </Field>
            <Field
              label={
                hc.method === "local_plane"
                  ? "조용한 구역 기준 백분위 (%) — 모델 잔차가 작은(=값이 일정한) 구역만 이 비율만큼 사용"
                  : "조용한 구간 기준 백분위 (%) — 낮을수록 더 엄격하게 조용한 지점만 사용"
              }
            >
              <input
                type="number"
                style={inputStyle}
                value={hc.quiet_percentile}
                onChange={(e) => updateHeading("quiet_percentile", parseFloat(e.target.value))}
              />
            </Field>
            {hc.method === "local_plane" && (
              <Field label="구역 반경 (측선 간격 배수) — 클수록 안정적이지만 '지질이 평면'이라는 가정이 약해집니다">
                <input
                  type="number"
                  step="0.5"
                  min="1.5"
                  max="6"
                  style={inputStyle}
                  value={hc.neighborhood_radius_factor}
                  onChange={(e) => updateHeading("neighborhood_radius_factor", parseFloat(e.target.value))}
                />
              </Field>
            )}
          </>
        )}

        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={sl.enabled} onChange={(e) => updateStatLevel("enabled", e.target.checked)} />
          <span title="지도에 세로 줄무늬(스트립)가 남아 있을 때 쓰는 보정입니다. 타이라인도 캘리브레이션 비행도 필요 없이, 각 측선을 좌우 이웃 측선과 비교해 측선마다 다른 레벨 오차를 찾아 없앱니다. 헤딩 보정은 전진/후진 두 방향에 공통된 오프셋 하나만 잡을 수 있어서, 측선마다 제각각인 오차는 이 보정으로만 제거됩니다.">
            통계적 레벨링 사용 — 이웃 측선 비교로 측선별 레벨 오차 제거 (줄무늬 완화의 핵심, 기본 켜짐) ⓘ
          </span>
        </label>
        {sl.enabled && (
          <>
            <Field
              label={
                <span title="측선 몇 개 범위로 추세(지질)를 맞출지 정합니다. '클수록 안전' 같은 방향성이 없고 적정값이 따로 있습니다. 너무 좁으면 추세가 오차 자체를 따라가 보정이 거의 안 되고, 너무 넓으면 추세가 지질 변화를 못 따라가 그 맞춤 오차를 레벨 오차인 양 빼버립니다(오차가 전혀 없는 자료에서도 수십 nT를 만들어냄). 기본값 9는 적정 구간에서 약간 보수적인 쪽입니다. 너무 넓게 잡으면 프로그램이 스스로 감지해 경고합니다.">
                  추세 창 (측선 수) — 기본값 9 근처를 권장 ⓘ
                </span>
              }
            >
              <input
                type="number"
                min="7"
                max="51"
                style={inputStyle}
                value={sl.trend_window_lines}
                onChange={(e) => updateStatLevel("trend_window_lines", parseInt(e.target.value, 10) || 9)}
              />
            </Field>
            <Field label="보정 형태">
              <select
                style={inputStyle}
                value={sl.order}
                onChange={(e) => updateStatLevel("order", parseInt(e.target.value, 10))}
              >
                <option value={0}>측선당 상수 (0차)</option>
                <option value={1}>측선 방향 1차 — 한 측선 안의 드리프트까지 보정</option>
              </select>
            </Field>
            <Field label="측선당 보정 상한 (nT) — 비워두면 제한 없음. 실제 이상대 위를 지난 측선이 깎이는 것을 막습니다">
              <input
                type="number"
                placeholder="제한 없음"
                style={inputStyle}
                value={sl.max_shift_nt ?? ""}
                onChange={(e) =>
                  updateStatLevel("max_shift_nt", e.target.value === "" ? null : parseFloat(e.target.value))
                }
              />
            </Field>
          </>
        )}

        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={cl.enabled} onChange={(e) => updateCrossover("enabled", e.target.checked)} />
          <span>타이라인(검교정 측선) 보정 사용 — 측선과 수직으로 비행한 타이라인이 있을 때만 유효 (선택사항)</span>
        </label>
        {cl.enabled && (
          <>
            <Field label="타이라인 방향 허용오차 (deg) — 주 측선과 수직 방향 기준">
              <input type="number" style={inputStyle} value={cl.tie_tolerance_deg} onChange={(e) => updateCrossover("tie_tolerance_deg", parseFloat(e.target.value))} />
            </Field>
            <Field label="교차점 매칭 거리 (m) — 측선과 타이라인이 이 거리 이내로 지나가면 교차점으로 인정">
              <input
                type="number"
                style={inputStyle}
                value={cl.max_crossover_distance_m}
                onChange={(e) => updateCrossover("max_crossover_distance_m", parseFloat(e.target.value))}
              />
            </Field>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input type="checkbox" checked={cl.iterative} onChange={(e) => updateCrossover("iterative", e.target.checked)} />
              <span>반복(iterative) 네트워크 보정 — 타이라인도 함께 보정해 오차를 양쪽에 고르게 분산 (해제 시 타이라인을 고정 기준으로 취급)</span>
            </label>
            <Field
              label={
                <span title="0차(상수): 측선 하나당 일정한 오프셋 하나만 보정합니다. 1차(폴리노미얼): 교차점 위치에 따라 보정값이 측선을 따라 선형으로 변하도록 fit합니다 - 타이라인이 2개 이상 있어 한 측선에 교차점이 여러 개 있을 때 더 정확합니다. 1차 선택 시 반복 네트워크 보정 여부와 무관하게 항상 단일 패스로 계산됩니다.">
                  보정 차수 ⓘ
                </span>
              }
            >
              <select style={inputStyle} value={cl.leveling_order} onChange={(e) => updateCrossover("leveling_order", parseInt(e.target.value, 10))}>
                <option value={0}>0차 (상수 오프셋, 기본값)</option>
                <option value={1}>1차 (폴리노미얼/선형 - 타이라인 2개 이상 권장)</option>
              </select>
            </Field>
          </>
        )}

        <button
          style={buttonStyle}
          disabled={processing || !droneSummary || (!baseSummary && dp.mode !== "assume_constant")}
          onClick={handleProcess}
        >
          {processing ? "처리 중..." : "자료 처리 실행 (필터+측선판별+보정+IGRF)"}
        </button>

        {processSummary && (
          <div style={{ color: "#4a3d28", marginTop: 4 }}>
            측선 {processSummary.n_lines}개 검출 / 유효 {processSummary.n_kept} / 자동제외 {processSummary.n_excluded_auto}
            <br />
            복각(Inclination): {processSummary.inclination_deg?.toFixed(2)}°, 편각(Declination): {processSummary.declination_deg?.toFixed(2)}°
            <br />
            좌표계: EPSG:{processSummary.utm_epsg}
            <br />
            {processSummary.gps_mag_lag?.lag_seconds !== 0 && (
              <>
                GPS-자력계 시간 오프셋 보정: {processSummary.gps_mag_lag.lag_seconds}초 적용 (경계 구간 {processSummary.gps_mag_lag.n_points_dropped}개 포인트 제외)
                <br />
              </>
            )}
            {processSummary.despike?.enabled && (
              <>
                스파이크 제거: {processSummary.despike.n_spikes_removed}개 ({processSummary.despike.pct_spikes_removed?.toFixed(2)}%)
                <br />
              </>
            )}
            {processSummary.heading_effect_calibration?.applied && (
              <>
                헤딩효과 캘리브레이션 보정 (
                {processSummary.heading_effect_calibration.calibration_source === "uploaded_file" ? "업로드된 비행" : "측선 턴 구간 자동 추출"}
                ): 측선 {processSummary.heading_effect_calibration.n_survey_points_corrected}개 포인트 보정 (
                {processSummary.heading_effect_calibration.pct_survey_points_corrected?.toFixed(1)}%, 평균{" "}
                {processSummary.heading_effect_calibration.mean_abs_correction_nt?.toFixed(2)} nT)
                {processSummary.heading_effect_calibration.quality_check?.available && (
                  <>
                    {" — 품질검증: "}
                    <span style={{ color: processSummary.heading_effect_calibration.quality_pass ? "#15803d" : "#b91c1c", fontWeight: 600 }}>
                      {processSummary.heading_effect_calibration.quality_pass ? "PASS" : "FAIL"}
                    </span>
                    {` (잔차 표준편차 ${processSummary.heading_effect_calibration.quality_check.residual_std_nt?.toFixed(2)} nT)`}
                  </>
                )}
                <br />
                {processSummary.heading_effect_calibration.coverage_warning && (
                  <>
                    <span style={{ color: "#b45309" }}>⚠ {processSummary.heading_effect_calibration.coverage_warning}</span>
                    <br />
                  </>
                )}
              </>
            )}
            {processSummary.heading_effect_calibration?.enabled &&
              !processSummary.heading_effect_calibration?.applied &&
              processSummary.heading_effect_calibration?.reason && (
                <>
                  <span style={{ color: "#ab9a78" }}>헤딩효과 캘리브레이션 보정: {processSummary.heading_effect_calibration.reason}</span>
                  <br />
                </>
              )}
            {processSummary.sway_detection?.enabled && processSummary.sway_detection?.available && (
              <>
                IMU 흔들림 검출: {processSummary.sway_detection.n_points_excluded}개 포인트 제외 ({processSummary.sway_detection.pct_points_flagged?.toFixed(2)}%, 신호:{" "}
                {processSummary.sway_detection.signal_used})
                <br />
              </>
            )}
            {processSummary.sway_detection?.enabled && processSummary.sway_detection?.available === false && (
              <>
                <span style={{ color: "#ab9a78" }}>IMU 흔들림 검출: 원본 파일에 자이로/가속도 데이터 없음 (건너뜀)</span>
                <br />
              </>
            )}
            {cl.enabled && processSummary.crossover_leveling && (
              <>
                {processSummary.crossover_leveling.applied ? (
                  <span>
                    타이라인 보정: 측선 {processSummary.crossover_leveling.n_survey_lines_corrected}개 보정 (교차점 {processSummary.crossover_leveling.n_crossovers}개), RMS{" "}
                    {processSummary.crossover_leveling.rms_before_nt?.toFixed(2)} → {processSummary.crossover_leveling.rms_after_nt?.toFixed(2)} nT
                  </span>
                ) : (
                  <span style={{ color: "#b45309" }}>타이라인 보정: {processSummary.crossover_leveling.reason}</span>
                )}
                <br />
              </>
            )}
            {processSummary.noise_qc?.available && (
              <>
                노이즈 QC (정규화 4th/8th difference): 전체 RMS {processSummary.noise_qc.overall_rms_4th_diff_nt?.toFixed(3)} /{" "}
                {processSummary.noise_qc.overall_rms_8th_diff_nt?.toFixed(3)} nT
                {processSummary.noise_qc.n_lines_flagged > 0 && (
                  <span style={{ color: "#b45309" }}> — 이상 측선 {processSummary.noise_qc.n_lines_flagged}개 감지됨</span>
                )}
                <br />
              </>
            )}
            {processSummary.sampling_qc?.available && (
              <>
                샘플링 거리 QC: 중앙값 {processSummary.sampling_qc.median_distance_m?.toFixed(2)}m, 최대{" "}
                {processSummary.sampling_qc.max_distance_m?.toFixed(2)}m
                {processSummary.sampling_qc.n_gaps_exceeding_tolerance > 0 && (
                  <span style={{ color: "#b45309" }}>
                    {" "}
                    — 허용기준({processSummary.sampling_qc.gap_tolerance_m}m) 초과 구간 {processSummary.sampling_qc.n_gaps_exceeding_tolerance}개 (
                    {processSummary.sampling_qc.pct_gaps_exceeding_tolerance?.toFixed(2)}%, GPS 순간 끊김 가능성)
                  </span>
                )}
                <br />
              </>
            )}
            {processSummary.file_level_check?.available && processSummary.file_level_check.flagged_any && (
              <div style={{ color: "#dc2626", marginTop: 4 }}>
                ⚠ 파일(타일) 간 레벨 불일치 감지: 베이스 위치 변경 등을 확인하세요 (
                {processSummary.file_level_check.files
                  .filter((f) => f.flagged)
                  .map((f) => `파일#${f.source_file_index} ${f.deviation_nt?.toFixed(1)}nT`)
                  .join(", ")}
                )
              </div>
            )}
            {processSummary.heading_correction?.measured ? (
              <span
                title="전진 방향으로 난 측선과 후진 방향으로 난 측선의 값 차이입니다. 헤딩 보정을 켜지 않아도 항상 측정해서 보여주므로, 방향별 단차가 실제로 있는지 눈이 아니라 숫자로 확인할 수 있습니다. 구역별 산포가 이 값보다 크면 방향 문제가 아니라 측선별 레벨 오차입니다."
              >
                방향별 단차(A↔B): {processSummary.heading_correction.offset_nt?.toFixed(2)} nT
                {processSummary.heading_correction.method === "local_plane" ? (
                  <>
                    {" "}
                    (국소 평면 방식 — 구역 {processSummary.heading_correction.n_matched_pairs}개 중 조용한{" "}
                    {processSummary.heading_correction.n_quiet_pairs}개 사용, 구역별 산포 ±
                    {processSummary.heading_correction.offset_spread_nt?.toFixed(2)} nT)
                  </>
                ) : (
                  <>
                    {" "}
                    (매칭 {processSummary.heading_correction.n_matched_pairs}쌍 중 조용한{" "}
                    {processSummary.heading_correction.n_quiet_pairs}쌍 사용)
                  </>
                )}
                {processSummary.heading_correction.applied ? (
                  <b style={{ color: "#15803d" }}> — 보정 적용됨</b>
                ) : hc.enabled ? (
                  <span style={{ color: "#b45309" }}> — 신뢰할 수 없어 적용 안 됨 (아래 경고 참고)</span>
                ) : (
                  <span style={{ color: "#8a7a5c" }}> — 측정만 함 (보정하려면 위 헤딩 보정을 켜세요)</span>
                )}{" "}
                ⓘ
              </span>
            ) : (
              processSummary.heading_correction?.reason && <span style={{ color: "#b45309" }}>헤딩 보정: {processSummary.heading_correction.reason}</span>
            )}
            {(processSummary.heading_correction?.warnings || []).map((w, i) => (
              <div key={i} style={{ color: "#b45309", marginTop: 2 }}>⚠ {w}</div>
            ))}
            <br />
            {processSummary.striping?.available && (
              <div
                style={{ marginTop: 2 }}
                title="측선이 좌우 이웃과 얼마나 어긋나 있는지를 nT로 잰 값입니다(줄무늬의 절대 크기). 괄호 안은 그 값을 이 탐사 자체의 자기이상 범위로 나눈 비율로, 서로 다른 탐사·시스템·기체를 비교할 때는 이 비율을 보세요 - 같은 2nT라도 자성이 강한 지역에서는 안 보이고 조용한 지역에서는 지배적입니다. 보정 전 값이라 설정을 바꿔가며 비교할 수 있습니다."
              >
                줄무늬 세기(보정 전): 측선간 어긋남 {processSummary.striping.line_level_jitter_nt?.toFixed(2)} nT
                {processSummary.striping.stripe_ratio_pct != null && (
                  <> (이상 범위의 {processSummary.striping.stripe_ratio_pct}%)</>
                )}{" "}
                ⓘ
              </div>
            )}
            {processSummary.statistical_leveling?.applied ? (
              <div style={{ marginTop: 2 }}>
                통계적 레벨링: 측선 {processSummary.statistical_leveling.n_lines}개 보정 (인접쌍{" "}
                {processSummary.statistical_leveling.n_pairs}개), 보정량 RMS{" "}
                {processSummary.statistical_leveling.rms_shift_nt?.toFixed(2)} / 최대{" "}
                {processSummary.statistical_leveling.max_shift_nt?.toFixed(2)} nT, 측선간 요철{" "}
                {processSummary.statistical_leveling.roughness_before_nt?.toFixed(2)} →{" "}
                {processSummary.statistical_leveling.roughness_after_nt?.toFixed(2)} nT
              </div>
            ) : (
              processSummary.statistical_leveling?.reason && (
                <div style={{ color: "#8a7a5c", marginTop: 2 }}>통계적 레벨링: {processSummary.statistical_leveling.reason}</div>
              )
            )}
            {(processSummary.statistical_leveling?.warnings || []).map((w, i) => (
              <div key={i} style={{ color: "#b45309", marginTop: 2 }}>⚠ {w}</div>
            ))}
            {processSummary.diurnal?.mode === "assume_constant" ? (
              <div style={{ color: "#b45309", marginTop: 4 }}>⚠ {processSummary.diurnal.note}</div>
            ) : (
              processSummary.diurnal && !processSummary.diurnal.has_overlap && (
                <div style={{ color: "#dc2626", marginTop: 4 }}>
                  ⚠ 베이스 자료와 드론 비행시간이 겹치지 않습니다. 베이스 시간범위: {processSummary.diurnal.base_time_range?.[0]} ~{" "}
                  {processSummary.diurnal.base_time_range?.[1]}. 시간 오프셋을 조정하세요.
                </div>
              )
            )}
          </div>
        )}
      </div>
    </details>
  );
}
