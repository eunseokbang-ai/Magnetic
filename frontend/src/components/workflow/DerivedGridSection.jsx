import { useApp } from "../../AppContext";
import ExportOptions from "../ExportOptions";
import { Field, TRANSFORM_INFO, bodyStyle, buttonStyle, inputStyle, sectionStyle, summaryStyle } from "./shared";

// Workflow step 10: derived grids (RTP, derivatives, continuation ...) and
// every export of the grid on screen.
// One section of components/WorkflowSteps.jsx; it reads App's state with useApp().
export default function DerivedGridSection() {
  const {
    activeTransform, exportGeotiffColored, exportingGeotiff, exportingGrd, exportingPointsCsv, exportingXyz,
    handleError, handleExportGeotiff, handleExportGpkg, handleExportGrd, handleExportPointsCsv,
    handleExportXyz, handleTransform, processSummary, projectId, setExportGeotiffColored,
    setTransformExtraParams, transformExtraParams, transformLoading,
  } = useApp();

  return (
    <details style={sectionStyle}>
      <summary style={summaryStyle}>10. 파생 그리드 (RTP / RTE / AS / 1VD 외)</summary>
      <div style={bodyStyle}>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {[
            ["none", "그리드(원본)"],
            ["rtp", "RTP"],
            ["rte", "RTE"],
            ["1vd", "1VD"],
            ["2vd", "2VD"],
            ["as", "AS"],
            ["thdr", "THDR"],
            ["tilt", "틸트각"],
            ["theta", "세타맵"],
            ["dx", "dX"],
            ["dy", "dY"],
            ["dxx", "dXX"],
            ["dyy", "dYY"],
            ["dxy", "dXY"],
            ["dxz", "dXZ"],
            ["dyz", "dYZ"],
            ["upward_continuation", "상방연속"],
            ["detrend", "추세면제거"],
            ["microlevel", "마이크로레벨링"],
          ].map(([key, label]) => (
            <button
              key={key}
              title={TRANSFORM_INFO[key]}
              onClick={() => handleTransform(key)}
              disabled={transformLoading || !processSummary}
              style={{
                ...buttonStyle,
                background: activeTransform === key ? "#a9631f" : "white",
                color: activeTransform === key ? "white" : "#a9631f",
              }}
            >
              {label}
            </button>
          ))}
        </div>
        {transformLoading && <div style={{ color: "#8a7a5c" }}>계산 중...</div>}
        <div style={{ border: "1px solid #e6dac0", borderRadius: 6, padding: "6px 8px", display: "flex", flexDirection: "column", gap: 6 }}>
          <div
            title="'측정값 - IGRF'의 0은 물리적 기준이 아닙니다(광역 지각 자기장, 일변화 수준에 따라 수십~100 nT 움직임). 배경을 빼면 지도의 색과 0 등치선이 '주변보다 높다/낮다'와 일치합니다. 원본 그리드, 내보내기, 모든 파생그리드가 이 기준 격자에서 만들어집니다. RTP·상방연속은 배경 수준을 그대로 옮기므로 눈에 띄게 달라지고, 미분 계열(AS, 틸트, 1VD 등)은 거의 그대로입니다. 수준 = 중앙값 하나를 뺌(가정이 가장 적음). 1차/2차 = 강한 이상에 끌려가지 않게 강건하게 맞춘 면을 뺌 - 2차는 구역만 한 이상을 같이 지울 수 있으니 지도 아래에 표시되는 제거량을 확인하세요."
          >
            <div style={{ marginBottom: 3 }}>기준 격자 ⓘ</div>
            <select
              data-testid="base-grid-regional"
              style={inputStyle}
              value={transformExtraParams.regional_removal ?? "none"}
              onChange={(e) => setTransformExtraParams((p) => ({ ...p, regional_removal: e.target.value }))}
            >
              <option value="none">자력이상 그대로 (측정값 − IGRF)</option>
              <option value="level">광역 배경 제거 — 수준 (중앙값)</option>
              <option value="plane">광역 배경 제거 — 1차 추세면</option>
              <option value="quadratic">광역 배경 제거 — 2차 추세면</option>
            </select>
          </div>
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input
              type="checkbox"
              data-testid="derive-from-rtp"
              checked={!!transformExtraParams.derive_from_rtp}
              onChange={(e) => setTransformExtraParams((p) => ({ ...p, derive_from_rtp: e.target.checked }))}
            />
            <span title="틸트각, 세타맵, 1VD/2VD, THDR, dX~dYZ처럼 경계나 정점의 위치를 읽는 격자를 기준 격자가 아니라 그 RTP에서 계산합니다. 복각 때문에 밀려 있던 경계가 암체 위로 옵니다(해석용 지도의 일반적인 만드는 순서: 자력이상 → RTP → RTP의 미분). AS는 자화 방향에 덜 민감하다는 점 때문에 쓰는 격자라 RTP를 거치지 않고, 상방연속은 순서를 바꿔도 같아서 해당이 없습니다.">
              경계·정점 격자(틸트, 미분 계열)를 RTP에서 계산 ⓘ
            </span>
          </label>
          <div style={{ fontSize: 11, color: "#8a7a5c" }}>바꾼 뒤 위의 격자 버튼을 다시 누르면 적용됩니다.</div>
        </div>
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input
            type="checkbox"
            checked={transformExtraParams.derivative_presmooth}
            onChange={(e) => setTransformExtraParams((p) => ({ ...p, derivative_presmooth: e.target.checked }))}
          />
          <span title="탐사는 측선 위에서만 값을 재고 측선 사이는 재지 않으므로, 측선 간격보다 좁은 구조는 측선 직각방향으로 '측정된 적이 없는' 값입니다 - 보간이 만들어낸 것입니다. 기본 보간법인 '원본 셀'은 이웃 셀의 80%가 값이 완전히 똑같은 계단 모양이라, 그대로 미분하면 지반이 아니라 그 계단을 재게 됩니다(합성자료 검증에서 AS가 참값 대비 101% 오차). 미분 전에 실제 분해능까지 평활하면 5%로 떨어집니다. 격자 줄무늬가 보이면 이 옵션을 확인하세요.">
            미분 전에 측선 간격 분해능까지 평활 (AS/1VD/dXY 등 격자 줄무늬 억제) ⓘ
          </span>
        </label>
        {transformExtraParams.derivative_presmooth && (
          <Field label="평활 폭 (측선 간격 배수) — 1.0 권장(0.4 이하는 10 m 격자에서 거의 거르지 못함). 키우면 줄무늬는 줄지만 실제 신호도 깎입니다">
            <input
              type="number"
              step="0.05"
              min="0.05"
              max="2"
              style={inputStyle}
              value={transformExtraParams.derivative_presmooth_factor}
              onChange={(e) => setTransformExtraParams((p) => ({ ...p, derivative_presmooth_factor: parseFloat(e.target.value) }))}
            />
          </Field>
        )}
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input
            type="checkbox"
            checked={transformExtraParams.boundary_continuation}
            onChange={(e) => setTransformExtraParams((p) => ({ ...p, boundary_continuation: e.target.checked }))}
          />
          <span title="FFT 파생그리드는 자료가 없는 탐사 바깥에도 반드시 어떤 값이 있어야 계산됩니다. 지금까지는 경계에서 가장 가까운 측정값을 바깥으로 그대로 복사했는데, 그러면 경계에 걸친 강한 이상대가 빈 땅 한 줄에 통째로 복제되고, 그 틀린 값과 실제 자기장의 차이가 격자 가로/세로 방향으로 뻗어 나갑니다. 이 옵션은 값을 추측하는 대신, 실제 측정값을 재현하는 얕은 등가 소스층을 풀어서 그 소스가 만드는 자기장을 바깥에 놓습니다(측정값은 한 자리도 바뀌지 않습니다). 정답을 아는 합성자료 3개로 검증: 조용한 구역에서 미분값이 경계로부터 물려받는 오차가 참값의 19~47배에서 1.4~2.2배로 줄고, 가장 강한 이상대 진폭은 100.0% 그대로 유지됩니다. 격자당 한 번 약 10초가 걸리고 이후에는 캐시되어 즉시 계산됩니다.">
            탐사 바깥을 등가 소스층으로 연속 (경계발 줄무늬 억제, 격자당 최초 1회 ~10초) ⓘ
          </span>
        </label>
        <Field
          label={
            <span title="파생그리드(특히 dXX/dXY/dYY)의 측선방향 줄무늬에 가장 잘 듣는 방법입니다. 측선 사이는 측정된 적이 없어서 격자가 지어낸 값인데, 이 옵션은 실측값을 재현하는 '지하 등가 소스층'을 풀어서 그 소스가 만드는 자기장으로 격자를 바꿉니다. 깊이 h의 소스 분포는 h보다 잔 구조를 만들 수 없으므로, 사라지는 것은 애초에 탐사가 분해하지 못한 성분입니다. 해남 자료 실측: dXX 줄무늬 에너지 24.7% → 3.7%(1.6배), 가장 강한 이상대의 미분 진폭은 75% 유지 — 같은 조건의 측선직각 필터(6.9% / 72%)보다 낫습니다. 참고로 마이크로레벨링(디코러게이션)은 이 자료에서 전혀 듣지 않았습니다(24.7% → 25.1%). 1.6배는 보수적, 2.0배는 더 깨끗하지만 진폭을 더 깎습니다. 적용 후 '실측 재현 오차(nT)'가 표시되니 납품 근거로 쓰세요.">
              등가 소스층으로 대체 (dXX/dXY 줄무늬 억제, 측선 간격 배수) ⓘ
            </span>
          }
        >
          <select
            style={inputStyle}
            value={transformExtraParams.equivalent_source_factor ?? ""}
            onChange={(e) =>
              setTransformExtraParams((p) => ({
                ...p,
                equivalent_source_factor: e.target.value === "" ? null : parseFloat(e.target.value),
              }))
            }
          >
            <option value="">사용 안 함</option>
            <option value="1.2">1.2배 (약하게)</option>
            <option value="1.6">1.6배 (권장 - 줄무늬 24.7%→3.7%, 진폭 75% 유지)</option>
            <option value="2.0">2.0배 (더 깨끗 - 줄무늬 0.5%, 진폭 64%)</option>
            <option value="2.5">2.5배 (강하게)</option>
          </select>
        </Field>
        <Field
          label={
            <span title="격자는 측선 위에서 잰 값을 측선 사이로 보간하지만, 탐사 경계 바깥쪽 가장자리는 보간이 아니라 '바깥으로 외삽'한 구간입니다. 파생그리드(AS/dXY 등)는 FFT 계산을 위해 자료가 없는 바깥을 반드시 무언가로 채워야 하는데, 경계에 걸친 강한 이상대가 있으면 그 채움값이 틀리는 만큼 격자 가로/세로 방향으로 줄무늬가 뻗어 나갑니다(합성자료: 탐사 한가운데 이상대는 0.000%, 경계에서 2셀 떨어진 같은 이상대는 0.757%). 채움 방식을 바꿔도(미러/조화연속 등) 줄지 않고, 경계를 테이퍼로 죽이면 실제 경계부 신호의 1/4을 잃습니다. 그래서 '보정'이 아니라 '어디까지가 못 믿을 구간인지 표시'하는 옵션입니다.">
              경계 여백 (파생그리드 가장자리 줄무늬 구간) ⓘ
            </span>
          }
        >
          <select
            style={inputStyle}
            value={transformExtraParams.boundary_margin_mode}
            onChange={(e) => setTransformExtraParams((p) => ({ ...p, boundary_margin_mode: e.target.value }))}
          >
            <option value="off">사용 안 함</option>
            <option value="outline">경계선만 표시 (자료는 그대로)</option>
            <option value="mask">여백 구간 가리기 (NaN 처리)</option>
          </select>
        </Field>
        {transformExtraParams.boundary_margin_mode !== "off" && (
          <Field label="여백 폭 (m) — 비워두면 격자 외삽 거리(측선 간격의 1.2배)를 씁니다">
            <input
              type="number"
              step="1"
              min="1"
              style={inputStyle}
              placeholder="자동"
              value={transformExtraParams.boundary_margin_m ?? ""}
              onChange={(e) =>
                setTransformExtraParams((p) => ({
                  ...p,
                  boundary_margin_m: e.target.value === "" ? null : parseFloat(e.target.value),
                }))
              }
            />
          </Field>
        )}
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input
            type="checkbox"
            checked={transformExtraParams.microlevel_pre_apply}
            onChange={(e) => setTransformExtraParams((p) => ({ ...p, microlevel_pre_apply: e.target.checked }))}
          />
          <span
            title="지금 마이크로레벨링 버튼을 눌러야만 적용되는 게 아니라, 이 체크박스를 켜두면 위 그리드(원본)를 포함한 다른 모든 파생그리드 계산 전에 먼저 마이크로레벨링(측선방향 줄무늬 완화)을 적용한 뒤 그 결과에 RTP/1VD/AS 등을 계산합니다. 미분 기반 변환일수록 줄무늬가 미리 지워진 상태에서 계산되어 결과가 더 매끈해집니다."
          >
            선택한 파생그리드 계산 전에 마이크로레벨링 먼저 적용 (원본 그리드 보기에도 적용)
          </span>
        </label>
        {activeTransform === "upward_continuation" && (
          <Field label="상방연속 고도 (m)">
            <input
              type="number"
              style={inputStyle}
              value={transformExtraParams.continuation_height_m}
              onChange={(e) => setTransformExtraParams((p) => ({ ...p, continuation_height_m: parseFloat(e.target.value) }))}
            />
          </Field>
        )}
        {activeTransform === "detrend" && (
          <Field label="추세면 차수 (1=평면, 2=2차, 3=3차 - 높을수록 더 복잡한 배경을 뺌)">
            <select
              style={inputStyle}
              value={transformExtraParams.trend_order}
              onChange={(e) => setTransformExtraParams((p) => ({ ...p, trend_order: parseInt(e.target.value, 10) }))}
            >
              <option value={1}>1차 (평면)</option>
              <option value={2}>2차</option>
              <option value={3}>3차</option>
            </select>
          </Field>
        )}
        {(activeTransform === "microlevel" || transformExtraParams.microlevel_pre_apply) && (
          <>
            <Field
              label={
                <span title="측선별 레벨 오차가 만드는 줄무늬는 파장 하나가 아니라 여러 파장에 걸쳐 있습니다. 측선마다 값이 다르면 측선 간격 파장에, 전진/후진이 번갈아 어긋나면 측선 간격의 2배 파장에, 그 아래로도 배음이 계속 생깁니다. 디코러게이션(고역통과)은 차단 파장보다 짧은 측선직각 성분을 한꺼번에 제거하므로 이 전부를 잡습니다. 노치는 측선 간격 한 파장만 좁게 제거하므로 지질 신호는 더 잘 보존하지만 줄무늬는 일부만 사라집니다.">
                  필터 방식 ⓘ
                </span>
              }
            >
              <select
                style={inputStyle}
                value={transformExtraParams.microlevel_mode}
                onChange={(e) => setTransformExtraParams((p) => ({ ...p, microlevel_mode: e.target.value }))}
              >
                <option value="decorrugation">디코러게이션 (고역통과, 권장)</option>
                <option value="notch">노치 (측선 간격 한 파장만)</option>
              </select>
            </Field>
            <Field label={`보정 강도: ${transformExtraParams.microlevel_strength}`}>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={transformExtraParams.microlevel_strength}
                onChange={(e) => setTransformExtraParams((p) => ({ ...p, microlevel_strength: parseFloat(e.target.value) }))}
              />
            </Field>
            <Field label="방향 허용오차 (deg) — 측선 직각방향 기준">
              <input
                type="number"
                style={inputStyle}
                value={transformExtraParams.microlevel_angle_tolerance_deg}
                onChange={(e) => setTransformExtraParams((p) => ({ ...p, microlevel_angle_tolerance_deg: parseFloat(e.target.value) }))}
              />
            </Field>
            {transformExtraParams.microlevel_mode === "decorrugation" ? (
              <Field
                label={
                  <span title="측선 간격의 몇 배보다 짧은 측선직각 성분을 제거할지 정합니다. UAV 자력탐사 가이드라인의 통상값은 측선 간격의 4배입니다. 키우면 더 긴 파장까지 제거되어 줄무늬는 더 잘 사라지지만, 측선과 나란한 실제 지질 신호도 함께 깎입니다.">
                    차단 파장 (측선 간격 배수) — 이보다 짧은 측선직각 성분 제거 ⓘ
                  </span>
                }
              >
                <input
                  type="number"
                  step="0.5"
                  min="1.5"
                  max="20"
                  style={inputStyle}
                  value={transformExtraParams.microlevel_cutoff_factor}
                  onChange={(e) => setTransformExtraParams((p) => ({ ...p, microlevel_cutoff_factor: parseFloat(e.target.value) }))}
                />
              </Field>
            ) : (
              <Field
                label={
                  <span title="노치의 폭입니다. 측선 간격 파장을 중심으로 이 배수만큼 위아래 대역을 완화합니다.">
                    파장 대역폭 배수 (측선 간격 기준, 클수록 더 넓은 파장대를 완화) ⓘ
                  </span>
                }
              >
                <input
                  type="number"
                  step="0.1"
                  min="1.01"
                  style={inputStyle}
                  value={transformExtraParams.microlevel_wavelength_factor}
                  onChange={(e) => setTransformExtraParams((p) => ({ ...p, microlevel_wavelength_factor: parseFloat(e.target.value) }))}
                />
              </Field>
            )}
          </>
        )}
        <ExportOptions projectId={projectId} ready={!!processSummary} projectEpsg={processSummary?.utm_epsg} onError={handleError} />
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={exportGeotiffColored} onChange={(e) => setExportGeotiffColored(e.target.checked)} />
          <span>컬러 GeoTIFF로 저장 (화면에 보이는 색상 그대로, 값 아님)</span>
        </label>
        <button
          style={{ ...buttonStyle, background: "white", color: "#a9631f" }}
          disabled={exportingGeotiff || !processSummary}
          onClick={() => handleExportGeotiff()}
        >
          {exportingGeotiff
            ? "내보내는 중..."
            : `현재 결과(${activeTransform === "none" ? "그리드" : activeTransform.toUpperCase()})를 GeoTIFF로 저장`}
        </button>
        <div style={{ color: "#ab9a78" }}>
          {exportGeotiffColored
            ? "화면에 보이는 컬러맵/힐쉐이드가 그대로 이미지로 저장됩니다 (Google Earth 등에서 바로 볼 때 적합, 값 재분석 불가)."
            : "실제 값(nT 등)이 그대로 저장되어 Oasis Montaj/QGIS/ArcGIS 등에서 다시 열 수 있습니다."}
        </div>
        <button style={{ ...buttonStyle, background: "white", color: "#a9631f" }} disabled={exportingXyz || !processSummary} onClick={() => handleExportXyz()}>
          {exportingXyz ? "내보내는 중..." : `현재 결과(${activeTransform === "none" ? "그리드" : activeTransform.toUpperCase()})를 XYZ(텍스트)로 저장`}
        </button>
        <div style={{ color: "#ab9a78" }}>경도·위도·값 3열의 공백 구분 텍스트 파일 — GeoTIFF를 지원하지 않는 다른 프로그램에서도 열람 가능.</div>
        <button style={{ ...buttonStyle, background: "white", color: "#a9631f" }} disabled={exportingGrd || !processSummary} onClick={() => handleExportGrd()}>
          {exportingGrd ? "내보내는 중..." : `현재 결과(${activeTransform === "none" ? "그리드" : activeTransform.toUpperCase()})를 Surfer GRD로 저장`}
        </button>
        <div style={{ color: "#ab9a78" }}>Surfer 6 Binary Grid(.grd, DSBB) 형식 — Golden Software Surfer에서 바로 열람 가능.</div>
        <button style={{ ...buttonStyle, background: "white", color: "#a9631f" }} disabled={!processSummary} onClick={() => handleExportGpkg()}>
          {`현재 결과(${activeTransform === "none" ? "그리드" : activeTransform.toUpperCase()})를 GeoPackage 격자로 저장`}
        </button>
        <div style={{ color: "#ab9a78" }}>값이 그대로 든 격자(.gpkg) — 점 레이어와 한 종류의 파일로 GIS에 넘길 때.</div>
        <button style={{ ...buttonStyle, background: "white", color: "#a9631f" }} disabled={exportingPointsCsv || !processSummary} onClick={() => handleExportPointsCsv()}>
          {exportingPointsCsv ? "내보내는 중..." : "처리된 포인트 전체를 CSV로 저장"}
        </button>
      </div>
    </details>
  );
}
