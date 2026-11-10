import { useApp } from "../../AppContext";
import { Field, bodyStyle, buttonStyle, inputStyle, sectionStyle, summaryStyle } from "./shared";

// Workflow step 9: gridding - method, cell size, colours and shading, the
// display boundary, the confidence layer.
// One section of components/WorkflowSteps.jsx; it reads App's state with useApp().
export default function GriddingSection() {
  const {
    activeTransform, alongLineSmooth, alongLineSmoothWavelength, autoBoundaryBusy, autoBoundaryInfo,
    boundaryBufferM, boundaryFilename, boundaryMode, boundaryPolygon, confidenceLoading, confidenceOverlay,
    contourInterval, contourNLevels, gridCellSize, gridMaxDistance, gridMethod, gridOpacity, gridding,
    handleAutoBoundary, handleClearBoundary, handleExportBoundary, handleExportBoundaryGis, handleGrid,
    handleGridConfidence, handleImportBoundaryFile, handleToggleBoundaryMode, hillshade, hillshadeAltitude,
    hillshadeAzimuth, hillshadeExaggeration, inspectMode, overlay, processSummary, setAlongLineSmooth,
    setAlongLineSmoothWavelength, setBoundaryBufferM, setBoundaryFilename, setContourInterval,
    setContourNLevels, setGridCellSize, setGridMaxDistance, setGridMethod, setGridOpacity, setHillshade,
    setHillshadeAltitude, setHillshadeAzimuth, setHillshadeExaggeration, setShowConfidenceOverlay,
    setShowContours, setStretch, showConfidenceOverlay, showContours, stretch, toggleInspectMode,
  } = useApp();

  const lineSpacingM = processSummary?.line_spacing_m;

  return (
    <details id="wf-section-grid" style={sectionStyle}>
      <summary style={summaryStyle}>9. 그리딩</summary>
      <div style={bodyStyle}>
        <Field label="셀 크기 (m)">
          <input type="number" style={inputStyle} value={gridCellSize} onChange={(e) => setGridCellSize(parseFloat(e.target.value))} />
        </Field>
        <Field label="보간 방법">
          <select style={inputStyle} value={gridMethod} onChange={(e) => setGridMethod(e.target.value)}>
            <option value="nearest">원본 셀 (가장 빠름, 기본값)</option>
            <option value="linear">선형(Linear) - 보통 속도</option>
            <option value="cubic">큐빅(Cubic) - 느림</option>
            <option value="spline">스플라인 (가장 부드러움, 가장 느림)</option>
            <option value="minimum_curvature">최소곡률(Minimum Curvature) — Surfer 기본 격자화 방식과 동일한 알고리즘</option>
            <option value="boxing">Boxing — 빈 셀은 보간하지 않고, 실측값이 있는 셀만 그 평균으로 채움 (자료 없는 구간을 있는 그대로 표시)</option>
          </select>
        </Field>
        <Field label="보간 반경 (m) — 비워두면 측선 간격 기반 자동 계산">
          <input
            type="number"
            style={inputStyle}
            placeholder={processSummary?.line_spacing_m ? `자동: ${Math.round(Math.max(2 * gridCellSize, 0.6 * processSummary.line_spacing_m))}` : "자동"}
            value={gridMaxDistance ?? ""}
            onChange={(e) => setGridMaxDistance(e.target.value === "" ? null : parseFloat(e.target.value))}
          />
        </Field>
        {processSummary?.line_spacing_m && (
          <div style={{ color: "#8a7a5c" }}>추정 측선 간격: {processSummary.line_spacing_m.toFixed(1)}m</div>
        )}
        <Field
          label={
            <span title="자료 처리를 실행하면 실제 비행한 측선을 기준으로 경계가 자동 생성되어, 측선이 없는 곳까지 내삽/외삽으로 채워지는 것을 막습니다. 버퍼는 가장 바깥 측선에서 경계까지의 거리입니다. 측선 사이의 정상 간격은 자동으로 이어 붙이므로 안쪽은 빈틈 없이 채워지고, 측선 간격보다 훨씬 넓은 빈 구간(결측 측선 등)만 경계 밖으로 빠집니다.">
              표시 경계 — 측선 기준 자동 생성 (버퍼 조절 가능) ⓘ
            </span>
          }
        >
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center", marginBottom: 6 }}>
            <span style={{ color: "#6b5c42" }}>버퍼</span>
            <input
              type="number"
              min="0.1"
              step="1"
              value={boundaryBufferM}
              // Blank is a real value here - it means "auto = 측선 간격" -
              // so keep it instead of coercing to a number.
              onChange={(e) => {
                const raw = e.target.value;
                setBoundaryBufferM(raw === "" ? "" : parseFloat(raw) || 0);
              }}
              placeholder="자동"
              style={{ ...inputStyle, width: 70 }}
              title="가장 바깥 측선에서 경계선까지의 거리(m). 비워두면 측선 간격으로 자동 설정됩니다."
            />
            <span style={{ color: "#6b5c42" }}>m</span>
            <button style={buttonStyle} onClick={() => handleAutoBoundary()} disabled={autoBoundaryBusy}>
              {autoBoundaryBusy ? "생성 중..." : boundaryBufferM === "" ? "자동 버퍼로 경계 생성" : "이 버퍼로 경계 다시 생성"}
            </button>
            {boundaryBufferM !== "" && (
              <button
                style={buttonStyle}
                onClick={() => setBoundaryBufferM("")}
                disabled={autoBoundaryBusy}
                title="버퍼를 비워 측선 간격 자동값으로 되돌립니다."
              >
                자동값으로
              </button>
            )}
          </div>
          {autoBoundaryInfo && !autoBoundaryInfo.failed && (
            <div style={{ fontSize: 11, color: "#6b5c42", marginBottom: 6 }}>
              자동 경계 적용됨: 버퍼 {autoBoundaryInfo.buffer_m}m
              {autoBoundaryInfo.buffer_auto ? " (측선 간격 자동)" : ""}, 면적 {autoBoundaryInfo.area_km2}km²
              {autoBoundaryInfo.n_parts > 1 ? `, 구역 ${autoBoundaryInfo.n_parts}개` : ""}
              {(autoBoundaryInfo.warnings || []).map((w, i) => (
                <div key={i} style={{ color: "#b45309", marginTop: 2 }}>
                  ⚠ {w}
                </div>
              ))}
            </div>
          )}
          {autoBoundaryInfo?.failed && (
            <div style={{ fontSize: 11, color: "#b45309", marginBottom: 6 }}>
              ⚠ 자동 경계를 만들지 못했습니다: {autoBoundaryInfo.reason} (그리드는 기존 방식대로 표시됩니다)
            </div>
          )}
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <button
              style={{
                ...buttonStyle,
                background: boundaryMode ? "#ea580c" : "white",
                color: boundaryMode ? "white" : "#ea580c",
                borderColor: "#ea580c",
              }}
              onClick={() => handleToggleBoundaryMode()}
            >
              {boundaryMode ? "경계 그리는 중 (다각형 완성하면 자동 적용)" : "경계 그리기"}
            </button>
            {boundaryPolygon && (
              <button style={buttonStyle} onClick={() => handleClearBoundary()}>
                경계 지우기
              </button>
            )}
            {boundaryPolygon && (
              <>
                <input
                  type="text"
                  value={boundaryFilename}
                  onChange={(e) => setBoundaryFilename(e.target.value)}
                  placeholder="저장할 파일명"
                  title="다른 프로젝트에서도 같은 경계를 재사용하려면 알아보기 쉬운 이름으로 바꿀 수 있습니다"
                  style={{ ...inputStyle, width: 160 }}
                />
                <button
                  style={buttonStyle}
                  onClick={() => handleExportBoundary()}
                  title="이 앱에서 다시 불러오기 위한 형식 (아래 '경계 파일 불러오기'로 재사용)"
                >
                  JSON 저장
                </button>
                <button
                  style={buttonStyle}
                  onClick={() => handleExportBoundaryGis("shp")}
                  title="ESRI 셰이프파일(.shp/.shx/.dbf/.prj)을 zip으로 내려받습니다 - QGIS/ArcGIS에서 바로 열립니다 (WGS84 경위도)"
                >
                  SHP 저장
                </button>
                <button
                  style={buttonStyle}
                  onClick={() => handleExportBoundaryGis("kml")}
                  title="구글어스 등에서 열 수 있는 KML로 내려받습니다 (WGS84 경위도)"
                >
                  KML 저장
                </button>
              </>
            )}
            <label style={{ ...buttonStyle, display: "inline-flex", alignItems: "center", cursor: "pointer" }}>
              경계 파일 불러오기
              <input
                type="file"
                accept=".json"
                style={{ display: "none" }}
                onChange={(e) => {
                  const file = e.target.files[0];
                  if (file) handleImportBoundaryFile(file);
                  e.target.value = "";
                }}
              />
            </label>
          </div>
        </Field>
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={alongLineSmooth} onChange={(e) => setAlongLineSmooth(e.target.checked)} />
          <span
            title="측선 방향 자료 간격(수 m)이 측선 사이 간격(보통 수십~수백 m)보다 훨씬 촘촘해서, 옆 측선과는 비교할 수 없는 미세한 변화까지 그리드에 그대로 반영되면 측선과 나란한 방향의 물결/주름 무늬(코러게이션)가 생깁니다. 측선 방향으로 이 정도 파장 이하의 세부 변화를 미리 눌러주면, 어떤 보간 방법을 쓰든 주름이 크게 줄어듭니다."
          >
            측선방향 평활화 — 측선/횡측선 간격 불일치로 인한 주름(코러게이션) 억제 (기본 켜짐)
          </span>
        </label>
        {alongLineSmooth && (
          <Field
            label={
              <span title="UAV 자력탐사 가이드라인의 디코러게이션 규칙: 저역통과 파장은 타이라인 간격의 2배, 고역통과 파장은 측선 간격의 2배가 일반적입니다. 여기 자동값(측선 간격 1배)은 우리 프로그램의 그리딩 전 앤티앨리어싱 목적에 맞춘 값으로, 가이드라인 수치와 반드시 같을 필요는 없지만 참고용으로 함께 조정해볼 수 있습니다.">
                {lineSpacingM
                  ? `평활화 파장 (m) — 비워두면 추정 측선 간격(${lineSpacingM.toFixed(1)}m) 사용 ⓘ`
                  : "평활화 파장 (m) — 비워두면 측선 간격 자동 추정값 사용 ⓘ"}
              </span>
            }
          >
            <input
              type="number"
              style={inputStyle}
              placeholder={lineSpacingM ? `자동: ${lineSpacingM.toFixed(1)}` : "자동"}
              value={alongLineSmoothWavelength ?? ""}
              onChange={(e) => setAlongLineSmoothWavelength(e.target.value === "" ? null : parseFloat(e.target.value))}
            />
          </Field>
        )}
        <Field label={`그리드 불투명도: ${Math.round(gridOpacity * 100)}%`}>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={gridOpacity}
            onChange={(e) => setGridOpacity(parseFloat(e.target.value))}
          />
        </Field>
        <Field label="색상 스트레치(대비) 방식">
          <select style={inputStyle} value={stretch} onChange={(e) => setStretch(e.target.value)}>
            <option value="linear">선형 (기본값)</option>
            <option value="equalize">히스토그램 균등화 — 값 분포에 맞춰 대비 자동 강조 (극단값에 덜 묻힘)</option>
            <option value="normal">정규분포 — 평균·표준편차 기준으로 색상 배분 (값이 정규분포에 가까울 때 적합)</option>
          </select>
        </Field>
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={hillshade} onChange={(e) => setHillshade(e.target.checked)} />
          <span style={{ color: "#6b5c42" }}>힐쉐이드 효과 (음영기복 - 값 변화를 입체감 있게 강조, Geosoft 스타일)</span>
        </label>
        {hillshade && (
          <>
            <Field label={`광원 방위각: ${hillshadeAzimuth}°`}>
              <input type="range" min="0" max="360" step="5" value={hillshadeAzimuth} onChange={(e) => setHillshadeAzimuth(parseFloat(e.target.value))} />
            </Field>
            <Field label={`광원 고도각: ${hillshadeAltitude}°`}>
              <input type="range" min="5" max="90" step="5" value={hillshadeAltitude} onChange={(e) => setHillshadeAltitude(parseFloat(e.target.value))} />
            </Field>
            <Field label={`강조 정도: ${hillshadeExaggeration}`}>
              <input type="range" min="0.5" max="20" step="0.5" value={hillshadeExaggeration} onChange={(e) => setHillshadeExaggeration(parseFloat(e.target.value))} />
            </Field>
          </>
        )}
        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={showContours} onChange={(e) => setShowContours(e.target.checked)} />
          <span style={{ color: "#6b5c42" }}>등고선(등자력선) 표시 — 격자를 만들 때 항상 계산되며, 이 체크는 보이기만 켜고 끕니다</span>
        </label>
        {(
          <>
            <Field label="등고선 간격 (nT) — 비워두면 아래 개수로 자동 분할 (바꾸면 그리드 생성을 다시)">
              <input
                type="number"
                style={inputStyle}
                value={contourInterval ?? ""}
                placeholder="자동"
                onChange={(e) => setContourInterval(e.target.value === "" ? null : parseFloat(e.target.value))}
              />
            </Field>
            <Field label="자동 등고선 개수 (간격을 비워둔 경우)">
              <input type="number" style={inputStyle} value={contourNLevels} onChange={(e) => setContourNLevels(parseInt(e.target.value, 10))} />
            </Field>
          </>
        )}
        <button style={buttonStyle} disabled={gridding || !processSummary} onClick={() => handleGrid()}>
          {gridding ? "그리딩 중..." : "그리드 생성"}
        </button>
        <button
          style={{
            ...buttonStyle,
            background: inspectMode ? "#a9631f" : "white",
            color: inspectMode ? "white" : "#a9631f",
          }}
          disabled={!overlay}
          onClick={() => toggleInspectMode()}
          title="켜면 지도를 클릭할 때마다 그 지점의 값(nT)이 지도 위에 표시됩니다. 여러 지점을 계속 클릭해 동시에 비교할 수 있고, 이 버튼을 다시 누르면 전부 지워집니다."
        >
          {inspectMode ? "지점값 확인 중 (클릭하면 종료)" : "지점값 확인"}
        </button>
        <button
          style={{ ...buttonStyle, background: "white", color: "#4a3d28", border: "1px solid #ddd0b2" }}
          disabled={!processSummary || confidenceLoading}
          onClick={() => handleGridConfidence()}
          title="자료점으로부터의 거리(같은 그리드 빈 공간 채우기 기준)를 바탕으로, 각 셀이 실제 자료에 얼마나 가까이 구속되어 있는지(1=자료점 바로 위, 0=보간 한계 지점)를 색으로 보여줍니다. 붉은/노란 구간은 보간에 크게 의존한 값이니 해석 시 주의하세요."
        >
          {confidenceLoading ? "신뢰도 계산 중..." : activeTransform && activeTransform !== "none" ? `신뢰도 레이어 계산 (지금 보는 파생 그리드: ${activeTransform})` : "격자 신뢰도 레이어 계산"}
        </button>
        {confidenceOverlay && (
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={showConfidenceOverlay} onChange={(e) => setShowConfidenceOverlay(e.target.checked)} />
            <span style={{ color: "#6b5c42" }}>지도에 신뢰도 레이어 표시 (초록=신뢰 높음, 빨강=보간 의존)</span>
          </label>
        )}
        {confidenceOverlay?.low_fraction != null && (
          <div style={{ color: "#6b5c42" }}>
            {confidenceOverlay.transform ? `파생 그리드(${confidenceOverlay.transform})` : "그리드"}에서 자료보다 보간에 더 기대는 셀: {(confidenceOverlay.low_fraction * 100).toFixed(1)}%
            {confidenceOverlay.base_low_fraction != null && ` (원 그리드 ${(confidenceOverlay.base_low_fraction * 100).toFixed(1)}%)`}
          </div>
        )}
      </div>
    </details>
  );
}
