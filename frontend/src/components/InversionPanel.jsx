import { useState } from "react";

const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #d1d5db" };
const rowStyle = { display: "flex", flexDirection: "column", gap: 2, marginBottom: 6 };
const labelStyle = { fontSize: 11, color: "#374151" };
const btnStyle = { padding: "6px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #2563eb", background: "#2563eb", color: "white", cursor: "pointer" };
const btnStyleAlt = { ...btnStyle, background: "white", color: "#2563eb" };
const selectStyle = { ...inputStyle };

function NumberField({ label, value, onChange, step = "1", min, max }) {
  return (
    <div style={rowStyle}>
      <label style={labelStyle}>{label}</label>
      <input
        type="number"
        style={inputStyle}
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value === "" ? "" : parseFloat(e.target.value))}
      />
    </div>
  );
}

function RangeThresholdFields({ min, setMin, max, setMax }) {
  return (
    <div style={{ display: "flex", gap: 6 }}>
      <div style={{ flex: 1 }}>
        <NumberField label="표시 하한 (SI)" value={min} step="0.01" onChange={setMin} />
      </div>
      <div style={{ flex: 1 }}>
        <NumberField label="표시 상한 (SI, 비우면 무제한)" value={max} step="0.01" onChange={setMax} />
      </div>
    </div>
  );
}

export default function InversionPanel({
  ready,
  demStatus,
  demUploading,
  onUploadDem,
  onClearDem,
  params,
  setParams,
  autoParams,
  setAutoParams,
  onRun,
  running,
  summary,
  error,
  sliceLayerIndex,
  setSliceLayerIndex,
  sliceThreshold,
  setSliceThreshold,
  sliceThresholdMax,
  setSliceThresholdMax,
  onShowSlice,
  sectionProfile,
  setSectionProfile,
  sectionPositionFrac,
  setSectionPositionFrac,
  sectionDrawMode,
  onToggleSectionDrawMode,
  onRunFixedSection,
  sectionThreshold,
  setSectionThreshold,
  sectionThresholdMax,
  setSectionThresholdMax,
  sectionLoading,
  onOpenVolume,
  volumeThreshold,
  setVolumeThreshold,
  volumeThresholdMax,
  setVolumeThresholdMax,
}) {
  const [demFileName, setDemFileName] = useState("");
  const nLayers = summary?.n_layers ?? params.n_layers ?? 8;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ fontSize: 12, color: "#374151" }}>
        given information(사전 지질정보) 없이 관측 자력이상만으로 지하 자화율(SI) 분포를 추정하는 실험적 3차원 역산입니다.
        측선 자료 처리를 먼저 완료해야 실행할 수 있습니다.
      </div>

      <div style={{ border: "1px solid #e5e7eb", borderRadius: 6, padding: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>지형(DEM)</div>
        <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 6 }}>
          DEM을 업로드하지 않으면 등고비행(일정 고도) 가정 하에 드론 GPS 고도에서 지형을 추정합니다 (상대적 굴곡만 정확, 절대 고도는 근사치).
        </div>
        <input
          type="file"
          accept=".tif,.tiff"
          style={inputStyle}
          disabled={demUploading}
          onChange={async (e) => {
            const f = e.target.files[0];
            if (!f) return;
            setDemFileName(f.name);
            await onUploadDem(f);
            e.target.value = "";
          }}
        />
        {demUploading && <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>{demFileName} 업로드 중...</div>}
        {demStatus && (
          <div style={{ fontSize: 11, color: "#059669", marginTop: 4, display: "flex", justifyContent: "space-between" }}>
            <span>DEM 사용 중: {demStatus.name}</span>
            <button onClick={onClearDem} style={{ border: "none", background: "none", color: "#dc2626", cursor: "pointer", fontSize: 11 }}>
              제거
            </button>
          </div>
        )}
        {!demStatus && (
          <NumberField
            label="추정 비행고도 AGL (m, DEM 없을 때)"
            value={params.assumed_agl_m}
            step="5"
            onChange={(v) => setParams((p) => ({ ...p, assumed_agl_m: v }))}
          />
        )}
      </div>

      <div style={{ border: "1px solid #e5e7eb", borderRadius: 6, padding: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>메쉬/역산 파라미터</div>
        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, marginBottom: 8 }}>
          <input type="checkbox" checked={autoParams} onChange={(e) => setAutoParams(e.target.checked)} />
          격자 크기·탐사심도·레이어 수 자동 설정 (취득 자료 기반)
        </label>
        {!autoParams && (
          <>
            <NumberField label="관측/메쉬 격자 크기 (m)" value={params.obs_cell_size_m} step="5" onChange={(v) => setParams((p) => ({ ...p, obs_cell_size_m: v }))} />
            <NumberField label="탐사 심도 (m)" value={params.depth_extent_m} step="10" onChange={(v) => setParams((p) => ({ ...p, depth_extent_m: v }))} />
            <NumberField label="깊이 레이어 수" value={params.n_layers} step="1" min="1" max="50" onChange={(v) => setParams((p) => ({ ...p, n_layers: v }))} />
          </>
        )}
        <NumberField
          label="정규화 강도 (클수록 완만/작을수록 뾰족)"
          value={params.regularization_strength}
          step="0.5"
          onChange={(v) => setParams((p) => ({ ...p, regularization_strength: v }))}
        />
        <NumberField label="IRLS 반복 횟수 (덩어리화)" value={params.n_irls_iterations} step="1" min="1" max="30" onChange={(v) => setParams((p) => ({ ...p, n_irls_iterations: v }))} />
        <div style={{ fontSize: 10, color: "#9ca3af", marginBottom: 8 }}>
          격자를 촘촘하게/레이어를 많이 설정할수록 더 상세하지만 실행 시간이 길어집니다(최대 1~2분 정도).
        </div>
        <button style={{ ...btnStyle, width: "100%", opacity: ready && !running ? 1 : 0.5 }} disabled={!ready || running} onClick={onRun}>
          {running ? "역산 실행 중... (최대 1~2분)" : "3차원 역산 실행"}
        </button>
        {error && <div style={{ fontSize: 11, color: "#dc2626", marginTop: 6 }}>{error}</div>}
      </div>

      {summary && (
        <div style={{ border: "1px solid #e5e7eb", borderRadius: 6, padding: 8, fontSize: 11, color: "#374151" }}>
          <div>관측점: {summary.n_obs} / 활성 셀: {summary.n_active_cells}</div>
          <div>RMS 잔차: {summary.rms_misfit_nt?.toFixed(1)} nT</div>
          <div>
            자화율(SI): 최대 {summary.susceptibility_stats?.max?.toFixed(4)} / 평균 {summary.susceptibility_stats?.mean?.toFixed(4)}
          </div>
          <div>고도 범위: {summary.elevation_range_m?.[0]?.toFixed(0)} ~ {summary.elevation_range_m?.[1]?.toFixed(0)} m</div>
          <div>DEM 사용: {summary.used_dem ? "예" : "아니오 (GPS 등고비행 추정)"}</div>
          {summary.auto_params && (
            <div style={{ marginTop: 4, paddingTop: 4, borderTop: "1px dashed #d1d5db" }}>
              <div>자동 선택된 격자 크기: {summary.obs_cell_size_m?.toFixed(1)} m</div>
              <div>자동 선택된 탐사 심도: {summary.depth_extent_m?.toFixed(0)} m ({summary.n_layers}개 레이어)</div>
              {summary.source_depth_estimate_m != null && (
                <div>스펙트럼 기반 추정 이상원 심도: 약 {summary.source_depth_estimate_m.toFixed(0)} m</div>
              )}
            </div>
          )}
        </div>
      )}

      {summary && (
        <div style={{ border: "1px solid #e5e7eb", borderRadius: 6, padding: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>수평 섹션 뷰 (Z, 깊이별 평면)</div>
          <div style={rowStyle}>
            <label style={labelStyle}>
              깊이 레이어: {sliceLayerIndex} / {nLayers - 1}
            </label>
            <input
              type="range"
              min="0"
              max={Math.max(0, nLayers - 1)}
              step="1"
              value={sliceLayerIndex}
              onChange={(e) => setSliceLayerIndex(parseInt(e.target.value, 10))}
            />
          </div>
          <RangeThresholdFields min={sliceThreshold} setMin={setSliceThreshold} max={sliceThresholdMax} setMax={setSliceThresholdMax} />
          <button style={{ ...btnStyleAlt, width: "100%" }} onClick={onShowSlice}>
            지도에 표시
          </button>
        </div>
      )}

      {summary && (
        <div style={{ border: "1px solid #e5e7eb", borderRadius: 6, padding: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>수직 섹션 뷰</div>
          <div style={rowStyle}>
            <label style={labelStyle}>단면 방향</label>
            <select style={selectStyle} value={sectionProfile} onChange={(e) => setSectionProfile(e.target.value)}>
              <option value="custom">자유선 그리기 (지도에 직접 그리기)</option>
              <option value="ew">동서 방향 단면</option>
              <option value="ns">남북 방향 단면</option>
            </select>
          </div>

          {sectionProfile === "custom" ? (
            <button style={{ ...(sectionDrawMode ? btnStyle : btnStyleAlt), width: "100%", marginBottom: 6 }} onClick={onToggleSectionDrawMode}>
              {sectionDrawMode ? "지도에 선을 그려주세요 (취소하려면 다시 클릭)" : "프로파일 선 그리기"}
            </button>
          ) : (
            <>
              <div style={rowStyle}>
                <label style={labelStyle}>
                  위치 ({sectionProfile === "ew" ? "남↔북" : "서↔동"}): {Math.round(sectionPositionFrac * 100)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={sectionPositionFrac}
                  onChange={(e) => setSectionPositionFrac(parseFloat(e.target.value))}
                />
              </div>
              <button style={{ ...btnStyleAlt, width: "100%", marginBottom: 6 }} onClick={onRunFixedSection}>
                단면 보기
              </button>
            </>
          )}

          <RangeThresholdFields min={sectionThreshold} setMin={setSectionThreshold} max={sectionThresholdMax} setMax={setSectionThresholdMax} />
          {sectionLoading && <div style={{ fontSize: 11, color: "#6b7280" }}>단면 계산 중...</div>}
        </div>
      )}

      {summary && (
        <div style={{ border: "1px solid #e5e7eb", borderRadius: 6, padding: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>3차원 이상대 뷰</div>
          <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4 }}>원하는 SI 범위만 지정하면 그 범위의 덩어리만 3D로 표시됩니다.</div>
          <RangeThresholdFields min={volumeThreshold} setMin={setVolumeThreshold} max={volumeThresholdMax} setMax={setVolumeThresholdMax} />
          <button style={{ ...btnStyleAlt, width: "100%" }} onClick={onOpenVolume}>
            3D 뷰 열기
          </button>
        </div>
      )}
    </div>
  );
}
