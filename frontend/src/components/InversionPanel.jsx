import { useState } from "react";

const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #ddd0b2" };
const rowStyle = { display: "flex", flexDirection: "column", gap: 2, marginBottom: 6 };
const labelStyle = { fontSize: 11, color: "#4a3d28" };
const btnStyle = { padding: "6px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #a9631f", background: "#a9631f", color: "white", cursor: "pointer" };
const btnStyleAlt = { ...btnStyle, background: "white", color: "#a9631f" };
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
  onExportInversion,
  onExportInversionCsv,
  onImportInversion,
  geologyUnits,
  geologyDrawMode,
  onToggleGeologyDrawMode,
  onUpdateGeologyUnit,
  onDeleteGeologyUnit,
}) {
  const [demFileName, setDemFileName] = useState("");
  const nLayers = summary?.n_layers ?? params.n_layers ?? 8;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ fontSize: 12, color: "#4a3d28" }}>
        관측 자력이상으로 지하 자화율(SI) 분포를 추정하는 실험적 3차원 역산입니다. 아래에서 지질도 등 given information(사전
        지질정보)을 선택적으로 제공하면 그 정보를 반영해 더 나은 결과를 얻을 수 있습니다. 측선 자료 처리를 먼저 완료해야 실행할
        수 있습니다.
      </div>

      <div style={{ border: "1px solid #e6dac0", borderRadius: 6, padding: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>이전 역산 결과 불러오기</div>
        <div style={{ fontSize: 11, color: "#8a7a5c", marginBottom: 6 }}>
          내보낸 .npz 파일을 불러오면 처음부터 다시 역산을 돌리지 않아도 바로 단면/3D 뷰를 확인할 수 있습니다.
        </div>
        <input
          type="file"
          accept=".npz"
          style={inputStyle}
          onChange={async (e) => {
            const f = e.target.files[0];
            if (!f) return;
            await onImportInversion(f);
            e.target.value = "";
          }}
        />
      </div>

      <div style={{ border: "1px solid #e6dac0", borderRadius: 6, padding: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>지형(DEM)</div>
        <div style={{ fontSize: 11, color: "#8a7a5c", marginBottom: 6 }}>
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
        {demUploading && <div style={{ fontSize: 11, color: "#8a7a5c", marginTop: 4 }}>{demFileName} 업로드 중...</div>}
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
        {demStatus && (
          <div title="대부분의 DEM은 정표고(orthometric, 지오이드 기준)이고 드론 GPS 고도는 타원체고(ellipsoidal, WGS84 기준)입니다 - 두 기준이 다르면(한국은 대략 +25m 정도 차이) 지형과 비행고도가 서로 다른 기준면에 놓여 심도가 그만큼 오차가 생깁니다. 이 지역의 지오이드고(N, m)를 알고 있다면 입력하세요 (h_타원체 = H_정표고 + N).">
            <NumberField
              label="DEM 지오이드 보정 (m, 선택)"
              value={params.dem_geoid_offset_m ?? 0}
              step="1"
              onChange={(v) => setParams((p) => ({ ...p, dem_geoid_offset_m: v }))}
            />
          </div>
        )}
      </div>

      <div style={{ border: "1px solid #e6dac0", borderRadius: 6, padding: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>지질 정보 (선택 — given information)</div>
        <div style={{ fontSize: 11, color: "#8a7a5c", marginBottom: 6 }}>
          지질도 래스터가 있다면 위 "12. 참조 레이어"에 업로드해 지도에 띄운 뒤, 아래 버튼으로 그 위에 암상 경계를 폴리곤으로
          그려 번호를 매기고 대자율(SI)을 입력하세요. 역산이 각 블록 내부에서는 0이 아니라 입력한 대자율 값을 향해 수렴하도록
          유도되어(참조모델), 정보가 없을 때보다 더 지질학적으로 그럴듯한 결과를 얻을 확률이 높아집니다. 단면 정보가 없으므로
          평면 경계가 깊이 방향으로 그대로 이어진다고 가정합니다(2.5D 근사).
        </div>
        <button
          style={{ ...(geologyDrawMode ? btnStyle : btnStyleAlt), width: "100%", marginBottom: 8 }}
          disabled={!ready}
          onClick={onToggleGeologyDrawMode}
        >
          {geologyDrawMode ? "지도에 블록 경계를 그려주세요 (취소하려면 다시 클릭)" : "+ 지질 블록 그리기"}
        </button>
        {geologyUnits && geologyUnits.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {geologyUnits.map((u) => (
              <div key={u.id} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11 }}>
                <span style={{ color: "#ab9a78", width: 18 }}>#{u.id}</span>
                <input
                  style={{ ...inputStyle, flex: 2 }}
                  value={u.name}
                  onChange={(e) => onUpdateGeologyUnit(u.id, { name: e.target.value })}
                  placeholder="암상명"
                />
                <input
                  type="number"
                  style={{ ...inputStyle, flex: 1 }}
                  value={u.susceptibility_si}
                  step="0.001"
                  min="0"
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    if (!Number.isNaN(v)) onUpdateGeologyUnit(u.id, { susceptibility_si: v });
                  }}
                  title="대자율 (SI)"
                />
                <button
                  onClick={() => onDeleteGeologyUnit(u.id)}
                  style={{ border: "none", background: "none", color: "#dc2626", cursor: "pointer", fontSize: 12 }}
                  title="이 블록 삭제"
                >
                  ✕
                </button>
              </div>
            ))}
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, marginTop: 4 }}>
              <input
                type="checkbox"
                checked={params.use_geology_reference !== false}
                onChange={(e) => setParams((p) => ({ ...p, use_geology_reference: e.target.checked }))}
              />
              역산에 위 지질 정보 반영 (해제하면 지질 정보 없이 실행해 비교 가능)
            </label>
          </div>
        )}
      </div>

      <div style={{ border: "1px solid #e6dac0", borderRadius: 6, padding: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>메쉬/역산 파라미터</div>
        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, marginBottom: 8 }}>
          <input type="checkbox" checked={autoParams} onChange={(e) => setAutoParams(e.target.checked)} />
          격자 크기·탐사심도·레이어 수 자동 설정 (취득 자료 기반)
        </label>
        {!autoParams && (
          <>
            <NumberField label="관측/메쉬 격자 크기 (m)" value={params.obs_cell_size_m} step="5" onChange={(v) => setParams((p) => ({ ...p, obs_cell_size_m: v }))} />
            <NumberField label="탐사 심도 (m)" value={params.depth_extent_m} step="10" onChange={(v) => setParams((p) => ({ ...p, depth_extent_m: v }))} />
            <NumberField label="깊이 레이어 수" value={params.n_layers} step="1" min="1" max="80" onChange={(v) => setParams((p) => ({ ...p, n_layers: v }))} />
            <NumberField
              label="깊이별 셀 두께 증가율 (1.0=균일, 클수록 표층은 얇고 심부는 두꺼움)"
              value={params.depth_growth_factor ?? 1.15}
              step="0.05"
              min="1"
              max="2"
              onChange={(v) => setParams((p) => ({ ...p, depth_growth_factor: v }))}
            />
          </>
        )}
        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, marginBottom: 8 }}>
          <input
            type="checkbox"
            checked={params.assumed_noise_nt != null}
            onChange={(e) => setParams((p) => ({ ...p, assumed_noise_nt: e.target.checked ? 2.0 : null }))}
          />
          정규화 강도 자동 선택 (discrepancy principle - 목표 잔차가 예상 잡음수준에 가깝도록)
        </label>
        {params.assumed_noise_nt != null ? (
          <NumberField
            label="예상 관측 잡음수준 (nT) - 클수록 더 완만한 결과"
            value={params.assumed_noise_nt}
            step="0.5"
            min="0.01"
            onChange={(v) => setParams((p) => ({ ...p, assumed_noise_nt: v }))}
          />
        ) : (
          <NumberField
            label="정규화 강도 (클수록 완만/작을수록 뾰족)"
            value={params.regularization_strength}
            step="0.5"
            onChange={(v) => setParams((p) => ({ ...p, regularization_strength: v }))}
          />
        )}
        <NumberField label="IRLS 반복 횟수 (덩어리화)" value={params.n_irls_iterations} step="1" min="1" max="30" onChange={(v) => setParams((p) => ({ ...p, n_irls_iterations: v }))} />
        <div style={{ fontSize: 10, color: "#ab9a78", marginBottom: 8 }}>
          격자를 촘촘하게/레이어를 많이 설정할수록 더 상세하지만 실행 시간이 길어집니다(최대 1~2분 정도).
        </div>
        <button style={{ ...btnStyle, width: "100%", opacity: ready && !running ? 1 : 0.5 }} disabled={!ready || running} onClick={onRun}>
          {running ? "역산 실행 중... (최대 1~2분)" : "3차원 역산 실행"}
        </button>
        {error && <div style={{ fontSize: 11, color: "#dc2626", marginTop: 6 }}>{error}</div>}
      </div>

      {summary && (
        <div style={{ border: "1px solid #e6dac0", borderRadius: 6, padding: 8, fontSize: 11, color: "#4a3d28" }}>
          <div>관측점: {summary.n_obs} / 활성 셀: {summary.n_active_cells}</div>
          <div>RMS 잔차: {summary.rms_misfit_nt?.toFixed(1)} nT</div>
          <div>
            자화율(SI): 최대 {summary.susceptibility_stats?.max?.toFixed(4)} / 평균 {summary.susceptibility_stats?.mean?.toFixed(4)}
          </div>
          <div>고도 범위: {summary.elevation_range_m?.[0]?.toFixed(0)} ~ {summary.elevation_range_m?.[1]?.toFixed(0)} m</div>
          <div>DEM 사용: {summary.used_dem ? "예" : "아니오 (GPS 등고비행 추정)"}</div>
          {summary.n_geology_units > 0 && (
            <div>
              지질 정보 반영: {summary.geology_reference_used ? "예" : "아니오 (해제됨)"}
              {summary.geology_reference_used &&
                summary.geology_reference_coverage != null &&
                ` (활성 셀의 ${(summary.geology_reference_coverage * 100).toFixed(0)}%가 지질 블록 안에 있음)`}
            </div>
          )}
          {summary.layer_thickness_m?.length > 1 && summary.depth_growth_factor != null && (
            <div>
              레이어 두께: 표층 {summary.layer_thickness_m[0]?.toFixed(1)} m → 최심부 {summary.layer_thickness_m.at(-1)?.toFixed(1)} m
              (증가율 ×{summary.depth_growth_factor.toFixed(2)})
            </div>
          )}
          {summary.auto_params && (
            <div style={{ marginTop: 4, paddingTop: 4, borderTop: "1px dashed #ddd0b2" }}>
              <div>자동 선택된 격자 크기: {summary.obs_cell_size_m?.toFixed(1)} m</div>
              <div>자동 선택된 탐사 심도: {summary.depth_extent_m?.toFixed(0)} m ({summary.n_layers}개 레이어)</div>
              {summary.source_depth_estimate_m != null && (
                <div>스펙트럼 기반 추정 이상원 심도: 약 {summary.source_depth_estimate_m.toFixed(0)} m</div>
              )}
            </div>
          )}
          {summary.resolution_warning && (
            <div style={{ marginTop: 4, paddingTop: 4, borderTop: "1px dashed #ddd0b2", color: "#92400e", background: "#fffbeb", padding: 6, borderRadius: 4 }}>
              ⚠ {summary.resolution_warning}
            </div>
          )}
          {summary.depth_resolution_warning && (
            <div style={{ marginTop: 4, color: "#92400e", background: "#fffbeb", padding: 6, borderRadius: 4 }}>
              ⚠ {summary.depth_resolution_warning}
            </div>
          )}
          {summary.auto_regularization && (
            <div style={{ marginTop: 4 }}>
              자동 선택된 정규화 강도: {summary.regularization_strength_used?.toFixed(3)} (목표 잡음수준 {summary.assumed_noise_nt} nT 기준)
            </div>
          )}
          {summary.depth_resolution?.per_layer_relative_sensitivity && (
            <div style={{ marginTop: 6, paddingTop: 4, borderTop: "1px dashed #ddd0b2" }}>
              <div style={{ marginBottom: 3 }}>레이어별 해상도(민감도) - 낮을수록 신뢰도 낮음</div>
              {summary.depth_resolution.per_layer_relative_sensitivity.map((rel, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 1 }}>
                  <span style={{ width: 46, color: "#8a7a5c" }}>{summary.layer_elevations_m?.[i]?.toFixed(0)}m</span>
                  <div style={{ flex: 1, height: 6, background: "#f3ecd9", borderRadius: 3, overflow: "hidden" }}>
                    <div style={{ width: `${Math.max(2, rel * 100)}%`, height: "100%", background: rel < 0.05 ? "#f59e0b" : "#a9631f" }} />
                  </div>
                </div>
              ))}
            </div>
          )}
          {summary.imported && <div style={{ marginTop: 4, color: "#059669" }}>불러온 결과입니다 (역산을 다시 돌리지 않았습니다).</div>}
          <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
            <button style={{ ...btnStyleAlt, flex: 1 }} onClick={onExportInversion}>
              결과 내보내기 (.npz)
            </button>
            <button style={{ ...btnStyleAlt, flex: 1 }} onClick={onExportInversionCsv}>
              CSV로 내보내기
            </button>
          </div>
        </div>
      )}

      {summary && (
        <div style={{ border: "1px solid #e6dac0", borderRadius: 6, padding: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>수평 섹션 뷰 (Z, 깊이별 평면)</div>
          <div style={rowStyle}>
            <label style={labelStyle}>
              깊이 레이어: {sliceLayerIndex} / {nLayers - 1}
              {summary.layer_elevations_m?.[sliceLayerIndex] != null && (
                <> — 고도 약 {summary.layer_elevations_m[sliceLayerIndex].toFixed(0)} m</>
              )}
            </label>
            <input
              type="range"
              min="0"
              max={Math.max(0, nLayers - 1)}
              step="1"
              value={sliceLayerIndex}
              onChange={(e) => setSliceLayerIndex(parseInt(e.target.value, 10))}
            />
            {summary.layer_elevations_m?.length > 1 && (
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#ab9a78" }}>
                <span>{summary.layer_elevations_m[0].toFixed(0)} m (얕음)</span>
                <span>{summary.layer_elevations_m.at(-1).toFixed(0)} m (깊음)</span>
              </div>
            )}
            {summary.depth_resolution?.poorly_resolved_layers?.includes(sliceLayerIndex) && (
              <div style={{ fontSize: 10, color: "#92400e", background: "#fffbeb", padding: 4, borderRadius: 4, marginTop: 2 }}>
                ⚠ 이 레이어는 측선 배치상 민감도가 낮아(최상층 대비 5% 미만) 결과 신뢰도가 낮습니다.
              </div>
            )}
          </div>
          <RangeThresholdFields min={sliceThreshold} setMin={setSliceThreshold} max={sliceThresholdMax} setMax={setSliceThresholdMax} />
          <button style={{ ...btnStyleAlt, width: "100%" }} onClick={onShowSlice}>
            지도에 표시
          </button>
        </div>
      )}

      {summary && (
        <div style={{ border: "1px solid #e6dac0", borderRadius: 6, padding: 8 }}>
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
          {sectionLoading && <div style={{ fontSize: 11, color: "#8a7a5c" }}>단면 계산 중...</div>}
        </div>
      )}

      {summary && (
        <div style={{ border: "1px solid #e6dac0", borderRadius: 6, padding: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>3차원 이상대 뷰</div>
          <div style={{ fontSize: 10, color: "#8a7a5c", marginBottom: 4 }}>원하는 SI 범위만 지정하면 그 범위의 덩어리만 3D로 표시됩니다.</div>
          <RangeThresholdFields min={volumeThreshold} setMin={setVolumeThreshold} max={volumeThresholdMax} setMax={setVolumeThresholdMax} />
          <button style={{ ...btnStyleAlt, width: "100%" }} onClick={onOpenVolume}>
            3D 뷰 열기
          </button>
        </div>
      )}
    </div>
  );
}
