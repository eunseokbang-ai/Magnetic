import { useState } from "react";

const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #ddd0b2" };
const buttonStyle = {
  padding: "6px 10px",
  fontSize: 12,
  borderRadius: 6,
  border: "1px solid #a9631f",
  background: "white",
  color: "#a9631f",
  cursor: "pointer",
};

export default function LayerManager({
  layers,
  onUpload,
  onToggleVisible,
  onSetOpacity,
  onMove,
  onRemove,
  uploading,
  error,
  onRegisterTileFolder,
  tileFolderRegistering,
  tileFolderError,
  onPickTileFolder,
  onRegisterRaster,
  onPickRasterFile,
}) {
  const [rasterPath, setRasterPath] = useState("");
  const [rasterLabel, setRasterLabel] = useState("");
  const [pickingRaster, setPickingRaster] = useState(false);
  const [fileName, setFileName] = useState("");
  const [folderPath, setFolderPath] = useState("");
  const [folderLabel, setFolderLabel] = useState("");
  const [folderScheme, setFolderScheme] = useState("auto");
  const [picking, setPicking] = useState(false);
  const [pickError, setPickError] = useState(null);

  const handlePickFolder = async () => {
    if (!onPickTileFolder) return;
    try {
      setPickError(null);
      setPicking(true);
      const picked = await onPickTileFolder();
      if (picked) setFolderPath(picked);
    } catch (e) {
      setPickError(e.message || String(e));
    } finally {
      setPicking(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ fontSize: 12, color: "#4a3d28" }}>
        좌표계가 내장된 GeoTIFF(지질도, 항공 자력도 등)를 업로드하면 배경지도 위에 겹쳐 표시됩니다. 시추 위치·시료 채취 지점·조사 경계 같은
        점·선·면 자료는 Google Earth의 KML/KMZ, GeoJSON, 또는 위도·경도 열(lat/lon, 위도/경도)이 있는 CSV로 올리면 이름표가 붙은 표시로
        그려집니다(위경도 WGS84만, 투영 좌표는 불가). 목록 아래쪽이 위 레이어입니다.
      </div>
      <input
        type="file"
        accept=".tif,.tiff,.kml,.kmz,.geojson,.json,.csv,.txt"
        style={inputStyle}
        disabled={uploading}
        onChange={async (e) => {
          const f = e.target.files[0];
          if (!f) return;
          setFileName(f.name);
          await onUpload(f);
          e.target.value = "";
        }}
      />
      {fileName && uploading && <div style={{ fontSize: 12, color: "#8a7a5c" }}>{fileName} 업로드/렌더링 중...</div>}
      {error && <div style={{ fontSize: 12, color: "#dc2626" }}>{error}</div>}

      {onRegisterRaster && (
        <div style={{ borderTop: "1px solid #e6dac0", paddingTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ fontSize: 12, color: "#4a3d28" }}>
            <b>대용량 GeoTIFF (수십 GB급 정사영상)</b>는 업로드하지 말고 파일 경로를 지정하세요. 이 프로그램이 파일을 직접
            읽어 화면에 필요한 타일만 그때그때 만들어 보여 줍니다(Metashape 등이 내보낸 GeoTIFF는 내부 오버뷰가 있어
            바로 빠릅니다. 오버뷰가 없는 파일은 <code>gdaladdo -r average 파일.tif 2 4 8 16 32 64 128 256</code>으로 먼저
            만드세요). Metashape 프로젝트의 <code>*.files\…\orthomosaic</code> 폴더는 Metashape 내부 형식이라 쓸 수
            없고, 정사영상을 GeoTIFF로 내보낸 파일을 지정합니다. 이 프로그램과 같은 컴퓨터에 파일이 있어야 합니다.
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <input
              type="text"
              placeholder="GeoTIFF 전체 경로 (예: D:\Andong20261028\Andong_ortho.tif)"
              style={{ ...inputStyle, flex: 1 }}
              value={rasterPath}
              onChange={(e) => setRasterPath(e.target.value)}
              disabled={tileFolderRegistering}
            />
            {onPickRasterFile && (
              <button
                type="button"
                style={{ ...buttonStyle, padding: "4px 10px" }}
                onClick={async () => {
                  try {
                    setPickingRaster(true);
                    const picked = await onPickRasterFile();
                    if (picked) setRasterPath(picked);
                  } catch (e) {
                    setPickError(e.message || String(e));
                  } finally {
                    setPickingRaster(false);
                  }
                }}
                disabled={tileFolderRegistering || pickingRaster}
                title="파일 선택 창 열기 (이 프로그램을 실행 중인 컴퓨터의 화면에 뜹니다)"
              >
                📄
              </button>
            )}
          </div>
          <input
            type="text"
            placeholder="레이어 이름 (선택, 비우면 파일명 사용)"
            style={inputStyle}
            value={rasterLabel}
            onChange={(e) => setRasterLabel(e.target.value)}
            disabled={tileFolderRegistering}
          />
          <button
            style={buttonStyle}
            disabled={!rasterPath.trim() || tileFolderRegistering}
            onClick={async () => {
              await onRegisterRaster(rasterPath.trim(), rasterLabel.trim() || null);
              setRasterPath("");
              setRasterLabel("");
            }}
          >
            {tileFolderRegistering ? "연동 중..." : "GeoTIFF 연동"}
          </button>
        </div>
      )}

      {onRegisterTileFolder && (
        <div style={{ borderTop: "1px solid #e6dac0", paddingTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ fontSize: 12, color: "#4a3d28" }}>
            <b>이미 타일로 만든 폴더</b>(QGIS의 "래스터 → 타일 생성(XYZ)", GDAL의 <code>gdal2tiles.py</code>, Metashape의
            "Export Tiles → Google Map Tiles"로 만든 0, 1, 2... 확대단계 숫자 이름의 하위 폴더 구조)는 아래에 경로로 지정하면
            업로드 없이 그 폴더를 직접 읽어 즉시 배경 레이어로 표시합니다 (이 프로그램과 같은 컴퓨터에 폴더가 있어야 합니다).
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <input
              type="text"
              placeholder="타일 폴더 전체 경로 (예: D:\survey\ortho_tiles)"
              style={{ ...inputStyle, flex: 1 }}
              value={folderPath}
              onChange={(e) => setFolderPath(e.target.value)}
              disabled={tileFolderRegistering}
            />
            {onPickTileFolder && (
              <button
                type="button"
                style={{ ...buttonStyle, padding: "4px 10px" }}
                onClick={handlePickFolder}
                disabled={tileFolderRegistering || picking}
                title="폴더 선택 창 열기 (이 프로그램을 실행 중인 컴퓨터의 화면에 뜹니다)"
              >
                📁
              </button>
            )}
          </div>
          {pickError && <div style={{ fontSize: 12, color: "#dc2626" }}>{pickError}</div>}
          <input
            type="text"
            placeholder="레이어 이름 (선택, 비우면 폴더명 사용)"
            style={inputStyle}
            value={folderLabel}
            onChange={(e) => setFolderLabel(e.target.value)}
            disabled={tileFolderRegistering}
          />
          <select style={inputStyle} value={folderScheme} onChange={(e) => setFolderScheme(e.target.value)} disabled={tileFolderRegistering}>
            <option value="auto">타일 행 순서 자동 감지 (권장)</option>
            <option value="xyz">XYZ (행 0 = 북쪽)</option>
            <option value="tms">TMS (행 0 = 남쪽, GDAL 기본 출력)</option>
          </select>
          <button
            style={buttonStyle}
            disabled={!folderPath.trim() || tileFolderRegistering}
            onClick={async () => {
              await onRegisterTileFolder(folderPath.trim(), folderLabel.trim() || null, folderScheme);
              setFolderPath("");
              setFolderLabel("");
            }}
          >
            {tileFolderRegistering ? "연동 중..." : "폴더 연동"}
          </button>
          {tileFolderError && <div style={{ fontSize: 12, color: "#dc2626" }}>{tileFolderError}</div>}
        </div>
      )}

      {layers.length === 0 ? (
        <div style={{ fontSize: 12, color: "#ab9a78" }}>추가된 레이어가 없습니다.</div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {layers.map((l, idx) => (
            <div key={l.id} style={{ border: "1px solid #e6dac0", borderRadius: 6, padding: 6, fontSize: 12 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                <input type="checkbox" checked={l.visible} onChange={(e) => onToggleVisible(l.id, e.target.checked)} />
                <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={l.name}>
                  {l.name}
                  {l.type === "raster" && (
                    <span style={{ color: "#8a7a5c" }}> (GeoTIFF {l.size_gb} GB, {l.res_m} m/px{l.has_overviews ? "" : ", 오버뷰 없음: 느림"})</span>
                  )}
                  {l.type === "vector" && (
                    <span style={{ color: "#8a7a5c" }}>
                      {" "}({[l.n_points ? `점 ${l.n_points}` : null, l.n_lines ? `선 ${l.n_lines}` : null, l.n_polygons ? `면 ${l.n_polygons}` : null].filter(Boolean).join(", ")})
                    </span>
                  )}
                </span>
                <button
                  disabled={idx === layers.length - 1}
                  onClick={() => onMove(l.id, "up")}
                  style={{ border: "none", background: "none", cursor: idx === layers.length - 1 ? "default" : "pointer", opacity: idx === layers.length - 1 ? 0.3 : 1 }}
                  title="위로"
                >
                  ▲
                </button>
                <button
                  disabled={idx === 0}
                  onClick={() => onMove(l.id, "down")}
                  style={{ border: "none", background: "none", cursor: idx === 0 ? "default" : "pointer", opacity: idx === 0 ? 0.3 : 1 }}
                  title="아래로"
                >
                  ▼
                </button>
                <button onClick={() => onRemove(l.id)} style={{ border: "none", background: "none", cursor: "pointer", color: "#dc2626" }} title="삭제">
                  ✕
                </button>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ color: "#8a7a5c" }}>불투명도</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={l.opacity}
                  onChange={(e) => onSetOpacity(l.id, parseFloat(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={{ color: "#8a7a5c", width: 32, textAlign: "right" }}>{Math.round(l.opacity * 100)}%</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
