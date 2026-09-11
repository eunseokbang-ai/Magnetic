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
}) {
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
        좌표계가 내장된 GeoTIFF(지질도 등)를 업로드하면 배경지도 위에 겹쳐 표시됩니다. 목록 아래쪽이 위 레이어입니다.
      </div>
      <input
        type="file"
        accept=".tif,.tiff"
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

      {onRegisterTileFolder && (
        <div style={{ borderTop: "1px solid #e6dac0", paddingTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ fontSize: 12, color: "#4a3d28" }}>
            <b>대용량 정사영상 (수십 GB급)</b>은 직접 업로드하면 렌더링이 매우 오래 걸립니다. 대신 QGIS의 "래스터 → 타일
            생성(XYZ)" 또는 GDAL의 <code>gdal2tiles.py</code>로 미리 타일링한 폴더(0, 1, 2... 확대단계 숫자 이름의
            하위 폴더 구조)를 아래에 경로로 지정하면, 업로드 없이 그 폴더를 직접 읽어 즉시 배경 레이어로 표시합니다
            (이 프로그램과 같은 컴퓨터에 폴더가 있어야 합니다).
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
