import { useState } from "react";

const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #d1d5db" };

export default function LayerManager({ layers, onUpload, onToggleVisible, onSetOpacity, onMove, onRemove, uploading, error }) {
  const [fileName, setFileName] = useState("");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ fontSize: 12, color: "#374151" }}>
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
      {fileName && uploading && <div style={{ fontSize: 12, color: "#6b7280" }}>{fileName} 업로드/렌더링 중...</div>}
      {error && <div style={{ fontSize: 12, color: "#dc2626" }}>{error}</div>}

      {layers.length === 0 ? (
        <div style={{ fontSize: 12, color: "#9ca3af" }}>추가된 레이어가 없습니다.</div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {layers.map((l, idx) => (
            <div key={l.id} style={{ border: "1px solid #e5e7eb", borderRadius: 6, padding: 6, fontSize: 12 }}>
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
                <span style={{ color: "#6b7280" }}>불투명도</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={l.opacity}
                  onChange={(e) => onSetOpacity(l.id, parseFloat(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={{ color: "#6b7280", width: 32, textAlign: "right" }}>{Math.round(l.opacity * 100)}%</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
