import { useEffect, useState } from "react";

const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #d1d5db" };
const buttonStyle = {
  padding: "6px 10px",
  fontSize: 12,
  borderRadius: 6,
  border: "1px solid #2563eb",
  background: "white",
  color: "#2563eb",
  cursor: "pointer",
};

// Lets the user pre-download basemap tiles for a specific area/zoom range
// so the map background still shows something when there's no internet
// connection in the field (e.g. downloaded once at the office before
// going out to survey a known area). Downloaded tiles are served back by
// this app's own backend (see MapView's "(오프라인 캐시)" basemap options),
// so once cached the browser never needs the public tile servers again.
export default function OfflineMapPanel({ mapBounds, onEstimate, onDownload, onRefreshStatus, status, estimate, downloadResult, loading, error }) {
  const [source, setSource] = useState("osm");
  const [minLat, setMinLat] = useState("");
  const [minLon, setMinLon] = useState("");
  const [maxLat, setMaxLat] = useState("");
  const [maxLon, setMaxLon] = useState("");
  const [minZoom, setMinZoom] = useState(10);
  const [maxZoom, setMaxZoom] = useState(15);

  useEffect(() => {
    onRefreshStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const useCurrentView = () => {
    if (!mapBounds) return;
    setMinLat(mapBounds.getSouth().toFixed(5));
    setMinLon(mapBounds.getWest().toFixed(5));
    setMaxLat(mapBounds.getNorth().toFixed(5));
    setMaxLon(mapBounds.getEast().toFixed(5));
  };

  const bbox = () => ({
    source,
    min_lat: parseFloat(minLat),
    min_lon: parseFloat(minLon),
    max_lat: parseFloat(maxLat),
    max_lon: parseFloat(maxLon),
    min_zoom: parseInt(minZoom, 10),
    max_zoom: parseInt(maxZoom, 10),
  });

  const hasValidBbox = [minLat, minLon, maxLat, maxLon].every((v) => v !== "" && !Number.isNaN(parseFloat(v)));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12 }}>
      <div style={{ color: "#6b7280" }}>
        인터넷이 안 되는 현장에서도 배경지도가 보이도록, 조사 예정 지역의 지도 타일을 미리 받아 저장해둘 수 있습니다.
        한 번 받아두면 오프라인 상태에서도 아래 지도 레이어 목록의 "(오프라인 캐시)" 항목으로 계속 볼 수 있습니다.
        조사 지역처럼 좁은 범위 + 필요한 확대 단계만 받는 것을 권장합니다 (넓은 지역·높은 확대단계는 타일이 매우
        많아집니다).
      </div>

      <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <span style={{ color: "#4b5563" }}>지도 소스</span>
        <select style={inputStyle} value={source} onChange={(e) => setSource(e.target.value)}>
          <option value="osm">OpenStreetMap</option>
          <option value="esri">Esri 위성 영상</option>
        </select>
      </label>

      <button style={buttonStyle} onClick={useCurrentView} disabled={!mapBounds}>
        현재 지도 화면 범위 사용
      </button>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ color: "#4b5563" }}>남쪽 위도</span>
          <input type="number" style={inputStyle} value={minLat} onChange={(e) => setMinLat(e.target.value)} />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ color: "#4b5563" }}>서쪽 경도</span>
          <input type="number" style={inputStyle} value={minLon} onChange={(e) => setMinLon(e.target.value)} />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ color: "#4b5563" }}>북쪽 위도</span>
          <input type="number" style={inputStyle} value={maxLat} onChange={(e) => setMaxLat(e.target.value)} />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ color: "#4b5563" }}>동쪽 경도</span>
          <input type="number" style={inputStyle} value={maxLon} onChange={(e) => setMaxLon(e.target.value)} />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ color: "#4b5563" }}>최소 확대단계 (zoom)</span>
          <input type="number" min="0" max="19" style={inputStyle} value={minZoom} onChange={(e) => setMinZoom(e.target.value)} />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ color: "#4b5563" }}>최대 확대단계 (zoom)</span>
          <input type="number" min="0" max="19" style={inputStyle} value={maxZoom} onChange={(e) => setMaxZoom(e.target.value)} />
        </label>
      </div>

      <div style={{ display: "flex", gap: 6 }}>
        <button style={buttonStyle} disabled={!hasValidBbox || loading} onClick={() => onEstimate(bbox())}>
          용량 확인
        </button>
        <button
          style={{ ...buttonStyle, background: "#2563eb", color: "white" }}
          disabled={!hasValidBbox || loading || (estimate && estimate.exceeds_max)}
          onClick={() => onDownload(bbox())}
        >
          {loading ? "다운로드 중..." : "다운로드 시작"}
        </button>
      </div>

      {estimate && (
        <div
          style={{
            border: `1px solid ${estimate.exceeds_max ? "#fecaca" : "#bfdbfe"}`,
            background: estimate.exceeds_max ? "#fef2f2" : "#eff6ff",
            borderRadius: 6,
            padding: "6px 8px",
          }}
        >
          예상 타일 수: {estimate.n_tiles.toLocaleString()}개 (약 {estimate.estimated_mb.toLocaleString()} MB)
          {estimate.exceeds_max && (
            <div style={{ color: "#dc2626", marginTop: 4 }}>
              ⚠ 한 번에 받을 수 있는 최대 타일 수({estimate.max_tiles.toLocaleString()}개)를 넘습니다. 영역을 좁히거나
              확대단계 범위를 줄이세요.
            </div>
          )}
        </div>
      )}

      {error && (
        <div style={{ color: "#dc2626", background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 6, padding: "6px 8px" }}>
          {error}
        </div>
      )}

      {downloadResult && (
        <div style={{ color: "#374151" }}>
          다운로드 완료: 신규 {downloadResult.n_downloaded}개, 이미 있던 타일 {downloadResult.n_already_cached}개
          {downloadResult.n_failed > 0 && <span style={{ color: "#b45309" }}> · 실패 {downloadResult.n_failed}개</span>}
        </div>
      )}

      <hr style={{ border: "none", borderTop: "1px solid #e5e7eb", margin: "4px 0" }} />

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontWeight: 600 }}>현재 캐시 상태</span>
        <button style={{ ...buttonStyle, padding: "3px 8px" }} onClick={onRefreshStatus}>
          새로고침
        </button>
      </div>
      {status &&
        Object.entries(status).map(([key, s]) => (
          <div key={key} style={{ color: "#6b7280" }}>
            {s.label}: {s.n_tiles.toLocaleString()}개 타일 ({s.size_mb} MB)
            {s.zoom_levels.length > 0 && ` — zoom ${s.zoom_levels.join(", ")}`}
          </div>
        ))}
    </div>
  );
}
