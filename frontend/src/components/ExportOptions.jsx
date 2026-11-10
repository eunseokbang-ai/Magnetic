import { useState } from "react";
import * as api from "../api";

const inputStyle = { fontSize: 12, padding: "4px 6px", borderRadius: 6, border: "1px solid #ddd0b2", width: "100%", boxSizing: "border-box" };
const buttonStyle = { padding: "7px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #a9631f", background: "white", color: "#a9631f", cursor: "pointer" };

// Coordinate systems a recipient in Korea is likely to ask for. "" = the
// program's default for each format (see the note under the select).
const CRS_CHOICES = [
  ["", "기본 (프로젝트 좌표계)"],
  ["4326", "WGS84 경위도 (EPSG:4326)"],
  ["5186", "Korea 2000 중부원점 (EPSG:5186)"],
  ["5185", "Korea 2000 서부원점 (EPSG:5185)"],
  ["5187", "Korea 2000 동부원점 (EPSG:5187)"],
  ["5188", "Korea 2000 동해원점 (EPSG:5188)"],
  ["5179", "Korea 2000 UTM-K (EPSG:5179)"],
  ["custom", "직접 입력 (EPSG 번호)"],
];

/**
 * The coordinate system exports leave in, and the point-data formats that
 * go with it (Geosoft XYZ for Oasis montaj, GeoPackage for GIS). The
 * choice applies to every grid export below it as well - GeoTIFF, XYZ,
 * Surfer GRD, GeoPackage - through api.setExportCrs. A grid in another
 * CRS is resampled once, here, rather than by whatever the recipient's
 * GIS does (magcore/export_formats.py).
 */
export default function ExportOptions({ projectId, ready, projectEpsg, onError }) {
  const [choice, setChoice] = useState("");
  const [custom, setCustom] = useState("");
  const [busy, setBusy] = useState(null);

  const epsg = choice === "custom" ? (custom.trim() ? Number(custom) : null) : choice ? Number(choice) : null;

  const pick = (value, customValue = custom) => {
    setChoice(value);
    const next = value === "custom" ? (customValue.trim() ? Number(customValue) : null) : value ? Number(value) : null;
    api.setExportCrs(Number.isFinite(next) ? next : null);
  };

  const run = async (key, fn) => {
    try {
      setBusy(key);
      await fn();
    } catch (e) {
      onError && onError(e);
    } finally {
      setBusy(null);
    }
  };

  const suffix = epsg ? `_epsg${epsg}` : "";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, padding: "8px", border: "1px solid #eadfc8", borderRadius: 6, background: "#fbf7ee" }}>
      <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <span style={{ color: "#6b5c42" }}>내보내기 좌표계 (아래 모든 격자·포인트 내보내기에 적용)</span>
        <select style={inputStyle} value={choice} onChange={(e) => pick(e.target.value)}>
          {CRS_CHOICES.map(([value, label]) => (
            <option key={value} value={value}>
              {value === "" && projectEpsg ? `기본 (프로젝트 좌표계 EPSG:${projectEpsg})` : label}
            </option>
          ))}
        </select>
      </label>
      {choice === "custom" && (
        <input
          type="number"
          aria-label="EPSG 번호"
          placeholder="예: 32652"
          style={inputStyle}
          value={custom}
          onChange={(e) => {
            setCustom(e.target.value);
            pick("custom", e.target.value);
          }}
        />
      )}
      <div style={{ color: "#ab9a78" }}>
        {epsg
          ? "격자는 이 좌표계로 한 번 재배열(쌍선형)되어 나갑니다 - 같은 자기장이지만 셀 하나하나가 화면의 격자와 같지는 않습니다. XYZ와 포인트는 좌표만 바뀝니다."
          : "기본: GeoTIFF·Surfer·GeoPackage는 프로젝트 좌표계, XYZ 텍스트는 경도·위도."}
      </div>
      <button style={{ ...buttonStyle, opacity: ready && !busy ? 1 : 0.5 }} disabled={!ready || !!busy} onClick={() => run("geosoft", () => api.exportPointsGeosoftXyz(projectId, epsg, `points${suffix}.xyz`))}>
        {busy === "geosoft" ? "내보내는 중..." : "측선별 측점을 Geosoft XYZ로 저장"}
      </button>
      <div style={{ color: "#ab9a78" }}>Oasis montaj의 XYZ 가져오기 형식 — 측선마다 "Line n" 블록, 타이라인은 "Tie n".</div>
      <button style={{ ...buttonStyle, opacity: ready && !busy ? 1 : 0.5 }} disabled={!ready || !!busy} onClick={() => run("gpkg", () => api.exportPointsGpkg(projectId, epsg, `points${suffix}.gpkg`))}>
        {busy === "gpkg" ? "내보내는 중..." : "처리된 포인트 전체를 GeoPackage로 저장"}
      </button>
      <div style={{ color: "#ab9a78" }}>QGIS·ArcGIS에서 바로 열리는 점 레이어(.gpkg) — 값, 측선 번호, 제외 여부가 속성으로 들어갑니다.</div>
    </div>
  );
}
