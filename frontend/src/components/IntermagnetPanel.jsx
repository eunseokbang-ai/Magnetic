import { useState } from "react";

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

// Lets the user substitute a public INTERMAGNET geomagnetic observatory's
// data for a local base station that was never measured. Two independent
// ways in: upload an IAGA-2002 file downloaded by hand from
// https://intermagnet.org (always works, no server-side network access
// needed), or have this server fetch it directly from the BGS GIN web
// service (needs outbound network access from wherever the app is
// deployed - not guaranteed in every environment). Either path lands in
// the same preview -> confirm ("이 자료 사용") flow so the user sees the
// station's name/location/distance before it's applied.
export default function IntermagnetPanel({
  onUploadIaga2002,
  onFetchIntermagnet,
  onApplyIntermagnet,
  onCancelPreview,
  preview,
  loading,
  applying,
  error,
  onFetchNearestIntermagnet,
  onApplyNearestIntermagnet,
  onCancelNearestPreview,
  onExportNearestCsv,
  onShowNearestComparison,
  nearestPreview,
  nearestLoading,
  nearestApplying,
  nearestError,
  nearestHasResult,
  nearestComparisonLoading,
}) {
  const [iagaCode, setIagaCode] = useState("");
  const [startDate, setStartDate] = useState("");
  const [days, setDays] = useState(2);
  const [nearestStartDate, setNearestStartDate] = useState("");
  const [nearestDays, setNearestDays] = useState(1);
  const [nStations, setNStations] = useState(4);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, paddingTop: 8, fontSize: 12 }}>
      <div style={{ color: "#6b7280" }}>
        측정 지역의 로컬 일변화까지는 반영하지 못하지만, 태양풍 등에 의한 지역/전지구적 자기장 변화는 인근 관측소 자료로
        어느 정도 추정할 수 있습니다. INTERMAGNET(intermagnet.org)에서 직접 받은 IAGA-2002 파일을 업로드하거나, 관측소
        코드와 날짜를 입력해 서버에서 바로 받아올 수 있습니다 (서버의 아웃바운드 네트워크 정책에 따라 자동 다운로드가
        막혀 있을 수 있으며, 이 경우 파일 업로드를 이용하세요).
      </div>

      {onFetchNearestIntermagnet && (
        <div
          style={{
            border: "1px solid #e5e7eb",
            borderRadius: 6,
            padding: "8px 10px",
            display: "flex",
            flexDirection: "column",
            gap: 6,
            background: "#fafafa",
          }}
        >
          <div style={{ fontWeight: 600 }}>주변 관측소 자동 선택 (추천)</div>
          <div style={{ color: "#6b7280" }}>
            조사지역 평균 좌표 주변 동서남북 방향에서 가장 가까운 관측소들을 자동으로 찾아 자료를 받아온 뒤, 거리 가중
            평균(IDW)으로 결합하여 하나의 가상 베이스 자료로 만듭니다. 관측소 코드를 몰라도 됩니다.
          </div>
          <div style={{ display: "flex", gap: 6, alignItems: "flex-end", flexWrap: "wrap" }}>
            <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <span style={{ color: "#4b5563" }}>시작일</span>
              <input
                type="date"
                style={{ ...inputStyle, width: 140 }}
                value={nearestStartDate}
                onChange={(e) => setNearestStartDate(e.target.value)}
              />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <span style={{ color: "#4b5563" }}>일수</span>
              <input
                type="number"
                min="1"
                max="31"
                style={{ ...inputStyle, width: 60 }}
                value={nearestDays}
                onChange={(e) => setNearestDays(parseInt(e.target.value, 10) || 1)}
              />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <span style={{ color: "#4b5563" }}>관측소 수</span>
              <input
                type="number"
                min="1"
                max="8"
                style={{ ...inputStyle, width: 60 }}
                value={nStations}
                onChange={(e) => setNStations(parseInt(e.target.value, 10) || 1)}
              />
            </label>
            <button
              style={buttonStyle}
              disabled={nearestLoading || !nearestStartDate}
              onClick={() =>
                onFetchNearestIntermagnet({ start_date: nearestStartDate, days: nearestDays, n_stations: nStations })
              }
            >
              {nearestLoading ? "탐색 중..." : "주변 관측소 탐색"}
            </button>
          </div>

          {nearestHasResult && onShowNearestComparison && (
            <button style={buttonStyle} disabled={nearestComparisonLoading} onClick={onShowNearestComparison}>
              {nearestComparisonLoading ? "불러오는 중..." : "📊 주변 관측소 자료와 비교 그래프 보기"}
            </button>
          )}

          {nearestError && (
            <div style={{ color: "#dc2626", background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 6, padding: "6px 8px" }}>
              {nearestError}
            </div>
          )}

          {nearestPreview && (
            <div
              style={{
                border: "1px solid #bfdbfe",
                background: "#eff6ff",
                borderRadius: 6,
                padding: "8px 10px",
                display: "flex",
                flexDirection: "column",
                gap: 4,
              }}
            >
              <div style={{ fontWeight: 600 }}>선택된 관측소 {nearestPreview.stations?.length}개</div>
              <ul style={{ margin: 0, paddingLeft: 16 }}>
                {nearestPreview.stations?.map((s) => (
                  <li key={s.iaga_code}>
                    {s.station_name} ({s.iaga_code}) — {s.distance_km?.toFixed(0)} km, 방위각 {s.bearing_deg?.toFixed(0)}°
                  </li>
                ))}
              </ul>
              <div>
                결합 자료 {nearestPreview.n_points}개, 기간: {nearestPreview.time_range?.[0]} ~ {nearestPreview.time_range?.[1]}
              </div>
              <div style={{ display: "flex", gap: 6, marginTop: 4, flexWrap: "wrap" }}>
                <button
                  style={{ ...buttonStyle, background: "#2563eb", color: "white" }}
                  disabled={nearestApplying}
                  onClick={onApplyNearestIntermagnet}
                >
                  {nearestApplying ? "적용 중..." : "이 자료를 베이스 자료로 사용"}
                </button>
                {onExportNearestCsv && (
                  <button style={buttonStyle} onClick={onExportNearestCsv}>
                    CSV로 파일 저장
                  </button>
                )}
                <button style={buttonStyle} disabled={nearestApplying} onClick={onCancelNearestPreview}>
                  취소
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      <div style={{ fontWeight: 600, marginTop: 4 }}>직접 관측소 지정</div>

      <label
        style={{
          display: "inline-block",
          padding: "6px 10px",
          fontSize: 12,
          borderRadius: 6,
          border: "1px solid #d1d5db",
          background: "white",
          cursor: "pointer",
          textAlign: "center",
        }}
      >
        IAGA-2002 파일 업로드
        <input
          type="file"
          style={{ display: "none" }}
          disabled={loading}
          onChange={(e) => {
            const f = e.target.files[0];
            if (f) onUploadIaga2002(f);
            e.target.value = "";
          }}
        />
      </label>

      <div style={{ display: "flex", gap: 6, alignItems: "flex-end", flexWrap: "wrap" }}>
        <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ color: "#4b5563" }}>IAGA 관측소 코드 (예: IRT)</span>
          <input
            type="text"
            style={{ ...inputStyle, width: 100 }}
            value={iagaCode}
            onChange={(e) => setIagaCode(e.target.value.toUpperCase())}
            placeholder="IRT"
          />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ color: "#4b5563" }}>시작일</span>
          <input type="date" style={{ ...inputStyle, width: 140 }} value={startDate} onChange={(e) => setStartDate(e.target.value)} />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ color: "#4b5563" }}>일수</span>
          <input
            type="number"
            min="1"
            max="31"
            style={{ ...inputStyle, width: 60 }}
            value={days}
            onChange={(e) => setDays(parseInt(e.target.value, 10) || 1)}
          />
        </label>
        <button
          style={buttonStyle}
          disabled={loading || !iagaCode || !startDate}
          onClick={() => onFetchIntermagnet({ iaga_code: iagaCode, start_date: startDate, days })}
        >
          {loading ? "요청 중..." : "자동 다운로드 시도"}
        </button>
      </div>

      {error && (
        <div style={{ color: "#dc2626", background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 6, padding: "6px 8px" }}>
          {error}
        </div>
      )}

      {preview && (
        <div style={{ border: "1px solid #bfdbfe", background: "#eff6ff", borderRadius: 6, padding: "8px 10px", display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ fontWeight: 600 }}>
            {preview.station_name} ({preview.iaga_code})
          </div>
          <div>
            위도 {preview.lat?.toFixed(3)}, 경도 {preview.lon?.toFixed(3)}
            {preview.distance_from_survey_km != null && ` — 조사지역까지 약 ${preview.distance_from_survey_km.toFixed(0)} km`}
          </div>
          <div>
            자료 {preview.n_points}개, 기간: {preview.time_range?.[0]} ~ {preview.time_range?.[1]}
          </div>
          <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
            <button
              style={{ ...buttonStyle, background: "#2563eb", color: "white" }}
              disabled={applying}
              onClick={onApplyIntermagnet}
            >
              {applying ? "적용 중..." : "이 자료를 베이스 자료로 사용"}
            </button>
            <button style={buttonStyle} disabled={applying} onClick={onCancelPreview}>
              취소
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
