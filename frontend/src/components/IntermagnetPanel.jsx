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
  nearestCsvFilename,
  onNearestCsvFilenameChange,
  onShowNearestComparison,
  nearestPreview,
  nearestLoading,
  nearestApplying,
  nearestError,
  nearestHasResult,
  nearestComparisonLoading,
  flightDates,
}) {
  const [iagaCode, setIagaCode] = useState("");
  const [dateMode, setDateMode] = useState("auto");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [nearestDateMode, setNearestDateMode] = useState("auto");
  const [nearestStartDate, setNearestStartDate] = useState("");
  const [nearestEndDate, setNearestEndDate] = useState("");
  const [nStations, setNStations] = useState(4);
  const [maxDistanceKm, setMaxDistanceKm] = useState(2000);

  const hasFlightDates = flightDates?.length > 0;
  const dateModeToggle = (mode, setMode) => (
    <div style={{ display: "flex", gap: 10, fontSize: 12 }}>
      <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
        <input type="radio" checked={mode === "auto"} onChange={() => setMode("auto")} />
        측선 자료 날짜 자동 사용 (권장)
      </label>
      <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
        <input type="radio" checked={mode === "manual"} onChange={() => setMode("manual")} />
        직접 기간 지정
      </label>
    </div>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, paddingTop: 8, fontSize: 12 }}>
      <div style={{ color: "#8a7a5c" }}>
        측정 지역의 로컬 일변화까지는 반영하지 못하지만, 태양풍 등에 의한 지역/전지구적 자기장 변화는 인근 관측소 자료로
        어느 정도 추정할 수 있습니다. INTERMAGNET(intermagnet.org)에서 직접 받은 IAGA-2002 파일을 업로드하거나, 관측소
        코드와 날짜를 입력해 서버에서 바로 받아올 수 있습니다 (서버의 아웃바운드 네트워크 정책에 따라 자동 다운로드가
        막혀 있을 수 있으며, 이 경우 파일 업로드를 이용하세요).
      </div>

      {onFetchNearestIntermagnet && (
        <div
          style={{
            border: "1px solid #e6dac0",
            borderRadius: 6,
            padding: "8px 10px",
            display: "flex",
            flexDirection: "column",
            gap: 6,
            background: "#faf7ef",
          }}
        >
          <div style={{ fontWeight: 600 }}>주변 관측소 자동 선택 (추천)</div>
          <div style={{ color: "#8a7a5c" }}>
            조사지역 평균 좌표 주변 동서남북 방향에서 가장 가까운 관측소들을 자동으로 찾아 자료를 받아온 뒤, 거리 가중
            평균(IDW)으로 결합하여 하나의 가상 베이스 자료로 만듭니다. 관측소 코드를 몰라도 됩니다.
          </div>
          {dateModeToggle(nearestDateMode, setNearestDateMode)}
          {nearestDateMode === "auto" ? (
            <div style={{ color: hasFlightDates ? "#6b5c42" : "#b45309" }}>
              {hasFlightDates
                ? `인식된 측선 날짜 (${flightDates.length}일): ${flightDates.join(", ")}`
                : "드론 자료를 먼저 업로드하면 촬영 날짜를 자동으로 인식합니다 (또는 직접 기간을 지정하세요)."}
            </div>
          ) : (
            <div style={{ display: "flex", gap: 6, alignItems: "flex-end", flexWrap: "wrap" }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <span style={{ color: "#6b5c42" }}>시작일</span>
                <input
                  type="date"
                  style={{ ...inputStyle, width: 140 }}
                  value={nearestStartDate}
                  onChange={(e) => setNearestStartDate(e.target.value)}
                />
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <span style={{ color: "#6b5c42" }}>종료일</span>
                <input
                  type="date"
                  style={{ ...inputStyle, width: 140 }}
                  value={nearestEndDate}
                  min={nearestStartDate || undefined}
                  onChange={(e) => setNearestEndDate(e.target.value)}
                />
              </label>
            </div>
          )}
          <div style={{ display: "flex", gap: 6, alignItems: "flex-end", flexWrap: "wrap" }}>
            <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <span style={{ color: "#6b5c42" }}>관측소 수 (최대)</span>
              <input
                type="number"
                min="1"
                max="8"
                style={{ ...inputStyle, width: 60 }}
                value={nStations}
                onChange={(e) => setNStations(parseInt(e.target.value, 10) || 1)}
              />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <span style={{ color: "#6b5c42" }} title="이 거리보다 먼 관측소는 아예 후보에서 제외합니다 - 방향(사분면)당 근처에 후보가 없으면 관측소 수보다 적게 선택될 수 있습니다">
                최대 거리 (km)
              </span>
              <input
                type="number"
                min="1"
                style={{ ...inputStyle, width: 80 }}
                value={maxDistanceKm}
                onChange={(e) => setMaxDistanceKm(e.target.value === "" ? "" : parseInt(e.target.value, 10) || 1)}
                placeholder="제한없음"
              />
            </label>
            <button
              style={buttonStyle}
              disabled={nearestLoading || (nearestDateMode === "manual" && (!nearestStartDate || !nearestEndDate))}
              onClick={() =>
                onFetchNearestIntermagnet({
                  ...(nearestDateMode === "manual" ? { start_date: nearestStartDate, end_date: nearestEndDate } : {}),
                  n_stations: nStations,
                  max_distance_km: maxDistanceKm === "" ? null : maxDistanceKm,
                })
              }
            >
              {nearestLoading ? "탐색 중..." : "주변 관측소 탐색"}
            </button>
          </div>
          <div style={{ color: "#ab9a78" }}>
            요청한 날짜 중 아직 게시되지 않은 자료나 특정일에 결측된 자료는, 그 전날과 다음날 자료로 추정하여
            채웁니다. 너무 먼 관측소는 일변화 위상/진폭이 달라질 수 있어(경도차 ≈ 지방시차, 위도차 ≈ 다른 자기위도대) 최대
            거리를 벗어난 후보는 자동 제외됩니다 - 빈칸으로 두면 거리 제한 없이 방향당 가장 가까운 관측소를 찾습니다.
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
                border: "1px solid #ecd0a3",
                background: "#faf0e2",
                borderRadius: 6,
                padding: "8px 10px",
                display: "flex",
                flexDirection: "column",
                gap: 4,
              }}
            >
              <div style={{ fontWeight: 600 }}>선택된 관측소 {nearestPreview.stations?.length}개</div>
              {nearestPreview.n_requested != null &&
                nearestPreview.stations?.length < nearestPreview.n_requested && (
                  <div style={{ color: "#b45309" }}>
                    요청한 {nearestPreview.n_requested}개 중 최대 거리
                    {nearestPreview.max_distance_km != null ? ` ${nearestPreview.max_distance_km}km` : ""} 조건 안에
                    드는 관측소가 {nearestPreview.stations?.length}개뿐이어서 그만큼만 선택됐습니다.
                  </div>
                )}
              <ul style={{ margin: 0, paddingLeft: 16 }}>
                {nearestPreview.stations?.map((s) => (
                  <li key={s.iaga_code}>
                    {s.station_name} ({s.iaga_code}) — {s.distance_km?.toFixed(0)} km, 방위각 {s.bearing_deg?.toFixed(0)}°
                    {s.estimated_dates?.length > 0 && (
                      <span style={{ color: "#b45309" }}> — 추정된 날짜: {s.estimated_dates.join(", ")}</span>
                    )}
                  </li>
                ))}
              </ul>
              <div>
                결합 자료 {nearestPreview.n_points}개, 기간: {nearestPreview.time_range?.[0]} ~ {nearestPreview.time_range?.[1]}
              </div>
              <div style={{ display: "flex", gap: 6, marginTop: 4, flexWrap: "wrap" }}>
                <button
                  style={{ ...buttonStyle, background: "#a9631f", color: "white" }}
                  disabled={nearestApplying}
                  onClick={onApplyNearestIntermagnet}
                >
                  {nearestApplying ? "적용 중..." : "이 자료를 베이스 자료로 사용"}
                </button>
                {onExportNearestCsv && (
                  <>
                    <input
                      type="text"
                      value={nearestCsvFilename}
                      onChange={(e) => onNearestCsvFilenameChange(e.target.value)}
                      placeholder="저장할 파일명"
                      title="다른 프로젝트에서 베이스 자료로 재사용할 때 알아보기 쉬운 이름으로 바꿀 수 있습니다"
                      style={{ ...inputStyle, width: 220 }}
                    />
                    <button style={buttonStyle} onClick={onExportNearestCsv}>
                      CSV로 파일 저장
                    </button>
                  </>
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
          border: "1px solid #ddd0b2",
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
          <span style={{ color: "#6b5c42" }}>IAGA 관측소 코드 (예: IRT)</span>
          <input
            type="text"
            style={{ ...inputStyle, width: 100 }}
            value={iagaCode}
            onChange={(e) => setIagaCode(e.target.value.toUpperCase())}
            placeholder="IRT"
          />
        </label>
      </div>
      {dateModeToggle(dateMode, setDateMode)}
      {dateMode === "auto" ? (
        <div style={{ color: hasFlightDates ? "#6b5c42" : "#b45309" }}>
          {hasFlightDates
            ? `인식된 측선 날짜 (${flightDates.length}일): ${flightDates.join(", ")}`
            : "드론 자료를 먼저 업로드하면 촬영 날짜를 자동으로 인식합니다 (또는 직접 기간을 지정하세요)."}
        </div>
      ) : (
        <div style={{ display: "flex", gap: 6, alignItems: "flex-end", flexWrap: "wrap" }}>
          <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <span style={{ color: "#6b5c42" }}>시작일</span>
            <input type="date" style={{ ...inputStyle, width: 140 }} value={startDate} onChange={(e) => setStartDate(e.target.value)} />
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <span style={{ color: "#6b5c42" }}>종료일</span>
            <input
              type="date"
              style={{ ...inputStyle, width: 140 }}
              value={endDate}
              min={startDate || undefined}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </label>
        </div>
      )}
      <div style={{ display: "flex", gap: 6 }}>
        <button
          style={buttonStyle}
          disabled={loading || !iagaCode || (dateMode === "manual" && (!startDate || !endDate))}
          onClick={() =>
            onFetchIntermagnet({
              iaga_code: iagaCode,
              ...(dateMode === "manual" ? { start_date: startDate, end_date: endDate } : {}),
            })
          }
        >
          {loading ? "요청 중..." : "자동 다운로드 시도"}
        </button>
      </div>
      <div style={{ color: "#ab9a78" }}>
        요청한 날짜 중 아직 게시되지 않은 자료나 특정일에 결측된 자료는, 그 전날과 다음날 자료로 추정하여 채웁니다.
      </div>

      {error && (
        <div style={{ color: "#dc2626", background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 6, padding: "6px 8px" }}>
          {error}
        </div>
      )}

      {preview && (
        <div style={{ border: "1px solid #ecd0a3", background: "#faf0e2", borderRadius: 6, padding: "8px 10px", display: "flex", flexDirection: "column", gap: 4 }}>
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
          {preview.estimated_dates?.length > 0 && (
            <div style={{ color: "#b45309" }}>추정된 날짜: {preview.estimated_dates.join(", ")}</div>
          )}
          <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
            <button
              style={{ ...buttonStyle, background: "#a9631f", color: "white" }}
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
