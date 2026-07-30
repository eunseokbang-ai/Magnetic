const PROFILE_LABELS = { custom: "자유선", ew: "동서", ns: "남북" };

// Mirrors InversionVolumeView's modal-style overlay so the section image
// gets real screen space instead of a cramped sidebar thumbnail.
export default function InversionSectionView({ data, onClose }) {
  if (!data) return null;

  const profileLabel = PROFILE_LABELS[data.profile] || "";

  return (
    <div
      style={{
        position: "absolute",
        top: 16,
        right: 16,
        bottom: 16,
        left: 16,
        background: "white",
        border: "1px solid #d1d5db",
        borderRadius: 8,
        boxShadow: "0 4px 20px rgba(0,0,0,0.25)",
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 12px", borderBottom: "1px solid #e5e7eb" }}>
        <div style={{ fontSize: 13, fontWeight: 600 }}>수직 단면 뷰 ({profileLabel})</div>
        <button onClick={onClose} style={{ border: "none", background: "none", cursor: "pointer", fontSize: 14 }}>
          ✕ 닫기
        </button>
      </div>
      <div style={{ flex: 1, padding: 16, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", overflow: "auto" }}>
        {/* This is an abstracted distance-vs-elevation plot, not a
            geo-referenced image, so stretching it to fill the panel
            (rather than preserving the raw pixel aspect ratio) is fine
            and makes thin, wide sections much easier to read. */}
        <img
          src={data.image_data_url}
          alt="수직 단면"
          style={{ width: "100%", height: "100%", objectFit: "fill", border: "1px solid #d1d5db", imageRendering: "pixelated" }}
        />
        <div style={{ fontSize: 12, color: "#6b7280", marginTop: 8, display: "flex", gap: 24 }}>
          <span>거리: 0 ~ {data.distance_m?.at(-1)?.toFixed(0)} m</span>
          <span>
            고도: {data.elevation_m?.at(-1)?.toFixed(0)} ~ {data.elevation_m?.[0]?.toFixed(0)} m
          </span>
          {data.stats?.max != null && (
            <span>
              자화율(SI): {data.stats.min?.toFixed(4)} ~ {data.stats.max?.toFixed(4)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
