
// Shared by the workflow sections: their common styles, the option notes, and two small layout helpers.

export const TRANSFORM_INFO = {
  none: "그리딩된 원본 자력 이상값을 그대로 표시합니다.",
  rtp: "RTP(Reduction to Pole, 자극환산) — 비스듬한 지자기장 방향 때문에 이상체 바로 위가 아닌 옆으로 치우쳐 보이는 이상대를, 자극(수직 자기장)에서 측정한 것처럼 보정해 이상체 바로 위에 오도록 만듭니다.",
  rte: "RTE(Reduction to Equator, 자기적도환산) — 저위도(적도 부근)처럼 복각이 낮아 RTP가 불안정한 지역에서 대신 사용하는 유사한 보정 기법입니다.",
  "1vd": "1VD(수직 1차 미분) — 값 자체가 아니라 수직 방향 변화율을 보여줘, 얕고 경계가 뚜렷한 이상체를 더 선명하게 강조합니다.",
  as: "AS(Analytic Signal, 해석 신호 진폭) — 자화 방향에 무관하게 이상체 바로 위에서 극대값을 갖는 값으로, 자화 방향을 모를 때도 이상체 위치를 판단하기 좋습니다.",
  thdr: "THDR(총수평미분) — 수평 방향 변화율의 크기로, 자화 방향에 무관하게 이상체 경계에서 뚜렷하게 나타나 접촉면·경계 파악에 유용합니다.",
  upward_continuation: "상방연속 — 지정한 높이만큼 더 높은 고도에서 측정한 것처럼 계산을 재구성합니다. 얕고 짧은 파장 잡음이 깊고 넓은 이상보다 훨씬 빠르게 약해져, 광역 추세만 강조하거나 잡음을 완화할 때 사용합니다.",
  detrend: "추세면 제거 — 완만하게 변화하는 광역 배경(다항식 추세면)을 최소자승으로 맞춰 뺀 나머지(잔차)만 표시합니다. 얕은 국지 이상을 광역 배경과 분리해 볼 때 유용합니다.",
  microlevel: "마이크로레벨링(디코러게이션) — 측선과 나란한 방향의 짧은 파장(측선 간격 규모) 줄무늬 잡음만 골라 완화합니다. 헤딩/타이라인 보정 후에도 남는 줄무늬가 있을 때 사용합니다.",
  "2vd": "2VD(수직 2차 미분) — 1VD보다 한 단계 더 미분해, 서로 가까이 붙은 여러 이상체를 분리해 보는 데 유용합니다 (잡음도 그만큼 더 증폭됩니다).",
  tilt: "틸트 각(Tilt Angle) — 1VD/THDR의 비율을 각도(-90°~90°)로 표현해, 이상체 진폭 크기와 무관하게 경계에서 0을 지나갑니다. 강한 이상체와 약한 이상체가 섞여 있어도 같은 기준으로 경계를 볼 수 있습니다.",
  theta: "세타 맵(Theta Map) — THDR을 해석신호 진폭(AS)으로 정규화한 각도(0°~90°)로, 틸트 각과 마찬가지로 진폭에 무관한 경계 탐지 보조 지표입니다.",
  dx: "1차 동서방향 미분(dX) — 동서(easting) 방향 변화율.",
  dy: "1차 남북방향 미분(dY) — 남북(northing) 방향 변화율.",
  dxx: "2차 동서방향 미분(dXX).",
  dyy: "2차 남북방향 미분(dYY).",
  dxy: "동서-남북 혼합 2차 미분(dXY).",
  dxz: "동서-수직 혼합 2차 미분(dXZ).",
  dyz: "남북-수직 혼합 2차 미분(dYZ).",
};

export const KOREA_PROJECTION_INFO = {
  "": "비워두면 측선 중심 좌표로 UTM 존을 자동 감지합니다 (국내외 공통).",
  korea_utm: "KoreaUTM (EPSG:5179) — GRS80 기반 한국 통합 좌표계 (KGD2002 Unified CS). 국내 조사에 권장.",
  korea2010: "Korea2010 (EPSG:5185~5188) — 한반도 권역별(서부/중부/동부/동해) 띠 좌표계 (KGD2002 Belt 2010).",
  utm: "UTM (EPSG:32651/32652) — 국내 지역 51N/52N을 경도 기준으로 자동 선택하는 표준 UTM.",
};

export const FILTER_METHOD_INFO = {
  butterworth: "Butterworth 저역통과 — 지정한 차단주파수 이상을 뚜렷하게 잘라내는 표준적인 영위상(zero-phase) 필터입니다.",
  savgol: "Savitzky-Golay — 이동창 안에서 다항식을 맞춰 평활화합니다. 이동평균보다 이상체의 봉우리 높이·폭을 더 잘 보존합니다.",
  moving_average: "이동평균 — 가장 단순한 평활화 방식으로, 뾰족한 이상체를 다소 무디게 만들 수 있습니다.",
};

export const sectionStyle = { border: "1px solid #e6dac0", borderRadius: 8, marginBottom: 10, background: "white" };

export const summaryStyle = { padding: "10px 12px", fontWeight: 600, fontSize: 13, cursor: "pointer" };

export const bodyStyle = { padding: "0 12px 12px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: 12 };

export const inputStyle = { width: "100%", padding: "4px 6px", fontSize: 12, borderRadius: 4, border: "1px solid #ddd0b2" };

export const buttonStyle = {
  padding: "7px 10px",
  fontSize: 12,
  fontWeight: 600,
  borderRadius: 6,
  border: "1px solid #a9631f",
  background: "#a9631f",
  color: "white",
  cursor: "pointer",
};

export function ProgressBar({ fraction }) {
  if (fraction == null) return null;
  const pct = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, height: 6, borderRadius: 3, background: "#e6dac0", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: "#a9631f", transition: "width 0.15s" }} />
      </div>
      <span style={{ color: "#8a7a5c", fontSize: 11, minWidth: 32, textAlign: "right" }}>{pct}%</span>
    </div>
  );
}

export function Field({ label, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ color: "#6b5c42" }}>{label}</span>
      {children}
    </label>
  );
}
