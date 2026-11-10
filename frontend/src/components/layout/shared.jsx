import { lazy } from "react";

// Pieces the layout components share: the two views loaded on demand, and the sensor-pair readout.

// plotly.js-dist-min alone is ~4.7MB unminified - only the 3D volume view
// needs it, and most sessions never open it, so it's split into its own
// chunk and fetched on demand instead of bloating the initial bundle.
export const InversionVolumeView = lazy(() => import("../InversionVolumeView"));

// GuidelinePanel also pulls in plotly.js-dist-min (for the power-spectrum
// chart) - same reasoning as InversionVolumeView above, split into its own
// chunk rather than bloating the initial bundle.
export const GuidelinePanel = lazy(() => import("../GuidelinePanel"));

// Two sensors on one carrier read the same field up to a fixed offset;
// what differs beyond that is system error, and it tells which sensor to
// process (magcore/sensor_pair.py). Shown only for two-sensor loggers.
export function SensorPairBox({ pair }) {
  const [a, b] = pair.labels;
  const usedLabel = pair.used === "mean" ? "두 센서 평균" : pair.used === "2" ? b : a;
  const signed = (v) => (v == null ? "—" : `${v > 0 ? "+" : ""}${v}`);
  const other = pair.cleaner && pair.used !== "mean" && pair.cleaner !== usedLabel;
  return (
    <div
      style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, borderRadius: 6, color: other ? "#b45309" : "#4a3d28",
               background: other ? "#fffbeb" : "#f7f2e6", border: `1px solid ${other ? "#fde68a" : "#e6dac0"}` }}
      title="두 센서는 같은 자기장을 일정한 차이를 두고 읽습니다. 그 차이가 비행 중 흔들리는 만큼, 그리고 왕복 방향에 따라 각 센서의 평균이 달라지는 만큼이 시스템 오차입니다."
    >
      <b>센서 쌍 진단</b> — 사용 중: {usedLabel} · {b} − {a} = {signed(pair.offset_nt)} nT
      {pair.difference_std_flight_nt != null && <> · 비행 중 차이의 산포 {pair.difference_std_flight_nt} nT</>} ⓘ
      {pair.direction_effect_nt && (
        <div style={{ marginTop: 3 }}>
          왕복 방향 효과: {a} {signed(pair.direction_effect_nt[a])} nT, {b} {signed(pair.direction_effect_nt[b])} nT
        </div>
      )}
      {pair.background_std_nt && (
        <div style={{ marginTop: 3 }}>
          배경 산포: {a} {pair.background_std_nt[a]} nT, {b} {pair.background_std_nt[b]} nT
          {pair.cleaner && <> — 단독으로 쓴다면 {pair.cleaner}</>}
          {other && <b> (지금은 {usedLabel}을(를) 쓰고 있습니다. 3번에서 바꿀 수 있습니다.)</b>}
        </div>
      )}
    </div>
  );
}
