import { useApp } from "../../AppContext";
import { bodyStyle, sectionStyle, summaryStyle } from "./shared";

// Workflow step 11: basemap and geology overlays.
// One section of components/WorkflowSteps.jsx; it reads App's state with useApp().
export default function BasemapSection() {
  const { geologyOpacity, setGeologyOpacity } = useApp();

  return (
    <details style={sectionStyle}>
      <summary style={summaryStyle}>11. 배경지도 · 지질도</summary>
      <div style={bodyStyle}>
        <div>지도 우측 상단 레이어 컨트롤에서 OSM / Esri 위성 / Google 위성을 전환할 수 있습니다.</div>
        <div style={{ marginTop: 6 }}>
          같은 컨트롤 아래쪽의 <b>KIGAM 지질도</b>(1:5만 / 1:25만 / 1:100만 / 단층)를 켜면 배경지도 위에 겹쳐집니다.
          출처: KIGAM 지오빅데이터 오픈플랫폼(data.kigam.re.kr).
        </div>
        <label
          style={{ display: "flex", flexDirection: "column", gap: 2, marginTop: 6 }}
          title="지질도 레이어의 투명도입니다. 배경 위성영상이나 자력 그리드와 함께 보려면 낮추세요."
        >
          <span style={{ color: "#6b5c42" }}>지질도 불투명도: {Math.round((geologyOpacity ?? 0.6) * 100)}%</span>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={geologyOpacity ?? 0.6}
            onChange={(e) => setGeologyOpacity && setGeologyOpacity(parseFloat(e.target.value))}
          />
        </label>
        <div style={{ color: "#8a7a5c", fontSize: 11, marginTop: 4 }}>
          1:5만은 축척이 커서 지도를 어느 정도 확대해야(줌 9 이상) 표시되고, 섬 지역 일부는 이 축척에 포함되지 않습니다.
          지질도 타일도 <b>16. 오프라인 지도</b>에서 미리 받아 두면 인터넷 없이 볼 수 있습니다.
        </div>
      </div>
    </details>
  );
}
