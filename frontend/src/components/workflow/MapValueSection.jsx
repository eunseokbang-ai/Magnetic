import { useApp } from "../../AppContext";
import InfoIcon from "../InfoIcon";
import { Field, bodyStyle, inputStyle, sectionStyle, summaryStyle } from "./shared";

// Workflow steps 6 and 8: which value the map shows.
// One section of components/WorkflowSteps.jsx; it reads App's state with useApp().
export default function MapValueSection() {
  const { handleSetValueField, valueField } = useApp();

  return (
    <details style={sectionStyle} open>
      <summary style={summaryStyle}>6, 8. 지도 표시 값</summary>
      <div style={bodyStyle}>
        <Field
          label={
            <>
              컬러로 표시할 값
              <InfoIcon text="TMI(Total Magnetic Intensity, 전자력) = 측정된 자기장 세기(일변화 보정 적용). IGRF(International Geomagnetic Reference Field) = 지구 자체의 배경 자기장 모델값. 자력 이상 = TMI - IGRF로, 지구 배경 자기장을 뺀 뒤 남는, 지하 자성체에 의한 자기장 변화만을 의미합니다 — 보통 이 값을 봅니다." />
            </>
          }
        >
          <select style={inputStyle} value={valueField} onChange={(e) => handleSetValueField(e.target.value)}>
            <option value="tmi">TMI (보정된 전자력)</option>
            <option value="anomaly">자력 이상 (TMI - IGRF)</option>
          </select>
        </Field>
      </div>
    </details>
  );
}
