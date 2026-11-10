import { useRef, useState } from "react";
import { formatArea, formatDistance, pathLength, polygonArea } from "../geoMeasure";

// 지도 위 거리·면적 재기.
//
// One of the hooks App() is composed of (split out of App.jsx in 2026-11;
// the statements are unchanged). It receives what the hooks called before
// it returned and returns everything it declares.
export default function useMeasureTool() {
  const [measureMode, setMeasureMode] = useState(null); // null | "distance" | "area"
  const [measurements, setMeasurements] = useState([]);
  const measureCounterRef = useRef({ distance: 0, area: 0 });

  const handleMeasureShapeDrawn = (coords) => {
    const latlngs = coords.map(([lat, lng]) => ({ lat, lng }));
    if (measureMode === "distance") {
      measureCounterRef.current.distance += 1;
      const value = pathLength(latlngs);
      setMeasurements((prev) => [
        ...prev,
        { id: `d${Date.now()}${Math.random()}`, type: "distance", latlngs, label: `거리 ${measureCounterRef.current.distance}`, valueText: formatDistance(value) },
      ]);
    } else if (measureMode === "area") {
      measureCounterRef.current.area += 1;
      const value = polygonArea(latlngs);
      setMeasurements((prev) => [
        ...prev,
        { id: `a${Date.now()}${Math.random()}`, type: "area", latlngs, label: `면적 ${measureCounterRef.current.area}`, valueText: formatArea(value) },
      ]);
    }
  };

  const handleRemoveMeasurement = (id) => {
    setMeasurements((prev) => prev.filter((m) => m.id !== id));
  };

  const handleClearMeasurements = () => {
    setMeasurements([]);
    measureCounterRef.current = { distance: 0, area: 0 };
  };

  return {
    measureMode, setMeasureMode, measurements, setMeasurements, measureCounterRef, handleMeasureShapeDrawn,
    handleRemoveMeasurement, handleClearMeasurements,
  };
}
