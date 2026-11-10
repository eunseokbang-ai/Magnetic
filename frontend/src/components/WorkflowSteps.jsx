import DroneUploadSection from "./workflow/DroneUploadSection";
import BaseStationSection from "./workflow/BaseStationSection";
import HeadingCalibrationSection from "./workflow/HeadingCalibrationSection";
import ProcessingSection from "./workflow/ProcessingSection";
import MapValueSection from "./workflow/MapValueSection";
import GriddingSection from "./workflow/GriddingSection";
import DerivedGridSection from "./workflow/DerivedGridSection";
import BasemapSection from "./workflow/BasemapSection";
import { useApp } from "../AppContext";
import { sectionStyle } from "./workflow/shared";

export default function WorkflowSteps() {
  const { error } = useApp();

  return (
    <div>
      {error && (
        <div style={{ ...sectionStyle, padding: 10, background: "#fef2f2", color: "#dc2626", fontSize: 12 }}>{error}</div>
      )}

      <DroneUploadSection />

      <BaseStationSection />

      <HeadingCalibrationSection />

      <ProcessingSection />

      <MapValueSection />

      <GriddingSection />

      <DerivedGridSection />

      <BasemapSection />
    </div>
  );
}
