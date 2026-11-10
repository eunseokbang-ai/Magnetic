import { Suspense } from "react";
import { useApp } from "../../AppContext";
import { toggleButtonStyle } from "../../appDefaults";
import BaseStationView from "../BaseStationView";
import InversionSectionView from "../InversionSectionView";
import LineProfileView from "../LineProfileView";
import MapView from "../MapView";
import NearestIntermagnetComparisonView from "../NearestIntermagnetComparisonView";
import { InversionVolumeView } from "./shared";

// The middle of the screen: the map, and whichever view is open over its lower
// part - the 3D inversion volume, a section, a line profile, the base station
// record, the observatory comparison.
// A region of App's screen; it reads App's state with useApp().
export default function MapArea() {
  const {
    asDepthResult, baseTimeseries, boundaryMode, boundaryPolygon, boxFacesData, boxFacesLoading,
    boxTopLayerIndex, candidateResult, candidateShow, cmapName, confidenceOverlay, contactResult, drawMode,
    editablePoints, eulerResult, flyToTarget, geologyDrawMode, geologyOpacity, geologyStatsResult,
    gridOpacity, handleAddVolumeSlice, handleApplySmoothing, handleBoundaryDrawn, handleInspectClick,
    handleLoadBoxFaces, handleMapBoundsChange, handleMapShapeDrawn, handleMeasureShapeDrawn,
    handleRemoveVolumeSlice, handleUpdateVolumeSlice, inspectMode, inspectPoints, inversionSummary,
    lastCustomSectionPath, legendRange, lineProfileData, lineamentResult, measureMode, measurements,
    multiscaleResult, nearestIntermagnetComparison, overlay, overlayLayers, processSummary,
    prospectivityOverlay, prospectivityResult, removalMethod, sectionDrawMode, sectionResult,
    selectedCandidateIndices, selectedStructureAnomalyIndices, selectedStructurePolygonIndices,
    setBaseTimeseries, setHoverPoint, setLeftSidebarOpen, setLineProfileData,
    setNearestIntermagnetComparison, setRightSidebarOpen, setSectionResult, setVolumeData, setVolumeOpacity,
    showAsDepth, showConfidenceOverlay, showContacts, showContours, showDetectedTargets, showEulerSolutions,
    showGeologyUnits, showLineLabels, showLineaments, showMultiscaleLayer, showPointsOverGrid,
    showProspectivityOverlay, showProspectivityTargets, showTiltDepth, smoothDrawMode, smoothing,
    structureScanResult, structureScanShow, targetDetectionResult, tiltDepthResult, valueField, volumeData,
    volumeOpacity, volumeSlices,
  } = useApp();

  return (
    <div style={{ flex: 1, position: "relative", minWidth: 0 }}>
      <div style={{ position: "absolute", top: 60, left: 8, zIndex: 1200, display: "flex", gap: 6 }}>
        <button className="app-sidebar-toggle" onClick={() => setLeftSidebarOpen((v) => !v)} style={toggleButtonStyle} title="자료 처리 패널 열기/닫기">
          ☰
        </button>
      </div>
      <div style={{ position: "absolute", top: 60, right: 8, zIndex: 1200, display: "flex", gap: 6 }}>
        <button className="app-sidebar-toggle" onClick={() => setRightSidebarOpen((v) => !v)} style={toggleButtonStyle} title="지도 도구 패널 열기/닫기">
          🛠
        </button>
      </div>
      <MapView
        geologyOpacity={geologyOpacity}
        points={editablePoints}
        colorRange={legendRange}
        cmapName={cmapName}
        overlay={overlay}
        showContours={showContours}
        gridOpacity={gridOpacity}
        confidenceOverlay={showConfidenceOverlay ? confidenceOverlay : null}
        showPointsOverGrid={showPointsOverGrid}
        overlayLayers={overlayLayers}
        lines={processSummary?.lines}
        showLineLabels={showLineLabels}
        onHoverPoint={setHoverPoint}
        drawMode={drawMode || sectionDrawMode || smoothDrawMode || geologyDrawMode}
        drawShapeType={sectionDrawMode ? "polyline" : "polygon"}
        onShapeDrawn={handleMapShapeDrawn}
        eulerSolutions={showEulerSolutions ? eulerResult?.solutions : null}
        detectedTargets={showDetectedTargets ? targetDetectionResult?.targets : null}
        multiscaleEdgePoints={showMultiscaleLayer ? multiscaleResult?.points : null}
        lineaments={showLineaments ? lineamentResult?.lineaments : null}
        tiltDepthPoints={showTiltDepth ? tiltDepthResult?.points : null}
        analyticSignalDepthPoints={showAsDepth ? asDepthResult?.points : null}
        contacts={showContacts ? contactResult?.contacts : null}
        geologyUnits={showGeologyUnits ? geologyStatsResult?.geojson : null}
        prospectivityTargets={showProspectivityTargets ? prospectivityResult?.targets : null}
        prospectivityOverlay={showProspectivityOverlay ? prospectivityOverlay : null}
        onBoundsChange={handleMapBoundsChange}
        inspectMode={inspectMode}
        inspectPoints={inspectPoints}
        onInspectClick={handleInspectClick}
        measureMode={measureMode}
        measurements={measurements}
        onMeasureShapeDrawn={handleMeasureShapeDrawn}
        anomalyCandidateResult={candidateShow ? candidateResult : null}
        anomalyCandidateRemovalMethod={removalMethod}
        selectedAnomalyCandidateIndices={selectedCandidateIndices}
        structureScanResult={structureScanShow ? structureScanResult : null}
        selectedStructurePolygonIndices={selectedStructurePolygonIndices}
        selectedStructureAnomalyIndices={selectedStructureAnomalyIndices}
        boundaryMode={boundaryMode}
        boundaryPolygon={boundaryPolygon}
        onBoundaryDrawn={handleBoundaryDrawn}
        flyToTarget={flyToTarget}
      />
      {volumeData && (
        <Suspense
          fallback={
            <div
              style={{
                position: "absolute",
                inset: 0,
                background: "rgba(255,255,255,0.9)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 14,
                color: "#4a3d28",
              }}
            >
              3D 뷰어 불러오는 중...
            </div>
          }
        >
          <InversionVolumeView
            data={volumeData}
            onClose={() => setVolumeData(null)}
            boxData={boxFacesData}
            boxTopLayerIndex={boxTopLayerIndex}
            onBoxTopLayerIndexChange={handleLoadBoxFaces}
            boxLoading={boxFacesLoading}
            nLayers={inversionSummary?.n_layers}
            slices={volumeSlices}
            onAddSlice={handleAddVolumeSlice}
            onRemoveSlice={handleRemoveVolumeSlice}
            onUpdateSlice={handleUpdateVolumeSlice}
            customSliceAvailable={!!lastCustomSectionPath}
            opacity={volumeOpacity}
            onOpacityChange={setVolumeOpacity}
          />
        </Suspense>
      )}
      {!volumeData && sectionResult && <InversionSectionView data={sectionResult} onClose={() => setSectionResult(null)} />}
      {!volumeData && !sectionResult && lineProfileData && (
        <LineProfileView
          data={lineProfileData}
          valueLabel={valueField === "anomaly" ? "자력 이상 (nT)" : "TMI (nT)"}
          onClose={() => setLineProfileData(null)}
          onApplySmoothing={handleApplySmoothing}
          smoothing={smoothing}
        />
      )}
      {!volumeData && !sectionResult && !lineProfileData && baseTimeseries && (
        <BaseStationView data={baseTimeseries} onClose={() => setBaseTimeseries(null)} />
      )}
      {!volumeData && !sectionResult && !lineProfileData && !baseTimeseries && nearestIntermagnetComparison && (
        <NearestIntermagnetComparisonView
          data={nearestIntermagnetComparison}
          onClose={() => setNearestIntermagnetComparison(null)}
        />
      )}
    </div>
  );
}
