import * as api from "../../api";
import { useApp } from "../../AppContext";
import AnomalyCandidatePanel from "../AnomalyCandidatePanel";
import BlockLevelingPanel from "../BlockLevelingPanel";
import FlightPathEditor from "../FlightPathEditor";
import HeightNormalizationPanel from "../HeightNormalizationPanel";
import LayerManager from "../LayerManager";
import LineEditor from "../LineEditor";
import SensorHeightPanel from "../SensorHeightPanel";
import RepeatPassPanel from "../RepeatPassPanel";
import StructureDistortionPanel from "../StructureDistortionPanel";

// Right column, first part: what changes the data after processing - height
// normalisation, line and point editing, block and repeat-pass levelling, anomaly
// candidates and ground structures - and the reference layers.
// A region of App's screen; it reads App's state with useApp().
export default function EditingPanels() {
  const {
    candidateApplying, candidateError, candidateParams, candidateResult, candidateRunning, candidateShow,
    drawAction, drawMode, editablePoints, exitAllDrawModes, exportingBln, flightPathEditorOpen,
    handleApplyCandidateSmoothing, handleApplyStructureSmoothing, handleError, handleExportBln,
    handleManualExcludePointIds, handleMoveOverlay, handlePickRasterFile, handlePickTileFolder,
    handleRegisterLocalRaster, handleRegisterLocalTileFolder, handleRemoveOverlay, handleResetAllSmoothing,
    handleResetManual, handleRunAnomalyCandidates, handleRunStructureScan, handleSetCandidateNote,
    handleSetOverlayOpacity, handleShowLineProfile, handleSourceRemovalHistory, handleToggleCandidate,
    handleToggleLines, handleToggleOverlayVisible, handleToggleStructureAnomaly,
    handleToggleStructurePolygon, handleUndoSmoothing, handleUploadOverlayImage, lastDrawnPolygon,
    lastRemovalAdded, lineProfileLoadingId, overlayError, overlayLayers, overlayUploading, pointsLoading,
    processSummary, projectId, refreshPoints, removalMethod, removalWorking, saveFilename,
    selectedCandidateIndices, selectedStructureAnomalyIndices, selectedStructurePolygonIndices,
    setCandidateParams, setCandidateShow, setDrawAction, setDrawMode, setError, setFlightPathEditorOpen,
    setFlyToTarget, setProcessParams, setProcessSummary, setRemovalMethod, setSelectedCandidateIndices,
    setShowLineLabels, setShowPointsOverGrid, setShowRampPoints, setSmoothDrawMode, setStructureScanParams,
    setStructureScanShow, showGeneratedLayer, showLineLabels, showPointsOverGrid, showRampPoints,
    smoothDrawMode, smoothHistory, smoothing, structureScanApplying, structureScanError, structureScanParams,
    structureScanResult, structureScanRunning, structureScanShow, tileFolderError, tileFolderRegistering,
    valueField, handleSelectRun, runSelectEnabled, selectedRun, setRunSelectEnabled, setSelectedRun,
  } = useApp();

  return (
    <>
      <SensorHeightPanel
        projectId={projectId}
        processSummary={processSummary}
        onApplied={async (resp) => {
          setProcessSummary(resp);
          await refreshPoints(projectId, valueField);
        }}
        onError={handleError}
      />

      <HeightNormalizationPanel
        projectId={projectId}
        processSummary={processSummary}
        onApplied={async (resp) => {
          setProcessSummary(resp);
          await refreshPoints(projectId, valueField);
        }}
        onError={handleError}
      />

      <h2 style={{ fontSize: 13, margin: "4px 0 10px 0" }}>7. 수동 측선 편집</h2>
      <LineEditor
        lines={processSummary?.lines}
        onToggleLines={handleToggleLines}
        drawMode={drawMode}
        drawAction={drawAction}
        onSetDrawAction={setDrawAction}
        onToggleDrawMode={() => {
          const next = !drawMode;
          exitAllDrawModes();
          if (next) setDrawMode(true);
        }}
        smoothDrawMode={smoothDrawMode}
        onToggleSmoothDrawMode={() => {
          const next = !smoothDrawMode;
          exitAllDrawModes();
          if (next) setSmoothDrawMode(true);
        }}
        smoothing={smoothing}
        nSmoothedActions={smoothHistory.length}
        onUndoSmoothing={handleUndoSmoothing}
        onResetAllSmoothing={handleResetAllSmoothing}
        removalMethod={removalMethod}
        setRemovalMethod={setRemovalMethod}
        nSourceRemovals={(processSummary?.source_removals || []).length}
        onUndoSourceRemoval={() => handleSourceRemovalHistory("undo")}
        onResetSourceRemovals={() => handleSourceRemovalHistory("reset")}
        onResetManual={handleResetManual}
        runSelectEnabled={runSelectEnabled}
        onToggleRunSelect={(on) => {
          setRunSelectEnabled(on);
          if (!on) setSelectedRun(null);
        }}
        transit={processSummary?.transit}
        onSelectRun={(runId) => {
          exitAllDrawModes();
          handleSelectRun(runId);
        }}
        selectedRunId={selectedRun?.runId}
        nManualIncluded={processSummary?.n_manual_included}
        nManualExcluded={processSummary?.n_manual_excluded}
        overlapFor={(lineId) =>
          (processSummary?.line_overlaps?.decisions || []).find((d) => d.line_id === lineId)
        }
        onToggleOverlap={(lineId, keep) => {
          // Re-processing with this line's choice flipped; the rest keep
          // whatever the automatic comparison decided.
          setProcessParams((p) => ({
            ...p,
            line_params: {
              ...p.line_params,
              overlap_overrides: { ...(p.line_params.overlap_overrides || {}), [lineId]: keep },
            },
          }));
          setError(null);
        }}
        showPointsOverGrid={showPointsOverGrid}
        onToggleShowPointsOverGrid={setShowPointsOverGrid}
        showLineLabels={showLineLabels}
        onToggleShowLineLabels={setShowLineLabels}
        onShowLineProfile={handleShowLineProfile}
        lineProfileLoadingId={lineProfileLoadingId}
        onOpenFlightPathEditor={() => setFlightPathEditorOpen(true)}
        onExportBln={handleExportBln}
        exportingBln={exportingBln}
        canExportBln={!!lastDrawnPolygon}
        showRampPoints={showRampPoints}
        onToggleShowRampPoints={setShowRampPoints}
      />

      <BlockLevelingPanel
        projectId={projectId}
        processSummary={processSummary}
        onApplied={async (summary) => {
          setProcessSummary(summary);
          await refreshPoints(projectId, valueField);
        }}
        onError={handleError}
      />

      <RepeatPassPanel
        projectId={projectId}
        processSummary={processSummary}
        onShowMap={showGeneratedLayer}
        onApplied={async (resp) => {
          setProcessSummary(resp);
          await refreshPoints(projectId, valueField);
        }}
        onError={handleError}
      />

      <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>7-1. 이상 후보 표시 후 수동 제거 (편집 모드)</h2>
      <AnomalyCandidatePanel
        ready={!!processSummary}
        params={candidateParams}
        setParams={setCandidateParams}
        onRun={handleRunAnomalyCandidates}
        running={candidateRunning}
        result={candidateResult}
        error={candidateError}
        showOnMap={candidateShow}
        setShowOnMap={setCandidateShow}
        selectedIndices={selectedCandidateIndices}
        onToggle={handleToggleCandidate}
        onSelectAll={() =>
          setSelectedCandidateIndices(new Set((candidateResult?.candidates || []).map((_c, i) => i)))
        }
        onSelectClean={() =>
          setSelectedCandidateIndices(
            new Set(
              (candidateResult?.candidates || []).flatMap((c, i) => (c.single_line || c.at_coverage_edge ? [] : [i]))
            )
          )
        }
        onClearSelection={() => setSelectedCandidateIndices(new Set())}
        onSetNote={handleSetCandidateNote}
        onExportNotes={() =>
          api.exportCandidateNotesCsv(projectId, `${saveFilename || "project"}_candidate_decisions.csv`).catch(handleError)
        }
        onFocusCandidate={(c) => setFlyToTarget({ lat: c.lat, lon: c.lon, nonce: Date.now() })}
        onApplySelected={handleApplyCandidateSmoothing}
        applying={candidateApplying}
        removalMethod={removalMethod}
        setRemovalMethod={setRemovalMethod}
        sourceRemovals={processSummary?.source_removals || []}
        lastRemovalAdded={lastRemovalAdded}
        onUndoSourceRemoval={() => handleSourceRemovalHistory("undo")}
        onResetSourceRemovals={() => handleSourceRemovalHistory("reset")}
        removalWorking={removalWorking}
      />

      <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>7-2. 지상구조물 왜곡 자동탐지 (OSM + 신호분석)</h2>
      <StructureDistortionPanel
        ready={!!processSummary}
        params={structureScanParams}
        setParams={setStructureScanParams}
        onRun={handleRunStructureScan}
        running={structureScanRunning}
        result={structureScanResult}
        error={structureScanError}
        showOnMap={structureScanShow}
        setShowOnMap={setStructureScanShow}
        selectedPolygonIndices={selectedStructurePolygonIndices}
        onTogglePolygon={handleToggleStructurePolygon}
        selectedAnomalyIndices={selectedStructureAnomalyIndices}
        onToggleAnomaly={handleToggleStructureAnomaly}
        onApplySelected={handleApplyStructureSmoothing}
        applying={structureScanApplying}
      />

      {flightPathEditorOpen && (
        <FlightPathEditor
          points={editablePoints}
          loading={pointsLoading}
          lines={processSummary?.lines}
          onApply={handleManualExcludePointIds}
          onClose={() => setFlightPathEditorOpen(false)}
        />
      )}

      <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>12. 참조 레이어 (GeoTIFF · KML/KMZ · GeoJSON · CSV)</h2>
      <LayerManager
        layers={overlayLayers}
        onUpload={handleUploadOverlayImage}
        onToggleVisible={handleToggleOverlayVisible}
        onSetOpacity={handleSetOverlayOpacity}
        onMove={handleMoveOverlay}
        onRemove={handleRemoveOverlay}
        uploading={overlayUploading}
        error={overlayError}
        onRegisterTileFolder={handleRegisterLocalTileFolder}
        tileFolderRegistering={tileFolderRegistering}
        tileFolderError={tileFolderError}
        onPickTileFolder={handlePickTileFolder}
        onRegisterRaster={handleRegisterLocalRaster}
        onPickRasterFile={handlePickRasterFile}
      />
    </>
  );
}
