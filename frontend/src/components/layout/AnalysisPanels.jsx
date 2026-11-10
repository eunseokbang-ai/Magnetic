import { Suspense } from "react";
import * as api from "../../api";
import { useApp } from "../../AppContext";
import ChatPanel from "../ChatPanel";
import ContactPanel from "../ContactPanel";
import DepthEstimationPanel from "../DepthEstimationPanel";
import EulerPanel from "../EulerPanel";
import GeologyStatsPanel from "../GeologyStatsPanel";
import InversionPanel from "../InversionPanel";
import LineamentPanel from "../LineamentPanel";
import OfflineMapPanel from "../OfflineMapPanel";
import ProspectivityPanel from "../ProspectivityPanel";
import QcCertificatePanel from "../QcCertificatePanel";
import SurveyDesignPanel from "../SurveyDesignPanel";
import TargetDetectionPanel from "../TargetDetectionPanel";
import { GuidelinePanel } from "./shared";

// Right column, second part: interpretation - 3D inversion, Euler, lineaments,
// depth estimates, contacts, prospectivity, geology statistics - and the tools
// around it: survey design, guideline diagnostics, target detection, QC
// certificate, the assistant, offline maps.
// A region of App's screen; it reads App's state with useApp().
export default function AnalysisPanels() {
  const {
    asDepthError, asDepthParams, asDepthResult, asDepthRunning, autoParams, boundaryPolygon, chatError,
    chatMessages, chatSending, contactError, contactParams, contactResult, contactRunning, demStatus,
    demUploading, depthEstimationCellSize, eulerError, eulerFlightAglM, eulerMaxUncertaintyPct, eulerResult,
    eulerRunning, eulerStructuralIndex, eulerWindowSize, geologyDrawMode, geologyStatsError,
    geologyStatsParams, geologyStatsResult, geologyStatsRunning, geologyUnits, guidelinePanelOpened,
    handleAddNotchFrequency, handleAnalyzeRepeatability, handleClearDem, handleDeleteGeologyUnit,
    handleDownloadTiles, handleError, handleEstimateTiles, handleExportInversion, handleExportInversionCsv,
    handleImportInversion, handleOpenVolume, handleRefreshTileStatus, handleRemoveNotchFrequency,
    handleRunAsDepth, handleRunContactDetection, handleRunEuler, handleRunFixedSection,
    handleRunGeologyStats, handleRunInversion, handleRunLineamentExtraction, handleRunMultiscaleEdges,
    handleRunProspectivity, handleRunQcCertificate, handleRunSpectralDepth, handleRunTargetDetection,
    handleRunTiltDepth, handleSendChatMessage, handleShowInversionSlice, handleToggleGeologyDrawMode,
    handleToggleSectionDrawMode, handleUpdateGeologyUnit, handleUploadDem, handleUploadRepeatability,
    handleViewSpectrum, inversionError, inversionParams, inversionRunning, inversionSummary, lineamentError,
    lineamentParams, lineamentResult, lineamentRunning, mapBounds, multiscaleError, multiscaleHeightsText,
    multiscalePercentile, multiscaleResult, multiscaleRunning, processParams, processSummary, projectId,
    prospectivityError, prospectivityParams, prospectivityResult, prospectivityRunning, purposeMode,
    qcCertificateError, qcCertificateParams, qcCertificateResult, qcCertificateRunning,
    repeatabilityAnalyzing, repeatabilityError, repeatabilityResult, repeatabilityUploadInfo,
    repeatabilityUploading, saveFilename, sectionDrawMode, sectionLoading, sectionPositionFrac,
    sectionProfile, sectionThreshold, sectionThresholdMax, setAsDepthParams, setAutoParams, setContactParams,
    setDepthEstimationCellSize, setEulerFlightAglM, setEulerMaxUncertaintyPct, setEulerStructuralIndex,
    setEulerWindowSize, setGeologyStatsParams, setGuidelinePanelOpened, setInversionParams,
    setLineamentParams, setMultiscaleHeightsText, setMultiscalePercentile, setProspectivityParams,
    setQcCertificateParams, setSectionPositionFrac, setSectionProfile, setSectionThreshold,
    setSectionThresholdMax, setShowAsDepth, setShowContacts, setShowDetectedTargets, setShowEulerSolutions,
    setShowGeologyUnits, setShowLineaments, setShowMultiscaleLayer, setShowProspectivityOverlay,
    setShowProspectivityTargets, setShowTiltDepth, setSliceLayerIndex, setSliceThreshold,
    setSliceThresholdMax, setSpectrumLineId, setSpectrumValue, setTargetDetectionParams, setTiltDepthParams,
    setVolumeThreshold, setVolumeThresholdMax, showAsDepth, showContacts, showDetectedTargets,
    showEulerSolutions, showGeneratedLayer, showGeologyUnits, showLineaments, showMultiscaleLayer,
    showProspectivityOverlay, showProspectivityTargets, showTiltDepth, sliceLayerIndex, sliceThreshold,
    sliceThresholdMax, spectralDepthError, spectralDepthResult, spectralDepthRunning, spectrumError,
    spectrumLineId, spectrumLoading, spectrumResult, spectrumValue, targetDetectionError,
    targetDetectionParams, targetDetectionResult, targetDetectionRunning, tileDownloadResult, tileError,
    tileEstimate, tileLoading, tileStatus, tiltDepthError, tiltDepthParams, tiltDepthResult,
    tiltDepthRunning, volumeThreshold, volumeThresholdMax,
  } = useApp();

  return (
    <>
      <details id="wf-section-advanced" open={purposeMode === "mineral"} style={{ marginBottom: 4 }}>
        <summary style={{ fontSize: 13, fontWeight: 600, cursor: "pointer", padding: "4px 0" }}>
          13~14. 광물자원탐사용 고급 분석 (3차원 역산 · 오일러 디컨볼루션){purposeMode === "target" && " — 근지표 표적탐지에는 보통 불필요"}
        </summary>
        <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>13. 3차원 역산 (실험적)</h2>
        <InversionPanel
          ready={!!processSummary}
          demStatus={demStatus}
          demUploading={demUploading}
          onUploadDem={handleUploadDem}
          onClearDem={handleClearDem}
          params={inversionParams}
          setParams={setInversionParams}
          autoParams={autoParams}
          setAutoParams={setAutoParams}
          onRun={handleRunInversion}
          running={inversionRunning}
          summary={inversionSummary}
          error={inversionError}
          sliceLayerIndex={sliceLayerIndex}
          setSliceLayerIndex={setSliceLayerIndex}
          sliceThreshold={sliceThreshold}
          setSliceThreshold={setSliceThreshold}
          sliceThresholdMax={sliceThresholdMax}
          setSliceThresholdMax={setSliceThresholdMax}
          onShowSlice={handleShowInversionSlice}
          sectionProfile={sectionProfile}
          setSectionProfile={setSectionProfile}
          sectionPositionFrac={sectionPositionFrac}
          setSectionPositionFrac={setSectionPositionFrac}
          sectionDrawMode={sectionDrawMode}
          onToggleSectionDrawMode={handleToggleSectionDrawMode}
          onRunFixedSection={handleRunFixedSection}
          sectionThreshold={sectionThreshold}
          setSectionThreshold={setSectionThreshold}
          sectionThresholdMax={sectionThresholdMax}
          setSectionThresholdMax={setSectionThresholdMax}
          sectionLoading={sectionLoading}
          onOpenVolume={handleOpenVolume}
          volumeThreshold={volumeThreshold}
          setVolumeThreshold={setVolumeThreshold}
          volumeThresholdMax={volumeThresholdMax}
          setVolumeThresholdMax={setVolumeThresholdMax}
          onExportInversion={handleExportInversion}
          onExportInversionCsv={handleExportInversionCsv}
          onImportInversion={handleImportInversion}
          geologyUnits={geologyUnits}
          geologyDrawMode={geologyDrawMode}
          onToggleGeologyDrawMode={handleToggleGeologyDrawMode}
          onUpdateGeologyUnit={handleUpdateGeologyUnit}
          onDeleteGeologyUnit={handleDeleteGeologyUnit}
        />

        <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>14. 오일러 디컨볼루션 (빠른 심도 추정)</h2>
        <EulerPanel
          ready={!!processSummary}
          structuralIndex={eulerStructuralIndex}
          setStructuralIndex={setEulerStructuralIndex}
          windowSize={eulerWindowSize}
          setWindowSize={setEulerWindowSize}
          maxUncertaintyPct={eulerMaxUncertaintyPct}
          setMaxUncertaintyPct={setEulerMaxUncertaintyPct}
          flightAglM={eulerFlightAglM}
          setFlightAglM={setEulerFlightAglM}
          onRun={handleRunEuler}
          running={eulerRunning}
          result={eulerResult}
          error={eulerError}
          showSolutions={showEulerSolutions}
          setShowSolutions={setShowEulerSolutions}
        />
      </details>

      <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>14-3. 자력 리니어먼트 추출 + 로즈다이어그램</h2>
      <LineamentPanel
        ready={!!processSummary}
        params={lineamentParams}
        setParams={setLineamentParams}
        onRun={handleRunLineamentExtraction}
        running={lineamentRunning}
        result={lineamentResult}
        error={lineamentError}
        showOnMap={showLineaments}
        setShowOnMap={setShowLineaments}
      />

      <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>14-4. 빠른 심도추정 (틸트각 / AS / 스펙트럼)</h2>
      <DepthEstimationPanel
        ready={!!processSummary}
        cellSizeM={depthEstimationCellSize}
        setCellSizeM={setDepthEstimationCellSize}
        tilt={{
          params: tiltDepthParams,
          setParams: setTiltDepthParams,
          onRun: handleRunTiltDepth,
          running: tiltDepthRunning,
          result: tiltDepthResult,
          error: tiltDepthError,
          showOnMap: showTiltDepth,
          setShowOnMap: setShowTiltDepth,
        }}
        analyticSignal={{
          params: asDepthParams,
          setParams: setAsDepthParams,
          onRun: handleRunAsDepth,
          running: asDepthRunning,
          result: asDepthResult,
          error: asDepthError,
          showOnMap: showAsDepth,
          setShowOnMap: setShowAsDepth,
        }}
        spectral={{
          onRun: handleRunSpectralDepth,
          running: spectralDepthRunning,
          result: spectralDepthResult,
          error: spectralDepthError,
        }}
      />

      <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>14-5. 자성 접촉면 탐지 (Magnetic Contact Detection)</h2>
      <ContactPanel
        ready={!!processSummary}
        params={contactParams}
        setParams={setContactParams}
        onRun={handleRunContactDetection}
        running={contactRunning}
        result={contactResult}
        error={contactError}
        showOnMap={showContacts}
        setShowOnMap={setShowContacts}
      />

      <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>14-6. 프로스펙티비티 분석 (광물 타깃 스코어링)</h2>
      <ProspectivityPanel
        ready={!!processSummary}
        params={prospectivityParams}
        setParams={setProspectivityParams}
        onRun={handleRunProspectivity}
        running={prospectivityRunning}
        result={prospectivityResult}
        error={prospectivityError}
        showTargetsOnMap={showProspectivityTargets}
        setShowTargetsOnMap={setShowProspectivityTargets}
        showOverlay={showProspectivityOverlay}
        setShowOverlay={setShowProspectivityOverlay}
      />

      <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>14-7. 지질 단위별 자력이상 통계 (KIGAM 지질도 교차 집계)</h2>
      <GeologyStatsPanel
        ready={!!processSummary}
        params={geologyStatsParams}
        setParams={setGeologyStatsParams}
        onRun={handleRunGeologyStats}
        running={geologyStatsRunning}
        result={geologyStatsResult}
        error={geologyStatsError}
        showOnMap={showGeologyUnits}
        setShowOnMap={setShowGeologyUnits}
        onExportCsv={() =>
          api.exportGeologyStatsCsv(projectId, `${saveFilename || "project"}_geology_units.csv`).catch(handleError)
        }
      />

      <details style={{ marginBottom: 10 }}>
        <summary style={{ cursor: "pointer", fontSize: 13, fontWeight: 600, color: "#4a3d28" }}>
          0. 탐사 설계 도우미 (권장 측선 간격 · 표적 탐지 가능성 · 비행 계획 KML)
        </summary>
        <SurveyDesignPanel mapBounds={mapBounds} boundaryPolygon={boundaryPolygon} onPreview={showGeneratedLayer} />
      </details>

      <details style={{ marginBottom: 10 }} onToggle={(e) => e.target.open && setGuidelinePanelOpened(true)}>
        <summary style={{ cursor: "pointer", fontSize: 13, fontWeight: 600, color: "#4a3d28" }}>
          15. UAV 자력탐사 가이드라인 진단 도구 (반복측선 · 파워스펙트럼/노치필터 · 멀티스케일 엣지)
        </summary>
        {guidelinePanelOpened && (
        <Suspense fallback={<div style={{ fontSize: 12, color: "#8a7a5c", padding: 8 }}>불러오는 중...</div>}>
        <GuidelinePanel
          ready={!!processSummary}
          repeatability={{
            onUploadFiles: handleUploadRepeatability,
            uploading: repeatabilityUploading,
            uploadInfo: repeatabilityUploadInfo,
            onAnalyze: handleAnalyzeRepeatability,
            analyzing: repeatabilityAnalyzing,
            result: repeatabilityResult,
            error: repeatabilityError,
          }}
          spectrum={{
            lineOptions: (processSummary?.lines || []).map((l) => l.line_id),
            selectedLineId: spectrumLineId,
            setSelectedLineId: setSpectrumLineId,
            spectrumValue,
            setSpectrumValue,
            onView: handleViewSpectrum,
            loading: spectrumLoading,
            result: spectrumResult,
            error: spectrumError,
            notchFrequencies: processParams.notch_filter.frequencies_hz,
            onAddNotch: handleAddNotchFrequency,
            onRemoveNotch: handleRemoveNotchFrequency,
          }}
          multiscaleEdges={{
            heightsText: multiscaleHeightsText,
            setHeightsText: setMultiscaleHeightsText,
            percentile: multiscalePercentile,
            setPercentile: setMultiscalePercentile,
            onRun: handleRunMultiscaleEdges,
            running: multiscaleRunning,
            result: multiscaleResult,
            error: multiscaleError,
            showLayer: showMultiscaleLayer,
            setShowLayer: setShowMultiscaleLayer,
          }}
        />
        </Suspense>
        )}
      </details>

      <details open={purposeMode === "target"} style={{ marginBottom: 4 }}>
        <summary style={{ fontSize: 13, fontWeight: 600, cursor: "pointer", padding: "4px 0" }}>
          🎯 근지표 표적탐지 (지뢰·불발탄·은닉 차량 등){purposeMode === "mineral" && " — 광물자원탐사에는 보통 불필요"}
        </summary>
        <div style={{ marginTop: 8 }}>
          <TargetDetectionPanel
            ready={!!processSummary}
            params={targetDetectionParams}
            setParams={setTargetDetectionParams}
            onRun={handleRunTargetDetection}
            running={targetDetectionRunning}
            result={targetDetectionResult}
            error={targetDetectionError}
            showTargets={showDetectedTargets}
            setShowTargets={setShowDetectedTargets}
            projectId={projectId}
            exportTargetsCsv={api.exportTargetsCsv}
            exportTargetsShapefile={api.exportTargetsShapefile}
          />
        </div>
      </details>

      <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>14-1. 표준 QC 인증서</h2>
      <QcCertificatePanel
        ready={!!processSummary}
        params={qcCertificateParams}
        setParams={setQcCertificateParams}
        onRun={handleRunQcCertificate}
        running={qcCertificateRunning}
        result={qcCertificateResult}
        error={qcCertificateError}
      />

      <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>15. AI 해석 도우미 (챗봇)</h2>
      <ChatPanel ready={!!processSummary} messages={chatMessages} onSend={handleSendChatMessage} sending={chatSending} error={chatError} />

      <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>16. 오프라인 지도 (인터넷 없이 배경지도 사용)</h2>
      <OfflineMapPanel
        mapBounds={mapBounds}
        onEstimate={handleEstimateTiles}
        onDownload={handleDownloadTiles}
        onRefreshStatus={handleRefreshTileStatus}
        status={tileStatus}
        estimate={tileEstimate}
        downloadResult={tileDownloadResult}
        loading={tileLoading}
        error={tileError}
      />
    </>
  );
}
