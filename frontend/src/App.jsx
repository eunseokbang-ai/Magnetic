import { useCallback, useMemo, useRef, useState } from "react";
import * as api from "./api";
import { minMax } from "./arrayUtils";
import MapView from "./components/MapView";
import Legend from "./components/Legend";
import WorkflowSteps from "./components/WorkflowSteps";
import LineEditor from "./components/LineEditor";
import LayerManager from "./components/LayerManager";
import InversionPanel from "./components/InversionPanel";
import InversionVolumeView from "./components/InversionVolumeView";
import InversionSectionView from "./components/InversionSectionView";

const DEFAULT_INVERSION_PARAMS = {
  obs_cell_size_m: 30.0,
  depth_extent_m: 150.0,
  n_layers: 8,
  assumed_agl_m: 50.0,
  regularization_strength: 1.0,
  n_irls_iterations: 6,
};

const DEFAULT_PARAMS = {
  filter_cutoff_hz: 1.0,
  line_params: {
    heading_lag_seconds: 1.0,
    heading_tolerance_deg: 20.0,
    min_speed_mps: 1.5,
    min_line_length_m: 150.0,
    turn_buffer_m: 15.0,
    max_gap_seconds: 1.0,
  },
  diurnal_params: {
    time_offset_seconds: 0.0,
    reference: "mean",
  },
  heading_correction: {
    enabled: true,
    quiet_percentile: 40.0,
    max_match_distance_m: null,
  },
};

export default function App() {
  const [projectId, setProjectId] = useState(null);
  const [droneSummary, setDroneSummary] = useState(null);
  const [baseSummary, setBaseSummary] = useState(null);
  const [processParams, setProcessParams] = useState(DEFAULT_PARAMS);
  const [processSummary, setProcessSummary] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [points, setPoints] = useState([]);
  const [valueField, setValueField] = useState("anomaly");
  const [hoverPoint, setHoverPoint] = useState(null);
  const [drawMode, setDrawMode] = useState(false);
  const [drawAction, setDrawAction] = useState("exclude");
  const [gridCellSize, setGridCellSize] = useState(10.0);
  const [gridMethod, setGridMethod] = useState("nearest");
  const [gridMaxDistance, setGridMaxDistance] = useState(null);
  const [gridding, setGridding] = useState(false);
  const [activeTransform, setActiveTransform] = useState("none");
  const [transformLoading, setTransformLoading] = useState(false);
  const [overlay, setOverlay] = useState(null);
  const [error, setError] = useState(null);
  const [cmapName, setCmapName] = useState("RdYlBu_r");
  const [manualRange, setManualRange] = useState({ enabled: false, vmin: null, vmax: null });
  const [gridOpacity, setGridOpacity] = useState(0.85);
  const [showPointsOverGrid, setShowPointsOverGrid] = useState(true);
  const [showLineLabels, setShowLineLabels] = useState(false);
  const [overlayLayers, setOverlayLayers] = useState([]);
  const [overlayUploading, setOverlayUploading] = useState(false);
  const [overlayError, setOverlayError] = useState(null);

  const [demStatus, setDemStatus] = useState(null);
  const [demUploading, setDemUploading] = useState(false);
  const [autoParams, setAutoParams] = useState(true);
  const [inversionParams, setInversionParams] = useState(DEFAULT_INVERSION_PARAMS);
  const [inversionRunning, setInversionRunning] = useState(false);
  const [inversionSummary, setInversionSummary] = useState(null);
  const [inversionError, setInversionError] = useState(null);
  const [sliceLayerIndex, setSliceLayerIndex] = useState(0);
  const [sliceThreshold, setSliceThreshold] = useState("");
  const [sliceThresholdMax, setSliceThresholdMax] = useState("");
  const [sectionProfile, setSectionProfile] = useState("custom");
  const [sectionPositionFrac, setSectionPositionFrac] = useState(0.5);
  const [sectionDrawMode, setSectionDrawMode] = useState(false);
  const [sectionThreshold, setSectionThreshold] = useState("");
  const [sectionThresholdMax, setSectionThresholdMax] = useState("");
  const [sectionResult, setSectionResult] = useState(null);
  const [sectionLoading, setSectionLoading] = useState(false);
  const [volumeThreshold, setVolumeThreshold] = useState("");
  const [volumeThresholdMax, setVolumeThresholdMax] = useState("");
  const [volumeData, setVolumeData] = useState(null);

  // Drone and base uploads can both fire ensureProject() before the
  // projectId state update from the first call has re-rendered, which
  // would otherwise create two separate backend projects. Sharing the
  // in-flight creation promise keeps both uploads on the same project.
  const projectCreationRef = useRef(null);
  const ensureProject = useCallback(async () => {
    if (projectId) return projectId;
    if (!projectCreationRef.current) {
      projectCreationRef.current = api.createProject().then((res) => {
        setProjectId(res.project_id);
        return res.project_id;
      });
    }
    return projectCreationRef.current;
  }, [projectId]);

  const handleError = (e) => setError(e.message || String(e));

  const handleUploadDrone = async (files) => {
    try {
      setError(null);
      const id = await ensureProject();
      const summary = await api.uploadDrone(id, files);
      setDroneSummary(summary);
    } catch (e) {
      handleError(e);
    }
  };

  const handleUploadBase = async (files) => {
    try {
      setError(null);
      const id = await ensureProject();
      const summary = await api.uploadBase(id, files);
      setBaseSummary(summary);
    } catch (e) {
      handleError(e);
    }
  };

  const refreshPoints = useCallback(async (id, value) => {
    const pts = await api.getPoints(id, value);
    setPoints(pts);
  }, []);

  const handleProcess = async () => {
    try {
      setError(null);
      setProcessing(true);
      const summary = await api.processProject(projectId, processParams);
      setProcessSummary(summary);
      setOverlay(null);
      setActiveTransform("none");
      setInversionSummary(null);
      setSectionResult(null);
      setVolumeData(null);
      await refreshPoints(projectId, valueField);
    } catch (e) {
      handleError(e);
    } finally {
      setProcessing(false);
    }
  };

  const handleSetValueField = async (value) => {
    setValueField(value);
    if (projectId && processSummary) {
      try {
        await refreshPoints(projectId, value);
      } catch (e) {
        handleError(e);
      }
    }
  };

  const handleToggleLines = async (lineIds, action) => {
    try {
      setError(null);
      const summary = await api.manualExclude(projectId, { mode: "lines", action, line_ids: lineIds });
      setProcessSummary(summary);
      await refreshPoints(projectId, valueField);
      setOverlay(null);
    } catch (e) {
      handleError(e);
    }
  };

  const handleShapeDrawn = async (latlngCoords) => {
    try {
      setError(null);
      const summary = await api.manualExclude(projectId, { mode: "polygon", action: drawAction, polygon: latlngCoords });
      setProcessSummary(summary);
      await refreshPoints(projectId, valueField);
      setOverlay(null);
    } catch (e) {
      handleError(e);
    }
  };

  const handleResetManual = async () => {
    if (!processSummary) return;
    try {
      setError(null);
      const summary = await api.manualExclude(projectId, { mode: "reset" });
      setProcessSummary(summary);
      await refreshPoints(projectId, valueField);
      setOverlay(null);
    } catch (e) {
      handleError(e);
    }
  };

  const handleGrid = async () => {
    try {
      setError(null);
      setGridding(true);
      const resp = await api.getGrid(projectId, {
        value: valueField,
        cell_size_m: gridCellSize,
        method: gridMethod,
        max_distance_m: gridMaxDistance,
        cmap: cmapName,
        vmin: manualRange.enabled ? manualRange.vmin : null,
        vmax: manualRange.enabled ? manualRange.vmax : null,
      });
      setOverlay(resp);
      setActiveTransform("none");
    } catch (e) {
      handleError(e);
    } finally {
      setGridding(false);
    }
  };

  const handleTransform = async (name) => {
    try {
      setError(null);
      if (name === "none") {
        setOverlay(null);
        setActiveTransform("none");
        return;
      }
      setTransformLoading(true);
      const resp = await api.getTransform(projectId, {
        transform: name,
        value: valueField,
        cell_size_m: gridCellSize,
        method: gridMethod,
        max_distance_m: gridMaxDistance,
        cmap: cmapName,
        vmin: manualRange.enabled ? manualRange.vmin : null,
        vmax: manualRange.enabled ? manualRange.vmax : null,
      });
      setOverlay(resp);
      setActiveTransform(name);
    } catch (e) {
      handleError(e);
    } finally {
      setTransformLoading(false);
    }
  };

  const handleUploadOverlayImage = async (file) => {
    try {
      setOverlayError(null);
      setOverlayUploading(true);
      const resp = await api.uploadOverlayImage(file);
      setOverlayLayers((prev) => [
        ...prev,
        { id: `${Date.now()}-${Math.random()}`, name: resp.name, image_data_url: resp.image_data_url, bounds: resp.bounds, opacity: 0.8, visible: true },
      ]);
    } catch (e) {
      setOverlayError(e.message || String(e));
    } finally {
      setOverlayUploading(false);
    }
  };

  const handleToggleOverlayVisible = (id, visible) =>
    setOverlayLayers((prev) => prev.map((l) => (l.id === id ? { ...l, visible } : l)));

  const handleSetOverlayOpacity = (id, opacity) =>
    setOverlayLayers((prev) => prev.map((l) => (l.id === id ? { ...l, opacity } : l)));

  const handleRemoveOverlay = (id) => setOverlayLayers((prev) => prev.filter((l) => l.id !== id));

  const handleMoveOverlay = (id, direction) => {
    setOverlayLayers((prev) => {
      const idx = prev.findIndex((l) => l.id === id);
      if (idx === -1) return prev;
      const swapWith = direction === "up" ? idx + 1 : idx - 1;
      if (swapWith < 0 || swapWith >= prev.length) return prev;
      const next = [...prev];
      [next[idx], next[swapWith]] = [next[swapWith], next[idx]];
      return next;
    });
  };

  const handleUploadDem = async (file) => {
    try {
      setInversionError(null);
      setDemUploading(true);
      const resp = await api.uploadDem(projectId, file);
      setDemStatus(resp);
    } catch (e) {
      setInversionError(e.message || String(e));
    } finally {
      setDemUploading(false);
    }
  };

  const handleClearDem = async () => {
    try {
      await api.clearDem(projectId);
      setDemStatus(null);
    } catch (e) {
      setInversionError(e.message || String(e));
    }
  };

  const handleRunInversion = async () => {
    try {
      setInversionError(null);
      setInversionRunning(true);
      const runParams = autoParams
        ? { ...inversionParams, obs_cell_size_m: null, depth_extent_m: null, n_layers: null }
        : inversionParams;
      const resp = await api.runInversion(projectId, runParams);
      setInversionSummary(resp);
      setSliceLayerIndex(0);
      setSectionResult(null);
      setVolumeData(null);
    } catch (e) {
      setInversionError(e.message || String(e));
    } finally {
      setInversionRunning(false);
    }
  };

  const handleShowInversionSlice = async () => {
    try {
      setError(null);
      const resp = await api.getInversionSlice(projectId, {
        layer_index: sliceLayerIndex,
        threshold: sliceThreshold === "" ? null : sliceThreshold,
        threshold_max: sliceThresholdMax === "" ? null : sliceThresholdMax,
      });
      setOverlay(resp);
      setActiveTransform("inversion_slice");
    } catch (e) {
      handleError(e);
    }
  };

  const handleToggleSectionDrawMode = () => setSectionDrawMode((v) => !v);

  const handleSectionPathDrawn = async (latlngCoords) => {
    setSectionDrawMode(false);
    try {
      setInversionError(null);
      setSectionLoading(true);
      const resp = await api.getInversionSection(projectId, {
        profile: "custom",
        path: latlngCoords,
        threshold: sectionThreshold === "" ? null : sectionThreshold,
        threshold_max: sectionThresholdMax === "" ? null : sectionThresholdMax,
      });
      setSectionResult(resp);
    } catch (e) {
      setInversionError(e.message || String(e));
    } finally {
      setSectionLoading(false);
    }
  };

  const handleRunFixedSection = async () => {
    try {
      setInversionError(null);
      setSectionLoading(true);
      const resp = await api.getInversionSection(projectId, {
        profile: sectionProfile,
        position_frac: sectionPositionFrac,
        threshold: sectionThreshold === "" ? null : sectionThreshold,
        threshold_max: sectionThresholdMax === "" ? null : sectionThresholdMax,
      });
      setSectionResult(resp);
    } catch (e) {
      setInversionError(e.message || String(e));
    } finally {
      setSectionLoading(false);
    }
  };

  const handleMapShapeDrawn = (coords) => {
    if (sectionDrawMode) {
      handleSectionPathDrawn(coords);
    } else {
      handleShapeDrawn(coords);
    }
  };

  const handleOpenVolume = async () => {
    try {
      setInversionError(null);
      const threshold = volumeThreshold === "" ? null : volumeThreshold;
      const thresholdMax = volumeThresholdMax === "" ? null : volumeThresholdMax;
      const resp = await api.getInversionVolume(projectId, threshold, thresholdMax);
      setVolumeData({ ...resp, threshold, thresholdMax });
    } catch (e) {
      setInversionError(e.message || String(e));
    }
  };

  const autoColorRange = useMemo(() => {
    const stats = valueField === "anomaly" ? processSummary?.anomaly_stats : processSummary?.tmi_stats;
    if (stats && stats.min != null) return { vmin: stats.min, vmax: stats.max };
    if (points.length > 0) {
      const values = points.map((p) => p.value);
      const [vmin, vmax] = minMax(values);
      return { vmin, vmax };
    }
    return { vmin: 0, vmax: 1 };
  }, [processSummary, valueField, points]);

  const colorRange = useMemo(() => {
    if (manualRange.enabled && manualRange.vmin != null && manualRange.vmax != null) {
      return { vmin: manualRange.vmin, vmax: manualRange.vmax };
    }
    return autoColorRange;
  }, [manualRange, autoColorRange]);

  const legendStats = overlay?.stats || (valueField === "anomaly" ? processSummary?.anomaly_stats : processSummary?.tmi_stats);
  const legendLabel =
    activeTransform === "inversion_slice"
      ? "역산 자화율 (SI)"
      : activeTransform !== "none"
        ? activeTransform.toUpperCase()
        : valueField === "anomaly"
          ? "자력 이상"
          : "TMI";
  const legendRange = overlay ? { vmin: overlay.vmin, vmax: overlay.vmax } : colorRange;

  return (
    <div style={{ display: "flex", height: "100%", fontFamily: "system-ui, sans-serif" }}>
      <div style={{ width: 320, borderRight: "1px solid #e5e7eb", overflowY: "auto", padding: 12, background: "#f9fafb" }}>
        <h1 style={{ fontSize: 16, margin: "4px 0 12px 0" }}>드론 자력탐사 자료 처리</h1>
        <WorkflowSteps
          onUploadDrone={handleUploadDrone}
          droneSummary={droneSummary}
          onUploadBase={handleUploadBase}
          baseSummary={baseSummary}
          processParams={processParams}
          setProcessParams={setProcessParams}
          onProcess={handleProcess}
          processSummary={processSummary}
          processing={processing}
          gridCellSize={gridCellSize}
          setGridCellSize={setGridCellSize}
          gridMethod={gridMethod}
          setGridMethod={setGridMethod}
          gridMaxDistance={gridMaxDistance}
          setGridMaxDistance={setGridMaxDistance}
          gridOpacity={gridOpacity}
          setGridOpacity={setGridOpacity}
          onGrid={handleGrid}
          gridding={gridding}
          activeTransform={activeTransform}
          onTransform={handleTransform}
          transformLoading={transformLoading}
          valueField={valueField}
          setValueField={handleSetValueField}
          error={error}
        />
      </div>

      <div style={{ flex: 1, position: "relative" }}>
        <MapView
          points={points}
          colorRange={colorRange}
          cmapName={cmapName}
          overlay={overlay}
          gridOpacity={gridOpacity}
          showPointsOverGrid={showPointsOverGrid}
          overlayLayers={overlayLayers}
          lines={processSummary?.lines}
          showLineLabels={showLineLabels}
          onHoverPoint={setHoverPoint}
          drawMode={drawMode || sectionDrawMode}
          drawShapeType={sectionDrawMode ? "polyline" : "polygon"}
          onShapeDrawn={handleMapShapeDrawn}
        />
        {volumeData && <InversionVolumeView data={volumeData} onClose={() => setVolumeData(null)} />}
        {!volumeData && sectionResult && <InversionSectionView data={sectionResult} onClose={() => setSectionResult(null)} />}
      </div>

      <div style={{ width: 280, borderLeft: "1px solid #e5e7eb", overflowY: "auto", padding: 12, background: "#f9fafb" }}>
        <h2 style={{ fontSize: 13, margin: "4px 0 10px 0" }}>7. 수동 측선 편집</h2>
        <LineEditor
          lines={processSummary?.lines}
          onToggleLines={handleToggleLines}
          drawMode={drawMode}
          drawAction={drawAction}
          onSetDrawAction={setDrawAction}
          onToggleDrawMode={() => setDrawMode((v) => !v)}
          onResetManual={handleResetManual}
          nManualIncluded={processSummary?.n_manual_included}
          nManualExcluded={processSummary?.n_manual_excluded}
          showPointsOverGrid={showPointsOverGrid}
          onToggleShowPointsOverGrid={setShowPointsOverGrid}
          showLineLabels={showLineLabels}
          onToggleShowLineLabels={setShowLineLabels}
        />

        <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>12. 참조 레이어 (지질도 등 GeoTIFF)</h2>
        <LayerManager
          layers={overlayLayers}
          onUpload={handleUploadOverlayImage}
          onToggleVisible={handleToggleOverlayVisible}
          onSetOpacity={handleSetOverlayOpacity}
          onMove={handleMoveOverlay}
          onRemove={handleRemoveOverlay}
          uploading={overlayUploading}
          error={overlayError}
        />

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
        />

        <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>범례</h2>
        <Legend
          label={legendLabel}
          unit="nT"
          vmin={legendRange.vmin}
          vmax={legendRange.vmax}
          cmapName={cmapName}
          onCmapChange={setCmapName}
          manualRange={manualRange}
          onManualRangeChange={setManualRange}
          stats={legendStats}
          hoverPoint={hoverPoint}
        />
      </div>
    </div>
  );
}
