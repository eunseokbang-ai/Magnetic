import { useCallback, useMemo, useState } from "react";
import * as api from "./api";
import MapView from "./components/MapView";
import Legend from "./components/Legend";
import WorkflowSteps from "./components/WorkflowSteps";
import LineEditor from "./components/LineEditor";

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
  const [gridCellSize, setGridCellSize] = useState(10.0);
  const [gridding, setGridding] = useState(false);
  const [activeTransform, setActiveTransform] = useState("none");
  const [transformLoading, setTransformLoading] = useState(false);
  const [overlay, setOverlay] = useState(null);
  const [error, setError] = useState(null);

  const ensureProject = useCallback(async () => {
    if (projectId) return projectId;
    const res = await api.createProject();
    setProjectId(res.project_id);
    return res.project_id;
  }, [projectId]);

  const handleError = (e) => setError(e.message || String(e));

  const handleUploadDrone = async (file) => {
    try {
      setError(null);
      const id = await ensureProject();
      const summary = await api.uploadDrone(id, file);
      setDroneSummary(summary);
    } catch (e) {
      handleError(e);
    }
  };

  const handleUploadBase = async (file) => {
    try {
      setError(null);
      const id = await ensureProject();
      const summary = await api.uploadBase(id, file);
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
      const summary = await api.manualExclude(projectId, { mode: "polygon", action: "exclude", polygon: latlngCoords });
      setProcessSummary(summary);
      await refreshPoints(projectId, valueField);
      setOverlay(null);
    } catch (e) {
      handleError(e);
    }
  };

  const handleClearManual = async () => {
    if (!processSummary) return;
    try {
      setError(null);
      const allLineIds = processSummary.lines.map((l) => l.line_id);
      const summary = await api.manualExclude(projectId, { mode: "lines", action: "include", line_ids: allLineIds });
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
      const resp = await api.getGrid(projectId, { value: valueField, cell_size_m: gridCellSize });
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
      const resp = await api.getTransform(projectId, { transform: name, value: valueField, cell_size_m: gridCellSize });
      setOverlay(resp);
      setActiveTransform(name);
    } catch (e) {
      handleError(e);
    } finally {
      setTransformLoading(false);
    }
  };

  const colorRange = useMemo(() => {
    const stats = valueField === "anomaly" ? processSummary?.anomaly_stats : processSummary?.tmi_stats;
    if (stats && stats.min != null) return { vmin: stats.min, vmax: stats.max };
    if (points.length > 0) {
      const values = points.map((p) => p.value);
      return { vmin: Math.min(...values), vmax: Math.max(...values) };
    }
    return { vmin: 0, vmax: 1 };
  }, [processSummary, valueField, points]);

  const legendStats = overlay?.stats || (valueField === "anomaly" ? processSummary?.anomaly_stats : processSummary?.tmi_stats);
  const legendLabel = activeTransform !== "none" ? activeTransform.toUpperCase() : valueField === "anomaly" ? "자력 이상" : "TMI";
  const legendCmapKind = valueField === "anomaly" || activeTransform !== "none" ? "anomaly" : "tmi";
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
          valueField={valueField}
          colorRange={colorRange}
          cmapKind={valueField === "anomaly" ? "anomaly" : "tmi"}
          overlay={overlay}
          onHoverPoint={setHoverPoint}
          drawMode={drawMode}
          onShapeDrawn={handleShapeDrawn}
        />
      </div>

      <div style={{ width: 280, borderLeft: "1px solid #e5e7eb", overflowY: "auto", padding: 12, background: "#f9fafb" }}>
        <h2 style={{ fontSize: 13, margin: "4px 0 10px 0" }}>7. 수동 측선 편집</h2>
        <LineEditor
          lines={processSummary?.lines}
          onToggleLines={handleToggleLines}
          drawMode={drawMode}
          onToggleDrawMode={() => setDrawMode((v) => !v)}
          onClearPolygonExclusions={handleClearManual}
          nExcludedManual={processSummary?.n_excluded_manual}
        />

        <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>범례</h2>
        <Legend
          label={legendLabel}
          unit="nT"
          vmin={legendRange.vmin}
          vmax={legendRange.vmax}
          cmapKind={legendCmapKind}
          stats={legendStats}
          hoverPoint={hoverPoint}
        />
      </div>
    </div>
  );
}
