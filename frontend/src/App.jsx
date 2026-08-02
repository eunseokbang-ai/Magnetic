import { lazy, Suspense, useCallback, useMemo, useRef, useState } from "react";
import * as api from "./api";
import { minMax } from "./arrayUtils";
import MapView from "./components/MapView";
import Legend from "./components/Legend";
import WorkflowSteps from "./components/WorkflowSteps";
import LineEditor from "./components/LineEditor";
import FlightPathEditor from "./components/FlightPathEditor";
import LayerManager from "./components/LayerManager";
import InversionPanel from "./components/InversionPanel";
import InversionSectionView from "./components/InversionSectionView";
import LineProfileView from "./components/LineProfileView";

// plotly.js-dist-min alone is ~4.7MB unminified - only the 3D volume view
// needs it, and most sessions never open it, so it's split into its own
// chunk and fetched on demand instead of bloating the initial bundle.
const InversionVolumeView = lazy(() => import("./components/InversionVolumeView"));
import EulerPanel from "./components/EulerPanel";
import WorkflowProgress from "./components/WorkflowProgress";
import ChatPanel from "./components/ChatPanel";
import TargetDetectionPanel from "./components/TargetDetectionPanel";

const toggleButtonStyle = {
  width: 36,
  height: 36,
  borderRadius: 8,
  border: "1px solid #d1d5db",
  background: "white",
  color: "#111827",
  fontSize: 16,
  cursor: "pointer",
  boxShadow: "0 1px 4px rgba(0,0,0,0.2)",
};

const DEFAULT_INVERSION_PARAMS = {
  obs_cell_size_m: 30.0,
  depth_extent_m: 150.0,
  n_layers: 8,
  assumed_agl_m: 50.0,
  regularization_strength: 1.0,
  n_irls_iterations: 6,
};

const DEFAULT_TARGET_DETECTION_PARAMS = {
  cell_size_m: 1.0,
  amplitude_threshold_nt: null,
  threshold_k: 4.0,
  min_footprint_m: 0.5,
  max_footprint_m: 15.0,
  fit_window_m: 8.0,
  max_depth_m: 5.0,
  min_fit_quality: 0.3,
};

const DEFAULT_TRANSFORM_EXTRA_PARAMS = {
  continuation_height_m: 20.0,
  trend_order: 1,
  microlevel_strength: 0.8,
  microlevel_angle_tolerance_deg: 15.0,
  microlevel_wavelength_factor: 1.5,
  microlevel_pre_apply: false,
};

const DEFAULT_PARAMS = {
  filter_method: "butterworth",
  filter_cutoff_hz: 1.0,
  filter_window_seconds: 1.0,
  filter_polyorder: 3,
  gps_mag_lag_seconds: 0.0,
  utm_epsg_override: null,
  korea_projection: null,
  despike_params: {
    enabled: true,
    window_size: 11,
    threshold_k: 4.0,
    adaptive: true,
    adaptive_gradient_threshold: 5.0,
    adaptive_expand_samples: 4,
  },
  sway_detection: {
    enabled: true,
    threshold_k: 4.0,
  },
  heading_effect_calibration: {
    enabled: true,
  },
  line_params: {
    heading_lag_seconds: 1.0,
    heading_tolerance_deg: 20.0,
    min_speed_mps: 1.5,
    min_line_length_m: 150.0,
    turn_buffer_m: 15.0,
    max_gap_seconds: 1.0,
    direction_method: "heading_histogram",
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
  crossover_leveling: {
    enabled: false,
    tie_tolerance_deg: 20.0,
    max_crossover_distance_m: 15.0,
    iterative: true,
  },
};

export default function App() {
  const [projectId, setProjectId] = useState(null);
  const [droneSummary, setDroneSummary] = useState(null);
  const [baseSummary, setBaseSummary] = useState(null);
  const [headingCalibrationSummary, setHeadingCalibrationSummary] = useState(null);
  const [headingCalibrationUploadProgress, setHeadingCalibrationUploadProgress] = useState(null);
  const [processParams, setProcessParams] = useState(DEFAULT_PARAMS);
  const [processSummary, setProcessSummary] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [points, setPoints] = useState([]);
  const [pointsLoading, setPointsLoading] = useState(false);
  const [valueField, setValueField] = useState("anomaly");
  const [hoverPoint, setHoverPoint] = useState(null);
  const [drawMode, setDrawMode] = useState(false);
  const [drawAction, setDrawAction] = useState("exclude");
  const [gridCellSize, setGridCellSize] = useState(10.0);
  const [gridMethod, setGridMethod] = useState("nearest");
  const [gridMaxDistance, setGridMaxDistance] = useState(null);
  const [alongLineSmooth, setAlongLineSmooth] = useState(true);
  const [alongLineSmoothWavelength, setAlongLineSmoothWavelength] = useState(null);
  const [gridding, setGridding] = useState(false);
  const [exportingGeotiff, setExportingGeotiff] = useState(false);
  const [savingProject, setSavingProject] = useState(false);
  const [loadingProject, setLoadingProject] = useState(false);
  const [exportingReport, setExportingReport] = useState(false);
  const [activeTransform, setActiveTransform] = useState("none");
  const [transformLoading, setTransformLoading] = useState(false);
  const [overlay, setOverlay] = useState(null);
  const [error, setError] = useState(null);
  const [cmapName, setCmapName] = useState("RdYlBu_r");
  const [manualRange, setManualRange] = useState({ enabled: false, vmin: null, vmax: null });
  const [gridOpacity, setGridOpacity] = useState(0.85);
  const [hillshade, setHillshade] = useState(false);
  const [hillshadeAzimuth, setHillshadeAzimuth] = useState(315);
  const [hillshadeAltitude, setHillshadeAltitude] = useState(45);
  const [hillshadeExaggeration, setHillshadeExaggeration] = useState(3);
  const [stretch, setStretch] = useState("linear");
  const [exportGeotiffColored, setExportGeotiffColored] = useState(false);
  const [droneUploadProgress, setDroneUploadProgress] = useState(null);
  const [baseUploadProgress, setBaseUploadProgress] = useState(null);
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(false);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(false);
  const [showContours, setShowContours] = useState(false);
  const [contourInterval, setContourInterval] = useState(null);
  const [contourNLevels, setContourNLevels] = useState(10);
  const [showPointsOverGrid, setShowPointsOverGrid] = useState(true);
  const [showLineLabels, setShowLineLabels] = useState(false);
  const [overlayLayers, setOverlayLayers] = useState([]);
  const [overlayUploading, setOverlayUploading] = useState(false);
  const [overlayError, setOverlayError] = useState(null);
  const [transformExtraParams, setTransformExtraParams] = useState(DEFAULT_TRANSFORM_EXTRA_PARAMS);
  const [exportingXyz, setExportingXyz] = useState(false);
  const [exportingGrd, setExportingGrd] = useState(false);
  const [exportingPointsCsv, setExportingPointsCsv] = useState(false);
  const [lastDrawnPolygon, setLastDrawnPolygon] = useState(null);
  const [exportingBln, setExportingBln] = useState(false);
  const [flightPathEditorOpen, setFlightPathEditorOpen] = useState(false);
  const [lineProfileData, setLineProfileData] = useState(null);
  const [lineProfileLoadingId, setLineProfileLoadingId] = useState(null);

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

  const [eulerStructuralIndex, setEulerStructuralIndex] = useState(1.0);
  const [eulerWindowSize, setEulerWindowSize] = useState(100.0);
  const [eulerMaxUncertaintyPct, setEulerMaxUncertaintyPct] = useState(30.0);
  const [eulerRunning, setEulerRunning] = useState(false);
  const [eulerResult, setEulerResult] = useState(null);
  const [eulerError, setEulerError] = useState(null);
  const [showEulerSolutions, setShowEulerSolutions] = useState(true);

  const [chatMessages, setChatMessages] = useState([]);
  const [chatSending, setChatSending] = useState(false);
  const [chatError, setChatError] = useState(null);

  // "용도" preset: mineral-exploration workflows want broad, coarse grids
  // and the 3D inversion/Euler panels front and center; near-surface
  // target detection (mines/UXO/hidden vehicles) wants a much finer grid
  // and the dipole-fit target panel front and center instead - switching
  // resets the grid cell size to that mode's usual scale.
  const [purposeMode, setPurposeMode] = useState("mineral"); // "mineral" | "target"
  const [targetDetectionParams, setTargetDetectionParams] = useState(DEFAULT_TARGET_DETECTION_PARAMS);
  const [targetDetectionRunning, setTargetDetectionRunning] = useState(false);
  const [targetDetectionResult, setTargetDetectionResult] = useState(null);
  const [targetDetectionError, setTargetDetectionError] = useState(null);
  const [showDetectedTargets, setShowDetectedTargets] = useState(true);

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
      setDroneUploadProgress(0);
      const id = await ensureProject();
      const summary = await api.uploadDrone(id, files, setDroneUploadProgress);
      setDroneSummary(summary);
    } catch (e) {
      handleError(e);
    } finally {
      setDroneUploadProgress(null);
    }
  };

  const handleUploadBase = async (files) => {
    try {
      setError(null);
      setBaseUploadProgress(0);
      const id = await ensureProject();
      const summary = await api.uploadBase(id, files, setBaseUploadProgress);
      setBaseSummary(summary);
    } catch (e) {
      handleError(e);
    } finally {
      setBaseUploadProgress(null);
    }
  };

  const handleUploadHeadingCalibration = async (files) => {
    try {
      setError(null);
      setHeadingCalibrationUploadProgress(0);
      const id = await ensureProject();
      const summary = await api.uploadHeadingCalibration(id, files, setHeadingCalibrationUploadProgress);
      setHeadingCalibrationSummary(summary);
    } catch (e) {
      handleError(e);
    } finally {
      setHeadingCalibrationUploadProgress(null);
    }
  };

  const refreshPoints = useCallback(async (id, value) => {
    setPointsLoading(true);
    try {
      const pts = await api.getPoints(id, value);
      setPoints(pts);
    } finally {
      setPointsLoading(false);
    }
  }, []);

  // manual-exclude responses carry just the (point_id -> excluded) delta,
  // not the whole point list (lat/lon/value/line_id/timestamp never
  // change from a manual edit) - patching it in place avoids re-fetching
  // and re-parsing the full point set, which gets slow at 100k+ points.
  const applyExclusionDelta = useCallback((exclusion) => {
    if (!exclusion) return;
    const excludedById = new Map(exclusion.point_id.map((id, i) => [id, exclusion.excluded[i]]));
    setPoints((prev) => prev.map((p) => (excludedById.has(p.point_id) ? { ...p, excluded: excludedById.get(p.point_id) } : p)));
  }, []);

  const handleSaveProject = async () => {
    try {
      setError(null);
      setSavingProject(true);
      await api.saveProject(projectId, "magnetic_project.zip");
    } catch (e) {
      handleError(e);
    } finally {
      setSavingProject(false);
    }
  };

  const handleExportReport = async () => {
    try {
      setError(null);
      setExportingReport(true);
      await api.exportReport(projectId, "processing_report.md");
    } catch (e) {
      handleError(e);
    } finally {
      setExportingReport(false);
    }
  };

  const handleLoadProject = async (file) => {
    try {
      setError(null);
      setLoadingProject(true);
      const id = await ensureProject();
      const resp = await api.loadProject(id, file);
      setDroneSummary(resp.drone_summary);
      setBaseSummary(resp.base_summary);
      setOverlay(null);
      setActiveTransform("none");
      setSectionResult(null);
      setVolumeData(null);
      if (resp.params) setProcessParams(resp.params);
      if (resp.processed) {
        setProcessSummary(resp.process_summary);
        await refreshPoints(id, valueField);
      } else {
        setProcessSummary(null);
        setPoints([]);
      }
      setInversionSummary(resp.inversion_summary || null);
    } catch (e) {
      handleError(e);
    } finally {
      setLoadingProject(false);
    }
  };

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
      setLineProfileData(null);
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
      applyExclusionDelta(summary.exclusion);
      setOverlay(null);
    } catch (e) {
      handleError(e);
    }
  };

  const handleShapeDrawn = async (latlngCoords) => {
    setLastDrawnPolygon(latlngCoords);
    try {
      setError(null);
      const summary = await api.manualExclude(projectId, { mode: "polygon", action: drawAction, polygon: latlngCoords });
      setProcessSummary(summary);
      applyExclusionDelta(summary.exclusion);
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
      applyExclusionDelta(summary.exclusion);
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
        along_line_smooth: alongLineSmooth,
        along_line_smooth_wavelength_m: alongLineSmoothWavelength,
        cmap: cmapName,
        vmin: manualRange.enabled ? manualRange.vmin : null,
        vmax: manualRange.enabled ? manualRange.vmax : null,
        hillshade,
        hillshade_azimuth_deg: hillshadeAzimuth,
        hillshade_altitude_deg: hillshadeAltitude,
        hillshade_exaggeration: hillshadeExaggeration,
        show_contours: showContours,
        contour_interval_nt: contourInterval,
        contour_n_levels: contourNLevels,
        stretch,
        microlevel_pre_apply: transformExtraParams.microlevel_pre_apply,
        microlevel_strength: transformExtraParams.microlevel_strength,
        microlevel_angle_tolerance_deg: transformExtraParams.microlevel_angle_tolerance_deg,
        microlevel_wavelength_factor: transformExtraParams.microlevel_wavelength_factor,
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
        // "그리드(원본)" must redisplay the plain gridded raster, not just
        // clear the overlay - clearing it left nothing but the raw survey
        // point layer showing through underneath, which looks like the
        // grid silently reverted to scattered points instead of the
        // interpolated surface.
        setTransformLoading(true);
        try {
          await handleGrid();
        } finally {
          setTransformLoading(false);
        }
        return;
      }
      setTransformLoading(true);
      const resp = await api.getTransform(projectId, {
        transform: name,
        value: valueField,
        cell_size_m: gridCellSize,
        method: gridMethod,
        max_distance_m: gridMaxDistance,
        along_line_smooth: alongLineSmooth,
        along_line_smooth_wavelength_m: alongLineSmoothWavelength,
        cmap: cmapName,
        vmin: manualRange.enabled ? manualRange.vmin : null,
        vmax: manualRange.enabled ? manualRange.vmax : null,
        hillshade,
        hillshade_azimuth_deg: hillshadeAzimuth,
        hillshade_altitude_deg: hillshadeAltitude,
        hillshade_exaggeration: hillshadeExaggeration,
        show_contours: showContours,
        contour_interval_nt: contourInterval,
        contour_n_levels: contourNLevels,
        stretch,
        ...transformExtraParams,
      });
      setOverlay(resp);
      setActiveTransform(name);
    } catch (e) {
      handleError(e);
    } finally {
      setTransformLoading(false);
    }
  };

  const handleExportGeotiff = async () => {
    try {
      setError(null);
      setExportingGeotiff(true);
      const base = {
        value: valueField,
        cell_size_m: gridCellSize,
        method: gridMethod,
        max_distance_m: gridMaxDistance,
        along_line_smooth: alongLineSmooth,
        along_line_smooth_wavelength_m: alongLineSmoothWavelength,
        colored: exportGeotiffColored,
        cmap: cmapName,
        vmin: manualRange.enabled ? manualRange.vmin : null,
        vmax: manualRange.enabled ? manualRange.vmax : null,
        hillshade,
        hillshade_azimuth_deg: hillshadeAzimuth,
        hillshade_altitude_deg: hillshadeAltitude,
        hillshade_exaggeration: hillshadeExaggeration,
        stretch,
        ...transformExtraParams,
      };
      const suffix = exportGeotiffColored ? "_colored" : "";
      if (activeTransform === "none") {
        await api.exportGridGeotiff(projectId, base, `${valueField}_${gridCellSize}m${suffix}.tif`);
      } else {
        await api.exportTransformGeotiff(
          projectId,
          { ...base, transform: activeTransform },
          `${activeTransform}_${gridCellSize}m${suffix}.tif`
        );
      }
    } catch (e) {
      handleError(e);
    } finally {
      setExportingGeotiff(false);
    }
  };

  const handleExportXyz = async () => {
    try {
      setError(null);
      setExportingXyz(true);
      const base = {
        value: valueField,
        cell_size_m: gridCellSize,
        method: gridMethod,
        max_distance_m: gridMaxDistance,
        along_line_smooth: alongLineSmooth,
        along_line_smooth_wavelength_m: alongLineSmoothWavelength,
        ...transformExtraParams,
      };
      if (activeTransform === "none") {
        await api.exportGridXyz(projectId, base, `${valueField}_${gridCellSize}m.xyz`);
      } else {
        await api.exportTransformXyz(projectId, { ...base, transform: activeTransform }, `${activeTransform}_${gridCellSize}m.xyz`);
      }
    } catch (e) {
      handleError(e);
    } finally {
      setExportingXyz(false);
    }
  };

  const handleExportGrd = async () => {
    try {
      setError(null);
      setExportingGrd(true);
      const base = {
        value: valueField,
        cell_size_m: gridCellSize,
        method: gridMethod,
        max_distance_m: gridMaxDistance,
        along_line_smooth: alongLineSmooth,
        along_line_smooth_wavelength_m: alongLineSmoothWavelength,
        ...transformExtraParams,
      };
      if (activeTransform === "none") {
        await api.exportGridGrd(projectId, base, `${valueField}_${gridCellSize}m.grd`);
      } else {
        await api.exportTransformGrd(projectId, { ...base, transform: activeTransform }, `${activeTransform}_${gridCellSize}m.grd`);
      }
    } catch (e) {
      handleError(e);
    } finally {
      setExportingGrd(false);
    }
  };

  const handleExportBln = async () => {
    if (!lastDrawnPolygon || lastDrawnPolygon.length < 3) {
      setError("먼저 지도에서 영역을 그려야 BLN으로 저장할 수 있습니다 (측선 편집의 '지도에서 영역 그리기' 기능 사용).");
      return;
    }
    try {
      setError(null);
      setExportingBln(true);
      await api.exportPolygonBln(projectId, lastDrawnPolygon, "area.bln");
    } catch (e) {
      handleError(e);
    } finally {
      setExportingBln(false);
    }
  };

  const handleManualExcludePointIds = async (pointIds, action) => {
    try {
      setError(null);
      const summary = await api.manualExclude(projectId, { mode: "point_ids", action, point_ids: pointIds });
      setProcessSummary(summary);
      applyExclusionDelta(summary.exclusion);
      setOverlay(null);
    } catch (e) {
      handleError(e);
    }
  };

  const handleExportPointsCsv = async () => {
    try {
      setError(null);
      setExportingPointsCsv(true);
      await api.exportPointsCsv(projectId, "points.csv");
    } catch (e) {
      handleError(e);
    } finally {
      setExportingPointsCsv(false);
    }
  };

  const handleShowLineProfile = async (lineId) => {
    try {
      setError(null);
      setLineProfileLoadingId(lineId);
      const resp = await api.getLineProfile(projectId, lineId, valueField);
      setLineProfileData(resp);
    } catch (e) {
      handleError(e);
    } finally {
      setLineProfileLoadingId(null);
    }
  };

  const handleUploadOverlayImage = async (file) => {
    try {
      setOverlayError(null);
      setOverlayUploading(true);
      const id = await ensureProject();
      const [resp] = await Promise.all([
        api.uploadOverlayImage(file),
        // Best-effort: also keep a project-scoped copy so the chat
        // assistant's sample_point tool can read real pixel values here.
        // Must never block the map overlay display if it fails.
        api.uploadReferenceLayer(id, file).catch(() => null),
      ]);
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

  const handleRemoveOverlay = (id) => {
    setOverlayLayers((prev) => {
      const layer = prev.find((l) => l.id === id);
      if (layer && projectId) {
        api.deleteReferenceLayer(projectId, layer.name).catch(() => {});
      }
      return prev.filter((l) => l.id !== id);
    });
  };

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

  const handleRunEuler = async () => {
    try {
      setEulerError(null);
      setEulerRunning(true);
      const resp = await api.runEulerDeconvolution(projectId, {
        value: valueField,
        cell_size_m: gridCellSize,
        method: gridMethod,
        max_distance_m: gridMaxDistance,
        structural_index: eulerStructuralIndex,
        window_size_m: eulerWindowSize,
        max_depth_uncertainty_pct: eulerMaxUncertaintyPct,
      });
      setEulerResult(resp);
    } catch (e) {
      setEulerError(e.message || String(e));
    } finally {
      setEulerRunning(false);
    }
  };

  const handleSetPurposeMode = (mode) => {
    setPurposeMode(mode);
    setGridCellSize(mode === "target" ? 1.0 : 10.0);
  };

  const handleRunTargetDetection = async () => {
    try {
      setTargetDetectionError(null);
      setTargetDetectionRunning(true);
      const resp = await api.runTargetDetection(projectId, targetDetectionParams);
      setTargetDetectionResult(resp);
    } catch (e) {
      setTargetDetectionError(e.message || String(e));
    } finally {
      setTargetDetectionRunning(false);
    }
  };

  const handleSendChatMessage = async (text) => {
    try {
      setChatError(null);
      setChatSending(true);
      const id = await ensureProject();
      const history = chatMessages.map((m) => ({ role: m.role, content: m.content }));
      setChatMessages((prev) => [...prev, { role: "user", content: text }]);
      const resp = await api.sendChatMessage(id, text, history);
      setChatMessages((prev) => [...prev, { role: "assistant", content: resp.reply, toolCalls: resp.tool_calls }]);
    } catch (e) {
      setChatError(e.message || String(e));
    } finally {
      setChatSending(false);
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

  const handleExportInversion = async () => {
    try {
      setInversionError(null);
      await api.exportInversionResult(projectId, "inversion_result.npz");
    } catch (e) {
      setInversionError(e.message || String(e));
    }
  };

  const handleExportInversionCsv = async () => {
    try {
      setInversionError(null);
      await api.exportInversionCsv(projectId, "inversion_result.csv");
    } catch (e) {
      setInversionError(e.message || String(e));
    }
  };

  const handleImportInversion = async (file) => {
    try {
      setInversionError(null);
      setInversionRunning(true);
      const id = await ensureProject();
      const resp = await api.importInversionResult(id, file);
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
      <div
        className={`app-sidebar-backdrop${leftSidebarOpen || rightSidebarOpen ? " visible" : ""}`}
        onClick={() => {
          setLeftSidebarOpen(false);
          setRightSidebarOpen(false);
        }}
      />
      <div
        className={`app-sidebar-left${leftSidebarOpen ? " open" : ""}`}
        style={{ width: 320, borderRight: "1px solid #e5e7eb", overflowY: "auto", padding: 12, background: "#f9fafb" }}
      >
        <h1 style={{ fontSize: 16, margin: "4px 0 12px 0" }}>드론 자력탐사 자료 처리</h1>
        <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
          <button
            style={{
              flex: 1,
              padding: "6px 8px",
              fontSize: 12,
              borderRadius: 6,
              border: purposeMode === "mineral" ? "1px solid #2563eb" : "1px solid #d1d5db",
              background: purposeMode === "mineral" ? "#2563eb" : "white",
              color: purposeMode === "mineral" ? "white" : "#374151",
              cursor: "pointer",
            }}
            onClick={() => handleSetPurposeMode("mineral")}
            title="넓은 지역의 완만한 지질체 - 3차원 역산/오일러 디컨볼루션 중심, 격자 크기 기본 10m"
          >
            🪨 광물자원탐사
          </button>
          <button
            style={{
              flex: 1,
              padding: "6px 8px",
              fontSize: 12,
              borderRadius: 6,
              border: purposeMode === "target" ? "1px solid #b91c1c" : "1px solid #d1d5db",
              background: purposeMode === "target" ? "#b91c1c" : "white",
              color: purposeMode === "target" ? "white" : "#374151",
              cursor: "pointer",
            }}
            onClick={() => handleSetPurposeMode("target")}
            title="작고 국지적인 근지표 표적(지뢰/불발탄/은닉차량) - 쌍극자 피팅 표적탐지 중심, 격자 크기 기본 1m"
          >
            🎯 근지표 표적탐지
          </button>
        </div>
        <WorkflowProgress
          droneSummary={droneSummary}
          baseSummary={baseSummary}
          processSummary={processSummary}
          overlay={overlay}
          inversionSummary={inversionSummary}
          eulerResult={eulerResult}
        />
        <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
          <button
            style={{
              flex: 1,
              padding: "6px 8px",
              fontSize: 12,
              borderRadius: 6,
              border: "1px solid #2563eb",
              background: "white",
              color: "#2563eb",
              cursor: droneSummary && !savingProject ? "pointer" : "default",
              opacity: droneSummary && !savingProject ? 1 : 0.5,
            }}
            disabled={!droneSummary || savingProject}
            onClick={handleSaveProject}
            title="업로드한 자료, 처리 파라미터, 수동 편집, DEM, 역산 결과를 모두 담아 저장합니다"
          >
            {savingProject ? "저장 중..." : "💾 프로젝트 저장"}
          </button>
          <label
            style={{
              flex: 1,
              padding: "6px 8px",
              fontSize: 12,
              borderRadius: 6,
              border: "1px solid #2563eb",
              background: "white",
              color: "#2563eb",
              cursor: "pointer",
              textAlign: "center",
              opacity: loadingProject ? 0.5 : 1,
            }}
            title="이전에 저장한 프로젝트(.zip)를 불러와 이어서 작업합니다"
          >
            {loadingProject ? "불러오는 중..." : "📂 프로젝트 불러오기"}
            <input
              type="file"
              accept=".zip"
              style={{ display: "none" }}
              disabled={loadingProject}
              onChange={async (e) => {
                const f = e.target.files[0];
                if (!f) return;
                await handleLoadProject(f);
                e.target.value = "";
              }}
            />
          </label>
        </div>
        <button
          style={{
            width: "100%",
            marginBottom: 12,
            padding: "6px 8px",
            fontSize: 12,
            borderRadius: 6,
            border: "1px solid #2563eb",
            background: "white",
            color: "#2563eb",
            cursor: processSummary && !exportingReport ? "pointer" : "default",
            opacity: processSummary && !exportingReport ? 1 : 0.5,
          }}
          disabled={!processSummary || exportingReport}
          onClick={handleExportReport}
          title="지금까지 적용된 보정, 측선/통계 요약, 역산·오일러 결과를 정리한 Markdown 보고서를 다운로드합니다"
        >
          {exportingReport ? "생성 중..." : "📄 처리 보고서 다운로드 (.md)"}
        </button>
        <WorkflowSteps
          onUploadDrone={handleUploadDrone}
          droneSummary={droneSummary}
          droneUploadProgress={droneUploadProgress}
          onUploadBase={handleUploadBase}
          baseSummary={baseSummary}
          baseUploadProgress={baseUploadProgress}
          onUploadHeadingCalibration={handleUploadHeadingCalibration}
          headingCalibrationSummary={headingCalibrationSummary}
          headingCalibrationUploadProgress={headingCalibrationUploadProgress}
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
          alongLineSmooth={alongLineSmooth}
          setAlongLineSmooth={setAlongLineSmooth}
          alongLineSmoothWavelength={alongLineSmoothWavelength}
          setAlongLineSmoothWavelength={setAlongLineSmoothWavelength}
          lineSpacingM={processSummary?.line_spacing_m}
          gridOpacity={gridOpacity}
          setGridOpacity={setGridOpacity}
          hillshade={hillshade}
          setHillshade={setHillshade}
          hillshadeAzimuth={hillshadeAzimuth}
          setHillshadeAzimuth={setHillshadeAzimuth}
          hillshadeAltitude={hillshadeAltitude}
          setHillshadeAltitude={setHillshadeAltitude}
          hillshadeExaggeration={hillshadeExaggeration}
          setHillshadeExaggeration={setHillshadeExaggeration}
          stretch={stretch}
          setStretch={setStretch}
          showContours={showContours}
          setShowContours={setShowContours}
          contourInterval={contourInterval}
          setContourInterval={setContourInterval}
          contourNLevels={contourNLevels}
          setContourNLevels={setContourNLevels}
          onGrid={handleGrid}
          gridding={gridding}
          activeTransform={activeTransform}
          onTransform={handleTransform}
          transformLoading={transformLoading}
          valueField={valueField}
          setValueField={handleSetValueField}
          onExportGeotiff={handleExportGeotiff}
          exportingGeotiff={exportingGeotiff}
          exportGeotiffColored={exportGeotiffColored}
          setExportGeotiffColored={setExportGeotiffColored}
          onExportXyz={handleExportXyz}
          exportingXyz={exportingXyz}
          onExportGrd={handleExportGrd}
          exportingGrd={exportingGrd}
          onExportPointsCsv={handleExportPointsCsv}
          exportingPointsCsv={exportingPointsCsv}
          transformExtraParams={transformExtraParams}
          setTransformExtraParams={setTransformExtraParams}
          error={error}
        />
      </div>

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
          eulerSolutions={showEulerSolutions ? eulerResult?.solutions : null}
          detectedTargets={showDetectedTargets ? targetDetectionResult?.targets : null}
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
                  color: "#374151",
                }}
              >
                3D 뷰어 불러오는 중...
              </div>
            }
          >
            <InversionVolumeView data={volumeData} onClose={() => setVolumeData(null)} />
          </Suspense>
        )}
        {!volumeData && sectionResult && <InversionSectionView data={sectionResult} onClose={() => setSectionResult(null)} />}
        {!volumeData && !sectionResult && lineProfileData && (
          <LineProfileView
            data={lineProfileData}
            valueLabel={valueField === "anomaly" ? "자력 이상 (nT)" : "TMI (nT)"}
            onClose={() => setLineProfileData(null)}
          />
        )}
      </div>

      <div
        className={`app-sidebar-right${rightSidebarOpen ? " open" : ""}`}
        style={{ width: 280, borderLeft: "1px solid #e5e7eb", overflowY: "auto", padding: 12, background: "#f9fafb" }}
      >
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
          onShowLineProfile={handleShowLineProfile}
          lineProfileLoadingId={lineProfileLoadingId}
          onOpenFlightPathEditor={() => setFlightPathEditorOpen(true)}
          onExportBln={handleExportBln}
          exportingBln={exportingBln}
          canExportBln={!!lastDrawnPolygon}
        />

        {flightPathEditorOpen && (
          <FlightPathEditor
            points={points}
            loading={pointsLoading}
            lines={processSummary?.lines}
            onApply={handleManualExcludePointIds}
            onClose={() => setFlightPathEditorOpen(false)}
          />
        )}

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

        <details open={purposeMode === "mineral"} style={{ marginBottom: 4 }}>
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
            onRun={handleRunEuler}
            running={eulerRunning}
            result={eulerResult}
            error={eulerError}
            showSolutions={showEulerSolutions}
            setShowSolutions={setShowEulerSolutions}
          />
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
            />
          </div>
        </details>

        <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>15. AI 해석 도우미 (챗봇)</h2>
        <ChatPanel ready={!!processSummary} messages={chatMessages} onSend={handleSendChatMessage} sending={chatSending} error={chatError} />

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
