import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import BaseStationView from "./components/BaseStationView";
import NearestIntermagnetComparisonView from "./components/NearestIntermagnetComparisonView";
import OfflineMapPanel from "./components/OfflineMapPanel";
import { pathLength, polygonArea, formatDistance, formatArea, circlePolygon } from "./geoMeasure";

// plotly.js-dist-min alone is ~4.7MB unminified - only the 3D volume view
// needs it, and most sessions never open it, so it's split into its own
// chunk and fetched on demand instead of bloating the initial bundle.
const InversionVolumeView = lazy(() => import("./components/InversionVolumeView"));
// GuidelinePanel also pulls in plotly.js-dist-min (for the power-spectrum
// chart) - same reasoning as InversionVolumeView above, split into its own
// chunk rather than bloating the initial bundle.
const GuidelinePanel = lazy(() => import("./components/GuidelinePanel"));
import EulerPanel from "./components/EulerPanel";
import WorkflowProgress from "./components/WorkflowProgress";
import ChatPanel from "./components/ChatPanel";
import TargetDetectionPanel from "./components/TargetDetectionPanel";
import QcCertificatePanel from "./components/QcCertificatePanel";
import LineamentPanel from "./components/LineamentPanel";
import DepthEstimationPanel from "./components/DepthEstimationPanel";
import StructureDistortionPanel from "./components/StructureDistortionPanel";
import ContactPanel from "./components/ContactPanel";
import ProspectivityPanel from "./components/ProspectivityPanel";

const toggleButtonStyle = {
  width: 36,
  height: 36,
  borderRadius: 8,
  border: "1px solid #ddd0b2",
  background: "white",
  color: "#2c2418",
  fontSize: 16,
  cursor: "pointer",
  boxShadow: "0 1px 4px rgba(0,0,0,0.2)",
};

const DEFAULT_INVERSION_PARAMS = {
  obs_cell_size_m: 30.0,
  depth_extent_m: 150.0,
  n_layers: 8,
  depth_growth_factor: null,
  assumed_agl_m: 50.0,
  dem_geoid_offset_m: 0.0,
  regularization_strength: 1.0,
  n_irls_iterations: 6,
  assumed_noise_nt: null,
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

const DEFAULT_QC_CERTIFICATE_PARAMS = {
  noise_threshold_multiplier: 2.0,
  max_repeatability_1sigma_nt: 5.0,
  max_sampling_gap_pct: 5.0,
  max_excluded_pct: 30.0,
};

const DEFAULT_LINEAMENT_PARAMS = {
  value: "anomaly",
  cell_size_m: 10.0,
  method: "nearest",
  source: "thd",
  percentile_threshold: 90.0,
  min_segment_points: 4,
  min_length_m: 0.0,
  rose_bin_width_deg: 10.0,
  max_gap_cells: 2.5,
};

const DEFAULT_TILT_DEPTH_PARAMS = { min_depth_m: 1.0, max_depth_m: 500.0 };
const DEFAULT_AS_DEPTH_PARAMS = { percentile_threshold: 90.0, search_radius_cells: 15 };

const DEFAULT_CONTACT_PARAMS = {
  value: "anomaly",
  cell_size_m: 10.0,
  method: "nearest",
  heights_m: [0.0, 20.0, 40.0, 60.0, 80.0],
  percentile_threshold: 80.0,
  min_persistence: 0.5,
  min_segment_points: 4,
  min_length_m: 0.0,
  max_gap_cells: 3.0,
};

const DEFAULT_PROSPECTIVITY_PARAMS = {
  value: "anomaly",
  cell_size_m: 10.0,
  method: "nearest",
  purpose: "magnetite_fe",
  weights: { asa: 0.3, thd: 0.2, structure: 0.2, contact: 0.2, susceptibility: 0.1 },
  decay_length_m: 200.0,
  score_threshold: 0.6,
  max_targets: 20,
};

const DEFAULT_STRUCTURE_SCAN_PARAMS = {
  cell_size_m: 2.0,
  building_buffer_m: 10.0,
  road_buffer_m: 6.0,
  threshold_k: 4.0,
  min_footprint_m: 1.0,
  max_footprint_m: 40.0,
  fit_window_m: 25.0,
  max_depth_m: 10.0,
  min_fit_quality: 0.2,
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
  base_qc_params: {
    trim_enabled: true,
    trim_window_seconds: 30.0,
    trim_threshold_k: 6.0,
    trim_confirm_seconds: 60.0,
    trim_max_fraction: 0.2,
    despike_enabled: true,
    despike_window_size: 11,
    despike_threshold_k: 5.0,
  },
  sway_detection: {
    enabled: false,
    threshold_k: 4.0,
  },
  duplicate_line_params: {
    enabled: false,
    perp_tolerance_m: 8.0,
    angle_tolerance_deg: 15.0,
  },
  heading_effect_calibration: {
    enabled: false,
    auto_calibrate_from_turns: true,
    quality_threshold_nt: 3.0,
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
    enabled: false,
    quiet_percentile: 40.0,
    max_match_distance_m: null,
  },
  crossover_leveling: {
    enabled: false,
    tie_tolerance_deg: 20.0,
    max_crossover_distance_m: 15.0,
    iterative: true,
    leveling_order: 0,
  },
  noise_qc: { enabled: true },
  notch_filter: { frequencies_hz: [], quality_factor: 30.0 },
};

export default function App() {
  const [backendVersion, setBackendVersion] = useState(null);
  useEffect(() => {
    api.getVersion().then(setBackendVersion).catch(() => setBackendVersion(null));
  }, []);
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
  const [measureMode, setMeasureMode] = useState(null); // null | "distance" | "area"
  const [measurements, setMeasurements] = useState([]);
  const measureCounterRef = useRef({ distance: 0, area: 0 });
  // 이착륙 램프 구간(이륙->측선시작, 측선종료->착륙)은 기본적으로 편집 화면에서
  // 숨기고 제외/복원 드로잉의 영향도 받지 않도록 보호한다 - 켜면 다시 보이고
  // 편집도 가능해진다 (backend: ManualExcludeRequest.include_ramp).
  const [showRampPoints, setShowRampPoints] = useState(false);
  const [gridCellSize, setGridCellSize] = useState(10.0);
  const [gridMethod, setGridMethod] = useState("nearest");
  const [gridMaxDistance, setGridMaxDistance] = useState(null);
  const [alongLineSmooth, setAlongLineSmooth] = useState(true);
  const [alongLineSmoothWavelength, setAlongLineSmoothWavelength] = useState(null);
  const [gridding, setGridding] = useState(false);
  const [boundaryMode, setBoundaryMode] = useState(false);
  const [boundaryPolygon, setBoundaryPolygon] = useState(null);
  const [boundaryFilename, setBoundaryFilename] = useState("boundary.json");
  // Buffer distance (m) for the auto-generated boundary - the one number
  // that decides how far past the outermost flight line the map extends.
  // Matches ProcessParams.display_boundary_buffer_m's default.
  const [boundaryBufferM, setBoundaryBufferM] = useState(10);
  const [autoBoundaryBusy, setAutoBoundaryBusy] = useState(false);
  const [autoBoundaryInfo, setAutoBoundaryInfo] = useState(null);
  const [nearestIntermagnetCsvFilename, setNearestIntermagnetCsvFilename] = useState("intermagnet_nearest_estimate.csv");
  const [exportingGeotiff, setExportingGeotiff] = useState(false);
  const [savingProject, setSavingProject] = useState(false);
  const [loadingProject, setLoadingProject] = useState(false);
  const [saveFilename, setSaveFilename] = useState("magnetic_project");
  const [exportingReport, setExportingReport] = useState(false);
  const [activeTransform, setActiveTransform] = useState("none");
  const [transformLoading, setTransformLoading] = useState(false);
  const [overlay, setOverlay] = useState(null);
  const [confidenceOverlay, setConfidenceOverlay] = useState(null);
  const [showConfidenceOverlay, setShowConfidenceOverlay] = useState(false);
  const [confidenceLoading, setConfidenceLoading] = useState(false);
  const [inspectMode, setInspectMode] = useState(false);
  const [inspectPoints, setInspectPoints] = useState([]);
  const [flyToTarget, setFlyToTarget] = useState(null);
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
  const [intermagnetPreview, setIntermagnetPreview] = useState(null);
  const [intermagnetLoading, setIntermagnetLoading] = useState(false);
  const [intermagnetApplying, setIntermagnetApplying] = useState(false);
  const [intermagnetError, setIntermagnetError] = useState(null);
  const [nearestIntermagnetPreview, setNearestIntermagnetPreview] = useState(null);
  const [nearestIntermagnetLoading, setNearestIntermagnetLoading] = useState(false);
  const [nearestIntermagnetApplying, setNearestIntermagnetApplying] = useState(false);
  const [nearestIntermagnetError, setNearestIntermagnetError] = useState(null);
  // Tracks whether a nearest-observatory fetch has ever succeeded, so the
  // comparison button stays enabled even after apply clears the preview
  // above (the backend keeps the underlying station data around regardless).
  const [nearestIntermagnetHasResult, setNearestIntermagnetHasResult] = useState(false);
  const [nearestIntermagnetComparison, setNearestIntermagnetComparison] = useState(null);
  const [nearestIntermagnetComparisonLoading, setNearestIntermagnetComparisonLoading] = useState(false);
  const [mapBounds, setMapBounds] = useState(null);
  const [tileEstimate, setTileEstimate] = useState(null);
  const [tileDownloadResult, setTileDownloadResult] = useState(null);
  const [tileStatus, setTileStatus] = useState(null);
  const [tileLoading, setTileLoading] = useState(false);
  const [tileError, setTileError] = useState(null);
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
  const [tileFolderRegistering, setTileFolderRegistering] = useState(false);
  const [tileFolderError, setTileFolderError] = useState(null);
  const [transformExtraParams, setTransformExtraParams] = useState(DEFAULT_TRANSFORM_EXTRA_PARAMS);
  const [exportingXyz, setExportingXyz] = useState(false);
  const [exportingGrd, setExportingGrd] = useState(false);
  const [exportingPointsCsv, setExportingPointsCsv] = useState(false);
  const [lastDrawnPolygon, setLastDrawnPolygon] = useState(null);
  const [exportingBln, setExportingBln] = useState(false);
  const [flightPathEditorOpen, setFlightPathEditorOpen] = useState(false);
  const [lineProfileData, setLineProfileData] = useState(null);
  const [lineProfileLoadingId, setLineProfileLoadingId] = useState(null);
  const [smoothing, setSmoothing] = useState(false);
  const [baseTimeseries, setBaseTimeseries] = useState(null);
  const [baseTimeseriesLoading, setBaseTimeseriesLoading] = useState(false);
  const [smoothDrawMode, setSmoothDrawMode] = useState(false);
  // Undo history for manual smoothing: each entry is the FULL set of
  // smoothed point_ids right after that apply call (not just the delta),
  // so undo can restore an exact prior state via reset + a single
  // point_ids call, regardless of whether entries came from the map
  // polygon tool or the time-series drag-select.
  const [smoothHistory, setSmoothHistory] = useState([]);

  const [demStatus, setDemStatus] = useState(null);
  const [demUploading, setDemUploading] = useState(false);
  const [autoParams, setAutoParams] = useState(true);
  const [inversionParams, setInversionParams] = useState(DEFAULT_INVERSION_PARAMS);
  // User-digitized geology blocks (polygons traced over an uploaded
  // geology map raster, see "12. 참조 레이어") - fed into the 3D
  // inversion as a reference model instead of always regularizing toward
  // zero susceptibility.
  const [geologyUnits, setGeologyUnits] = useState([]);
  const [geologyDrawMode, setGeologyDrawMode] = useState(false);
  const [inversionRunning, setInversionRunning] = useState(false);
  const [inversionSummary, setInversionSummary] = useState(null);
  const [inversionError, setInversionError] = useState(null);
  const [sliceLayerIndex, setSliceLayerIndex] = useState(0);
  const [sliceThreshold, setSliceThreshold] = useState("");
  const [sliceThresholdMax, setSliceThresholdMax] = useState("");
  const [sectionProfile, setSectionProfile] = useState("custom");
  const [sectionPositionFrac, setSectionPositionFrac] = useState(0.5);
  const [sectionDrawMode, setSectionDrawMode] = useState(false);

  // MapView mounts up to three independent Leaflet-draw DrawControl
  // instances that can each be "enabled" - one shared by
  // drawMode/sectionDrawMode/smoothDrawMode/geologyDrawMode (mux'd into a
  // single `enabled` prop, see the drawMode= line below), one for
  // measureMode, one for boundaryMode. Each of those six toggle handlers
  // used to only clear whichever subset of the others its own author
  // happened to think of at the time, so two (or more) could end up
  // enabled together - either two real DrawControls both firing off a
  // single Leaflet-draw CREATED event, or (within the mux'd group) the
  // wrong one of sectionDrawMode/smoothDrawMode/geologyDrawMode/drawMode
  // winning handleMapShapeDrawn's if/else priority order. Routing every
  // toggle through this one helper first guarantees at most one mode is
  // ever active at a time.
  const exitAllDrawModes = () => {
    setDrawMode(false);
    setSectionDrawMode(false);
    setSmoothDrawMode(false);
    setGeologyDrawMode(false);
    setMeasureMode(null);
    setBoundaryMode(false);
  };
  const [sectionThreshold, setSectionThreshold] = useState("");
  const [sectionThresholdMax, setSectionThresholdMax] = useState("");
  const [sectionResult, setSectionResult] = useState(null);
  const [sectionLoading, setSectionLoading] = useState(false);
  const [volumeThreshold, setVolumeThreshold] = useState("");
  const [volumeThresholdMax, setVolumeThresholdMax] = useState("");
  const [volumeData, setVolumeData] = useState(null);
  const [boxFacesData, setBoxFacesData] = useState(null);
  const [boxTopLayerIndex, setBoxTopLayerIndex] = useState(0);
  const [boxFacesLoading, setBoxFacesLoading] = useState(false);
  // Interior section planes shown together with the isosurface "blob" in
  // one combined 3D scene (volume + any number of simultaneous EW/NS/
  // 기울기(azimuth)/자유선 sections) - "자유선" reuses whatever path was
  // last drawn for the 2D 수직 섹션 뷰. Each entry:
  // {id, orientation, positionFrac, azimuthDeg, path, data, loading}.
  const [lastCustomSectionPath, setLastCustomSectionPath] = useState(null);
  const [volumeSlices, setVolumeSlices] = useState([]);
  const [volumeOpacity, setVolumeOpacity] = useState(1.0);

  const [eulerStructuralIndex, setEulerStructuralIndex] = useState(1.0);
  const [eulerWindowSize, setEulerWindowSize] = useState(100.0);
  const [eulerMaxUncertaintyPct, setEulerMaxUncertaintyPct] = useState(30.0);
  const [eulerFlightAglM, setEulerFlightAglM] = useState(null);
  const [eulerRunning, setEulerRunning] = useState(false);
  const [eulerResult, setEulerResult] = useState(null);
  const [eulerError, setEulerError] = useState(null);
  const [showEulerSolutions, setShowEulerSolutions] = useState(true);

  const [repeatabilityUploading, setRepeatabilityUploading] = useState(false);
  const [repeatabilityUploadInfo, setRepeatabilityUploadInfo] = useState(null);
  const [repeatabilityAnalyzing, setRepeatabilityAnalyzing] = useState(false);
  const [repeatabilityResult, setRepeatabilityResult] = useState(null);
  const [repeatabilityError, setRepeatabilityError] = useState(null);

  const [spectrumLineId, setSpectrumLineId] = useState(null);
  const [spectrumValue, setSpectrumValue] = useState("mag_raw");
  const [spectrumLoading, setSpectrumLoading] = useState(false);
  const [spectrumResult, setSpectrumResult] = useState(null);
  const [spectrumError, setSpectrumError] = useState(null);

  const [multiscaleHeightsText, setMultiscaleHeightsText] = useState("0, 25, 50, 100, 200");
  const [multiscalePercentile, setMultiscalePercentile] = useState(80.0);
  const [multiscaleRunning, setMultiscaleRunning] = useState(false);
  const [multiscaleResult, setMultiscaleResult] = useState(null);
  const [multiscaleError, setMultiscaleError] = useState(null);
  const [showMultiscaleLayer, setShowMultiscaleLayer] = useState(true);
  // GuidelinePanel (and its lazy-loaded plotly chunk, ~4.4MB) must not mount
  // until this accordion section is actually opened - <details> only hides
  // its children visually, React still mounts them on first render, so a
  // lazy()+Suspense component sitting directly inside an always-rendered
  // <details> fetches its chunk immediately regardless of open/closed state.
  const [guidelinePanelOpened, setGuidelinePanelOpened] = useState(false);

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
  const [qcCertificateParams, setQcCertificateParams] = useState(DEFAULT_QC_CERTIFICATE_PARAMS);
  const [qcCertificateRunning, setQcCertificateRunning] = useState(false);
  const [qcCertificateResult, setQcCertificateResult] = useState(null);
  const [qcCertificateError, setQcCertificateError] = useState(null);
  const [lineamentParams, setLineamentParams] = useState(DEFAULT_LINEAMENT_PARAMS);
  const [lineamentRunning, setLineamentRunning] = useState(false);
  const [lineamentResult, setLineamentResult] = useState(null);
  const [lineamentError, setLineamentError] = useState(null);
  const [showLineaments, setShowLineaments] = useState(true);
  const [depthEstimationCellSize, setDepthEstimationCellSize] = useState(10.0);
  const [tiltDepthParams, setTiltDepthParams] = useState(DEFAULT_TILT_DEPTH_PARAMS);
  const [tiltDepthRunning, setTiltDepthRunning] = useState(false);
  const [tiltDepthResult, setTiltDepthResult] = useState(null);
  const [tiltDepthError, setTiltDepthError] = useState(null);
  const [showTiltDepth, setShowTiltDepth] = useState(true);
  const [asDepthParams, setAsDepthParams] = useState(DEFAULT_AS_DEPTH_PARAMS);
  const [asDepthRunning, setAsDepthRunning] = useState(false);
  const [asDepthResult, setAsDepthResult] = useState(null);
  const [asDepthError, setAsDepthError] = useState(null);
  const [showAsDepth, setShowAsDepth] = useState(true);
  const [spectralDepthRunning, setSpectralDepthRunning] = useState(false);
  const [spectralDepthResult, setSpectralDepthResult] = useState(null);
  const [spectralDepthError, setSpectralDepthError] = useState(null);
  const [contactParams, setContactParams] = useState(DEFAULT_CONTACT_PARAMS);
  const [contactRunning, setContactRunning] = useState(false);
  const [contactResult, setContactResult] = useState(null);
  const [contactError, setContactError] = useState(null);
  const [showContacts, setShowContacts] = useState(true);
  const [prospectivityParams, setProspectivityParams] = useState(DEFAULT_PROSPECTIVITY_PARAMS);
  const [prospectivityRunning, setProspectivityRunning] = useState(false);
  const [prospectivityResult, setProspectivityResult] = useState(null);
  const [prospectivityError, setProspectivityError] = useState(null);
  const [showProspectivityTargets, setShowProspectivityTargets] = useState(true);
  const [showProspectivityOverlay, setShowProspectivityOverlay] = useState(false);
  const [prospectivityOverlay, setProspectivityOverlay] = useState(null);
  const [structureScanParams, setStructureScanParams] = useState(DEFAULT_STRUCTURE_SCAN_PARAMS);
  const [structureScanRunning, setStructureScanRunning] = useState(false);
  const [structureScanResult, setStructureScanResult] = useState(null);
  const [structureScanError, setStructureScanError] = useState(null);
  const [structureScanShow, setStructureScanShow] = useState(true);
  const [structureScanApplying, setStructureScanApplying] = useState(false);
  const [selectedStructurePolygonIndices, setSelectedStructurePolygonIndices] = useState(new Set());
  const [selectedStructureAnomalyIndices, setSelectedStructureAnomalyIndices] = useState(new Set());

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

  const handleUploadIaga2002 = async (file) => {
    try {
      setIntermagnetError(null);
      setIntermagnetLoading(true);
      const id = await ensureProject();
      const preview = await api.uploadIaga2002(id, file);
      setIntermagnetPreview(preview);
    } catch (e) {
      setIntermagnetError(e.message || String(e));
    } finally {
      setIntermagnetLoading(false);
    }
  };

  const handleFetchIntermagnet = async (req) => {
    try {
      setIntermagnetError(null);
      setIntermagnetLoading(true);
      const id = await ensureProject();
      const preview = await api.fetchIntermagnet(id, req);
      setIntermagnetPreview(preview);
    } catch (e) {
      setIntermagnetError(e.message || String(e));
    } finally {
      setIntermagnetLoading(false);
    }
  };

  const handleApplyIntermagnet = async () => {
    try {
      setIntermagnetError(null);
      setIntermagnetApplying(true);
      const summary = await api.applyIntermagnet(projectId);
      setBaseSummary(summary);
      setIntermagnetPreview(null);
    } catch (e) {
      setIntermagnetError(e.message || String(e));
    } finally {
      setIntermagnetApplying(false);
    }
  };

  const handleCancelIntermagnetPreview = () => {
    setIntermagnetPreview(null);
    setIntermagnetError(null);
  };

  const handleFetchNearestIntermagnet = async (req) => {
    try {
      setNearestIntermagnetError(null);
      setNearestIntermagnetLoading(true);
      const id = await ensureProject();
      const preview = await api.fetchNearestIntermagnet(id, req);
      setNearestIntermagnetPreview(preview);
      setNearestIntermagnetHasResult(true);
    } catch (e) {
      setNearestIntermagnetError(e.message || String(e));
    } finally {
      setNearestIntermagnetLoading(false);
    }
  };

  const handleApplyNearestIntermagnet = async () => {
    try {
      setNearestIntermagnetError(null);
      setNearestIntermagnetApplying(true);
      const summary = await api.applyNearestIntermagnet(projectId);
      setBaseSummary(summary);
      setNearestIntermagnetPreview(null);
    } catch (e) {
      setNearestIntermagnetError(e.message || String(e));
    } finally {
      setNearestIntermagnetApplying(false);
    }
  };

  const handleCancelNearestIntermagnetPreview = () => {
    setNearestIntermagnetPreview(null);
    setNearestIntermagnetError(null);
  };

  const handleExportNearestIntermagnetCsv = async () => {
    try {
      const base = nearestIntermagnetCsvFilename.trim().replace(/\.csv$/i, "") || "intermagnet_nearest_estimate";
      await api.exportNearestIntermagnetCsv(projectId, `${base}.csv`);
    } catch (e) {
      setNearestIntermagnetError(e.message || String(e));
    }
  };

  const handleShowNearestIntermagnetComparison = async () => {
    try {
      setNearestIntermagnetError(null);
      setNearestIntermagnetComparisonLoading(true);
      const comparison = await api.getNearestIntermagnetComparison(projectId);
      setNearestIntermagnetComparison(comparison);
    } catch (e) {
      setNearestIntermagnetError(e.message || String(e));
    } finally {
      setNearestIntermagnetComparisonLoading(false);
    }
  };

  const handleMapBoundsChange = useCallback((bounds) => setMapBounds(bounds), []);

  const handleEstimateTiles = async (req) => {
    try {
      setTileError(null);
      setTileDownloadResult(null);
      const resp = await api.estimateOfflineTiles(req);
      setTileEstimate(resp);
    } catch (e) {
      setTileError(e.message || String(e));
    }
  };

  const handleDownloadTiles = async (req) => {
    try {
      setTileError(null);
      setTileLoading(true);
      const resp = await api.downloadOfflineTiles(req);
      setTileDownloadResult(resp);
      const status = await api.getOfflineTileStatus();
      setTileStatus(status);
    } catch (e) {
      setTileError(e.message || String(e));
    } finally {
      setTileLoading(false);
    }
  };

  const handleRefreshTileStatus = async () => {
    try {
      const status = await api.getOfflineTileStatus();
      setTileStatus(status);
    } catch (e) {
      setTileError(e.message || String(e));
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
      const base = saveFilename.trim().replace(/\.zip$/i, "") || "magnetic_project";
      await api.saveProject(projectId, `${base}.zip`);
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
        try {
          const geo = await api.listGeologyUnits(id);
          setGeologyUnits(geo.units);
        } catch {
          setGeologyUnits([]);
        }
      } else {
        setProcessSummary(null);
        setPoints([]);
        setGeologyUnits([]);
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
      const summary = await api.manualExclude(projectId, { mode: "polygon", action: drawAction, polygon: latlngCoords, include_ramp: showRampPoints });
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
      setConfidenceOverlay(null);
    } catch (e) {
      handleError(e);
    } finally {
      setGridding(false);
    }
  };

  const handleGridConfidence = async () => {
    try {
      setError(null);
      setConfidenceLoading(true);
      const resp = await api.getGridConfidence(projectId, {
        value: valueField,
        cell_size_m: gridCellSize,
        method: gridMethod,
        max_distance_m: gridMaxDistance,
      });
      setConfidenceOverlay(resp);
      setShowConfidenceOverlay(true);
    } catch (e) {
      handleError(e);
    } finally {
      setConfidenceLoading(false);
    }
  };

  // Pinned points are only valid for the overlay they were read from - if a
  // new grid/transform gets generated while some are still pinned, clear
  // them rather than leaving stale values floating over the new surface.
  useEffect(() => {
    setInspectPoints([]);
  }, [overlay]);

  // Keep the boundary shown on the map in sync with what the backend
  // actually has stored (e.g. after loading a saved project, or after any
  // process_summary refresh) rather than only updating it locally when the
  // user draws/clears it themselves.
  useEffect(() => {
    if (processSummary && "display_boundary_polygon" in processSummary) {
      setBoundaryPolygon(processSummary.display_boundary_polygon || null);
    }
    // Processing reports what its own automatic boundary did (buffer used,
    // area, any warning about split blocks) - show that, not a stale
    // result from a previous manual regeneration.
    if (processSummary && "auto_boundary" in processSummary) {
      setAutoBoundaryInfo(processSummary.auto_boundary || null);
      if (processSummary.auto_boundary?.buffer_m) {
        setBoundaryBufferM(processSummary.auto_boundary.buffer_m);
      }
    }
  }, [processSummary]);

  const handleToggleBoundaryMode = () => {
    const next = !boundaryMode;
    exitAllDrawModes();
    if (next) setBoundaryMode(true);
  };

  // Re-request whichever overlay (plain grid or an active derived
  // transform) is currently shown, so a boundary draw/clear is reflected
  // on the map immediately instead of only taking effect the next time the
  // user happens to regenerate the grid/transform themselves.
  const refreshCurrentOverlay = async () => {
    if (!overlay) return;
    if (activeTransform === "none") {
      await handleGrid();
    } else {
      await handleTransform(activeTransform);
    }
  };

  const handleBoundaryDrawn = async (coords) => {
    setBoundaryMode(false);
    if (!projectId) return;
    try {
      const resp = await api.setDisplayBoundary(projectId, coords);
      setBoundaryPolygon(resp.display_boundary_polygon || null);
      await refreshCurrentOverlay();
    } catch (e) {
      handleError(e);
    }
  };

  const handleClearBoundary = async () => {
    if (!projectId) return;
    try {
      const resp = await api.setDisplayBoundary(projectId, null);
      setBoundaryPolygon(resp.display_boundary_polygon || null);
      await refreshCurrentOverlay();
    } catch (e) {
      handleError(e);
    }
  };

  // Lets a boundary drawn once be reused later - on a re-processed version
  // of the same survey, or on an entirely different project covering the
  // same physical area - without having to redraw the polygon by hand.
  const handleExportBoundary = () => {
    if (!boundaryPolygon) return;
    const base = boundaryFilename.trim().replace(/\.json$/i, "") || "boundary";
    api.downloadJson({ polygon: boundaryPolygon }, `${base}.json`);
  };

  // Regenerate the boundary from the flown lines at the current buffer.
  // Processing already does this once with the default buffer (see
  // ProcessParams.auto_display_boundary); this is how the user retunes it
  // without paying for a full reprocess.
  const handleAutoBoundary = async () => {
    if (!projectId) return;
    try {
      setError(null);
      setAutoBoundaryBusy(true);
      const resp = await api.autoDisplayBoundary(projectId, boundaryBufferM);
      setBoundaryPolygon(resp.display_boundary_polygon || null);
      setAutoBoundaryInfo(resp);
      await refreshCurrentOverlay();
    } catch (e) {
      handleError(e);
    } finally {
      setAutoBoundaryBusy(false);
    }
  };

  const handleExportBoundaryGis = async (format) => {
    if (!projectId || !boundaryPolygon) return;
    try {
      setError(null);
      const base = boundaryFilename.trim().replace(/\.(json|kml|zip)$/i, "") || "boundary";
      await api.exportDisplayBoundary(projectId, format, base);
    } catch (e) {
      handleError(e);
    }
  };

  const handleImportBoundaryFile = async (file) => {
    if (!projectId || !file) return;
    try {
      setError(null);
      const text = await file.text();
      const data = JSON.parse(text);
      const polygon = Array.isArray(data) ? data : data.polygon;
      if (!Array.isArray(polygon) || polygon.length < 3) {
        throw new Error("올바른 경계 파일이 아닙니다 (최소 3개의 [위도, 경도] 좌표 배열이 필요합니다).");
      }
      const resp = await api.setDisplayBoundary(projectId, polygon);
      setBoundaryPolygon(resp.display_boundary_polygon || null);
      await refreshCurrentOverlay();
    } catch (e) {
      handleError(e);
    }
  };

  const toggleInspectMode = () => {
    setInspectMode((v) => !v);
    setInspectPoints([]);
  };

  const handleInspectClick = async (lat, lon) => {
    if (!projectId) return;
    try {
      const resp = await api.sampleOverlayValue(projectId, lat, lon);
      setInspectPoints((prev) => [...prev, resp]);
    } catch (e) {
      handleError(e);
    }
  };

  // "최댓값/최솟값 위치로 이동" - clicking the exact extreme cell by hand is
  // genuinely hard once cells are only a few screen pixels wide at typical
  // map zoom, so this flies the map straight to the backend-reported
  // extremum and drops an inspect pin there instead of relying on the
  // user's click precision.
  const handleJumpToExtremum = (which) => {
    const loc = overlay?.extrema?.[which];
    if (!loc) return;
    setFlyToTarget({ lat: loc.lat, lon: loc.lon, nonce: Date.now() });
    setInspectMode(true);
    handleInspectClick(loc.lat, loc.lon);
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
      const summary = await api.manualExclude(projectId, { mode: "point_ids", action, point_ids: pointIds, include_ramp: showRampPoints });
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
      setBaseTimeseries(null);
      setLineProfileLoadingId(lineId);
      const resp = await api.getLineProfile(projectId, lineId, valueField);
      setLineProfileData(resp);
    } catch (e) {
      handleError(e);
    } finally {
      setLineProfileLoadingId(null);
    }
  };

  // Removes a user-identified ground-structure distortion (house, fence,
  // parked vehicle...) by interpolating anomaly/tmi across the selected
  // points - see backend set_manual_smoothing. Always recomputed fresh
  // from the pristine post-pipeline snapshot server-side, so this is safe
  // to call repeatedly without compounding smoothing on smoothing.
  //
  // If a grid/overlay was already being shown, regenerate it with the
  // same settings instead of just clearing it - otherwise the map simply
  // goes blank after applying smoothing, with nothing visually confirming
  // whether the edit actually worked (the underlying point values do
  // update either way; this only affects the gridded raster view).
  const _afterSmoothingApplied = async (summary) => {
    setProcessSummary(summary);
    await refreshPoints(projectId, valueField);
    if (lineProfileData) {
      const refreshed = await api.getLineProfile(projectId, lineProfileData.line_id, valueField);
      setLineProfileData(refreshed);
    }
    if (overlay) {
      await handleGrid();
    } else {
      setOverlay(null);
    }
  };

  const handleApplySmoothing = async (pointIds) => {
    try {
      setError(null);
      setSmoothing(true);
      const summary = await api.applySmoothing(projectId, { mode: "point_ids", point_ids: pointIds });
      if (Array.isArray(summary.manual_smooth_point_ids)) {
        setSmoothHistory((prev) => [...prev, summary.manual_smooth_point_ids]);
      }
      await _afterSmoothingApplied(summary);
    } catch (e) {
      handleError(e);
    } finally {
      setSmoothing(false);
    }
  };

  const handleSmoothPolygonDrawn = async (latlngCoords) => {
    try {
      setError(null);
      setSmoothing(true);
      const summary = await api.applySmoothing(projectId, { mode: "polygon", polygon: latlngCoords });
      if (Array.isArray(summary.manual_smooth_point_ids)) {
        setSmoothHistory((prev) => [...prev, summary.manual_smooth_point_ids]);
      }
      await _afterSmoothingApplied(summary);
    } catch (e) {
      handleError(e);
    } finally {
      setSmoothing(false);
    }
  };

  // Undoes only the most recent smoothing action, restoring whatever the
  // full smoothed-point set looked like right before it - reset (clear
  // manual_smooth_point_ids to empty) then, if any earlier actions
  // remain, re-apply their combined result in one point_ids call, since
  // the backend only ever unions point_ids into the set and has no
  // "shrink to exactly this" mode on its own.
  const handleUndoSmoothing = async () => {
    if (smoothHistory.length === 0) return;
    try {
      setError(null);
      setSmoothing(true);
      const newHistory = smoothHistory.slice(0, -1);
      const target = newHistory.length > 0 ? newHistory[newHistory.length - 1] : [];
      let summary = await api.applySmoothing(projectId, { mode: "reset" });
      if (target.length > 0) {
        summary = await api.applySmoothing(projectId, { mode: "point_ids", point_ids: target });
      }
      setSmoothHistory(newHistory);
      await _afterSmoothingApplied(summary);
    } catch (e) {
      handleError(e);
    } finally {
      setSmoothing(false);
    }
  };

  const handleResetAllSmoothing = async () => {
    if (smoothHistory.length === 0) return;
    try {
      setError(null);
      setSmoothing(true);
      const summary = await api.applySmoothing(projectId, { mode: "reset" });
      setSmoothHistory([]);
      await _afterSmoothingApplied(summary);
    } catch (e) {
      handleError(e);
    } finally {
      setSmoothing(false);
    }
  };

  const handleShowBaseTimeseries = async () => {
    try {
      setError(null);
      setLineProfileData(null);
      setBaseTimeseriesLoading(true);
      const resp = await api.getBaseTimeseries(projectId);
      setBaseTimeseries(resp);
    } catch (e) {
      handleError(e);
    } finally {
      setBaseTimeseriesLoading(false);
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
      if (layer?.type === "tiles") {
        api.unregisterLocalTileFolder(layer.tile_layer_id).catch(() => {});
      } else if (layer && projectId) {
        api.deleteReferenceLayer(projectId, layer.name).catch(() => {});
      }
      return prev.filter((l) => l.id !== id);
    });
  };

  const handleRegisterLocalTileFolder = async (path, label, scheme) => {
    try {
      setTileFolderError(null);
      setTileFolderRegistering(true);
      const resp = await api.registerLocalTileFolder({ path, label, scheme });
      setOverlayLayers((prev) => [
        ...prev,
        {
          id: `${Date.now()}-${Math.random()}`,
          name: resp.label,
          type: "tiles",
          tile_layer_id: resp.id,
          bounds: resp.bounds,
          min_zoom: resp.min_zoom,
          max_zoom: resp.max_zoom,
          opacity: 1,
          visible: true,
        },
      ]);
    } catch (e) {
      setTileFolderError(e.message || String(e));
    } finally {
      setTileFolderRegistering(false);
    }
  };

  const handlePickTileFolder = async () => {
    const resp = await api.pickLocalTileFolder();
    return resp.path;
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
        flight_agl_m: eulerFlightAglM,
      });
      setEulerResult(resp);
    } catch (e) {
      setEulerError(e.message || String(e));
    } finally {
      setEulerRunning(false);
    }
  };

  const handleUploadRepeatability = async (files) => {
    try {
      setRepeatabilityError(null);
      setRepeatabilityUploading(true);
      const id = await ensureProject();
      const summary = await api.uploadRepeatability(id, files);
      setRepeatabilityUploadInfo(summary);
      setRepeatabilityResult(null);
    } catch (e) {
      setRepeatabilityError(e.message || String(e));
    } finally {
      setRepeatabilityUploading(false);
    }
  };

  const handleAnalyzeRepeatability = async () => {
    try {
      setRepeatabilityError(null);
      setRepeatabilityAnalyzing(true);
      const resp = await api.analyzeRepeatability(projectId);
      setRepeatabilityResult(resp);
    } catch (e) {
      setRepeatabilityError(e.message || String(e));
    } finally {
      setRepeatabilityAnalyzing(false);
    }
  };

  const handleViewSpectrum = async () => {
    try {
      setSpectrumError(null);
      setSpectrumLoading(true);
      const resp = await api.getPowerSpectrum(projectId, { line_id: spectrumLineId, value: spectrumValue });
      setSpectrumResult(resp);
    } catch (e) {
      setSpectrumError(e.message || String(e));
    } finally {
      setSpectrumLoading(false);
    }
  };

  const handleAddNotchFrequency = (freqHz) => {
    setProcessParams((p) => ({
      ...p,
      notch_filter: { ...p.notch_filter, frequencies_hz: [...p.notch_filter.frequencies_hz, freqHz] },
    }));
  };

  const handleRemoveNotchFrequency = (freqHz) => {
    setProcessParams((p) => ({
      ...p,
      notch_filter: { ...p.notch_filter, frequencies_hz: p.notch_filter.frequencies_hz.filter((f) => f !== freqHz) },
    }));
  };

  const handleRunMultiscaleEdges = async () => {
    try {
      setMultiscaleError(null);
      setMultiscaleRunning(true);
      const heights_m = multiscaleHeightsText
        .split(",")
        .map((s) => parseFloat(s.trim()))
        .filter((v) => !Number.isNaN(v));
      const resp = await api.runMultiscaleEdges(projectId, {
        value: valueField,
        cell_size_m: gridCellSize,
        method: gridMethod,
        max_distance_m: gridMaxDistance,
        heights_m,
        percentile: multiscalePercentile,
      });
      setMultiscaleResult(resp);
    } catch (e) {
      setMultiscaleError(e.message || String(e));
    } finally {
      setMultiscaleRunning(false);
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

  const handleRunQcCertificate = async () => {
    try {
      setQcCertificateError(null);
      setQcCertificateRunning(true);
      const resp = await api.runQcCertificate(projectId, qcCertificateParams);
      setQcCertificateResult(resp);
    } catch (e) {
      setQcCertificateError(e.message || String(e));
    } finally {
      setQcCertificateRunning(false);
    }
  };

  const handleRunLineamentExtraction = async () => {
    try {
      setLineamentError(null);
      setLineamentRunning(true);
      const resp = await api.runLineamentExtraction(projectId, lineamentParams);
      setLineamentResult(resp);
    } catch (e) {
      setLineamentError(e.message || String(e));
    } finally {
      setLineamentRunning(false);
    }
  };

  const handleRunTiltDepth = async () => {
    try {
      setTiltDepthError(null);
      setTiltDepthRunning(true);
      const resp = await api.runTiltDepth(projectId, { value: "anomaly", cell_size_m: depthEstimationCellSize, method: "nearest", ...tiltDepthParams });
      setTiltDepthResult(resp);
    } catch (e) {
      setTiltDepthError(e.message || String(e));
    } finally {
      setTiltDepthRunning(false);
    }
  };

  const handleRunAsDepth = async () => {
    try {
      setAsDepthError(null);
      setAsDepthRunning(true);
      const resp = await api.runAnalyticSignalDepth(projectId, { value: "anomaly", cell_size_m: depthEstimationCellSize, method: "nearest", ...asDepthParams });
      setAsDepthResult(resp);
    } catch (e) {
      setAsDepthError(e.message || String(e));
    } finally {
      setAsDepthRunning(false);
    }
  };

  const handleRunSpectralDepth = async () => {
    try {
      setSpectralDepthError(null);
      setSpectralDepthRunning(true);
      const resp = await api.runSpectralDepth(projectId, { value: "anomaly", cell_size_m: depthEstimationCellSize, method: "nearest" });
      setSpectralDepthResult(resp);
    } catch (e) {
      setSpectralDepthError(e.message || String(e));
    } finally {
      setSpectralDepthRunning(false);
    }
  };

  const handleRunContactDetection = async () => {
    try {
      setContactError(null);
      setContactRunning(true);
      const resp = await api.runContactDetection(projectId, contactParams);
      setContactResult(resp);
    } catch (e) {
      setContactError(e.message || String(e));
    } finally {
      setContactRunning(false);
    }
  };

  const handleRunProspectivity = async () => {
    try {
      setProspectivityError(null);
      setProspectivityRunning(true);
      const resp = await api.runProspectivity(projectId, prospectivityParams);
      setProspectivityResult(resp);
      if (resp.available) {
        const overlay = await api.getProspectivityOverlay(projectId, "viridis");
        setProspectivityOverlay(overlay);
      } else {
        setProspectivityOverlay(null);
      }
    } catch (e) {
      setProspectivityError(e.message || String(e));
    } finally {
      setProspectivityRunning(false);
    }
  };

  const handleRunStructureScan = async () => {
    try {
      setStructureScanError(null);
      setStructureScanRunning(true);
      const resp = await api.scanStructureDistortion(projectId, structureScanParams);
      setStructureScanResult(resp);
      // Mapped buildings/roads are real structures almost by definition,
      // so pre-select all of them for a one-click bulk apply (this is the
      // whole point - replacing drawing one polygon per structure by
      // hand). A signal spike with no matching mapped structure gets left
      // unchecked though, since that could just as easily be real geology
      // as an unmapped structure - worth a deliberate look before
      // smoothing it away.
      setSelectedStructurePolygonIndices(new Set((resp.structure_polygons || []).map((_p, i) => i)));
      setSelectedStructureAnomalyIndices(
        new Set((resp.anomalies || []).map((a, i) => (a.matched_structure ? i : null)).filter((i) => i !== null))
      );
    } catch (e) {
      setStructureScanError(e.message || String(e));
    } finally {
      setStructureScanRunning(false);
    }
  };

  const handleToggleStructurePolygon = (i) => {
    setSelectedStructurePolygonIndices((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  };

  const handleToggleStructureAnomaly = (i) => {
    setSelectedStructureAnomalyIndices((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  };

  const handleApplyStructureSmoothing = async () => {
    if (!structureScanResult) return;
    try {
      setError(null);
      setStructureScanApplying(true);
      const polygons = [];
      for (const i of selectedStructurePolygonIndices) {
        polygons.push(structureScanResult.structure_polygons[i]);
      }
      for (const i of selectedStructureAnomalyIndices) {
        const a = structureScanResult.anomalies[i];
        // point candidates aren't polygons on their own - a small circle
        // covering the detected footprint stands in for one, same as a
        // hand-drawn region would.
        polygons.push(circlePolygon(a.lat, a.lon, Math.max(a.footprint_m, 3)));
      }
      if (polygons.length === 0) return;
      const summary = await api.applySmoothing(projectId, { mode: "polygons", polygons });
      if (Array.isArray(summary.manual_smooth_point_ids)) {
        setSmoothHistory((prev) => [...prev, summary.manual_smooth_point_ids]);
      }
      await _afterSmoothingApplied(summary);
    } catch (e) {
      handleError(e);
    } finally {
      setStructureScanApplying(false);
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

  const handleToggleSectionDrawMode = () => {
    const next = !sectionDrawMode;
    exitAllDrawModes();
    if (next) setSectionDrawMode(true);
  };

  const handleSectionPathDrawn = async (latlngCoords) => {
    setSectionDrawMode(false);
    setLastCustomSectionPath(latlngCoords);
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

  // Lets the WorkflowProgress checklist double as navigation in the long
  // (17-section) accordion sidebar: clicking a step opens its <details>
  // and its sidebar (drone/base/process/grid live in the left panel,
  // the advanced step in the right one) and scrolls it into view, instead
  // of the checklist being purely informational and leaving the user to
  // hunt for the matching section themselves.
  const handleWorkflowStepClick = (sectionId) => {
    if (!sectionId) return;
    const el = document.getElementById(sectionId);
    if (!el) return;
    if (typeof el.open === "boolean") el.open = true;
    if (sectionId === "wf-section-advanced") {
      setRightSidebarOpen(true);
    } else {
      setLeftSidebarOpen(true);
    }
    requestAnimationFrame(() => el.scrollIntoView({ behavior: "smooth", block: "start" }));
  };

  const handleMapShapeDrawn = (coords) => {
    if (sectionDrawMode) {
      handleSectionPathDrawn(coords);
    } else if (smoothDrawMode) {
      handleSmoothPolygonDrawn(coords);
    } else if (geologyDrawMode) {
      handleGeologyPolygonDrawn(coords);
    } else {
      handleShapeDrawn(coords);
    }
  };

  const handleToggleGeologyDrawMode = () => {
    const next = !geologyDrawMode;
    exitAllDrawModes();
    if (next) setGeologyDrawMode(true);
  };

  const handleGeologyPolygonDrawn = async (latlngCoords) => {
    setGeologyDrawMode(false);
    try {
      const resp = await api.addGeologyUnit(projectId, {
        name: `지질블록 ${geologyUnits.length + 1}`,
        susceptibility_si: 0.01,
        path: latlngCoords,
      });
      setGeologyUnits(resp.units);
    } catch (e) {
      handleError(e);
    }
  };

  const handleUpdateGeologyUnit = async (unitId, patch) => {
    // Optimistic local update so typing in the name/SI fields feels
    // immediate; the server response (awaited but not blocking the UI)
    // then reconciles in case of a validation error.
    setGeologyUnits((prev) => prev.map((u) => (u.id === unitId ? { ...u, ...patch } : u)));
    try {
      const resp = await api.updateGeologyUnit(projectId, unitId, patch);
      setGeologyUnits(resp.units);
    } catch (e) {
      handleError(e);
    }
  };

  const handleDeleteGeologyUnit = async (unitId) => {
    try {
      const resp = await api.deleteGeologyUnit(projectId, unitId);
      setGeologyUnits(resp.units);
    } catch (e) {
      handleError(e);
    }
  };

  // 거리/면적 측정 - 다른 그리기 모드(측선 제외, 스무딩 영역, 단면 프로파일)와
  // 동시에 켜져 있으면 지도 위에 두 개의 그리기 컨트롤이 겹쳐 혼란스러우므로
  // 서로 배타적으로 켭니다.
  const toggleMeasureMode = (mode) => {
    const next = measureMode === mode ? null : mode;
    exitAllDrawModes();
    if (next) setMeasureMode(next);
  };

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

  const handleOpenVolume = async () => {
    try {
      setInversionError(null);
      const threshold = volumeThreshold === "" ? null : volumeThreshold;
      const thresholdMax = volumeThresholdMax === "" ? null : volumeThresholdMax;
      const resp = await api.getInversionVolume(projectId, threshold, thresholdMax);
      setVolumeData({ ...resp, threshold, thresholdMax });
      setBoxFacesData(null);
      setVolumeSlices([]);
    } catch (e) {
      setInversionError(e.message || String(e));
    }
  };

  const handleLoadBoxFaces = async (topLayerIndex) => {
    try {
      setBoxTopLayerIndex(topLayerIndex);
      setBoxFacesLoading(true);
      const resp = await api.getInversionBoxFaces(projectId, topLayerIndex);
      setBoxFacesData(resp);
    } catch (e) {
      setInversionError(e.message || String(e));
    } finally {
      setBoxFacesLoading(false);
    }
  };

  // Fetches one interior section plane (동서/남북/기울기/자유선) for the
  // given slice spec and stores the result back onto that spec by id -
  // any number of these can be added at once to overlay on the
  // isosurface "blob" in the same 3D scene. Continuous SI coloring, no
  // threshold gating, matching the "박스" view's convention so each
  // section reads as full context rather than being clipped to whatever
  // range the isosurface threshold uses.
  const fetchVolumeSliceData = async (spec) => {
    setVolumeSlices((prev) => prev.map((s) => (s.id === spec.id ? { ...s, loading: true } : s)));
    try {
      const body =
        spec.orientation === "custom"
          ? { orientation: "custom", path: spec.path }
          : spec.orientation === "azimuth"
          ? { orientation: "azimuth", azimuth_deg: spec.azimuthDeg }
          : { orientation: spec.orientation, position_frac: spec.positionFrac };
      const resp = await api.getInversionSlice3D(projectId, body);
      setVolumeSlices((prev) => prev.map((s) => (s.id === spec.id ? { ...s, data: resp, loading: false } : s)));
    } catch (e) {
      setInversionError(e.message || String(e));
      setVolumeSlices((prev) => prev.map((s) => (s.id === spec.id ? { ...s, loading: false } : s)));
    }
  };

  const handleAddVolumeSlice = (orientation) => {
    if (orientation === "custom" && !lastCustomSectionPath) return;
    const spec = {
      id: `${orientation}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      orientation,
      positionFrac: 0.5,
      azimuthDeg: 45,
      path: orientation === "custom" ? lastCustomSectionPath : null,
      data: null,
      loading: false,
    };
    setVolumeSlices((prev) => [...prev, spec]);
    fetchVolumeSliceData(spec);
  };

  const handleRemoveVolumeSlice = (id) => {
    setVolumeSlices((prev) => prev.filter((s) => s.id !== id));
  };

  const handleUpdateVolumeSlice = (id, patch) => {
    setVolumeSlices((prev) => {
      const next = prev.map((s) => (s.id === id ? { ...s, ...patch } : s));
      const updated = next.find((s) => s.id === id);
      if (updated) fetchVolumeSliceData(updated);
      return next;
    });
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

  // 이착륙 램프 구간은 기본적으로 지도/측선 편집 화면에서 숨긴다 (표시 옵션을
  // 켜기 전까지는 제외/복원 드로잉의 대상도 되지 않도록 백엔드에서도 별도로
  // 보호함 - handleShapeDrawn의 include_ramp 참고).
  const editablePoints = useMemo(() => {
    if (showRampPoints) return points;
    return points.filter((p) => !(p.is_ramp && p.excluded));
  }, [points, showRampPoints]);

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
      ? `역산 자화율 (SI) - 고도 약 ${overlay?.elevation_m?.toFixed(0) ?? "?"} m (레이어 ${overlay?.layer_index ?? "?"}/${(overlay?.n_layers ?? 1) - 1})`
      : activeTransform !== "none"
        ? activeTransform.toUpperCase()
        : valueField === "anomaly"
          ? "자력 이상"
          : "TMI";
  const legendRange = overlay ? { vmin: overlay.vmin, vmax: overlay.vmax } : colorRange;

  return (
    <div style={{ display: "flex", height: "100%", fontFamily: '"Pretendard", system-ui, "Malgun Gothic", "Apple SD Gothic Neo", sans-serif' }}>
      <div
        className={`app-sidebar-backdrop${leftSidebarOpen || rightSidebarOpen ? " visible" : ""}`}
        onClick={() => {
          setLeftSidebarOpen(false);
          setRightSidebarOpen(false);
        }}
      />
      <div
        className={`app-sidebar-left${leftSidebarOpen ? " open" : ""}`}
        style={{ width: 320, borderRight: "1px solid #e6dac0", overflowY: "auto", padding: 12, background: "#faf6ec" }}
      >
        <h1 style={{ fontSize: 16, margin: "4px 0 2px 0" }}>드론 자력탐사 자료 처리</h1>
        <div
          style={{ fontSize: 10, color: "#ab9a78", margin: "0 0 12px 0" }}
          title="이 화면이 백엔드 서버로부터 실제로 받은 버전 정보입니다 - '분명히 고쳐달라고 한 문제가 그대로'라면 run.bat을 다시 실행해 이 커밋 해시가 바뀌는지부터 확인하세요"
        >
          {backendVersion?.commit ? `버전: ${backendVersion.commit} (${backendVersion.commit_date})` : ""}
        </div>
        <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
          <button
            style={{
              flex: 1,
              padding: "6px 8px",
              fontSize: 12,
              borderRadius: 6,
              border: purposeMode === "mineral" ? "1px solid #a9631f" : "1px solid #ddd0b2",
              background: purposeMode === "mineral" ? "#a9631f" : "white",
              color: purposeMode === "mineral" ? "white" : "#4a3d28",
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
              border: purposeMode === "target" ? "1px solid #b91c1c" : "1px solid #ddd0b2",
              background: purposeMode === "target" ? "#b91c1c" : "white",
              color: purposeMode === "target" ? "white" : "#4a3d28",
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
          onStepClick={handleWorkflowStepClick}
        />
        <input
          type="text"
          value={saveFilename}
          onChange={(e) => setSaveFilename(e.target.value)}
          placeholder="저장할 파일명 (확장자 제외, 여러 프로젝트는 다른 이름으로 저장)"
          title="여러 프로젝트를 구분해 저장/불러오려면 프로젝트마다 다른 파일명을 지정하세요"
          style={{
            width: "100%",
            marginBottom: 6,
            padding: "5px 8px",
            fontSize: 12,
            borderRadius: 6,
            border: "1px solid #ddd0b2",
            boxSizing: "border-box",
          }}
        />
        <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
          <button
            style={{
              flex: 1,
              padding: "6px 8px",
              fontSize: 12,
              borderRadius: 6,
              border: "1px solid #a9631f",
              background: "white",
              color: "#a9631f",
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
              border: "1px solid #a9631f",
              background: "white",
              color: "#a9631f",
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
            border: "1px solid #a9631f",
            background: "white",
            color: "#a9631f",
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
          onShowBaseTimeseries={handleShowBaseTimeseries}
          baseTimeseriesLoading={baseTimeseriesLoading}
          onUploadIaga2002={handleUploadIaga2002}
          onFetchIntermagnet={handleFetchIntermagnet}
          onApplyIntermagnet={handleApplyIntermagnet}
          onCancelIntermagnetPreview={handleCancelIntermagnetPreview}
          intermagnetPreview={intermagnetPreview}
          intermagnetLoading={intermagnetLoading}
          intermagnetApplying={intermagnetApplying}
          intermagnetError={intermagnetError}
          onFetchNearestIntermagnet={handleFetchNearestIntermagnet}
          onApplyNearestIntermagnet={handleApplyNearestIntermagnet}
          onCancelNearestIntermagnetPreview={handleCancelNearestIntermagnetPreview}
          onExportNearestIntermagnetCsv={handleExportNearestIntermagnetCsv}
          nearestIntermagnetCsvFilename={nearestIntermagnetCsvFilename}
          setNearestIntermagnetCsvFilename={setNearestIntermagnetCsvFilename}
          onShowNearestIntermagnetComparison={handleShowNearestIntermagnetComparison}
          nearestIntermagnetPreview={nearestIntermagnetPreview}
          nearestIntermagnetLoading={nearestIntermagnetLoading}
          nearestIntermagnetApplying={nearestIntermagnetApplying}
          nearestIntermagnetError={nearestIntermagnetError}
          nearestIntermagnetHasResult={nearestIntermagnetHasResult}
          nearestIntermagnetComparisonLoading={nearestIntermagnetComparisonLoading}
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
          overlay={overlay}
          onGridConfidence={handleGridConfidence}
          confidenceLoading={confidenceLoading}
          confidenceOverlay={confidenceOverlay}
          showConfidenceOverlay={showConfidenceOverlay}
          setShowConfidenceOverlay={setShowConfidenceOverlay}
          boundaryMode={boundaryMode}
          onToggleBoundaryMode={handleToggleBoundaryMode}
          boundaryPolygon={boundaryPolygon}
          onClearBoundary={handleClearBoundary}
          onExportBoundary={handleExportBoundary}
          boundaryFilename={boundaryFilename}
          setBoundaryFilename={setBoundaryFilename}
          onImportBoundaryFile={handleImportBoundaryFile}
          boundaryBufferM={boundaryBufferM}
          setBoundaryBufferM={setBoundaryBufferM}
          onAutoBoundary={handleAutoBoundary}
          autoBoundaryBusy={autoBoundaryBusy}
          autoBoundaryInfo={autoBoundaryInfo}
          onExportBoundaryGis={handleExportBoundaryGis}
          inspectMode={inspectMode}
          onToggleInspectMode={toggleInspectMode}
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
          points={editablePoints}
          colorRange={legendRange}
          cmapName={cmapName}
          overlay={overlay}
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
          prospectivityTargets={showProspectivityTargets ? prospectivityResult?.targets : null}
          prospectivityOverlay={showProspectivityOverlay ? prospectivityOverlay : null}
          onBoundsChange={handleMapBoundsChange}
          inspectMode={inspectMode}
          inspectPoints={inspectPoints}
          onInspectClick={handleInspectClick}
          measureMode={measureMode}
          measurements={measurements}
          onMeasureShapeDrawn={handleMeasureShapeDrawn}
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

      <div
        className={`app-sidebar-right${rightSidebarOpen ? " open" : ""}`}
        style={{ width: 280, borderLeft: "1px solid #e6dac0", overflowY: "auto", padding: 12, background: "#faf6ec" }}
      >
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
          showRampPoints={showRampPoints}
          onToggleShowRampPoints={setShowRampPoints}
        />

        <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>7-1. 지상구조물 왜곡 자동탐지 (OSM + 신호분석)</h2>
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
          onRegisterTileFolder={handleRegisterLocalTileFolder}
          tileFolderRegistering={tileFolderRegistering}
          tileFolderError={tileFolderError}
          onPickTileFolder={handlePickTileFolder}
        />

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

        <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>17. 거리/면적 측정</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 10 }}>
          <div style={{ display: "flex", gap: 6 }}>
            <button
              onClick={() => toggleMeasureMode("distance")}
              style={{
                flex: 1,
                padding: "6px 8px",
                fontSize: 12,
                borderRadius: 6,
                border: `1px solid ${measureMode === "distance" ? "#dc2626" : "#ddd0b2"}`,
                background: measureMode === "distance" ? "#dc2626" : "white",
                color: measureMode === "distance" ? "white" : "#4a3d28",
                cursor: "pointer",
              }}
              title="지도를 클릭해 선을 그리면(더블클릭으로 종료) 그 경로의 총 거리를 계산해 표시합니다. 이상대 길이 등을 잴 때 사용하세요. 계속 여러 선을 그릴 수 있고, 버튼을 다시 누르면 그리기 모드만 꺼집니다(측정 결과는 남습니다)."
            >
              {measureMode === "distance" ? "거리 측정 중 (다시 누르면 종료)" : "거리 측정"}
            </button>
            <button
              onClick={() => toggleMeasureMode("area")}
              style={{
                flex: 1,
                padding: "6px 8px",
                fontSize: 12,
                borderRadius: 6,
                border: `1px solid ${measureMode === "area" ? "#7c3aed" : "#ddd0b2"}`,
                background: measureMode === "area" ? "#7c3aed" : "white",
                color: measureMode === "area" ? "white" : "#4a3d28",
                cursor: "pointer",
              }}
              title="지도를 클릭해 다각형을 그리면(더블클릭 또는 시작점 클릭으로 종료) 그 영역의 면적을 계산해 표시합니다. 이상대 면적 등을 잴 때 사용하세요. 계속 여러 영역을 그릴 수 있고, 버튼을 다시 누르면 그리기 모드만 꺼집니다(측정 결과는 남습니다)."
            >
              {measureMode === "area" ? "면적 측정 중 (다시 누르면 종료)" : "면적 측정"}
            </button>
          </div>
          {measurements.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {measurements.map((m) => (
                <div
                  key={m.id}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    fontSize: 12,
                    background: "white",
                    border: "1px solid #e6dac0",
                    borderRadius: 6,
                    padding: "4px 8px",
                  }}
                >
                  <span>
                    {m.label}: {m.valueText}
                  </span>
                  <button
                    onClick={() => handleRemoveMeasurement(m.id)}
                    style={{ border: "none", background: "transparent", color: "#ab9a78", cursor: "pointer", fontSize: 13, lineHeight: 1 }}
                    title="이 측정 삭제"
                  >
                    ✕
                  </button>
                </div>
              ))}
              <button
                onClick={handleClearMeasurements}
                style={{ alignSelf: "flex-start", fontSize: 11, color: "#8a7a5c", background: "transparent", border: "none", cursor: "pointer", textDecoration: "underline", padding: 0 }}
              >
                전체 지우기
              </button>
            </div>
          )}
        </div>

        <h2 style={{ fontSize: 13, margin: "16px 0 10px 0" }}>범례</h2>
        <Legend
          label={legendLabel}
          unit="nT"
          vmin={legendRange.vmin}
          vmax={legendRange.vmax}
          ticks={overlay?.legend_ticks}
          stretch={overlay?.stretch}
          cmapName={cmapName}
          onCmapChange={setCmapName}
          manualRange={manualRange}
          onManualRangeChange={setManualRange}
          stats={legendStats}
          hoverPoint={hoverPoint}
          extrema={overlay?.extrema}
          onJumpToExtremum={handleJumpToExtremum}
        />
        {overlay?.cell_size_guideline_warning && (
          <div style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, color: "#b45309", background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 6 }}>
            ⚠ {overlay.cell_size_guideline_warning}
          </div>
        )}
        {overlay?.rtp_latitude_warning && (
          <div style={{ marginTop: 8, padding: "6px 8px", fontSize: 11, color: "#b45309", background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 6 }}>
            ⚠ {overlay.rtp_latitude_warning}
          </div>
        )}
      </div>
    </div>
  );
}
