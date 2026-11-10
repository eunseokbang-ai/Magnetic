import { useEffect } from "react";
import * as api from "./api";
import JobProgress from "./components/JobProgress";
import useChat from "./hooks/useChat";
import useOfflineTiles from "./hooks/useOfflineTiles";
import useReferenceLayers from "./hooks/useReferenceLayers";
import useProjectSession from "./hooks/useProjectSession";
import useSurveyUploads from "./hooks/useSurveyUploads";
import useGridDisplay from "./hooks/useGridDisplay";
import useInversion from "./hooks/useInversion";
import useProcessingEdits from "./hooks/useProcessingEdits";
import useInterpretation from "./hooks/useInterpretation";
import useStructureRemoval from "./hooks/useStructureRemoval";
import useMeasureTool from "./hooks/useMeasureTool";
import LeftSidebar from "./components/layout/LeftSidebar";
import MapArea from "./components/layout/MapArea";
import EditingPanels from "./components/layout/EditingPanels";
import AnalysisPanels from "./components/layout/AnalysisPanels";
import MapReadout from "./components/layout/MapReadout";
import { AppContext } from "./AppContext";

export default function App() {
  // State and handlers live in the hooks below, one per part of the program.
  // Each is given what the earlier ones returned. What is left in this
  // function is what ties them together: loading a project touches all of
  // them, and so do the handlers that dispatch on which drawing mode is on.
  const session = useProjectSession();
  const {
    projectId, setProjectId, setDroneSummary, setBaseSummary, setProcessSummary, setPoints, valueField,
    setLoadingProject, setError, leftSidebarOpen, setLeftSidebarOpen, rightSidebarOpen, setRightSidebarOpen,
    setPurposeMode, projectCreationRef, ensureProject, handleError, refreshPoints,
  } = session;
  const uploads = useSurveyUploads(session);
  const { setBaseTimeseries, setBaseTimeseriesLoading } = uploads;
  const grid = useGridDisplay({ ...session, ...uploads });
  const { setGridCellSize, boundaryMode, setBoundaryMode, setActiveTransform, setOverlay } = grid;
  const inversion = useInversion({ ...session, ...uploads, ...grid });
  const {
    setGeologyUnits, geologyDrawMode, setGeologyDrawMode, setInversionSummary, sectionDrawMode,
    setSectionDrawMode, setSectionResult, setVolumeData, handleSectionPathDrawn, handleGeologyPolygonDrawn,
  } = inversion;
  const editing = useProcessingEdits({ ...session, ...uploads, ...grid, ...inversion });
  const {
    setProcessParams, setDrawMode, setLineProfileData, setSmoothing, smoothDrawMode, setSmoothDrawMode,
    setSmoothHistory, handleShapeDrawn, _afterSmoothingApplied,
  } = editing;
  const interpretation = useInterpretation({ ...session, ...uploads, ...grid, ...inversion, ...editing });
  const structures = useStructureRemoval({ ...session, ...uploads, ...grid, ...inversion, ...editing, ...interpretation });
  const { removalMethod, _removeStructuresByModel } = structures;
  const measure = useMeasureTool();
  const { measureMode, setMeasureMode } = measure;

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

  // Self-contained parts of the screen keep their state and handlers in
  // hooks (src/hooks/), with their list logic in src/lib/.
  const {
    overlayLayers, overlayUploading, overlayError, tileFolderRegistering, tileFolderError, restoreReferenceLayers, showGeneratedLayer,
    handleUploadOverlayImage, handleToggleOverlayVisible, handleSetOverlayOpacity, handleRemoveOverlay, handleMoveOverlay,
    handleRegisterLocalTileFolder, handleRegisterLocalRaster, handlePickRasterFile, handlePickTileFolder,
  } = useReferenceLayers({ projectId, ensureProject });

  const { tileEstimate, tileDownloadResult, tileStatus, tileLoading, tileError, handleEstimateTiles, handleDownloadTiles, handleRefreshTileStatus } =
    useOfflineTiles();

  const { chatMessages, chatSending, chatError, handleSendChatMessage } = useChat({ ensureProject });

  // Shared by "프로젝트 불러오기" and by restoring an autosave: both hand
  // back the same payload and have to put the screen back into the same
  // state, so this must not be duplicated between them.
  const applyLoadedProject = async (id, resp) => {
    // Reference layers saved with the project - before the first await, see
    // useReferenceLayers.restoreReferenceLayers for why.
    restoreReferenceLayers(resp.reference_layers);
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
  };

  const handleLoadProject = async (file) => {
    try {
      setError(null);
      setLoadingProject(true);
      const id = await ensureProject();
      const resp = await api.loadProject(id, file);
      await applyLoadedProject(id, resp);
    } catch (e) {
      handleError(e);
    } finally {
      setLoadingProject(false);
    }
  };

  useEffect(() => {
    const wanted = new URLSearchParams(window.location.search).get("project");
    if (!wanted) return;
    let cancelled = false;
    (async () => {
      setLoadingProject(true);
      try {
        const resp = await api.openProject(wanted);
        if (cancelled) return;
        projectCreationRef.current = Promise.resolve(resp.project_id);
        setProjectId(resp.project_id);
        await applyLoadedProject(resp.project_id, resp);
      } catch (e) {
        if (cancelled) return;
        // gone from memory and from disk: start clean rather than keep pointing at it
        const url = new URL(window.location.href);
        url.searchParams.delete("project");
        window.history.replaceState(null, "", url);
        setError(`주소가 가리키는 이전 작업을 찾지 못해 새로 시작합니다. (${e.message || e})`);
      } finally {
        if (!cancelled) setLoadingProject(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // once, on load
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRestoreAutosave = async (autosaveProjectId) => {
    setError(null);
    setLoadingProject(true);
    try {
      const resp = await api.restoreAutosave(autosaveProjectId);
      // The backend restores under the saved project's own id, so the UI
      // has to follow it there rather than keeping whatever empty project
      // this session had created.
      projectCreationRef.current = Promise.resolve(resp.project_id);
      setProjectId(resp.project_id);
      await applyLoadedProject(resp.project_id, resp);
    } finally {
      setLoadingProject(false);
    }
  };

  const handleToggleBoundaryMode = () => {
    const next = !boundaryMode;
    exitAllDrawModes();
    if (next) setBoundaryMode(true);
  };

  const handleSmoothPolygonDrawn = async (latlngCoords) => {
    try {
      setError(null);
      setSmoothing(true);
      if (removalMethod === "model") {
        await _removeStructuresByModel([latlngCoords], [null]);
        return;
      }
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

  const handleSetPurposeMode = (mode) => {
    setPurposeMode(mode);
    setGridCellSize(mode === "target" ? 1.0 : 10.0);
  };

  const handleToggleSectionDrawMode = () => {
    const next = !sectionDrawMode;
    exitAllDrawModes();
    if (next) setSectionDrawMode(true);
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

  // 거리/면적 측정 - 다른 그리기 모드(측선 제외, 스무딩 영역, 단면 프로파일)와
  // 동시에 켜져 있으면 지도 위에 두 개의 그리기 컨트롤이 겹쳐 혼란스러우므로
  // 서로 배타적으로 켭니다.
  const toggleMeasureMode = (mode) => {
    const next = measureMode === mode ? null : mode;
    exitAllDrawModes();
    if (next) setMeasureMode(next);
  };

  // Everything above, under one roof, for the parts of the screen that live in
  // their own files (components/layout/, components/workflow/): they read what
  // they need with useApp() instead of being handed a hundred props.
  const app = {
    ...session, ...uploads, ...grid, ...inversion, ...editing, ...interpretation, ...structures, ...measure,
    exitAllDrawModes, overlayLayers, overlayUploading, overlayError, tileFolderRegistering, tileFolderError,
    restoreReferenceLayers, showGeneratedLayer, handleUploadOverlayImage, handleToggleOverlayVisible,
    handleSetOverlayOpacity, handleRemoveOverlay, handleMoveOverlay, handleRegisterLocalTileFolder,
    handleRegisterLocalRaster, handlePickRasterFile, handlePickTileFolder, tileEstimate, tileDownloadResult,
    tileStatus, tileLoading, tileError, handleEstimateTiles, handleDownloadTiles, handleRefreshTileStatus,
    chatMessages, chatSending, chatError, handleSendChatMessage, applyLoadedProject, handleLoadProject,
    handleRestoreAutosave, handleToggleBoundaryMode, handleSmoothPolygonDrawn, handleShowBaseTimeseries,
    handleSetPurposeMode, handleToggleSectionDrawMode, handleWorkflowStepClick, handleMapShapeDrawn,
    handleToggleGeologyDrawMode, toggleMeasureMode,
  };

  return (
    <AppContext.Provider value={app}>
      <div style={{ display: "flex", height: "100%", fontFamily: '"Pretendard", system-ui, "Malgun Gothic", "Apple SD Gothic Neo", sans-serif' }}>
        <JobProgress />
        <div
          className={`app-sidebar-backdrop${leftSidebarOpen || rightSidebarOpen ? " visible" : ""}`}
          onClick={() => {
            setLeftSidebarOpen(false);
            setRightSidebarOpen(false);
          }}
        />
        <LeftSidebar />

        <MapArea />

        <div
          className={`app-sidebar-right${rightSidebarOpen ? " open" : ""}`}
          style={{ width: 280, borderLeft: "1px solid #e6dac0", overflowY: "auto", padding: 12, background: "#faf6ec" }}
        >
          <EditingPanels />

          <AnalysisPanels />

          <MapReadout />
        </div>
      </div>
    </AppContext.Provider>
  );
}
