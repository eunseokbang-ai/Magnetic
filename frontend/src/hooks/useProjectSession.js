import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../api";

// 프로젝트 한 건의 뼈대: 번호, 요약, 측점, 오류 표시, 저장 이름.
//
// One of the hooks App() is composed of (split out of App.jsx in 2026-11;
// the statements are unchanged). It receives what the hooks called before
// it returned and returns everything it declares.
export default function useProjectSession() {
  const [backendVersion, setBackendVersion] = useState(null);

  useEffect(() => {
    api.getVersion().then(setBackendVersion).catch(() => setBackendVersion(null));
  }, []);

  const [projectId, setProjectId] = useState(null);
  const [droneSummary, setDroneSummary] = useState(null);
  const [baseSummary, setBaseSummary] = useState(null);
  const [processSummary, setProcessSummary] = useState(null);
  const [points, setPoints] = useState([]);
  const [pointsLoading, setPointsLoading] = useState(false);
  const [valueField, setValueField] = useState("anomaly");
  const [hoverPoint, setHoverPoint] = useState(null);
  const [savingProject, setSavingProject] = useState(false);
  const [loadingProject, setLoadingProject] = useState(false);
  const [saveFilename, setSaveFilename] = useState("magnetic_project");
  const [exportingReport, setExportingReport] = useState(false);
  const [flyToTarget, setFlyToTarget] = useState(null);
  const [error, setError] = useState(null);
  const [mapBounds, setMapBounds] = useState(null);
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(false);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(false);

  // "용도" preset: mineral-exploration workflows want broad, coarse grids
  // and the 3D inversion/Euler panels front and center; near-surface
  // target detection (mines/UXO/hidden vehicles) wants a much finer grid
  // and the dipole-fit target panel front and center instead - switching
  // resets the grid cell size to that mode's usual scale.
  const [purposeMode, setPurposeMode] = useState("mineral"); // "mineral" | "target"

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

  const handleMapBoundsChange = useCallback((bounds) => setMapBounds(bounds), []);

  const refreshPoints = useCallback(async (id, value) => {
    setPointsLoading(true);
    try {
      const pts = await api.getPoints(id, value);
      setPoints(pts);
    } finally {
      setPointsLoading(false);
    }
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

  const handleExportReport = async (format = "md") => {
    try {
      setError(null);
      setExportingReport(true);
      if (format === "md") await api.exportReport(projectId, "processing_report.md");
      else await api.exportReportDocument(projectId, format, `processing_report.${format}`);
    } catch (e) {
      handleError(e);
    } finally {
      setExportingReport(false);
    }
  };

  // The address names the project (?project=<id>), so a refresh, a
  // reopened tab or a bookmark comes back to it instead of to an empty
  // screen. Written whenever the id changes; read once, on load.
  useEffect(() => {
    if (!projectId) return;
    const url = new URL(window.location.href);
    if (url.searchParams.get("project") !== projectId) {
      url.searchParams.set("project", projectId);
      window.history.replaceState(null, "", url);
    }
  }, [projectId]);

  return {
    backendVersion, setBackendVersion, projectId, setProjectId, droneSummary, setDroneSummary, baseSummary,
    setBaseSummary, processSummary, setProcessSummary, points, setPoints, pointsLoading, setPointsLoading,
    valueField, setValueField, hoverPoint, setHoverPoint, savingProject, setSavingProject, loadingProject,
    setLoadingProject, saveFilename, setSaveFilename, exportingReport, setExportingReport, flyToTarget,
    setFlyToTarget, error, setError, mapBounds, setMapBounds, leftSidebarOpen, setLeftSidebarOpen,
    rightSidebarOpen, setRightSidebarOpen, purposeMode, setPurposeMode, projectCreationRef, ensureProject,
    handleError, handleMapBoundsChange, refreshPoints, handleSaveProject, handleExportReport,
  };
}
