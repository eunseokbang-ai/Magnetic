import { useEffect, useMemo, useState } from "react";
import * as api from "../api";
import { minMax } from "../arrayUtils";
import { DEFAULT_TRANSFORM_EXTRA_PARAMS } from "../appDefaults";

// 그리드와 파생 그리드의 계산·표시·내보내기, 색과 음영, 신뢰도 레이어, 표시 경계, 지점값 확인.
//
// One of the hooks App() is composed of (split out of App.jsx in 2026-11;
// the statements are unchanged). It receives what the hooks called before
// it returned and returns everything it declares.
export default function useGridDisplay(app) {
  const {
    handleError, points, processSummary, projectId, setError, setFlyToTarget, valueField,
  } = app;

  const [gridCellSize, setGridCellSize] = useState(10.0);
  const [gridMethod, setGridMethod] = useState("linear");
  const [gridMaxDistance, setGridMaxDistance] = useState(null);
  const [alongLineSmooth, setAlongLineSmooth] = useState(true);
  const [alongLineSmoothWavelength, setAlongLineSmoothWavelength] = useState(null);
  const [gridding, setGridding] = useState(false);
  const [boundaryMode, setBoundaryMode] = useState(false);
  const [boundaryPolygon, setBoundaryPolygon] = useState(null);
  const [boundaryFilename, setBoundaryFilename] = useState("boundary.json");
  // Buffer distance (m) for the auto-generated boundary - the one number
  // that decides how far past the outermost flight line the map extends.
  // Empty means "let the backend pick one line spacing", which is what
  // processing itself does (ProcessParams.display_boundary_buffer_m=None);
  // it gets filled in with the resolved value once processing reports back.
  const [boundaryBufferM, setBoundaryBufferM] = useState("");
  const [autoBoundaryBusy, setAutoBoundaryBusy] = useState(false);
  const [autoBoundaryInfo, setAutoBoundaryInfo] = useState(null);
  const [exportingGeotiff, setExportingGeotiff] = useState(false);
  const [activeTransform, setActiveTransform] = useState("none");
  const [transformLoading, setTransformLoading] = useState(false);
  const [overlay, setOverlay] = useState(null);
  const [confidenceOverlay, setConfidenceOverlay] = useState(null);
  const [showConfidenceOverlay, setShowConfidenceOverlay] = useState(false);
  const [confidenceLoading, setConfidenceLoading] = useState(false);
  const [inspectMode, setInspectMode] = useState(false);
  const [inspectPoints, setInspectPoints] = useState([]);
  const [cmapName, setCmapName] = useState("RdYlBu_r");
  const [manualRange, setManualRange] = useState({ enabled: false, vmin: null, vmax: null });
  const [gridOpacity, setGridOpacity] = useState(0.85);
  const [hillshade, setHillshade] = useState(false);
  const [hillshadeAzimuth, setHillshadeAzimuth] = useState(315);
  const [hillshadeAltitude, setHillshadeAltitude] = useState(45);
  const [hillshadeExaggeration, setHillshadeExaggeration] = useState(3);
  const [stretch, setStretch] = useState("linear");
  const [exportGeotiffColored, setExportGeotiffColored] = useState(false);
  const [showContours, setShowContours] = useState(true);    // display only: contours always come with the grid
  const [contourInterval, setContourInterval] = useState(null);
  const [contourNLevels, setContourNLevels] = useState(10);
  // Off by default. Points are drawn *over* the grid image, and since
  // that image is itself semi-transparent over the basemap while the
  // points are not, a point always composites more saturated than the
  // grid cell holding the same value - so every flight line shows as a
  // colored stripe whether or not the data has anything wrong with it.
  // No alpha fixes that (matching the grid exactly needs alpha 0), so the
  // overlay is opt-in: it answers "where did I actually fly", not "what
  // does the field look like", which is what the grid is for.
  const [showPointsOverGrid, setShowPointsOverGrid] = useState(false);
  const [showLineLabels, setShowLineLabels] = useState(false);
  const [transformExtraParams, setTransformExtraParams] = useState(DEFAULT_TRANSFORM_EXTRA_PARAMS);
  const [exportingXyz, setExportingXyz] = useState(false);
  const [exportingGrd, setExportingGrd] = useState(false);
  const [exportingPointsCsv, setExportingPointsCsv] = useState(false);
  const [lastDrawnPolygon, setLastDrawnPolygon] = useState(null);
  const [exportingBln, setExportingBln] = useState(false);
  // Opacity of the KIGAM geology overlays (MapView.GEOLOGY_LAYERS).
  const [geologyOpacity, setGeologyOpacity] = useState(0.6);

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
        regional_removal: transformExtraParams.regional_removal,
        microlevel_pre_apply: transformExtraParams.microlevel_pre_apply,
        microlevel_strength: transformExtraParams.microlevel_strength,
        microlevel_angle_tolerance_deg: transformExtraParams.microlevel_angle_tolerance_deg,
        microlevel_wavelength_factor: transformExtraParams.microlevel_wavelength_factor,
        microlevel_mode: transformExtraParams.microlevel_mode,
        microlevel_cutoff_factor: transformExtraParams.microlevel_cutoff_factor,
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
      // for the layer that is on the map: a derived grid has its own, carried through the transform
      const derived = activeTransform && activeTransform !== "none";
      const resp = derived
        ? await api.getTransformConfidence(projectId, {
            transform: activeTransform,
            value: valueField,
            cell_size_m: gridCellSize,
            method: gridMethod,
            max_distance_m: gridMaxDistance,
            along_line_smooth: alongLineSmooth,
            along_line_smooth_wavelength_m: alongLineSmoothWavelength,
            ...transformExtraParams,
          })
        : await api.getGridConfidence(projectId, {
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
      // Blank field = null = "one line spacing", same as processing's default.
      const bufferM = boundaryBufferM === "" || boundaryBufferM === null ? null : Number(boundaryBufferM);
      const resp = await api.autoDisplayBoundary(projectId, bufferM);
      setBoundaryPolygon(resp.display_boundary_polygon || null);
      setAutoBoundaryInfo(resp);
      // Show whatever the backend actually used, so an auto run stops
      // looking like an empty box.
      if (resp.buffer_m) setBoundaryBufferM(resp.buffer_m);
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

  const handleExportGpkg = async () => {
    try {
      setError(null);
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
        await api.exportGridGpkg(projectId, base, `${valueField}_${gridCellSize}m.gpkg`);
      } else {
        await api.exportTransformGpkg(projectId, { ...base, transform: activeTransform }, `${activeTransform}_${gridCellSize}m.gpkg`);
      }
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

  return {
    gridCellSize, setGridCellSize, gridMethod, setGridMethod, gridMaxDistance, setGridMaxDistance,
    alongLineSmooth, setAlongLineSmooth, alongLineSmoothWavelength, setAlongLineSmoothWavelength, gridding,
    setGridding, boundaryMode, setBoundaryMode, boundaryPolygon, setBoundaryPolygon, boundaryFilename,
    setBoundaryFilename, boundaryBufferM, setBoundaryBufferM, autoBoundaryBusy, setAutoBoundaryBusy,
    autoBoundaryInfo, setAutoBoundaryInfo, exportingGeotiff, setExportingGeotiff, activeTransform,
    setActiveTransform, transformLoading, setTransformLoading, overlay, setOverlay, confidenceOverlay,
    setConfidenceOverlay, showConfidenceOverlay, setShowConfidenceOverlay, confidenceLoading,
    setConfidenceLoading, inspectMode, setInspectMode, inspectPoints, setInspectPoints, cmapName,
    setCmapName, manualRange, setManualRange, gridOpacity, setGridOpacity, hillshade, setHillshade,
    hillshadeAzimuth, setHillshadeAzimuth, hillshadeAltitude, setHillshadeAltitude, hillshadeExaggeration,
    setHillshadeExaggeration, stretch, setStretch, exportGeotiffColored, setExportGeotiffColored,
    showContours, setShowContours, contourInterval, setContourInterval, contourNLevels, setContourNLevels,
    showPointsOverGrid, setShowPointsOverGrid, showLineLabels, setShowLineLabels, transformExtraParams,
    setTransformExtraParams, exportingXyz, setExportingXyz, exportingGrd, setExportingGrd,
    exportingPointsCsv, setExportingPointsCsv, lastDrawnPolygon, setLastDrawnPolygon, exportingBln,
    setExportingBln, geologyOpacity, setGeologyOpacity, handleGrid, handleGridConfidence,
    refreshCurrentOverlay, handleBoundaryDrawn, handleClearBoundary, handleExportBoundary,
    handleAutoBoundary, handleExportBoundaryGis, handleImportBoundaryFile, toggleInspectMode,
    handleInspectClick, handleJumpToExtremum, handleTransform, handleExportGeotiff, handleExportXyz,
    handleExportGrd, handleExportBln, handleExportGpkg, handleExportPointsCsv, autoColorRange, colorRange,
    legendStats, legendLabel, legendRange,
  };
}
