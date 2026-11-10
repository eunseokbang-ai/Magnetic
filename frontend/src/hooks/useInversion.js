import { useState } from "react";
import * as api from "../api";
import { DEFAULT_INVERSION_PARAMS } from "../appDefaults";

// 3차원 역산: DEM과 지질 단위 입력, 실행, 수평 단면·수직 단면·3차원 보기, 가져오기와 내보내기.
//
// One of the hooks App() is composed of (split out of App.jsx in 2026-11;
// the statements are unchanged). It receives what the hooks called before
// it returned and returns everything it declares.
export default function useInversion(app) {
  const {
    ensureProject, handleError, projectId, setActiveTransform, setError, setOverlay,
  } = app;

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
  const [showGeologyUnits, setShowGeologyUnits] = useState(true);

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

  return {
    demStatus, setDemStatus, demUploading, setDemUploading, autoParams, setAutoParams, inversionParams,
    setInversionParams, geologyUnits, setGeologyUnits, geologyDrawMode, setGeologyDrawMode, inversionRunning,
    setInversionRunning, inversionSummary, setInversionSummary, inversionError, setInversionError,
    sliceLayerIndex, setSliceLayerIndex, sliceThreshold, setSliceThreshold, sliceThresholdMax,
    setSliceThresholdMax, sectionProfile, setSectionProfile, sectionPositionFrac, setSectionPositionFrac,
    sectionDrawMode, setSectionDrawMode, sectionThreshold, setSectionThreshold, sectionThresholdMax,
    setSectionThresholdMax, sectionResult, setSectionResult, sectionLoading, setSectionLoading,
    volumeThreshold, setVolumeThreshold, volumeThresholdMax, setVolumeThresholdMax, volumeData,
    setVolumeData, boxFacesData, setBoxFacesData, boxTopLayerIndex, setBoxTopLayerIndex, boxFacesLoading,
    setBoxFacesLoading, lastCustomSectionPath, setLastCustomSectionPath, volumeSlices, setVolumeSlices,
    volumeOpacity, setVolumeOpacity, showGeologyUnits, setShowGeologyUnits, handleUploadDem, handleClearDem,
    handleRunInversion, handleShowInversionSlice, handleSectionPathDrawn, handleRunFixedSection,
    handleGeologyPolygonDrawn, handleUpdateGeologyUnit, handleDeleteGeologyUnit, handleOpenVolume,
    handleLoadBoxFaces, fetchVolumeSliceData, handleAddVolumeSlice, handleRemoveVolumeSlice,
    handleUpdateVolumeSlice, handleExportInversion, handleExportInversionCsv, handleImportInversion,
  };
}
