import { useCallback, useMemo, useState } from "react";
import * as api from "../api";
import { DEFAULT_PARAMS } from "../appDefaults";

// 처리 실행과 그 뒤의 편집: 처리 설정, 측점 제외, 측선 프로파일, 평활, 측선 편집기.
//
// One of the hooks App() is composed of (split out of App.jsx in 2026-11;
// the statements are unchanged). It receives what the hooks called before
// it returned and returns everything it declares.
export default function useProcessingEdits(app) {
  const {
    handleError, handleGrid, overlay, points, processSummary, projectId, refreshPoints, setActiveTransform,
    setBaseTimeseries, setError, setInversionSummary, setLastDrawnPolygon, setOverlay, setPoints,
    setProcessSummary, setSectionResult, setValueField, setVolumeData, valueField,
  } = app;

  const [processParams, setProcessParams] = useState(DEFAULT_PARAMS);
  const [processing, setProcessing] = useState(false);
  const [drawMode, setDrawMode] = useState(false);
  const [drawAction, setDrawAction] = useState("exclude");
  // 이착륙 램프 구간(이륙->측선시작, 측선종료->착륙)은 기본적으로 편집 화면에서
  // 숨기고 제외/복원 드로잉의 영향도 받지 않도록 보호한다 - 켜면 다시 보이고
  // 편집도 가능해진다 (backend: ManualExcludeRequest.include_ramp).
  const [showRampPoints, setShowRampPoints] = useState(false);
  const [flightPathEditorOpen, setFlightPathEditorOpen] = useState(false);
  const [lineProfileData, setLineProfileData] = useState(null);
  const [lineProfileLoadingId, setLineProfileLoadingId] = useState(null);
  const [smoothing, setSmoothing] = useState(false);
  const [smoothDrawMode, setSmoothDrawMode] = useState(false);
  // Undo history for manual smoothing: each entry is the FULL set of
  // smoothed point_ids right after that apply call (not just the delta),
  // so undo can restore an exact prior state via reset + a single
  // point_ids call, regardless of whether entries came from the map
  // polygon tool or the time-series drag-select.
  const [smoothHistory, setSmoothHistory] = useState([]);

  // manual-exclude responses carry just the (point_id -> excluded) delta,
  // not the whole point list (lat/lon/value/line_id/timestamp never
  // change from a manual edit) - patching it in place avoids re-fetching
  // and re-parsing the full point set, which gets slow at 100k+ points.
  const applyExclusionDelta = useCallback((exclusion) => {
    if (!exclusion) return;
    const excludedById = new Map(exclusion.point_id.map((id, i) => [id, exclusion.excluded[i]]));
    setPoints((prev) => prev.map((p) => (excludedById.has(p.point_id) ? { ...p, excluded: excludedById.get(p.point_id) } : p)));
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

  // 이착륙 램프 구간은 기본적으로 지도/측선 편집 화면에서 숨긴다 (표시 옵션을
  // 켜기 전까지는 제외/복원 드로잉의 대상도 되지 않도록 백엔드에서도 별도로
  // 보호함 - handleShapeDrawn의 include_ramp 참고).
  const editablePoints = useMemo(() => {
    if (showRampPoints) return points;
    return points.filter((p) => !(p.is_ramp && p.excluded));
  }, [points, showRampPoints]);

  return {
    processParams, setProcessParams, processing, setProcessing, drawMode, setDrawMode, drawAction,
    setDrawAction, showRampPoints, setShowRampPoints, flightPathEditorOpen, setFlightPathEditorOpen,
    lineProfileData, setLineProfileData, lineProfileLoadingId, setLineProfileLoadingId, smoothing,
    setSmoothing, smoothDrawMode, setSmoothDrawMode, smoothHistory, setSmoothHistory, applyExclusionDelta,
    handleProcess, handleSetValueField, handleToggleLines, handleShapeDrawn, handleResetManual,
    handleManualExcludePointIds, handleShowLineProfile, _afterSmoothingApplied, handleApplySmoothing,
    handleUndoSmoothing, handleResetAllSmoothing, editablePoints,
  };
}
