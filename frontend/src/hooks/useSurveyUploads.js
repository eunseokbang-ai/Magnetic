import { useState } from "react";
import * as api from "../api";

// 자료 올리기: 드론·베이스 파일, INTERMAGNET 관측소, 헤딩 캘리브레이션 비행, 베이스 기록 보기.
//
// One of the hooks App() is composed of (split out of App.jsx in 2026-11;
// the statements are unchanged). It receives what the hooks called before
// it returned and returns everything it declares.
export default function useSurveyUploads(app) {
  const {
    ensureProject, handleError, projectId, setBaseSummary, setDroneSummary, setError,
  } = app;

  const [headingCalibrationSummary, setHeadingCalibrationSummary] = useState(null);
  const [headingCalibrationUploadProgress, setHeadingCalibrationUploadProgress] = useState(null);
  const [nearestIntermagnetCsvFilename, setNearestIntermagnetCsvFilename] = useState("intermagnet_nearest_estimate.csv");
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
  const [baseTimeseries, setBaseTimeseries] = useState(null);
  const [baseTimeseriesLoading, setBaseTimeseriesLoading] = useState(false);

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

  return {
    headingCalibrationSummary, setHeadingCalibrationSummary, headingCalibrationUploadProgress,
    setHeadingCalibrationUploadProgress, nearestIntermagnetCsvFilename, setNearestIntermagnetCsvFilename,
    droneUploadProgress, setDroneUploadProgress, baseUploadProgress, setBaseUploadProgress,
    intermagnetPreview, setIntermagnetPreview, intermagnetLoading, setIntermagnetLoading,
    intermagnetApplying, setIntermagnetApplying, intermagnetError, setIntermagnetError,
    nearestIntermagnetPreview, setNearestIntermagnetPreview, nearestIntermagnetLoading,
    setNearestIntermagnetLoading, nearestIntermagnetApplying, setNearestIntermagnetApplying,
    nearestIntermagnetError, setNearestIntermagnetError, nearestIntermagnetHasResult,
    setNearestIntermagnetHasResult, nearestIntermagnetComparison, setNearestIntermagnetComparison,
    nearestIntermagnetComparisonLoading, setNearestIntermagnetComparisonLoading, baseTimeseries,
    setBaseTimeseries, baseTimeseriesLoading, setBaseTimeseriesLoading, handleUploadDrone, handleUploadBase,
    handleUploadIaga2002, handleFetchIntermagnet, handleApplyIntermagnet, handleCancelIntermagnetPreview,
    handleFetchNearestIntermagnet, handleApplyNearestIntermagnet, handleCancelNearestIntermagnetPreview,
    handleExportNearestIntermagnetCsv, handleShowNearestIntermagnetComparison,
    handleUploadHeadingCalibration,
  };
}
