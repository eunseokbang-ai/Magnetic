import { useState } from "react";
import * as api from "../api";
import { DEFAULT_AS_DEPTH_PARAMS, DEFAULT_CONTACT_PARAMS, DEFAULT_LINEAMENT_PARAMS, DEFAULT_PROSPECTIVITY_PARAMS, DEFAULT_QC_CERTIFICATE_PARAMS, DEFAULT_TARGET_DETECTION_PARAMS, DEFAULT_TILT_DEPTH_PARAMS } from "../appDefaults";

// 해석 도구: 오일러, 반복측선·스펙트럼·멀티스케일 엣지, 표적탐지, QC 인증서, 선구조, 심도 추정, 접촉면, 유망도, 지질 통계.
//
// One of the hooks App() is composed of (split out of App.jsx in 2026-11;
// the statements are unchanged). It receives what the hooks called before
// it returned and returns everything it declares.
export default function useInterpretation(app) {
  const {
    ensureProject, gridCellSize, gridMaxDistance, gridMethod, projectId, setProcessParams, valueField,
  } = app;

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
  const [geologyStatsParams, setGeologyStatsParams] = useState({ scale: "50k", value: "anomaly", cell_size_m: 10.0, use_cache: true });
  const [geologyStatsResult, setGeologyStatsResult] = useState(null);
  const [geologyStatsRunning, setGeologyStatsRunning] = useState(false);
  const [geologyStatsError, setGeologyStatsError] = useState(null);
  const [prospectivityError, setProspectivityError] = useState(null);
  const [showProspectivityTargets, setShowProspectivityTargets] = useState(true);
  const [showProspectivityOverlay, setShowProspectivityOverlay] = useState(false);
  const [prospectivityOverlay, setProspectivityOverlay] = useState(null);

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
      const resp = await api.runTiltDepth(projectId, { value: "anomaly", cell_size_m: depthEstimationCellSize, method: gridMethod, ...tiltDepthParams });
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
      const resp = await api.runAnalyticSignalDepth(projectId, { value: "anomaly", cell_size_m: depthEstimationCellSize, method: gridMethod, ...asDepthParams });
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
      const resp = await api.runSpectralDepth(projectId, { value: "anomaly", cell_size_m: depthEstimationCellSize, method: gridMethod });
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

  const handleRunGeologyStats = async () => {
    try {
      setGeologyStatsError(null);
      setGeologyStatsRunning(true);
      setGeologyStatsResult(await api.runGeologyStats(projectId, geologyStatsParams));
    } catch (e) {
      setGeologyStatsError(e.message || String(e));
    } finally {
      setGeologyStatsRunning(false);
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

  return {
    eulerStructuralIndex, setEulerStructuralIndex, eulerWindowSize, setEulerWindowSize,
    eulerMaxUncertaintyPct, setEulerMaxUncertaintyPct, eulerFlightAglM, setEulerFlightAglM, eulerRunning,
    setEulerRunning, eulerResult, setEulerResult, eulerError, setEulerError, showEulerSolutions,
    setShowEulerSolutions, repeatabilityUploading, setRepeatabilityUploading, repeatabilityUploadInfo,
    setRepeatabilityUploadInfo, repeatabilityAnalyzing, setRepeatabilityAnalyzing, repeatabilityResult,
    setRepeatabilityResult, repeatabilityError, setRepeatabilityError, spectrumLineId, setSpectrumLineId,
    spectrumValue, setSpectrumValue, spectrumLoading, setSpectrumLoading, spectrumResult, setSpectrumResult,
    spectrumError, setSpectrumError, multiscaleHeightsText, setMultiscaleHeightsText, multiscalePercentile,
    setMultiscalePercentile, multiscaleRunning, setMultiscaleRunning, multiscaleResult, setMultiscaleResult,
    multiscaleError, setMultiscaleError, showMultiscaleLayer, setShowMultiscaleLayer, guidelinePanelOpened,
    setGuidelinePanelOpened, targetDetectionParams, setTargetDetectionParams, targetDetectionRunning,
    setTargetDetectionRunning, targetDetectionResult, setTargetDetectionResult, targetDetectionError,
    setTargetDetectionError, showDetectedTargets, setShowDetectedTargets, qcCertificateParams,
    setQcCertificateParams, qcCertificateRunning, setQcCertificateRunning, qcCertificateResult,
    setQcCertificateResult, qcCertificateError, setQcCertificateError, lineamentParams, setLineamentParams,
    lineamentRunning, setLineamentRunning, lineamentResult, setLineamentResult, lineamentError,
    setLineamentError, showLineaments, setShowLineaments, depthEstimationCellSize,
    setDepthEstimationCellSize, tiltDepthParams, setTiltDepthParams, tiltDepthRunning, setTiltDepthRunning,
    tiltDepthResult, setTiltDepthResult, tiltDepthError, setTiltDepthError, showTiltDepth, setShowTiltDepth,
    asDepthParams, setAsDepthParams, asDepthRunning, setAsDepthRunning, asDepthResult, setAsDepthResult,
    asDepthError, setAsDepthError, showAsDepth, setShowAsDepth, spectralDepthRunning,
    setSpectralDepthRunning, spectralDepthResult, setSpectralDepthResult, spectralDepthError,
    setSpectralDepthError, contactParams, setContactParams, contactRunning, setContactRunning, contactResult,
    setContactResult, contactError, setContactError, showContacts, setShowContacts, prospectivityParams,
    setProspectivityParams, prospectivityRunning, setProspectivityRunning, prospectivityResult,
    setProspectivityResult, geologyStatsParams, setGeologyStatsParams, geologyStatsResult,
    setGeologyStatsResult, geologyStatsRunning, setGeologyStatsRunning, geologyStatsError,
    setGeologyStatsError, prospectivityError, setProspectivityError, showProspectivityTargets,
    setShowProspectivityTargets, showProspectivityOverlay, setShowProspectivityOverlay, prospectivityOverlay,
    setProspectivityOverlay, handleRunEuler, handleUploadRepeatability, handleAnalyzeRepeatability,
    handleViewSpectrum, handleAddNotchFrequency, handleRemoveNotchFrequency, handleRunMultiscaleEdges,
    handleRunTargetDetection, handleRunQcCertificate, handleRunLineamentExtraction, handleRunTiltDepth,
    handleRunAsDepth, handleRunSpectralDepth, handleRunContactDetection, handleRunGeologyStats,
    handleRunProspectivity,
  };
}
