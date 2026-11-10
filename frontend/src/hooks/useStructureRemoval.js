import { useState } from "react";
import * as api from "../api";
import { circlePolygon } from "../geoMeasure";
import { DEFAULT_ANOMALY_CANDIDATE_PARAMS, DEFAULT_STRUCTURE_SCAN_PARAMS } from "../appDefaults";

// 지상 구조물 처리: 구조물 왜곡 탐지, 이상 후보 목록과 판정, 구조물 이상 제거와 그 이력.
//
// One of the hooks App() is composed of (split out of App.jsx in 2026-11;
// the statements are unchanged). It receives what the hooks called before
// it returned and returns everything it declares.
export default function useStructureRemoval(app) {
  const {
    _afterSmoothingApplied, handleError, projectId, setError, setSmoothHistory,
  } = app;

  const [structureScanParams, setStructureScanParams] = useState(DEFAULT_STRUCTURE_SCAN_PARAMS);
  const [structureScanRunning, setStructureScanRunning] = useState(false);
  const [structureScanResult, setStructureScanResult] = useState(null);
  const [structureScanError, setStructureScanError] = useState(null);
  const [structureScanShow, setStructureScanShow] = useState(true);
  const [structureScanApplying, setStructureScanApplying] = useState(false);
  const [candidateParams, setCandidateParams] = useState(DEFAULT_ANOMALY_CANDIDATE_PARAMS);
  const [candidateRunning, setCandidateRunning] = useState(false);
  const [candidateResult, setCandidateResult] = useState(null);
  const [candidateError, setCandidateError] = useState(null);
  const [candidateShow, setCandidateShow] = useState(true);
  const [candidateApplying, setCandidateApplying] = useState(false);
  const [selectedCandidateIndices, setSelectedCandidateIndices] = useState(new Set());
  // How a marked ground structure is removed: "model" fits its field and
  // subtracts it (processing/source_removal.py); "interpolate" cuts the
  // region out and fills it along each line (processing/manual_smooth.py).
  const [removalMethod, setRemovalMethod] = useState("model");
  const [removalWorking, setRemovalWorking] = useState(false);
  const [lastRemovalAdded, setLastRemovalAdded] = useState(null);
  const [selectedStructurePolygonIndices, setSelectedStructurePolygonIndices] = useState(new Set());
  const [selectedStructureAnomalyIndices, setSelectedStructureAnomalyIndices] = useState(new Set());

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

  const handleRunAnomalyCandidates = async () => {
    try {
      setCandidateError(null);
      setCandidateRunning(true);
      const resp = await api.scanAnomalyCandidates(projectId, candidateParams);
      setCandidateResult(resp);
      // nothing pre-selected: which of these are structures and which are
      // geology is the operator's call, and the panel says so
      setSelectedCandidateIndices(new Set());
    } catch (e) {
      setCandidateResult(null);
      setCandidateError(e.message || String(e));
    } finally {
      setCandidateRunning(false);
    }
  };

  // Records what the operator decided about one candidate and refreshes
  // the list in place, so the decision is visible on the row it was made
  // on rather than only after the next scan.
  const handleSetCandidateNote = async (candidate, verdict, note) => {
    try {
      setError(null);
      const resp = await api.setCandidateNote(projectId, {
        lat: candidate.lat,
        lon: candidate.lon,
        verdict,
        note,
        peak_anomaly_nt: candidate.peak_anomaly_nt,
        depth_m: candidate.depth_m,
        radius_m: candidate.radius_m,
        magnetization: candidate.magnetization ? candidate.magnetization.verdict : null,
      });
      setCandidateResult((prev) =>
        prev
          ? {
              ...prev,
              candidates: prev.candidates.map((c) =>
                c.lat === candidate.lat && c.lon === candidate.lon ? { ...c, note: resp.note } : c
              ),
            }
          : prev
      );
    } catch (e) {
      handleError(e);
    }
  };

  const handleToggleCandidate = (i) => {
    setSelectedCandidateIndices((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  };

  const _removeStructuresByModel = async (polygons, depthHints) => {
    const summary = await api.applySourceRemoval(projectId, {
      mode: "add",
      polygons,
      depth_hints_m: depthHints,
    });
    setLastRemovalAdded(summary.added || null);
    await _afterSmoothingApplied(summary);
    return summary;
  };

  const handleSourceRemovalHistory = async (mode) => {
    try {
      setError(null);
      setRemovalWorking(true);
      const summary = await api.applySourceRemoval(projectId, { mode });
      setLastRemovalAdded(null);
      await _afterSmoothingApplied(summary);
    } catch (e) {
      handleError(e);
    } finally {
      setRemovalWorking(false);
    }
  };

  const handleApplyCandidateSmoothing = async () => {
    if (!candidateResult) return;
    try {
      setError(null);
      setCandidateApplying(true);
      const polygons = [];
      const depthHints = [];
      const applied = [];
      for (const i of selectedCandidateIndices) {
        const c = candidateResult.candidates[i];
        applied.push(c);
        // a model needs room for the whole source, a cut only the reach
        // of its field - see AnomalyCandidate.source_polygon_xy
        const ring = removalMethod === "model" ? c?.source_polygon : c?.polygon;
        if (ring?.length >= 3) {
          polygons.push(ring);
          depthHints.push(c.depth_m ?? null);
        }
      }
      if (polygons.length === 0) return;
      if (removalMethod === "model") {
        await _removeStructuresByModel(polygons, depthHints);
      } else {
        const summary = await api.applySmoothing(projectId, { mode: "polygons", polygons });
        if (Array.isArray(summary.manual_smooth_point_ids)) {
          setSmoothHistory((prev) => [...prev, summary.manual_smooth_point_ids]);
        }
        await _afterSmoothingApplied(summary);
      }
      // Removing a candidate is itself a decision about it, so it goes
      // into the log without the operator having to record it twice. An
      // existing verdict is kept - only the method is added.
      for (const c of applied) {
        try {
          await api.setCandidateNote(projectId, {
            lat: c.lat,
            lon: c.lon,
            verdict: c.note?.verdict && c.note.verdict !== "clear" ? c.note.verdict : "structure",
            note: c.note?.note || null,
            peak_anomaly_nt: c.peak_anomaly_nt,
            depth_m: c.depth_m,
            radius_m: c.radius_m,
            magnetization: c.magnetization ? c.magnetization.verdict : null,
            removed_method: removalMethod,
          });
        } catch {
          // the removal itself succeeded; a failed log entry must not undo it
        }
      }
      // the field has changed underneath them, so the old ranking no
      // longer describes it - rerun to see what is left
      setSelectedCandidateIndices(new Set());
      await handleRunAnomalyCandidates();
    } catch (e) {
      handleError(e);
    } finally {
      setCandidateApplying(false);
    }
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

  return {
    structureScanParams, setStructureScanParams, structureScanRunning, setStructureScanRunning,
    structureScanResult, setStructureScanResult, structureScanError, setStructureScanError,
    structureScanShow, setStructureScanShow, structureScanApplying, setStructureScanApplying,
    candidateParams, setCandidateParams, candidateRunning, setCandidateRunning, candidateResult,
    setCandidateResult, candidateError, setCandidateError, candidateShow, setCandidateShow,
    candidateApplying, setCandidateApplying, selectedCandidateIndices, setSelectedCandidateIndices,
    removalMethod, setRemovalMethod, removalWorking, setRemovalWorking, lastRemovalAdded,
    setLastRemovalAdded, selectedStructurePolygonIndices, setSelectedStructurePolygonIndices,
    selectedStructureAnomalyIndices, setSelectedStructureAnomalyIndices, handleRunStructureScan,
    handleToggleStructurePolygon, handleToggleStructureAnomaly, handleRunAnomalyCandidates,
    handleSetCandidateNote, handleToggleCandidate, _removeStructuresByModel, handleSourceRemovalHistory,
    handleApplyCandidateSmoothing, handleApplyStructureSmoothing,
  };
}
