const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: options.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") {
        detail = data.detail;
      } else if (Array.isArray(data.detail)) {
        // FastAPI/Pydantic validation errors: [{loc, msg, type}, ...]
        detail = data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
      } else if (data.detail) {
        detail = JSON.stringify(data.detail);
      } else {
        detail = JSON.stringify(data);
      }
    } catch {
      // ignore, keep statusText
    }
    throw new Error(detail);
  }
  return res.json();
}

// fetch() has no upload-progress event, so file uploads that want a
// progress bar go through XMLHttpRequest instead - the rest of the app
// still uses the plain fetch-based request() above.
function uploadWithProgress(path, form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}${path}`);
    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total);
      };
    }
    xhr.onload = () => {
      let data;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        data = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(data);
      } else {
        let detail = xhr.statusText;
        if (data) {
          if (typeof data.detail === "string") detail = data.detail;
          else if (Array.isArray(data.detail)) detail = data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
          else if (data.detail) detail = JSON.stringify(data.detail);
        }
        reject(new Error(detail));
      }
    };
    xhr.onerror = () => reject(new Error("네트워크 오류로 업로드에 실패했습니다."));
    xhr.send(form);
  });
}

export function createProject() {
  return request("/projects", { method: "POST" });
}

export function uploadDrone(projectId, files, onProgress) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  return uploadWithProgress(`/projects/${projectId}/upload/drone`, form, onProgress);
}

export function uploadBase(projectId, files, onProgress) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  return uploadWithProgress(`/projects/${projectId}/upload/base`, form, onProgress);
}

export function uploadIaga2002(projectId, file) {
  const form = new FormData();
  form.append("file", file);
  return request(`/projects/${projectId}/base/iaga2002/upload`, { method: "POST", body: form });
}

export function fetchIntermagnet(projectId, req) {
  return request(`/projects/${projectId}/base/intermagnet/fetch`, { method: "POST", body: JSON.stringify(req) });
}

export function applyIntermagnet(projectId) {
  return request(`/projects/${projectId}/base/intermagnet/apply`, { method: "POST" });
}

export function fetchNearestIntermagnet(projectId, req) {
  return request(`/projects/${projectId}/base/intermagnet/nearest/fetch`, { method: "POST", body: JSON.stringify(req) });
}

export function applyNearestIntermagnet(projectId) {
  return request(`/projects/${projectId}/base/intermagnet/nearest/apply`, { method: "POST" });
}

export function getNearestIntermagnetComparison(projectId) {
  return request(`/projects/${projectId}/base/intermagnet/nearest/comparison`);
}

export function uploadHeadingCalibration(projectId, files, onProgress) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  return uploadWithProgress(`/projects/${projectId}/upload/heading_calibration`, form, onProgress);
}

export function processProject(projectId, params) {
  return request(`/projects/${projectId}/process`, { method: "POST", body: JSON.stringify(params) });
}

export function getSummary(projectId) {
  return request(`/projects/${projectId}/summary`);
}

export function getPoints(projectId, value) {
  return request(`/projects/${projectId}/points?value=${value}`);
}

export function manualExclude(projectId, req) {
  return request(`/projects/${projectId}/manual-exclude`, { method: "POST", body: JSON.stringify(req) });
}

export function getGrid(projectId, req) {
  return request(`/projects/${projectId}/grid`, { method: "POST", body: JSON.stringify(req) });
}

export function getTransform(projectId, req) {
  return request(`/projects/${projectId}/transform`, { method: "POST", body: JSON.stringify(req) });
}

export function runEulerDeconvolution(projectId, req) {
  return request(`/projects/${projectId}/euler-deconvolution`, { method: "POST", body: JSON.stringify(req) });
}

export function runTargetDetection(projectId, req) {
  return request(`/projects/${projectId}/target-detection`, { method: "POST", body: JSON.stringify(req) });
}

export function getGridConfidence(projectId, req) {
  return request(`/projects/${projectId}/grid/confidence`, { method: "POST", body: JSON.stringify(req) });
}

export function runQcCertificate(projectId, req) {
  return request(`/projects/${projectId}/qc-certificate`, { method: "POST", body: JSON.stringify(req) });
}

export function runMultiscaleEdges(projectId, req) {
  return request(`/projects/${projectId}/multiscale-edges`, { method: "POST", body: JSON.stringify(req) });
}

export function runLineamentExtraction(projectId, req) {
  return request(`/projects/${projectId}/lineaments`, { method: "POST", body: JSON.stringify(req) });
}

export function runTiltDepth(projectId, req) {
  return request(`/projects/${projectId}/depth-estimation/tilt`, { method: "POST", body: JSON.stringify(req) });
}

export function runAnalyticSignalDepth(projectId, req) {
  return request(`/projects/${projectId}/depth-estimation/analytic-signal`, { method: "POST", body: JSON.stringify(req) });
}

export function runSpectralDepth(projectId, req) {
  return request(`/projects/${projectId}/depth-estimation/spectral`, { method: "POST", body: JSON.stringify(req) });
}

export function runContactDetection(projectId, req) {
  return request(`/projects/${projectId}/contacts`, { method: "POST", body: JSON.stringify(req) });
}

export function runProspectivity(projectId, req) {
  return request(`/projects/${projectId}/prospectivity`, { method: "POST", body: JSON.stringify(req) });
}

export function getProspectivityOverlay(projectId, colormap) {
  const query = colormap ? `?colormap=${encodeURIComponent(colormap)}` : "";
  return request(`/projects/${projectId}/prospectivity/overlay${query}`);
}

export function getPowerSpectrum(projectId, req) {
  return request(`/projects/${projectId}/spectrum`, { method: "POST", body: JSON.stringify(req) });
}

export function uploadRepeatability(projectId, files, onProgress) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  return uploadWithProgress(`/projects/${projectId}/upload/repeatability`, form, onProgress);
}

export function analyzeRepeatability(projectId) {
  return request(`/projects/${projectId}/repeatability/analyze`, { method: "POST" });
}

export function getLineProfile(projectId, lineId, value) {
  return request(`/projects/${projectId}/line-profile?line_id=${lineId}&value=${value}`);
}

export function getBaseTimeseries(projectId) {
  return request(`/projects/${projectId}/base/timeseries`);
}

export function applySmoothing(projectId, req) {
  return request(`/projects/${projectId}/smooth`, { method: "POST", body: JSON.stringify(req) });
}

export function scanStructureDistortion(projectId, req) {
  return request(`/projects/${projectId}/structure-scan`, { method: "POST", body: JSON.stringify(req) });
}

export function setDisplayBoundary(projectId, polygon) {
  return request(`/projects/${projectId}/display-boundary`, { method: "POST", body: JSON.stringify({ polygon }) });
}

export function estimateOfflineTiles(req) {
  return request(`/tiles/estimate`, { method: "POST", body: JSON.stringify(req) });
}

export function downloadOfflineTiles(req) {
  return request(`/tiles/download`, { method: "POST", body: JSON.stringify(req) });
}

export function getOfflineTileStatus() {
  return request(`/tiles/status`);
}

export function registerLocalTileFolder(req) {
  return request(`/local-tiles/register`, { method: "POST", body: JSON.stringify(req) });
}

export function unregisterLocalTileFolder(layerId) {
  return request(`/local-tiles/${layerId}`, { method: "DELETE" });
}

export function pickLocalTileFolder() {
  return request(`/local-tiles/pick-folder`, { method: "POST" });
}

export function uploadOverlayImage(file) {
  const form = new FormData();
  form.append("file", file);
  return request(`/overlay-images`, { method: "POST", body: form });
}

// Project-scoped copy of an uploaded reference layer (e.g. geology map) so
// the chat assistant's sample_point tool can read real pixel values at a
// point - separate from uploadOverlayImage above, which only produces a
// display image for the map and is not tied to a project.
export function uploadReferenceLayer(projectId, file) {
  const form = new FormData();
  form.append("file", file);
  return request(`/projects/${projectId}/reference-layers`, { method: "POST", body: form });
}

export function deleteReferenceLayer(projectId, name) {
  return request(`/projects/${projectId}/reference-layers/${encodeURIComponent(name)}`, { method: "DELETE" });
}

export function sendChatMessage(projectId, message, history) {
  return request(`/projects/${projectId}/chat`, { method: "POST", body: JSON.stringify({ message, history }) });
}

export function sampleOverlayValue(projectId, lat, lon) {
  return request(`/projects/${projectId}/overlay/sample`, { method: "POST", body: JSON.stringify({ lat, lon }) });
}

export function uploadDem(projectId, file) {
  const form = new FormData();
  form.append("file", file);
  return request(`/projects/${projectId}/upload/dem`, { method: "POST", body: form });
}

export function clearDem(projectId) {
  return request(`/projects/${projectId}/dem`, { method: "DELETE" });
}

export function runInversion(projectId, params) {
  return request(`/projects/${projectId}/inversion`, { method: "POST", body: JSON.stringify(params) });
}

export function getInversionSlice(projectId, req) {
  return request(`/projects/${projectId}/inversion/slice`, { method: "POST", body: JSON.stringify(req) });
}

export function getInversionSection(projectId, req) {
  return request(`/projects/${projectId}/inversion/section`, { method: "POST", body: JSON.stringify(req) });
}

export function getInversionVolume(projectId, threshold, thresholdMax) {
  const params = new URLSearchParams();
  if (threshold != null) params.set("threshold", threshold);
  if (thresholdMax != null) params.set("threshold_max", thresholdMax);
  const q = params.toString() ? `?${params.toString()}` : "";
  return request(`/projects/${projectId}/inversion/volume${q}`);
}

export function getInversionBoxFaces(projectId, topLayerIndex) {
  const params = new URLSearchParams();
  if (topLayerIndex != null) params.set("top_layer_index", topLayerIndex);
  const q = params.toString() ? `?${params.toString()}` : "";
  return request(`/projects/${projectId}/inversion/box_faces${q}`);
}

async function requestBlob(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: options.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? data);
    } catch {
      // ignore, keep statusText
    }
    throw new Error(detail);
  }
  return res.blob();
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// Purely client-side (no backend round-trip needed - the data is already
// in frontend state) - used for the display-boundary polygon so it can be
// saved to a file and re-loaded later or reused across projects.
export function downloadJson(data, filename) {
  downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }), filename);
}

export async function exportGridGeotiff(projectId, req, filename) {
  const blob = await requestBlob(`/projects/${projectId}/grid/geotiff`, { method: "POST", body: JSON.stringify(req) });
  downloadBlob(blob, filename);
}

export async function exportTransformGeotiff(projectId, req, filename) {
  const blob = await requestBlob(`/projects/${projectId}/transform/geotiff`, { method: "POST", body: JSON.stringify(req) });
  downloadBlob(blob, filename);
}

export async function exportGridXyz(projectId, req, filename) {
  const blob = await requestBlob(`/projects/${projectId}/grid/xyz`, { method: "POST", body: JSON.stringify(req) });
  downloadBlob(blob, filename);
}

export async function exportTransformXyz(projectId, req, filename) {
  const blob = await requestBlob(`/projects/${projectId}/transform/xyz`, { method: "POST", body: JSON.stringify(req) });
  downloadBlob(blob, filename);
}

export async function exportPointsCsv(projectId, filename) {
  const blob = await requestBlob(`/projects/${projectId}/points/csv`);
  downloadBlob(blob, filename);
}

export async function exportTargetsCsv(projectId, filename) {
  const blob = await requestBlob(`/projects/${projectId}/target-detection/csv`);
  downloadBlob(blob, filename);
}

export async function exportTargetsShapefile(projectId, filename) {
  const blob = await requestBlob(`/projects/${projectId}/target-detection/shapefile`);
  downloadBlob(blob, filename);
}

export async function exportGridGrd(projectId, req, filename) {
  const blob = await requestBlob(`/projects/${projectId}/grid/grd`, { method: "POST", body: JSON.stringify(req) });
  downloadBlob(blob, filename);
}

export async function exportTransformGrd(projectId, req, filename) {
  const blob = await requestBlob(`/projects/${projectId}/transform/grd`, { method: "POST", body: JSON.stringify(req) });
  downloadBlob(blob, filename);
}

export async function exportPolygonBln(projectId, polygon, filename) {
  const blob = await requestBlob(`/projects/${projectId}/export/bln`, { method: "POST", body: JSON.stringify({ polygon }) });
  downloadBlob(blob, filename);
}

export async function exportInversionSliceGeotiff(projectId, req, filename) {
  const blob = await requestBlob(`/projects/${projectId}/inversion/slice/geotiff`, { method: "POST", body: JSON.stringify(req) });
  downloadBlob(blob, filename);
}

export async function exportInversionResult(projectId, filename) {
  const blob = await requestBlob(`/projects/${projectId}/inversion/export`);
  downloadBlob(blob, filename);
}

export async function exportNearestIntermagnetCsv(projectId, filename) {
  const blob = await requestBlob(`/projects/${projectId}/base/intermagnet/nearest/csv`);
  downloadBlob(blob, filename);
}

export async function exportInversionCsv(projectId, filename) {
  const blob = await requestBlob(`/projects/${projectId}/inversion/export/csv`);
  downloadBlob(blob, filename);
}

export function importInversionResult(projectId, file) {
  const form = new FormData();
  form.append("file", file);
  return request(`/projects/${projectId}/inversion/import`, { method: "POST", body: form });
}

export async function saveProject(projectId, filename) {
  const blob = await requestBlob(`/projects/${projectId}/save`);
  downloadBlob(blob, filename);
}

export async function exportReport(projectId, filename) {
  const blob = await requestBlob(`/projects/${projectId}/report`);
  downloadBlob(blob, filename);
}

export function loadProject(projectId, file) {
  const form = new FormData();
  form.append("file", file);
  return request(`/projects/${projectId}/load`, { method: "POST", body: form });
}

// Which commit the connected backend is actually running - lets the UI
// show this directly, since "이미 고친 버그가 재현된다" reports have
// repeatedly turned out to be a stale run.bat build rather than a real
// regression (see main.py::_detect_running_version).
export function getVersion() {
  return request("/version");
}
