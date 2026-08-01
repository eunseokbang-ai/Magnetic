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

export function getLineProfile(projectId, lineId, value) {
  return request(`/projects/${projectId}/line-profile?line_id=${lineId}&value=${value}`);
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

export async function exportInversionSliceGeotiff(projectId, req, filename) {
  const blob = await requestBlob(`/projects/${projectId}/inversion/slice/geotiff`, { method: "POST", body: JSON.stringify(req) });
  downloadBlob(blob, filename);
}

export async function exportInversionResult(projectId, filename) {
  const blob = await requestBlob(`/projects/${projectId}/inversion/export`);
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
