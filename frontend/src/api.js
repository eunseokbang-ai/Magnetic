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

export function createProject() {
  return request("/projects", { method: "POST" });
}

export function uploadDrone(projectId, files) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  return request(`/projects/${projectId}/upload/drone`, { method: "POST", body: form });
}

export function uploadBase(projectId, files) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  return request(`/projects/${projectId}/upload/base`, { method: "POST", body: form });
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

export function uploadOverlayImage(file) {
  const form = new FormData();
  form.append("file", file);
  return request(`/overlay-images`, { method: "POST", body: form });
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

export function getInversionVolume(projectId, threshold) {
  const q = threshold != null ? `?threshold=${threshold}` : "";
  return request(`/projects/${projectId}/inversion/volume${q}`);
}
