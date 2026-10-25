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
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? data);
    } catch {
      // keep statusText
    }
    throw new Error(detail);
  }
  return res.json();
}

const post = (path, body) => request(path, { method: "POST", body: JSON.stringify(body || {}) });

export const getVersion = () => request("/version");
export const createProject = () => request("/projects", { method: "POST" });

export function uploadFiles(projectId, files) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  return request(`/projects/${projectId}/upload`, { method: "POST", body: form });
}

export const getSummary = (projectId) => request(`/projects/${projectId}/summary`);
export const getPoints = (projectId, sensor, maxPoints = 8000, processed = false) =>
  request(`/projects/${projectId}/points?max_points=${maxPoints}${sensor ? `&sensor=${sensor}` : ""}${processed ? "&processed=true" : ""}`);

export const runQc = (projectId, req) => post(`/projects/${projectId}/qc`, req);
export const runPreprocess = (projectId, req) => post(`/projects/${projectId}/preprocess`, req);
export const resetPreprocess = (projectId) => request(`/projects/${projectId}/preprocess`, { method: "DELETE" });
export const makeGrid = (projectId, req) => post(`/projects/${projectId}/grid`, req);
export const compareRepeats = (projectId, req) => post(`/projects/${projectId}/repeat`, req);
export const pickTargets = (projectId, req) => post(`/projects/${projectId}/targets`, req);
export const setArea = (projectId, polygon) => request(`/projects/${projectId}/area`, { method: "PUT", body: JSON.stringify({ polygon }) });
export const clearArea = (projectId) => request(`/projects/${projectId}/area`, { method: "DELETE" });
export const getArea = (projectId) => request(`/projects/${projectId}/area`);
export const sampleAt = (projectId, lat, lon) => post(`/projects/${projectId}/sample`, { lat, lon });
export const lineProfile = (projectId, fileIndex, lineId, sensor) =>
  request(`/projects/${projectId}/line-profile?file_index=${fileIndex}&line_id=${lineId}${sensor ? `&sensor=${sensor}` : ""}`);
export const targetsCsvUrl = (projectId) => `${BASE}/projects/${projectId}/targets.csv`;
export const targetsGeojsonUrl = (projectId) => `${BASE}/projects/${projectId}/targets.geojson`;

export function uploadBasemap(projectId, file) {
  const form = new FormData();
  form.append("file", file);
  return request(`/projects/${projectId}/basemaps`, { method: "POST", body: form });
}
export const listBasemaps = (projectId) => request(`/projects/${projectId}/basemaps`);
export const getBasemap = (projectId, name) => request(`/projects/${projectId}/basemaps/${encodeURIComponent(name)}`);
export const deleteBasemap = (projectId, name) => request(`/projects/${projectId}/basemaps/${encodeURIComponent(name)}`, { method: "DELETE" });
export const registerLocalTiles = (req) => post("/local-tiles/register", req);
export const pickLocalTileFolder = () => request("/local-tiles/pick-folder", { method: "POST" });
export const unregisterLocalTiles = (id) => request(`/local-tiles/${id}`, { method: "DELETE" });
export const localTileUrl = (id) => `${BASE}/local-tiles/${id}/{z}/{x}/{y}`;

/** Grid exports are POSTs (the grid parameters travel in the body); the
 * browser is handed the file as a download. */
export async function downloadGrid(projectId, req, fmt) {
  const res = await fetch(`${BASE}/projects/${projectId}/grid/export?fmt=${fmt}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req || {}),
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `minemag_${req.kind || "anomaly"}.${fmt}`;
  a.click();
  URL.revokeObjectURL(url);
}
