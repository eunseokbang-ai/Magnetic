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
export const getPoints = (projectId, sensor, maxPoints = 8000) =>
  request(`/projects/${projectId}/points?max_points=${maxPoints}${sensor ? `&sensor=${sensor}` : ""}`);

export const runQc = (projectId, req) => post(`/projects/${projectId}/qc`, req);
export const makeGrid = (projectId, req) => post(`/projects/${projectId}/grid`, req);
export const compareRepeats = (projectId, req) => post(`/projects/${projectId}/repeat`, req);
