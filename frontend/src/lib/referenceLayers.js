// The layer manager's rows (12. 참조 레이어) as plain data: how a row is
// built from what the backend answers, how the list is reordered, and
// what of it is mirrored back to the project. No React here, so these
// are the parts of the reference-layer feature that have tests
// (referenceLayers.test.js).
//
// A row is { id, name, opacity, visible, type?, ... }. `type` is absent
// for an uploaded GeoTIFF (the original kind), and "vector", "tiles" or
// "raster" otherwise. `id` is this session's handle for the row; tile and
// raster layer ids come from the backend and are only valid for the
// server run that issued them.

export function newLayerId() {
  return `${Date.now()}-${Math.random()}`;
}

// A layer the user just uploaded (POST /overlay-images or /vector-layers).
export function uploadedLayer(resp) {
  if (resp.type === "vector") {
    return {
      id: newLayerId(), name: resp.name, type: "vector", features: resp.features, bounds: resp.bounds,
      n_points: resp.n_points, n_lines: resp.n_lines, n_polygons: resp.n_polygons, opacity: 1, visible: true,
    };
  }
  return { id: newLayerId(), name: resp.name, image_data_url: resp.image_data_url, bounds: resp.bounds, opacity: 0.8, visible: true };
}

// A folder of tiles on this machine (POST /local-tiles/register).
export function tileFolderLayer(resp) {
  return {
    id: newLayerId(), name: resp.label, type: "tiles", tile_layer_id: resp.id, bounds: resp.bounds,
    min_zoom: resp.min_zoom, max_zoom: resp.max_zoom, path: resp.path, scheme: resp.scheme, opacity: 1, visible: true,
  };
}

// A large GeoTIFF on this machine, served as tiles (POST /local-rasters/register).
export function rasterLayer(resp) {
  return {
    id: newLayerId(), name: resp.label, type: "raster", raster_layer_id: resp.id, bounds: resp.bounds,
    min_zoom: resp.min_zoom, max_zoom: resp.max_zoom, has_overviews: resp.has_overviews, size_gb: resp.size_gb,
    res_m: resp.res_m, path: resp.path, opacity: 1, visible: true,
  };
}

// A saved layer as the backend hands it back on load / reopen -> a row.
export function layerFromDescriptor(d) {
  const base = { id: newLayerId(), name: d.name, opacity: d.opacity ?? 1, visible: d.visible ?? true };
  if (d.type === "vector") {
    return { ...base, type: "vector", features: d.features, bounds: d.bounds, n_points: d.n_points, n_lines: d.n_lines, n_polygons: d.n_polygons };
  }
  if (d.type === "tiles") {
    return { ...base, type: "tiles", tile_layer_id: d.tile_layer_id, bounds: d.bounds, min_zoom: d.min_zoom, max_zoom: d.max_zoom, path: d.path, scheme: d.scheme };
  }
  if (d.type === "raster") {
    return {
      ...base, type: "raster", raster_layer_id: d.raster_layer_id, bounds: d.bounds, min_zoom: d.min_zoom, max_zoom: d.max_zoom,
      has_overviews: d.has_overviews, size_gb: d.size_gb, res_m: d.res_m, path: d.path,
    };
  }
  return { ...base, image_data_url: d.image_data_url, bounds: d.bounds };
}

// The saved layers of a loaded project: the rows to show, and a message
// naming the ones that could not be connected again (a moved folder, an
// unplugged disk) - those come back with `error` instead of content.
export function layersFromDescriptors(descriptors) {
  const lost = descriptors.filter((d) => d.error);
  return {
    layers: descriptors.filter((d) => !d.error).map(layerFromDescriptor),
    message: lost.length ? `다시 연결하지 못한 참조 레이어: ${lost.map((d) => `${d.name} — ${d.error}`).join(" / ")}` : null,
  };
}

// What the project keeps of the list (PUT .../reference-layers/state):
// order, opacity, visibility and, for path-based layers, the path. The
// content itself was sent when each layer was added.
export function layerSyncState(layers) {
  return layers.filter((l) => !l.generated).map((l) => ({ type: l.type || "image", name: l.name, opacity: l.opacity, visible: l.visible, path: l.path || null, scheme: l.scheme || null }));
}

// "up" moves a row toward the end of the list: the list is drawn bottom
// to top, so the last row is the top layer on the map.
export function moveLayer(layers, id, direction) {
  const idx = layers.findIndex((l) => l.id === id);
  if (idx === -1) return layers;
  const swapWith = direction === "up" ? idx + 1 : idx - 1;
  if (swapWith < 0 || swapWith >= layers.length) return layers;
  const next = [...layers];
  [next[idx], next[swapWith]] = [next[swapWith], next[idx]];
  return next;
}
