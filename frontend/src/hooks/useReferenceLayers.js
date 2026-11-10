import { useEffect, useState } from "react";
import * as api from "../api";
import { layerSyncState, layersFromDescriptors, moveLayer, rasterLayer, tileFolderLayer, uploadedLayer } from "../lib/referenceLayers";

// State and actions of the layer manager (12. 참조 레이어): uploaded
// GeoTIFFs and vector files, tile folders and large GeoTIFFs linked by
// path. Moved out of App.jsx in 2026-11; the row shapes and list logic
// are in lib/referenceLayers.js.
export default function useReferenceLayers({ projectId, ensureProject }) {
  const [overlayLayers, setOverlayLayers] = useState([]);
  const [overlayUploading, setOverlayUploading] = useState(false);
  const [overlayError, setOverlayError] = useState(null);
  const [tileFolderRegistering, setTileFolderRegistering] = useState(false);
  const [tileFolderError, setTileFolderError] = useState(null);

  // Mirror the layer list to the project (debounced) so 저장 and the
  // autosave carry it. Content was already sent when each layer was added;
  // this is only order / opacity / visibility / paths.
  useEffect(() => {
    if (!projectId) return undefined;
    const timer = setTimeout(() => {
      api.setReferenceLayerState(projectId, layerSyncState(overlayLayers)).catch(() => {});
    }, 500);
    return () => clearTimeout(timer);
  }, [overlayLayers, projectId]);

  // Reference layers saved with a project that was just loaded. Call it
  // before the first await of the load handler, so it lands in the same
  // render as the changed project id - the sync effect above must never
  // pair the new project with the old list. `descriptors` null = a bundle
  // from before layers were saved: leave the screen alone.
  const restoreReferenceLayers = (descriptors) => {
    if (!Array.isArray(descriptors)) return;
    const { layers, message } = layersFromDescriptors(descriptors);
    setOverlayLayers(layers);
    setOverlayError(message);
  };

  // A layer made by the program itself (a flight plan preview): shown like
  // an uploaded vector file, replacing an earlier one of the same kind so
  // re-planning does not pile lines on the map. `kind` is its "format".
  const showGeneratedLayer = (layer) =>
    setOverlayLayers((prev) => [...prev.filter((l) => !(l.generated && l.generated === layer.format)), { ...uploadedLayer(layer), generated: layer.format }]);

  const handleUploadOverlayImage = async (file) => {
    try {
      setOverlayError(null);
      setOverlayUploading(true);
      const id = await ensureProject();
      const [resp] = await Promise.all([
        api.isVectorLayerFile(file.name) ? api.uploadVectorLayer(file) : api.uploadOverlayImage(file),
        // Best-effort: also keep a project-scoped copy so the chat
        // assistant's sample_point tool can read real pixel values (or
        // the nearest drill hole) here. Must never block the map display.
        api.uploadReferenceLayer(id, file).catch(() => null),
      ]);
      setOverlayLayers((prev) => [...prev, uploadedLayer(resp)]);
    } catch (e) {
      setOverlayError(e.message || String(e));
    } finally {
      setOverlayUploading(false);
    }
  };

  const handleToggleOverlayVisible = (id, visible) => setOverlayLayers((prev) => prev.map((l) => (l.id === id ? { ...l, visible } : l)));

  const handleSetOverlayOpacity = (id, opacity) => setOverlayLayers((prev) => prev.map((l) => (l.id === id ? { ...l, opacity } : l)));

  const handleRemoveOverlay = (id) => {
    setOverlayLayers((prev) => {
      const layer = prev.find((l) => l.id === id);
      if (layer?.type === "tiles") {
        api.unregisterLocalTileFolder(layer.tile_layer_id).catch(() => {});
      } else if (layer?.type === "raster") {
        api.unregisterLocalRaster(layer.raster_layer_id).catch(() => {});
      } else if (layer && projectId) {
        api.deleteReferenceLayer(projectId, layer.name).catch(() => {});
      }
      return prev.filter((l) => l.id !== id);
    });
  };

  const handleMoveOverlay = (id, direction) => setOverlayLayers((prev) => moveLayer(prev, id, direction));

  // Both path-based kinds share one busy flag and one error line in the panel.
  const registerByPath = async (register, toLayer) => {
    try {
      setTileFolderError(null);
      setTileFolderRegistering(true);
      const resp = await register();
      setOverlayLayers((prev) => [...prev, toLayer(resp)]);
    } catch (e) {
      setTileFolderError(e.message || String(e));
    } finally {
      setTileFolderRegistering(false);
    }
  };

  const handleRegisterLocalTileFolder = (path, label, scheme) => registerByPath(() => api.registerLocalTileFolder({ path, label, scheme }), tileFolderLayer);

  const handleRegisterLocalRaster = (path, label) => registerByPath(() => api.registerLocalRaster({ path, label }), rasterLayer);

  const handlePickRasterFile = async () => (await api.pickLocalRasterFile()).path;

  const handlePickTileFolder = async () => (await api.pickLocalTileFolder()).path;

  return {
    overlayLayers,
    overlayUploading,
    overlayError,
    tileFolderRegistering,
    tileFolderError,
    restoreReferenceLayers,
    showGeneratedLayer,
    handleUploadOverlayImage,
    handleToggleOverlayVisible,
    handleSetOverlayOpacity,
    handleRemoveOverlay,
    handleMoveOverlay,
    handleRegisterLocalTileFolder,
    handleRegisterLocalRaster,
    handlePickRasterFile,
    handlePickTileFolder,
  };
}
