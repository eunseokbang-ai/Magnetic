import { useState } from "react";
import * as api from "../api";

// State and actions of 16. 오프라인 지도: how many tiles a download would
// take, the download itself, and what is already cached. Moved out of
// App.jsx in 2026-11.
export default function useOfflineTiles() {
  const [tileEstimate, setTileEstimate] = useState(null);
  const [tileDownloadResult, setTileDownloadResult] = useState(null);
  const [tileStatus, setTileStatus] = useState(null);
  const [tileLoading, setTileLoading] = useState(false);
  const [tileError, setTileError] = useState(null);

  const handleEstimateTiles = async (req) => {
    try {
      setTileError(null);
      setTileDownloadResult(null);
      setTileEstimate(await api.estimateOfflineTiles(req));
    } catch (e) {
      setTileError(e.message || String(e));
    }
  };

  const handleDownloadTiles = async (req) => {
    try {
      setTileError(null);
      setTileLoading(true);
      setTileDownloadResult(await api.downloadOfflineTiles(req));
    } catch (e) {
      setTileError(e.message || String(e));
    } finally {
      setTileLoading(false);
      // also after a cancelled or failed download: the tiles fetched before it stopped are cached
      api.getOfflineTileStatus().then(setTileStatus).catch(() => {});
    }
  };

  const handleRefreshTileStatus = async () => {
    try {
      setTileStatus(await api.getOfflineTileStatus());
    } catch (e) {
      setTileError(e.message || String(e));
    }
  };

  return { tileEstimate, tileDownloadResult, tileStatus, tileLoading, tileError, handleEstimateTiles, handleDownloadTiles, handleRefreshTileStatus };
}
