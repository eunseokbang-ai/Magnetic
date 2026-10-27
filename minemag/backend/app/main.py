"""MineMag API - landmine / UXO detection from MagDrone R3 surveys.

Same skeleton as DroneMag Studio's backend (CORS limited to this machine,
gzip, the built frontend served from one port), with its own, much
shorter pipeline: load the two-probe export, QC the platform noise, grid
it, compare repeated runs, pick and grade targets, export the list.
"""
from __future__ import annotations

import io
import pathlib
import subprocess

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from magcore import local_tiles, tile_cache
from magcore.colormaps import register_custom_colormaps
from magcore.local_tiles import LocalTileError
from magcore.overlay_image import OverlayImageError
from magcore.tile_cache import TileCacheError

from . import paths
from .models import AreaRequest, GridRequest, LocalTileFolderRequest, PreprocessRequest, QcRequest, RepeatRequest, SampleRequest, TargetRequest, TileBboxRequest
from .store import ProjectError, store

register_custom_colormaps()

app = FastAPI(title="MineMag API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)


def _running_version() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parent, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        commit = None
    return {"product": "MineMag", "commit": commit}


_VERSION = _running_version()


@app.exception_handler(ProjectError)
async def _project_error(_request: Request, exc: ProjectError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(OverlayImageError)
async def _overlay_error(_request: Request, exc: OverlayImageError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(TileCacheError)
async def _tile_cache_error(_request: Request, exc: TileCacheError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(LocalTileError)
async def _tile_error(_request: Request, exc: LocalTileError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def _missing_project(_request: Request, exc: KeyError):
    return JSONResponse(status_code=404, content={"detail": f"프로젝트를 찾을 수 없습니다: {exc.args[0] if exc.args else ''}"})


@app.get("/api/version")
def version():
    return _VERSION


@app.post("/api/projects")
def create_project():
    return {"project_id": store.create().id}


@app.post("/api/projects/{project_id}/upload")
async def upload(project_id: str, files: list[UploadFile] = File(...)):
    project = store.get(project_id)
    buffers = [io.BytesIO(await f.read()) for f in files]
    return project.load_files(buffers, [f.filename or "" for f in files])


@app.get("/api/projects/{project_id}/summary")
def summary(project_id: str):
    return store.get(project_id).summary()


@app.get("/api/projects/{project_id}/points")
def points(project_id: str, sensor: int | None = None, max_points: int = 20000, processed: bool = False):
    if max_points < 100 or max_points > 200000:
        raise HTTPException(status_code=422, detail="max_points는 100~200000 사이여야 합니다.")
    return store.get(project_id).points(sensor=sensor, max_points=max_points, processed=processed)


@app.post("/api/projects/{project_id}/preprocess")
def run_preprocess(project_id: str, req: PreprocessRequest):
    return store.get(project_id).run_preprocess(req)


@app.delete("/api/projects/{project_id}/preprocess")
def reset_preprocess(project_id: str):
    return store.get(project_id).reset_preprocess()


@app.post("/api/projects/{project_id}/qc")
def platform_qc(project_id: str, req: QcRequest):
    return store.get(project_id).qc(req)


@app.post("/api/projects/{project_id}/grid")
def grid(project_id: str, req: GridRequest):
    return store.get(project_id).grid(req)


@app.put("/api/projects/{project_id}/area")
def set_area(project_id: str, req: AreaRequest):
    return store.get(project_id).set_area(req)


@app.delete("/api/projects/{project_id}/area")
def clear_area(project_id: str):
    return store.get(project_id).set_area(None)


@app.get("/api/projects/{project_id}/area")
def get_area(project_id: str):
    return store.get(project_id).area_status()


@app.post("/api/projects/{project_id}/sample")
def sample(project_id: str, req: SampleRequest):
    return store.get(project_id).sample(req)


@app.get("/api/projects/{project_id}/line-profile")
def line_profile(project_id: str, file_index: int, line_id: int, sensor: int | None = None):
    return store.get(project_id).line_profile(file_index, line_id, sensor)


@app.post("/api/projects/{project_id}/grid/export")
def grid_export(project_id: str, req: GridRequest, fmt: str = "grd"):
    data = store.get(project_id).grid_export(req, fmt)
    media = "application/octet-stream" if fmt == "grd" else "image/tiff"
    return Response(content=data, media_type=media,
                    headers={"Content-Disposition": f"attachment; filename=minemag_{req.kind}.{fmt}"})


@app.post("/api/projects/{project_id}/repeat")
def repeat(project_id: str, req: RepeatRequest):
    return store.get(project_id).repeat(req)


@app.post("/api/projects/{project_id}/targets")
def targets(project_id: str, req: TargetRequest):
    return store.get(project_id).targets(req)


@app.get("/api/projects/{project_id}/targets.geojson")
def targets_geojson(project_id: str):
    return store.get(project_id).targets_geojson()


@app.get("/api/projects/{project_id}/targets.csv")
def targets_csv(project_id: str):
    text = store.get(project_id).targets_csv()
    return Response(
        content=text.encode("utf-8-sig"), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=minemag_targets.csv"},
    )


# ---------------------------------------------------------- basemaps
@app.post("/api/projects/{project_id}/basemaps")
async def upload_basemap(project_id: str, file: UploadFile = File(...)):
    content = await file.read()
    return store.get(project_id).add_basemap(file.filename or "orthophoto", content)


@app.get("/api/projects/{project_id}/basemaps")
def list_basemaps(project_id: str):
    return store.get(project_id).list_basemaps()


@app.get("/api/projects/{project_id}/basemaps/{name}")
def get_basemap(project_id: str, name: str):
    return store.get(project_id).basemap(name)


@app.delete("/api/projects/{project_id}/basemaps/{name}")
def delete_basemap(project_id: str, name: str):
    return store.get(project_id).remove_basemap(name)


# Esri imagery through the offline tile cache (magcore.tile_cache, rooted
# in MineMag's own data dir): the map's Esri layer asks this app for its
# tiles, so a tile seen while online is served from disk afterwards, and
# /api/tiles/download fills a survey area in advance for a site with no
# internet. OSM and Google are direct, uncached browser layers: OSM's
# policy forbids bulk use and its servers blocked our fetches with 403,
# Google's terms do not allow it either.
_CACHED_SOURCES = ("esri",)
@app.post("/api/tiles/estimate")
def estimate_tiles(req: TileBboxRequest):
    n_tiles = tile_cache.count_tiles(req.source, req.min_lat, req.min_lon, req.max_lat, req.max_lon, req.min_zoom, req.max_zoom)
    avg_kb_per_tile = 25 if req.source == "esri" else 15
    return {"n_tiles": n_tiles, "estimated_mb": round(n_tiles * avg_kb_per_tile / 1024, 1),
            "max_tiles": tile_cache.MAX_TILES_PER_REQUEST, "exceeds_max": n_tiles > tile_cache.MAX_TILES_PER_REQUEST}


@app.post("/api/tiles/download")
def download_tiles(req: TileBboxRequest):
    r = tile_cache.download_tiles(req.source, req.min_lat, req.min_lon, req.max_lat, req.max_lon, req.min_zoom, req.max_zoom)
    return {"n_total": r.n_total, "n_already_cached": r.n_already_cached, "n_downloaded": r.n_downloaded, "n_failed": r.n_failed}


@app.get("/api/tiles/status")
def tiles_status():
    status = tile_cache.cache_status()
    return {k: v for k, v in status.items() if k in _CACHED_SOURCES}


@app.get("/api/tiles/{source}/{z}/{x}/{y}")
def get_tile(source: str, z: int, x: int, y: int):
    if source not in _CACHED_SOURCES:
        raise HTTPException(status_code=404, detail=f"캐시를 거치지 않는 지도 소스입니다: {source}")
    content = tile_cache.get_cached_tile(source, z, x, y) or tile_cache.fetch_and_cache_tile(source, z, x, y)
    if content is None:
        raise HTTPException(status_code=404, detail="타일이 캐시에 없고 인터넷에서도 받아오지 못했습니다.")
    return Response(content=content, media_type=tile_cache.TILE_SOURCES[source]["media_type"],
                    headers={"Cache-Control": "public, max-age=86400"})


# Large orthophotos: an already-tiled {z}/{x}/{y} folder served from disk
# (tile once with QGIS "Generate XYZ tiles" or gdal2tiles.py). Not
# project-scoped, like DroneMag Studio's.
@app.post("/api/local-tiles/register")
def register_local_tile_folder(req: LocalTileFolderRequest):
    scheme_override = None if req.scheme == "auto" else req.scheme
    layer = local_tiles.register_local_tile_folder(req.path, req.label, scheme_override)
    south, west, north, east = layer.bounds
    return {"id": layer.id, "label": layer.label, "min_zoom": layer.min_zoom, "max_zoom": layer.max_zoom,
            "scheme": layer.scheme, "bounds": [[south, west], [north, east]]}


@app.post("/api/local-tiles/pick-folder")
def pick_local_tile_folder():
    return {"path": local_tiles.pick_folder_dialog()}


@app.delete("/api/local-tiles/{layer_id}")
def unregister_local_tile_folder(layer_id: str):
    local_tiles.unregister_local_tile_folder(layer_id)
    return {"status": "ok"}


@app.get("/api/local-tiles/{layer_id}/{z}/{x}/{y}")
def get_local_tile(layer_id: str, z: int, x: int, y: int):
    content, media_type = local_tiles.get_tile(layer_id, z, x, y)
    if content is None:
        raise HTTPException(status_code=404, detail="타일을 찾을 수 없습니다.")
    return Response(content=content, media_type=media_type)


@app.middleware("http")
async def _no_cache_for_the_page(request: Request, call_next):
    """index.html must not be cached: the bundle it names changes with
    every build, and a browser holding yesterday's page keeps running
    yesterday's app (seen 2026-10-27 after run_minemag.bat rebuilt)."""
    response = await call_next(request)
    if request.url.path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


_frontend_dist = paths.frontend_dist()
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
