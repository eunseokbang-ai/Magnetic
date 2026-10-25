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

from magcore.colormaps import register_custom_colormaps

from . import paths
from .models import GridRequest, PreprocessRequest, QcRequest, RepeatRequest, TargetRequest
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


_frontend_dist = paths.frontend_dist()
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
