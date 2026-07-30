from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .io_.base_loader import BaseLoadError
from .io_.drone_loader import DroneLoadError
from .models import (
    GridRequest,
    InversionParams,
    InversionSectionRequest,
    InversionSliceRequest,
    ManualExcludeRequest,
    ProcessParams,
    TransformRequest,
)
from .processing.colormaps import register_custom_colormaps
from .processing.overlay_image import OverlayImageError, load_geotiff_overlay
from .store import ProjectError, store

register_custom_colormaps()

app = FastAPI(title="드론 자력탐사 자료 처리 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ProjectError)
async def project_error_handler(request, exc: ProjectError):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(DroneLoadError)
async def drone_error_handler(request, exc: DroneLoadError):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(BaseLoadError)
async def base_error_handler(request, exc: BaseLoadError):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(OverlayImageError)
async def overlay_image_error_handler(request, exc: OverlayImageError):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.post("/api/projects")
def create_project():
    project = store.create()
    return {"project_id": project.id}


@app.post("/api/projects/{project_id}/upload/drone")
async def upload_drone(project_id: str, files: list[UploadFile] = File(...)):
    project = store.get(project_id)
    buffers = [io.BytesIO(await f.read()) for f in files]
    summary = project.load_drone(buffers)
    return summary


@app.post("/api/projects/{project_id}/upload/base")
async def upload_base(project_id: str, files: list[UploadFile] = File(...)):
    project = store.get(project_id)
    buffers = [io.BytesIO(await f.read()) for f in files]
    summary = project.load_base(buffers)
    return summary


@app.post("/api/projects/{project_id}/process")
def process(project_id: str, params: ProcessParams):
    project = store.get(project_id)
    return project.run_pipeline(params)


@app.get("/api/projects/{project_id}/summary")
def summary(project_id: str):
    project = store.get(project_id)
    if project.processed is None:
        raise ProjectError("자료 처리를 먼저 실행하세요.")
    return project.process_summary()


@app.get("/api/projects/{project_id}/points")
def points(project_id: str, value: str = "anomaly"):
    project = store.get(project_id)
    return project.get_points(value)


@app.post("/api/projects/{project_id}/manual-exclude")
def manual_exclude(project_id: str, req: ManualExcludeRequest):
    project = store.get(project_id)
    return project.set_manual_exclude(req)


@app.post("/api/projects/{project_id}/grid")
def grid(project_id: str, req: GridRequest):
    project = store.get(project_id)
    return project.get_grid_overlay(req)


@app.post("/api/projects/{project_id}/transform")
def transform(project_id: str, req: TransformRequest):
    project = store.get(project_id)
    return project.get_transform_overlay(req)


@app.post("/api/overlay-images")
async def upload_overlay_image(file: UploadFile):
    content = await file.read()
    name = file.filename or "overlay"
    return load_geotiff_overlay(io.BytesIO(content), name=name)


@app.post("/api/projects/{project_id}/upload/dem")
async def upload_dem(project_id: str, file: UploadFile):
    project = store.get(project_id)
    content = await file.read()
    name = file.filename or "dem.tif"
    return project.load_dem(content, name)


@app.delete("/api/projects/{project_id}/dem")
def delete_dem(project_id: str):
    project = store.get(project_id)
    return project.clear_dem()


@app.post("/api/projects/{project_id}/inversion")
def run_inversion(project_id: str, params: InversionParams):
    project = store.get(project_id)
    return project.run_inversion(params)


@app.post("/api/projects/{project_id}/inversion/slice")
def inversion_slice(project_id: str, req: InversionSliceRequest):
    project = store.get(project_id)
    return project.get_inversion_horizontal_slice(req)


@app.post("/api/projects/{project_id}/inversion/section")
def inversion_section(project_id: str, req: InversionSectionRequest):
    project = store.get(project_id)
    return project.get_inversion_vertical_section(req)


@app.get("/api/projects/{project_id}/inversion/volume")
def inversion_volume(project_id: str, threshold: float | None = None, threshold_max: float | None = None):
    project = store.get(project_id)
    return project.get_inversion_volume(threshold, threshold_max)


@app.get("/api/health")
def health():
    return {"status": "ok"}
