from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .io_.base_loader import BaseLoadError
from .io_.drone_loader import DroneLoadError
from .models import GridRequest, ManualExcludeRequest, ProcessParams, TransformRequest
from .store import ProjectError, store

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


@app.get("/api/health")
def health():
    return {"status": "ok"}
