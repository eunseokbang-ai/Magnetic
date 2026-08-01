from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .io_.base_loader import BaseLoadError
from .io_.drone_loader import DroneLoadError
from .chat import ChatError
from .models import (
    ChatRequest,
    EulerDeconvolutionRequest,
    GridRequest,
    InversionParams,
    InversionSectionRequest,
    InversionSliceRequest,
    ManualExcludeRequest,
    ProcessParams,
    TargetDetectionRequest,
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
# The 3D inversion volume payload (flattened isosurface coordinate arrays)
# can run into the tens of MB uncompressed; JSON full of repeated float
# patterns compresses very well.
app.add_middleware(GZipMiddleware, minimum_size=1000)


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


@app.exception_handler(ChatError)
async def chat_error_handler(request, exc: ChatError):
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


@app.post("/api/projects/{project_id}/grid/geotiff")
def export_grid_geotiff(project_id: str, req: GridRequest):
    project = store.get(project_id)
    data = project.export_grid_geotiff(req)
    return Response(content=data, media_type="image/tiff", headers={"Content-Disposition": "attachment; filename=grid.tif"})


@app.post("/api/projects/{project_id}/transform/geotiff")
def export_transform_geotiff(project_id: str, req: TransformRequest):
    project = store.get(project_id)
    data = project.export_transform_geotiff(req)
    return Response(
        content=data, media_type="image/tiff", headers={"Content-Disposition": f"attachment; filename={req.transform}.tif"}
    )


@app.post("/api/projects/{project_id}/euler-deconvolution")
def euler_deconvolution(project_id: str, req: EulerDeconvolutionRequest):
    project = store.get(project_id)
    return project.run_euler_deconvolution(req)


@app.post("/api/projects/{project_id}/target-detection")
def target_detection(project_id: str, req: TargetDetectionRequest):
    project = store.get(project_id)
    return project.run_target_detection(req)


@app.post("/api/overlay-images")
async def upload_overlay_image(file: UploadFile):
    content = await file.read()
    name = file.filename or "overlay"
    return load_geotiff_overlay(io.BytesIO(content), name=name)


@app.post("/api/projects/{project_id}/reference-layers")
async def upload_reference_layer(project_id: str, file: UploadFile):
    project = store.get(project_id)
    content = await file.read()
    name = file.filename or "reference"
    return project.add_reference_layer(name, content)


@app.delete("/api/projects/{project_id}/reference-layers/{name}")
def delete_reference_layer(project_id: str, name: str):
    project = store.get(project_id)
    return project.remove_reference_layer(name)


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


@app.post("/api/projects/{project_id}/inversion/slice/geotiff")
def export_inversion_slice_geotiff(project_id: str, req: InversionSliceRequest):
    project = store.get(project_id)
    data = project.export_inversion_slice_geotiff(req)
    return Response(content=data, media_type="image/tiff", headers={"Content-Disposition": "attachment; filename=inversion_slice.tif"})


@app.post("/api/projects/{project_id}/inversion/section")
def inversion_section(project_id: str, req: InversionSectionRequest):
    project = store.get(project_id)
    return project.get_inversion_vertical_section(req)


@app.get("/api/projects/{project_id}/inversion/volume")
def inversion_volume(project_id: str, threshold: float | None = None, threshold_max: float | None = None):
    project = store.get(project_id)
    return project.get_inversion_volume(threshold, threshold_max)


@app.get("/api/projects/{project_id}/inversion/export")
def export_inversion(project_id: str):
    project = store.get(project_id)
    data = project.export_inversion_npz()
    return Response(
        content=data, media_type="application/octet-stream", headers={"Content-Disposition": "attachment; filename=inversion_result.npz"}
    )


@app.post("/api/projects/{project_id}/inversion/import")
async def import_inversion(project_id: str, file: UploadFile):
    project = store.get(project_id)
    content = await file.read()
    return project.import_inversion_npz(content)


@app.get("/api/projects/{project_id}/inversion/export/csv")
def export_inversion_csv(project_id: str):
    project = store.get(project_id)
    data = project.export_inversion_csv()
    return Response(
        content=data, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=inversion_result.csv"}
    )


@app.get("/api/projects/{project_id}/save")
def save_project(project_id: str):
    project = store.get(project_id)
    data = project.save_project_bundle()
    return Response(
        content=data, media_type="application/zip", headers={"Content-Disposition": "attachment; filename=magnetic_project.zip"}
    )


@app.post("/api/projects/{project_id}/load")
async def load_project(project_id: str, file: UploadFile):
    project = store.get(project_id)
    content = await file.read()
    return project.load_project_bundle(content)


@app.get("/api/projects/{project_id}/report")
def export_report(project_id: str):
    project = store.get(project_id)
    md = project.generate_report()
    return Response(
        content=md.encode("utf-8"),
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=processing_report.md"},
    )


@app.post("/api/projects/{project_id}/chat")
def chat(project_id: str, req: ChatRequest):
    project = store.get(project_id)
    history = [m.model_dump() for m in req.history]
    return project.run_chat(req.message, history)


@app.get("/api/health")
def health():
    return {"status": "ok"}
