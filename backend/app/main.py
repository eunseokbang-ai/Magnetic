from __future__ import annotations

import io

import orjson
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
    PolygonExportRequest,
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
# and the raw point list (tens of MB at 100k+ points) compress very well,
# but Starlette's default compresslevel=9 spends ~4x longer than
# compresslevel=6 for a ~0.5% size gain on this kind of repeated-float
# JSON - at ~20MB payloads that is 1-2s of pure CPU time added to every
# request, independent of how fast the actual endpoint logic is.
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)


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


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    # Catch-all for plain ValueError raised deep in processing/* (grid size
    # caps, parameter validation, etc.) that isn't already one of the more
    # specific *Error types above - without this, FastAPI has no handler
    # for a bare ValueError and it surfaces as an opaque 500, hiding the
    # actual (often quite actionable) Korean message the exception carries.
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _fast_json_response(data) -> Response:
    """FastAPI's default return-value path runs every payload through
    jsonable_encoder (a slow, fully recursive type-dispatch pass) before
    handing it to json.dumps - for a 90k-row point list that's ~1.2s of
    pure Python overhead that has nothing to do with how fast the actual
    endpoint logic is. orjson serializes the already-JSON-native records
    directly (and, as a bonus, turns NaN/Infinity into `null` instead of
    the non-standard `NaN` literal json.dumps would emit, which browsers'
    JSON.parse cannot read). Falls back to the normal FastAPI path for
    the rare payload orjson can't handle."""
    try:
        return Response(content=orjson.dumps(data), media_type="application/json")
    except TypeError:
        from fastapi.encoders import jsonable_encoder
        from fastapi.responses import JSONResponse

        return JSONResponse(content=jsonable_encoder(data))


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


@app.post("/api/projects/{project_id}/upload/heading_calibration")
async def upload_heading_calibration(project_id: str, files: list[UploadFile] = File(...)):
    project = store.get(project_id)
    buffers = [io.BytesIO(await f.read()) for f in files]
    summary = project.load_heading_calibration(buffers)
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
    return _fast_json_response(project.get_points(value))


@app.get("/api/projects/{project_id}/line-profile")
def line_profile(project_id: str, line_id: int, value: str = "anomaly"):
    project = store.get(project_id)
    return project.get_line_profile(line_id, value)


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


@app.post("/api/projects/{project_id}/grid/xyz")
def export_grid_xyz(project_id: str, req: GridRequest):
    project = store.get(project_id)
    data = project.export_grid_xyz(req)
    return Response(content=data, media_type="text/plain", headers={"Content-Disposition": "attachment; filename=grid.xyz"})


@app.post("/api/projects/{project_id}/transform/xyz")
def export_transform_xyz(project_id: str, req: TransformRequest):
    project = store.get(project_id)
    data = project.export_transform_xyz(req)
    return Response(
        content=data, media_type="text/plain", headers={"Content-Disposition": f"attachment; filename={req.transform}.xyz"}
    )


@app.post("/api/projects/{project_id}/grid/grd")
def export_grid_grd(project_id: str, req: GridRequest):
    project = store.get(project_id)
    data = project.export_grid_surfer_grd(req)
    return Response(
        content=data, media_type="application/octet-stream", headers={"Content-Disposition": "attachment; filename=grid.grd"}
    )


@app.post("/api/projects/{project_id}/transform/grd")
def export_transform_grd(project_id: str, req: TransformRequest):
    project = store.get(project_id)
    data = project.export_transform_surfer_grd(req)
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={req.transform}.grd"},
    )


@app.post("/api/projects/{project_id}/export/bln")
def export_polygon_bln(project_id: str, req: PolygonExportRequest):
    project = store.get(project_id)
    data = project.export_polygon_bln(req.polygon)
    return Response(
        content=data, media_type="text/plain", headers={"Content-Disposition": "attachment; filename=area.bln"}
    )


@app.get("/api/projects/{project_id}/points/csv")
def export_points_csv(project_id: str):
    project = store.get(project_id)
    data = project.export_points_csv()
    return Response(content=data, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=points.csv"})


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
    return _fast_json_response(project.get_inversion_volume(threshold, threshold_max))


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
