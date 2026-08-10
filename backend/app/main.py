from __future__ import annotations

import io
import pathlib
import subprocess
from datetime import date, timedelta

import orjson
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from .io_.base_loader import BaseLoadError
from .io_.drone_loader import DroneLoadError
from .chat import ChatError
from .models import (
    AnalyticSignalDepthRequest,
    ChatRequest,
    ContactDetectionRequest,
    EulerDeconvolutionRequest,
    GeologyUnitInput,
    GeologyUnitUpdate,
    GridConfidenceRequest,
    GridRequest,
    InversionParams,
    InversionSectionRequest,
    InversionSlice3DRequest,
    InversionSliceRequest,
    IntermagnetFetchRequest,
    LineamentRequest,
    LocalTileFolderRequest,
    ManualExcludeRequest,
    DisplayBoundaryRequest,
    ManualSmoothRequest,
    MultiscaleEdgeRequest,
    NearestIntermagnetRequest,
    OverlaySampleRequest,
    PolygonExportRequest,
    PowerSpectrumRequest,
    ProcessParams,
    ProspectivityRequest,
    QcCertificateRequest,
    SpectralDepthRequest,
    StructureScanRequest,
    TargetDetectionRequest,
    TileBboxRequest,
    TiltDepthRequest,
    TransformRequest,
)
from .processing import local_tiles, tile_cache
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


def _detect_running_version() -> dict:
    """Which commit this server process is actually running - computed
    once at startup (not per-request) since the answer can't change until
    the process is restarted. Exists because "이미 고친 버그가 여전히
    재현된다"는 제보가 실제로는 run.bat이 최신 커밋을 받아오지 못했거나
    (git pull 실패/충돌) 빌드가 새로 되지 않은 채 예전 버전이 계속 실행
    중인 경우로 여러 차례 밝혀졌다 - 사용자가 화면에서 직접 커밋 해시를
    확인해 "정말 최신 버전을 실행 중인지"를 스스로 판단할 수 있게 한다."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        commit_date = subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M"],
            cwd=pathlib.Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return {"commit": commit, "commit_date": commit_date, "branch": branch}
    except Exception:
        return {"commit": None, "commit_date": None, "branch": None}


_RUNNING_VERSION = _detect_running_version()


@app.get("/api/version")
def get_version():
    return _RUNNING_VERSION


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
    filenames = [f.filename for f in files]
    summary = project.load_base(buffers, filenames)
    return summary


@app.post("/api/projects/{project_id}/base/iaga2002/upload")
async def upload_iaga2002(project_id: str, file: UploadFile = File(...)):
    """Preview (not yet apply) a manually-downloaded IAGA-2002 file as a
    substitute base station - works with no outbound network access from
    this server at all, unlike the /intermagnet/fetch endpoint below."""
    project = store.get(project_id)
    text = (await file.read()).decode("utf-8", errors="replace")
    return project.preview_intermagnet_text(text)


def _resolve_requested_dates(project, start_str: str | None, end_str: str | None, max_days: int = 31) -> list[date]:
    """Either an explicit [start_date, end_date] override, or (when both
    are omitted) the project's own distinct drone survey flight dates -
    see Project._survey_dates. Both drone timestamps and INTERMAGNET
    publication times are GPS/UTC-based, so the survey's own calendar
    dates line up directly with INTERMAGNET's without any timezone
    conversion."""
    if start_str and end_str:
        try:
            start_date = date.fromisoformat(start_str)
        except ValueError:
            raise ProjectError(f"올바르지 않은 시작일 형식입니다: {start_str!r} (예: 2026-07-24)")
        try:
            end_date = date.fromisoformat(end_str)
        except ValueError:
            raise ProjectError(f"올바르지 않은 종료일 형식입니다: {end_str!r} (예: 2026-07-24)")
        if end_date < start_date:
            raise ProjectError("종료일이 시작일보다 빠릅니다.")
        n_days = (end_date - start_date).days + 1
        if n_days > max_days:
            raise ProjectError(f"한 번에 조회할 수 있는 기간은 최대 {max_days}일입니다.")
        return [start_date + timedelta(days=i) for i in range(n_days)]

    if start_str or end_str:
        raise ProjectError("시작일과 종료일을 모두 입력하거나, 둘 다 비워 측선 자료의 촬영 날짜를 자동으로 사용하세요.")

    dates = project._survey_dates()
    if not dates:
        raise ProjectError("드론 자료를 먼저 업로드하거나, 시작일·종료일을 직접 입력하세요.")
    if len(dates) > max_days:
        raise ProjectError(
            f"측선 자료의 날짜 수가 너무 많습니다 (최대 {max_days}일, 현재 {len(dates)}일). "
            "시작일·종료일을 직접 입력해 범위를 좁혀주세요."
        )
    return dates


@app.post("/api/projects/{project_id}/base/intermagnet/fetch")
def fetch_intermagnet(project_id: str, req: IntermagnetFetchRequest):
    project = store.get(project_id)
    dates = _resolve_requested_dates(project, req.start_date, req.end_date)
    return project.fetch_intermagnet_preview(req.iaga_code, dates)


@app.post("/api/projects/{project_id}/base/intermagnet/apply")
def apply_intermagnet(project_id: str):
    project = store.get(project_id)
    return project.apply_intermagnet_preview()


@app.post("/api/projects/{project_id}/base/intermagnet/nearest/fetch")
def fetch_nearest_intermagnet(project_id: str, req: NearestIntermagnetRequest):
    project = store.get(project_id)
    dates = _resolve_requested_dates(project, req.start_date, req.end_date)
    return project.fetch_nearest_intermagnet_preview(
        dates, req.n_stations, req.target_lat, req.target_lon, req.max_distance_km
    )


@app.post("/api/projects/{project_id}/base/intermagnet/nearest/apply")
def apply_nearest_intermagnet(project_id: str):
    project = store.get(project_id)
    return project.apply_nearest_intermagnet_preview()


@app.get("/api/projects/{project_id}/base/intermagnet/nearest/comparison")
def get_nearest_intermagnet_comparison(project_id: str):
    project = store.get(project_id)
    return project.get_nearest_intermagnet_comparison()


@app.get("/api/projects/{project_id}/base/intermagnet/nearest/csv")
def export_nearest_intermagnet_csv(project_id: str):
    project = store.get(project_id)
    data = project.export_nearest_intermagnet_csv()
    return Response(
        content=data, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=intermagnet_nearest_estimate.csv"}
    )


@app.post("/api/projects/{project_id}/upload/heading_calibration")
async def upload_heading_calibration(project_id: str, files: list[UploadFile] = File(...)):
    project = store.get(project_id)
    buffers = [io.BytesIO(await f.read()) for f in files]
    summary = project.load_heading_calibration(buffers)
    return summary


@app.post("/api/projects/{project_id}/upload/repeatability")
async def upload_repeatability(project_id: str, files: list[UploadFile] = File(...)):
    project = store.get(project_id)
    buffers = [io.BytesIO(await f.read()) for f in files]
    summary = project.load_repeatability(buffers)
    return summary


@app.post("/api/projects/{project_id}/repeatability/analyze")
def repeatability_analyze(project_id: str):
    project = store.get(project_id)
    return project.run_repeatability_analysis()


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


@app.get("/api/projects/{project_id}/base/timeseries")
def base_timeseries(project_id: str):
    project = store.get(project_id)
    return project.get_base_timeseries()


@app.post("/api/projects/{project_id}/smooth")
def smooth(project_id: str, req: ManualSmoothRequest):
    project = store.get(project_id)
    return project.set_manual_smoothing(req)


@app.post("/api/projects/{project_id}/display-boundary")
def display_boundary(project_id: str, req: DisplayBoundaryRequest):
    project = store.get(project_id)
    return project.set_display_boundary(req)


@app.post("/api/projects/{project_id}/grid")
def grid(project_id: str, req: GridRequest):
    project = store.get(project_id)
    return project.get_grid_overlay(req)


@app.post("/api/projects/{project_id}/transform")
def transform(project_id: str, req: TransformRequest):
    project = store.get(project_id)
    return project.get_transform_overlay(req)


@app.post("/api/projects/{project_id}/overlay/sample")
def sample_overlay(project_id: str, req: OverlaySampleRequest):
    project = store.get(project_id)
    return project.sample_overlay_value(req.lat, req.lon)


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


@app.get("/api/projects/{project_id}/target-detection/csv")
def export_targets_csv(project_id: str):
    project = store.get(project_id)
    data = project.export_targets_csv()
    return Response(content=data, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=targets.csv"})


@app.get("/api/projects/{project_id}/target-detection/shapefile")
def export_targets_shapefile(project_id: str):
    project = store.get(project_id)
    data = project.export_targets_shapefile()
    return Response(
        content=data, media_type="application/zip", headers={"Content-Disposition": "attachment; filename=targets_shapefile.zip"}
    )


@app.post("/api/projects/{project_id}/grid/confidence")
def grid_confidence_overlay(project_id: str, req: GridConfidenceRequest):
    project = store.get(project_id)
    return project.get_grid_confidence_overlay(req)


@app.post("/api/projects/{project_id}/qc-certificate")
def qc_certificate(project_id: str, req: QcCertificateRequest):
    project = store.get(project_id)
    return project.generate_qc_certificate(req)


@app.post("/api/projects/{project_id}/structure-scan")
def structure_scan(project_id: str, req: StructureScanRequest):
    project = store.get(project_id)
    return project.scan_structure_distortion(req)


@app.post("/api/projects/{project_id}/multiscale-edges")
def multiscale_edges(project_id: str, req: MultiscaleEdgeRequest):
    project = store.get(project_id)
    return project.run_multiscale_edges(req)


@app.post("/api/projects/{project_id}/lineaments")
def lineaments(project_id: str, req: LineamentRequest):
    project = store.get(project_id)
    return project.run_lineament_extraction(req)


@app.post("/api/projects/{project_id}/depth-estimation/tilt")
def tilt_depth(project_id: str, req: TiltDepthRequest):
    project = store.get(project_id)
    return project.run_tilt_depth(req)


@app.post("/api/projects/{project_id}/depth-estimation/analytic-signal")
def analytic_signal_depth(project_id: str, req: AnalyticSignalDepthRequest):
    project = store.get(project_id)
    return project.run_analytic_signal_depth(req)


@app.post("/api/projects/{project_id}/depth-estimation/spectral")
def spectral_depth(project_id: str, req: SpectralDepthRequest):
    project = store.get(project_id)
    return project.run_spectral_depth(req)


@app.post("/api/projects/{project_id}/contacts")
def magnetic_contacts(project_id: str, req: ContactDetectionRequest):
    project = store.get(project_id)
    return project.run_magnetic_contact_detection(req)


@app.post("/api/projects/{project_id}/prospectivity")
def prospectivity(project_id: str, req: ProspectivityRequest):
    project = store.get(project_id)
    return project.run_prospectivity(req)


@app.get("/api/projects/{project_id}/prospectivity/overlay")
def prospectivity_overlay(project_id: str, colormap: str = "viridis"):
    project = store.get(project_id)
    return project.get_prospectivity_overlay(colormap)


@app.post("/api/projects/{project_id}/spectrum")
def power_spectrum(project_id: str, req: PowerSpectrumRequest):
    project = store.get(project_id)
    return project.get_power_spectrum(req)


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


@app.get("/api/projects/{project_id}/geology/units")
def list_geology_units(project_id: str):
    project = store.get(project_id)
    return project.get_geology_units()


@app.post("/api/projects/{project_id}/geology/units")
def add_geology_unit(project_id: str, req: GeologyUnitInput):
    project = store.get(project_id)
    return project.add_geology_unit(req)


@app.patch("/api/projects/{project_id}/geology/units/{unit_id}")
def update_geology_unit(project_id: str, unit_id: int, req: GeologyUnitUpdate):
    project = store.get(project_id)
    return project.update_geology_unit(unit_id, req)


@app.delete("/api/projects/{project_id}/geology/units/{unit_id}")
def remove_geology_unit(project_id: str, unit_id: int):
    project = store.get(project_id)
    return project.remove_geology_unit(unit_id)


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


@app.get("/api/projects/{project_id}/inversion/box_faces")
def inversion_box_faces(project_id: str, top_layer_index: int = 0):
    project = store.get(project_id)
    return _fast_json_response(project.get_inversion_box_faces(top_layer_index))


@app.post("/api/projects/{project_id}/inversion/slice_3d")
def inversion_slice_3d(project_id: str, req: InversionSlice3DRequest):
    project = store.get(project_id)
    return _fast_json_response(project.get_inversion_slice_3d(req))


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


# Offline basemap tile cache - not project-scoped, since map imagery is
# generic and reusable across projects/sessions (see processing/tile_cache.py).
_TILE_MEDIA_TYPES = {"png": "image/png", "jpg": "image/jpeg"}


@app.post("/api/tiles/estimate")
def estimate_tiles(req: TileBboxRequest):
    n_tiles = tile_cache.count_tiles(req.source, req.min_lat, req.min_lon, req.max_lat, req.max_lon, req.min_zoom, req.max_zoom)
    avg_kb_per_tile = 25 if req.source == "esri" else 15
    return {
        "n_tiles": n_tiles,
        "estimated_mb": round(n_tiles * avg_kb_per_tile / 1024, 1),
        "max_tiles": tile_cache.MAX_TILES_PER_REQUEST,
        "exceeds_max": n_tiles > tile_cache.MAX_TILES_PER_REQUEST,
    }


@app.post("/api/tiles/download")
def download_tiles(req: TileBboxRequest):
    result = tile_cache.download_tiles(req.source, req.min_lat, req.min_lon, req.max_lat, req.max_lon, req.min_zoom, req.max_zoom)
    return {
        "n_total": result.n_total,
        "n_already_cached": result.n_already_cached,
        "n_downloaded": result.n_downloaded,
        "n_failed": result.n_failed,
    }


@app.get("/api/tiles/status")
def tiles_status():
    return tile_cache.cache_status()


@app.get("/api/tiles/{source}/{z}/{x}/{y}")
def get_tile(source: str, z: int, x: int, y: int):
    if source not in tile_cache.TILE_SOURCES:
        raise HTTPException(status_code=404, detail=f"지원하지 않는 지도 소스입니다: {source}")
    content = tile_cache.get_cached_tile(source, z, x, y)
    if content is None:
        # Opportunistic: not cached yet, but if the internet happens to be
        # reachable right now, fetch it live and cache it for next time -
        # so simply browsing this layer while online gradually builds the
        # offline cache too, not just the explicit bulk-download endpoint.
        content = tile_cache.fetch_and_cache_tile(source, z, x, y)
    if content is None:
        raise HTTPException(status_code=404, detail="타일을 찾을 수 없습니다 (캐시에 없고 인터넷에서도 받아오지 못했습니다).")
    media_type = _TILE_MEDIA_TYPES[tile_cache.TILE_SOURCES[source]["ext"]]
    return Response(content=content, media_type=media_type)


# Serves an already-tiled raster pyramid straight off local disk (e.g. a
# very large orthophoto tiled once with QGIS/gdal2tiles.py) - not
# project-scoped, same rationale as the offline tile cache above. See
# processing/local_tiles.py for why this exists instead of just uploading
# the raw GeoTIFF through /overlay-images.
@app.post("/api/local-tiles/register")
def register_local_tile_folder(req: LocalTileFolderRequest):
    scheme_override = None if req.scheme == "auto" else req.scheme
    layer = local_tiles.register_local_tile_folder(req.path, req.label, scheme_override)
    south, west, north, east = layer.bounds
    return {
        "id": layer.id,
        "label": layer.label,
        "min_zoom": layer.min_zoom,
        "max_zoom": layer.max_zoom,
        "scheme": layer.scheme,
        "bounds": [[south, west], [north, east]],
    }


@app.post("/api/local-tiles/pick-folder")
def pick_local_tile_folder():
    path = local_tiles.pick_folder_dialog()
    return {"path": path}


@app.delete("/api/local-tiles/{layer_id}")
def unregister_local_tile_folder(layer_id: str):
    local_tiles.unregister_local_tile_folder(layer_id)
    return {"status": "ok"}


@app.get("/api/local-tiles/{layer_id}/{z}/{x}/{y}")
def get_local_tile(layer_id: str, z: int, x: int, y: int):
    content, media_type = local_tiles.get_tile(layer_id, z, x, y)
    if content is None:
        raise HTTPException(status_code=404, detail="타일을 찾을 수 없습니다 (등록되지 않은 레이어이거나 해당 좌표에 타일 파일이 없습니다).")
    return Response(content=content, media_type=media_type)


# When a production build exists (frontend/dist, produced by `npm run
# build` - see the Windows launcher's run.bat), serve it from this same
# process so the whole app runs as a single executable/port instead of
# needing a separate `npm run dev` terminal. Mounted last and after every
# /api/* route above so those always win; html=True serves index.html for
# unmatched paths, which is what the client-side router needs. In local
# dev (no dist/ built yet, frontend served by its own `vite` dev server
# instead) this mount is simply skipped - the API-only behavior is
# unchanged.
_frontend_dist = pathlib.Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
