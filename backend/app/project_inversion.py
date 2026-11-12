"""3D susceptibility inversion: the DEM and geology-unit inputs, the run
itself, slices / sections / volume views of the result, and its import
and export. Part of store.Project.

Split out of store.py in 2026-11; the methods are unchanged. They rely on
the attributes store.Project declares (this class has no state of its
own) and are only meant to be used through it.
"""
from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pandas as pd
import rasterio
from magcore import progress
from magcore.gridding import GridResult, grid_points
from magcore.igrf import mean_field_intensity_nt
from magcore.inversion import (
    InversionError,
    InversionMesh,
    InversionResult,
    azimuth_line,
    box_faces,
    build_mesh,
    build_sensitivity_matrix,
    horizontal_slice,
    internal_slice,
    invert,
    path_slice_3d as _inversion_path_slice_3d,
    render_section_png,
    upsample_susceptibility,
    vertical_section as _inversion_vertical_section,
)
from magcore.inversion_auto import DEFAULT_DEPTH_GROWTH_FACTOR, suggest_mesh_params
from magcore.render import grid_to_geotiff_bytes, grid_to_png_overlay
from magcore.terrain import TerrainError, estimate_ground_elevation, load_dem_geotiff
from magcore.trend import separate_regional
from matplotlib.path import Path as MplPath
from pyproj import Transformer
from .models import (
    AnomalyCandidateRequest,
    GeologyUnitInput,
    GeologyUnitUpdate,
    InversionParams,
    InversionSectionRequest,
    InversionSlice3DRequest,
    InversionSliceRequest,
)
from .project_common import ProjectError, _as_job, _stats


# How many of the block's strongest anomalies the inversion's remanence check looks at.
REMANENCE_CHECK_CANDIDATES = 6


class InversionMixin:
    def _remanence_check(self, n_candidates: int = REMANENCE_CHECK_CANDIDATES) -> dict:
        """Whether the inversion's one assumption about magnetization holds
        where it matters most.

        The inversion solves for susceptibility: every cell is magnetized
        along today's field, in proportion to it. A source carrying
        remanence is not, and the solver then explains its anomaly with
        susceptibility in the wrong place - shifted sideways and in
        depth, with a halo of compensating cells around it - while still
        fitting the data. Nothing in the misfit shows it.

        What can be checked without a second inversion: the strongest
        anomalies of the block are each fitted with a dipole whose
        direction is free and with one held to the induced direction
        (magnetization.py, the same test the candidate scan shows per
        candidate). Sources whose direction is far from the field's, with
        the freedom buying a real drop in misfit, are reported, and the
        summary carries a warning naming how many and how far off. It is
        a warning about where not to trust the model, not a correction;
        a magnetization-vector inversion would be the correction.
        """
        if self.processed is None or self.inclination_deg is None:
            return {"available": False, "reason": "자기장 방향을 알 수 없어 확인하지 못했습니다."}
        cell = max(2.0, float(self.line_spacing_m or 20.0) / 4.0)
        try:
            scan = self.scan_anomaly_candidates(AnomalyCandidateRequest(n_candidates=n_candidates, cell_size_m=cell))
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail the inversion it describes
            return {"available": False, "reason": f"잔류자화 확인 실패: {exc}"}
        sources = []
        for c in scan["candidates"]:
            m = c.get("magnetization")
            if not m:
                continue
            sources.append({
                "rank": c["rank"], "lat": c["lat"], "lon": c["lon"], "peak_anomaly_nt": c["peak_anomaly_nt"],
                "angle_from_induced_deg": m["angle_from_induced_deg"], "improvement": m["improvement"],
                "verdict": m["verdict"], "label": m["label"],
            })
        remanent = [s for s in sources if s["verdict"] == "remanent"]
        warning = None
        if remanent:
            angles = [s["angle_from_induced_deg"] for s in remanent]
            span = f"{min(angles):.0f}°" if len(angles) == 1 else f"{min(angles):.0f}~{max(angles):.0f}°"
            warning = (
                f"가장 강한 이상 {len(sources)}곳 중 {len(remanent)}곳의 자화 방향이 현재 지자기장 방향과 {span} 어긋납니다(잔류자화). "
                "이 역산은 유도자화만 가정하므로 그 주변에서는 자화율의 크기·위치·심도를 믿기 어렵습니다. "
                "잔류자화를 띤 암체일 수도, 인공 구조물일 수도 있습니다 - 구조물이라면 7-1에서 제거한 뒤 다시 역산하세요."
            )
        return {
            "available": True,
            "n_checked": len(sources),
            "n_remanent": len(remanent),
            "n_unclear": sum(1 for s in sources if s["verdict"] == "unclear"),
            "sources": sources,
            "warning": warning,
        }

    def load_dem(self, data: bytes, name: str) -> dict:
        try:
            with rasterio.open(io.BytesIO(data)) as src:
                if src.crs is None:
                    raise ProjectError("DEM GeoTIFF에 좌표계(CRS) 정보가 없습니다.")
                bounds = src.bounds
                width, height = src.width, src.height
        except rasterio.errors.RasterioIOError as exc:
            raise ProjectError(f"DEM 파일을 열 수 없습니다: {exc}") from exc
        self.dem_bytes = data
        self.dem_name = name
        self.mark_changed()
        return {"name": name, "width": width, "height": height, "bounds": list(bounds)}

    def clear_dem(self) -> dict:
        self.dem_bytes = None
        self.dem_name = None
        return {"cleared": True}

    def add_geology_unit(self, req: GeologyUnitInput) -> dict:
        """Add one user-digitized geology block (a polygon traced over an
        uploaded geology map, e.g. the "12. 참조 레이어" GeoTIFF, plus its
        assigned susceptibility) to the inversion's reference model. Later-
        added units win where polygons overlap - see run_inversion's
        rasterization, which walks self.geology_units in order."""
        if self.utm_epsg is None:
            raise ProjectError("좌표계 정보가 없습니다 - 먼저 드론 자료를 업로드/처리하세요.")
        unit = {
            "id": self.geology_unit_next_id,
            "name": req.name,
            "susceptibility_si": req.susceptibility_si,
            "path": req.path,
        }
        self.geology_unit_next_id += 1
        self.geology_units.append(unit)
        self.mark_changed()
        return {"unit": unit, "units": self.geology_units}

    def update_geology_unit(self, unit_id: int, req: GeologyUnitUpdate) -> dict:
        for unit in self.geology_units:
            if unit["id"] == unit_id:
                if req.name is not None:
                    unit["name"] = req.name
                if req.susceptibility_si is not None:
                    unit["susceptibility_si"] = req.susceptibility_si
                self.mark_changed()
                return {"unit": unit, "units": self.geology_units}
        raise ProjectError(f"지질 블록 {unit_id}을(를) 찾을 수 없습니다.")

    def remove_geology_unit(self, unit_id: int) -> dict:
        before = len(self.geology_units)
        self.geology_units = [u for u in self.geology_units if u["id"] != unit_id]
        if len(self.geology_units) == before:
            raise ProjectError(f"지질 블록 {unit_id}을(를) 찾을 수 없습니다.")
        self.mark_changed()
        return {"units": self.geology_units}

    def get_geology_units(self) -> dict:
        return {"units": self.geology_units}

    def _rasterize_geology_reference(self, x_centers: np.ndarray, y_centers: np.ndarray) -> np.ndarray:
        """(ny, nx) array of the reference susceptibility assigned to each
        mesh column by self.geology_units, NaN where no digitized polygon
        covers that column - same lat/lon-to-local-UTM MplPath technique
        already used for structure-scan polygon matching."""
        ref = np.full((len(y_centers), len(x_centers)), np.nan)
        if not self.geology_units:
            return ref
        x_grid, y_grid = np.meshgrid(x_centers, y_centers)  # (ny, nx)
        points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        to_local = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
        for unit in self.geology_units:
            xs, ys = to_local.transform([p[1] for p in unit["path"]], [p[0] for p in unit["path"]])
            poly_path = MplPath(list(zip(xs, ys)))
            inside = poly_path.contains_points(points).reshape(ref.shape)
            ref[inside] = unit["susceptibility_si"]
        return ref

    @_as_job("3차원 역산")
    def run_inversion(self, params: InversionParams) -> dict:
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed
        active = self._active_mask()
        sub = df.loc[active]
        if len(sub) < 20:
            raise ProjectError("역산을 위한 유효 포인트가 부족합니다.")
        col = "anomaly" if params.value == "anomaly" else "tmi"

        auto_used = params.obs_cell_size_m is None or params.depth_extent_m is None or params.n_layers is None
        auto_suggestion = None
        if auto_used:
            auto_suggestion = suggest_mesh_params(
                sub["x"].to_numpy(), sub["y"].to_numpy(), sub[col].to_numpy(),
                self.line_spacing_m, N_OBS_CAP, N_ACTIVE_CAP,
            )
        obs_cell_size_m = params.obs_cell_size_m or auto_suggestion["obs_cell_size_m"]
        depth_extent_m = params.depth_extent_m or auto_suggestion["depth_extent_m"]
        n_layers = params.n_layers or auto_suggestion["n_layers"]

        max_distance = self._resolve_max_distance(obs_cell_size_m, None)
        obs_grid = grid_points(
            sub["x"].to_numpy(), sub["y"].to_numpy(), sub[col].to_numpy(),
            obs_cell_size_m, method="nearest", max_distance_m=max_distance,
        )
        alt_grid = grid_points(
            sub["x"].to_numpy(), sub["y"].to_numpy(), sub["altitude_ellipsoidal_m"].to_numpy(),
            obs_cell_size_m, method="nearest", max_distance_m=max_distance,
        )

        # The background the model is not asked to explain (it could only do
        # so with susceptibility that is not there). What is inverted, kept
        # and shown from here on is the residual.
        try:
            residual, _regional, regional_info = separate_regional(
                obs_grid.values, obs_grid.easting, obs_grid.northing, params.regional_removal)
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
        obs_grid = replace(obs_grid, values=residual)

        ny, nx = obs_grid.values.shape
        if nx * ny > N_OBS_CAP:
            raise ProjectError(
                f"관측 격자가 너무 촘촘합니다 ({nx}x{ny}={nx*ny}점, 최대 {N_OBS_CAP}점) - 격자 크기(obs_cell_size_m)를 늘려주세요."
            )

        if self.dem_bytes is not None:
            try:
                ground_elev = load_dem_geotiff(io.BytesIO(self.dem_bytes), obs_grid.easting, obs_grid.northing, self.utm_epsg)
            except TerrainError as exc:
                raise ProjectError(str(exc)) from exc
            # Reconcile the DEM's vertical datum (usually orthometric)
            # with the drone GPS's own ellipsoidal altitude - see
            # InversionParams.dem_geoid_offset_m's docstring.
            ground_elev = ground_elev + params.dem_geoid_offset_m
        else:
            point_ground = estimate_ground_elevation(sub["altitude_ellipsoidal_m"].to_numpy(), params.assumed_agl_m)
            ground_result = grid_points(
                sub["x"].to_numpy(), sub["y"].to_numpy(), point_ground,
                obs_cell_size_m, method="nearest", max_distance_m=max_distance,
            )
            ground_elev = ground_result.values

        growth_factor = params.depth_growth_factor or DEFAULT_DEPTH_GROWTH_FACTOR
        try:
            progress.report("메시 구성", 0.05)
            mesh = build_mesh(
                obs_grid.easting, obs_grid.northing, ground_elev,
                cell_size_m=obs_cell_size_m, depth_extent_m=depth_extent_m, n_layers=n_layers,
                growth_factor=growth_factor,
            )
        except InversionError as exc:
            raise ProjectError(str(exc)) from exc

        n_mesh_cells = int(mesh.active.sum())
        if n_mesh_cells > N_ACTIVE_CAP:
            raise ProjectError(
                f"역산 메쉬가 너무 큽니다 ({n_mesh_cells}셀, 최대 {N_ACTIVE_CAP}셀) - 격자 크기를 늘리거나 레이어 수/심도를 줄여주세요."
            )

        easting_2d, northing_2d = np.meshgrid(obs_grid.easting, obs_grid.northing)
        obs_mask = np.isfinite(obs_grid.values) & np.isfinite(alt_grid.values)
        obs_x = easting_2d[obs_mask]
        obs_y = northing_2d[obs_mask]
        obs_z = alt_grid.values[obs_mask]
        data_nt = obs_grid.values[obs_mask]
        if len(data_nt) < 10:
            raise ProjectError("역산을 위한 유효 관측 격자점이 부족합니다.")

        field_intensity_nt = mean_field_intensity_nt(
            sub["lat"].to_numpy(), sub["lon"].to_numpy(), sub["altitude_ellipsoidal_m"].to_numpy(), sub["timestamp"]
        )
        self.inversion_field_intensity_nt = field_intensity_nt

        m_ref_flat = None
        geology_coverage_frac = None
        try:
            progress.report("감도 행렬 계산", 0.10)
            G, rows, cols, layers = build_sensitivity_matrix(
                obs_x, obs_y, obs_z, mesh, self.inclination_deg, self.declination_deg, field_intensity_nt
            )
            if self.geology_units and params.use_geology_reference:
                # Repeat each digitized geology block's susceptibility
                # straight down through every active layer in that column
                # (a "2.5D" reference model - see GeologyUnitInput's
                # docstring on why: early-stage geology maps rarely carry
                # dip information to build a true 3D unit boundary from).
                geology_ref_2d = self._rasterize_geology_reference(mesh.x_centers, mesh.y_centers)
                ref_at_cells = geology_ref_2d[rows, cols]
                m_ref_flat = np.where(np.isfinite(ref_at_cells), ref_at_cells, 0.0)
                geology_coverage_frac = float(np.mean(np.isfinite(ref_at_cells))) if ref_at_cells.size else 0.0
            with progress.span(0.35, 0.95):
                result = invert(
                    G, data_nt, mesh, rows, cols, layers,
                    regularization_strength=params.regularization_strength,
                    n_irls_iterations=params.n_irls_iterations,
                    assumed_noise_nt=params.assumed_noise_nt,
                    m_ref=m_ref_flat,
                )
        except InversionError as exc:
            raise ProjectError(str(exc)) from exc

        progress.report("결과 정리", 0.96)
        self.inversion_result = result
        self.inversion_params = params
        self.inversion_obs_grid = obs_grid
        self.inversion_value_field = params.value

        active_chi = result.susceptibility[result.susceptibility > 0]
        summary = {
            "n_obs": result.n_obs,
            "n_active_cells": result.n_active_cells,
            "rms_misfit_nt": result.rms_misfit_nt,
            "iterations": result.iterations,
            "n_layers": n_layers,
            "obs_cell_size_m": obs_cell_size_m,
            "depth_extent_m": depth_extent_m,
            "cell_size_m": mesh.cell_size_m,
            "layer_thickness_m": [float(t) for t in mesh.layer_thickness_m],
            "depth_growth_factor": growth_factor,
            "elevation_range_m": [float(mesh.z_centers.min()), float(mesh.z_centers.max())],
            "field": {
                "inclination_deg": self.inclination_deg,
                "declination_deg": self.declination_deg,
                "field_intensity_nt": field_intensity_nt,
            },
            "susceptibility_stats": _stats(pd.Series(active_chi)) if active_chi.size else _stats(pd.Series(dtype=float)),
            "used_dem": self.dem_bytes is not None,
            "regional_removal": regional_info,
            "regional_warning": _regional_warning(regional_info),
            "auto_params": auto_used,
            "source_depth_estimate_m": auto_suggestion["source_depth_estimate_m"] if auto_suggestion else None,
            "resolution_warning": _resolution_warning(obs_cell_size_m, self.line_spacing_m, nx, ny),
            "regularization_strength_used": result.regularization_strength_used,
            "auto_regularization": params.assumed_noise_nt is not None,
            "assumed_noise_nt": params.assumed_noise_nt,
            "layer_elevations_m": [float(z) for z in mesh.z_centers],
            "depth_resolution": result.depth_resolution,
            "depth_resolution_warning": _depth_resolution_warning(result.depth_resolution, mesh.z_centers),
            "geology_reference_used": m_ref_flat is not None,
            "geology_reference_coverage": geology_coverage_frac,
            "n_geology_units": len(self.geology_units),
        }
        summary["remanence_check"] = self._remanence_check()
        self.inversion_summary_cache = summary
        self.mark_changed()
        return summary

    def get_inversion_horizontal_slice(self, req: InversionSliceRequest) -> dict:
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        mesh = self.inversion_result.mesh
        try:
            slice_2d = horizontal_slice(self.inversion_result, req.layer_index, req.threshold, req.threshold_max)
            overlay = grid_to_png_overlay(
                slice_2d, mesh.x_centers, mesh.y_centers, self.utm_epsg,
                cmap_name=req.cmap or "geosoft_rainbow", symmetric=False, vmin=req.vmin, vmax=req.vmax,
                hillshade=req.hillshade,
                hillshade_azimuth_deg=req.hillshade_azimuth_deg,
                hillshade_altitude_deg=req.hillshade_altitude_deg,
                hillshade_exaggeration=req.hillshade_exaggeration,
                cell_size_m=mesh.cell_size_m,
            )
        except (InversionError, ValueError) as exc:
            raise ProjectError(str(exc)) from exc
        layer_index = int(np.clip(req.layer_index, 0, mesh.active.shape[2] - 1))
        overlay["stats"] = _stats(pd.Series(slice_2d.ravel()))
        overlay["layer_index"] = layer_index
        overlay["n_layers"] = int(mesh.active.shape[2])
        overlay["elevation_m"] = float(mesh.z_centers[layer_index])
        return overlay

    def export_inversion_slice_geotiff(self, req: InversionSliceRequest) -> bytes:
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        mesh = self.inversion_result.mesh
        try:
            slice_2d = horizontal_slice(self.inversion_result, req.layer_index, req.threshold, req.threshold_max)
        except InversionError as exc:
            raise ProjectError(str(exc)) from exc
        return grid_to_geotiff_bytes(slice_2d, mesh.x_centers, mesh.y_centers, self.utm_epsg)

    def get_inversion_vertical_section(self, req: InversionSectionRequest) -> dict:
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        mesh = self.inversion_result.mesh

        if req.profile == "custom":
            if not req.path or len(req.path) < 2:
                raise ProjectError("자유선 단면을 위해서는 경로(path)가 필요합니다.")
            if self.utm_epsg is None:
                raise ProjectError("좌표계 정보가 없습니다.")
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
            lats = [pt[0] for pt in req.path]
            lons = [pt[1] for pt in req.path]
            vx, vy = transformer.transform(lons, lats)
            path_x, path_y = _densify_path(np.asarray(vx), np.asarray(vy), req.sample_spacing_m)
        else:
            if req.position_frac is None:
                raise ProjectError("동서/남북 단면을 위해서는 position_frac이 필요합니다.")
            n_samples = max(2, int((mesh.x_centers.max() - mesh.x_centers.min() if req.profile == "ew" else mesh.y_centers.max() - mesh.y_centers.min()) / req.sample_spacing_m) + 1)
            if req.profile == "ew":
                # straight line at a fixed north-south position, spanning
                # the mesh's full east-west extent - local UTM x/y is
                # close enough to true east/west, north/south at this
                # survey scale.
                fixed_y = mesh.y_centers.min() + req.position_frac * (mesh.y_centers.max() - mesh.y_centers.min())
                path_x = np.linspace(mesh.x_centers.min(), mesh.x_centers.max(), n_samples)
                path_y = np.full(n_samples, fixed_y)
            else:  # "ns"
                fixed_x = mesh.x_centers.min() + req.position_frac * (mesh.x_centers.max() - mesh.x_centers.min())
                path_y = np.linspace(mesh.y_centers.min(), mesh.y_centers.max(), n_samples)
                path_x = np.full(n_samples, fixed_x)

        try:
            section, distance, ground_elev_path = _inversion_vertical_section(
                self.inversion_result, path_x, path_y, req.threshold, req.threshold_max
            )
            png = render_section_png(
                section, distance, self.inversion_result.mesh.z_centers, ground_elev_path,
                path_x, path_y, mesh.x_centers, mesh.y_centers, profile=req.profile,
                cmap_name=req.cmap or "geosoft_rainbow", vmin=req.vmin, vmax=req.vmax,
            )
        except InversionError as exc:
            raise ProjectError(str(exc)) from exc
        png["stats"] = _stats(pd.Series(section.ravel()))
        png["profile"] = req.profile
        return png

    def get_inversion_volume(self, threshold: float | None = None, threshold_max: float | None = None) -> dict:
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        result = self.inversion_result
        mesh = result.mesh
        x0 = float(mesh.x_centers.mean())
        y0 = float(mesh.y_centers.mean())

        # Interpolate onto a much finer grid before thresholding, purely
        # so the isosurface renders as a smooth "blob" instead of a
        # blocky voxel shape - the inversion itself is unaffected, this
        # only changes how the already-solved model is displayed. See
        # processing/inversion.upsample_susceptibility.
        fine_x, fine_y, fine_z, fine_chi = upsample_susceptibility(mesh, result.susceptibility)

        chi = fine_chi
        if threshold is not None:
            chi = np.where(chi >= threshold, chi, 0.0)
        if threshold_max is not None:
            chi = np.where(chi <= threshold_max, chi, 0.0)

        Y, X, Z = np.meshgrid(fine_y - y0, fine_x - x0, fine_z, indexing="ij")
        active_chi = result.susceptibility[result.susceptibility > 0]

        # Drape the actual observed anomaly/TMI map used as inversion
        # input on the terrain-following top of the mesh, matching the
        # common "2D magnetic map on a 3D inversion box" presentation -
        # gives geographic context for where the recovered bodies sit.
        top = None
        if self.inversion_obs_grid is not None:
            top = {
                "x": (mesh.x_centers - x0).tolist(),
                "y": (mesh.y_centers - y0).tolist(),
                "z": mesh.ground_elevation.tolist(),
                "color": np.where(
                    np.isfinite(self.inversion_obs_grid.values), self.inversion_obs_grid.values, None
                ).tolist(),
                "value_field": self.inversion_value_field,
            }

        return {
            "x": X.ravel().tolist(),
            "y": Y.ravel().tolist(),
            "z": Z.ravel().tolist(),
            "value": chi.ravel().tolist(),
            "shape": list(fine_chi.shape),
            "stats": _stats(pd.Series(active_chi)) if active_chi.size else _stats(pd.Series(dtype=float)),
            "top": top,
        }

    def get_inversion_box_faces(self, top_layer_index: int = 0) -> dict:
        """Fence-diagram style 3D view: top horizontal slice + the 4
        vertical boundary walls of the mesh, one continuous SI
        colorscale, no threshold gating - see processing/inversion.py:
        box_faces for the layout this mirrors."""
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        mesh = self.inversion_result.mesh
        x0 = float(mesh.x_centers.mean())
        y0 = float(mesh.y_centers.mean())
        faces = box_faces(self.inversion_result, top_layer_index)
        for key in ("top", "south", "north", "west", "east"):
            face = faces[key]
            face["x"] = (np.asarray(face["x"]) - x0).tolist()
            face["y"] = (np.asarray(face["y"]) - y0).tolist()
        return faces

    def get_inversion_slice_3d(self, req: InversionSlice3DRequest) -> dict:
        """One interior cutting plane (ew/ns at an arbitrary interior
        position, or an arbitrary-direction custom path) in the same
        mesh-centered local coordinates as get_inversion_volume's
        isosurface and get_inversion_box_faces' walls, so it can be added
        as an extra Plotly surface trace into the same 3D scene - the
        "volume + several simultaneous section planes at once" combined
        view (mirrors the standard multi-panel presentation in
        mining-industry 3D modeling packages)."""
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        mesh = self.inversion_result.mesh
        x0 = float(mesh.x_centers.mean())
        y0 = float(mesh.y_centers.mean())

        if req.orientation == "custom":
            if not req.path or len(req.path) < 2:
                raise ProjectError("자유선 단면을 위해서는 경로(path)가 필요합니다.")
            if self.utm_epsg is None:
                raise ProjectError("좌표계 정보가 없습니다.")
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
            lats = [pt[0] for pt in req.path]
            lons = [pt[1] for pt in req.path]
            vx, vy = transformer.transform(lons, lats)
            path_x, path_y = _densify_path(np.asarray(vx), np.asarray(vy), req.sample_spacing_m)
            try:
                face = _inversion_path_slice_3d(self.inversion_result, path_x, path_y, req.threshold, req.threshold_max)
            except InversionError as exc:
                raise ProjectError(str(exc)) from exc
            face["orientation"] = "custom"
        elif req.orientation == "azimuth":
            if req.azimuth_deg is None:
                raise ProjectError("자유방향 단면을 위해서는 azimuth_deg가 필요합니다.")
            path_x, path_y = azimuth_line(mesh, req.azimuth_deg, req.sample_spacing_m)
            try:
                face = _inversion_path_slice_3d(self.inversion_result, path_x, path_y, req.threshold, req.threshold_max)
            except InversionError as exc:
                raise ProjectError(str(exc)) from exc
            face["orientation"] = "azimuth"
            face["azimuth_deg"] = req.azimuth_deg
        else:
            if req.position_frac is None:
                raise ProjectError("동서/남북 단면을 위해서는 position_frac이 필요합니다.")
            try:
                face = internal_slice(self.inversion_result, req.orientation, req.position_frac, req.threshold, req.threshold_max)
            except InversionError as exc:
                raise ProjectError(str(exc)) from exc

        face["x"] = (np.asarray(face["x"]) - x0).tolist()
        face["y"] = (np.asarray(face["y"]) - y0).tolist()
        active_chi = self.inversion_result.susceptibility[self.inversion_result.susceptibility > 0]
        face["vmin"] = 0.0
        face["vmax"] = float(active_chi.max()) if active_chi.size else 1.0
        return face

    def export_inversion_npz(self) -> bytes:
        """Serialize the mesh + solved model (and enough context to
        re-render slices/sections/volume) into a self-contained .npz so a
        later session can reload the result via import_inversion_npz
        without rerunning the inversion solve."""
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        result = self.inversion_result
        mesh = result.mesh

        payload = dict(
            x_centers=mesh.x_centers,
            y_centers=mesh.y_centers,
            z_centers=mesh.z_centers,
            cell_size_m=np.array(mesh.cell_size_m),
            layer_thickness_m=np.array(mesh.layer_thickness_m),
            ground_elevation=mesh.ground_elevation,
            active=mesh.active,
            susceptibility=result.susceptibility,
            predicted_nt=result.predicted_nt,
            observed_nt=result.observed_nt,
            rms_misfit_nt=np.array(result.rms_misfit_nt),
            n_active_cells=np.array(result.n_active_cells),
            n_obs=np.array(result.n_obs),
            iterations=np.array(result.iterations),
            utm_epsg=np.array(self.utm_epsg if self.utm_epsg is not None else -1),
            inclination_deg=np.array(self.inclination_deg if self.inclination_deg is not None else np.nan),
            declination_deg=np.array(self.declination_deg if self.declination_deg is not None else np.nan),
            field_intensity_nt=np.array(self.inversion_field_intensity_nt if self.inversion_field_intensity_nt is not None else np.nan),
            value_field=np.array(self.inversion_value_field or "anomaly"),
        )
        if self.inversion_obs_grid is not None:
            g = self.inversion_obs_grid
            payload.update(
                obs_easting=g.easting,
                obs_northing=g.northing,
                obs_values=g.values,
                obs_cell_size_m=np.array(g.cell_size_m),
                obs_region=np.array(g.region, dtype=float),
            )

        buf = io.BytesIO()
        np.savez_compressed(buf, **payload)
        return buf.getvalue()

    def import_inversion_npz(self, data: bytes) -> dict:
        """Restore a previously exported inversion result (see
        export_inversion_npz) so slice/section/volume endpoints work
        immediately - no drone/base upload or processing needed first."""
        try:
            with np.load(io.BytesIO(data), allow_pickle=False) as npz:
                mesh = InversionMesh(
                    x_centers=npz["x_centers"],
                    y_centers=npz["y_centers"],
                    z_centers=npz["z_centers"],
                    cell_size_m=float(npz["cell_size_m"]),
                    layer_thickness_m=np.atleast_1d(npz["layer_thickness_m"]).astype(float),
                    ground_elevation=npz["ground_elevation"],
                    active=npz["active"],
                )
                result = InversionResult(
                    mesh=mesh,
                    susceptibility=npz["susceptibility"],
                    predicted_nt=npz["predicted_nt"],
                    observed_nt=npz["observed_nt"],
                    rms_misfit_nt=float(npz["rms_misfit_nt"]),
                    n_active_cells=int(npz["n_active_cells"]),
                    n_obs=int(npz["n_obs"]),
                    iterations=int(npz["iterations"]),
                )
                utm_epsg = int(npz["utm_epsg"])
                inclination_deg = float(npz["inclination_deg"])
                declination_deg = float(npz["declination_deg"])
                field_intensity_nt = float(npz["field_intensity_nt"])
                value_field = str(npz["value_field"])

                obs_grid = None
                if "obs_easting" in npz:
                    obs_grid = GridResult(
                        values=npz["obs_values"],
                        easting=npz["obs_easting"],
                        northing=npz["obs_northing"],
                        cell_size_m=float(npz["obs_cell_size_m"]),
                        region=tuple(npz["obs_region"].tolist()),
                    )
        except Exception as exc:  # noqa: BLE001 - deliberately broad: validating an untrusted uploaded file
            raise ProjectError(f"역산 결과 파일을 불러올 수 없습니다 (손상되었거나 올바른 형식이 아닙니다): {exc}") from exc

        self.inversion_result = result
        self.utm_epsg = utm_epsg if utm_epsg >= 0 else None
        self.inclination_deg = None if np.isnan(inclination_deg) else inclination_deg
        self.declination_deg = None if np.isnan(declination_deg) else declination_deg
        self.inversion_field_intensity_nt = None if np.isnan(field_intensity_nt) else field_intensity_nt
        self.inversion_value_field = value_field
        self.inversion_obs_grid = obs_grid

        active_chi = result.susceptibility[result.susceptibility > 0]
        summary = {
            "n_obs": result.n_obs,
            "n_active_cells": result.n_active_cells,
            "rms_misfit_nt": result.rms_misfit_nt,
            "iterations": result.iterations,
            "n_layers": int(mesh.active.shape[2]),
            "obs_cell_size_m": mesh.cell_size_m,
            "depth_extent_m": float(np.sum(mesh.layer_thickness_m)),
            "cell_size_m": mesh.cell_size_m,
            "layer_thickness_m": [float(t) for t in mesh.layer_thickness_m],
            "elevation_range_m": [float(mesh.z_centers.min()), float(mesh.z_centers.max())],
            "field": {
                "inclination_deg": self.inclination_deg,
                "declination_deg": self.declination_deg,
                "field_intensity_nt": self.inversion_field_intensity_nt,
            },
            "susceptibility_stats": _stats(pd.Series(active_chi)) if active_chi.size else _stats(pd.Series(dtype=float)),
            "used_dem": None,
            # not recorded in an imported model file
            "regional_removal": None,
            "regional_warning": None,
            "auto_params": False,
            "source_depth_estimate_m": None,
            "resolution_warning": None,
            "imported": True,
        }
        # Kept like a fresh run's summary, so a restored or reopened project
        # still has its inversion on screen and available to the chat
        # assistant (get_inversion_summary used to answer "아직 역산을
        # 실행하지 않았습니다" after a restore that had just restored one).
        summary["remanence_check"] = self._remanence_check()
        self.inversion_summary_cache = summary
        return summary

    def export_inversion_csv(self) -> bytes:
        """Active-cell (x, y, z, susceptibility) point cloud - a portable
        format for other 3D tools (ParaView, Voxler, GIS point layers,
        even a spreadsheet) that don't understand our mesh format."""
        if self.inversion_result is None:
            raise ProjectError("역산을 먼저 실행하세요.")
        result = self.inversion_result
        mesh = result.mesh
        rows, cols, layers = np.nonzero(mesh.active)
        out = pd.DataFrame(
            {
                "easting_m": mesh.x_centers[cols],
                "northing_m": mesh.y_centers[rows],
                "elevation_m": mesh.z_centers[layers],
                "susceptibility_si": result.susceptibility[rows, cols, layers],
            }
        )
        return out.to_csv(index=False).encode("utf-8")


# Size caps for the 3D inversion mesh/observation grid. The normal
# equations are solved in "data space" (see processing/inversion.py),
# whose cost scales with n_obs^2 * n_active - these limits were picked
# from a benchmark of that solve (n_obs=3000, n_active=60000 with 6 IRLS
# iterations takes ~90s; n_obs=3500, n_active=90000 extrapolates to
# roughly 3 minutes), so a user asking for more detail still finishes in
# a bounded time rather than growing unbounded. For a large-area survey
# these caps - not just the (now fixed) depth_extent_m ceiling - can
# still force the auto-suggested cell size above the line spacing; see
# the resolution_warning field in run_inversion's response.
def _regional_warning(info: dict) -> str | None:
    """Said when the data go into the inversion sitting well off zero:
    more than a quarter of their own spread. Below that a level does
    little; above it the susceptibilities start to be about the level."""
    if info["method"] != "none":
        return None
    level, spread = info["data_median_after_nt"], info["data_std_nt"]
    if spread <= 0 or abs(level) < 0.25 * spread:
        return None
    return (
        f"역산에 들어간 자료의 중앙값이 {level:+.0f} nT로 0에서 벗어나 있습니다(자료 표준편차 {spread:.0f} nT). "
        "이 모델에는 배경 수준을 나타내는 미지수가 없어서, 그 수준까지 자화율로 설명하게 됩니다 - "
        "자화율 크기가 부풀고 실제로 없는 암체가 생길 수 있습니다. '광역 배경 제거'를 켜고 다시 역산하세요."
    )


N_OBS_CAP = 3500


N_ACTIVE_CAP = 90000


def _resolution_warning(obs_cell_size_m: float, line_spacing_m: float | None, nx: int, ny: int) -> str | None:
    """A cell size coarser than the flight-line spacing means the mesh is
    under-using the resolution the survey actually captured (this can
    happen for large-area surveys, where the observation/mesh size caps
    force the cell size up regardless of line spacing) - flag it rather
    than silently returning a possibly-suboptimal mesh."""
    if not line_spacing_m or obs_cell_size_m <= line_spacing_m * 1.15:
        return None
    return (
        f"선택된 격자 크기({obs_cell_size_m:.0f}m)가 측선 간격({line_spacing_m:.0f}m)보다 큽니다 - "
        f"조사 영역이 넓어 계산량 상한({nx}x{ny}점 관측 격자) 때문에 해상도가 낮아진 것으로, "
        "측선 자료가 가진 만큼의 해상도를 다 살리지 못합니다. 더 세밀한 결과가 필요하면 "
        "관심 영역만 잘라서(예: 측선 일부만 남기고 나머지를 수동 제외) 다시 처리하거나, "
        "자동 설정을 끄고 격자 크기를 직접 줄여보세요(실행 시간이 늘어납니다)."
    )


def _depth_resolution_warning(depth_resolution: dict | None, z_centers: np.ndarray) -> str | None:
    """Human-readable caveat listing which depth layers the survey's own
    geometry barely constrains at all (see
    processing/inversion.py:resolution_diagnostics) - surfaced so a user
    doesn't read the deepest part of the recovered model with the same
    confidence as the well-resolved shallow part."""
    if not depth_resolution:
        return None
    poor = depth_resolution.get("poorly_resolved_layers") or []
    if not poor:
        return None
    elevations = [float(z_centers[i]) for i in poor]
    return (
        f"심도 레이어 {len(poor)}개(고도 {min(elevations):.0f}~{max(elevations):.0f}m, 대체로 더 깊은 쪽)는 "
        "이 측선 배치/고도로는 실제 민감도가 최상층 대비 5% 미만입니다 - 해당 구간의 역산 결과는 "
        "참고용으로만 보고, 얕은 구간보다 신뢰도를 낮게 두세요."
    )


def _densify_path(x: np.ndarray, y: np.ndarray, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    seg_len = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 0:
        return x, y
    n_samples = max(2, int(total / spacing_m) + 1)
    sample_dist = np.linspace(0.0, total, n_samples)
    sample_x = np.interp(sample_dist, cum, x)
    sample_y = np.interp(sample_dist, cum, y)
    return sample_x, sample_y
