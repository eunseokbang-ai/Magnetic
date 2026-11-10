"""What makes a project outlive the session: the project bundle (save,
load, the summary handed back to the screen) and the reference layers
that travel with it, plus point sampling across those layers. Part of
store.Project.

Split out of store.py in 2026-11; the methods are unchanged. They rely on
the attributes store.Project declares (this class has no state of its
own) and are only meant to be used through it.
"""
from __future__ import annotations

import io
import json
import numpy as np
import pandas as pd
import zipfile
from magcore import local_raster, local_tiles, progress
from magcore.geology_sample import GeologySampleError, sample_geotiff_at_point
from magcore.gridding import _nearest_axis_index
from magcore.height_normalization import HeightNormalization
from magcore.overlay_image import OverlayImageError, load_geotiff_overlay
from magcore.source_removal import SourceRemoval
from magcore.vector_layer import detect_vector_format, load_vector_layer
from pyproj import Transformer
from .bundle_frames import BundleFrameError, frame_from_bytes, frame_from_legacy_pickle, frame_to_bytes
from .models import DisplayBoundaryRequest, ManualSmoothRequest, ProcessParams
from .project_common import ProjectError, _as_job


class PersistenceMixin:
    def add_reference_layer(self, name: str, data: bytes) -> dict:
        """Keep a project-scoped copy of an uploaded reference GeoTIFF (in
        addition to the stateless preview endpoint the map layer manager
        uses) so the chat assistant can sample real pixel values at a
        point instead of only showing the layer as a picture."""
        if detect_vector_format(name, data) is not None and not name.lower().endswith((".tif", ".tiff")):
            layer = load_vector_layer(data, name)
            self.reference_vectors[layer["name"]] = layer
            self.mark_changed()
            return layer
        try:
            preview = load_geotiff_overlay(io.BytesIO(data), name=name)
        except OverlayImageError as exc:
            raise ProjectError(str(exc)) from exc
        self.reference_layers[name] = data
        self.mark_changed()
        return preview

    def remove_reference_layer(self, name: str) -> dict:
        self.reference_layers.pop(name, None)
        self.reference_vectors.pop(name, None)
        self.reference_layer_state = [s for s in self.reference_layer_state if s.get("name") != name]
        self.mark_changed()
        return {"removed": name}

    def set_reference_layer_state(self, layers: list[dict]) -> dict:
        """The layer manager's list as the frontend has it now (order,
        opacity, visibility, and for path-based layers the path). Uploaded
        content stays where add_reference_layer put it; this is only the
        list. Re-sending the same list does not count as a change, so an
        idle screen does not keep the autosave writing."""
        clean = []
        for item in layers:
            entry = {"type": item["type"], "name": str(item["name"]),
                     "opacity": float(item.get("opacity", 1.0)), "visible": bool(item.get("visible", True))}
            if item.get("path"):
                entry["path"] = str(item["path"])
            if item.get("scheme"):
                entry["scheme"] = str(item["scheme"])
            clean.append(entry)
        if clean != self.reference_layer_state:
            self.reference_layer_state = clean
            self.mark_changed()
        return {"n_layers": len(self.reference_layer_state)}

    def reference_layer_descriptors(self) -> list[dict]:
        """Everything the map needs to draw the project's reference layers
        again, in the saved order: uploaded GeoTIFFs re-rendered from their
        stored bytes, vector layers as parsed, and tile folders / on-disk
        GeoTIFFs registered again from their paths (ids are per server
        run). A layer that cannot come back - the folder moved, the file
        is gone - is returned with `error` instead of being dropped, so
        the screen can say which one."""
        state = list(self.reference_layer_state)
        known = {(s["type"], s["name"]) for s in state}
        # content that was uploaded but whose list entry never arrived
        state += [{"type": "image", "name": n, "opacity": 0.8, "visible": True}
                  for n in self.reference_layers if ("image", n) not in known]
        state += [{"type": "vector", "name": n, "opacity": 1.0, "visible": True}
                  for n in self.reference_vectors if ("vector", n) not in known]
        out = []
        for s in state:
            base = {"type": s["type"], "name": s["name"], "opacity": s.get("opacity", 1.0), "visible": s.get("visible", True)}
            try:
                if s["type"] == "image":
                    data = self.reference_layers.get(s["name"])
                    if data is None:
                        continue                       # the upload's project copy never landed
                    preview = load_geotiff_overlay(io.BytesIO(data), name=s["name"])
                    out.append({**base, "image_data_url": preview["image_data_url"], "bounds": preview["bounds"]})
                elif s["type"] == "vector":
                    layer = self.reference_vectors.get(s["name"])
                    if layer is None:
                        continue
                    out.append({**layer, **base})
                elif s["type"] == "tiles":
                    layer = local_tiles.register_local_tile_folder(s.get("path") or "", s["name"], s.get("scheme"))
                    south, west, north, east = layer.bounds
                    out.append({**base, "tile_layer_id": layer.id, "bounds": [[south, west], [north, east]],
                                "min_zoom": layer.min_zoom, "max_zoom": layer.max_zoom,
                                "path": str(layer.path), "scheme": layer.scheme})
                elif s["type"] == "raster":
                    layer = local_raster.register_local_raster(s.get("path") or "", s["name"])
                    south, west, north, east = layer.bounds
                    out.append({**base, "raster_layer_id": layer.id, "bounds": [[south, west], [north, east]],
                                "min_zoom": layer.min_zoom, "max_zoom": layer.max_zoom, "path": str(layer.path),
                                "has_overviews": layer.has_overviews, "size_gb": layer.size_gb, "res_m": layer.res_m})
            except (OverlayImageError, local_tiles.LocalTileError, local_raster.LocalRasterError) as exc:
                out.append({**base, "path": s.get("path"), "error": str(exc)})
        return out

    def sample_point(self, lat: float, lon: float) -> dict:
        """Look up everything this project currently knows about a single
        lat/lon: the nearest processed survey point's field values, the
        3D inversion susceptibility profile at that column (if an
        inversion has been run), and each uploaded reference layer's
        pixel value there - the grounding tool behind the chat assistant."""
        result: dict = {"lat": lat, "lon": lon}

        if self.processed is not None and self.utm_epsg is not None:
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
            x, y = transformer.transform(lon, lat)
            active = self._active_mask()
            df = self.processed.loc[active]
            if len(df):
                dist = np.hypot(df["x"].to_numpy() - x, df["y"].to_numpy() - y)
                idx = int(np.argmin(dist))
                row = df.iloc[idx]
                result["nearest_survey_point"] = {
                    "distance_m": float(dist[idx]),
                    "anomaly_nt": float(row["anomaly"]),
                    "tmi_nt": float(row["tmi"]),
                }

            if self.inversion_result is not None:
                mesh = self.inversion_result.mesh
                col = int(np.argmin(np.abs(mesh.x_centers - x)))
                row_i = int(np.argmin(np.abs(mesh.y_centers - y)))
                col_dist_m = float(np.hypot(mesh.x_centers[col] - x, mesh.y_centers[row_i] - y))
                if col_dist_m <= 2.0 * mesh.cell_size_m:
                    chi = self.inversion_result.susceptibility[row_i, col, :]
                    result["inversion_susceptibility_profile"] = {
                        "distance_from_mesh_column_m": col_dist_m,
                        "layers": [
                            {"elevation_m": float(mesh.z_centers[k]), "susceptibility_si": float(chi[k])}
                            for k in range(len(mesh.z_centers))
                        ],
                    }
                else:
                    result["inversion_susceptibility_profile"] = {"note": "역산 메쉬 범위 밖의 좌표입니다."}

        if self.reference_vectors:
            nearest = None
            for layer_name, layer in self.reference_vectors.items():
                for f in layer["features"]:
                    for la, lo in f["coordinates"]:
                        d = float(np.hypot((la - lat) * 110540.0, (lo - lon) * 111320.0 * np.cos(np.radians(lat))))
                        if nearest is None or d < nearest["distance_m"]:
                            nearest = {"layer": layer_name, "name": f["name"], "geometry": f["geometry"], "distance_m": round(d, 1),
                                       "properties": f.get("properties") or {}}
            if nearest is not None:
                result["nearest_vector_feature"] = nearest

        if self.reference_layers:
            geology = {}
            for layer_name, data in self.reference_layers.items():
                try:
                    geology[layer_name] = sample_geotiff_at_point(data, lat, lon)
                except GeologySampleError as exc:
                    geology[layer_name] = {"error": str(exc)}
            result["reference_layers"] = geology

        return result

    def sample_overlay_value(self, lat: float, lon: float) -> dict:
        """Value of the nearest cell in the most recently displayed
        grid/derivative overlay to a clicked map point - matches exactly
        what the overlay's color at that pixel represents, unlike
        sample_point() which reports the nearest raw survey point.
        Powers the map's click-to-inspect tool.

        Deliberately nearest-cell, not a bilinear blend between
        neighboring cells: the PNG overlay (see
        processing/render.py::grid_to_png_overlay) renders one flat color
        per cell with hard edges (no smoothing baked into the image
        itself), so a bilinearly-interpolated read-out could return a
        value blended from a neighboring cell that visibly differs from
        (and doesn't actually match) the exact color the user clicked on -
        most noticeable right at a sharp anomaly edge, exactly where
        getting this right matters most. Nearest-cell guarantees the
        number always matches the pixel."""
        overlay = self.last_overlay  # single read - see OverlayState's docstring on why this must stay one reference
        if overlay is None:
            raise ProjectError("먼저 그리드를 생성하세요.")
        if self.utm_epsg is None:
            raise ProjectError("좌표계 정보가 없습니다.")
        easting = overlay.easting
        northing = overlay.northing
        values = overlay.values
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
        x, y = transformer.transform(lon, lat)
        result = {"lat": lat, "lon": lon, "value_nt": None, "in_bounds": False, "label": overlay.label}
        # Half a cell of slack on each side matches the overlay image's own
        # rendered extent (see grid_to_png_overlay's "pixel is area" bounds)
        # - without it, a click inside the visually-drawn edge cell but
        # past the last cell *center* would wrongly report out-of-bounds.
        half_e = (easting[1] - easting[0]) / 2.0 if len(easting) > 1 else 0.0
        half_n = (northing[1] - northing[0]) / 2.0 if len(northing) > 1 else 0.0
        if x < easting[0] - half_e or x > easting[-1] + half_e or y < northing[0] - half_n or y > northing[-1] + half_n:
            return result
        col = int(_nearest_axis_index(easting, np.array([x]))[0])
        row = int(_nearest_axis_index(northing, np.array([y]))[0])
        value = float(values[row, col])
        if not np.isfinite(value):
            return result
        result["value_nt"] = value
        result["in_bounds"] = True
        return result

    def save_project_bundle(self) -> bytes:
        """Zip up everything needed to resume this project later without
        re-uploading raw files: the raw (already-parsed) drone/base
        dataframes, the last-used processing params, manual overrides,
        DEM, and inversion result if present. Re-loading replays
        run_pipeline(last_params) on the raw data rather than trying to
        serialize every derived column/scalar, which keeps this in sync
        with the pipeline logic for free."""
        if self.drone_raw is None:
            raise ProjectError("저장할 자료가 없습니다 (드론 자료를 먼저 업로드하세요).")

        # Everything the bundle describes is read at one moment, under the
        # same lock the mutating methods take.
        with self._edit_lock:
            drone_raw = self.drone_raw
            base_raw = self.base_raw
            calibration_raw = self.calibration_raw
            meta = self._bundle_meta()
            dem_bytes = self.dem_bytes
            inversion_npz = self.export_inversion_npz() if self.inversion_result is not None else None
            # same order as meta["reference_files"] (dict order is insertion order)
            reference_files = {f"reference/{i}.tif": data for i, data in enumerate(self.reference_layers.values())}
            reference_vectors = dict(self.reference_vectors)
        return _write_bundle(drone_raw, base_raw, calibration_raw, meta, dem_bytes, inversion_npz,
                             reference_files, reference_vectors)

    def _bundle_meta(self) -> dict:
        meta = {
            "last_params": self.last_params.model_dump() if self.last_params is not None else None,
            "manual_overrides": {str(k): v for k, v in self.manual_overrides.items()},
            "manual_smooth_point_ids": sorted(int(p) for p in self.manual_smooth_point_ids),
            "source_removals": [r.to_dict() for r in self.source_removals],
            "candidate_notes": self.candidate_notes,
            "repeat_pass_offsets": {str(k): float(v) for k, v in self.repeat_pass_offsets.items()},
            "height_normalization": (
                self.height_normalization.to_dict() if self.height_normalization is not None else None
            ),
            "display_boundary_polygon": self.display_boundary_polygon,
            "dem_name": self.dem_name,
            "has_base": self.base_raw is not None,
            "geology_units": self.geology_units,
            "geology_unit_next_id": self.geology_unit_next_id,
            # Reference layers (12. 참조 레이어): the list as shown, and
            # which bundle member holds each uploaded GeoTIFF. Vector layers
            # are in reference_vectors.json; tile folders and on-disk
            # GeoTIFFs are paths in the list, registered again on load.
            "reference_layer_state": self.reference_layer_state,
            "reference_files": {name: f"reference/{i}.tif" for i, name in enumerate(self.reference_layers)},
        }
        return meta

    def load_project_bundle(self, data: bytes) -> dict:
        with self._edit_lock:
            return self._load_project_bundle_unlocked(data)

    @_as_job("프로젝트 불러오기", rollback=True)
    def _load_project_bundle_unlocked(self, data: bytes) -> dict:
        """Restore a project saved by save_project_bundle. Returns the
        same shape as process_summary() (plus a couple of extra flags) so
        the frontend can jump straight back to where the user left off."""
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = set(zf.namelist())
                if "meta.json" not in names or not ({"drone_raw.npz", "drone_raw.pkl", "drone_raw.csv"} & names):
                    raise ProjectError("올바른 프로젝트 저장 파일이 아닙니다.")

                progress.report("측점 자료 읽기", 0.02)
                self.drone_raw = _read_frame(zf, names, "drone_raw")
                self.base_raw = _read_frame(zf, names, "base_raw")
                self.calibration_raw = _read_frame(zf, names, "calibration_raw")

                meta = json.loads(zf.read("meta.json").decode("utf-8"))

                if "dem.tif" in names:
                    self.dem_bytes = zf.read("dem.tif")
                    self.dem_name = meta.get("dem_name")
                else:
                    self.dem_bytes = None
                    self.dem_name = None

                self.geology_units = meta.get("geology_units") or []
                self.geology_unit_next_id = meta.get("geology_unit_next_id", len(self.geology_units) + 1)

                # Reference layers. A bundle from before they were saved has
                # no key at all: then whatever the screen shows is left alone
                # (has_reference_layers False) rather than cleared.
                has_reference_layers = "reference_layer_state" in meta
                if has_reference_layers:
                    self.reference_layer_state = list(meta.get("reference_layer_state") or [])
                    self.reference_layers = {
                        name: zf.read(member) for name, member in (meta.get("reference_files") or {}).items() if member in names
                    }
                    self.reference_vectors = (
                        json.loads(zf.read("reference_vectors.json").decode("utf-8")) if "reference_vectors.json" in names else {}
                    )

                processed = False
                last_params = meta.get("last_params")
                # A saved "베이스 자료 없이 진행" (assume_constant) bundle
                # legitimately has no base_raw at all - requiring one here
                # meant such a project silently reloaded as unprocessed
                # (blank map, cleared summary) even though it was saved in
                # a perfectly valid, already-processed state.
                can_replay = last_params is not None and (
                    self.base_raw is not None
                    or (last_params.get("diurnal_params") or {}).get("mode") == "assume_constant"
                )
                if can_replay:
                    params = ProcessParams(**last_params)
                    with progress.span(0.10, 0.80):
                        self.run_pipeline(params)
                    progress.report("편집 내용 복원", 0.82)
                    overrides = meta.get("manual_overrides") or {}
                    if overrides:
                        self.manual_overrides = {int(k): v for k, v in overrides.items()}
                        self.grid_cache = {}
                        self.transform_cache = {}
                    # Stored as fitted models, not refitted: the result is
                    # exactly what was saved.
                    self.source_removals = [
                        SourceRemoval.from_dict(d) for d in (meta.get("source_removals") or [])
                    ]
                    self.repeat_pass_offsets = {
                        int(k): float(v) for k, v in (meta.get("repeat_pass_offsets") or {}).items()
                    }
                    # Likewise the fitted layers, not a refit.
                    saved_hn = meta.get("height_normalization")
                    self.height_normalization = HeightNormalization.from_dict(saved_hn) if saved_hn else None
                    self._height_correction_cache = None
                    # The operator's structure-or-geology calls. Kept
                    # whatever the processing did - they are about what is
                    # on the ground, not about this run's settings.
                    self.candidate_notes = list(meta.get("candidate_notes") or [])
                    smooth_ids = meta.get("manual_smooth_point_ids") or []
                    if smooth_ids:
                        self.set_manual_smoothing(ManualSmoothRequest(mode="point_ids", point_ids=smooth_ids))
                    elif self.source_removals or self.height_normalization is not None:
                        self._rebuild_processed()
                    # Restore the saved boundary state exactly, including
                    # "none". The run_pipeline replay above regenerates an
                    # automatic boundary (ProcessParams.auto_display_boundary
                    # defaults on), which would otherwise both overwrite a
                    # hand-drawn boundary and resurrect one the user had
                    # deliberately cleared before saving. Keyed on the key's
                    # presence, not its truthiness, so a save file old enough
                    # to predate the field keeps the new default instead of
                    # being forced to "no boundary".
                    if "display_boundary_polygon" in meta:
                        self.set_display_boundary(
                            DisplayBoundaryRequest(polygon=meta.get("display_boundary_polygon"))
                        )
                    processed = True

                progress.report("역산 결과 복원", 0.92)
                if "inversion.npz" in names and processed:
                    self.import_inversion_npz(zf.read("inversion.npz"))
        except zipfile.BadZipFile as exc:
            raise ProjectError(f"프로젝트 파일을 열 수 없습니다 (손상되었거나 zip 형식이 아닙니다): {exc}") from exc
        except BundleFrameError as exc:
            raise ProjectError(str(exc)) from exc
        except KeyError as exc:
            raise ProjectError(f"프로젝트 파일 내용이 올바르지 않습니다: {exc}") from exc

        result = self.session_summary()
        if not has_reference_layers:
            # this bundle predates saved reference layers: keep what is on screen
            result["reference_layers"] = None
        self.mark_changed()
        return result

    def session_summary(self) -> dict:
        """Everything the screen needs to pick this project up where it
        is now: what a bundle load answers with, and what reopening a
        project that is still in memory answers with (a browser refresh,
        a second tab) - so both paths put the same state on screen."""
        processed = self.processed is not None
        inversion_summary = self.inversion_summary_cache if self.inversion_result is not None else None
        result = {
            "drone_summary": self.drone_summary(),
            "base_summary": self.base_summary(),
            "processed": processed,
            "inversion_restored": inversion_summary is not None,
            "inversion_summary": inversion_summary or None,
            "params": self.last_params.model_dump() if self.last_params is not None else None,
            "reference_layers": self.reference_layer_descriptors(),
        }
        if processed:
            result["process_summary"] = self.process_summary()
        return result


def _write_bundle(drone_raw, base_raw, calibration_raw, meta, dem_bytes, inversion_npz,
                  reference_files: dict | None = None, reference_vectors: dict | None = None) -> bytes:
    """The project as one zip. The readings go in as binary column arrays
    (bundle_frames.py: an npz read with allow_pickle=False), not CSV:
    measured on the 74-file HaeNam block (965k rows), CSV took 6.9 s to
    write and the bundle 13.9 s, and the autosave ran that every two
    minutes with the GIL held, so the screen stalled with it. They are not
    pickled frames either (the format from 2026-09 to 2026-11): a pickle
    runs the code it names when opened, which a file people pass around
    must not. Bundles written as CSV or as pickle by earlier versions are
    still read (_read_frame) - the pickles through an allowlist."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.writestr("drone_raw.npz", frame_to_bytes(drone_raw))
        if base_raw is not None:
            zf.writestr("base_raw.npz", frame_to_bytes(base_raw))
        if calibration_raw is not None:
            zf.writestr("calibration_raw.npz", frame_to_bytes(calibration_raw))
        zf.writestr("meta.json", json.dumps(meta))
        if dem_bytes is not None:
            zf.writestr("dem.tif", dem_bytes)
        if inversion_npz is not None:
            zf.writestr("inversion.npz", inversion_npz)
        # Uploaded reference GeoTIFFs go in stored, not deflated: they are
        # usually compressed already, and a second pass over tens of MB is
        # time the autosave would spend for nothing.
        for member, data in (reference_files or {}).items():
            zf.writestr(zipfile.ZipInfo(member), data, compress_type=zipfile.ZIP_STORED)
        if reference_vectors:
            zf.writestr("reference_vectors.json", json.dumps(reference_vectors, ensure_ascii=False))
    return buf.getvalue()


def _read_frame(zf: zipfile.ZipFile, names: set, stem: str) -> pd.DataFrame | None:
    """A frame from a bundle, whichever way it was written: column arrays
    by this version, a pickle (read through LegacyFrameUnpickler's
    allowlist, never unpickled freely) or CSV by the versions before it."""
    if f"{stem}.npz" in names:
        return frame_from_bytes(zf.read(f"{stem}.npz"))
    if f"{stem}.pkl" in names:
        return frame_from_legacy_pickle(zf.read(f"{stem}.pkl"))
    if f"{stem}.csv" in names:
        frame = pd.read_csv(io.BytesIO(zf.read(f"{stem}.csv")))
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        return frame
    return None
