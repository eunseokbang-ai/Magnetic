"""Interpretation products computed from the grids: Euler deconvolution,
edges, contacts, lineaments, depth estimates, prospectivity, target
detection, structure and anomaly-candidate scans, geology statistics and
the operator's candidate notes. Part of store.Project.

Split out of store.py in 2026-11; the methods are unchanged. They rely on
the attributes store.Project declares (this class has no state of its
own) and are only meant to be used through it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import time
from magcore.anomaly_candidates import SUPPORT_RADIUS_LINE_SPACINGS, find_anomaly_candidates
from magcore.contacts import detect_magnetic_contacts
from magcore.depth_estimation import (
    analytic_signal_depth_estimates,
    spectral_depth_diagnostic,
    tilt_depth_estimates,
)
from magcore.dipole_fit import classify_moment, detect_targets
from magcore.euler_deconvolution import run_euler_deconvolution as _euler_deconvolution_solve
from magcore.geology_stats import (
    GeologyServiceError,
    fetch_lithology,
    stats_to_csv as geology_stats_to_csv,
    summarize_by_unit,
    units_geojson,
)
from magcore.gridding import grid_points
from magcore.lineaments import extract_lineaments
from magcore.magnetization import estimate_magnetization
from magcore.multiscale_edges import run_multiscale_edges as _multiscale_edges_solve
from magcore.osm_structures import OsmFetchError, buffer_structures_to_polygons, fetch_osm_structures
from magcore.prospectivity import compute_prospectivity
from magcore.render import grid_to_png_overlay
from magcore.spectrum import compute_power_spectrum
from magcore.transforms import (
    analytic_signal,
    reduction_to_pole,
    tilt_angle,
    total_horizontal_derivative,
    vertical_derivative,
)
from magcore.trend import remove_regional_trend
from matplotlib.path import Path as MplPath
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from .models import (
    AnalyticSignalDepthRequest,
    AnomalyCandidateRequest,
    ContactDetectionRequest,
    DEFAULT_GRID_METHOD,
    EulerDeconvolutionRequest,
    GeologyStatsRequest,
    LineamentRequest,
    MultiscaleEdgeRequest,
    PowerSpectrumRequest,
    ProspectivityRequest,
    SpectralDepthRequest,
    StructureScanRequest,
    TargetDetectionRequest,
    TiltDepthRequest,
)
from .project_common import ProjectError, _as_job, _stats
from .models import CandidateNoteRequest


from magcore.uncertainty import derived_confidence, window_mean  # noqa: E402
from .project_grids import LOW_CONFIDENCE  # noqa: E402


class InterpretationMixin:
    def export_targets_csv(self) -> bytes:
        """The last run_target_detection() result (self.target_summary_cache)
        as a flat CSV table - lat/lon plus every dipole-fit attribute - for
        handing a detected-target list to a GIS or spreadsheet without the
        map UI. Empty (header-only) rather than an error when detection
        found nothing, since "ran clean, zero targets" is a valid result a
        caller may still want a well-formed CSV for."""
        if self.target_summary_cache is None:
            raise ProjectError("타겟 탐지를 먼저 실행하세요.")
        targets = self.target_summary_cache.get("targets", [])
        columns = ["lat", "lon", "depth_m", "moment_am2", "size_class", "peak_anomaly_nt", "footprint_m", "fit_quality", "background_nt"]
        out = pd.DataFrame(targets, columns=columns) if targets else pd.DataFrame(columns=columns)
        out.insert(0, "target_id", range(1, len(out) + 1))
        return out.to_csv(index=False).encode("utf-8")

    def export_targets_shapefile(self) -> bytes:
        """The same detected-target list as export_targets_csv, packaged as
        a zipped ESRI shapefile (.shp/.shx/.dbf/.prj point layer) for direct
        import into GIS software - the standard exchange format the
        "coverage/target handoff to GIS" use case actually wants, CSV alone
        requiring a manual re-projection/import step every time."""
        import io
        import zipfile

        import shapefile as pyshp

        if self.target_summary_cache is None:
            raise ProjectError("타겟 탐지를 먼저 실행하세요.")
        targets = self.target_summary_cache.get("targets", [])

        shp_buf, shx_buf, dbf_buf = io.BytesIO(), io.BytesIO(), io.BytesIO()
        writer = pyshp.Writer(shp=shp_buf, shx=shx_buf, dbf=dbf_buf, shapeType=pyshp.POINT)
        writer.field("target_id", "N")
        writer.field("depth_m", "F", decimal=2)
        writer.field("moment_am2", "F", decimal=4)
        writer.field("size_class", "C", size=40)
        writer.field("peak_nt", "F", decimal=2)
        # dbf field names are capped at 10 bytes - "footprint_m"/"fit_quality"
        # would silently truncate (and collide with nothing here, but it's
        # a latent footgun for any future field starting the same way).
        writer.field("footprnt_m", "F", decimal=2)
        writer.field("fit_qual", "F", decimal=3)
        writer.field("bg_nt", "F", decimal=2)
        for i, t in enumerate(targets, start=1):
            writer.point(t["lon"], t["lat"])
            writer.record(i, t["depth_m"], t["moment_am2"], t["size_class"], t["peak_anomaly_nt"], t["footprint_m"], t["fit_quality"], t["background_nt"])
        writer.close()

        # WGS84 geographic .prj - point() above was fed lon/lat, matching
        # this CRS, so the shapefile's coordinates and declared CRS agree
        # regardless of which UTM zone the project itself works in.
        prj_wkt = (
            'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("targets.shp", shp_buf.getvalue())
            zf.writestr("targets.shx", shx_buf.getvalue())
            zf.writestr("targets.dbf", dbf_buf.getvalue())
            zf.writestr("targets.prj", prj_wkt)
        return zip_buf.getvalue()

    @_as_job("오일러 디컨볼루션")
    def run_euler_deconvolution(self, req: EulerDeconvolutionRequest) -> dict:
        # Defaults to along-line smoothing on (see _grid_for) - Euler
        # solutions are as sensitive to corrugation-driven spurious
        # gradients as any other derivative-based analysis, and this
        # request type has no UI toggle of its own for it.
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)

        altitude_grid = None
        if req.flight_agl_m is not None:
            if self.processed is None:
                raise ProjectError("자료 처리를 먼저 실행하세요.")
            df = self.processed
            active = self._active_mask()
            # Gridded with the exact same points/cell size/max_distance as
            # the anomaly grid above so the two line up cell-for-cell (same
            # pattern run_inversion uses for its obs_grid/alt_grid pair).
            altitude_grid = grid_points(
                df.loc[active, "x"].to_numpy(),
                df.loc[active, "y"].to_numpy(),
                df.loc[active, "altitude_ellipsoidal_m"].to_numpy(),
                grid.cell_size_m,
                method=req.method,
                max_distance_m=req.max_distance_m,
                line_id=df.loc[active, "line_id"].to_numpy(),
                typical_line_spacing_m=self.line_spacing_m,
            ).values

        solutions = _euler_deconvolution_solve(
            grid.values,
            grid.easting,
            grid.northing,
            grid.cell_size_m,
            self.utm_epsg,
            structural_index=req.structural_index,
            window_size_m=req.window_size_m,
            max_depth_uncertainty_pct=req.max_depth_uncertainty_pct,
            altitude_grid=altitude_grid,
            flight_agl_m=req.flight_agl_m,
        )
        depths = [s.depth_m for s in solutions]
        confidence = self._euler_confidence(grid, solutions, req)
        summary = {
            "n_solutions": len(solutions),
            # the method assumes a level observation plane - said when the survey is not on one
            "altitude_warning": self._altitude_warning(),
            # how many solutions sit where the derivatives they were solved from rest on interpolation
            "n_low_confidence": int(sum(1 for c in confidence if c is not None and c < LOW_CONFIDENCE)),
            "low_confidence_threshold": LOW_CONFIDENCE,
            "structural_index": req.structural_index,
            "depth_reference": solutions[0].depth_reference if solutions else (
                "ground_surface" if req.flight_agl_m is not None else "flat_datum"
            ),
            "depth_stats": _stats(pd.Series(depths)) if depths else None,
            "solutions": [
                {
                    "lat": s.lat,
                    "lon": s.lon,
                    "depth_m": s.depth_m,
                    "base_level_nt": s.base_level_nt,
                    "uncertainty_m": s.uncertainty_m,
                    "confidence": c,
                }
                for s, c in zip(solutions, confidence)
            ],
        }
        self.euler_summary_cache = summary
        return summary

    def _euler_confidence(self, grid, solutions, req) -> list:
        """Support score of each Euler solution: the grid's confidence
        carried through the three first derivatives the method solves
        from, averaged over the solution's window, read at the solution.
        A solution in a gap between lines gets a low score here even when
        its own least-squares uncertainty is small - the window fitted
        interpolated gradients very consistently."""
        if not solutions:
            return []
        try:
            base = self._grid_confidence(grid, req.max_distance_m)
            derivative = derived_confidence(base, self._transform_components("as", req, grid.cell_size_m))
            window = window_mean(derivative, int(round(req.window_size_m / grid.cell_size_m)))
            to_xy = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
            x, y = to_xy.transform([s.lon for s in solutions], [s.lat for s in solutions])
            ix = np.clip(np.round((np.asarray(x) - grid.easting[0]) / grid.cell_size_m).astype(int), 0, len(grid.easting) - 1)
            iy = np.clip(np.round((np.asarray(y) - grid.northing[0]) / grid.cell_size_m).astype(int), 0, len(grid.northing) - 1)
            values = window[iy, ix]
            return [None if not np.isfinite(v) else round(float(v), 3) for v in values]
        except Exception:  # noqa: BLE001 - the score is an annotation; the solutions stand without it
            return [None] * len(solutions)

    def run_multiscale_edges(self, req: MultiscaleEdgeRequest) -> dict:
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        points = _multiscale_edges_solve(
            grid.values,
            grid.easting,
            grid.northing,
            grid.cell_size_m,
            self.utm_epsg,
            heights_m=req.heights_m,
            percentile=req.percentile,
        )
        summary = {
            "n_points": len(points),
            "heights_m": req.heights_m,
            "percentile": req.percentile,
            "points": [
                {"lat": p.lat, "lon": p.lon, "height_m": p.height_m, "thdr_value": p.thdr_value} for p in points
            ],
        }
        self.multiscale_edges_summary_cache = summary
        return summary

    def run_magnetic_contact_detection(self, req: ContactDetectionRequest) -> dict:
        """Magnetic contact detection - see processing/contacts.py. Reuses
        the same multi-height THDR worming as run_multiscale_edges, but
        keeps only ridge points that persist across several heights and
        vectorizes them into contact segments (a proxy for geological
        contacts/rock-unit boundaries), meant to be compared against an
        uploaded geological-map reference layer."""
        if self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        result = detect_magnetic_contacts(
            grid.values,
            grid.easting,
            grid.northing,
            grid.cell_size_m,
            self.utm_epsg,
            heights_m=tuple(req.heights_m),
            percentile_threshold=req.percentile_threshold,
            min_persistence=req.min_persistence,
            min_segment_points=req.min_segment_points,
            min_length_m=req.min_length_m,
            max_gap_cells=req.max_gap_cells,
        )
        self.contact_summary_cache = result
        return result

    def run_lineament_extraction(self, req: LineamentRequest) -> dict:
        """Magnetic lineament extraction + rose-diagram structural
        statistics - see processing/lineaments.py. Along-line smoothing
        stays on (the default) for the underlying grid: lineament
        extraction reads a derivative of that grid, which would otherwise
        amplify any flight-line-parallel corrugation into spurious
        "lineaments" running exactly along the survey's own line
        direction."""
        if self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        source_grid = {
            "thd": lambda: total_horizontal_derivative(grid.values, grid.cell_size_m),
            "as": lambda: analytic_signal(grid.values, grid.cell_size_m),
            "tilt": lambda: tilt_angle(grid.values, grid.cell_size_m),
            "1vd": lambda: vertical_derivative(grid.values, grid.cell_size_m, order=1),
        }[req.source]()
        result = extract_lineaments(
            source_grid,
            grid.easting,
            grid.northing,
            grid.cell_size_m,
            self.utm_epsg,
            percentile_threshold=req.percentile_threshold,
            min_segment_points=req.min_segment_points,
            min_length_m=req.min_length_m,
            rose_bin_width_deg=req.rose_bin_width_deg,
            max_gap_cells=req.max_gap_cells,
        )
        result["source"] = req.source
        self.lineament_summary_cache = result
        return result

    def run_tilt_depth(self, req: TiltDepthRequest) -> dict:
        if self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        result = tilt_depth_estimates(
            grid.values, grid.easting, grid.northing, grid.cell_size_m, self.utm_epsg,
            min_depth_m=req.min_depth_m, max_depth_m=req.max_depth_m,
        )
        self.tilt_depth_cache = result
        return result

    def run_analytic_signal_depth(self, req: AnalyticSignalDepthRequest) -> dict:
        if self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        result = analytic_signal_depth_estimates(
            grid.values, grid.easting, grid.northing, grid.cell_size_m, self.utm_epsg,
            percentile_threshold=req.percentile_threshold, search_radius_cells=req.search_radius_cells,
        )
        self.as_depth_cache = result
        return result

    def run_spectral_depth(self, req: SpectralDepthRequest) -> dict:
        if self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        result = spectral_depth_diagnostic(grid.values, grid.cell_size_m)
        self.spectral_depth_cache = result
        return result

    def _near_surface_susceptibility_grid(self, easting: np.ndarray, northing: np.ndarray) -> np.ndarray | None:
        """Average susceptibility over the shallowest 2 layers of the most
        recent 3D inversion (index 0 = shallowest, see processing/
        inversion.py), resampled from the (coarse) inversion mesh onto the
        current display grid's easting/northing via linear interpolation -
        the same coarse-grid-and-interpolate pattern processing/igrf.py
        uses for IGRF, since the inversion mesh is far coarser than a
        typical display grid. Returns None (rather than raising) when no
        inversion has been run yet - susceptibility is an optional
        prospectivity layer, not a required one."""
        if self.inversion_result is None:
            return None
        mesh = self.inversion_result.mesh
        n_layers = min(2, mesh.z_centers.size)
        near_surface = np.nanmean(self.inversion_result.susceptibility[:, :, :n_layers], axis=2)
        if mesh.x_centers.size < 2 or mesh.y_centers.size < 2:
            return None
        interpolator = RegularGridInterpolator(
            (mesh.y_centers, mesh.x_centers), near_surface, bounds_error=False, fill_value=np.nan
        )
        x2d, y2d = np.meshgrid(easting, northing)
        return interpolator(np.column_stack([y2d.ravel(), x2d.ravel()])).reshape(x2d.shape)

    def run_prospectivity(self, req: ProspectivityRequest) -> dict:
        """Rule-based mineral prospectivity ("target score") mapping - see
        processing/prospectivity.py. Combines ASA/THD with (if already
        computed) structural-lineament and magnetic-contact proximity, and
        (if a 3D inversion has been run) near-surface susceptibility, into
        one weighted 0-1 score grid, then reports ranked, explained
        local-maximum targets."""
        if self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        asa_grid = analytic_signal(grid.values, grid.cell_size_m)
        thd_grid = total_horizontal_derivative(grid.values, grid.cell_size_m)
        finite_mask = np.isfinite(grid.values)

        lineament_points = None
        if self.lineament_summary_cache and self.lineament_summary_cache.get("available"):
            lineament_points = [p for l in self.lineament_summary_cache["lineaments"] for p in l["points_latlon"]]
        contact_points = None
        if self.contact_summary_cache and self.contact_summary_cache.get("available"):
            contact_points = [p for c in self.contact_summary_cache["contacts"] for p in c["points_latlon"]]
        susceptibility_grid = self._near_surface_susceptibility_grid(grid.easting, grid.northing)

        result = compute_prospectivity(
            asa_grid,
            thd_grid,
            finite_mask,
            grid.easting,
            grid.northing,
            self.utm_epsg,
            lineament_points_latlon=lineament_points,
            contact_points_latlon=contact_points,
            susceptibility_grid=susceptibility_grid,
            purpose=req.purpose,
            custom_weights=req.weights,
            decay_length_m=req.decay_length_m,
            score_threshold=req.score_threshold,
            max_targets=req.max_targets,
            min_target_separation_m=req.min_target_separation_m,
        )
        score_grid = result.pop("score_grid", None)
        if score_grid is not None:
            self.prospectivity_score_grid = score_grid
            self.prospectivity_score_easting = grid.easting
            self.prospectivity_score_northing = grid.northing
            self.prospectivity_score_cell_size_m = grid.cell_size_m
        self.prospectivity_summary_cache = result
        return result

    def get_prospectivity_overlay(self, colormap: str = "Viridis") -> dict:
        """PNG overlay of the most recent run_prospectivity() score grid -
        same rendering pattern as get_grid_confidence_overlay, since the
        0-1 score grid isn't a value the generic get_grid_overlay/
        get_transform_overlay cache keys already cover."""
        if self.prospectivity_score_grid is None:
            raise ProjectError("프로스펙티비티 분석을 먼저 실행하세요.")
        overlay = grid_to_png_overlay(
            self.prospectivity_score_grid,
            self.prospectivity_score_easting,
            self.prospectivity_score_northing,
            self.utm_epsg,
            cmap_name=colormap,
            symmetric=False,
            vmin=0.0,
            vmax=1.0,
            cell_size_m=self.prospectivity_score_cell_size_m,
        )
        overlay["stats"] = _stats(pd.Series(self.prospectivity_score_grid.ravel()))
        overlay["cell_size_m"] = self.prospectivity_score_cell_size_m
        return overlay

    def get_power_spectrum(self, req: PowerSpectrumRequest) -> dict:
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed
        line_df = df[df["line_id"] == req.line_id].sort_values("timestamp")
        if line_df.empty:
            raise ProjectError(f"측선 {req.line_id}을 찾을 수 없습니다.")
        return compute_power_spectrum(line_df[req.value].to_numpy(), line_df["timestamp"])

    def run_target_detection(self, req: TargetDetectionRequest) -> dict:
        """Near-surface compact-target detection (mines, buried ordnance,
        hidden vehicles): grids the magnetic anomaly at a fine cell size,
        picks out localized (non-regional) local-maximum anomalies, and
        fits each one to a single induced-magnetic-dipole model - see
        processing/dipole_fit.py for the physics and the caveats around
        what a dipole-moment size class can and cannot tell you."""
        if self.processed is None or self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")

        active = self._active_mask()
        df = self.processed.loc[active]
        x_span = float(df["x"].max() - df["x"].min())
        y_span = float(df["y"].max() - df["y"].min())
        est_cells = (x_span / req.cell_size_m + 1) * (y_span / req.cell_size_m + 1)
        if est_cells > TARGET_DETECTION_GRID_CELL_CAP:
            raise ProjectError(
                f"탐지 격자가 너무 촘촘합니다 (예상 셀 수 약 {int(est_cells):,}개, 상한 {TARGET_DETECTION_GRID_CELL_CAP:,}개). "
                "탐지 격자 크기(m)를 늘리거나, 폴리곤으로 관심 영역만 남기고 나머지 측선을 제외한 뒤 다시 시도하세요."
            )

        # Along-line smoothing is deliberately off here: it low-passes at a
        # wavelength matched to the (much larger) line spacing, which would
        # blur out exactly the small, compact, near-surface anomalies
        # (mines/ordnance/vehicles) this feature exists to find - unlike
        # every other _grid_for caller, this one wants full native
        # along-line resolution preserved.
        grid = self._grid_for("anomaly", req.cell_size_m, req.method, req.max_distance_m, along_line_smooth=False)

        if req.amplitude_threshold_nt is not None:
            threshold_nt = req.amplitude_threshold_nt
        else:
            finite = grid.values[np.isfinite(grid.values)]
            robust_std = float(1.4826 * np.median(np.abs(finite - np.median(finite)))) if finite.size else 0.0
            threshold_nt = max(req.threshold_k * robust_std, 1e-6)

        targets = detect_targets(
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            df["anomaly"].to_numpy(),
            grid.values,
            grid.easting,
            grid.northing,
            grid.cell_size_m,
            self.inclination_deg,
            self.declination_deg,
            threshold_nt=threshold_nt,
            min_footprint_m=req.min_footprint_m,
            max_footprint_m=req.max_footprint_m,
            fit_window_m=req.fit_window_m,
            max_depth_m=req.max_depth_m,
            min_fit_quality=req.min_fit_quality,
        )

        transformer = Transformer.from_crs(f"EPSG:{self.utm_epsg}", "EPSG:4326", always_xy=True)
        target_dicts = []
        if targets:
            lons, lats = transformer.transform([t.x for t in targets], [t.y for t in targets])
        else:
            lons, lats = [], []
        for t, lat, lon in zip(targets, lats, lons):
            target_dicts.append(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "depth_m": t.depth_m,
                    "moment_am2": t.moment_am2,
                    "size_class": classify_moment(t.moment_am2),
                    "peak_anomaly_nt": t.peak_anomaly_nt,
                    "footprint_m": t.footprint_m,
                    "fit_quality": t.fit_quality,
                    "background_nt": t.background_nt,
                }
            )
        # sort strongest/most-confident first, since these lists are most
        # useful read top-down as a triage order rather than in scan order.
        target_dicts.sort(key=lambda d: -d["fit_quality"])

        summary = {
            "n_targets": len(target_dicts),
            "amplitude_threshold_nt": threshold_nt,
            "cell_size_m": grid.cell_size_m,
            "targets": target_dicts,
        }
        self.target_summary_cache = summary
        return summary

    @_as_job("지상구조물 왜곡 탐지")
    def scan_structure_distortion(self, req: StructureScanRequest) -> dict:
        """Combine an OpenStreetMap building/road location prior with the
        same compact-anomaly signal detector used for near-surface target
        detection (run_target_detection) to auto-generate the regions the
        "지도에서 왜곡 영역 그려 스무딩" tool would otherwise need one
        hand-drawn polygon per structure for - see
        processing/osm_structures.py.

        Returns both the buffered structure polygons (regardless of
        whether a signal anomaly matched them - a subtle building without
        a peak reaching the detection threshold is still worth a visual
        check) and the compact anomalies the signal-only side found, each
        tagged with whether it falls inside a mapped structure's buffered
        region - lets the caller show "this spike lines up with a mapped
        building" separately from "no mapped structure here at all, worth
        a closer look (unmapped structure, or possibly real geology)"
        instead of blindly trusting either signal alone."""
        if self.processed is None or self.inclination_deg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        if self.utm_epsg is None:
            raise ProjectError("좌표계 정보가 없습니다.")

        active = self._active_mask()
        df = self.processed.loc[active]
        if df.empty:
            raise ProjectError("유효한 측선 포인트가 없습니다.")

        lat_min, lat_max = float(df["lat"].min()), float(df["lat"].max())
        lon_min, lon_max = float(df["lon"].min()), float(df["lon"].max())
        margin_m = 50.0
        lat_margin = margin_m / 111_000.0
        lon_margin = margin_m / (111_000.0 * max(np.cos(np.radians((lat_min + lat_max) / 2)), 0.1))

        try:
            structures = fetch_osm_structures(
                lat_min - lat_margin, lon_min - lon_margin, lat_max + lat_margin, lon_max + lon_margin
            )
        except OsmFetchError as exc:
            raise ProjectError(str(exc)) from exc

        structure_polygons = buffer_structures_to_polygons(
            structures, self.utm_epsg, req.building_buffer_m, req.road_buffer_m
        )

        x_span = float(df["x"].max() - df["x"].min())
        y_span = float(df["y"].max() - df["y"].min())
        est_cells = (x_span / req.cell_size_m + 1) * (y_span / req.cell_size_m + 1)
        if est_cells > TARGET_DETECTION_GRID_CELL_CAP:
            raise ProjectError(
                f"탐지 격자가 너무 촘촘합니다 (예상 셀 수 약 {int(est_cells):,}개, 상한 {TARGET_DETECTION_GRID_CELL_CAP:,}개). "
                "탐지 격자 크기(m)를 늘리거나, 폴리곤으로 관심 영역만 남기고 나머지 측선을 제외한 뒤 다시 시도하세요."
            )

        # Along-line smoothing off, same reasoning as run_target_detection:
        # this hunts for compact, localized anomalies, which a low-pass
        # tuned to the (much larger) line spacing would blur away.
        grid = self._grid_for("anomaly", req.cell_size_m, req.method, req.max_distance_m, along_line_smooth=False)

        if req.amplitude_threshold_nt is not None:
            threshold_nt = req.amplitude_threshold_nt
        else:
            finite = grid.values[np.isfinite(grid.values)]
            robust_std = float(1.4826 * np.median(np.abs(finite - np.median(finite)))) if finite.size else 0.0
            threshold_nt = max(req.threshold_k * robust_std, 1e-6)

        candidates = detect_targets(
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            df["anomaly"].to_numpy(),
            grid.values,
            grid.easting,
            grid.northing,
            grid.cell_size_m,
            self.inclination_deg,
            self.declination_deg,
            threshold_nt=threshold_nt,
            min_footprint_m=req.min_footprint_m,
            max_footprint_m=req.max_footprint_m,
            fit_window_m=req.fit_window_m,
            max_depth_m=req.max_depth_m,
            min_fit_quality=req.min_fit_quality,
        )

        structure_polys_local = []
        if structure_polygons:
            to_local = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
            for ring in structure_polygons:
                xs, ys = to_local.transform([p[1] for p in ring], [p[0] for p in ring])
                structure_polys_local.append(MplPath(list(zip(xs, ys))))

        transformer = Transformer.from_crs(f"EPSG:{self.utm_epsg}", "EPSG:4326", always_xy=True)
        anomaly_dicts = []
        if candidates:
            lons, lats = transformer.transform([c.x for c in candidates], [c.y for c in candidates])
        else:
            lons, lats = [], []
        for c, lat, lon in zip(candidates, lats, lons):
            matched = any(path.contains_point((c.x, c.y)) for path in structure_polys_local)
            anomaly_dicts.append(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "peak_anomaly_nt": c.peak_anomaly_nt,
                    "footprint_m": c.footprint_m,
                    "fit_quality": c.fit_quality,
                    "matched_structure": matched,
                }
            )
        # matched-and-confident first - the most trustworthy "this is
        # almost certainly cultural noise" candidates, read top-down.
        anomaly_dicts.sort(key=lambda d: (not d["matched_structure"], -d["fit_quality"]))

        return {
            "n_buildings": len(structures.buildings),
            "n_roads": len(structures.roads),
            "structure_polygons": structure_polygons,
            "n_structure_polygons": len(structure_polygons),
            "amplitude_threshold_nt": threshold_nt,
            "cell_size_m": grid.cell_size_m,
            "anomalies": anomaly_dicts,
            "n_anomalies": len(anomaly_dicts),
            "n_matched": sum(1 for a in anomaly_dicts if a["matched_structure"]),
        }

    def scan_anomaly_candidates(self, req: AnomalyCandidateRequest) -> dict:
        """Rank the strongest local anomalies in the block and suggest a
        removal region for each - the "start editing" step of cultural
        noise removal, on a survey where the structures are stronger than
        the geology being looked for.

        Unlike scan_structure_distortion this needs no OpenStreetMap
        footprints and makes no claim about what any candidate *is*: it
        reports where the field is most extreme, how compact it is, how
        deep its source sits, and how many flight lines actually crossed
        it, and leaves the structure-or-geology call to the operator.
        That is deliberate - the two are only separable by judgement where
        they overlap, and the stated workflow is to take the cleanly
        isolated cultural anomalies first.

        See processing/anomaly_candidates.py for why candidates are ranked
        on the analytic signal by default and why the suggested region is
        sized from the anomaly's own half-width.
        """
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        if req.field == "rtp" and self.inclination_deg is None:
            raise ProjectError("RTP 기준으로 순위를 매기려면 지자기 경사각 정보가 필요합니다.")
        if self.utm_epsg is None:
            raise ProjectError("좌표계 정보가 없습니다.")

        active = self._active_mask()
        df = self.processed.loc[active]
        if df.empty:
            raise ProjectError("유효한 측선 포인트가 없습니다.")

        # Along-line smoothing stays ON here, unlike run_target_detection.
        # This hunts for sources big enough to be seen across several
        # flight lines; anything finer than the line spacing is corrugation
        # as far as the grid is concerned, and letting it through would
        # rank levelling stripes among the ten strongest "anomalies".
        grid = self._grid_for("anomaly", req.cell_size_m, req.method, req.max_distance_m)
        anomaly = grid.values

        if req.field == "as":
            ranking = analytic_signal(anomaly, grid.cell_size_m)
        elif req.field == "rtp":
            ranking = reduction_to_pole(
                anomaly, grid.cell_size_m, self.inclination_deg, self.declination_deg
            )
        elif req.field == "residual":
            ranking, _trend = remove_regional_trend(
                anomaly, grid.easting, grid.northing, order=req.trend_order
            )
        else:
            ranking = anomaly

        candidates = find_anomaly_candidates(
            ranking,
            anomaly,
            grid.easting,
            grid.northing,
            grid.cell_size_m,
            n_candidates=req.n_candidates,
            region_level_fraction=req.region_level_fraction,
            region_margin_m=req.region_margin_m,
            max_radius_m=req.max_radius_m,
            min_separation_m=req.min_separation_m,
            point_x=df["x"].to_numpy(),
            point_y=df["y"].to_numpy(),
            point_line_id=df["line_id"].to_numpy(),
            support_radius_m=(
                req.support_radius_m
                if req.support_radius_m is not None
                else (
                    SUPPORT_RADIUS_LINE_SPACINGS * self.line_spacing_m
                    if self.line_spacing_m
                    else None
                )
            ),
        )

        # Magnetization direction per candidate - see
        # processing/magnetization.py. Steel magnetized by anything other
        # than the present field is evidence of something man-made, which
        # is the call the operator is being asked to make here. Fitted on
        # the samples, and only when the field direction is known.
        point_x = df["x"].to_numpy()
        point_y = df["y"].to_numpy()
        point_values = df["anomaly"].to_numpy()

        transformer = Transformer.from_crs(f"EPSG:{self.utm_epsg}", "EPSG:4326", always_xy=True)
        out = []
        for c in candidates:
            lon, lat = transformer.transform(c.x, c.y)
            magnetization = None
            if self.inclination_deg is not None and req.estimate_magnetization:
                try:
                    estimate = estimate_magnetization(
                        point_x, point_y, point_values, c.x, c.y, c.depth_m,
                        self.inclination_deg, self.declination_deg, self.line_spacing_m,
                    )
                except Exception:
                    estimate = None      # a diagnostic must not break the scan
                magnetization = estimate.to_dict() if estimate is not None else None
            def to_latlon(ring):
                if not ring:
                    return []
                ring_lon, ring_lat = transformer.transform([q[0] for q in ring], [q[1] for q in ring])
                return [[float(la), float(lo)] for la, lo in zip(ring_lat, ring_lon)]

            polygon = to_latlon(c.polygon_xy)
            source_polygon = to_latlon(c.source_polygon_xy)
            out.append(
                {
                    "rank": c.rank,
                    "lat": float(lat),
                    "lon": float(lon),
                    "ranking_value": c.ranking_value,
                    "peak_anomaly_nt": c.peak_anomaly_nt,
                    "polarity": c.polarity,
                    "half_width_m": c.half_width_m,
                    "depth_m": c.depth_m,
                    "radius_m": c.radius_m,
                    "n_lines": c.n_lines,
                    "n_points": c.n_points,
                    "single_line": c.single_line,
                    "at_coverage_edge": c.at_coverage_edge,
                    "region_capped": c.region_capped,
                    "depth_resolved": c.depth_resolved,
                    "polygon": polygon,
                    "source_polygon": source_polygon,
                    "magnetization": magnetization,
                    "note": self._note_match(c.x, c.y, c.radius_m),
                }
            )

        return {
            "field": req.field,
            "cell_size_m": grid.cell_size_m,
            "line_spacing_m": self.line_spacing_m,
            "n_candidates": len(out),
            "candidates": out,
        }

    # ------------------------------------------------------------ geology
    def compute_geology_stats(self, req: GeologyStatsRequest) -> dict:
        """Per-rock-unit statistics of the readings against the national
        geology map - see processing/geology_stats.py."""
        if self.processed is None or self.utm_epsg is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed.loc[self._active_mask()]
        if df.empty:
            raise ProjectError("유효한 측선 포인트가 없습니다.")
        lat = df["lat"].to_numpy(dtype=float)
        lon = df["lon"].to_numpy(dtype=float)
        pad = 0.002
        try:
            collection = fetch_lithology(
                float(lat.min()) - pad, float(lon.min()) - pad, float(lat.max()) + pad, float(lon.max()) + pad,
                scale=req.scale, use_cache=req.use_cache,
            )
        except GeologyServiceError as exc:
            raise ProjectError(str(exc)) from exc

        signal_at_points = None
        if req.cell_size_m:
            grid = self._grid_for("anomaly", req.cell_size_m, DEFAULT_GRID_METHOD, None)
            signal = analytic_signal(grid.values, grid.cell_size_m)
            col = np.clip(np.round((df["x"].to_numpy() - grid.easting[0]) / grid.cell_size_m).astype(int), 0, len(grid.easting) - 1)
            row = np.clip(np.round((df["y"].to_numpy() - grid.northing[0]) / grid.cell_size_m).astype(int), 0, len(grid.northing) - 1)
            signal_at_points = signal[row, col]

        units, report = summarize_by_unit(
            lon, lat, df[req.value].to_numpy(dtype=float), df["line_id"].to_numpy(),
            collection, signal=signal_at_points,
        )
        result = {
            "scale": req.scale,
            "value": req.value,
            "units": [u.to_dict() for u in units],
            "report": report,
            "geojson": units_geojson(collection),
            "attribution": "지질도 © KIGAM 지오빅데이터 오픈플랫폼",
        }
        self.geology_stats_cache = {k: v for k, v in result.items() if k != "geojson"}
        return result

    def export_geology_stats_csv(self) -> bytes:
        if not self.geology_stats_cache:
            raise ProjectError("지질 단위별 통계를 먼저 계산하세요.")
        from magcore.geology_stats import UnitStats

        units = [UnitStats(**{
            "symbol": u["symbol"], "name": u["name"], "age": u.get("age"), "n_points": u["n_points"],
            "n_lines": u["n_lines"], "mean_nt": u["mean_nt"], "std_nt": u["std_nt"], "median_nt": u["median_nt"],
            "p10_nt": u["p10_nt"], "p90_nt": u["p90_nt"], "mean_signal": u.get("mean_signal"),
            "area_share_pct": u["share_pct"],
        }) for u in self.geology_stats_cache["units"]]
        return geology_stats_to_csv(units)

    # ------------------------------------------------------------ candidate notes
    #
    # Why a separate log instead of a flag on each candidate: a candidate
    # only exists inside one scan's results. Re-running the scan with a
    # different cell size, ranking field or candidate count renumbers
    # everything and can drop a peak entirely, while the thing on the
    # ground that the operator went and checked against the orthophoto is
    # still there. So the decision is stored against its position and
    # matched back to whatever candidate lands near it.
    #
    # How near: the decision is about a source, and two sources closer
    # together than the region being removed are not separable in this
    # data anyway. The tolerance is therefore the candidate's own region
    # radius, floored so a small region still matches after a re-scan
    # moves the peak by a cell or two.
    _NOTE_MIN_TOLERANCE_M = 30.0

    def _note_match(self, x: float, y: float, radius_m: float | None = None) -> dict | None:
        tolerance = max(self._NOTE_MIN_TOLERANCE_M, float(radius_m or 0.0))
        best, best_distance = None, tolerance
        for note in self.candidate_notes:
            distance = float(np.hypot(note["x"] - x, note["y"] - y))
            if distance <= best_distance:
                best, best_distance = note, distance
        return best

    def set_candidate_note(self, req: CandidateNoteRequest) -> dict:
        """Record (or clear) what the operator decided about the candidate
        at this position."""
        if self.utm_epsg is None:
            raise ProjectError("좌표계 정보가 없습니다 (자료 처리를 먼저 실행하세요).")
        to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
        x, y = to_utm.transform(req.lon, req.lat)
        existing = self._note_match(float(x), float(y), req.radius_m)

        if req.verdict == "clear":
            if existing is not None:
                self.candidate_notes.remove(existing)
            self.mark_changed()
            return {"notes": self.candidate_notes, "note": None}

        note = existing if existing is not None else {"x": float(x), "y": float(y)}
        note.update(
            {
                "lat": float(req.lat),
                "lon": float(req.lon),
                "verdict": req.verdict,
                "note": req.note or "",
                "decided_at": time.time(),
            }
        )
        for key in ("peak_anomaly_nt", "depth_m", "radius_m", "magnetization", "removed_method"):
            value = getattr(req, key)
            if value is not None:
                note[key] = value
        if existing is None:
            self.candidate_notes.append(note)
        self.mark_changed()
        return {"notes": self.candidate_notes, "note": note}

    def get_candidate_notes(self) -> dict:
        return {"notes": self.candidate_notes}

    def export_candidate_notes_csv(self) -> bytes:
        """The decision log as a table, for the report: what was found,
        what it was judged to be, and what was done about it."""
        rows = []
        for note in sorted(self.candidate_notes, key=lambda n: n.get("decided_at", 0.0)):
            rows.append(
                {
                    "latitude": note.get("lat"),
                    "longitude": note.get("lon"),
                    "easting": note.get("x"),
                    "northing": note.get("y"),
                    "verdict": {"structure": "지상구조물", "geology": "지질", "hold": "보류"}.get(
                        note.get("verdict"), note.get("verdict")
                    ),
                    "peak_anomaly_nt": note.get("peak_anomaly_nt"),
                    "depth_m": note.get("depth_m"),
                    "radius_m": note.get("radius_m"),
                    "magnetization": note.get("magnetization"),
                    "removed_method": note.get("removed_method"),
                    "note": note.get("note"),
                    "decided_at": (
                        pd.to_datetime(note["decided_at"], unit="s").isoformat()
                        if note.get("decided_at")
                        else None
                    ),
                }
            )
        frame = pd.DataFrame(rows, columns=[
            "latitude", "longitude", "easting", "northing", "verdict", "peak_anomaly_nt",
            "depth_m", "radius_m", "magnetization", "removed_method", "note", "decided_at",
        ])
        return frame.to_csv(index=False).encode("utf-8-sig")


# Target detection grids at a fine (often ~1m) cell size to resolve compact
# near-surface objects; on a large-area survey that same fine cell size
# multiplied by the full flight extent can balloon into a many-million-cell
# grid (benchmarked at ~1.5M cells / ~4-9s total on the sample survey at
# 1m/98m line spacing), so this bounds it - a user surveying a large area
# should raise cell_size_m (or narrow the area) rather than the request
# hanging or exhausting memory.
TARGET_DETECTION_GRID_CELL_CAP = 4_000_000
