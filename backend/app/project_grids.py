"""Gridding, grid transforms (RTP, derivatives, continuation), the display
boundary, and grid exports. Part of store.Project.

Split out of store.py in 2026-11; the methods are unchanged. They rely on
the attributes store.Project declares (this class has no state of its
own) and are only meant to be used through it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import replace
from magcore import progress
from magcore.boundary import auto_survey_boundary
from magcore.boundary_export import boundary_to_kml, boundary_to_shapefile_zip
from magcore.continuation import (
    continuation_report,
    equivalent_source_field,
    field_depth_cells,
    source_continuation_fill,
)
from magcore.contours import compute_contours
from magcore.edge_margin import apply_margin, margin_mask, margin_outline
from magcore.grid_diagnostics import diagnose_striping
from magcore.gridding import GridResult, grid_confidence, grid_points
from magcore.microlevel import apply_microleveling
from magcore.render import (
    grid_to_geotiff_bytes,
    grid_to_geotiff_bytes_colored,
    grid_to_png_overlay,
    grid_to_surfer_grd_bytes,
    grid_to_xyz_bytes,
)
from magcore.transforms import (
    RTP_LOW_LATITUDE_INCLINATION_DEG,
    analytic_signal,
    derivative_easting,
    derivative_northing,
    limit_to_line_spacing_resolution,
    reduction_to_equator,
    reduction_to_pole,
    resolution_cutoff_m,
    second_derivative_ee,
    second_derivative_en,
    second_derivative_ez,
    second_derivative_nn,
    second_derivative_nz,
    theta_map,
    tilt_angle,
    total_horizontal_derivative,
    upward_continuation,
    vertical_derivative,
)
from magcore.trend import remove_regional_trend
from pyproj import Transformer
from magcore.export_formats import ExportError, geotiff_grid, geotiff_to_gpkg, grid_to_xyz_in_crs, reproject_geotiff
from magcore.uncertainty import derived_confidence
from .models import TransformConfidenceRequest
from .models import DisplayBoundaryRequest, GridConfidenceRequest, GridRequest, TransformRequest
from .project_common import (
    OverlayState,
    ProjectError,
    _GRID_CACHE_MAX_ENTRIES,
    _TRANSFORM_CACHE_MAX_ENTRIES,
    _as_job,
    _stats,
    boundary_rings,
)


# Below this a cell rests more on interpolation than on readings - the line the
# confidence layers' "low" share and the Euler solutions' flag are drawn at.
LOW_CONFIDENCE = 0.5


class GridMixin:
    def set_display_boundary(self, req: DisplayBoundaryRequest) -> dict:
        """Set or clear the optional user-drawn boundary polygon that
        clips grid overlay/export output to exactly that outline - see
        display_boundary_polygon and _apply_display_boundary.

        Deliberately does *not* touch grid_cache/transform_cache: the
        boundary is applied as a cheap mask at render time (see
        _apply_display_boundary), not baked into the cached grid values,
        so drawing/clearing it doesn't force expensive regridding."""
        if req.polygon is not None:
            rings = boundary_rings(req.polygon)
            if not rings or any(len(ring) < 3 for ring in rings):
                raise ProjectError("polygon은 최소 3개의 [lat, lon] 좌표가 필요합니다.")
        self.display_boundary_polygon = req.polygon
        self.mark_changed()
        return {"display_boundary_polygon": self.display_boundary_polygon}

    def auto_display_boundary(self, buffer_m: float | None = None) -> dict:
        """Derive the display boundary from the flown lines themselves -
        a corridor buffer_m outside the outermost line, with the regular
        line-to-line spacing closed over so the interior stays solid. See
        processing/boundary.py for the closing operation and why the
        merge radius is taken from the line spacing rather than exposed.

        buffer_m=None means "one line spacing", which is the default: the
        useful buffer scales with the survey, since the grid is only
        trustworthy about that far off the outermost line anyway, and a
        fixed value chokes a 100m-spaced survey while ballooning a
        5m-spaced one. Falls back to _FALLBACK_BOUNDARY_BUFFER_M when the
        spacing is unknown (single line / irregular lines).

        Replaces whatever boundary is currently set. Like
        set_display_boundary this leaves grid_cache/transform_cache alone
        (the boundary is a render-time mask, not baked into the grid)."""
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        if self.utm_epsg is None:
            raise ProjectError("좌표계가 설정되지 않았습니다 (자료 처리를 먼저 실행하세요).")

        df = self.processed
        active = self._active_mask()
        if not active.any():
            raise ProjectError("경계를 만들 활성 측선 자료가 없습니다.")

        auto_buffer = buffer_m is None
        extra_warnings: list[str] = []
        if auto_buffer:
            if self.line_spacing_m and self.line_spacing_m > 0:
                buffer_m = float(self.line_spacing_m)
            else:
                buffer_m = _FALLBACK_BOUNDARY_BUFFER_M
                extra_warnings.append(
                    f"측선 간격을 추정하지 못해 버퍼를 기본값 {_FALLBACK_BOUNDARY_BUFFER_M:.0f}m로 잡았습니다 - "
                    "경계가 좁으면 버퍼를 직접 지정하세요."
                )

        try:
            result = auto_survey_boundary(
                df.loc[active, "x"].to_numpy(),
                df.loc[active, "y"].to_numpy(),
                df.loc[active, "line_id"].to_numpy(),
                buffer_m=buffer_m,
                line_spacing_m=self.line_spacing_m,
            )
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc

        to_wgs84 = Transformer.from_crs(f"EPSG:{self.utm_epsg}", "EPSG:4326", always_xy=True)
        rings = []
        for ring_xy in result.rings_xy:
            lon, lat = to_wgs84.transform(ring_xy[:, 0], ring_xy[:, 1])
            rings.append([[float(la), float(lo)] for la, lo in zip(lat, lon)])
        # Single-block surveys (the normal case) keep the plain flat-ring
        # shape, so saved projects and exported boundary JSON stay readable
        # by anything that predates multi-part support - see boundary_rings.
        self.display_boundary_polygon = rings[0] if len(rings) == 1 else rings

        return {
            "display_boundary_polygon": self.display_boundary_polygon,
            # Rounded only for display/round-tripping through the UI input;
            # the geometry above used the full-precision value.
            "buffer_m": round(result.buffer_m, 1),
            # True when the buffer above was derived from the line spacing
            # rather than given by the caller, so the UI can say so.
            "buffer_auto": auto_buffer,
            "merge_radius_m": round(result.merge_radius_m, 1),
            "area_km2": round(result.area_m2 / 1e6, 4),
            "n_parts": result.n_parts,
            "n_vertices": sum(len(r) for r in rings),
            "dropped_holes": result.dropped_holes,
            "warnings": extra_warnings + result.warnings,
        }

    def export_display_boundary(self, fmt: str, name: str = "boundary") -> tuple[bytes, str, str]:
        """(bytes, media type, filename) for the current boundary as a
        zipped shapefile or a KML - see processing/boundary_export.py."""
        rings = boundary_rings(self.display_boundary_polygon)
        if not rings:
            raise ProjectError("먼저 경계를 그리거나 자동 생성하세요.")
        safe = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or "boundary"
        if fmt == "shp":
            return boundary_to_shapefile_zip(rings, safe), "application/zip", f"{safe}_shapefile.zip"
        if fmt == "kml":
            return boundary_to_kml(rings, safe), "application/vnd.google-earth.kml+xml", f"{safe}.kml"
        raise ProjectError(f"지원하지 않는 경계 내보내기 형식입니다: {fmt} (shp 또는 kml)")

    def _apply_display_boundary(self, values: np.ndarray, grid: GridResult) -> np.ndarray:
        """Mask `values` (a grid.values-shaped array) to NaN outside the
        user-drawn display_boundary_polygon, if one is set - complements
        the automatic convex-hull extrapolation cap in
        processing/gridding.py::grid_points, which still leaves the full
        convex hull filled even for a concave (e.g. L-shaped) survey
        footprint. Always returns a fresh array (never mutates `values` in
        place), since callers commonly pass in a grid_cache/transform_cache
        entry that must stay valid (boundary-free) for other callers/later
        boundary changes."""
        rings = boundary_rings(self.display_boundary_polygon)
        if not rings or self.utm_epsg is None:
            return values
        from shapely import contains_xy, unary_union
        from shapely.geometry import Polygon

        to_local = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg}", always_xy=True)
        polygons = []
        for ring in rings:
            lat = [pt[0] for pt in ring]
            lon = [pt[1] for pt in ring]
            bx, by = to_local.transform(lon, lat)
            polygons.append(Polygon(np.column_stack([bx, by])))
        # A survey split into separate blocks keeps every block; a cell is
        # shown when it falls inside any of them.
        area = polygons[0] if len(polygons) == 1 else unary_union(polygons)
        easting_2d, northing_2d = np.meshgrid(grid.easting, grid.northing)
        inside = contains_xy(area, easting_2d.ravel(), northing_2d.ravel()).reshape(values.shape)
        return np.where(inside, values, np.nan)

    def _boundary_margin(self, values: np.ndarray, grid: GridResult, req, with_outline: bool = True) -> tuple[np.ndarray, dict]:
        """Mark - or blank out - the band along the edge of the data where a
        derived grid is partly measuring its own boundary.

        Every FFT transform has to invent values where the grid is NaN, and
        an anomaly cut by the coverage boundary throws a streak along the
        grid axes from that invented continuation; on synthetic data the
        same anomaly mid-survey leaks 0.000% and two cells from the edge
        leaks 0.757%, still visible 200 cells away. No fill fixes it (the
        best harmonic continuation measured *worse* than nearest-fill) and
        tapering costs a quarter of the real edge signal, so this marks the
        band instead of pretending to correct it - see
        processing/edge_margin.py for the full table.

        The distance is measured from the *base grid's* coverage, not from
        whatever the display boundary polygon leaves visible: it is the
        gridding's extrapolation that fabricates values, so a user polygon
        drawn well inside the data should shrink the margin to nothing
        rather than carve a fresh band out of good interior cells.

        Returns (values, report); `values` is untouched unless the mode is
        "mask". Set with_outline=False for exports, which have nowhere to
        draw a line and would only pay for the contouring.
        """
        mode = getattr(req, "boundary_margin_mode", "off")
        if mode == "off":
            return values, {"mode": "off", "applied": False}
        margin_m = getattr(req, "boundary_margin_m", None)
        auto = margin_m is None
        if auto:
            margin_m = self._resolve_max_distance(req.cell_size_m, req.max_distance_m)
        report = {
            "mode": mode,
            "applied": True,
            "margin_m": round(float(margin_m), 1),
            "margin_cells": round(float(margin_m) / grid.cell_size_m, 1),
            "from_extrapolation_radius": auto,
            # The streak decays slowly, so "outside the margin" is not a
            # clean bill of health - say so wherever this is displayed.
            "note": (
                "경계 여백은 격자가 측선 사이를 보간한 것이 아니라 바깥으로 외삽한 구간입니다. "
                "파생그리드에서 경계에 걸친 이상대가 만드는 줄무늬는 이 구간에서 가장 강하지만 "
                "멀리까지 천천히 약해지므로, 여백 바깥이라고 해서 줄무늬가 없다는 뜻은 아닙니다."
            ),
        }
        mask = margin_mask(grid.values, grid.cell_size_m, float(margin_m))
        n_valid = int(np.isfinite(values).sum())
        n_masked = int((mask & np.isfinite(values)).sum())
        report["n_cells_in_margin"] = n_masked
        report["pct_of_grid"] = round(100.0 * n_masked / n_valid, 1) if n_valid else 0.0

        if mode == "mask":
            if n_valid and n_masked >= n_valid:
                # Blanking the entire map is never the useful answer; the
                # margin is simply wider than the survey is.
                report["applied"] = False
                report["reason"] = (
                    f"여백 폭({report['margin_m']}m)이 탐사 폭보다 넓어 전체가 가려집니다 - 폭을 줄이세요."
                )
                return values, report
            values = apply_margin(values, mask)
        elif with_outline:
            paths = margin_outline(grid.values, grid.cell_size_m, float(margin_m), grid.easting, grid.northing)
            to_wgs84 = Transformer.from_crs(f"EPSG:{self.utm_epsg}", "EPSG:4326", always_xy=True)
            outline = []
            for path in paths:
                lon, lat = to_wgs84.transform(path[:, 0], path[:, 1])
                outline.append([[float(a), float(b)] for a, b in zip(lat, lon)])
            report["outline"] = outline
            if not outline:
                report["reason"] = (
                    f"여백 폭({report['margin_m']}m) 안쪽에 남는 영역이 없어 선을 그릴 수 없습니다 - 폭을 줄이세요."
                )
        return values, report

    def _resolve_max_distance(self, cell_size_m: float, max_distance_m: float | None) -> float:
        if max_distance_m is not None:
            return max_distance_m
        if self.line_spacing_m:
            # Matches processing/gridding.py's _INTERIOR_FILL_FRACTION - see
            # its docstring for why 0.6 left too thin a margin.
            return max(2.0 * cell_size_m, 1.2 * self.line_spacing_m)
        return 2.0 * cell_size_m

    def _grid_for(
        self,
        value: str,
        cell_size_m: float,
        method: str = "spline",
        max_distance_m: float | None = None,
        along_line_smooth: bool = True,
        along_line_smooth_wavelength_m: float | None = None,
    ) -> GridResult:
        progress.report("격자 보간", None)
        resolved_max_distance = self._resolve_max_distance(cell_size_m, max_distance_m)
        # None (auto) = the estimated cross-line spacing itself: cross-line
        # interpolation cannot resolve anything finer than that anyway, so
        # any along-line detail below it is fabricated anisotropy
        # ("corrugation") rather than real resolvable structure - see
        # processing.gridding._along_line_lowpass.
        resolved_wavelength = along_line_smooth_wavelength_m if along_line_smooth_wavelength_m is not None else self.line_spacing_m
        effective_wavelength = resolved_wavelength if along_line_smooth else None
        # Both the raw max_distance_m (None vs an explicit float that
        # happens to equal what auto-resolution would have produced) and
        # resolved_max_distance are needed here: grid_points is called
        # below with the raw value and picks a genuinely different masking
        # algorithm depending on whether it's None (per-cell adaptive
        # local-line-gap threshold) or an explicit float (one uniform
        # project-wide threshold) - keying on resolved_max_distance alone
        # would let an explicit call collide with an auto call that
        # resolves to the same number and incorrectly reuse its result.
        key = (value, cell_size_m, method, max_distance_m, resolved_max_distance, effective_wavelength)
        cached = _lru_get(self.grid_cache, key)
        if cached is not None:
            return cached
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed
        active = self._active_mask()
        col = "anomaly" if value == "anomaly" else "tmi"
        result = grid_points(
            df.loc[active, "x"].to_numpy(),
            df.loc[active, "y"].to_numpy(),
            df.loc[active, col].to_numpy(),
            cell_size_m,
            method=method,
            # the raw (possibly None/auto) value, not resolved_max_distance -
            # None lets grid_points compute a per-cell local-line-gap-based
            # threshold instead of one project-wide constant (see its
            # docstring and _local_line_gap_m); resolved_max_distance above
            # is only used for the cache key, which just needs a stable,
            # unique-enough value, not the actual masking threshold.
            max_distance_m=max_distance_m,
            line_id=df.loc[active, "line_id"].to_numpy(),
            along_line_smooth_wavelength_m=effective_wavelength,
            typical_line_spacing_m=self.line_spacing_m,
        )
        _lru_put(self.grid_cache, key, result, _GRID_CACHE_MAX_ENTRIES)
        return result

    def _rtp_latitude_warning(self, transform: str) -> str | None:
        """RTP's operator diverges toward the magnetic equator (see
        processing/transforms.py::reduction_to_pole) - the amplitude cap
        there bounds the blowup, but a low-inclination RTP map is still
        unreliable and streaky along strike. Purely informational, and
        points at RTE, which is stable at any latitude and is already
        offered as its own transform."""
        if transform != "rtp" or self.inclination_deg is None:
            return None
        if abs(self.inclination_deg) >= RTP_LOW_LATITUDE_INCLINATION_DEG:
            return None
        return (
            f"자기 복각이 낮아({self.inclination_deg:.1f}°) 극자기 변환(RTP)이 수치적으로 불안정한 구간입니다 - "
            "측선 방향으로 늘어진 줄무늬 인공 신호가 나타날 수 있습니다. 저위도에서도 안정적인 "
            "적도자기 변환(RTE)을 대신 사용하는 것을 권장합니다."
        )

    def _cell_size_guideline_warning(self, cell_size_m: float) -> str | None:
        """The UAV magnetics survey guidelines' rule of thumb (Section
        11.1): grid to a cell size of 1/4 to 1/5 of the nominal line
        spacing - coarser under-resolves the data (unnecessary smoothing),
        finer risks fabricating detail the line spacing can't actually
        support (gridding artefacts). Purely informational (never blocks
        the request); a generous tolerance band around that rule since it
        is itself only a guideline, not a hard limit."""
        if not self.line_spacing_m or self.line_spacing_m <= 0:
            return None
        ratio = self.line_spacing_m / cell_size_m
        if ratio < 3.0:
            return (
                f"셀 크기({cell_size_m:.1f}m)가 측선 간격({self.line_spacing_m:.1f}m)에 비해 너무 큽니다 - "
                "가이드라인 권장 비율은 측선 간격의 1/4~1/5이며, 이보다 크면 자료가 불필요하게 뭉개질 수 있습니다."
            )
        if ratio > 8.0:
            return (
                f"셀 크기({cell_size_m:.1f}m)가 측선 간격({self.line_spacing_m:.1f}m)에 비해 너무 작습니다 - "
                "가이드라인 권장 비율(측선 간격의 1/4~1/5)보다 촘촘해 측선 사이 보간이 실제로 뒷받침하지 못하는 "
                "디테일이 만들어질 위험이 있습니다."
            )
        return None

    @_as_job("그리드 생성")
    def get_grid_overlay(self, req: GridRequest) -> dict:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        grid = self._maybe_pre_level(grid, req)
        grid = replace(grid, values=self._apply_display_boundary(grid.values, grid))
        cmap = req.cmap or (DEFAULT_CMAPS["anomaly_grid"] if req.value == "anomaly" else DEFAULT_CMAPS["tmi_grid"])
        overlay = grid_to_png_overlay(
            grid.values,
            grid.easting,
            grid.northing,
            self.utm_epsg,
            cmap_name=cmap,
            symmetric=(req.value == "anomaly"),
            vmin=req.vmin,
            vmax=req.vmax,
            hillshade=req.hillshade,
            hillshade_azimuth_deg=req.hillshade_azimuth_deg,
            hillshade_altitude_deg=req.hillshade_altitude_deg,
            hillshade_exaggeration=req.hillshade_exaggeration,
            cell_size_m=grid.cell_size_m,
            stretch=req.stretch,
        )
        overlay["stats"] = _stats(pd.Series(grid.values.ravel()))
        overlay["extrema"] = _extrema_locations(grid.values, grid.easting, grid.northing, self.utm_epsg)
        overlay["cell_size_m"] = grid.cell_size_m
        overlay["cell_size_guideline_warning"] = self._cell_size_guideline_warning(grid.cell_size_m)
        overlay["striping"] = self._striping_diagnostic(grid.values, grid.cell_size_m)
        # contours always come with the grid; the map's checkbox only shows or
        # hides them, so toggling it never needs a re-grid
        overlay["contours"] = compute_contours(
            grid.values, grid.easting, grid.northing, self.utm_epsg,
            interval=req.contour_interval_nt, n_levels=req.contour_n_levels,
        )
        self.last_overlay = OverlayState(grid.values, grid.easting, grid.northing, req.value)
        return overlay

    def get_grid_confidence_overlay(self, req: GridConfidenceRequest) -> dict:
        """A companion "how much should this cell be trusted" layer for
        get_grid_overlay's result - see processing/gridding.py::grid_confidence
        for what the 0-1 score means. Deliberately grids at the same cell
        size/method/max_distance the caller would use for the value layer
        itself (and shares its cache via _grid_for) so the two overlays
        line up cell-for-cell; reuses that same GridResult's own cached
        tree_dist/max_distance_grid/hull_mask (see GridResult's docstring)
        rather than recomputing them, since grid_points already computed
        them from this exact point set/parameters to build its own NaN
        mask - the confidence score is the only thing genuinely computed
        fresh here."""
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(req.value, req.cell_size_m, req.method, req.max_distance_m)
        return self._confidence_overlay(self._grid_confidence(grid, req.max_distance_m), grid, req.colormap)

    def _grid_confidence(self, grid: GridResult, max_distance_m: float | None = None) -> np.ndarray:
        """The 0-1 support score of each cell of `grid` (gridding.grid_confidence)."""
        df = self.processed
        active = self._active_mask()
        easting_2d, northing_2d = np.meshgrid(grid.easting, grid.northing)
        return grid_confidence(
            df.loc[active, "x"].to_numpy(),
            df.loc[active, "y"].to_numpy(),
            easting_2d,
            northing_2d,
            grid.cell_size_m,
            max_distance_m=max_distance_m,
            line_id=df.loc[active, "line_id"].to_numpy(),
            typical_line_spacing_m=self.line_spacing_m,
            tree_dist=grid.tree_dist,
            max_distance_grid=grid.max_distance_grid,
            hull_mask=grid.hull_mask,
        )

    def _confidence_overlay(self, confidence: np.ndarray, grid: GridResult, colormap: str) -> dict:
        overlay = grid_to_png_overlay(
            confidence,
            grid.easting,
            grid.northing,
            self.utm_epsg,
            cmap_name=colormap,
            symmetric=False,
            vmin=0.0,
            vmax=1.0,
            cell_size_m=grid.cell_size_m,
        )
        overlay["stats"] = _stats(pd.Series(confidence.ravel()))
        overlay["cell_size_m"] = grid.cell_size_m
        finite = confidence[np.isfinite(confidence)]
        # share of the covered cells that rest more on interpolation than on readings
        overlay["low_fraction"] = round(float(np.mean(finite < LOW_CONFIDENCE)), 4) if finite.size else None
        return overlay

    def _transform_components(self, transform: str, req, cell_size_m: float) -> list:
        """The linear operators a derived product is made of, as functions
        of a grid - what magcore.uncertainty propagates confidence
        through. One for a linear product; the field's derivatives for a
        product that is not linear in the field; none for products that
        do not read across cells (their confidence is the base layer's)."""
        inc, dec = self.inclination_deg, self.declination_deg
        linear = {
            "rtp": lambda v: reduction_to_pole(v, cell_size_m, inc, dec),
            "rte": lambda v: reduction_to_equator(v, cell_size_m, inc, dec),
            "1vd": lambda v: vertical_derivative(v, cell_size_m, order=1),
            "2vd": lambda v: vertical_derivative(v, cell_size_m, order=2),
            "dx": lambda v: derivative_easting(v, cell_size_m),
            "dy": lambda v: derivative_northing(v, cell_size_m),
            "dxx": lambda v: second_derivative_ee(v, cell_size_m),
            "dyy": lambda v: second_derivative_nn(v, cell_size_m),
            "dxy": lambda v: second_derivative_en(v, cell_size_m),
            "dxz": lambda v: second_derivative_ez(v, cell_size_m),
            "dyz": lambda v: second_derivative_nz(v, cell_size_m),
            "upward_continuation": lambda v: upward_continuation(v, cell_size_m, getattr(req, "continuation_height_m", None) or 2.0 * cell_size_m),
        }
        made_of = {"as": ("dx", "dy", "1vd"), "thdr": ("dx", "dy"), "tilt": ("dx", "dy", "1vd"), "theta": ("dx", "dy", "1vd")}
        names = made_of.get(transform) or ((transform,) if transform in linear else ())
        if transform in ("rtp", "rte") and inc is None:
            raise ProjectError("자기장 방향을 알 수 없어 극자화 변환의 신뢰도를 계산할 수 없습니다.")
        components = [linear[n] for n in names]
        # the same pre-smoothing the product itself gets: it widens the footprint, so it belongs in it
        if components and transform in _RESOLUTION_LIMITED_TRANSFORMS and getattr(req, "derivative_presmooth", True) and self.line_spacing_m:
            factor = getattr(req, "derivative_presmooth_factor", 1.0)
            smooth = lambda v: limit_to_line_spacing_resolution(v, cell_size_m, self.line_spacing_m, factor=factor)
            components = [(lambda v, op=op: op(smooth(v))) for op in components]
        return components

    @_as_job("파생 그리드 신뢰도")
    def get_transform_confidence_overlay(self, req: TransformConfidenceRequest) -> dict:
        """The base grid's confidence carried through a transform - see
        magcore/uncertainty.py. Lines up cell for cell with
        get_transform_overlay's picture of the same request."""
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        base = self._grid_confidence(grid, req.max_distance_m)
        progress.report("신뢰도 전파", None)
        confidence = derived_confidence(base, self._transform_components(req.transform, req, grid.cell_size_m))
        overlay = self._confidence_overlay(confidence, grid, req.colormap)
        overlay["transform"] = req.transform
        base_finite = base[np.isfinite(base)]
        overlay["base_low_fraction"] = round(float(np.mean(base_finite < LOW_CONFIDENCE)), 4) if base_finite.size else None
        return overlay

    def _transform_values(self, grid: GridResult, req: TransformRequest) -> tuple[np.ndarray, bool]:
        resolved_max_distance = self._resolve_max_distance(req.cell_size_m, req.max_distance_m)
        resolved_smooth_wavelength = req.along_line_smooth_wavelength_m if req.along_line_smooth_wavelength_m is not None else self.line_spacing_m
        effective_smooth_wavelength = resolved_smooth_wavelength if req.along_line_smooth else None
        cache_key = (
            req.value,
            req.cell_size_m,
            req.method,
            # Both the raw and resolved max_distance_m, for the same reason
            # as _grid_for's cache key: the transform result depends on
            # which underlying grid was used, and auto (None) vs an
            # explicit value that happens to equal the auto-resolved number
            # produce genuinely different grids (per-cell adaptive vs
            # uniform masking) - keying on the resolved number alone would
            # let one silently return the other's cached transform.
            req.max_distance_m,
            resolved_max_distance,
            effective_smooth_wavelength,
            req.transform,
            req.continuation_height_m,
            req.trend_order,
            req.microlevel_strength,
            req.microlevel_angle_tolerance_deg,
            req.microlevel_wavelength_factor,
            req.microlevel_mode,
            req.microlevel_cutoff_factor,
            req.microlevel_pre_apply,
            req.derivative_presmooth,
            req.derivative_presmooth_factor,
            getattr(req, "equivalent_source_factor", None),
        )
        cache_key = cache_key + (getattr(req, "boundary_continuation", True),)
        cached = _lru_get(self.transform_cache, cache_key)
        if cached is not None:
            return cached
        values, symmetric = self._compute_transform_values(grid, req)
        # The continuation hands the transform a grid with no holes in it,
        # so the transform no longer masks its own output - put the survey
        # footprint back. A no-op on every other path, where the transform
        # has already masked exactly these cells.
        values = np.where(np.isnan(grid.values), np.nan, values)
        result = (self._post_filter_transform(values, grid, req), symmetric)
        _lru_put(self.transform_cache, cache_key, result, _TRANSFORM_CACHE_MAX_ENTRIES)
        return result

    def _maybe_pre_level(self, grid: GridResult, req) -> GridResult:
        """Apply microleveling to `grid` when req.microlevel_pre_apply is
        set - shared by every GridRequest/TransformRequest call site so
        "먼저 마이크로레벨링 적용" behaves identically whether the caller
        ends up looking at the plain grid or a derived transform of it,
        instead of microleveling only being reachable as its own
        mutually-exclusive transform choice. No-op (returns grid
        unchanged) when the flag isn't set."""
        if not getattr(req, "microlevel_pre_apply", False):
            return grid
        if not self.line_spacing_m:
            raise ProjectError("측선 간격을 추정할 수 없어 micro-leveling을 적용할 수 없습니다 (측선이 2개 이상 필요).")
        leveled = apply_microleveling(
            grid.values,
            grid.cell_size_m,
            self.dominant_azimuth_deg,
            self.line_spacing_m,
            strength=req.microlevel_strength,
            angle_tolerance_deg=req.microlevel_angle_tolerance_deg,
            wavelength_bandwidth_factor=req.microlevel_wavelength_factor,
            mode=req.microlevel_mode,
            cutoff_spacing_factor=req.microlevel_cutoff_factor,
        )
        return replace(grid, values=leveled)

    def _compute_transform_values(self, grid: GridResult, req: TransformRequest) -> tuple[np.ndarray, bool]:
        transform = req.transform
        if self.inclination_deg is None:
            raise ProjectError("IGRF 계산이 필요합니다 (자료 처리를 먼저 실행하세요).")
        # Continue the field into the empty ground with a fitted source
        # layer before anything else touches the grid, so both the
        # across-line filter below and the transform itself see one
        # complete field instead of measurements next to invented values.
        # See processing/continuation.py for what this is worth: in quiet
        # ground the error a derivative inherits from the boundary drops
        # from 19-47x the true signal to 1.4-2.2x, with the strongest
        # anomalies keeping 100.0% of their amplitude.
        if transform in _CONTINUATION_TRANSFORMS and getattr(req, "boundary_continuation", True):
            grid = replace(grid, values=source_continuation_fill(grid.values))
        # Then, optionally, replace the field itself with what a source
        # layer at depth predicts. This is the striping remedy that
        # measured best on the real grid - see processing/continuation.py
        # ::equivalent_source_field for the table it is chosen from.
        depth = self._equivalent_source_depth(req, grid.cell_size_m)
        if transform in _CONTINUATION_TRANSFORMS and depth is not None:
            grid = replace(grid, values=equivalent_source_field(grid.values, depth)[0])
        # Derivative-based transforms amplify whatever the interpolator
        # invented between the flight lines; smoothing to the grid's real
        # across-line resolution first is what keeps them measuring the
        # ground instead of the gridding. See
        # processing/transforms.py:limit_to_line_spacing_resolution.
        if transform in _RESOLUTION_LIMITED_TRANSFORMS and getattr(req, "derivative_presmooth", True):
            grid = replace(
                grid,
                values=limit_to_line_spacing_resolution(
                    grid.values, grid.cell_size_m, self.line_spacing_m,
                    factor=req.derivative_presmooth_factor,
                ),
            )
        if transform != "microlevel":
            # matters most for derivative-based transforms (RTP/1VD/tilt/
            # etc.), which amplify whatever line-parallel corrugation is
            # still present in the input grid.
            grid = self._maybe_pre_level(grid, req)
        if transform == "rtp":
            return reduction_to_pole(grid.values, grid.cell_size_m, self.inclination_deg, self.declination_deg), True
        if transform == "rte":
            return reduction_to_equator(grid.values, grid.cell_size_m, self.inclination_deg, self.declination_deg), True
        if transform == "1vd":
            return vertical_derivative(grid.values, grid.cell_size_m, order=1), True
        if transform == "2vd":
            return vertical_derivative(grid.values, grid.cell_size_m, order=2), True
        if transform == "as":
            return analytic_signal(grid.values, grid.cell_size_m), False
        if transform == "thdr":
            return total_horizontal_derivative(grid.values, grid.cell_size_m), False
        if transform == "tilt":
            return tilt_angle(grid.values, grid.cell_size_m), True
        if transform == "theta":
            return theta_map(grid.values, grid.cell_size_m), False
        if transform == "dx":
            return derivative_easting(grid.values, grid.cell_size_m), True
        if transform == "dy":
            return derivative_northing(grid.values, grid.cell_size_m), True
        if transform == "dxx":
            return second_derivative_ee(grid.values, grid.cell_size_m), True
        if transform == "dyy":
            return second_derivative_nn(grid.values, grid.cell_size_m), True
        if transform == "dxy":
            return second_derivative_en(grid.values, grid.cell_size_m), True
        if transform == "dxz":
            return second_derivative_ez(grid.values, grid.cell_size_m), True
        if transform == "dyz":
            return second_derivative_nz(grid.values, grid.cell_size_m), True
        if transform == "upward_continuation":
            height_m = req.continuation_height_m or (2.0 * grid.cell_size_m)
            return upward_continuation(grid.values, grid.cell_size_m, height_m), True
        if transform == "detrend":
            residual, _trend = remove_regional_trend(grid.values, grid.easting, grid.northing, order=req.trend_order)
            return residual, True
        if transform == "microlevel":
            if not self.line_spacing_m:
                raise ProjectError("측선 간격을 추정할 수 없어 micro-leveling을 적용할 수 없습니다 (측선이 2개 이상 필요).")
            return (
                apply_microleveling(
                    grid.values,
                    grid.cell_size_m,
                    self.dominant_azimuth_deg,
                    self.line_spacing_m,
                    strength=req.microlevel_strength,
                    angle_tolerance_deg=req.microlevel_angle_tolerance_deg,
                    wavelength_bandwidth_factor=req.microlevel_wavelength_factor,
                    mode=req.microlevel_mode,
                    cutoff_spacing_factor=req.microlevel_cutoff_factor,
                ),
                True,
            )
        raise ProjectError(f"알 수 없는 변환입니다: {transform}")

    def _post_filter_transform(self, values: np.ndarray, grid: GridResult, req) -> np.ndarray:
        """Run the across-line filter over the transform's *output* as well
        as its input.

        Filtering the input alone leaves the input at the numerical floor
        but the output still carrying cell-scale texture, because the
        derivative multiplies whatever is left by the wavenumber - on the
        real survey grid an input at 0.0001% of variance in the 2-4 cell
        band came out as 0.32% of the second derivative's. A second pass
        over the output takes that to 0.068%, and the analytic signal from
        0.013% to 0.001%. It is close to free: the two hundred strongest
        analytic-signal peaks keep 98.4% of their amplitude, because a
        compact target is isotropic and this filter only touches
        across-line directions.

        Non-negative transforms (AS, THDR) are clipped back at zero - the
        filter can push a near-zero cell slightly below, and a negative
        analytic signal is meaningless."""
        if req.transform not in _RESOLUTION_LIMITED_TRANSFORMS or not getattr(req, "derivative_presmooth", True):
            return values
        filtered = limit_to_line_spacing_resolution(
            values, grid.cell_size_m, self.line_spacing_m, factor=req.derivative_presmooth_factor
        )
        if filtered is values:
            return values
        if np.nanmin(values) >= 0.0:
            filtered = np.where(np.isnan(filtered), filtered, np.maximum(filtered, 0.0))
        return filtered

    def _equivalent_source_depth(self, req, cell_size_m: float) -> float | None:
        """Depth in cells for the equivalent-source field, or None when the
        option is off or the line spacing is unknown."""
        factor = getattr(req, "equivalent_source_factor", None)
        if not factor:
            return None
        return field_depth_cells(self.line_spacing_m, cell_size_m, factor)

    def _equivalent_source_report(self, req, grid: GridResult) -> dict:
        """Whether the grid was replaced by a fitted source layer's field,
        how deep the layer sat, and how well it reproduces the readings.
        The misfit is the number that makes this defensible to a client:
        it says in nanotesla what the smoothing cost."""
        factor = getattr(req, "equivalent_source_factor", None)
        if req.transform not in _CONTINUATION_TRANSFORMS or not factor:
            return {"applies": False}
        depth = self._equivalent_source_depth(req, grid.cell_size_m)
        if depth is None:
            return {
                "applies": True, "applied": False,
                "reason": "측선 간격을 추정할 수 없어 등가층 깊이를 정할 수 없습니다 (측선이 2개 이상 필요).",
            }
        _field, misfit = equivalent_source_field(grid.values, depth)
        measured = grid.values[np.isfinite(grid.values)]
        return {
            "applies": True, "applied": True,
            "factor": factor,
            "depth_m": round(depth * grid.cell_size_m, 1),
            "depth_cells": round(depth, 1),
            "misfit_nt": round(misfit, 2),
            "field_std_nt": round(float(measured.std()), 1) if measured.size else None,
            "misfit_pct": round(100.0 * misfit / float(measured.std()), 1) if measured.size and measured.std() else None,
        }

    def _continuation_report(self, req, grid: GridResult) -> dict:
        """Whether the field outside the survey was continued with a fitted
        source layer, and how much ground that was. Reported because it can
        decline invisibly - a grid with no gaps, or too few readings to fit
        a layer to - and because a user who turns it off should see that
        the derivative is back to inheriting its boundary."""
        if req.transform not in _CONTINUATION_TRANSFORMS:
            return {"applies": False}
        if not getattr(req, "boundary_continuation", True):
            return {"applies": True, "applied": False, "reason": "사용자가 껐습니다."}
        info = continuation_report(grid.values)
        if not info["applicable"]:
            return {"applies": True, "applied": False, "reason": info["reason"]}
        return {"applies": True, "applied": True, "gap_pct": info["gap_pct"], "n_gap_cells": info["n_gap_cells"]}

    def _presmooth_report(self, req, cell_size_m: float) -> dict:
        """Whether the across-line filter ran on this transform and at what
        cutoff. Reported because it can decline for reasons nobody can see
        from the map - no line-spacing estimate, or cells already coarser
        than the cutoff - and a filter that quietly does nothing is
        indistinguishable from one that is broken."""
        if req.transform not in _RESOLUTION_LIMITED_TRANSFORMS:
            return {"applies": False, "reason": "이 변환은 미분 계열이 아니라 평활 대상이 아닙니다."}
        if not getattr(req, "derivative_presmooth", True):
            return {"applies": True, "applied": False, "reason": "사용자가 껐습니다."}
        cutoff = resolution_cutoff_m(cell_size_m, self.line_spacing_m, req.derivative_presmooth_factor)
        if cutoff is None:
            return {
                "applies": True, "applied": False,
                "reason": f"셀 크기({cell_size_m:.0f}m)가 차단 파장보다 커서 지울 것이 없습니다.",
            }
        return {
            "applies": True, "applied": True,
            "cutoff_m": round(cutoff, 1),
            "cutoff_cells": round(cutoff / cell_size_m, 1),
            "line_spacing_m": self.line_spacing_m,
            "from_line_spacing": bool(self.line_spacing_m and self.line_spacing_m > 0),
        }

    def _raw_cell_derivative_warning(self, req) -> str | None:
        """Derivatives of a "raw cell" grid measure the interpolator. Each
        cell takes its nearest sounding's value, so the grid is flat blocks
        with steps at exactly the line spacing, and differentiating that
        put the analytic signal 101% away from a known answer on test data.
        No post-filter rescues it: removing steps at the line spacing needs
        a cutoff around twice that, which also costs a third of the
        amplitude of every compact target. The fix is to grid smoothly, so
        say so rather than quietly producing a map of the gridding."""
        if req.transform not in _RESOLUTION_LIMITED_TRANSFORMS or req.method != "nearest":
            return None
        return (
            "보간 방법이 '원본 셀'입니다 - 각 셀이 가장 가까운 측점 값을 그대로 받는 계단형 격자라, "
            "미분 계열(AS/1VD/dXY 등)은 지반이 아니라 그 계단을 재게 됩니다(검증자료에서 AS 오차 101%). "
            "필터로는 측선 간격 단차를 지우려면 실제 이상대 진폭까지 1/3을 잃어야 해서 해결되지 않습니다 - "
            "파생그리드를 볼 때는 보간 방법을 '선형'이나 '큐빅' 이상으로 바꾸세요."
        )

    def _striping_diagnostic(self, values: np.ndarray, cell_size_m: float) -> dict:
        """Corrugation wavelength/strength of a displayed grid - see
        processing/grid_diagnostics.py. Never fatal: a diagnostic that can
        break the map it is describing is worse than no diagnostic."""
        try:
            return diagnose_striping(values, cell_size_m, self.dominant_azimuth_deg, self.line_spacing_m)
        except Exception as exc:  # noqa: BLE001 - diagnostics must not break rendering
            return {"available": False, "reason": f"줄무늬 분석 실패: {exc}"}

    @_as_job("파생 그리드 계산")
    def get_transform_overlay(self, req: TransformRequest) -> dict:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        progress.report("변환 계산", None)
        values, symmetric = self._transform_values(grid, req)
        progress.report("그림 만들기", None)
        values = self._apply_display_boundary(values, grid)
        # Before the colour stretch, so a masked margin cannot stretch the
        # scale around values the user has just said not to trust.
        values, margin_report = self._boundary_margin(values, grid, req)

        cmap = req.cmap or DEFAULT_CMAPS["derivative"]
        overlay = grid_to_png_overlay(
            values, grid.easting, grid.northing, self.utm_epsg, cmap_name=cmap, symmetric=symmetric, vmin=req.vmin, vmax=req.vmax,
            hillshade=req.hillshade,
            hillshade_azimuth_deg=req.hillshade_azimuth_deg,
            hillshade_altitude_deg=req.hillshade_altitude_deg,
            hillshade_exaggeration=req.hillshade_exaggeration,
            cell_size_m=grid.cell_size_m,
            stretch=req.stretch,
        )
        overlay["stats"] = _stats(pd.Series(values.ravel()))
        overlay["extrema"] = _extrema_locations(values, grid.easting, grid.northing, self.utm_epsg)
        overlay["cell_size_m"] = grid.cell_size_m
        overlay["cell_size_guideline_warning"] = self._cell_size_guideline_warning(grid.cell_size_m)
        # Both the displayed transform and the grid it came from: which of
        # the two carries the corrugation is exactly what says whether a
        # transform introduced it or only revealed what was already there.
        overlay["striping"] = self._striping_diagnostic(values, grid.cell_size_m)
        overlay["striping_base_grid"] = self._striping_diagnostic(grid.values, grid.cell_size_m)
        overlay["rtp_latitude_warning"] = self._rtp_latitude_warning(req.transform)
        overlay["raw_cell_derivative_warning"] = self._raw_cell_derivative_warning(req)
        overlay["derivative_presmooth"] = self._presmooth_report(req, grid.cell_size_m)
        overlay["boundary_continuation"] = self._continuation_report(req, grid)
        overlay["equivalent_source"] = self._equivalent_source_report(req, grid)
        overlay["boundary_margin"] = margin_report
        overlay["transform"] = req.transform
        overlay["contours"] = compute_contours(
            values, grid.easting, grid.northing, self.utm_epsg,
            interval=req.contour_interval_nt, n_levels=req.contour_n_levels,
        )
        self.last_overlay = OverlayState(values, grid.easting, grid.northing, req.transform)
        return overlay

    @_as_job("그리드 내보내기")
    def export_grid_geotiff(self, req: GridRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        grid = self._maybe_pre_level(grid, req)
        grid = replace(grid, values=self._apply_display_boundary(grid.values, grid))
        if req.colored:
            cmap = req.cmap or (DEFAULT_CMAPS["anomaly_grid"] if req.value == "anomaly" else DEFAULT_CMAPS["tmi_grid"])
            return self._in_export_crs(grid_to_geotiff_bytes_colored(
                grid.values, grid.easting, grid.northing, self.utm_epsg,
                cmap_name=cmap, symmetric=(req.value == "anomaly"), vmin=req.vmin, vmax=req.vmax,
                hillshade=req.hillshade, hillshade_azimuth_deg=req.hillshade_azimuth_deg,
                hillshade_altitude_deg=req.hillshade_altitude_deg, hillshade_exaggeration=req.hillshade_exaggeration,
                stretch=req.stretch,
            ), req, coloured=True)
        return self._in_export_crs(grid_to_geotiff_bytes(grid.values, grid.easting, grid.northing, self.utm_epsg), req)

    def _in_export_crs(self, geotiff: bytes, req, coloured: bool = False) -> bytes:
        """`geotiff` (in the project CRS) warped into the CRS the request
        asks for, if it asks for another one - see magcore/export_formats.py
        for what that does to the numbers."""
        epsg = getattr(req, "export_epsg", None)
        if epsg is None or int(epsg) == int(self.utm_epsg):
            return geotiff
        try:
            return reproject_geotiff(geotiff, int(epsg), coloured=coloured)
        except ExportError as exc:
            raise ProjectError(str(exc)) from exc

    def _export_values(self, req, transform: bool) -> tuple[np.ndarray, GridResult]:
        """The grid a request describes, as every export of it writes it:
        boundary applied, and for a derived grid the margin decision too."""
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        if transform:
            values, _symmetric = self._transform_values(grid, req)
            values = self._apply_display_boundary(values, grid)
            values, _margin = self._boundary_margin(values, grid, req, with_outline=False)
            return values, grid
        grid = self._maybe_pre_level(grid, req)
        return self._apply_display_boundary(grid.values, grid), grid

    def _xyz_bytes(self, values: np.ndarray, grid: GridResult, req) -> bytes:
        epsg = getattr(req, "export_epsg", None)
        if epsg is None:
            return grid_to_xyz_bytes(values, grid.easting, grid.northing, self.utm_epsg)     # lon lat value, as always
        try:
            return grid_to_xyz_in_crs(values, grid.easting, grid.northing, self.utm_epsg, int(epsg))
        except ExportError as exc:
            raise ProjectError(str(exc)) from exc

    def _grd_bytes(self, values: np.ndarray, grid: GridResult, req) -> bytes:
        epsg = getattr(req, "export_epsg", None)
        if epsg is None or int(epsg) == int(self.utm_epsg):
            return grid_to_surfer_grd_bytes(values, grid.easting, grid.northing)
        warped = self._in_export_crs(grid_to_geotiff_bytes(values, grid.easting, grid.northing, self.utm_epsg), req)
        return grid_to_surfer_grd_bytes(*geotiff_grid(warped))

    @_as_job("그리드 내보내기")
    def export_gpkg(self, req, transform: bool = False) -> bytes:
        """The grid (or derived grid) as a GeoPackage raster, in the
        requested CRS."""
        values, grid = self._export_values(req, transform)
        geotiff = self._in_export_crs(grid_to_geotiff_bytes(values, grid.easting, grid.northing, self.utm_epsg), req)
        try:
            return geotiff_to_gpkg(geotiff, layer=(req.transform if transform else req.value))
        except ExportError as exc:
            raise ProjectError(str(exc)) from exc

    @_as_job("파생 그리드 내보내기")
    def export_transform_geotiff(self, req: TransformRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        values, symmetric = self._transform_values(grid, req)
        values = self._apply_display_boundary(values, grid)
        # Exported files carry the same margin decision as the screen -
        # a GeoTIFF that quietly puts back what the map is hiding would be
        # the one that ends up in a report.
        values, _margin = self._boundary_margin(values, grid, req, with_outline=False)
        if req.colored:
            cmap = req.cmap or DEFAULT_CMAPS["derivative"]
            return self._in_export_crs(grid_to_geotiff_bytes_colored(
                values, grid.easting, grid.northing, self.utm_epsg,
                cmap_name=cmap, symmetric=symmetric, vmin=req.vmin, vmax=req.vmax,
                hillshade=req.hillshade, hillshade_azimuth_deg=req.hillshade_azimuth_deg,
                hillshade_altitude_deg=req.hillshade_altitude_deg, hillshade_exaggeration=req.hillshade_exaggeration,
                stretch=req.stretch,
            ), req, coloured=True)
        return self._in_export_crs(grid_to_geotiff_bytes(values, grid.easting, grid.northing, self.utm_epsg), req)

    def export_grid_xyz(self, req: GridRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        grid = self._maybe_pre_level(grid, req)
        grid = replace(grid, values=self._apply_display_boundary(grid.values, grid))
        return self._xyz_bytes(grid.values, grid, req)

    def export_transform_xyz(self, req: TransformRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        values, _symmetric = self._transform_values(grid, req)
        values = self._apply_display_boundary(values, grid)
        values, _margin = self._boundary_margin(values, grid, req, with_outline=False)
        return self._xyz_bytes(values, grid, req)

    def export_grid_surfer_grd(self, req: GridRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        grid = self._maybe_pre_level(grid, req)
        grid = replace(grid, values=self._apply_display_boundary(grid.values, grid))
        return self._grd_bytes(grid.values, grid, req)

    def export_transform_surfer_grd(self, req: TransformRequest) -> bytes:
        grid = self._grid_for(
            req.value, req.cell_size_m, req.method, req.max_distance_m,
            req.along_line_smooth, req.along_line_smooth_wavelength_m,
        )
        values, _symmetric = self._transform_values(grid, req)
        values = self._apply_display_boundary(values, grid)
        values, _margin = self._boundary_margin(values, grid, req, with_outline=False)
        return self._grd_bytes(values, grid, req)


DEFAULT_CMAPS = {
    "point": "viridis",
    "anomaly_grid": "RdYlBu_r",
    "tmi_grid": "viridis",
    "derivative": "RdYlBu_r",
}


# LRU entry caps for the two per-project result caches below. Both caches
# are only fully cleared on reprocess/manual-edit, so without a cap a user
# experimenting with cell sizes/interpolation methods/masking distances
# accumulates entries indefinitely - and a single grid_cache entry can be
# large (GridResult carries values + tree_dist + max_distance_grid, up to
# the ~3M-cell gridding cap: roughly 25MB per float64 array, so up to
# ~100MB per entry). Grid entries are the heavy ones (several full-grid
# arrays each) so they get the smaller cap; transform entries hold one
# values array each. Module-level (not per-instance) so tests can lower
# them without building huge grids.
# Transforms that differentiate (or otherwise amplify high wavenumbers)
# and so must not see structure finer than the survey actually resolves.
# "none"/"detrend"/"upward_continuation" don't amplify, and "microlevel"
# is itself a filter aimed at that scale, so none of them are pre-smoothed.
_RESOLUTION_LIMITED_TRANSFORMS = frozenset({
    "rtp", "rte", "1vd", "2vd", "as", "thdr", "tilt", "theta",
    "dx", "dy", "dxx", "dyy", "dxy", "dxz", "dyz",
})


# Transforms whose FFT has to invent values outside the survey, and which
# therefore benefit from the equivalent-source continuation (see
# processing/continuation.py). Upward continuation is in here too - it is
# an FFT filter like the rest - while "detrend" (a least-squares surface)
# and "microlevel" (its own directional filter) are not.
_CONTINUATION_TRANSFORMS = _RESOLUTION_LIMITED_TRANSFORMS | frozenset({"upward_continuation"})


# Display-boundary buffer used when the caller asks for the automatic
# default (buffer_m=None) but the line spacing could not be estimated -
# a single-line survey, or lines too irregular to read a spacing from.
# See Project.auto_display_boundary.
_FALLBACK_BOUNDARY_BUFFER_M = 10.0


def _lru_get(cache: dict, key):
    """Return cache[key] (refreshing its recency) or None. Python dicts
    iterate in insertion order, so re-inserting on hit makes the FIRST key
    the least-recently-USED one, which _lru_put evicts."""
    if key not in cache:
        return None
    value = cache.pop(key)
    cache[key] = value
    return value


def _lru_put(cache: dict, key, value, max_entries: int) -> None:
    cache[key] = value
    while len(cache) > max_entries:
        cache.pop(next(iter(cache)))


def _extrema_locations(values: np.ndarray, easting: np.ndarray, northing: np.ndarray, utm_epsg: int) -> dict:
    """lat/lon of the single cell holding the grid's max and min value -
    powers the "최댓값/최솟값 위치로 이동" map jump button. Exists because
    manually finding and clicking the exact extreme cell on a busy map is
    genuinely hard once cells are only a few screen pixels wide (adjacent
    cells can look nearly identical at typical zoom, especially with the
    "nearest" method's sharp cell-to-cell jumps) - jumping straight there
    sidesteps that precision problem entirely rather than just tolerating
    it."""
    finite = np.isfinite(values)
    if not finite.any():
        return {"max": None, "min": None}
    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

    def _location(idx_func):
        row, col = idx_func(np.where(finite, values, np.nan))
        lon, lat = transformer.transform(float(easting[col]), float(northing[row]))
        return {"lat": float(lat), "lon": float(lon), "value_nt": float(values[row, col])}

    return {
        "max": _location(lambda v: np.unravel_index(np.nanargmax(v), v.shape)),
        "min": _location(lambda v: np.unravel_index(np.nanargmin(v), v.shape)),
    }
