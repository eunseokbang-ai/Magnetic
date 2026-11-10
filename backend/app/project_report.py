"""The processing report with its figures: what the project draws of
itself, and the Word / HTML / PDF documents made of that - see
magcore/report_document.py. Part of store.Project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from magcore import progress
from magcore.report_document import grid_figure, label, lines_figure, series_figure, to_docx, to_html, to_pdf
from magcore.transforms import analytic_signal, reduction_to_pole, vertical_derivative

from .project_common import ProjectError, _as_job

# A report map does not need the screen's finest grid, and a 10 Hz survey
# over several km2 at 2 m cells is millions of them: the cell is a quarter
# of the line spacing, coarsened until the grid fits this many cells.
REPORT_MAX_CELLS = 600_000


class ReportMixin:
    def _report_cell_size(self) -> float:
        cell = max(2.0, float(self.line_spacing_m or 20.0) / 4.0)
        active = self.processed.loc[self._active_mask()]
        area = float(np.ptp(active["x"])) * float(np.ptp(active["y"]))
        while area / (cell * cell) > REPORT_MAX_CELLS:
            cell *= 1.5
        return round(cell, 1)

    def report_figures(self) -> list[dict]:
        """The maps of this project a report needs, each as {key, title,
        caption, png, after}: `after` is the report section it belongs
        under. Drawn from the data as it stands now - the same grid the
        screen would make at this cell size, with the display boundary
        applied - and only those the project can support (no RTP without
        a field direction, no base station plot without a base record)."""
        if self.processed is None:
            raise ProjectError("자료 처리를 먼저 실행하세요.")
        df = self.processed
        active = self._active_mask().to_numpy()
        cell = self._report_cell_size()
        figures: list[dict] = []

        def add(key, title, caption, png, after):
            figures.append({"key": key, "title": title, "caption": caption, "png": png, "after": after})

        progress.report("측선도", 0.05)
        add("lines", label("측선과 측점", "Flight lines and readings"),
            label(f"색은 자력 이상(nT), 회색은 처리에서 제외된 측점(선회·이착륙·수동 제외). 측선 {int(df.loc[active, 'line_id'].nunique())}개, 유효 측점 {int(active.sum()):,}개.",
                  f"Colour: anomaly (nT); grey: readings left out. {int(df.loc[active, 'line_id'].nunique())} lines, {int(active.sum()):,} readings."),
            lines_figure(df["x"].to_numpy(), df["y"].to_numpy(), df["anomaly"].to_numpy(), active, label("측선과 측점", "Flight lines and readings"), "nT"),
            "4.")

        progress.report("격자 보간", 0.2)
        grid = self._grid_for("anomaly", cell)
        values = self._apply_display_boundary(grid.values, grid)
        note = label(f"셀 {cell:g} m, 등치선 포함. 색 범위는 2~98 백분위.", f"Cell {cell:g} m, with contours. Colour range: 2nd-98th percentile.")
        add("anomaly", label("자력 이상도", "Magnetic anomaly"), note,
            grid_figure(values, grid.easting, grid.northing, label("자력 이상도", "Magnetic anomaly"), "nT", contours=True), "5.")

        filled = grid.values
        if self.inclination_deg is not None:
            progress.report("극자화 변환", 0.4)
            rtp = self._apply_display_boundary(reduction_to_pole(filled, grid.cell_size_m, self.inclination_deg, self.declination_deg), grid)
            add("rtp", label("극자화 변환 (RTP)", "Reduction to the pole"),
                label(f"복각 {self.inclination_deg:.1f}°, 편각 {self.declination_deg:.1f}°. 유도자화를 가정한 변환이다.",
                      f"Inclination {self.inclination_deg:.1f} deg, declination {self.declination_deg:.1f} deg; assumes induced magnetization."),
                grid_figure(rtp, grid.easting, grid.northing, label("극자화 변환 (RTP)", "Reduction to the pole"), "nT", contours=True), "5.")
        progress.report("미분 산출물", 0.55)
        asig = self._apply_display_boundary(analytic_signal(filled, grid.cell_size_m), grid)
        add("as", label("해석신호 (AS)", "Analytic signal"),
            label("자화 방향과 무관하게 소스 위에서 커지는 양.", "Peaks over sources whatever their magnetization direction."),
            grid_figure(asig, grid.easting, grid.northing, label("해석신호 (AS)", "Analytic signal"), "nT/m", cmap="magma", symmetric=False), "5.")
        vd = self._apply_display_boundary(vertical_derivative(filled, grid.cell_size_m, order=1), grid)
        add("1vd", label("1차 수직미분", "First vertical derivative"),
            label("얕은 소스와 경계를 강조한다.", "Sharpens shallow sources and edges."),
            grid_figure(vd, grid.easting, grid.northing, label("1차 수직미분", "First vertical derivative"), "nT/m"), "5.")

        progress.report("격자 신뢰도", 0.7)
        confidence = self._grid_confidence(grid)
        add("confidence", label("격자 신뢰도", "Grid confidence"),
            label("1 = 측점 바로 위, 0 = 보간에만 의존. 위 지도들을 어디까지 믿을지의 기준.", "1 = on a reading, 0 = interpolation only."),
            grid_figure(confidence, grid.easting, grid.northing, label("격자 신뢰도", "Grid confidence"), "0-1", cmap="RdYlGn", symmetric=False, vmin=0.0, vmax=1.0), "5.")

        if self.base_raw is not None and len(self.base_raw):
            progress.report("베이스 기록", 0.8)
            raw = self.base_raw.sort_values("timestamp")
            series = [(label("기록", "recorded"), _hours(raw["timestamp"]), raw["mag"].to_numpy())]
            if self.base_processed is not None and len(self.base_processed):
                used = self.base_processed.sort_values("timestamp")
                series.append((label("보정에 쓴 값", "used"), _hours(used["timestamp"], raw["timestamp"].min()), used["mag"].to_numpy()))
            add("base", label("베이스(일변화) 기록", "Base station record"),
                label(f"기록 {len(raw):,}점. 가로축은 첫 기록부터의 시간.", f"{len(raw):,} samples; time from the first sample."),
                series_figure(series, label("베이스(일변화) 기록", "Base station record"), label("시간 (h)", "Time (h)"), "nT"), "3.")

        if self.inversion_result is not None:
            progress.report("역산 단면", 0.9)
            result = self.inversion_result
            chi = result.susceptibility
            top = np.where(result.mesh.active[:, :, 0], chi[:, :, 0], np.nan) if chi.ndim == 3 else None
            if top is not None and np.isfinite(top).any():
                add("inversion", label("역산 자화율 (최상부 층)", "Inverted susceptibility (top layer)"),
                    label("유도자화만 가정한 역산이다(보고서의 잔류자화 확인 참조).", "Induced magnetization assumed."),
                    grid_figure(top, result.mesh.x_centers, result.mesh.y_centers, label("역산 자화율 (최상부 층)", "Inverted susceptibility (top layer)"),
                                "SI", cmap="viridis", symmetric=False), "7.")
        return figures

    def _report_summary_lines(self) -> list[str]:
        s = self.process_summary()
        d = self.drone_summary()
        a = s.get("anomaly_stats") or {}
        boundary = s.get("auto_boundary") or {}
        lines = [
            label(f"측점 {d.get('n_points', 0):,}개 · 측선 {s.get('n_lines')}개 · 유효 측점 {s.get('n_kept', 0):,}개", f"{d.get('n_points', 0):,} readings, {s.get('n_lines')} lines, {s.get('n_kept', 0):,} kept"),
            label(f"비행 시간: {(d.get('time_range') or ['', ''])[0][:19]} ~ {(d.get('time_range') or ['', ''])[1][:19]}", f"Flown: {(d.get('time_range') or ['', ''])[0][:19]} - {(d.get('time_range') or ['', ''])[1][:19]}"),
            label(f"측선 간격 {s.get('line_spacing_m') or 0:.1f} m · 좌표계 EPSG:{s.get('utm_epsg')}", f"Line spacing {s.get('line_spacing_m') or 0:.1f} m, EPSG:{s.get('utm_epsg')}"),
            label(f"자력 이상: {a.get('min', 0):.1f} ~ {a.get('max', 0):.1f} nT (표준편차 {a.get('std', 0):.1f} nT)", f"Anomaly: {a.get('min', 0):.1f} to {a.get('max', 0):.1f} nT (std {a.get('std', 0):.1f} nT)"),
        ]
        if boundary.get("area_km2"):
            lines.append(label(f"조사 면적 {boundary['area_km2']:.3f} km²", f"Area {boundary['area_km2']:.3f} km2"))
        lines.append("")
        lines.append(label("이 PDF는 그림 모음입니다. 보정 내역과 수치 전체는 Word·HTML 보고서에 있습니다.", "Figures only; the full text is in the Word and HTML reports."))
        return lines

    @_as_job("보고서 만들기")
    def generate_report_document(self, fmt: str) -> bytes:
        """The report with figures, as "docx", "html" or "pdf"."""
        if fmt not in ("docx", "html", "pdf"):
            raise ProjectError(f"알 수 없는 보고서 형식입니다: {fmt}")
        with progress.span(0.0, 0.9):
            figures = self.report_figures()
        progress.report("문서 조립", 0.92)
        if fmt == "pdf":
            return to_pdf(self._report_summary_lines(), figures)
        markdown = self.generate_report()
        return to_docx(markdown, figures) if fmt == "docx" else to_html(markdown, figures)


def _hours(timestamps: pd.Series, origin=None) -> np.ndarray:
    origin = timestamps.min() if origin is None else origin
    return ((timestamps - origin).dt.total_seconds() / 3600.0).to_numpy()
