"""The processing report as a document with its figures in it: Word,
self-contained HTML (prints to PDF from any browser), and a PDF of the
figures with a summary page.

report.py writes the report as Markdown text - every correction that ran
and its numbers, but no picture. A report that goes to a client needs
the maps, and until 2026-11 those were screenshots pasted in by hand.
Here the same Markdown is laid out as a document and the figures the
project can draw of itself are placed after the section they illustrate.

Two parts, kept apart on purpose:

- `figure_png` and friends draw one map each from plain arrays. They use
  matplotlib's object interface (no pyplot), so nothing here depends on
  or disturbs global figure state, and they pick a font that has Hangul
  if the machine has one (`korean_font`) - without one, titles fall back
  to the English given alongside rather than printing empty boxes.
- `to_html`, `to_docx`, `to_pdf` take the Markdown and a list of figures
  (dicts with title, caption, png, after) and know nothing of magnetics.
  The Markdown subset is exactly what report.py emits: #-headings,
  "- " bullets with **bold**, pipe tables, fenced code, paragraphs.
"""
from __future__ import annotations

import base64
import html
import io
import re
from functools import lru_cache

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

_KOREAN_FONTS = ("Malgun Gothic", "NanumGothic", "NanumBarunGothic", "AppleGothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR")


@lru_cache(maxsize=1)
def korean_font() -> str | None:
    """Name of an installed font with Hangul glyphs, or None."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in _KOREAN_FONTS if name in available), None)


def _rc() -> dict:
    font = korean_font()
    rc = {"axes.unicode_minus": False, "font.size": 9}
    if font:
        rc["font.family"] = font
    return rc


def label(korean: str, english: str) -> str:
    """`korean` when it can be drawn, else `english`."""
    return korean if korean_font() else english


def _png(fig: Figure, dpi: int = 150) -> bytes:
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def _axes_km(ax, easting: np.ndarray, northing: np.ndarray):
    """Axes in kilometres from the survey's south-west corner: projected
    coordinates of six or seven digits tell a reader nothing, a distance
    does, and the corner's coordinates go in the caption."""
    ax.set_aspect("equal")
    ax.set_xlabel(label("동서 거리 (km)", "Easting from SW corner (km)"))
    ax.set_ylabel(label("남북 거리 (km)", "Northing from SW corner (km)"))
    ax.tick_params(length=3)


def grid_figure(values: np.ndarray, easting: np.ndarray, northing: np.ndarray, title: str, unit: str, cmap: str = "RdBu_r",
                symmetric: bool = True, contours: bool = False, vmin: float | None = None, vmax: float | None = None) -> bytes:
    """One gridded quantity as a map. Colour limits are the 2nd-98th
    percentiles (symmetric about zero for a signed quantity) so a single
    extreme cell does not flatten the rest of the map."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("그릴 값이 없습니다.")
    if vmin is None or vmax is None:
        lo, hi = np.percentile(finite, [2, 98])
        if symmetric:
            m = max(abs(lo), abs(hi)) or 1.0
            lo, hi = -m, m
        vmin, vmax = (lo, hi) if hi > lo else (lo - 1.0, hi + 1.0)
    x0, y0 = float(easting[0]), float(northing[0])
    cell = float(easting[1] - easting[0]) if len(easting) > 1 else 1.0
    extent = [(-cell / 2) / 1000.0, (easting[-1] - x0 + cell / 2) / 1000.0, (-cell / 2) / 1000.0, (northing[-1] - y0 + cell / 2) / 1000.0]
    width = extent[1] - extent[0]
    height = extent[3] - extent[2]
    with matplotlib.rc_context(_rc()):
        fig = Figure(figsize=(7.0, max(3.0, min(9.0, 7.0 * height / max(width, 1e-9) * 0.85 + 0.8))))
        ax = fig.add_subplot(111)
        im = ax.imshow(values, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        if contours:
            xs = (easting - x0) / 1000.0
            ys = (northing - y0) / 1000.0
            try:
                ax.contour(xs, ys, np.where(np.isfinite(values), values, np.nan), levels=8, colors="k", linewidths=0.3, alpha=0.5)
            except Exception:  # noqa: BLE001 - contours are decoration; an all-NaN row must not lose the map
                pass
        _axes_km(ax, easting, northing)
        ax.set_title(title, fontsize=11)
        bar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        bar.set_label(unit)
        return _png(fig)


def lines_figure(x: np.ndarray, y: np.ndarray, values: np.ndarray, kept: np.ndarray, title: str, unit: str) -> bytes:
    """The readings where they were taken: those kept, coloured by value;
    those left out (turns, take-off, manual exclusions), in grey."""
    x0, y0 = float(np.nanmin(x)), float(np.nanmin(y))
    kx, ky, kv = (x[kept] - x0) / 1000.0, (y[kept] - y0) / 1000.0, values[kept]
    step = max(1, len(kx) // 60000)                     # a map does not need every 10 Hz sample
    finite = kv[np.isfinite(kv)]
    m = float(np.percentile(np.abs(finite - np.median(finite)), 98)) if finite.size else 1.0
    mid = float(np.median(finite)) if finite.size else 0.0
    width, height = float(np.ptp(x)) or 1.0, float(np.ptp(y)) or 1.0
    with matplotlib.rc_context(_rc()):
        fig = Figure(figsize=(7.0, max(3.0, min(9.0, 7.0 * height / width * 0.85 + 0.8))))
        ax = fig.add_subplot(111)
        out_step = max(1, int((~kept).sum()) // 20000)
        ax.scatter((x[~kept] - x0)[::out_step] / 1000.0, (y[~kept] - y0)[::out_step] / 1000.0, s=0.3, c="#b8b8b8", linewidths=0)
        sc = ax.scatter(kx[::step], ky[::step], s=0.6, c=kv[::step], cmap="RdBu_r", vmin=mid - m, vmax=mid + m, linewidths=0)
        _axes_km(ax, x, y)
        ax.set_title(title, fontsize=11)
        fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02).set_label(unit)
        return _png(fig)


def series_figure(series: list[tuple[str, np.ndarray, np.ndarray]], title: str, xlabel: str, ylabel: str) -> bytes:
    """Lines against a shared x axis (the base station record)."""
    with matplotlib.rc_context(_rc()):
        fig = Figure(figsize=(7.0, 2.8))
        ax = fig.add_subplot(111)
        for name, xs, ys in series:
            ax.plot(xs, ys, linewidth=0.8, label=name)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        if len(series) > 1:
            ax.legend(fontsize=8, frameon=False)
        return _png(fig)


# ------------------------------------------------------------------ layout
def _blocks(markdown: str) -> list[tuple]:
    """The Markdown as a flat list of blocks: ("h", level, text),
    ("p", text), ("ul", [items]), ("table", [rows]), ("code", text)."""
    out: list[tuple] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append(("code", "\n".join(lines[i + 1:j])))
            i = j + 1
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(("h", level, line[level:].strip()))
            i += 1
        elif line.lstrip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(lines[i].lstrip()[2:])
                i += 1
            out.append(("ul", items))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append(("table", rows))
        elif line.strip():
            out.append(("p", line.strip().lstrip("> ").strip()))
            i += 1
        else:
            i += 1
    return out


def _spans(text: str) -> list[tuple[str, bool]]:
    """(text, bold) runs of a line with **bold** in it."""
    parts = re.split(r"\*\*(.+?)\*\*", text)
    return [(p, k % 2 == 1) for k, p in enumerate(parts) if p]


def _figures_after(figures: list[dict], heading: str) -> list[dict]:
    return [f for f in figures if f.get("after") and heading.startswith(f["after"])]


def _with_figures(markdown: str, figures: list[dict]) -> list[tuple]:
    """Blocks with each figure placed at the end of the section whose
    heading starts with its `after`; figures matching no section go last."""
    blocks = _blocks(markdown)
    out: list[tuple] = []
    pending: list[dict] = []
    placed: set[int] = set()
    for block in blocks:
        if block[0] == "h" and block[1] <= 2:
            out += [("fig", f) for f in pending]
            pending = _figures_after(figures, block[2])
            placed |= {id(f) for f in pending}
        out.append(block)
    out += [("fig", f) for f in pending]
    out += [("fig", f) for f in figures if id(f) not in placed]
    return out


def to_html(markdown: str, figures: list[dict], title: str = "드론 자력탐사 자료 처리 보고서") -> bytes:
    """One self-contained HTML file (figures embedded), laid out for A4
    so the browser's print dialog gives a PDF."""
    esc = html.escape
    rich = lambda t: "".join(f"<strong>{esc(s)}</strong>" if bold else esc(s) for s, bold in _spans(t))
    body = []
    number = 0
    for block in _with_figures(markdown, figures):
        kind = block[0]
        if kind == "h":
            body.append(f"<h{block[1]}>{esc(block[2])}</h{block[1]}>")
        elif kind == "p":
            body.append(f"<p>{rich(block[1])}</p>")
        elif kind == "ul":
            body.append("<ul>" + "".join(f"<li>{rich(item)}</li>" for item in block[1]) + "</ul>")
        elif kind == "table" and block[1]:
            head, *rows = block[1]
            body.append("<table><thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in head) + "</tr></thead><tbody>"
                        + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>")
        elif kind == "code":
            body.append(f"<pre>{esc(block[1])}</pre>")
        elif kind == "fig":
            number += 1
            f = block[1]
            body.append(f'<figure><img alt="{esc(f["title"])}" src="data:image/png;base64,{base64.b64encode(f["png"]).decode()}">'
                        f'<figcaption><strong>그림 {number}.</strong> {esc(f["title"])}{" — " + esc(f["caption"]) if f.get("caption") else ""}</figcaption></figure>')
    css = """
    :root { color-scheme: light; }
    body { font-family: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; font-size: 10.5pt; line-height: 1.6; color: #1f1a12;
           max-width: 180mm; margin: 0 auto; padding: 12mm 4mm; background: white; }
    h1 { font-size: 18pt; border-bottom: 2px solid #a9631f; padding-bottom: 4px; }
    h2 { font-size: 13pt; margin-top: 1.6em; border-bottom: 1px solid #e6dac0; padding-bottom: 2px; }
    h3 { font-size: 11pt; margin-top: 1.2em; }
    table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 0.6em 0; }
    th, td { border: 1px solid #d9cdb2; padding: 3px 6px; text-align: right; }
    th { background: #f7f2e6; } th:first-child, td:first-child { text-align: left; }
    pre { background: #f7f2e6; padding: 8px; font-size: 8.5pt; overflow-x: auto; white-space: pre-wrap; word-break: break-all; }
    figure { margin: 1.2em 0; text-align: center; break-inside: avoid; }
    figure img { max-width: 100%; height: auto; }
    figcaption { font-size: 9.5pt; color: #4a3d28; margin-top: 4px; text-align: left; }
    @page { size: A4; margin: 15mm; }
    @media print { body { padding: 0; max-width: none; } h2 { break-after: avoid; } }
    """
    doc = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)}</title><style>{css}</style></head><body>{"".join(body)}</body></html>'
    return doc.encode("utf-8")


def to_docx(markdown: str, figures: list[dict]) -> bytes:
    """A Word document: the report's headings, lists and tables as Word's
    own, each figure with a numbered caption."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.0)
    section.top_margin = section.bottom_margin = Cm(2.0)
    for style_name in ("Normal", "Heading 1", "Heading 2", "Heading 3", "List Bullet", "Caption"):
        style = doc.styles[style_name]
        style.font.name = "맑은 고딕"
        style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "맑은 고딕")     # Hangul uses the East Asian slot
    doc.styles["Normal"].font.size = Pt(10)

    def write_runs(paragraph, text):
        for chunk, bold in _spans(text):
            paragraph.add_run(chunk).bold = bold

    number = 0
    for block in _with_figures(markdown, figures):
        kind = block[0]
        if kind == "h":
            doc.add_heading(block[2], level=min(block[1], 3))
        elif kind == "p":
            write_runs(doc.add_paragraph(), block[1])
        elif kind == "ul":
            for item in block[1]:
                write_runs(doc.add_paragraph(style="List Bullet"), item)
        elif kind == "table" and block[1]:
            rows = block[1]
            table = doc.add_table(rows=len(rows), cols=max(len(r) for r in rows))
            table.style = "Light Grid Accent 1"
            for r, cells in enumerate(rows):
                for c, text in enumerate(cells):
                    cell = table.cell(r, c)
                    cell.text = text
                    for run in cell.paragraphs[0].runs:
                        run.font.size = Pt(9)
                        run.bold = r == 0
        elif kind == "code":
            p = doc.add_paragraph()
            run = p.add_run(block[1])
            run.font.name = "Consolas"
            run.font.size = Pt(8)
        elif kind == "fig":
            number += 1
            f = block[1]
            doc.add_picture(io.BytesIO(f["png"]), width=Cm(16.0))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption = doc.add_paragraph(style="Caption")
            caption.add_run(f"그림 {number}. ").bold = True
            caption.add_run(f["title"] + (f" — {f['caption']}" if f.get("caption") else ""))
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def to_pdf(summary_lines: list[str], figures: list[dict], title: str = "드론 자력탐사 자료 처리 보고서") -> bytes:
    """A PDF of the figures, one to a page, behind a page of key numbers.
    The full text of the report is in the Word and HTML versions; this one
    is the set of maps to hand over or print."""
    buf = io.BytesIO()
    with matplotlib.rc_context(_rc()), PdfPages(buf) as pdf:
        page = Figure(figsize=(8.27, 11.69))
        FigureCanvasAgg(page)
        page.text(0.08, 0.92, title if korean_font() else "Drone magnetic survey - processing report", fontsize=18, weight="bold")
        y = 0.86
        for line in summary_lines:
            page.text(0.08, y, line, fontsize=10.5)
            y -= 0.028
            if y < 0.08:
                break
        pdf.savefig(page)
        for number, f in enumerate(figures, start=1):
            page = Figure(figsize=(8.27, 11.69))
            FigureCanvasAgg(page)
            image = matplotlib.image.imread(io.BytesIO(f["png"]), format="png")
            h, w = image.shape[:2]
            width = 0.84
            height = min(0.72, width * (h / w) * (8.27 / 11.69))
            width = height * (w / h) * (11.69 / 8.27)
            ax = page.add_axes([(1 - width) / 2, 0.90 - height, width, height])
            ax.imshow(image)
            ax.axis("off")
            caption = f"{label('그림', 'Figure')} {number}. {f['title']}" + (f" — {f['caption']}" if f.get("caption") else "")
            page.text(0.08, 0.90 - height - 0.035, caption, fontsize=10, wrap=True)
            pdf.savefig(page)
    return buf.getvalue()
