"""Render docs/TECHNICAL_MANUAL.md into a printable A4 PDF.

The manual itself stays in Markdown (one source of truth, reviewable in a
diff); this only typesets it. Run after editing the manual:

    pip install reportlab pymupdf
    python docs/build_manual.py

Korean PDF rendering has three traps, all of which cost a day each to
find, so they are handled here deliberately:

1. reportlab's bundled CJK CID fonts (HYGothic-Medium, HYSMyeongJo-Medium)
   miscompute advance widths on mixed Korean/Latin runs and the glyphs
   overlap. Windows' own Malgun Gothic, registered as a TTF, does not.

2. Malgun Gothic - and every other legacy-codepage Korean font, DotumChe
   included - draws U+005C REVERSE SOLIDUS as the won sign. A Windows path
   typeset in it reads "C:\\magnetic" as "C:(won)magnetic". Courier draws a
   real backslash but has no Korean glyphs at all. So inline code is
   typeset in Courier when it is pure ASCII and in the Korean font
   otherwise, and a code block that needs both is reported rather than
   silently mangled.

3. Box-drawing characters are missing from Courier but present in both
   Malgun Gothic and DotumChe, so the pipeline diagram has to be set in a
   Korean font - DotumChe, because it is the one that is monospaced (and
   is East-Asian monospace: one Korean glyph is exactly two Latin ones,
   which is what keeps the diagram's columns aligned).
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS / "TECHNICAL_MANUAL.md"
OUTPUT = DOCS / "Magnetic_Processing_Technical_Manual.pdf"

KR = "Malgun"
KR_BOLD = "MalgunBold"
MONO_KR = "DotumChe"   # monospaced *and* Korean-capable; see module docstring
MONO_ASCII = "Courier"  # the only one that draws a real backslash

CONTENT_W = A4[0] - 40 * mm


def register_fonts() -> None:
    fonts = [
        (KR, r"C:\Windows\Fonts\malgun.ttf", None),
        (KR_BOLD, r"C:\Windows\Fonts\malgunbd.ttf", None),
        (MONO_KR, r"C:\Windows\Fonts\gulim.ttc", 3),
    ]
    for name, path, subfont in fonts:
        if not Path(path).exists():
            raise SystemExit(f"필요한 글꼴을 찾을 수 없습니다: {path}")
        if subfont is None:
            pdfmetrics.registerFont(TTFont(name, path))
        else:
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=subfont))
    pdfmetrics.registerFontFamily(KR, normal=KR, bold=KR_BOLD, italic=KR, boldItalic=KR_BOLD)


# --------------------------------------------------------------- styles
def build_styles() -> dict:
    base = ParagraphStyle("body", fontName=KR, fontSize=9.5, leading=15,
                          alignment=TA_LEFT, spaceAfter=5)
    return {
        "title": ParagraphStyle("title", parent=base, fontName=KR_BOLD, fontSize=23,
                                leading=32, alignment=TA_CENTER, spaceAfter=6),
        "subtitle": ParagraphStyle("subtitle", parent=base, fontSize=11, leading=18,
                                   alignment=TA_CENTER, textColor=colors.HexColor("#555555")),
        "h1": ParagraphStyle("h1", parent=base, fontName=KR_BOLD, fontSize=16, leading=22,
                             spaceBefore=2, spaceAfter=9,
                             textColor=colors.HexColor("#1a3a5c")),
        "h2": ParagraphStyle("h2", parent=base, fontName=KR_BOLD, fontSize=12.5, leading=18,
                             spaceBefore=11, spaceAfter=5,
                             textColor=colors.HexColor("#1a3a5c")),
        "h3": ParagraphStyle("h3", parent=base, fontName=KR_BOLD, fontSize=10.5, leading=16,
                             spaceBefore=8, spaceAfter=3,
                             textColor=colors.HexColor("#34495e")),
        "body": base,
        "bullet": ParagraphStyle("bullet", parent=base, leftIndent=11, bulletIndent=2,
                                 spaceAfter=2),
        "code": ParagraphStyle("code", parent=base, fontName=MONO_KR, fontSize=8,
                               leading=11.5, spaceAfter=0, spaceBefore=0),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.3, leading=12, spaceAfter=0),
        "cellhead": ParagraphStyle("cellhead", parent=base, fontName=KR_BOLD, fontSize=8.3,
                                   leading=12, spaceAfter=0, textColor=colors.white),
        "toc1": ParagraphStyle("toc1", parent=base, fontName=KR_BOLD, fontSize=10,
                               leading=17, spaceBefore=3),
        "toc2": ParagraphStyle("toc2", parent=base, fontSize=9, leading=14, leftIndent=13),
    }


# Characters the manual reads better with but no installed font can draw.
# A missing glyph is not an error in reportlab - it silently emits notdef,
# so the page just has a hole in it where a minus sign should be. Mapped to
# the nearest character that every font here does have.
GLYPH_SUBSTITUTIONS = {
    "\u2212": "-",   # MINUS SIGN -> HYPHEN-MINUS (absent from Malgun Gothic)
    "\u2080": "0",   # SUBSCRIPT ZERO -> digit zero (absent from all three)
}


def font_safe(text: str) -> str:
    for bad, good in GLYPH_SUBSTITUTIONS.items():
        text = text.replace(bad, good)
    return text


def check_coverage(text: str) -> None:
    """Warn at build time about any character no registered font can draw,
    so a later edit to the manual cannot quietly punch holes in the PDF."""
    try:
        from fontTools.ttLib import TTCollection
        from fontTools.ttLib import TTFont as FTFont
    except ImportError:
        return

    def cmap(path: str, index: int | None = None) -> set[int]:
        font = TTCollection(path).fonts[index] if index is not None else FTFont(path)
        covered: set[int] = set()
        for table in font["cmap"].tables:
            covered |= set(table.cmap.keys())
        return covered

    available = (cmap(r"C:\Windows\Fonts\malgun.ttf")
                 & cmap(r"C:\Windows\Fonts\malgunbd.ttf"))
    missing = sorted({c for c in text if ord(c) > 127 and ord(c) not in available})
    if missing:
        detail = ", ".join(f"{c!r} (U+{ord(c):04X})" for c in missing)
        print(f"  [경고] 글꼴에 없는 문자가 있어 PDF에서 빈칸으로 나옵니다: {detail}\n"
              f"         docs/build_manual.py 의 GLYPH_SUBSTITUTIONS 에 대체 문자를 "
              f"추가하세요.", file=sys.stderr)


# ------------------------------------------------------- inline markup
_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def inline(text: str) -> str:
    """Markdown inline -> reportlab mini-HTML, with the font rules above."""
    text = _LINK.sub(r"\1", text)

    placeholders: list[str] = []

    def stash(markup: str) -> str:
        placeholders.append(markup)
        return f"\x00{len(placeholders) - 1}\x00"

    def code_repl(m: re.Match) -> str:
        raw = m.group(1)
        # Courier renders a real backslash but no Korean; the Korean font is
        # the other way round. Pick per span (see module docstring).
        font = MONO_ASCII if raw.isascii() else MONO_KR
        return stash(f'<font face="{font}" size="8.6">{html.escape(raw)}</font>')

    text = _INLINE_CODE.sub(code_repl, text)
    text = _BOLD.sub(lambda m: stash(f"<b>{html.escape(m.group(1))}</b>"), text)
    text = html.escape(text)
    # Reverse order matters: a bold span stashed later can contain the
    # placeholder of an inline-code span stashed earlier (`**...`Mag`...**`),
    # so the outer one has to be substituted back in first or the inner
    # marker is re-inserted after the loop has already passed its index -
    # which printed the raw placeholder number into the page.
    for i in range(len(placeholders) - 1, -1, -1):
        text = text.replace(f"\x00{i}\x00", placeholders[i])
    return text


# ------------------------------------------------------------ parsing
def split_table_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_table_divider(line: str) -> bool:
    return bool(re.fullmatch(r"\|?[\s:|-]+\|?", line.strip())) and "-" in line


def make_table(rows: list[list[str]], styles: dict) -> Table:
    header, *body = rows
    ncols = len(header)
    data = [[Paragraph(inline(c), styles["cellhead"]) for c in header]]
    for r in body:
        r = (r + [""] * ncols)[:ncols]
        data.append([Paragraph(inline(c), styles["cell"]) for c in r])

    # First column tends to be the label, so give it more room; spread the rest.
    if ncols == 1:
        widths = [CONTENT_W]
    else:
        first = min(CONTENT_W * 0.34, max(CONTENT_W * 0.18, CONTENT_W / ncols * 1.35))
        widths = [first] + [(CONTENT_W - first) / (ncols - 1)] * (ncols - 1)

    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34506b")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c6d2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return t


def make_code_block(lines: list[str], styles: dict) -> Table:
    text = "\n".join(lines)
    if "\\" in text and not text.isascii():
        print("  [경고] 코드블록에 백슬래시와 한글이 함께 있습니다 - 백슬래시가 원화 기호로 "
              "표시됩니다:\n    " + lines[0][:70], file=sys.stderr)
    body = []
    for raw in lines:
        # NBSP keeps reportlab from collapsing the diagram's indentation.
        safe = html.escape(raw).replace(" ", "&nbsp;")
        body.append(Paragraph(safe or "&nbsp;", styles["code"]))
    t = Table([[body]], colWidths=[CONTENT_W], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f4f0")),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6d6cc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def make_quote(flowables: list, styles: dict) -> Table:
    t = Table([[flowables]], colWidths=[CONTENT_W], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef4ea")),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, colors.HexColor("#7aa06a")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


class ManualDoc(BaseDocTemplate):
    """Registers headings with the TOC as they are laid out."""

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style = flowable.style.name
        if style in ("h1", "h2"):
            text = re.sub(r"<[^>]+>", "", flowable.getPlainText())
            if text.strip() == "목차":  # the contents page is not its own entry
                return
            level = 0 if style == "h1" else 1
            self.notify("TOCEntry", (level, text, self.page))


def parse(md: str, styles: dict) -> list:
    lines = md.split("\n")
    out: list = []
    i = 0
    pending_bullets: list = []

    def flush_bullets():
        nonlocal pending_bullets
        if pending_bullets:
            out.extend(pending_bullets)
            out.append(Spacer(1, 3))
            pending_bullets = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_bullets()
            i += 1
            continue

        # fenced code
        if stripped.startswith("```"):
            flush_bullets()
            i += 1
            block: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append(Spacer(1, 3))
            out.append(make_code_block(block, styles))
            out.append(Spacer(1, 7))
            continue

        # blockquote (may itself contain tables/lists)
        if stripped.startswith(">"):
            flush_bullets()
            quoted: list[str] = []
            while i < len(lines) and (lines[i].strip().startswith(">") or
                                      (quoted and not lines[i].strip())):
                if not lines[i].strip():
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith(">"):
                        quoted.append("")
                        i += 1
                        continue
                    break
                quoted.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append(Spacer(1, 3))
            out.append(make_quote(parse("\n".join(quoted), styles), styles))
            out.append(Spacer(1, 7))
            continue

        # table
        if stripped.startswith("|") and i + 1 < len(lines) and is_table_divider(lines[i + 1]):
            flush_bullets()
            rows = [split_table_row(lines[i])]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_table_row(lines[i]))
                i += 1
            out.append(Spacer(1, 3))
            out.append(make_table(rows, styles))
            out.append(Spacer(1, 8))
            continue

        # headings
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            flush_bullets()
            level, text = len(m.group(1)), m.group(2)
            if level == 1:
                i += 1
                continue  # the document title is set on the cover page
            style = {2: "h1", 3: "h2", 4: "h3"}[level]
            out.append(Paragraph(inline(text), styles[style]))
            i += 1
            continue

        if re.fullmatch(r"-{3,}", stripped):
            flush_bullets()
            out.append(Spacer(1, 4))
            out.append(HRFlowable(width="100%", thickness=0.6,
                                  color=colors.HexColor("#c3ccd4")))
            out.append(Spacer(1, 6))
            i += 1
            continue

        # list items (an item may wrap onto following indented lines)
        m_bullet = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        m_number = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if m_bullet or m_number:
            if m_bullet:
                indent, text = len(m_bullet.group(1)), m_bullet.group(2)
                marker = "·" if indent < 2 else "-"
            else:
                indent, text = len(m_number.group(1)), m_number.group(3)
                marker = f"{m_number.group(2)}."
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if (not nxt.strip() or len(nxt) - len(nxt.lstrip()) <= indent
                        or re.match(r"^\s*([-*]\s|\d+\.\s)", nxt)
                        or nxt.strip().startswith(("#", ">", "|", "```"))):
                    break
                text += " " + nxt.strip()
                i += 1
            pending_bullets.append(Paragraph(inline(text), styles["bullet"],
                                             bulletText=marker))
            continue

        # paragraph (join continuation lines)
        flush_bullets()
        para = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", ">", "|", "```", "- ", "* "))
                    or re.match(r"^\d+\.\s", nxt) or re.fullmatch(r"-{3,}", nxt)):
                break
            para.append(nxt)
            i += 1
        out.append(Paragraph(inline(" ".join(para)), styles["body"]))

    flush_bullets()
    return out


def page_furniture(canvas, doc):
    canvas.saveState()
    canvas.setFont(KR, 8)
    canvas.setFillColor(colors.HexColor("#7b8794"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, str(canvas.getPageNumber()))
    if canvas.getPageNumber() > 1:
        canvas.drawRightString(A4[0] - 20 * mm, A4[1] - 13 * mm,
                               "드론 자력탐사 자료처리 프로그램 — 기술 매뉴얼")
        canvas.setStrokeColor(colors.HexColor("#d5dbe1"))
        canvas.setLineWidth(0.4)
        canvas.line(20 * mm, A4[1] - 15.5 * mm, A4[0] - 20 * mm, A4[1] - 15.5 * mm)
    canvas.restoreState()


def build() -> Path:
    register_fonts()
    styles = build_styles()
    md = font_safe(SOURCE.read_text(encoding="utf-8"))
    check_coverage(md)

    doc = ManualDoc(str(OUTPUT), pagesize=A4,
                    leftMargin=20 * mm, rightMargin=20 * mm,
                    topMargin=20 * mm, bottomMargin=18 * mm,
                    title="드론 자력탐사 자료처리 프로그램 기술 매뉴얼",
                    author="Magnetic Survey Processing")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=page_furniture)])

    story: list = []

    # cover
    story.append(Spacer(1, 52 * mm))
    story.append(Paragraph("드론 자력탐사<br/>자료처리 프로그램", styles["title"]))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("기술 매뉴얼", styles["title"]))
    story.append(Spacer(1, 9 * mm))
    story.append(HRFlowable(width="45%", thickness=1.1, color=colors.HexColor("#34506b"),
                            hAlign="CENTER"))
    story.append(Spacer(1, 9 * mm))
    story.append(Paragraph("이론적 배경 · 자료처리 절차 · 단계별 알고리듬과 그 근거",
                           styles["subtitle"]))
    story.append(PageBreak())

    # table of contents
    toc = TableOfContents()
    toc.levelStyles = [styles["toc1"], styles["toc2"]]
    story.append(Paragraph("목차", styles["h1"]))
    story.append(Spacer(1, 4))
    story.append(toc)
    story.append(PageBreak())

    story.extend(parse(md, styles))

    doc.multiBuild(story)
    return OUTPUT


if __name__ == "__main__":
    path = build()
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path}  ({size_kb:.0f} KB)")
