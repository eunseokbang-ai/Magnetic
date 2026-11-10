"""The processing report with its figures, as Word, HTML and PDF
(magcore/report_document.py, app/project_report.py)."""
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from docx import Document
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.store import store
from magcore.report_document import _blocks, _with_figures, grid_figure, to_html

DRONE_CSV = "tests/fixtures/sample_drone_survey.csv"
BASE_CSV = "tests/fixtures/sample_base_station.csv"

MARKDOWN = """# 보고서

생성 시각: 2026-11-10

## 1. 원본 자료

- 포인트 수: 100
- **스파이크 제거**: 3개 제거

```json
{"a": 1}
```

## 4. 측선 판별 결과

| 측선 | 포인트 수 |
|---|---|
| #0 | 165 |
| #1 | 4744 |

## 5. 자력값 통계
"""


def _png(color):
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), color).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------- the layout
def test_the_markdown_the_report_writes_is_read_into_blocks():
    blocks = _blocks(MARKDOWN)
    assert [b[0] for b in blocks] == ["h", "p", "h", "ul", "code", "h", "table", "h"]
    assert blocks[0] == ("h", 1, "보고서") and blocks[3] == ("ul", ["포인트 수: 100", "**스파이크 제거**: 3개 제거"])
    assert blocks[4] == ("code", '{"a": 1}')
    assert blocks[6] == ("table", [["측선", "포인트 수"], ["#0", "165"], ["#1", "4744"]])        # the |---| rule is not a row


def test_figures_go_at_the_end_of_their_section_and_orphans_at_the_end():
    figures = [{"title": "측선도", "png": _png("red"), "after": "4."}, {"title": "이상도", "png": _png("blue"), "after": "5."},
               {"title": "어디에도 없는 절", "png": _png("green"), "after": "9."}]
    kinds = [(b[0], b[1]["title"] if b[0] == "fig" else b[2] if b[0] == "h" else None) for b in _with_figures(MARKDOWN, figures)]
    order = [k for k in kinds if k[0] in ("h", "fig")]
    assert order == [("h", "보고서"), ("h", "1. 원본 자료"), ("h", "4. 측선 판별 결과"), ("fig", "측선도"), ("h", "5. 자력값 통계"),
                     ("fig", "이상도"), ("fig", "어디에도 없는 절")]


def test_html_is_self_contained_and_escapes_what_it_is_given():
    figures = [{"title": "지도 <1>", "caption": "셀 10 m", "png": _png("red"), "after": "4."}]
    page = to_html(MARKDOWN + "\n<script>alert(1)</script>\n", figures).decode("utf-8")
    assert page.startswith("<!doctype html>") and '<html lang="ko">' in page
    assert "<h2>4. 측선 판별 결과</h2>" in page and "<strong>스파이크 제거</strong>" in page and "<th>측선</th>" in page
    assert page.count("data:image/png;base64,") == 1 and "<strong>그림 1.</strong> 지도 &lt;1&gt; — 셀 10 m" in page
    assert "<script>" not in page and "&lt;script&gt;" in page                       # text is text
    assert "http://" not in page and "https://" not in page                           # nothing fetched from anywhere


def test_a_map_figure_is_a_png_of_sensible_shape():
    values = np.random.default_rng(0).normal(0, 50, (60, 120))
    values[:5, :] = np.nan
    png = grid_figure(values, 500000 + np.arange(120) * 10.0, 4000000 + np.arange(60) * 10.0, "자력 이상도", "nT", contours=True)
    image = Image.open(io.BytesIO(png))
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and image.width > image.height > 300      # a wide survey gives a wide figure
    with pytest.raises(ValueError):
        grid_figure(np.full((4, 4), np.nan), np.arange(4.0), np.arange(4.0), "t", "nT")


# ---------------------------------------------------------- from a project
@pytest.fixture(scope="module")
def project():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    with open(DRONE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/drone", files=[("files", ("d.csv", f.read(), "text/csv"))])
    with open(BASE_CSV, "rb") as f:
        client.post(f"/api/projects/{pid}/upload/base", files=[("files", ("b.csv", f.read(), "text/csv"))])
    assert client.post(f"/api/projects/{pid}/process", json={}).status_code == 200
    return client, pid


def test_the_project_draws_the_maps_it_can_support(project):
    client, pid = project
    p = store.get(pid)
    figures = p.report_figures()
    assert [f["key"] for f in figures] == ["lines", "anomaly", "rtp", "as", "1vd", "confidence", "base"]
    assert all(f["png"][:8] == b"\x89PNG\r\n\x1a\n" and f["title"] and f["after"] for f in figures)
    # the report grid is coarse enough to stay small whatever the survey
    assert p._report_cell_size() >= 2.0
    # an inversion adds its own figure
    assert client.post(f"/api/projects/{pid}/inversion", json={"regularization_strength": 1.0}).status_code == 200
    assert [f["key"] for f in p.report_figures()][-1] == "inversion"


def test_word_report_has_the_text_and_every_figure(project):
    client, pid = project
    r = client.get(f"/api/projects/{pid}/report/docx")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/vnd.openxmlformats")
    doc = Document(io.BytesIO(r.content))
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
    assert headings[0] == "드론 자력탐사 자료 처리 보고서" and any(h.startswith("3. 적용된 보정") for h in headings)
    n_figures = len(store.get(pid).report_figures())
    assert len(doc.inline_shapes) == n_figures
    captions = [p.text for p in doc.paragraphs if p.style.name == "Caption"]
    # numbered in the order they appear: the base record belongs to section 3, so it comes first
    assert len(captions) == n_figures and captions[0].startswith("그림 1. 베이스") and captions[1].startswith("그림 2. 측선")
    assert len(doc.tables) >= 1 and doc.tables[0].cell(0, 0).text == "측선"
    # the base station plot sits in the corrections section, before the line table's section
    texts = [p.text for p in doc.paragraphs]
    assert texts.index(next(t for t in texts if "베이스(일변화) 기록" in t and t.startswith("그림"))) < texts.index(next(t for t in texts if t.startswith("4. 측선 판별")))


def test_html_and_pdf_reports(project):
    client, pid = project
    n_figures = len(store.get(pid).report_figures())
    r = client.get(f"/api/projects/{pid}/report/html")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    page = r.content.decode("utf-8")
    assert page.count("<figure>") == n_figures and "<h2>5. 자력값 통계" in page
    r = client.get(f"/api/projects/{pid}/report/pdf")
    assert r.status_code == 200 and r.content[:5] == b"%PDF-"
    assert r.content.count(b"/Type /Page\n") + r.content.count(b"/Type /Page ") >= 1
    import fitz

    assert len(fitz.open(stream=r.content, filetype="pdf")) == n_figures + 1          # a summary page, then one figure a page
    # the Markdown report is still what it was
    assert client.get(f"/api/projects/{pid}/report").headers["content-type"].startswith("text/markdown")
    assert client.get(f"/api/projects/{pid}/report/odt").status_code == 404


def test_a_report_needs_processed_data():
    client = TestClient(app)
    pid = client.post("/api/projects").json()["project_id"]
    r = client.get(f"/api/projects/{pid}/report/docx")
    assert r.status_code == 400 and "자료 처리를 먼저" in r.json()["detail"]
