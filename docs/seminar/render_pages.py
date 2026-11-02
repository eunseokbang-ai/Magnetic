"""PDF -> page PNGs for visual QA (PyMuPDF; no poppler on this machine)."""
import sys
from pathlib import Path

import fitz

pdf = Path(sys.argv[1])
out = pdf.parent / "qa"
out.mkdir(exist_ok=True)
for f in out.glob("slide-*.png"):
    f.unlink()
doc = fitz.open(pdf)
for i, page in enumerate(doc, 1):
    page.get_pixmap(dpi=int(sys.argv[2]) if len(sys.argv) > 2 else 80).save(out / f"slide-{i:02d}.png")
print(len(doc), "pages ->", out)
