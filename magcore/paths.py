"""Where magcore writes its caches (basemap tiles, geology polygons).

The library is shared by more than one program, and an installed program
cannot write next to its own code, so the program that embeds magcore
tells it where to write with configure(). Until then the default keeps
the location the development checkout always used.
"""
from __future__ import annotations

import os
from pathlib import Path

_configured: Path | None = None


def configure(data_dir: str | os.PathLike) -> None:
    """Called once by the embedding program, before any cache module is
    imported (the caches fix their root at import time)."""
    global _configured
    _configured = Path(data_dir)


def data_dir() -> Path:
    """Writable per-user state. Resolution order: configure(), the
    MAGCORE_DATA_DIR environment variable, then backend/.data in the
    repository (the development default)."""
    if _configured is not None:
        base = _configured
    elif os.environ.get("MAGCORE_DATA_DIR"):
        base = Path(os.environ["MAGCORE_DATA_DIR"])
    else:
        base = Path(__file__).resolve().parent.parent / "backend" / ".data"
    base.mkdir(parents=True, exist_ok=True)
    return base
