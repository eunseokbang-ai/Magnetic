"""Where MineMag's files live, in a checkout and in a packaged build.
Mirrors backend/app/paths.py of DroneMag Studio; see there for why."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent          # minemag/backend/app
_BACKEND_DIR = _APP_DIR.parent                      # minemag/backend
_PRODUCT_DIR = _BACKEND_DIR.parent                  # minemag
REPO_DIR = _PRODUCT_DIR.parent                      # repository root (holds magcore/)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", _BACKEND_DIR))
    return _BACKEND_DIR


def frontend_dist() -> Path:
    if is_frozen():
        return resource_dir() / "frontend_dist"
    return _PRODUCT_DIR / "frontend" / "dist"


def data_dir() -> Path:
    """Writable per-user state. Its own folder, separate from DroneMag
    Studio's, so the two programs never read each other's caches."""
    if is_frozen():
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "MineMag"
    else:
        base = _BACKEND_DIR / ".data"
    base.mkdir(parents=True, exist_ok=True)
    return base
