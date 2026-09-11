"""Where the app's files live, in both a source checkout and the packaged
(PyInstaller) build.

Two things move when the app is frozen into an executable:

* Files shipped *with* the program (the built frontend) are unpacked into
  PyInstaller's own directory, not the repository layout, so anything that
  located them by walking up from ``__file__`` stops finding them.
* Files the program *writes* (the basemap tile cache) can no longer live
  next to the code: an installed program's own directory is read-only for
  a standard user, so writes have to go to the user's profile instead.

Both are resolved here rather than at each call site, so the rest of the
code never has to ask whether it is running frozen.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent          # backend/app
_BACKEND_DIR = _APP_DIR.parent                      # backend
_REPO_DIR = _BACKEND_DIR.parent                     # repository root


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> Path:
    """Root of the read-only files shipped alongside the program."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", _BACKEND_DIR))
    return _BACKEND_DIR


def frontend_dist() -> Path:
    """The built React app, served by the backend so the whole thing runs
    on one port. Absent in local dev (vite serves it instead)."""
    if is_frozen():
        return resource_dir() / "frontend_dist"
    return _REPO_DIR / "frontend" / "dist"


def data_dir() -> Path:
    """Writable per-user state (basemap tiles). Under the user's profile
    when installed, so the program works without administrator rights and
    two accounts on one machine do not share a cache."""
    if is_frozen():
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "MagneticSurvey"
    else:
        base = _BACKEND_DIR / ".data"
    base.mkdir(parents=True, exist_ok=True)
    return base
