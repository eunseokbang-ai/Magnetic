# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the packaged desktop build.

Build with the backend venv (not the global interpreter), from the
repository root:

    backend\\venv\\Scripts\\python.exe -m PyInstaller packaging\\magnetic.spec --noconfirm

Produces packaging/dist/DroneMagStudio/ - a self-contained folder that needs
neither Python nor Node on the target machine. packaging/installer.iss
then wraps that folder into a single setup executable.

The collect_all() calls are not defensive padding: each of those packages
loads something at runtime that a static import scan cannot see -
coordinate and datum tables (pyproj), GDAL's own data directory
(rasterio), the IGRF coefficient files (ppigrf), matplotlib's fonts and
style sheets, and numba/choclo's JIT machinery. Without them the program
builds cleanly and then fails on first use.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

REPO = Path(SPECPATH).resolve().parent
BACKEND = REPO / "backend"

datas = [(str(REPO / "frontend" / "dist"), "frontend_dist")]

# What this build was made from, for the version line the screen shows.
# The installed program has no git checkout to ask, so the answer is
# written here at build time and read by main.py::_detect_running_version.
import json
import re
import subprocess


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except Exception:
        return None


_iss = (Path(SPECPATH) / "installer.iss").read_text(encoding="utf-8")
_version = re.search(r'#define AppVersion "([^"]+)"', _iss)
_build_info = Path(SPECPATH) / "build" / "build_info.json"
_build_info.parent.mkdir(parents=True, exist_ok=True)
_build_info.write_text(json.dumps({
    "version": _version.group(1) if _version else None,
    "commit": _git("rev-parse", "--short", "HEAD"),
    "commit_date": _git("log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M"),
    "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
}), encoding="utf-8")
datas.append((str(_build_info), "."))
binaries = []
hiddenimports = []

for package in ("pyproj", "rasterio", "ppigrf", "verde", "matplotlib",
                "choclo", "numba", "llvmlite", "shapely", "certifi",
                "anthropic", "pyogrio"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception:
        continue  # optional dependency not installed in this environment
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# uvicorn resolves its protocol/loop/lifespan implementations by string at
# startup, so none of them appear in the import graph.
hiddenimports += collect_submodules("uvicorn")
hiddenimports += [
    "encodings.idna",
    "httptools",
    "websockets",
    "websockets.legacy",
    "anyio._backends._asyncio",
]

a = Analysis(
    [str(BACKEND / "launcher.py")],
    pathex=[str(BACKEND)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Trimmed only where the saving is large and the package is genuinely
    # unused at runtime: the test suite's own dependencies, and the GUI
    # toolkits matplotlib would otherwise drag in (it renders to PNG here,
    # never to a window).
    # tkinter stays in: the "타일 폴더 선택" dialog (processing/local_tiles
    # .py) is the one place the program opens a native file chooser, and
    # without it the installed copy could only take a typed path.
    excludes=["pytest", "PyQt5", "PyQt6", "PySide2", "PySide6",
              "IPython", "notebook", "sphinx"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DroneMagStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,          # the console window is the app's stop button
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DroneMagStudio",
)
