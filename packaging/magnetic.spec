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
    excludes=["pytest", "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
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
