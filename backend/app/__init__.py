"""DroneMag Studio's backend package.

The processing library lives one level up, in magcore/, because the
landmine-detection program shares it. Put it on the path here so running
the app (launcher.py, uvicorn --app-dir backend) and the tests need no
install step, and tell magcore where this program keeps its caches.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from magcore import paths as _core_paths  # noqa: E402
from . import paths as _app_paths  # noqa: E402

_core_paths.configure(_app_paths.data_dir())
