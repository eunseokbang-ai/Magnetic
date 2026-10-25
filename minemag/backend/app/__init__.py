"""MineMag - landmine / UXO detection from MagDrone R3 surveys.

The processing library is magcore/ at the repository root, shared with
DroneMag Studio. Put it on the path and tell it where this program's
caches live, the same way backend/app/__init__.py does for the mineral
program.
"""
from __future__ import annotations

import sys

from . import paths as _app_paths

if str(_app_paths.REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_app_paths.REPO_DIR))

from magcore import paths as _core_paths  # noqa: E402

_core_paths.configure(_app_paths.data_dir())
