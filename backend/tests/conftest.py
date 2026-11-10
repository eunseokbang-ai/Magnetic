"""Shared test setup."""
import pathlib
import sys

# magcore lives at the repository root, next to backend/. The application
# puts it on sys.path when `app` is imported; a test file that imports
# magcore before app (several do) only worked when some other file had
# been collected first, so it failed when run on its own. Done here, once.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _path in (_ROOT / "backend", _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pytest


@pytest.fixture(autouse=True)
def _isolated_intermagnet_cache(tmp_path, monkeypatch):
    """Observatory days are cached on disk between runs. Tests mock the
    network, so each one gets its own empty cache - otherwise mocked data
    would be served to the next test, or to the real app."""
    from magcore import intermagnet

    monkeypatch.setenv(intermagnet.CACHE_DIR_ENV, str(tmp_path / "intermagnet_cache"))
    monkeypatch.setattr(intermagnet, "_sleep", lambda _s: None)
