"""Guards against leftovers of the magcore split: a module that still
reaches for `..paths` (its old home under backend/app/processing) only
fails at call time, deep inside a feature - the nearest-INTERMAGNET fetch
returned 500 for weeks because of one such line in intermagnet._cache_dir.
"""
import importlib
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import app  # noqa: E402,F401 - puts the repository root (magcore) on sys.path and configures its data dir

import magcore
from magcore import intermagnet, paths

MAGCORE_DIR = pathlib.Path(magcore.__file__).resolve().parent


def test_no_parent_relative_imports_in_magcore():
    offenders = []
    for py in MAGCORE_DIR.glob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*(from \.\.|import \.\.)", line):
                offenders.append(f"{py.name}:{i}: {line.strip()}")
    assert not offenders, offenders


def test_every_magcore_module_imports():
    for py in MAGCORE_DIR.glob("*.py"):
        if py.name.startswith("_"):
            continue
        importlib.import_module(f"magcore.{py.stem}")


def test_intermagnet_cache_dir_is_under_the_configured_data_dir(tmp_path):
    paths.configure(tmp_path)
    try:
        cache = intermagnet._cache_dir()
        assert cache.is_dir() and tmp_path in cache.parents
        assert intermagnet._stations_file().parent == cache
    finally:
        paths.configure(pathlib.Path(__file__).resolve().parent.parent / ".data")
