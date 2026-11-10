"""How a project bundle stores its data frames - without pickle.

A bundle is a file people keep, send to colleagues and open again. Until
2026-11 its frames were `pickle.dumps(DataFrame)`, and unpickling a file
runs whatever code the file names: opening a project somebody else made
could execute arbitrary code on this machine. Frames are now written as
an `.npz` of plain arrays (`numpy.load(..., allow_pickle=False)` refuses
anything but array data) plus a JSON manifest:

    __manifest__   0-d unicode array holding JSON:
                   {"version": 1, "n_rows", "attrs": {...},
                    "columns": [{"name", "kind", "array", "mask"?}, ...]}
    c0, c1, ...    one array per column, in manifest order
    m0, m1, ...    null mask of a string column (only when it has nulls)

kinds: "num" (any numeric or bool dtype, stored as is), "datetime"
(datetime64[ns] as int64, NaT kept), "str" (object column of strings, as
fixed-width unicode + a null mask). The loaders produce only the first
two today (measured: 12 float64, 2 int64, 1 datetime64[ns], 1 bool,
RangeIndex, JSON-able attrs); "str" is there so a future text column
does not force a format change. Anything else is refused at save time
rather than written in a form that cannot be read back safely.

Speed is why pickle was chosen in the first place (CSV took 6.9 s on the
965k-row HaeNam block and stalled the autosave). The npz costs the same
where it matters. Measured on a 965k x 16 frame (117 MB either way):
the frame alone, pickle 0.02 s write / 0.02 s read, npz 0.17 s / 0.08 s;
inside the bundle's level-1 deflate, 2.23 s for both - the compression
is the whole cost and it does not care which bytes it is given.

Bundles written before this change still open. Their `.pkl` members go
through LegacyFrameUnpickler, which resolves only the handful of names a
pandas frame of these dtypes pickles to (recorded from real pickles under
numpy 1.26 and 2.2) and raises on any other name - so an old bundle of
ours loads, and a crafted one that reaches for `os.system` or `builtins.
eval` is rejected before anything runs.
"""
from __future__ import annotations

import io
import json
import pickle

import numpy as np
import pandas as pd

FORMAT_VERSION = 1
_MANIFEST = "__manifest__"


class BundleFrameError(ValueError):
    pass


def _json_safe(value):
    """attrs as JSON: numpy scalars to Python ones; refuse what would not
    survive (so nothing is silently dropped)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise BundleFrameError(f"프레임 속성에 저장할 수 없는 값이 있습니다: {type(value).__name__}")


def frame_to_bytes(df: pd.DataFrame) -> bytes:
    arrays: dict[str, np.ndarray] = {}
    columns = []
    for i, name in enumerate(df.columns):
        col = df[name]
        key = f"c{i}"
        entry = {"name": str(name), "array": key}
        if pd.api.types.is_datetime64_any_dtype(col.dtype):
            if getattr(col.dtype, "tz", None) is not None:
                raise BundleFrameError(f"시간대가 있는 시각 열은 저장하지 않습니다: {name}")
            entry["kind"] = "datetime"
            arrays[key] = col.to_numpy(dtype="datetime64[ns]").view("int64")
        elif pd.api.types.is_bool_dtype(col.dtype) or pd.api.types.is_numeric_dtype(col.dtype):
            values = col.to_numpy()
            if values.dtype == object:
                raise BundleFrameError(f"열 {name}의 형식({col.dtype})은 저장하지 않습니다.")
            entry["kind"] = "num"
            arrays[key] = values
        elif col.dtype == object or pd.api.types.is_string_dtype(col.dtype):
            null = col.isna().to_numpy()
            values = col.to_numpy(dtype=object)
            if not all(isinstance(v, str) for v in values[~null]):
                raise BundleFrameError(f"열 {name}에 문자열이 아닌 객체가 있어 저장할 수 없습니다.")
            text = np.where(null, "", values).astype(str)
            entry["kind"] = "str"
            arrays[key] = text
            if null.any():
                entry["mask"] = f"m{i}"
                arrays[f"m{i}"] = null
        else:
            raise BundleFrameError(f"열 {name}의 형식({col.dtype})은 저장하지 않습니다.")
        columns.append(entry)
    if not isinstance(df.index, pd.RangeIndex) or df.index.start != 0 or df.index.step != 1:
        raise BundleFrameError("0부터 시작하는 기본 색인의 프레임만 저장합니다.")
    manifest = {"version": FORMAT_VERSION, "n_rows": int(len(df)), "attrs": _json_safe(dict(df.attrs)), "columns": columns}
    arrays[_MANIFEST] = np.array(json.dumps(manifest, ensure_ascii=False))
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def frame_from_bytes(data: bytes) -> pd.DataFrame:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            manifest = json.loads(str(npz[_MANIFEST][()]))
            if manifest.get("version") != FORMAT_VERSION:
                raise BundleFrameError(f"알 수 없는 프레임 형식 버전입니다: {manifest.get('version')}")
            out = {}
            for entry in manifest["columns"]:
                values = npz[entry["array"]]
                if len(values) != manifest["n_rows"]:
                    raise BundleFrameError(f"열 {entry['name']}의 길이가 맞지 않습니다.")
                if entry["kind"] == "datetime":
                    values = values.astype("int64").view("datetime64[ns]")
                elif entry["kind"] == "str":
                    values = values.astype(object)
                    if entry.get("mask"):
                        values[npz[entry["mask"]]] = None
                elif entry["kind"] != "num":
                    raise BundleFrameError(f"알 수 없는 열 종류입니다: {entry['kind']}")
                out[entry["name"]] = values
    except BundleFrameError:
        raise
    except Exception as exc:                      # noqa: BLE001 - a damaged or foreign member
        raise BundleFrameError(f"프레임을 읽지 못했습니다: {exc}") from exc
    df = pd.DataFrame(out, columns=[e["name"] for e in manifest["columns"]])
    df.attrs = dict(manifest.get("attrs") or {})
    return df


# ----------------------------------------------------------------- legacy
# Every global a pickled frame of our dtypes resolves, recorded with a
# find_class hook on real pickles (protocols 4 and 5, with and without a
# reset index, with a string column) under numpy 1.26.4 and 2.2.6 with
# pandas 2.2.3. Nothing outside this set is ever needed to rebuild one.
_LEGACY_ALLOWED = frozenset({
    ("builtins", "slice"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),          # a numpy scalar among the attrs
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
    ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.arrays.datetimes", "DatetimeArray"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.internals.managers", "BlockManager"),
})


class LegacyFrameUnpickler(pickle.Unpickler):
    """Unpickles a frame written by the pre-2026-11 bundle format and
    nothing else: a name outside _LEGACY_ALLOWED raises before it is
    imported, let alone called."""

    def find_class(self, module, name):
        if (module, name) not in _LEGACY_ALLOWED:
            raise BundleFrameError(
                f"프로젝트 파일에 허용되지 않은 내용이 들어 있어 열지 않았습니다 ({module}.{name}). "
                "이 프로그램이 저장한 파일이 아니거나 변조된 파일일 수 있습니다."
            )
        return super().find_class(module, name)


def frame_from_legacy_pickle(data: bytes) -> pd.DataFrame:
    try:
        obj = LegacyFrameUnpickler(io.BytesIO(data)).load()
    except BundleFrameError:
        raise
    except Exception as exc:                      # noqa: BLE001 - truncated or foreign pickle
        raise BundleFrameError(f"이전 형식의 프레임을 읽지 못했습니다: {exc}") from exc
    if not isinstance(obj, pd.DataFrame):
        raise BundleFrameError("이전 형식의 프레임이 아닙니다.")
    return obj
