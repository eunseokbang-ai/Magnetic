"""One entry point for every survey file format the program reads.

The format is told from the file's first line, not its extension (both
formats arrive as text and operators rename files): a MagDrone R3 ASC
export starts with "Timestamp", a Log_Drone optically-pumped CSV with
the MagField_J*/TOW header. Each loader returns the same long-format
frame (one row per sensor per sample; timestamp, t_ms, track, sensor,
lat, lon, alt, tfa, bx, by, bz, b_total), so everything after loading is
format-agnostic. Files of different formats may be mixed in one project
(e.g. a fluxgate run and an optically pumped run of the same field).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .log_drone_csv import is_log_drone_csv, load_log_drone_csv
from .magdrone_r3 import MagDroneLoadError, load_magdrone_r3

FORMATS = {
    "magdrone_r3": "SENSYS MagDrone R3 DataTool ASC (2 fluxgate probes, 200 Hz)",
    "log_drone_csv": "Log_Drone CSV (optically pumped scalar pair J1/J2, 50 Hz)",
}


def detect_format(data: bytes | str) -> str:
    if is_log_drone_csv(data):
        return "log_drone_csv"
    return "magdrone_r3"


def load_survey_file(data: bytes | str, filename: str = "") -> pd.DataFrame:
    fmt = detect_format(data)
    df = load_log_drone_csv(data, filename) if fmt == "log_drone_csv" else load_magdrone_r3(data, filename)
    df.attrs["file_info"].setdefault("format", fmt)
    return df


def load_survey_files(buffers: list, filenames: list[str] | None = None) -> pd.DataFrame:
    """Several files (several flights or runs, any mix of formats) -> one
    frame sorted by time, with file_index and point_id. attrs["files"]
    lists each file's FileInfo dict (with "format")."""
    frames, infos = [], []
    for i, buf in enumerate(buffers):
        name = filenames[i] if filenames and i < len(filenames) and filenames[i] else f"file_{i}"
        data = buf.read() if hasattr(buf, "read") else buf
        df = load_survey_file(data, name)
        df["file_index"] = i
        infos.append(df.attrs["file_info"])
        frames.append(df)
    if not frames:
        raise MagDroneLoadError("올릴 파일이 없습니다.")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["timestamp", "sensor"], kind="stable").reset_index(drop=True)
    out["point_id"] = np.arange(len(out))
    out.attrs["files"] = infos
    return out
