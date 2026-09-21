"""Ground the survey flew more than once, and what it says about drift.

A magnetometer survey has one measurement it can make of its own
repeatability without any extra equipment: fly a stretch twice and
compare. The field did not change between the two passes (the diurnal
correction has already taken what did), so whatever difference is left is
the survey's own error - residual diurnal, sensor drift over a long
flight, a height difference, a heading effect that varies with attitude.

This module finds those stretches after the fact, in data that was not
flown with repeats in mind, and turns them into two things:

1. **A measurement.** How well each repeated stretch agrees, split into
   the constant part (a level difference between the two passes) and what
   is left after removing it (the part no constant can fix).
2. **A correction.** If the same flight appears in several pairs, its
   level can be solved for: the differences form a small network, exactly
   like tie-line crossovers, and least squares gives one offset per flight
   that makes them agree. This is the only handle this app has on
   flight-to-flight level differences when there are no tie lines - see
   processing/crossover_leveling.py for the tie-line case, which is the
   same algebra on crossing points instead of repeated ground.

What it deliberately does not do: invent a drift model in time. On the
2026-09 HaeNam block the within-flight trend and the across-track position
were rank-correlated at -1.000, because the lines were flown in order
from one edge to the other - so "the sensor drifted" and "the field rises
to the east" are the same numbers, and no amount of processing separates
them. Repeats are what break that tie, which is why the report says so
when there are none.

Definitions used here: two *lines* repeat each other when their tracks sit
within a fraction of the line spacing of one another and they share enough
along-track length to compare. The comparison is done on binned medians
along the track, not sample by sample: the two passes never sample the
same points, and a median over a bin is robust to a spike in either.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# How close two lines' tracks have to be, as a fraction of the line
# spacing, before they count as the same ground. A quarter of the spacing
# is well inside "the same line re-flown" and well outside "the next line
# over" for any survey that keeps its tracks to the tolerance this app
# already measures (2.6 m on the HaeNam block, 50.6 m spacing).
ACROSS_TRACK_FRACTION = 0.25
MIN_ACROSS_TRACK_M = 5.0
# Shorter than this and the comparison is dominated by whatever single
# anomaly happens to sit in it.
MIN_OVERLAP_M = 100.0
BIN_M = 5.0
MIN_BINS = 20


@dataclass
class RepeatPair:
    """One stretch of ground flown by two different lines."""
    line_a: int
    line_b: int
    flight_a: int | None
    flight_b: int | None
    n_bins: int
    overlap_m: float
    across_track_m: float
    offset_nt: float                # median(a - b): the constant part
    residual_rms_nt: float          # what is left after removing it
    hours_apart: float
    time_a: pd.Timestamp
    time_b: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "line_a": int(self.line_a),
            "line_b": int(self.line_b),
            "flight_a": None if self.flight_a is None else int(self.flight_a),
            "flight_b": None if self.flight_b is None else int(self.flight_b),
            "n_bins": self.n_bins,
            "overlap_m": round(self.overlap_m, 1),
            "across_track_m": round(self.across_track_m, 2),
            "offset_nt": round(self.offset_nt, 2),
            "residual_rms_nt": round(self.residual_rms_nt, 2),
            "hours_apart": round(self.hours_apart, 2),
            "time_a": str(self.time_a),
            "time_b": str(self.time_b),
        }


def _line_frame(x: np.ndarray, y: np.ndarray, azimuth_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(along-track, across-track) in the flight lines' own frame.

    Every geometric test here has to happen in this frame. Measured on
    real data: a line flown at azimuth 1.5 degrees moves 93 m east over
    3.6 km, so comparing easting would call two passes of one line 93 m
    apart.
    """
    theta = np.radians(azimuth_deg)
    direction = np.array([np.sin(theta), np.cos(theta)])          # x east, y north
    normal = np.array([np.cos(theta), -np.sin(theta)])
    return x * direction[0] + y * direction[1], x * normal[0] + y * normal[1]


def find_repeat_passes(
    df: pd.DataFrame,
    azimuth_deg: float,
    line_spacing_m: float | None,
    value: str = "anomaly",
    min_overlap_m: float = MIN_OVERLAP_M,
    across_tolerance_m: float | None = None,
) -> list[RepeatPair]:
    """Every pair of lines that flew the same ground, with how well they
    agree. `df` is the processed point table (x, y, line_id, timestamp,
    `value`, optionally source_file_index)."""
    needed = {"x", "y", "line_id", "timestamp", value}
    if not needed.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(needed - set(df.columns))}")
    data = df[df["line_id"] >= 0]
    data = data[np.isfinite(data[value].to_numpy())]
    if data.empty:
        return []

    tolerance = across_tolerance_m
    if tolerance is None:
        tolerance = max(MIN_ACROSS_TRACK_M, ACROSS_TRACK_FRACTION * (line_spacing_m or 50.0))

    along, across = _line_frame(data["x"].to_numpy(), data["y"].to_numpy(), azimuth_deg)
    work = pd.DataFrame(
        {
            "line_id": data["line_id"].to_numpy(),
            "flight": (
                data["source_file_index"].to_numpy()
                if "source_file_index" in data.columns
                else np.full(len(data), -1)
            ),
            "along": along,
            "across": across,
            "value": data[value].to_numpy(dtype=float),
            "timestamp": pd.to_datetime(data["timestamp"]).to_numpy(),
            "bin": np.round(along / BIN_M).astype(np.int64),
        }
    )

    profiles: dict[int, dict] = {}
    for line_id, group in work.groupby("line_id"):
        binned = group.groupby("bin")["value"].median()
        profiles[int(line_id)] = {
            "across": float(np.median(group["across"])),
            "flight": int(np.median(group["flight"])),
            "time": pd.Timestamp(group["timestamp"].min()),
            "bins": binned,
        }

    pairs: list[RepeatPair] = []
    ids = sorted(profiles)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pa, pb = profiles[a], profiles[b]
            separation = abs(pa["across"] - pb["across"])
            if separation > tolerance:
                continue
            shared = pa["bins"].index.intersection(pb["bins"].index)
            if len(shared) < MIN_BINS or len(shared) * BIN_M < min_overlap_m:
                continue
            difference = pa["bins"].loc[shared].to_numpy() - pb["bins"].loc[shared].to_numpy()
            offset = float(np.median(difference))
            residual = difference - offset
            pairs.append(
                RepeatPair(
                    line_a=a,
                    line_b=b,
                    flight_a=pa["flight"] if pa["flight"] >= 0 else None,
                    flight_b=pb["flight"] if pb["flight"] >= 0 else None,
                    n_bins=int(len(shared)),
                    overlap_m=float(len(shared) * BIN_M),
                    across_track_m=separation,
                    offset_nt=offset,
                    residual_rms_nt=float(np.sqrt(np.mean(residual**2))),
                    hours_apart=abs((pa["time"] - pb["time"]).total_seconds()) / 3600.0,
                    time_a=pa["time"],
                    time_b=pb["time"],
                )
            )
    pairs.sort(key=lambda p: -p.overlap_m)
    return pairs


def solve_offsets(pairs: list[RepeatPair], by: str = "flight") -> dict[int, float]:
    """One level offset per flight (or per line) that makes the repeated
    stretches agree, in the least-squares sense.

    Each pair gives one equation, `offset_a - offset_b = measured`. That
    system fixes the offsets only up to a common constant - nothing here
    knows the absolute level, and the diurnal correction does not care
    either - so a zero-mean constraint is added, and the result is
    subtracted as a correction rather than read as an absolute level.

    Returns {} when the pairs form no network to solve (no repeats, or
    every pair inside one flight).
    """
    if by not in ("flight", "line"):
        raise ValueError("by must be 'flight' or 'line'")

    def key(pair, side):
        if by == "flight":
            return getattr(pair, f"flight_{side}")
        return getattr(pair, f"line_{side}")

    usable = [p for p in pairs if key(p, "a") is not None and key(p, "b") is not None
              and key(p, "a") != key(p, "b")]
    if not usable:
        return {}
    groups = sorted({key(p, side) for p in usable for side in ("a", "b")})
    index = {g: i for i, g in enumerate(groups)}

    A = np.zeros((len(usable) + 1, len(groups)))
    d = np.zeros(len(usable) + 1)
    for row, pair in enumerate(usable):
        A[row, index[key(pair, "a")]] = 1.0
        A[row, index[key(pair, "b")]] = -1.0
        d[row] = pair.offset_nt
    A[-1, :] = 1.0                      # zero-mean: the network's own datum
    d[-1] = 0.0

    solution, *_ = np.linalg.lstsq(A, d, rcond=None)
    return {int(g): float(v) for g, v in zip(groups, solution)}


def summarize(pairs: list[RepeatPair], offsets: dict[int, float]) -> dict:
    """What the report shows: how much ground was repeated, how well it
    agreed, and what is left after levelling on it."""
    if not pairs:
        return {
            "n_pairs": 0,
            "advice": (
                "같은 구간을 두 번 이상 비행한 자료가 없습니다. 비행 중 센서 드리프트와 광역 경사는 "
                "측선을 한쪽 끝부터 차례로 날면 같은 신호가 되어 자료만으로는 분리할 수 없습니다. "
                "다음 탐사에서는 비행마다 직전 비행의 측선 2~3개를 다시 날거나, 타이라인을 추가하세요."
            ),
        }
    offsets_nt = np.array([abs(p.offset_nt) for p in pairs])
    residuals = np.array([p.residual_rms_nt for p in pairs])
    corrected = []
    for pair in pairs:
        a, b = offsets.get(pair.flight_a), offsets.get(pair.flight_b)
        if a is None or b is None:
            continue
        corrected.append(abs(pair.offset_nt - (a - b)))
    return {
        "n_pairs": len(pairs),
        "total_overlap_m": round(float(sum(p.overlap_m for p in pairs)), 1),
        "median_offset_nt": round(float(np.median(offsets_nt)), 2),
        "max_offset_nt": round(float(offsets_nt.max()), 2),
        "median_residual_rms_nt": round(float(np.median(residuals)), 2),
        "median_hours_apart": round(float(np.median([p.hours_apart for p in pairs])), 2),
        "n_flights_levelled": len(offsets),
        "median_offset_after_nt": round(float(np.median(corrected)), 2) if corrected else None,
        "offset_range_nt": (
            round(float(max(offsets.values()) - min(offsets.values())), 2) if offsets else None
        ),
        # The part a constant per flight cannot fix. It is the honest
        # repeatability figure for the survey, and the one to quote.
        "advice": (
            "재비행 구간에서 남는 차이(상수 보정 후)는 고도차·자세에 따른 헤딩 효과·잔여 일변화처럼 "
            "측선별 상수로는 설명되지 않는 성분입니다."
        ),
    }
