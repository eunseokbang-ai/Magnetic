"""Does the line spacing resolve the field the survey is measuring?

Every derived grid on a line-based survey shows some line-parallel
structure, and the usual reflex is to blame the processing. On the Haenam
west block that reflex was measured and it was wrong. Six candidates were
eliminated on the raw 10 Hz files of five flights (about twelve lines):

  * boundary leakage from the FFT - swapping the gap fill for an
    equivalent-source continuation changed the picture almost not at all,
    though on synthetic data the same swap cuts boundary error twentyfold
  * diurnal variation - within a line pair, the disagreement does not grow
    where the two passes are further apart in time (correlation +0.03)
  * flight height - regressing the height difference out of the field
    difference removes 3% of it
  * a GPS/magnetometer timing offset - sliding each line along its own
    flight direction is minimised at exactly 0.0 m, and same-direction
    pairs disagree as much as opposite-direction ones (16.1 vs 14.2 nT)
  * per-line level or trend error - removing a per-line polynomial of
    order 0 through 5 leaves the disagreement at 12.0 -> 10.7 nT, and the
    striping in the derived grid carries no per-column offset at all
    (0.0% of its variance)
  * along-line noise - between 20 and 60 m wavelength the compensated
    data carries 0.03-0.19 nT, next to 5-17 nT at 100-300 m

What is left is the survey's own sampling. Comparing each line with the
average of its two neighbours, 55% of a line's 40-600 m content is
invisible to its neighbours - and smoothing along the line does not
reduce that share (56% keeping wavelengths over 60 m, 48% keeping only
those over 300 m). The field simply differs between the lines, at every
wavelength, because the sources are shallow relative to the spacing:
Spector-Grant on the along-line spectra puts them about 42 m from the
sensor, so the field spreads over roughly that scale, and the lines are
50 m apart. That is about one sample per wavelength across-line, where
faithful sampling needs two.

The flying itself is not at fault, and it is worth saying so because the
opposite is easy to "measure" by accident: these lines hold their track
to 2.6 m and sit 50.6 m apart with a p90/p10 spread of 1.1. An earlier
pass at this analysis read the spacing as 51-77 m and the line keeping as
72 m of wander, both of which were artifacts of measuring easting instead
of the across-line coordinate - the lines fly at azimuth 1.5 degrees, so
3.6 km of perfectly straight line moves 93 m east on its own. Anything
that assesses line geometry has to rotate into line coordinates first,
which is what this module does.

The east block, five more flights over the far side of the same survey,
came out at 46% unresolved against the west's 55%, with the same 50 m
spacing and a source distance of 38 m - measurably better, and the
difference matches which block looks cleaner on the map.

So the striping is not an artifact to be removed - it is the part of the
field that was never measured, appearing where the gridder had to invent
it. This module measures that share and says so, because a number the
user can see beats a filter that pretends to fix it.
"""
from __future__ import annotations

import numpy as np

__all__ = ["assess_line_resolution"]

# Wavelength window the assessment looks at, in metres along the line.
# Below the short end a 10 m grid carries nothing; above the long end the
# regional field dominates and neighbouring lines always agree.
_BAND_SHORT_M = 40.0
_BAND_LONG_M = 600.0
# A neighbour triple needs this much shared along-line extent to say
# anything - shorter overlaps are dominated by the ends of the profiles.
_MIN_OVERLAP_M = 800.0
_RESAMPLE_STEP_M = 5.0
_MIN_POINTS_PER_LINE = 200
# Share of a line's content its neighbours cannot see. Under a quarter the
# spacing is comfortable; over a half the across-line sampling is past the
# point where a grid can represent the field, whatever the processing.
_COMFORTABLE = 0.25
_UNDERSAMPLED = 0.50


def _source_distance_m(profile: np.ndarray, step: float, short_m: float = 40.0, long_m: float = 400.0) -> float:
    """Distance from the sensor to the sources that dominate the field,
    from the slope of the log power spectrum (Spector & Grant, 1970:
    power falls off as exp(-2*h*k), so the slope gives h).

    This is what sets how fast the field can change sideways, and so what
    line spacing it takes to sample it - and unlike a flight height it
    needs no DEM, which most projects here do not load. On the Haenam west
    block it reads 30-53 m per line against a GPS altitude of 51-65 m.
    """
    n = len(profile)
    spectrum = np.abs(np.fft.rfft((profile - profile.mean()) * np.hanning(n))) ** 2
    k = 2 * np.pi * np.fft.rfftfreq(n, d=step)
    wavelength = np.full(len(k), np.inf)
    wavelength[1:] = 2 * np.pi / k[1:]
    keep = (wavelength >= short_m) & (wavelength <= long_m) & (spectrum > 0)
    if int(keep.sum()) < 8:
        return float("nan")
    slope = np.polyfit(k[keep], np.log(spectrum[keep]), 1)[0]
    return float(-slope / 2.0)


def _profile(y: np.ndarray, v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    order = np.argsort(y)
    return np.interp(axis, y[order], v[order])


def _band(values: np.ndarray, step: float, short_m: float, long_m: float) -> np.ndarray:
    n = len(values)
    spectrum = np.fft.rfft(values - values.mean())
    freq = np.fft.rfftfreq(n, d=step)
    wavelength = np.full(len(spectrum), np.inf)
    wavelength[1:] = 1.0 / freq[1:]
    keep = (wavelength >= short_m) & (wavelength < long_m)
    return np.fft.irfft(np.where(keep, spectrum, 0.0), n=n)


def assess_line_resolution(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    line_id: np.ndarray,
    azimuth_deg: float | None,
    short_m: float = _BAND_SHORT_M,
) -> dict:
    """How much of what each line measures its neighbours cannot see.

    Each interior line is compared against the average of the two lines
    either side of it. If the spacing resolves the field, that average is
    a good prediction of the middle line and the residual is small; the
    residual is exactly the part that has to appear as line-parallel
    structure once the field is gridded.

    `azimuth_deg` is the flight direction, used only to split position
    into along-line and across-line; None falls back to reading it from
    the line centres. Returns a dict that is safe to serialise, with
    "available": False and a reason when there is not enough to say.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    line_id = np.asarray(line_id)

    if azimuth_deg is None:
        return {"available": False, "reason": "측선 방위각을 알 수 없습니다."}
    along = np.radians(azimuth_deg)
    # Along-line and across-line coordinates: the azimuth is measured from
    # north, so along = x*sin + y*cos and across is its perpendicular.
    s = x * np.sin(along) + y * np.cos(along)
    t = x * np.cos(along) - y * np.sin(along)

    lines = []
    for lid in np.unique(line_id[line_id >= 0]):
        sel = (line_id == lid) & np.isfinite(values)
        if int(sel.sum()) < _MIN_POINTS_PER_LINE:
            continue
        lines.append((float(np.median(t[sel])), s[sel], values[sel]))
    if len(lines) < 3:
        return {"available": False, "reason": f"이웃 비교에는 측선이 3개 이상 필요합니다 (현재 {len(lines)}개)."}
    lines.sort(key=lambda item: item[0])

    distances = [d for d in (_source_distance_m(v, np.median(np.diff(np.sort(ss))) or _RESAMPLE_STEP_M)
                             for _t, ss, v in lines) if np.isfinite(d) and d > 0]

    shares, spacings, contents = [], [], []
    for i in range(1, len(lines) - 1):
        (ta, sa, va), (tm, sm, vm), (tb, sb, vb) = lines[i - 1], lines[i], lines[i + 1]
        lo = max(sa.min(), sm.min(), sb.min())
        hi = min(sa.max(), sm.max(), sb.max())
        if hi - lo < _MIN_OVERLAP_M:
            continue
        axis = np.arange(lo, hi, _RESAMPLE_STEP_M)
        a, m, b = (
            _band(_profile(ss, vv, axis), _RESAMPLE_STEP_M, short_m, _BAND_LONG_M)
            for ss, vv in ((sa, va), (sm, vm), (sb, vb))
        )
        content = float(np.std(m))
        if content <= 0:
            continue
        # What the neighbours cannot predict about the middle line.
        unresolved = float(np.std(m - 0.5 * (a + b)))
        shares.append(unresolved / content)
        contents.append(content)
        spacings.append(abs(tb - ta) / 2.0)

    if not shares:
        return {"available": False, "reason": "이웃한 세 측선이 겹치는 구간이 짧아 비교할 수 없습니다."}

    share = float(np.median(shares))
    spacing = float(np.median(spacings))
    result = {
        "available": True,
        "n_triples": len(shares),
        "unresolved_pct": round(100.0 * share, 0),
        "line_spacing_m": round(spacing, 1),
        "line_content_nt": round(float(np.median(contents)), 1),
        "band_m": [short_m, _BAND_LONG_M],
    }
    if distances:
        distance = float(np.median(distances))
        result["source_distance_m"] = round(distance, 1)
        result["spacing_over_distance"] = round(spacing / distance, 2)
        # A sensor h from its sources sees them spread over a horizontal
        # scale of about h, and sampling a scale twice - the least that
        # represents it - needs a spacing of h/2.
        result["recommended_spacing_m"] = round(distance / 2.0, 0)

    if share <= _COMFORTABLE:
        verdict = (
            f"측선 간격({spacing:.0f}m)이 자기장 변화를 따라잡고 있습니다 - 이웃 측선이 못 보는 성분은 "
            f"{100 * share:.0f}%뿐이라, 파생그리드의 측선방향 줄무늬는 자료처리로 다룰 수 있는 수준입니다."
        )
    elif share < _UNDERSAMPLED:
        verdict = (
            f"측선 간격({spacing:.0f}m)이 경계선상입니다 - 각 측선이 재는 값의 {100 * share:.0f}%를 "
            f"이웃 측선은 보지 못합니다. 그만큼은 격자가 '지어낸' 값이 되므로 파생그리드에 측선방향 "
            f"줄무늬로 남습니다."
        )
    else:
        verdict = (
            f"측선 간격({spacing:.0f}m)이 이 자기장을 담기에 넓습니다 - 각 측선이 재는 값의 "
            f"{100 * share:.0f}%를 이웃 측선이 보지 못합니다. 측선 사이는 측정된 적이 없으므로, "
            f"파생그리드의 측선방향 줄무늬는 자료처리 오류가 아니라 '측정하지 않은 부분'이 드러난 "
            f"것입니다. 필터로 지우려면 실제 신호를 같이 잘라내야 합니다."
        )
    if "recommended_spacing_m" in result and share >= _COMFORTABLE:
        verdict += (
            f" 스펙트럼으로 추정한 센서-이상대 거리는 약 {result['source_distance_m']:.0f}m로, 이상대가 그 정도 "
            f"규모로 퍼져 보인다는 뜻입니다. 이를 격자에 제대로 담으려면 측선 간격이 그 절반인 "
            f"{result['recommended_spacing_m']:.0f}m 이하여야 합니다 (현재 간격은 "
            f"{result['spacing_over_distance']:.1f}배)."
        )
    result["verdict"] = verdict
    return result
