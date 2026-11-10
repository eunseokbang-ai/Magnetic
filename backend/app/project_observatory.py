"""INTERMAGNET observatory data as the base station: preview, apply, the
nearest-observatory virtual base. Part of store.Project.

Split out of store.py in 2026-11; the methods are unchanged. They rely on
the attributes store.Project declares (this class has no state of its
own) and are only meant to be used through it.
"""
from __future__ import annotations

from magcore.intermagnet import (
    DEFAULT_MAX_STATION_DISTANCE_KM,
    IagaParseError,
    IntermagnetFetchError,
    _bearing_deg,
    estimate_base_from_observatories,
    fetch_observatory_dates,
    haversine_km,
    parse_iaga2002,
    select_nearest_observatories,
)
from .project_common import ProjectError


class ObservatoryMixin:
    def _preview_from_iaga(self, data, estimated_dates: list | None = None) -> dict:
        self.intermagnet_preview = data
        centroid = self._survey_centroid()
        distance_km = (
            haversine_km(centroid[0], centroid[1], data.lat, data.lon)
            if centroid is not None and data.lat is not None and data.lon is not None
            else None
        )
        return {
            "station_name": data.station_name,
            "iaga_code": data.iaga_code,
            "lat": data.lat,
            "lon": data.lon,
            "elevation_m": data.elevation_m,
            "reported": data.reported,
            "n_points": len(data.df),
            "time_range": [data.df["timestamp"].min().isoformat(), data.df["timestamp"].max().isoformat()],
            "distance_from_survey_km": distance_km,
            "estimated_dates": estimated_dates or [],
        }

    def preview_intermagnet_text(self, text: str) -> dict:
        """Parse a manually-downloaded IAGA-2002 file (from
        https://intermagnet.org or any GIN) without touching the network at
        all - the fallback path when fetch_intermagnet_preview can't reach
        the data service from wherever this server is deployed."""
        try:
            data = parse_iaga2002(text)
        except IagaParseError as exc:
            raise ProjectError(str(exc)) from exc
        return self._preview_from_iaga(data)

    def fetch_intermagnet_preview(self, iaga_code: str, dates: list) -> dict:
        """Download and parse an INTERMAGNET observatory's IAGA-2002 data
        for exactly the given calendar dates (typically the survey's own
        distinct flight dates - see _survey_dates - not a padded
        min-to-max range). Requires this server's own outbound network to
        reach the BGS GIN web service - see
        intermagnet.py::fetch_iaga2002_text for what to do when that's
        blocked (e.g. a locked-down deployment network) - use
        preview_intermagnet_text with a manually downloaded file instead.

        Any requested date that comes back entirely or mostly missing
        (definitive data for the most recent day or two often isn't
        published yet, and any specific day may simply be absent) is
        estimated from its own immediate neighbors - see
        processing/intermagnet.py::fetch_observatory_dates - and disclosed
        via estimated_dates in the returned preview."""
        try:
            data, estimated_dates = fetch_observatory_dates(iaga_code, dates)
        except IntermagnetFetchError as exc:
            raise ProjectError(str(exc)) from exc
        return self._preview_from_iaga(data, estimated_dates)

    def apply_intermagnet_preview(self) -> dict:
        """Commit the most recently previewed observatory data as this
        project's base (diurnal) station series - a separate confirm step
        so the user sees the station name/location/distance before it's
        used, rather than it being silently substituted."""
        if self.intermagnet_preview is None:
            raise ProjectError("먼저 INTERMAGNET 관측소 자료를 업로드하거나 다운로드하여 미리보기를 확인하세요.")
        data = self.intermagnet_preview
        self.base_raw = data.df
        self.base_source = {
            "type": "intermagnet",
            "station_name": data.station_name,
            "iaga_code": data.iaga_code,
            "lat": data.lat,
            "lon": data.lon,
        }
        self.intermagnet_preview = None
        self.mark_changed()
        return self.base_summary()

    def fetch_nearest_intermagnet_preview(
        self,
        dates: list,
        n_stations: int = 4,
        target_lat: float | None = None,
        target_lon: float | None = None,
        max_distance_km: float | None = DEFAULT_MAX_STATION_DISTANCE_KM,
    ) -> dict:
        """Auto-select a directionally spread set of nearby INTERMAGNET
        observatories (defaulting to the survey's own average GPS
        position when target_lat/target_lon aren't given) and combine
        their data into one substitute base station series - see
        processing/intermagnet.py::select_nearest_observatories /
        estimate_base_from_observatories. Requires this server's own
        outbound network to reach the data service.

        `dates` is typically the survey's own distinct flight dates (see
        _survey_dates), not a padded min-to-max range. Any requested date
        that comes back entirely or mostly missing for a selected station
        is estimated from its own immediate neighbors - see
        select_nearest_observatories's docstring and the per-station
        estimated_dates in the returned preview.

        max_distance_km caps candidate distance (see
        select_nearest_observatories's docstring) - n_stations selected
        may come back fewer than requested if the cutoff leaves fewer
        than n_stations candidates; the returned dict's n_requested vs.
        len(stations) tells the caller whether that happened."""
        if target_lat is None or target_lon is None:
            centroid = self._survey_centroid()
            if centroid is None:
                raise ProjectError("대상 좌표가 없습니다 - 드론 자료를 먼저 업로드하거나 좌표를 직접 입력하세요.")
            target_lat, target_lon = centroid
        try:
            selection_report: dict = {}
            stations, estimated_dates_by_code = select_nearest_observatories(
                target_lat, target_lon, dates, n_stations=n_stations, max_distance_km=max_distance_km,
                report=selection_report,
            )
            combined = estimate_base_from_observatories(stations, target_lat, target_lon)
        except IntermagnetFetchError as exc:
            raise ProjectError(str(exc)) from exc

        result = {
            "target_lat": target_lat,
            "target_lon": target_lon,
            "stations": stations,
            "df": combined,
            "estimated_dates_by_code": estimated_dates_by_code,
        }
        self.intermagnet_nearest_preview = result
        self.intermagnet_nearest_last_result = result
        return {
            "target_lat": target_lat,
            "target_lon": target_lon,
            "stations": [
                {
                    "station_name": s.station_name,
                    "iaga_code": s.iaga_code,
                    "lat": s.lat,
                    "lon": s.lon,
                    "distance_km": haversine_km(target_lat, target_lon, s.lat, s.lon),
                    "bearing_deg": _bearing_deg(target_lat, target_lon, s.lat, s.lon),
                    "estimated_dates": estimated_dates_by_code.get(s.iaga_code, []),
                }
                for s in stations
            ],
            "n_requested": n_stations,
            "max_distance_km": max_distance_km,
            "n_points": len(combined),
            "time_range": [combined["timestamp"].min().isoformat(), combined["timestamp"].max().isoformat()],
            # Where the set of contributing observatories changed and the
            # blend had to be levelled across the handover. Reported rather
            # than left silent: a handover is where this estimate is least
            # trustworthy, and because station data is fetched a calendar
            # day at a time, handovers land at midnight - which is exactly
            # where a flight that starts at 00:00 sits. See
            # processing/intermagnet.py::_remove_handover_steps.
            "station_handovers": combined.attrs.get("station_handovers", []),
            # Stations passed over because they could not supply every
            # flight date (usually a network timeout or an unpublished
            # day), and whether the result had to fall back to incomplete
            # stations because none could.
            "dropped_stations": selection_report.get("dropped", []),
            "used_incomplete_stations": selection_report.get("used_incomplete_stations", False),
        }

    def apply_nearest_intermagnet_preview(self) -> dict:
        """Commit the most recently previewed multi-observatory estimate
        as this project's base (diurnal) station series - a separate
        confirm step, matching apply_intermagnet_preview's UX."""
        if self.intermagnet_nearest_preview is None:
            raise ProjectError("먼저 주변 관측소 자료를 조회하여 미리보기를 확인하세요.")
        preview = self.intermagnet_nearest_preview
        self.base_raw = preview["df"]
        self.base_source = {
            "type": "intermagnet_nearest",
            "station_name": " + ".join(s.iaga_code for s in preview["stations"]),
            "iaga_code": None,
            "lat": preview["target_lat"],
            "lon": preview["target_lon"],
        }
        self.intermagnet_nearest_preview = None
        self.mark_changed()
        return self.base_summary()

    def get_nearest_intermagnet_comparison(self) -> dict:
        """The individual raw series of each station used in the most
        recent nearest-observatory fetch, alongside the combined (IDW)
        estimate built from them - for the user to visually sanity-check
        the estimate against the real station data (applied or still just
        previewed; see intermagnet_nearest_last_result)."""
        if self.intermagnet_nearest_last_result is None:
            raise ProjectError("먼저 주변 관측소 자료를 조회하세요.")
        result = self.intermagnet_nearest_last_result
        target_lat, target_lon = result["target_lat"], result["target_lon"]
        combined = result["df"]
        estimated_dates_by_code = result.get("estimated_dates_by_code", {})
        return {
            "combined": {
                "timestamp": combined["timestamp"].astype(str).tolist(),
                "mag": combined["mag"].tolist(),
            },
            "stations": [
                {
                    "station_name": s.station_name,
                    "iaga_code": s.iaga_code,
                    "distance_km": haversine_km(target_lat, target_lon, s.lat, s.lon),
                    "bearing_deg": _bearing_deg(target_lat, target_lon, s.lat, s.lon),
                    "timestamp": s.df["timestamp"].astype(str).tolist(),
                    "mag": s.df["mag"].tolist(),
                    "estimated_dates": estimated_dates_by_code.get(s.iaga_code, []),
                }
                for s in result["stations"]
            ],
        }

    def export_nearest_intermagnet_csv(self) -> bytes:
        """The combined multi-observatory estimate as plain CSV - for
        inspection, or reuse in another project via the IAGA-2002-style
        manual upload path (see preview_intermagnet_text) - though this
        export is plain (timestamp, mag) CSV, not IAGA-2002 fixed-width."""
        if self.intermagnet_nearest_preview is None:
            raise ProjectError("먼저 주변 관측소 자료를 조회하여 미리보기를 확인하세요.")
        df = self.intermagnet_nearest_preview["df"]
        out = df.rename(columns={"mag": "mag_nT"}).copy()
        out["timestamp"] = out["timestamp"].astype(str)
        return out.to_csv(index=False).encode("utf-8")
