"""Download and parse NASA POWER daily weather for a single point.

This module only handles weather data. It does not build features and it does
not touch the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

import config


class WeatherDataError(Exception):
    """Raised when weather data cannot be downloaded or used."""


@dataclass(frozen=True)
class WeatherData:
    """A usable weather window plus the date the data actually ends on."""

    frame: pd.DataFrame
    as_of: date
    requested: date

    @property
    def is_stale(self) -> bool:
        """True when the newest available day is older than the requested day."""
        return self.as_of < self.requested


def download_daily_weather(
    lat: float,
    lon: float,
    end_date: date,
    lookback_days: int = config.HISTORY_DAYS,
    timeout: float = config.REQUEST_TIMEOUT_SECONDS,
) -> WeatherData:
    """Download daily weather and return the latest usable history window.

    If NASA POWER does not have the selected date yet, the newest available
    day is used instead. The caller can show ``WeatherData.as_of`` in the UI.
    """
    requested = end_date
    api_end = min(requested, date.today())

    # Ask for extra days so we can still return a full window when POWER lags.
    api_start = api_end - timedelta(
        days=lookback_days + config.POWER_REQUEST_BUFFER_DAYS - 1
    )

    payload = _request_power(lat, lon, api_start, api_end, timeout)
    frame = _parse_power_payload(payload)
    frame = _normalize_weather_frame(frame)

    if frame.empty:
        raise WeatherDataError("NASA POWER returned no weather data for this point.")

    # Use the selected date when it exists, otherwise use the newest day we have.
    available = frame.loc[frame.index <= pd.Timestamp(requested)]
    if available.empty:
        raise WeatherDataError(
            f"No weather data is available on or before {requested:%Y-%m-%d}."
        )

    as_of = available.index.max().date()
    window = frame.loc[frame.index <= pd.Timestamp(as_of)].tail(lookback_days).copy()

    if len(window) < config.MIN_HISTORY_DAYS:
        raise WeatherDataError(
            "Not enough weather history was returned. Please try an earlier date."
        )

    return WeatherData(frame=window, as_of=as_of, requested=requested)


def _request_power(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    timeout: float = config.REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Call the NASA POWER daily point endpoint."""
    params = {
        "parameters": ",".join(config.POWER_PARAMETERS),
        "community": config.POWER_COMMUNITY,
        "longitude": lon,
        "latitude": lat,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
    }

    try:
        response = requests.get(
            config.POWER_BASE_URL,
            params=params,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise WeatherDataError(
            "The weather service timed out. Please try again."
        ) from exc
    except requests.RequestException as exc:
        raise WeatherDataError(
            "Could not reach the weather service. Please try again."
        ) from exc

    if response.status_code != 200:
        raise WeatherDataError(
            f"The weather service returned an error (HTTP {response.status_code})."
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise WeatherDataError(
            "The weather service returned an unreadable response."
        ) from exc

    _raise_for_power_error(payload)
    return payload


def _raise_for_power_error(payload: dict[str, Any]) -> None:
    """Turn NASA POWER error messages into a clear WeatherDataError."""
    messages = payload.get("messages") or []
    for message in messages:
        if isinstance(message, dict):
            text = message.get("message")
            kind = str(message.get("type", "")).lower()
        else:
            text = str(message)
            kind = ""

        if text and (kind == "error" or "error" in text.lower()):
            raise WeatherDataError(f"NASA POWER error: {text}")

    parameter = payload.get("properties", {}).get("parameter")
    if not parameter:
        raise WeatherDataError("NASA POWER returned no weather parameters.")


def _parse_power_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert the nested POWER JSON into a date-indexed DataFrame."""
    parameter = payload.get("properties", {}).get("parameter")
    if not parameter:
        raise WeatherDataError("NASA POWER returned no weather parameters.")

    frame = pd.DataFrame(parameter)
    if frame.empty:
        raise WeatherDataError("NASA POWER returned an empty weather table.")

    try:
        frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    except (TypeError, ValueError) as exc:
        raise WeatherDataError(
            "NASA POWER returned dates in an unexpected format."
        ) from exc

    frame.index.name = "date"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _normalize_weather_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the six expected columns, clean missing values, and sort by date."""
    missing = [name for name in config.POWER_PARAMETERS if name not in frame.columns]
    if missing:
        raise WeatherDataError(
            "NASA POWER did not return these weather fields: "
            + ", ".join(missing)
            + "."
        )

    frame = frame.loc[:, list(config.POWER_PARAMETERS)].copy()
    frame = frame.apply(pd.to_numeric, errors="coerce")

    # POWER uses -999 for missing values.
    frame = frame.mask(frame <= -999.0)

    # NASA POWER marks days it does not have yet with -999 in every column.
    # Drop those days completely instead of filling them with older values,
    # otherwise the app would claim "weather data as of" a day that is fake.
    frame = frame.dropna(how="all")

    if frame.empty:
        raise WeatherDataError("NASA POWER returned no usable weather data.")

    # Small gaps in the middle of the window are safe to fill.
    frame = frame.interpolate(method="linear", limit_direction="both")
    frame = frame.ffill().bfill()

    if frame.isna().any().any():
        raise WeatherDataError(
            "Some weather values are missing for this point and date."
        )

    return frame.sort_index()
