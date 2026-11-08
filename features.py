"""Build the 18 model features from daily weather.

The 7/14/30-day formulas here are placeholders. They are kept in this file so
they can be replaced later without touching the UI or the predictor.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

import config


class FeatureBuildError(Exception):
    """Raised when the weather window cannot produce all features."""


def build_features(weather_df: pd.DataFrame, target_date: date) -> dict[str, float]:
    """Build the 18 features for ``target_date``.

    The returned dictionary follows ``config.FEATURES`` exactly.
    """
    _require_history(weather_df, target_date, config.MIN_HISTORY_DAYS)

    frame = weather_df.sort_index().loc[: pd.Timestamp(target_date)]
    row = frame.loc[pd.Timestamp(target_date)]
    doy_sin, doy_cos = _day_of_year_encoding(target_date)

    features = {
        "T2M": float(row["T2M"]),
        "T2M_MAX": float(row["T2M_MAX"]),
        "RH2M": float(row["RH2M"]),
        "WS2M": float(row["WS2M"]),
        "PRECTOTCORR": float(row["PRECTOTCORR"]),
        "GWETROOT": float(row["GWETROOT"]),
        "rain_7d": _window_sum(frame["PRECTOTCORR"], target_date, 7),
        "tmax_7d": _window_max(frame["T2M_MAX"], target_date, 7),
        "rhmin_7d": _window_min(frame["RH2M"], target_date, 7),
        "rain_14d": _window_sum(frame["PRECTOTCORR"], target_date, 14),
        "tmax_14d": _window_max(frame["T2M_MAX"], target_date, 14),
        "rhmin_14d": _window_min(frame["RH2M"], target_date, 14),
        "rain_30d": _window_sum(frame["PRECTOTCORR"], target_date, 30),
        "tmax_30d": _window_max(frame["T2M_MAX"], target_date, 30),
        "rhmin_30d": _window_min(frame["RH2M"], target_date, 30),
        "dry_days": _count_dry_days(frame, target_date, 30),
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
    }
    return features


def build_feature_vector(features: dict[str, float]) -> list[float]:
    """Return values in the exact order required by the model."""
    missing = [name for name in config.FEATURES if name not in features]
    extra = [name for name in features if name not in config.FEATURES]

    if missing or extra:
        problems = []
        if missing:
            problems.append("missing: " + ", ".join(missing))
        if extra:
            problems.append("unexpected: " + ", ".join(extra))
        raise FeatureBuildError(
            "Feature set does not match config.FEATURES (" + "; ".join(problems) + ")."
        )

    return [float(features[name]) for name in config.FEATURES]


def _require_history(
    weather_df: pd.DataFrame,
    target_date: date,
    days: int,
) -> None:
    """Check that the target day and enough history are present."""
    if weather_df.empty:
        raise FeatureBuildError("No weather data was provided.")

    frame = weather_df.sort_index()
    target = pd.Timestamp(target_date)

    if target not in frame.index:
        raise FeatureBuildError(
            f"No weather row exists for {target_date:%Y-%m-%d}."
        )

    available = frame.loc[:target]
    if len(available) < days:
        raise FeatureBuildError(
            f"Need at least {days} days of weather history, "
            f"but only {len(available)} are available."
        )


def _window(series: pd.Series, target_date: date, days: int) -> pd.Series:
    """Return the last ``days`` values ending on ``target_date``."""
    return series.loc[: pd.Timestamp(target_date)].tail(days)


def _window_sum(series: pd.Series, target_date: date, days: int) -> float:
    return float(_window(series, target_date, days).sum())


def _window_max(series: pd.Series, target_date: date, days: int) -> float:
    return float(_window(series, target_date, days).max())


def _window_min(series: pd.Series, target_date: date, days: int) -> float:
    return float(_window(series, target_date, days).min())


def _count_dry_days(
    weather_df: pd.DataFrame,
    target_date: date,
    days: int,
) -> int:
    """Count days in the window with almost no rain."""
    window = _window(weather_df["PRECTOTCORR"], target_date, days)
    return int((window < config.DRY_DAY_MM).sum())


def _day_of_year_encoding(target_date: date) -> tuple[float, float]:
    """Encode the day of year as a point on a circle."""
    day_of_year = target_date.timetuple().tm_yday
    angle = 2.0 * math.pi * day_of_year / 365.25
    return math.sin(angle), math.cos(angle)
