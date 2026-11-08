"""Scan a 3x3 grid of points around the selected location.

The scan is best-effort: points that fail are skipped so one bad download does
not stop the whole analysis. Weather downloads run in parallel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from typing import Callable

import config
from features import build_features
from predictor import Prediction, predict
from weather import WeatherData

# A weather fetcher takes (lat, lon, end_date, timeout) and returns WeatherData.
WeatherFetcher = Callable[[float, float, date, float], WeatherData]


@dataclass(frozen=True)
class GridPoint:
    """One position in the 3x3 scan grid."""

    lat: float
    lon: float
    is_center: bool


@dataclass(frozen=True)
class ScanPoint:
    """A successful scan result for one grid point."""

    lat: float
    lon: float
    prediction: Prediction
    as_of: date
    is_center: bool


@dataclass(frozen=True)
class ScanResult:
    """The successful points from a scan and the total grid size."""

    points: list[ScanPoint]
    total: int = config.SCAN_GRID_POINTS

    @property
    def center(self) -> ScanPoint | None:
        """Return the selected point when it was analyzed successfully."""
        for point in self.points:
            if point.is_center:
                return point
        return None

    @property
    def indices(self) -> list[int]:
        """Return the risk index of every successful point."""
        return [point.prediction.index for point in self.points]

    @property
    def average_index(self) -> int:
        """Return the rounded average index, or 0 when nothing succeeded."""
        values = self.indices
        if not values:
            return 0
        return round(sum(values) / len(values))

    @property
    def min_index(self) -> int:
        """Return the smallest index, or 0 when nothing succeeded."""
        values = self.indices
        return min(values) if values else 0

    @property
    def max_index(self) -> int:
        """Return the largest index, or 0 when nothing succeeded."""
        values = self.indices
        return max(values) if values else 0


def grid_points(lat: float, lon: float) -> list[GridPoint]:
    """Build the 3x3 grid around a point, 1 degree apart.

    Latitude is clamped to -90..90 and longitude is wrapped into -180..180.
    """
    step = config.SCAN_GRID_STEP_DEGREES
    offsets = (-step, 0.0, step)
    points: list[GridPoint] = []

    for lat_offset in offsets:
        for lon_offset in offsets:
            point_lat = max(-90.0, min(90.0, lat + lat_offset))
            point_lon = config.normalize_longitude(lon + lon_offset)
            points.append(
                GridPoint(
                    lat=round(point_lat, 5),
                    lon=round(point_lon, 5),
                    is_center=lat_offset == 0.0 and lon_offset == 0.0,
                )
            )

    return points


def scan_area_points(
    lat: float,
    lon: float,
    target_date: date,
    fetch_weather: WeatherFetcher,
) -> ScanResult:
    """Analyze all 9 grid points, ignoring any point that fails."""
    points = grid_points(lat, lon)
    results: list[ScanPoint] = []

    with ThreadPoolExecutor(max_workers=config.SCAN_WORKERS) as executor:
        futures = [
            executor.submit(
                _analyze_point,
                point,
                target_date,
                fetch_weather,
            )
            for point in points
        ]

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                # One bad point must not stop the rest of the scan.
                continue

    return ScanResult(points=results)


def _analyze_point(
    point: GridPoint,
    target_date: date,
    fetch_weather: WeatherFetcher,
) -> ScanPoint:
    """Download weather and predict one grid point."""
    weather_data = fetch_weather(
        point.lat,
        point.lon,
        target_date,
        config.SCAN_REQUEST_TIMEOUT_SECONDS,
    )
    features = build_features(weather_data.frame, weather_data.as_of)
    prediction = predict(features)

    return ScanPoint(
        lat=point.lat,
        lon=point.lon,
        prediction=prediction,
        as_of=weather_data.as_of,
        is_center=point.is_center,
    )
