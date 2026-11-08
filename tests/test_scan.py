from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import pytest

import config
import scan
from predictor import Prediction
from weather import WeatherData


def make_prediction(index: int = 50) -> Prediction:
    return Prediction(
        score=0.5,
        index=index,
        risk_level="High",
        level_color="#e5b93c",
        is_demo=True,
        model_version="test",
        message="",
    )


def make_weather(as_of: date = date(2026, 11, 4)) -> WeatherData:
    return WeatherData(
        frame=pd.DataFrame(),
        as_of=as_of,
        requested=as_of,
    )


@pytest.fixture
def fake_analysis(monkeypatch):
    """Skip real feature building and model calls during scan tests."""
    monkeypatch.setattr(scan, "build_features", lambda frame, as_of: {})
    monkeypatch.setattr(scan, "predict", lambda features: make_prediction())


def test_grid_points_build_nine_points_with_one_center():
    points = scan.grid_points(36.7538, 3.0588)

    assert len(points) == 9
    centers = [point for point in points if point.is_center]
    assert len(centers) == 1
    assert centers[0].lat == pytest.approx(36.7538)
    assert centers[0].lon == pytest.approx(3.0588)

    assert sorted({point.lat for point in points}) == pytest.approx(
        [35.7538, 36.7538, 37.7538]
    )
    assert sorted({point.lon for point in points}) == pytest.approx(
        [2.0588, 3.0588, 4.0588]
    )


def test_grid_points_clamp_latitude_and_wrap_longitude():
    points = scan.grid_points(90.0, 179.0)

    assert all(-90.0 <= point.lat <= 90.0 for point in points)
    assert all(-180.0 <= point.lon <= 180.0 for point in points)
    assert {point.lat for point in points} == {89.0, 90.0}
    assert -180.0 in {point.lon for point in points}


def test_grid_points_wrap_at_the_south_west_corner():
    points = scan.grid_points(-90.0, -180.0)

    assert all(-90.0 <= point.lat <= 90.0 for point in points)
    assert all(-180.0 <= point.lon <= 180.0 for point in points)
    assert {point.lat for point in points} == {-90.0, -89.0}
    assert 179.0 in {point.lon for point in points}


def test_scan_area_points_analyzes_all_nine_points(fake_analysis):
    calls = []

    def fetch_weather(lat, lon, target_date, timeout):
        calls.append((lat, lon, target_date, timeout))
        return make_weather()

    result = scan.scan_area_points(
        10.0,
        20.0,
        date(2026, 11, 4),
        fetch_weather,
    )

    assert result.total == 9
    assert len(result.points) == 9
    assert result.center is not None
    assert result.center.lat == pytest.approx(10.0)
    assert result.center.lon == pytest.approx(20.0)
    assert len(calls) == 9
    assert {call[3] for call in calls} == {
        config.SCAN_REQUEST_TIMEOUT_SECONDS
    }


def test_scan_uses_five_workers(fake_analysis, monkeypatch):
    recorded = {}

    class RecordingExecutor(ThreadPoolExecutor):
        def __init__(self, max_workers=None, **kwargs):
            recorded["max_workers"] = max_workers
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(scan, "ThreadPoolExecutor", RecordingExecutor)

    scan.scan_area_points(
        0.0,
        0.0,
        date(2026, 11, 4),
        lambda lat, lon, target_date, timeout: make_weather(),
    )

    assert recorded["max_workers"] == 5


def test_scan_ignores_points_that_fail(fake_analysis):
    def fetch_weather(lat, lon, target_date, timeout):
        if (lat, lon) in {(-1.0, -1.0), (1.0, 1.0)}:
            raise RuntimeError("simulated download failure")
        return make_weather()

    result = scan.scan_area_points(
        0.0,
        0.0,
        date(2026, 11, 4),
        fetch_weather,
    )

    assert result.total == 9
    assert len(result.points) == 7
    assert result.center is not None


def test_scan_keeps_other_points_when_the_center_fails(fake_analysis):
    def fetch_weather(lat, lon, target_date, timeout):
        if lat == 0.0 and lon == 0.0:
            raise RuntimeError("center failed")
        return make_weather()

    result = scan.scan_area_points(
        0.0,
        0.0,
        date(2026, 11, 4),
        fetch_weather,
    )

    assert len(result.points) == 8
    assert result.center is None


def test_scan_result_summary_values():
    points = [
        scan.ScanPoint(
            lat=0.0,
            lon=0.0,
            prediction=make_prediction(10),
            as_of=date(2026, 11, 4),
            is_center=True,
        ),
        scan.ScanPoint(
            lat=1.0,
            lon=0.0,
            prediction=make_prediction(20),
            as_of=date(2026, 11, 4),
            is_center=False,
        ),
        scan.ScanPoint(
            lat=0.0,
            lon=1.0,
            prediction=make_prediction(30),
            as_of=date(2026, 11, 4),
            is_center=False,
        ),
    ]

    result = scan.ScanResult(points=points, total=9)

    assert result.indices == [10, 20, 30]
    assert result.average_index == 20
    assert result.min_index == 10
    assert result.max_index == 30


def test_empty_scan_summary_is_zero():
    result = scan.ScanResult(points=[], total=9)

    assert result.average_index == 0
    assert result.min_index == 0
    assert result.max_index == 0
