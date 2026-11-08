from datetime import date, timedelta

import pandas as pd
import pytest
import requests

import config
import weather


def make_payload(dates):
    values = {
        "T2M": 20.0,
        "T2M_MAX": 30.0,
        "RH2M": 40.0,
        "WS2M": 5.0,
        "PRECTOTCORR": 0.0,
        "GWETROOT": 0.2,
    }
    return {
        "properties": {
            "parameter": {
                name: {day.strftime("%Y%m%d"): value for day in dates}
                for name, value in values.items()
            }
        }
    }


def test_parse_power_payload_builds_date_indexed_frame():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")

    frame = weather._parse_power_payload(make_payload(dates))

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert list(frame.columns) == list(config.POWER_PARAMETERS)
    assert len(frame) == 3


def test_normalize_replaces_missing_values_and_sorts():
    dates = pd.date_range("2024-01-03", periods=2, freq="D")
    payload = make_payload(dates)
    payload["properties"]["parameter"]["T2M"][dates[0].strftime("%Y%m%d")] = -999.0

    frame = weather._parse_power_payload(payload)
    frame = weather._normalize_weather_frame(frame)

    assert not frame.isna().any().any()
    assert frame.index.is_monotonic_increasing


def test_download_uses_selected_date_when_available(monkeypatch):
    requested = date(2024, 2, 7)
    dates = pd.date_range(end=pd.Timestamp(requested), periods=40, freq="D")
    monkeypatch.setattr(
        weather,
        "_request_power",
        lambda *args, **kwargs: make_payload(dates),
    )

    data = weather.download_daily_weather(30.0, 2.0, requested)

    assert data.requested == requested
    assert data.as_of == requested
    assert data.is_stale is False
    assert len(data.frame) == config.HISTORY_DAYS


def test_download_uses_latest_available_day(monkeypatch):
    requested = date(2024, 2, 10)
    latest = date(2024, 2, 7)
    dates = pd.date_range(end=pd.Timestamp(latest), periods=40, freq="D")
    monkeypatch.setattr(
        weather,
        "_request_power",
        lambda *args, **kwargs: make_payload(dates),
    )

    data = weather.download_daily_weather(30.0, 2.0, requested)

    assert data.requested == requested
    assert data.as_of == latest
    assert data.is_stale is True
    assert len(data.frame) == config.HISTORY_DAYS
    assert data.frame.index.max().date() == latest


def test_download_uses_latest_day_when_date_is_in_future(monkeypatch):
    requested = date.today() + timedelta(days=30)
    latest = date.today()
    dates = pd.date_range(end=pd.Timestamp(latest), periods=40, freq="D")
    monkeypatch.setattr(
        weather,
        "_request_power",
        lambda *args, **kwargs: make_payload(dates),
    )

    data = weather.download_daily_weather(30.0, 2.0, requested)

    assert data.as_of == latest
    assert data.is_stale is True


def test_download_ignores_trailing_missing_days(monkeypatch):
    """NASA POWER sends -999 for days it does not have yet.

    Those days must not be filled with older values, because then ``as_of``
    would point at a day with no real weather data.
    """
    requested = date(2024, 2, 10)
    latest_real = date(2024, 2, 7)
    dates = pd.date_range(end=pd.Timestamp(requested), periods=40, freq="D")
    payload = make_payload(dates)

    # Mark the last three days as missing, the way NASA POWER does.
    for day in dates[-3:]:
        key = day.strftime("%Y%m%d")
        for name in config.POWER_PARAMETERS:
            payload["properties"]["parameter"][name][key] = -999.0

    monkeypatch.setattr(
        weather,
        "_request_power",
        lambda *args, **kwargs: payload,
    )

    data = weather.download_daily_weather(30.0, 2.0, requested)

    assert data.as_of == latest_real
    assert data.is_stale is True
    assert data.frame.index.max().date() == latest_real
    assert len(data.frame) == config.HISTORY_DAYS


def test_download_raises_when_every_day_is_missing(monkeypatch):
    dates = pd.date_range(end=pd.Timestamp("2024-02-07"), periods=40, freq="D")
    payload = make_payload(dates)

    for day in dates:
        key = day.strftime("%Y%m%d")
        for name in config.POWER_PARAMETERS:
            payload["properties"]["parameter"][name][key] = -999.0

    monkeypatch.setattr(
        weather,
        "_request_power",
        lambda *args, **kwargs: payload,
    )

    with pytest.raises(weather.WeatherDataError, match="no usable weather data"):
        weather.download_daily_weather(30.0, 2.0, date(2024, 2, 7))


def test_download_raises_when_no_parameters(monkeypatch):
    monkeypatch.setattr(
        weather,
        "_request_power",
        lambda *args, **kwargs: {"properties": {"parameter": {}}},
    )

    with pytest.raises(weather.WeatherDataError):
        weather.download_daily_weather(30.0, 2.0, date(2024, 1, 1))


def test_download_raises_when_history_is_too_short(monkeypatch):
    dates = pd.date_range(end=pd.Timestamp("2024-02-07"), periods=20, freq="D")
    monkeypatch.setattr(
        weather,
        "_request_power",
        lambda *args, **kwargs: make_payload(dates),
    )

    with pytest.raises(weather.WeatherDataError):
        weather.download_daily_weather(30.0, 2.0, date(2024, 2, 7))


def test_request_power_maps_timeout(monkeypatch):
    def fake_get(*args, **kwargs):
        raise requests.Timeout()

    monkeypatch.setattr(weather.requests, "get", fake_get)

    with pytest.raises(weather.WeatherDataError, match="timed out"):
        weather._request_power(
            30.0,
            2.0,
            date(2024, 1, 1),
            date(2024, 1, 2),
        )


def test_download_passes_timeout_to_request(monkeypatch):
    requested = date(2024, 2, 7)
    dates = pd.date_range(end=pd.Timestamp(requested), periods=40, freq="D")
    captured = {}

    def fake_request(lat, lon, start_date, end_date, timeout):
        captured["timeout"] = timeout
        return make_payload(dates)

    monkeypatch.setattr(weather, "_request_power", fake_request)

    weather.download_daily_weather(30.0, 2.0, requested, timeout=25)

    assert captured["timeout"] == 25


def test_request_power_uses_given_timeout(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "properties": {
                    "parameter": {"T2M": {"20240101": 20.0}}
                }
            }

    def fake_get(*args, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(weather.requests, "get", fake_get)

    weather._request_power(
        30.0,
        2.0,
        date(2024, 1, 1),
        date(2024, 1, 2),
        timeout=25,
    )

    assert captured["timeout"] == 25
