from unittest.mock import Mock

import pytest
import requests

import config
import geocoding


def test_search_places_returns_parsed_results(monkeypatch):
    def fake_get(url, params, timeout):
        assert url == config.GEOCODING_URL
        assert params["name"] == "Algiers"
        assert params["count"] == config.GEOCODING_RESULT_COUNT

        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "results": [
                {
                    "name": "Algiers",
                    "latitude": 36.7538,
                    "longitude": 3.0588,
                    "country": "Algeria",
                    "admin1": "Algiers",
                },
                {
                    "name": "Algiers",
                    "latitude": 30.0,
                    "longitude": 2.0,
                },
                {
                    "name": "Missing coordinates",
                    "longitude": 2.0,
                },
            ]
        }
        return response

    monkeypatch.setattr(geocoding.requests, "get", fake_get)

    places = geocoding.search_places("Algiers")

    assert len(places) == 2
    assert places[0].name == "Algiers"
    assert places[0].latitude == pytest.approx(36.7538)
    assert places[0].longitude == pytest.approx(3.0588)
    assert places[0].label == "Algiers, Algeria"
    assert places[1].label == "Algiers"


def test_search_places_empty_query_returns_empty_list():
    assert geocoding.search_places("   ") == []


def test_search_places_returns_empty_list_when_no_results(monkeypatch):
    response = Mock()
    response.status_code = 200
    response.json.return_value = {}
    monkeypatch.setattr(geocoding.requests, "get", lambda *args, **kwargs: response)

    assert geocoding.search_places("Nowhere") == []


def test_search_places_maps_timeout_to_clear_error(monkeypatch):
    def fake_get(*args, **kwargs):
        raise requests.Timeout()

    monkeypatch.setattr(geocoding.requests, "get", fake_get)

    with pytest.raises(geocoding.GeocodingError, match="timed out"):
        geocoding.search_places("Algiers")


def test_search_places_maps_http_error(monkeypatch):
    response = Mock()
    response.status_code = 500
    monkeypatch.setattr(geocoding.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(geocoding.GeocodingError, match="HTTP 500"):
        geocoding.search_places("Algiers")
