from datetime import date
from pathlib import Path
from unittest.mock import patch

import folium
import pytest
from streamlit.testing.v1 import AppTest

import app as app_module
import config
from geocoding import Place
from predictor import Prediction
from scan import ScanPoint, ScanResult

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_starts_with_world_view():
    with patch("streamlit_folium.st_folium", return_value=None):
        at = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    assert not at.exception
    assert at.session_state["lat"] == config.DEFAULT_MAP_CENTER[0]
    assert at.session_state["lon"] == config.DEFAULT_MAP_CENTER[1]
    assert at.session_state["map_focus"] == {
        "center": list(config.DEFAULT_MAP_CENTER),
        "zoom": config.DEFAULT_MAP_ZOOM,
    }
    assert at.session_state["map_nonce"] == 0
    assert at.checkbox(key="scan_area").value is True


def test_place_search_moves_map_and_updates_coordinates():
    places = [
        Place(
            name="Algiers",
            latitude=36.7538,
            longitude=3.0588,
            country="Algeria",
            admin1="Algiers",
        ),
        Place(
            name="Paris",
            latitude=48.8566,
            longitude=2.3522,
            country="France",
        ),
    ]

    with (
        patch("streamlit_folium.st_folium", return_value=None),
        patch("geocoding.search_places", return_value=places),
    ):
        at = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
        at.text_input(key="place_query").set_value("Algiers").run()
        at.button(key="place_search").click().run()

        assert not at.exception
        at.selectbox(key="place_result").select("Algiers, Algeria").run()

    assert not at.exception
    assert at.session_state["lat"] == pytest.approx(places[0].latitude)
    assert at.session_state["lon"] == pytest.approx(places[0].longitude)
    assert at.session_state["map_focus"]["center"] == pytest.approx(
        [places[0].latitude, places[0].longitude]
    )
    assert at.session_state["map_focus"]["zoom"] == config.PLACE_MAP_ZOOM
    assert at.session_state["map_nonce"] == 1


def test_map_has_three_base_layers_and_a_control(monkeypatch):
    captured = {}

    def fake_st_folium(fig, **kwargs):
        captured["map"] = fig
        captured["kwargs"] = kwargs
        return None

    monkeypatch.setattr("streamlit_folium.st_folium", fake_st_folium)

    at = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    assert not at.exception

    # Only clicks may rerun the app. Zoom, pan and layer changes must not.
    assert captured["kwargs"]["returned_objects"] == ["last_clicked"]

    fig = captured["map"]
    tiles = [
        child
        for child in fig._children.values()
        if isinstance(child, folium.TileLayer)
    ]

    assert [tile.layer_name for tile in tiles] == [
        "Dark",
        "Satellite",
        "Streets",
    ]
    assert all(tile.options["no_wrap"] is True for tile in tiles)

    # "Dark" is the visible layer at start, the other two wait for the control.
    assert [tile.show for tile in tiles] == [True, False, False]

    controls = [
        child
        for child in fig._children.values()
        if isinstance(child, folium.LayerControl)
    ]
    assert len(controls) == 1
    assert controls[0].options["position"] == "topleft"
    assert controls[0].options["collapsed"] is False

    rendered = fig.get_root().render()
    # Dark now uses the free Esri dark canvas tiles, no API key needed.
    assert "Canvas/World_Dark_Gray_Base/MapServer/tile" in rendered
    assert "Tiles \\u0026copy; Esri" in rendered
    assert "cartocdn.com" not in rendered
    assert "World_Imagery" in rendered
    assert "tile.openstreetmap.org" in rendered
    assert "L.control.layers" in rendered

    # Esri dark tiles stop at zoom 16.
    dark = tiles[0]
    assert dark.options["max_zoom"] == 16


def test_map_shows_small_marker_on_the_selected_point(monkeypatch):
    captured = {}

    def fake_st_folium(fig, **kwargs):
        captured["map"] = fig
        captured["group"] = kwargs.get("feature_group_to_add")
        return None

    monkeypatch.setattr("streamlit_folium.st_folium", fake_st_folium)

    at = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    assert not at.exception

    # The marker is dynamic, so it lives in the feature group, not the base map.
    markers = [
        child
        for child in captured["group"]._children.values()
        if isinstance(child, folium.CircleMarker)
        and not isinstance(child, folium.Circle)
    ]
    assert len(markers) == 1
    assert not any(
        isinstance(child, folium.CircleMarker)
        for child in captured["map"]._children.values()
    )

    marker = markers[0]
    assert marker.location == pytest.approx(list(config.DEFAULT_MAP_CENTER))
    assert marker.options["radius"] <= 10
    assert "Selected point" in {
        getattr(child, "text", None)
        for child in marker._children.values()
    }


def test_map_keeps_view_after_a_click(monkeypatch):
    captured = {}
    click = {"last_clicked": {"lat": 10.0, "lng": 20.0}}

    def fake_st_folium(fig, **kwargs):
        captured["map"] = fig
        captured["kwargs"] = kwargs
        return dict(click)

    monkeypatch.setattr("streamlit_folium.st_folium", fake_st_folium)

    at = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    assert not at.exception

    # The click moved the selected point...
    assert at.session_state["lat"] == pytest.approx(10.0)
    assert at.session_state["lon"] == pytest.approx(20.0)
    # ...but a click must not change the map focus or force a remount.
    assert at.session_state["map_focus"] == {
        "center": list(config.DEFAULT_MAP_CENTER),
        "zoom": config.DEFAULT_MAP_ZOOM,
    }
    assert at.session_state["map_nonce"] == 0
    assert captured["map"].location == pytest.approx(list(config.DEFAULT_MAP_CENTER))
    assert captured["map"].options["zoom"] == config.DEFAULT_MAP_ZOOM


def test_place_search_changes_the_map_focus_and_nonce(monkeypatch):
    places = [
        Place(
            name="Algiers",
            latitude=36.7538,
            longitude=3.0588,
            country="Algeria",
            admin1="Algiers",
        )
    ]
    captured = {}

    def fake_st_folium(fig, **kwargs):
        captured["map"] = fig
        captured["kwargs"] = kwargs
        return {"last_clicked": None}

    monkeypatch.setattr("streamlit_folium.st_folium", fake_st_folium)

    with patch("geocoding.search_places", return_value=places):
        at = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
        at.text_input(key="place_query").set_value("Algiers").run()
        at.button(key="place_search").click().run()
        at.selectbox(key="place_result").select("Algiers, Algeria").run()

    assert not at.exception
    assert at.session_state["map_focus"]["center"] == pytest.approx(
        [36.7538, 3.0588]
    )
    assert at.session_state["map_focus"]["zoom"] == config.PLACE_MAP_ZOOM
    # The nonce changes so the map remounts once at the new place.
    assert at.session_state["map_nonce"] == 1
    assert captured["map"].location == pytest.approx([36.7538, 3.0588])
    assert captured["map"].options["zoom"] == config.PLACE_MAP_ZOOM
    assert captured["kwargs"]["key"] == "fire_map_1"


def test_training_rectangle_is_only_drawn_when_bounds_are_set(monkeypatch):
    captured = {}

    def fake_st_folium(fig, **kwargs):
        captured["group"] = kwargs.get("feature_group_to_add")
        return None

    monkeypatch.setattr("streamlit_folium.st_folium", fake_st_folium)

    at = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    assert not at.exception
    assert not _has_rectangle(captured["group"])

    monkeypatch.setattr(
        config,
        "TRAINING_BOUNDS",
        {
            "lat_min": 19.5,
            "lat_max": 37.5,
            "lon_min": -8.5,
            "lon_max": 10.5,
        },
    )

    at = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    assert not at.exception
    assert _has_rectangle(captured["group"])


def _has_rectangle(group: folium.FeatureGroup) -> bool:
    return any(
        isinstance(child, folium.Rectangle) for child in group._children.values()
    )


def _sample_prediction() -> Prediction:
    return Prediction(
        score=0.70,
        index=56,
        risk_level="High",
        level_color="#e5b93c",
        is_demo=True,
        model_version="0.1.0-demo",
        message="Demo score.",
    )


def test_gauge_html_shows_index_and_level_color():
    html = app_module._gauge_html(56, "#e5b93c")

    assert "RISK INDEX" in html
    assert ">56<" in html
    assert "#e5b93c" in html
    assert "stroke-dasharray" in html
    assert 'aria-label="Risk index 56 out of 100"' in html


def test_result_card_html_shows_level_coordinates_and_dates():
    html = app_module._result_card_html(
        _sample_prediction(),
        lat=36.7538,
        lon=3.0588,
        weather_as_of=date(2026, 11, 4),
        requested_date=date(2026, 11, 8),
    )

    assert "RISK INDEX" in html
    assert "High" in html
    assert "36.7538, 3.0588" in html
    assert "Date: 2026-11-08" in html
    assert "Weather data as of 2026-11-04" in html


def test_legend_html_lists_all_five_levels():
    html = app_module._legend_html()

    assert "Fire Risk Levels" in html
    assert "Relative index, not a probability" in html
    assert "position: absolute" in html
    assert "top: 12px" in html
    assert "right: 12px" in html
    assert "background: #ffffff" in html

    for level in config.RISK_LEVELS:
        assert level["name"] in html
        assert level["color"] in html

    for band in ("0-20", "20-40", "40-60", "60-80", "80-100"):
        assert band in html


def test_map_contains_the_legend(monkeypatch):
    captured = {}

    def fake_st_folium(fig, **kwargs):
        captured["map"] = fig
        return None

    monkeypatch.setattr("streamlit_folium.st_folium", fake_st_folium)

    at = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    assert not at.exception

    rendered = captured["map"].get_root().render()
    assert "Fire Risk Levels" in rendered
    assert "Relative index, not a probability" in rendered


def test_base_map_is_identical_when_results_change(monkeypatch):
    """The base map must not change when new results appear.

    If the base map HTML changed, Streamlit would rebuild the map component,
    and the browser would drop the user's zoom and chosen base layer.
    """
    from streamlit_folium import generate_js_hash

    captured = []

    def fake_st_folium(fig, **kwargs):
        captured.append((fig, kwargs))
        return None

    monkeypatch.setattr("streamlit_folium.st_folium", fake_st_folium)

    at = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    assert not at.exception

    at.session_state["scan_points"] = [
        _sample_scan_point(56, 20.0, 0.0, is_center=True),
        _sample_scan_point(90, 21.0, 0.0),
    ]
    at.run()
    assert not at.exception

    (first_map, first_kwargs), (second_map, second_kwargs) = captured[-2:]

    def component_key(fig, kwargs):
        # Same normalisation streamlit-folium uses to decide whether to remount.
        return generate_js_hash(fig.get_root().render(), kwargs["key"])

    assert first_kwargs["key"] == second_kwargs["key"] == "fire_map_0"
    assert component_key(first_map, first_kwargs) == component_key(
        second_map, second_kwargs
    )
    # The dynamic part really did change: two more scan circles.
    assert len(second_kwargs["feature_group_to_add"]._children) == (
        len(first_kwargs["feature_group_to_add"]._children) + 2
    )


def test_result_card_renders_after_a_stored_prediction():
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["result"] = _sample_prediction()
    at.session_state["result_lat"] = 36.7538
    at.session_state["result_lon"] = 3.0588
    at.session_state["weather_as_of"] = date(2026, 11, 4)
    at.session_state["requested_date"] = date(2026, 11, 8)

    with patch("streamlit_folium.st_folium", return_value=None):
        at.run()

    assert not at.exception
    markdown_text = "\n".join(item.value for item in at.markdown)
    assert "RISK INDEX" in markdown_text
    assert "High" in markdown_text
    assert "36.7538, 3.0588" in markdown_text
    assert "Weather data as of 2026-11-04" in markdown_text
    assert "DEMO MODE" in "\n".join(item.value for item in at.error)


def _sample_scan_point(
    index: int,
    lat: float,
    lon: float,
    is_center: bool = False,
) -> ScanPoint:
    prediction = Prediction(
        score=index / 100,
        index=index,
        risk_level="High" if index < 80 else "Extreme",
        level_color="#e5b93c" if index < 80 else "#d32f2f",
        is_demo=True,
        model_version="0.1.0-demo",
        message="Demo score.",
    )
    return ScanPoint(
        lat=lat,
        lon=lon,
        prediction=prediction,
        as_of=date(2026, 11, 4),
        is_center=is_center,
    )


def test_scan_summary_html_shows_stats_and_count():
    points = [
        _sample_scan_point(40, 0.0, 0.0, is_center=True),
        _sample_scan_point(60, 1.0, 0.0),
        _sample_scan_point(80, 0.0, 1.0),
    ]

    html = app_module._scan_summary_html(points, 9)

    assert "Average" in html
    assert "Min" in html
    assert "Max" in html
    assert ">60<" in html
    assert ">40<" in html
    assert ">80<" in html
    assert "Points scanned: 3 of 9" in html


def test_map_draws_scan_circles(monkeypatch):
    captured = {}

    def fake_st_folium(fig, **kwargs):
        captured["group"] = kwargs.get("feature_group_to_add")
        return None

    monkeypatch.setattr("streamlit_folium.st_folium", fake_st_folium)

    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["scan_points"] = [
        _sample_scan_point(56, 20.0, 0.0, is_center=True),
        _sample_scan_point(90, 21.0, 0.0),
    ]
    at.run()

    assert not at.exception
    circles = [
        child
        for child in captured["group"]._children.values()
        if isinstance(child, folium.Circle)
    ]
    assert len(circles) == 2

    for circle in circles:
        assert circle.options["radius"] == config.SCAN_CIRCLE_RADIUS_METERS
        assert circle.options["fillOpacity"] == 0.35

    assert {circle.options["fillColor"] for circle in circles} == {
        "#e5b93c",
        "#d32f2f",
    }

    tooltips = {
        getattr(child, "text", None)
        for circle in circles
        for child in circle._children.values()
    }
    assert "High - index 56" in tooltips
    assert "Extreme - index 90" in tooltips


def test_analyze_with_scan_shows_summary_and_circles(monkeypatch):
    captured = {}

    def fake_st_folium(fig, **kwargs):
        captured["group"] = kwargs.get("feature_group_to_add")
        return None

    monkeypatch.setattr("streamlit_folium.st_folium", fake_st_folium)

    center = _sample_scan_point(56, 20.0, 0.0, is_center=True)
    other = _sample_scan_point(72, 21.0, 0.0)
    fake_result = ScanResult(points=[center, other], total=9)

    def fake_scan_area_points(lat, lon, target_date, fetch_weather):
        return fake_result

    monkeypatch.setattr("scan.scan_area_points", fake_scan_area_points)

    at = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    at.button(key="analyze").click().run()

    assert not at.exception
    assert at.session_state["scan_requested"] is True
    assert at.session_state["result"] is center.prediction
    assert at.session_state["scan_points"] == [center, other]

    text = "\n".join(item.value for item in at.markdown)
    assert "Points scanned: 2 of 9" in text
    assert "Average" in text

    circles = [
        child
        for child in captured["group"]._children.values()
        if isinstance(child, folium.Circle)
    ]
    assert len(circles) == 2


def test_analyze_without_scan_skips_the_grid(monkeypatch):
    import pandas as pd

    from weather import WeatherData

    captured = {}

    def fake_st_folium(fig, **kwargs):
        captured["group"] = kwargs.get("feature_group_to_add")
        return None

    monkeypatch.setattr("streamlit_folium.st_folium", fake_st_folium)

    def fake_download(lat, lon, target_date, timeout=30):
        return WeatherData(
            frame=pd.DataFrame(),
            as_of=date(2026, 11, 4),
            requested=target_date,
        )

    monkeypatch.setattr("weather.download_daily_weather", fake_download)
    monkeypatch.setattr("features.build_features", lambda frame, as_of: {})
    monkeypatch.setattr(
        "predictor.predict",
        lambda features: _sample_prediction(),
    )

    called = {"scan": False}

    def fake_scan_area_points(*args, **kwargs):
        called["scan"] = True
        return ScanResult(points=[], total=9)

    monkeypatch.setattr("scan.scan_area_points", fake_scan_area_points)

    at = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    at.checkbox(key="scan_area").uncheck().run()
    at.button(key="analyze").click().run()

    assert not at.exception
    assert called["scan"] is False
    assert at.session_state["scan_requested"] is False
    assert at.session_state["scan_points"] == []

    text = "\n".join(item.value for item in at.markdown)
    assert "Points scanned" not in text

    circles = [
        child
        for child in captured["group"]._children.values()
        if isinstance(child, folium.Circle)
    ]
    assert circles == []


def test_weather_downloads_are_cached(monkeypatch):
    calls = []

    def fake_download(lat, lon, target_date, timeout=30):
        calls.append((lat, lon, target_date, timeout))
        return "weather"

    monkeypatch.setattr(app_module, "download_daily_weather", fake_download)
    app_module.cached_weather.clear()

    try:
        first = app_module.cached_weather(
            12.5,
            34.5,
            date(2026, 11, 4),
            25,
        )
        second = app_module.cached_weather(
            12.5,
            34.5,
            date(2026, 11, 4),
            25,
        )
    finally:
        app_module.cached_weather.clear()

    assert config.WEATHER_CACHE_TTL_SECONDS == 3600
    assert first == second == "weather"
    assert len(calls) == 1
