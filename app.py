"""Streamlit interface for the fire risk app.

This file only handles the interface and the thin orchestration between
weather, features, and predictor.
"""

from __future__ import annotations

import math
from datetime import date

import folium
import streamlit as st
from streamlit_folium import st_folium

import config
from features import FeatureBuildError, build_features
from geocoding import GeocodingError, Place, search_places
from predictor import (
    ModelLoadError,
    Prediction,
    PredictionError,
    predict,
)
from scan import ScanPoint, scan_area_points
from weather import WeatherDataError, download_daily_weather

# One CSS block for the whole dashboard. It hides the Streamlit chrome,
# turns the sidebar into a dark navy control panel, and lets the map fill
# the rest of the window.
DASHBOARD_CSS = """
<style>
/* ---- Hide Streamlit header, footer and toolbar ---- */
#MainMenu,
header[data-testid="stHeader"],
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}

/* ---- Page background and zero padding so the map is edge to edge ---- */
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #0b1230 !important;
}

[data-testid="stMainBlockContainer"],
.block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}

[data-testid="stMainBlockContainer"] .stVerticalBlock {
    gap: 0 !important;
}

/* ---- Left panel: dark navy gradient, fixed width ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1a3a 0%, #0b1230 100%) !important;
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

[data-testid="stSidebar"]:not([data-collapsed="true"]) {
    width: 380px !important;
    min-width: 380px !important;
    max-width: 380px !important;
}

[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    background: transparent !important;
}

[data-testid="stSidebarUserContent"] {
    padding: 1.4rem 1.15rem 2rem 1.15rem !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #eef2ff !important;
}

/* ---- Panel title ---- */
.risk-panel-title {
    color: #ffffff;
    font-size: 1.55rem;
    font-weight: 750;
    letter-spacing: 0.01em;
    margin: 0 0 1.1rem 0;
}

/* ---- Rounded cards with spacing between them ---- */
.st-key-card-search,
.st-key-card-date,
.st-key-card-point,
.st-key-card-result,
.st-key-card-scan {
    background: #151f45 !important;
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 12px !important;
    padding: 0.9rem !important;
    margin-bottom: 0.85rem !important;
    box-shadow: 0 10px 24px rgba(2, 6, 23, 0.35);
}

[data-testid="stSidebar"] h5 {
    font-size: 1rem !important;
    margin: 0 0 0.4rem 0 !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #93a4c8 !important;
    font-size: 0.72rem !important;
    line-height: 1.25 !important;
}

/* ---- Dark inputs: text box, date box and select box ----
   Streamlit 1.67 renders these with data-testid wrappers, not BaseWeb tags. */
[data-testid="stSidebar"] [data-testid="stTextInputRootElement"],
[data-testid="stSidebar"] [data-testid="stDateInputField"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {
    background: #0f1a3a !important;
    background-color: #0f1a3a !important;
    border: 1px solid #2a3766 !important;
    border-radius: 8px !important;
}

/* Text typed into the inputs must be white, not the default dark colour. */
[data-testid="stSidebar"] [data-testid="stTextInputField"],
[data-testid="stSidebar"] [data-testid="stDateInputField"] input,
[data-testid="stSidebar"] [data-testid="stDateInputFieldsScroller"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] input {
    background: transparent !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* The date shows as year / month / day segments, not a plain input. */
[data-testid="stSidebar"] [data-testid="stDateInputField"] span[role="spinbutton"] {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Placeholder text stays readable but quieter than real text. */
[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
    color: #8a96c4 !important;
    -webkit-text-fill-color: #8a96c4 !important;
    opacity: 1 !important;
}

/* The calendar icon inside the date box and the select box arrow. */
[data-testid="stSidebar"] [data-testid="stDateInputField"] svg,
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
    fill: #8a96c4 !important;
    color: #8a96c4 !important;
}

/* ---- Buttons ---- */
[data-testid="stSidebar"] .stButton > button {
    background: #1e2b5a !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    font-weight: 600;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #27376e !important;
    border-color: rgba(148, 163, 184, 0.35) !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(234, 88, 12, 0.35);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #fb923c 0%, #f97316 100%) !important;
}

/* ---- Alerts and result text ---- */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    border-radius: 10px !important;
}

[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* ---- Dark dropdown list and date picker popup ----
   These render outside the sidebar in a portal, so they are not scoped to
   [data-testid="stSidebar"]. */

/* The selectbox results list. */
[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stSelectboxVirtualDropdown"] [role="listbox"] {
    background: #0f1a3a !important;
    border: 1px solid #2a3766 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

[data-testid="stSelectboxVirtualDropdown"] [role="option"],
[data-testid="stSelectboxVirtualDropdown"] li {
    background: #0f1a3a !important;
    color: #ffffff !important;
}

[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
[data-testid="stSelectboxVirtualDropdown"] li:hover {
    background: #1e2b5a !important;
    color: #ffffff !important;
}

/* Calendar popup for the date input. */
[data-testid="stDateInputCalendar"] {
    background: #0f1a3a !important;
    border: 1px solid #2a3766 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

[data-testid="stDateInputCalendar"] * {
    color: #ffffff !important;
}

[data-testid="stDateInputCalendar"] button {
    background: transparent !important;
    color: #ffffff !important;
}

[data-testid="stDateInputCalendar"] button:hover,
[data-testid="stDateInputCalendar"] div[role="button"]:hover {
    background: #1e2b5a !important;
    border-radius: 6px !important;
}

/* The selected day and the weekday header row. */
[data-testid="stDateInputCalendar"] [data-selected="true"] {
    background: #f97316 !important;
    color: #ffffff !important;
    border-radius: 6px !important;
}

[data-testid="stDateInputCalendar"] th {
    color: #8a96c4 !important;
}

/* The month / year dropdowns inside the calendar (visually hidden but kept
   dark so nothing flashes white). */
[data-testid="stDateInputCalendar"] select {
    background: #0f1a3a !important;
    color: #ffffff !important;
}

/* ---- Result card: gauge and details ---- */
.risk-gauge {
    display: flex;
    justify-content: center;
    margin: 0.1rem 0 0.5rem 0;
}

.risk-gauge svg {
    width: 164px;
    height: 164px;
}

.risk-gauge-value {
    font-size: 34px;
    font-weight: 750;
}

.risk-gauge-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.16em;
}

.risk-level {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.45rem;
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0.1rem 0 0.65rem 0;
}

.risk-level-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}

.risk-meta {
    color: #dbe4ff !important;
    font-size: 0.85rem;
    line-height: 1.5;
    text-align: center;
}

.risk-meta-soft {
    color: #93a4c8 !important;
    font-size: 0.76rem;
    margin-top: 0.2rem;
}

/* ---- Surrounding-area summary ---- */
.scan-stats {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.5rem;
}

.scan-stat {
    background: #0d1738;
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 10px;
    padding: 0.5rem 0.25rem;
    text-align: center;
}

.scan-stat-label {
    color: #93a4c8;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.scan-stat-value {
    color: #ffffff;
    font-size: 1.3rem;
    font-weight: 750;
    line-height: 1.25;
}

.scan-count {
    color: #93a4c8;
    font-size: 0.76rem;
    margin-top: 0.55rem;
    text-align: center;
}

/* ---- Map fills the whole main area ---- */
.st-key-map-area {
    height: 100vh !important;
    padding: 0 !important;
    margin: 0 !important;
}

.st-key-map-area [data-testid="stVerticalBlock"] {
    height: 100vh !important;
    gap: 0 !important;
}

.st-key-map-area iframe {
    display: block;
    height: 100vh !important;
    width: 100% !important;
    border: 0 !important;
}
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title="Fire Risk Map",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    _init_state()

    # Render the map first so a map click can update the coordinate widgets
    # before the sidebar creates them.
    render_map(st.session_state["lat"], st.session_state["lon"])
    render_sidebar()


@st.cache_data(ttl=config.WEATHER_CACHE_TTL_SECONDS, show_spinner=False)
def cached_weather(
    lat: float,
    lon: float,
    target_date: date,
    timeout: float = config.REQUEST_TIMEOUT_SECONDS,
):
    """Cache weather downloads for one hour so repeated scans are fast."""
    return download_daily_weather(lat, lon, target_date, timeout=timeout)


def _init_state() -> None:
    """Set the default map view once per session.

    The map view is only changed by a place search. The browser keeps the
    user's zoom and pan on its own, so we never read them back.
    """
    if "lat" not in st.session_state:
        st.session_state["lat"] = config.DEFAULT_MAP_CENTER[0]
    if "lon" not in st.session_state:
        st.session_state["lon"] = config.DEFAULT_MAP_CENTER[1]
    if "map_focus" not in st.session_state:
        # Initial view: only used on the first run, and again when a place is
        # picked in the search results.
        st.session_state["map_focus"] = {
            "center": list(config.DEFAULT_MAP_CENTER),
            "zoom": config.DEFAULT_MAP_ZOOM,
        }
    if "map_nonce" not in st.session_state:
        # Bumped on place search so the map remounts once at the new place.
        st.session_state["map_nonce"] = 0


def render_sidebar() -> None:
    """Render the left control panel, top to bottom."""
    with st.sidebar:
        st.markdown(
            '<div class="risk-panel-title">Risk Assessment</div>',
            unsafe_allow_html=True,
        )

        render_place_search_card()
        target_date = render_date_card()

        scan_area = st.checkbox(
            "Scan surrounding area",
            value=True,
            key="scan_area",
        )

        lat = float(st.session_state["lat"])
        lon = float(st.session_state["lon"])

        analyze = st.button(
            "Analyze fire risk",
            type="primary",
            width="stretch",
            key="analyze",
        )

        inside = config.is_inside_training_area(lat, lon)
        if not inside:
            render_error(_outside_area_message())
        elif analyze:
            with st.spinner("Downloading weather and building features..."):
                success = handle_prediction(
                    lat,
                    lon,
                    target_date,
                    scan_area,
                )
            if success:
                # Rerun so the map can draw the new scan circles.
                st.rerun()

        if inside and "result" in st.session_state:
            prediction = st.session_state["result"]
            render_demo_banner(prediction.is_demo)
            render_result_card(
                prediction,
                st.session_state["result_lat"],
                st.session_state["result_lon"],
                st.session_state["weather_as_of"],
                st.session_state["requested_date"],
            )

            if st.session_state.get("scan_requested"):
                render_scan_summary(
                    st.session_state.get("scan_points") or [],
                    st.session_state.get("scan_total", config.SCAN_GRID_POINTS),
                )

        render_point_card()


def render_place_search_card() -> None:
    """Place search card at the top of the panel."""
    with st.container(key="card-search"):
        st.markdown("##### Search a place")
        query = st.text_input(
            "Place name",
            placeholder="e.g. Algiers, Paris, Sydney",
            key="place_query",
            label_visibility="collapsed",
        )

        if st.button("Search", key="place_search", width="stretch"):
            _run_place_search(query)

        error = st.session_state.get("place_error")
        if error:
            st.warning(error)

        places = st.session_state.get("place_results") or []
        if places:
            st.selectbox(
                "Matches",
                options=_place_labels(places),
                index=None,
                placeholder="Choose a place",
                key="place_result",
                on_change=_apply_selected_place,
                label_visibility="collapsed",
            )

        st.caption("Place search uses the free Open-Meteo geocoding API.")


def render_date_card() -> date:
    """Date card below the place search."""
    with st.container(key="card-date"):
        st.markdown("##### Date")
        return st.date_input(
            "Date",
            value=date.today(),
            max_value=date.today(),
            key="target_date",
            label_visibility="collapsed",
        )


def render_point_card() -> None:
    """Small card with the selected coordinates."""
    with st.container(key="card-point"):
        st.markdown("##### Selected point")
        col1, col2 = st.columns(2)

        with col1:
            st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                step=0.01,
                format="%.5f",
                key="lat",
            )

        with col2:
            st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                step=0.01,
                format="%.5f",
                key="lon",
            )

        st.caption("Click the map or search a place to move the marker.")


def _run_place_search(query: str) -> None:
    """Call the geocoding API and store the results for the selectbox."""
    if "place_result" in st.session_state:
        del st.session_state["place_result"]

    if not query.strip():
        st.session_state["place_results"] = []
        st.session_state["place_error"] = "Type a place name first."
        return

    try:
        places = search_places(query)
    except GeocodingError as exc:
        st.session_state["place_results"] = []
        st.session_state["place_error"] = str(exc)
        return

    st.session_state["place_results"] = places
    st.session_state["place_error"] = (
        None if places else "No places found. Try a different name."
    )


def _place_labels(places: list[Place]) -> list[str]:
    """Return unique labels for the place selectbox."""
    labels: list[str] = []
    seen: dict[str, int] = {}

    for place in places:
        label = place.label
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label} ({seen[label]})"
        labels.append(label)

    return labels


def _apply_selected_place() -> None:
    """Move the map to the place chosen in the sidebar."""
    label = st.session_state.get("place_result")
    places = st.session_state.get("place_results") or []
    labels = _place_labels(places)

    if label not in labels:
        return

    place: Place = places[labels.index(label)]
    place_lat = round(place.latitude, 5)
    place_lon = round(
        config.normalize_longitude(place.longitude),
        5,
    )
    st.session_state["lat"] = place_lat
    st.session_state["lon"] = place_lon
    # Move the map view to the new place too, not just the marker. Changing the
    # nonce makes the map remount once at this center and zoom.
    st.session_state["map_focus"] = {
        "center": [place_lat, place_lon],
        "zoom": config.PLACE_MAP_ZOOM,
    }
    st.session_state["map_nonce"] = st.session_state.get("map_nonce", 0) + 1
    _clear_result()


def _clear_result() -> None:
    """Remove an old result so it cannot describe a new point."""
    for key in (
        "result",
        "result_lat",
        "result_lon",
        "weather_as_of",
        "requested_date",
        "scan_points",
        "scan_total",
        "scan_requested",
    ):
        if key in st.session_state:
            del st.session_state[key]


def render_map(lat: float, lon: float) -> None:
    """Show a clickable world map with the selected point and scan circles.

    The base map (tiles, layer control, legend) is identical on every run, so
    the browser keeps the user's zoom, pan and chosen base layer. Only the
    dynamic features (marker, scan circles, training rectangle) are sent in a
    feature group, which can change without rebuilding the map.
    """
    bounds = config.TRAINING_BOUNDS
    focus = st.session_state.get("map_focus") or {
        "center": list(config.DEFAULT_MAP_CENTER),
        "zoom": config.DEFAULT_MAP_ZOOM,
    }
    nonce = st.session_state.get("map_nonce", 0)

    with st.container(key="map-area"):
        map_object = folium.Map(
            location=focus["center"],
            zoom_start=int(focus["zoom"]),
            min_zoom=config.MIN_MAP_ZOOM,
            max_bounds=True,
            tiles=None,
            control_scale=True,
        )

        # Base layers for the layer control. "Dark" is shown first and the
        # other two are switched on from the control. no_wrap stops the
        # tiles from repeating at the edges of the world. This block must
        # stay identical between runs so the map is not re-created.
        for index, layer in enumerate(config.MAP_BASE_LAYERS):
            folium.TileLayer(
                tiles=layer["tiles"],
                name=layer["name"],
                attr=layer["attr"],
                subdomains=layer["subdomains"],
                max_zoom=layer["max_zoom"],
                no_wrap=True,
                show=index == 0,
            ).add_to(map_object)

        folium.LayerControl(
            position="topleft",
            collapsed=False,
        ).add_to(map_object)

        # White legend card in the top-right corner of the map. It is static,
        # so it belongs to the base map.
        map_object.get_root().html.add_child(
            folium.Element(_legend_html())
        )

        # Everything that can change lives in this feature group. The base map
        # above is untouched, so adding circles does not reset the view.
        dynamic = folium.FeatureGroup(name="results")

        if bounds is not None:
            folium.Rectangle(
                bounds=[
                    [bounds["lat_min"], bounds["lon_min"]],
                    [bounds["lat_max"], bounds["lon_max"]],
                ],
                color="#c2410c",
                weight=2,
                fill=True,
                fill_opacity=0.08,
                tooltip="Training area",
            ).add_to(dynamic)

        # Semi-transparent circles for every successfully scanned point.
        for point in st.session_state.get("scan_points") or []:
            prediction = point.prediction
            folium.Circle(
                location=[point.lat, point.lon],
                radius=config.SCAN_CIRCLE_RADIUS_METERS,
                color=prediction.level_color,
                weight=1,
                opacity=0.75,
                fill=True,
                fill_color=prediction.level_color,
                fill_opacity=0.35,
                tooltip=f"{prediction.risk_level} - index {prediction.index}",
            ).add_to(dynamic)

        # Small dot that shows which point was selected or clicked.
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="#ffffff",
            weight=2,
            fill=True,
            fill_color="#d32f2f",
            fill_opacity=1.0,
            tooltip="Selected point",
        ).add_to(dynamic)

        map_data = st_folium(
            map_object,
            height=800,
            width=None,
            # The nonce only changes after a place search, which remounts the
            # map once at the new place. Clicks leave it alone.
            key=f"fire_map_{nonce}",
            # Only clicks come back. Zoom, pan and layer changes stay in the
            # browser and never trigger a rerun.
            returned_objects=["last_clicked"],
            feature_group_to_add=dynamic,
            wrap_longitude=False,
        )

    if not map_data:
        return

    clicked = map_data.get("last_clicked")
    if not clicked:
        return

    clicked_lat = round(float(clicked["lat"]), 5)
    clicked_lon = round(config.normalize_longitude(float(clicked["lng"])), 5)

    if (
        clicked_lat == st.session_state["lat"]
        and clicked_lon == st.session_state["lon"]
    ):
        return

    # A click moves the marker only. The map view and the nonce stay as they
    # are, so the user's zoom, pan and chosen layer are kept.
    st.session_state["lat"] = clicked_lat
    st.session_state["lon"] = clicked_lon
    st.rerun()


def handle_prediction(
    lat: float,
    lon: float,
    target_date: date,
    scan_area: bool,
) -> bool:
    """Run the weather -> features -> predictor flow and store the result.

    Returns True when the selected point was analyzed successfully.
    """
    if scan_area:
        scan_result = scan_area_points(
            lat,
            lon,
            target_date,
            cached_weather,
        )
        center = scan_result.center

        if center is None:
            _clear_result()
            render_error(
                "Could not analyze the selected point. "
                "Please try again or turn off the area scan."
            )
            return False

        prediction = center.prediction
        weather_as_of = center.as_of
        scan_points = scan_result.points
        scan_total = scan_result.total
    else:
        try:
            weather_data = cached_weather(lat, lon, target_date)
            features = build_features(weather_data.frame, weather_data.as_of)
            prediction = predict(features)
        except (
            WeatherDataError,
            FeatureBuildError,
            ModelLoadError,
            PredictionError,
        ) as exc:
            _clear_result()
            render_error(str(exc))
            return False

        weather_as_of = weather_data.as_of
        scan_points = []
        scan_total = config.SCAN_GRID_POINTS

    st.session_state["result"] = prediction
    st.session_state["result_lat"] = lat
    st.session_state["result_lon"] = lon
    st.session_state["weather_as_of"] = weather_as_of
    st.session_state["requested_date"] = target_date
    st.session_state["scan_points"] = scan_points
    st.session_state["scan_total"] = scan_total
    st.session_state["scan_requested"] = bool(scan_area)
    return True


def render_demo_banner(is_demo: bool) -> None:
    """Show a large warning when the app is using the fake demo score."""
    if is_demo:
        st.error(f"## {config.DEMO_BANNER_TEXT}")


def _gauge_html(index: int, color: str) -> str:
    """Build the SVG donut gauge for the 0-100 risk index."""
    radius = 52.0
    circumference = 2.0 * math.pi * radius
    filled = circumference * max(0, min(100, index)) / 100.0

    return (
        '<div class="risk-gauge">'
        '<svg viewBox="0 0 140 140" '
        f'role="img" aria-label="Risk index {index} out of 100">'
        f'<circle cx="70" cy="70" r="{radius:.1f}" fill="none" '
        'stroke="rgba(148, 163, 184, 0.18)" stroke-width="12" />'
        f'<circle cx="70" cy="70" r="{radius:.1f}" fill="none" '
        f'stroke="{color}" stroke-width="12" stroke-linecap="round" '
        f'stroke-dasharray="{filled:.2f} {circumference:.2f}" '
        'transform="rotate(-90 70 70)" />'
        '<text x="70" y="72" text-anchor="middle" '
        f'class="risk-gauge-value" fill="#ffffff">{index}</text>'
        '<text x="70" y="94" text-anchor="middle" '
        'class="risk-gauge-label" fill="#93a4c8">RISK INDEX</text>'
        "</svg>"
        "</div>"
    )


def _result_card_html(
    prediction: Prediction,
    lat: float,
    lon: float,
    weather_as_of: date,
    requested_date: date,
) -> str:
    """Build the gauge, level name, coordinates, and dates for the result card."""
    return (
        _gauge_html(prediction.index, prediction.level_color)
        + '<div class="risk-level" '
        + f'style="color: {prediction.level_color}">'
        + '<span class="risk-level-dot" '
        + f'style="background: {prediction.level_color}"></span>'
        + f"{prediction.risk_level}</div>"
        + f'<div class="risk-meta">{lat:.4f}, {lon:.4f}</div>'
        + f'<div class="risk-meta">Date: {requested_date:%Y-%m-%d}</div>'
        + '<div class="risk-meta risk-meta-soft">'
        + f"Weather data as of {weather_as_of:%Y-%m-%d}</div>"
    )


def render_result_card(
    prediction: Prediction,
    lat: float,
    lon: float,
    weather_as_of: date,
    requested_date: date,
) -> None:
    """Show the 0-100 relative risk index, never a probability."""
    with st.container(key="card-result"):
        st.markdown(
            _result_card_html(
                prediction,
                lat,
                lon,
                weather_as_of,
                requested_date,
            ),
            unsafe_allow_html=True,
        )

        if weather_as_of != requested_date:
            st.caption(
                f"Weather for {requested_date:%Y-%m-%d} is not available yet."
            )

        st.caption("Relative index, not a probability.")
        st.caption(f"Model version: {prediction.model_version}")


def _scan_stat_html(label: str, value: int) -> str:
    """Build one small statistic card."""
    return (
        '<div class="scan-stat">'
        f'<div class="scan-stat-label">{label}</div>'
        f'<div class="scan-stat-value">{value}</div>'
        "</div>"
    )


def _scan_summary_html(points: list[ScanPoint], total: int) -> str:
    """Build the Average / Min / Max cards and the scanned-point count."""
    values = [point.prediction.index for point in points]

    if values:
        average = round(sum(values) / len(values))
        minimum = min(values)
        maximum = max(values)
    else:
        average = minimum = maximum = 0

    return (
        '<div class="scan-stats">'
        + _scan_stat_html("Average", average)
        + _scan_stat_html("Min", minimum)
        + _scan_stat_html("Max", maximum)
        + "</div>"
        + '<div class="scan-count">'
        + f"Points scanned: {len(points)} of {total}"
        + "</div>"
    )


def render_scan_summary(points: list[ScanPoint], total: int) -> None:
    """Show the surrounding-area summary in the left panel."""
    with st.container(key="card-scan"):
        st.markdown("##### Surrounding area")
        st.markdown(
            _scan_summary_html(points, total),
            unsafe_allow_html=True,
        )


def _legend_html() -> str:
    """Build the white map legend as a positioned HTML card."""
    rows: list[str] = []

    for position, level in enumerate(config.RISK_LEVELS):
        low = position * 20
        high = (position + 1) * 20
        rows.append(
            '<div style="display: flex; align-items: center; gap: 8px; '
            'margin: 5px 0;">'
            '<span style="display: inline-block; width: 11px; height: 11px; '
            f'border-radius: 3px; background: {level["color"]};"></span>'
            '<span style="flex: 1; color: #1f2937; font-size: 12px; '
            f'font-weight: 600;">{level["name"]}</span>'
            '<span style="color: #6b7280; font-size: 11px;">'
            f"{low}-{high}</span>"
            "</div>"
        )

    return (
        '<div style="position: absolute; top: 12px; right: 12px; '
        "z-index: 9999; background: #ffffff; border-radius: 12px; "
        "padding: 12px 14px; min-width: 172px; pointer-events: none; "
        "box-shadow: 0 8px 22px rgba(2, 6, 23, 0.28); "
        'font-family: system-ui, -apple-system, Segoe UI, sans-serif;">'
        '<div style="color: #0b1230; font-size: 13px; font-weight: 700; '
        'margin-bottom: 7px;">Fire Risk Levels</div>'
        + "".join(rows)
        + '<div style="border-top: 1px solid #e5e7eb; margin-top: 8px; '
        'padding-top: 7px; color: #6b7280; font-size: 10px;">'
        "Relative index, not a probability</div>"
        "</div>"
    )


def render_error(message: str) -> None:
    st.error(message)


def _outside_area_message() -> str:
    bounds = config.TRAINING_BOUNDS

    if bounds is None:
        return (
            "This point is outside the valid coordinate range. "
            "Latitude must be between -90 and 90, and longitude between "
            "-180 and 180."
        )

    return (
        "This point is outside the training area. "
        f"Choose a point between latitude {bounds['lat_min']} and "
        f"{bounds['lat_max']} and longitude {bounds['lon_min']} and "
        f"{bounds['lon_max']}."
    )


if __name__ == "__main__":
    main()
