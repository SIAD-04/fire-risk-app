"""Central configuration for the fire risk app.

Keep this file to constants and tiny pure helpers. Model logic belongs in
predictor.py, weather logic in weather.py, and feature logic in features.py.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

APP_TITLE = "Fire risk web app"

# The 18 model features, in the exact order the model expects.
FEATURES = [
    "T2M",
    "T2M_MAX",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
    "GWETROOT",
    "rain_7d",
    "tmax_7d",
    "rhmin_7d",
    "rain_14d",
    "tmax_14d",
    "rhmin_14d",
    "rain_30d",
    "tmax_30d",
    "rhmin_30d",
    "dry_days",
    "doy_sin",
    "doy_cos",
]

# Set to None to allow the whole world. To limit the app later, use a dict:
# {"lat_min": 19.5, "lat_max": 37.5, "lon_min": -8.5, "lon_max": 10.5}
TRAINING_BOUNDS = None

MODEL_PATH = PROJECT_ROOT / "fire_model.joblib"
LEVELS_PATH = PROJECT_ROOT / "levels.json"
MODEL_VERSION = "0.1.0"

# Risk levels for the 0-100 index. Each level covers 20 index points, in this
# order: 0-20, 20-40, 40-60, 60-80, 80-100. The score cut-offs that separate
# them live in levels.json.
RISK_LEVELS = [
    {"name": "Low", "color": "#3fb68b"},
    {"name": "Moderate", "color": "#9acd32"},
    {"name": "High", "color": "#e5b93c"},
    {"name": "Very High", "color": "#f07f2d"},
    {"name": "Extreme", "color": "#d32f2f"},
]

# Weather settings.
HISTORY_DAYS = 35
MIN_HISTORY_DAYS = 30
POWER_REQUEST_BUFFER_DAYS = 14
POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_COMMUNITY = "AG"
POWER_PARAMETERS = ("T2M", "T2M_MAX", "RH2M", "WS2M", "PRECTOTCORR", "GWETROOT")
REQUEST_TIMEOUT_SECONDS = 30
DRY_DAY_MM = 1.0

# Surrounding-area scan (3x3 grid, 1 degree apart).
SCAN_GRID_POINTS = 9
SCAN_GRID_STEP_DEGREES = 1.0
SCAN_WORKERS = 5
SCAN_REQUEST_TIMEOUT_SECONDS = 25
SCAN_CIRCLE_RADIUS_METERS = 40_000

# Weather downloads are cached for one hour.
WEATHER_CACHE_TTL_SECONDS = 3600

DEMO_BANNER_TEXT = "DEMO MODE - fake score, not a real prediction"

# Free Open-Meteo place search. No API key is needed.
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
GEOCODING_RESULT_COUNT = 5
GEOCODING_TIMEOUT_SECONDS = 10

# Map defaults (world view).
DEFAULT_MAP_CENTER = (20.0, 0.0)
DEFAULT_MAP_ZOOM = 2
MIN_MAP_ZOOM = 2
PLACE_MAP_ZOOM = 10

# Base map layers shown in the map layer control. The first entry ("Dark") is
# the default. CartoDB's "dark matter" style is served from the dark_all URL.
MAP_BASE_LAYERS = [
    {
        "name": "Dark",
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": "Tiles &copy; Esri",
        "subdomains": "abc",
        "max_zoom": 16,
    },
    {
        "name": "Satellite",
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": "Tiles &copy; Esri",
        "subdomains": "abc",
        "max_zoom": 18,
    },
    {
        "name": "Streets",
        "tiles": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": "&copy; OpenStreetMap contributors",
        "subdomains": "abc",
        "max_zoom": 19,
    },
]


def is_inside_training_area(lat: float, lon: float) -> bool:
    """Return True when the point is inside the allowed area.

    When TRAINING_BOUNDS is None, every valid latitude/longitude is allowed.
    When it is a dict, only points inside those bounds are allowed.
    """
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return False

    bounds = TRAINING_BOUNDS
    if bounds is None:
        return True

    return (
        bounds["lat_min"] <= lat <= bounds["lat_max"]
        and bounds["lon_min"] <= lon <= bounds["lon_max"]
    )


def normalize_longitude(lon: float) -> float:
    """Wrap a longitude into the -180..180 range."""
    return ((lon + 180.0) % 360.0) - 180.0
