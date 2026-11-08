"""The only module that talks to the fire risk model.

If ``fire_model.joblib`` does not exist, this module uses a deterministic DEMO
score. If the file exists, it loads the model and uses it. Level names and
colors come from config.py; the score cut-offs between levels come from
levels.json.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

import config
from features import build_feature_vector


class ModelLoadError(Exception):
    """Raised when the model file exists but cannot be loaded."""


class PredictionError(Exception):
    """Raised when the model cannot produce a prediction."""


@dataclass(frozen=True)
class Prediction:
    """The result shown by the UI. This is a relative index, not a probability."""

    score: float
    index: int
    risk_level: str
    level_color: str
    is_demo: bool
    model_version: str
    message: str


# Loaded models are kept in memory so Streamlit reruns do not reload them.
_model_cache: dict[str, Any] = {}


def is_demo_mode() -> bool:
    """Return True when no model file is available."""
    return not Path(config.MODEL_PATH).is_file()


def predict(features: dict[str, float]) -> Prediction:
    """Predict a relative risk level from the 18 features."""
    vector = build_feature_vector(features)

    if is_demo_mode():
        score = _demo_score(features)
        return _make_prediction(
            score,
            is_demo=True,
            model_version=f"{config.MODEL_VERSION}-demo",
            message=(
                "Demo score built from placeholder formulas. "
                "It is not a real prediction."
            ),
        )

    model = _load_model()
    raw_score = _predict_with_model(model, vector)
    score = _normalize_score(raw_score)

    return _make_prediction(
        score,
        is_demo=False,
        model_version=config.MODEL_VERSION,
        message=(
            "Relative risk level from the trained model. "
            "The score is not a probability."
        ),
    )


def _load_model() -> Any:
    """Load and cache the joblib model."""
    path = Path(config.MODEL_PATH)
    cache_key = str(path.resolve())

    if cache_key not in _model_cache:
        try:
            _model_cache[cache_key] = joblib.load(path)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load the model file {path.name}: {exc}"
            ) from exc

    return _model_cache[cache_key]


def _predict_with_model(model: Any, vector: list[float]) -> float:
    """Call the model and return its raw score."""
    try:
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba([vector]))
            return float(probabilities[0, -1])

        if hasattr(model, "predict"):
            raw = np.asarray(model.predict([vector])).ravel()
            return float(raw[0])
    except Exception as exc:
        raise PredictionError(
            f"The model could not make a prediction: {exc}"
        ) from exc

    raise PredictionError(
        "The loaded model does not have a predict or predict_proba method."
    )


def _demo_score(features: dict[str, float]) -> float:
    """Build a stable placeholder score from the feature values.

    Hotter, drier, windier, and less rainy conditions push the score up.
    There is no randomness, so Streamlit reruns show the same result.
    """
    temperature = _clamp(features["T2M_MAX"] / 50.0)
    dryness = _clamp(1.0 - features["RH2M"] / 100.0)
    wind = _clamp(features["WS2M"] / 15.0)
    recent_rain = _clamp(features["rain_7d"] / 30.0)
    soil_dryness = _clamp(1.0 - features["GWETROOT"])
    dry_days = _clamp(features["dry_days"] / 30.0)

    score = (
        0.20 * temperature
        + 0.25 * dryness
        + 0.15 * wind
        + 0.15 * (1.0 - recent_rain)
        + 0.10 * soil_dryness
        + 0.15 * dry_days
    )
    return _clamp(score)


def _normalize_score(raw_score: float) -> float:
    """Keep model output in the 0-1 range used by levels.json."""
    score = float(raw_score)
    if 1.0 < score <= 100.0:
        # Some models return a 0-100 score instead of 0-1.
        score = score / 100.0
    return _clamp(score)


def score_to_index(score: float) -> float:
    """Map a model score to a 0-100 relative risk index.

    The score is clamped to 0-1. The four cut-offs in levels.json split that
    range into five levels of 20 index points each. Inside a level the index
    moves linearly, so the mapping works with any model score range.
    """
    cutoffs = _load_cutoffs()
    boundaries = [0.0, *cutoffs, 1.0]
    points_per_level = 100.0 / (len(boundaries) - 1)
    value = _clamp(score)

    # Find the level band that contains this score.
    band = len(boundaries) - 2
    for position in range(len(boundaries) - 1):
        if value < boundaries[position + 1]:
            band = position
            break

    low = boundaries[band]
    high = boundaries[band + 1]
    span = high - low
    fraction = 0.0 if span <= 0.0 else (value - low) / span

    index = (band + fraction) * points_per_level
    return _clamp(index, 0.0, 100.0)


def _make_prediction(
    score: float,
    *,
    is_demo: bool,
    model_version: str,
    message: str,
) -> Prediction:
    """Build the Prediction object, including the 0-100 index and color."""
    index_value = score_to_index(score)
    level = _level_for_index(index_value)
    index = max(0, min(100, int(math.floor(index_value))))

    return Prediction(
        score=score,
        index=index,
        risk_level=str(level["name"]),
        level_color=str(level["color"]),
        is_demo=is_demo,
        model_version=model_version,
        message=message,
    )


def _level_for_index(index: float) -> dict[str, Any]:
    """Return the config level dict for an index in the 0-100 range."""
    levels = config.RISK_LEVELS
    points_per_level = 100.0 / len(levels)
    band = int(float(index) // points_per_level)
    band = max(0, min(len(levels) - 1, band))
    return levels[band]


def _score_to_risk_level(score: float) -> str:
    """Map a score to the level name from config.RISK_LEVELS."""
    return str(_level_for_index(score_to_index(score))["name"])


def _load_cutoffs() -> list[float]:
    """Read the four score cut-offs that separate the five index levels."""
    data = _load_levels_data()
    expected = len(config.RISK_LEVELS) - 1
    cutoffs = data.get("cutoffs")

    if not isinstance(cutoffs, list) or len(cutoffs) != expected:
        raise ModelLoadError(
            f"{Path(config.LEVELS_PATH).name} must contain a 'cutoffs' list "
            f"with exactly {expected} numbers."
        )

    values: list[float] = []
    for raw in cutoffs:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ModelLoadError(
                f"Each value in 'cutoffs' must be a number: {raw!r}."
            ) from exc

        if not math.isfinite(value):
            raise ModelLoadError("Cut-off values must be finite numbers.")

        # Allow a 0-100 scale as well as the usual 0-1 scale.
        if 1.0 < value <= 100.0:
            value = value / 100.0

        values.append(value)

    previous = 0.0
    for value in values:
        if not 0.0 < value < 1.0:
            raise ModelLoadError("Cut-off values must be between 0 and 1.")
        if value <= previous:
            raise ModelLoadError("Cut-off values must be in ascending order.")
        previous = value

    return values


def _load_levels_data() -> dict[str, Any]:
    """Read levels.json and return its contents."""
    path = Path(config.LEVELS_PATH)

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ModelLoadError(f"Risk level file not found: {path.name}.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError(f"Could not read {path.name}: {exc}") from exc

    if not isinstance(data, dict):
        raise ModelLoadError(f"{path.name} must contain a JSON object.")

    return data


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Keep a value inside a range."""
    return max(low, min(high, float(value)))
