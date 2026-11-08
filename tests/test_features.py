from datetime import date

import pandas as pd
import pytest

import config
from features import FeatureBuildError, build_feature_vector, build_features


def make_weather(days: int = 35) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    frame = pd.DataFrame(
        {
            "T2M": [20.0] * days,
            "T2M_MAX": [30.0] * days,
            "RH2M": [40.0] * days,
            "WS2M": [5.0] * days,
            "PRECTOTCORR": [0.0] * days,
            "GWETROOT": [0.2] * days,
        },
        index=dates,
    )
    frame.index.name = "date"
    return frame


def test_feature_keys_match_config_order():
    frame = make_weather()
    target = frame.index[-1].date()
    features = build_features(frame, target)

    assert list(features) == config.FEATURES
    assert build_feature_vector(features) == [
        features[name] for name in config.FEATURES
    ]


def test_target_day_values_are_used():
    frame = make_weather()
    frame.loc[frame.index[-1], "T2M"] = 27.5

    features = build_features(frame, frame.index[-1].date())

    assert features["T2M"] == pytest.approx(27.5)


def test_window_features_use_expected_days():
    frame = make_weather()
    frame.loc[frame.index[-7:], "PRECTOTCORR"] = 2.0
    frame.loc[frame.index[-7:], "T2M_MAX"] = 35.0
    frame.loc[frame.index[-7:], "RH2M"] = 20.0

    features = build_features(frame, frame.index[-1].date())

    assert features["rain_7d"] == pytest.approx(14.0)
    assert features["rain_14d"] == pytest.approx(14.0)
    assert features["rain_30d"] == pytest.approx(14.0)
    assert features["tmax_7d"] == pytest.approx(35.0)
    assert features["tmax_14d"] == pytest.approx(35.0)
    assert features["tmax_30d"] == pytest.approx(35.0)
    assert features["rhmin_7d"] == pytest.approx(20.0)
    assert features["rhmin_14d"] == pytest.approx(20.0)
    assert features["rhmin_30d"] == pytest.approx(20.0)
    assert features["dry_days"] == 23


def test_day_of_year_encoding_is_on_unit_circle():
    frame = make_weather()
    features = build_features(frame, frame.index[-1].date())

    assert features["doy_sin"] ** 2 + features["doy_cos"] ** 2 == pytest.approx(1.0)


def test_insufficient_history_raises():
    frame = make_weather(days=29)

    with pytest.raises(FeatureBuildError):
        build_features(frame, frame.index[-1].date())


def test_missing_target_date_raises():
    frame = make_weather()

    with pytest.raises(FeatureBuildError):
        build_features(frame, date(2023, 1, 1))


def test_feature_vector_rejects_missing_keys():
    frame = make_weather()
    features = build_features(frame, frame.index[-1].date())
    features.pop("T2M")

    with pytest.raises(FeatureBuildError):
        build_feature_vector(features)


def test_training_area_accepts_the_world_when_unbounded():
    assert config.TRAINING_BOUNDS is None
    assert config.is_inside_training_area(0.0, 0.0)
    assert config.is_inside_training_area(-89.9, 179.9)
    assert config.is_inside_training_area(90.0, -180.0)
    assert not config.is_inside_training_area(91.0, 0.0)
    assert not config.is_inside_training_area(0.0, 181.0)


def test_training_area_uses_bounds_when_set(monkeypatch):
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

    assert config.is_inside_training_area(30.0, 2.0)
    assert not config.is_inside_training_area(0.0, 0.0)


def test_normalize_longitude_wraps_into_range():
    assert config.normalize_longitude(0.0) == pytest.approx(0.0)
    assert config.normalize_longitude(190.0) == pytest.approx(-170.0)
    assert config.normalize_longitude(-190.0) == pytest.approx(170.0)
    assert config.normalize_longitude(181.0) == pytest.approx(-179.0)
    assert config.normalize_longitude(540.0) == pytest.approx(-180.0)
    assert config.normalize_longitude(-540.0) == pytest.approx(-180.0)
