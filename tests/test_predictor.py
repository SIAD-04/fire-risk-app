import json

import pandas as pd
import pytest

import config
import predictor
from features import build_features

LEVEL_NAMES = [level["name"] for level in config.RISK_LEVELS]
LEVEL_COLORS = [level["color"] for level in config.RISK_LEVELS]


def make_features():
    days = 35
    frame = pd.DataFrame(
        {
            "T2M": [20.0] * days,
            "T2M_MAX": [30.0] * days,
            "RH2M": [40.0] * days,
            "WS2M": [5.0] * days,
            "PRECTOTCORR": [0.0] * days,
            "GWETROOT": [0.2] * days,
        },
        index=pd.date_range("2024-01-01", periods=days, freq="D"),
    )
    return build_features(frame, frame.index[-1].date())


def write_cutoffs(tmp_path, cutoffs, monkeypatch):
    path = tmp_path / "levels.json"
    path.write_text(
        json.dumps({"demo": True, "cutoffs": cutoffs}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "LEVELS_PATH", path)
    return path


class DummyModel:
    def predict(self, rows):
        return [0.42]


def test_config_has_the_five_expected_levels():
    assert LEVEL_NAMES == [
        "Low",
        "Moderate",
        "High",
        "Very High",
        "Extreme",
    ]
    assert LEVEL_COLORS == [
        "#3fb68b",
        "#9acd32",
        "#e5b93c",
        "#f07f2d",
        "#d32f2f",
    ]


def test_levels_file_is_marked_demo_with_four_cutoffs():
    with open(config.LEVELS_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    assert data["demo"] is True
    cutoffs = data["cutoffs"]
    assert len(cutoffs) == len(config.RISK_LEVELS) - 1
    assert cutoffs == sorted(cutoffs)
    assert all(0.0 < value < 1.0 for value in cutoffs)


def test_demo_mode_when_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.joblib")
    assert predictor.is_demo_mode() is True

    result = predictor.predict(make_features())

    assert result.is_demo is True
    assert 0.0 <= result.score <= 1.0
    assert 0 <= result.index <= 100
    assert result.risk_level in LEVEL_NAMES
    assert result.level_color in LEVEL_COLORS
    assert "not a real prediction" in result.message


def test_predict_uses_model_when_file_exists(monkeypatch, tmp_path):
    model_path = tmp_path / "fire_model.joblib"
    model_path.write_bytes(b"placeholder")
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    monkeypatch.setattr(predictor, "_load_model", lambda: DummyModel())

    result = predictor.predict(make_features())

    assert result.is_demo is False
    assert result.score == pytest.approx(0.42)
    assert result.index == 33
    assert result.risk_level == "Moderate"
    assert result.level_color == "#9acd32"


def test_score_to_index_interpolates_inside_each_level(monkeypatch, tmp_path):
    write_cutoffs(tmp_path, [0.2, 0.4, 0.6, 0.8], monkeypatch)

    assert predictor.score_to_index(0.0) == pytest.approx(0.0)
    assert predictor.score_to_index(0.1) == pytest.approx(10.0)
    assert predictor.score_to_index(0.2) == pytest.approx(20.0)
    assert predictor.score_to_index(0.3) == pytest.approx(30.0)
    assert predictor.score_to_index(0.5) == pytest.approx(50.0)
    assert predictor.score_to_index(0.7) == pytest.approx(70.0)
    assert predictor.score_to_index(0.9) == pytest.approx(90.0)
    assert predictor.score_to_index(1.0) == pytest.approx(100.0)


def test_score_to_index_clamps_outside_scores(monkeypatch, tmp_path):
    write_cutoffs(tmp_path, [0.2, 0.4, 0.6, 0.8], monkeypatch)

    assert predictor.score_to_index(-5.0) == pytest.approx(0.0)
    assert predictor.score_to_index(5.0) == pytest.approx(100.0)


def test_cutoffs_accept_a_0_to_100_scale(monkeypatch, tmp_path):
    write_cutoffs(tmp_path, [20, 40, 60, 80], monkeypatch)

    assert predictor._load_cutoffs() == pytest.approx([0.2, 0.4, 0.6, 0.8])
    assert predictor.score_to_index(0.5) == pytest.approx(50.0)


@pytest.mark.parametrize(
    "cutoffs",
    [
        [0.2, 0.4, 0.6],
        [0.2, 0.4, 0.6, 1.0],
        [0.2, 0.2, 0.6, 0.8],
        [0.5, 0.4, 0.6, 0.8],
        ["x", 0.4, 0.6, 0.8],
    ],
)
def test_invalid_cutoffs_raise(monkeypatch, tmp_path, cutoffs):
    write_cutoffs(tmp_path, cutoffs, monkeypatch)

    with pytest.raises(predictor.ModelLoadError):
        predictor.score_to_index(0.5)


def test_missing_levels_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LEVELS_PATH", tmp_path / "missing.json")

    with pytest.raises(predictor.ModelLoadError):
        predictor.score_to_index(0.5)


def test_load_model_raises_for_bad_file(monkeypatch, tmp_path):
    model_path = tmp_path / "bad.joblib"
    model_path.write_text("not a model", encoding="utf-8")
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    predictor._model_cache.clear()

    with pytest.raises(predictor.ModelLoadError):
        predictor._load_model()
