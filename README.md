# Fire Risk Map

A worldwide Streamlit dashboard that shows a **relative fire risk index** for any
point on the map.

Pick a point by clicking the map or by searching a place name, choose a date, and
press **Analyze fire risk**. The app downloads daily weather for that point from
NASA POWER, builds 18 features from it, and turns them into a risk index from 0
to 100. The index is shown as one of five levels: Low, Moderate, High, Very High
or Extreme.

You can also tick **Scan surrounding area** to score a 3x3 grid of points around
the selected location. The results are drawn as colored circles on the map, with
average, minimum and maximum values.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m streamlit run app.py
```

Then open the local URL that Streamlit prints (usually http://localhost:8501).

## Project files

| File | What it does |
| --- | --- |
| `app.py` | The Streamlit interface only: sidebar, map, result cards. |
| `weather.py` | Downloads NASA POWER daily weather for one point. |
| `features.py` | Builds the 18 model features from the weather. |
| `predictor.py` | The only file that talks to the model. |
| `scan.py` | Runs the 3x3 surrounding-area scan. |
| `geocoding.py` | Place search with the free Open-Meteo API. |
| `config.py` | Features list, risk levels, scan settings, map layers. |
| `levels.json` | Score cut-offs that separate the five risk levels. |
| `tests/` | Automated tests (`pytest`). |

The 18 features, in this exact order:

```
T2M, T2M_MAX, RH2M, WS2M, PRECTOTCORR, GWETROOT, rain_7d, tmax_7d, rhmin_7d,
rain_14d, tmax_14d, rhmin_14d, rain_30d, tmax_30d, rhmin_30d, dry_days,
doy_sin, doy_cos
```

## DEMO mode

The machine learning model is **not ready yet**. As long as the file
`fire_model.joblib` is missing from the project folder, the app runs in
**DEMO MODE**:

- A red banner says **"DEMO MODE - fake score, not a real prediction"**.
- The score comes from simple placeholder formulas in `predictor.py`
  (hotter, drier, windier and less rainy conditions give a higher score).
- The 7/14/30-day features in `features.py` are placeholders too.

To use a real model, put a trained `fire_model.joblib` in the project folder.
`predictor.py` loads it automatically and the demo banner disappears. Nothing
else needs to change. The file supports models with `predict_proba` (the last
class probability is used) or `predict`.

## Model limits

- The number shown is a **relative risk index from 0 to 100**, not a
  probability. It is only useful for comparing places or dates with each other.
- It is **not** a forecast of fire ignition, and it must not be used for safety
  decisions, evacuation or emergency planning.
- The score depends on the weather data available for that day. Recent dates may
  fall back to the last day that NASA POWER has, and the app shows that date.
- Accuracy is limited by the training data of the model, which is still missing.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Author

Student code **IASD01**.
