# Project: Fire risk web app (worldwide), Streamlit

The ML model is NOT ready yet. Build the app so the model can be plugged in later.

Architecture (keep these files separate):
- app.py            : Streamlit interface only (map, date, button, result card)
- weather.py        : downloads NASA POWER daily weather for a point
- features.py       : builds the 18 features from the weather (placeholder formulas for now)
- predictor.py      : the ONLY file that talks to the model.
                      If fire_model.joblib does not exist -> DEMO mode.
                      If it exists -> use it.
- config.py         : FEATURES list, training-area bounds, model version

The 18 features, in this exact order:
T2M, T2M_MAX, RH2M, WS2M, PRECTOTCORR, GWETROOT, rain_7d, tmax_7d, rhmin_7d, rain_14d, tmax_14d, rhmin_14d, rain_30d, tmax_30d, rhmin_30d, dry_days, doy_sin, doy_cos

Rules:
- In DEMO mode, show a big visible banner: "DEMO MODE - fake score, not a real prediction".
- Never present any score as a real probability. Use "relative risk level".
- Reject points outside the training area with a clear message.
- Handle weather download errors with a clear message (timeout, no data).
- Keep the code simple and commented in plain English.
- No secrets, no real names, no Co-Authored-By lines.
- After each change, explain in simple words what you did.
