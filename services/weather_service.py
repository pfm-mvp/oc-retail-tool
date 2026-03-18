# services/forecast_service.py

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests

# Pro-model support (LightGBM). Als dit niet geïnstalleerd is,
# valt de pro-forecast automatisch terug op de simpele variant.
try:
    import lightgbm as lgb  # type: ignore
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

# Retail-kalenderfeatures (feestdagen, Black Friday, kerstperiode, zomer, etc.)
from services.event_service import add_retail_calendar_features


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def _ensure_sales_per_visitor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zorgt dat 'sales_per_visitor' bestaat als 'turnover' en 'footfall' aanwezig zijn.
    Past df in-place NIET aan; returned een copy.
    """
    df = df.copy()
    if "sales_per_visitor" not in df.columns:
        if "turnover" in df.columns and "footfall" in df.columns:
            df["sales_per_visitor"] = np.where(
                df["footfall"] > 0,
                df["turnover"] / df["footfall"],
                np.nan,
            )
        else:
            df["sales_per_visitor"] = np.nan
    return df


def _fetch_weather_daily_visualcrossing(
    location_str: str,
    start_date: dt.date,
    end_date: dt.date,
    api_key: str,
) -> pd.DataFrame:
    """
    Haalt dagelijkse weerdata op via Visual Crossing (history/forecast in één endpoint).

    Return DataFrame: kolommen ['date', 'temp', 'precip', 'windspeed']
    """
    if not api_key:
        return pd.DataFrame()

    base_url = (
        "https://weather.visualcrossing.com/"
        "VisualCrossingWebServices/rest/services/timeline"
    )

    url = f"{base_url}/{location_str}/{start_date.isoformat()}/{end_date.isoformat()}"
    params = {
        "unitGroup": "metric",
        "key": api_key,
        "include": "days",
    }

    resp = requests.get(url, params=params, timeout=40)
    resp.raise_for_status()
    data = resp.json()

    days = data.get("days", [])
    if not days:
        return pd.DataFrame()

    wdf = pd.DataFrame(days)
    wdf["date"] = pd.to_datetime(wdf["datetime"], errors="coerce")
    wdf = wdf.dropna(subset=["date"])
    wdf = wdf[["date", "temp", "precip", "windspeed"]].copy()
    return wdf


def _prepare_weather_for_training_and_forecast(
    df: pd.DataFrame,
    weather_cfg: Optional[Dict[str, Any]],
    use_weather: bool,
    horizon: int,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    - Voegt weerdata toe aan de historische df (voor training).
    - Haalt ook weerdata op voor de forecast-periode (voor features in de toekomst).

    Return: (df_with_weather, future_weather_df or None)
    """
    if not use_weather or not weather_cfg:
        return df, None

    api_key = weather_cfg.get("api_key") or ""
    mode = weather_cfg.get("mode", "city_country")
    if not api_key:
        return df, None

    if mode == "city_country":
        city = weather_cfg.get("city", "Amsterdam")
        country = weather_cfg.get("country", "Netherlands")
        location_str = f"{city},{country}"
    else:
        # Onbekende mode → geen weer
        return df, None

    df = df.copy()
    if df.empty or "date" not in df.columns:
        return df, None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return df, None

    hist_start = df["date"].min().date()
    hist_end = df["date"].max().date()

    try:
        w_hist = _fetch_weather_daily_visualcrossing(
            location_str=location_str,
            start_date=hist_start,
            end_date=hist_end,
            api_key=api_key,
        )
        if not w_hist.empty:
            df = df.merge(w_hist, on="date", how="left")
    except Exception:
        # Als weer faalt: geen crash, gewoon zonder weer verder
        return df, None

    # Weer voor forecast-periode
    last_date = df["date"].max().date()
    fut_start = last_date + dt.timedelta(days=1)
    fut_end = last_date + dt.timedelta(days=horizon)

    try:
        w_future = _fetch_weather_daily_visualcrossing(
            location_str=location_str,
            start_date=fut_start,
            end_date=fut_end,
            api_key=api_key,
        )
        if w_future.empty:
            w_future = None
    except Exception:
        w_future = None

    return df, w_future


# -----------------------------------------------------------
# SIMPLE BASELINE FORECAST (DoW-gemiddelden)
# -----------------------------------------------------------

def build_simple_footfall_turnover_forecast(
    df_all_raw: pd.DataFrame,
    horizon: int = 14,
    min_history_days: int = 30,
) -> dict:
    """
    Simpele day-of-week forecast voor footfall & omzet.

    Verwacht:
    - df_all_raw met minimaal kolommen: 'date', 'footfall'
    - optioneel: 'turnover'
    - 'sales_per_visitor' wordt berekend als die nog niet bestaat

    Return dict:
    {
        "enough_history": bool,
        "hist_recent": DataFrame,
        "forecast": DataFrame,
        "recent_footfall_sum": float,
        "recent_turnover_sum": float or NaN,
        "forecast_footfall_sum": float,
        "forecast_turnover_sum": float,
        "last_date": Timestamp,
        "model_type": str,               # "simple_dow"
        "used_simple_fallback": bool,    # altijd False hier
    }
    """

    df = df_all_raw.copy()

    if df.empty or "date" not in df.columns or "footfall" not in df.columns:
        return {"enough_history": False}

    # Zorg dat date datetime is
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        return {"enough_history": False}

    # Sales per visitor berekenen als dat nog niet bestaat
    df = _ensure_sales_per_visitor(df)

    # Minimaal aantal dagen historie
    unique_days = df["date"].dt.normalize().nunique()
    if unique_days < min_history_days:
        return {"enough_history": False}

    # Day-of-week statistieken
    df["dow"] = df["date"].dt.weekday  # 0=maandag
    grp = df.groupby("dow")

    dow_stats = grp["footfall"].agg(["mean", "std"]).rename(
        columns={"mean": "footfall_mean", "std": "footfall_std"}
    )

    if "sales_per_visitor" in df.columns:
        dow_spv = grp["sales_per_visitor"].mean().rename("spv_mean")
        dow_stats = dow_stats.join(dow_spv)
    else:
        dow_stats["spv_mean"] = np.nan

    last_date = df["date"].max()

    # Forecast horizon
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    fc = pd.DataFrame({"date": future_dates})
    fc["dow"] = fc["date"].dt.weekday

    fc = fc.merge(dow_stats.reset_index(), on="dow", how="left")

    # Fallback SPV: globaal gemiddelde
    global_spv = df["sales_per_visitor"].mean()
    fc["spv_mean"] = fc["spv_mean"].fillna(global_spv)

    # Footfall & omzet forecast
    fc["footfall_forecast"] = fc["footfall_mean"].clip(lower=0)
    fc["turnover_forecast"] = fc["footfall_forecast"] * fc["spv_mean"]

    # Vergelijk met afgelopen 14 dagen
    last_14_start = last_date - pd.Timedelta(days=13)
    recent = df[(df["date"] >= last_14_start) & (df["date"] <= last_date)].copy()

    recent_foot = recent["footfall"].sum()
    recent_turn = recent["turnover"].sum() if "turnover" in recent.columns else np.nan

    fut_foot = fc["footfall_forecast"].sum()
    fut_turn = fc["turnover_forecast"].sum()

    # Historiek voor grafiek: laatste 28 dagen
    hist_recent = df[df["date"] >= (last_date - pd.Timedelta(days=27))].copy()

    return {
        "enough_history": True,
        "hist_recent": hist_recent,
        "forecast": fc,
        "recent_footfall_sum": float(recent_foot),
        "recent_turnover_sum": float(recent_turn) if not pd.isna(recent_turn) else np.nan,
        "forecast_footfall_sum": float(fut_foot),
        "forecast_turnover_sum": float(fut_turn),
        "last_date": last_date,
        "model_type": "simple_dow",
        "used_simple_fallback": False,
    }


# -----------------------------------------------------------
# PRO FORECAST (LightGBM + lags/rolling stats + calendar/events + weer)
# -----------------------------------------------------------

def build_pro_footfall_turnover_forecast(
    df_all_raw: pd.DataFrame,
    horizon: int = 14,
    min_history_days: int = 60,
    weather_cfg: Optional[Dict[str, Any]] = None,
    use_weather: bool = False,
) -> dict:
    """
    Pro-forecast met:
    - LightGBM (als beschikbaar)
    - Retail-kalenderfeatures (dow, maand, Q4, kerstperiode, Black Friday, feestdagen, zomer)
    - Lags & rolling means
    - Optioneel: weerfeatures (temp, precip, windspeed) via Visual Crossing

    Als LightGBM niet beschikbaar is of er te weinig historie is:
    -> automatische fallback naar build_simple_footfall_turnover_forecast(...),
       met 'used_simple_fallback' = True.
    """

    # Als LightGBM ontbreekt → fallback
    if not HAS_LIGHTGBM:
        simple = build_simple_footfall_turnover_forecast(
            df_all_raw,
            horizon=horizon,
            min_history_days=min_history_days,
        )
        simple["model_type"] = "simple_dow"
        simple["used_simple_fallback"] = True
        return simple

    df = df_all_raw.copy()

    if df.empty or "date" not in df.columns or "footfall" not in df.columns:
        # directe fallback
        simple = build_simple_footfall_turnover_forecast(
            df_all_raw,
            horizon=horizon,
            min_history_days=min_history_days,
        )
        simple["used_simple_fallback"] = True
        return simple

    # Date normaliseren
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        simple = build_simple_footfall_turnover_forecast(
            df_all_raw,
            horizon=horizon,
            min_history_days=min_history_days,
        )
        simple["used_simple_fallback"] = True
        return simple

    # Sales per visitor
    df = _ensure_sales_per_visitor(df)

    # Minimaal aantal dagen historie
    unique_days = df["date"].dt.normalize().nunique()
    if unique_days < min_history_days:
        simple = build_simple_footfall_turnover_forecast(
            df_all_raw,
            horizon=horizon,
            min_history_days=min_history_days,
        )
        simple["used_simple_fallback"] = True
        return simple

    # Sorteren op datum
    df = df.sort_values("date").reset_index(drop=True)

    # Retail-kalenderfeatures (dow, maand, Q4, kerstperiode, Black Friday, zomer, etc.)
    df = add_retail_calendar_features(df, date_col="date")

    # Lags & rolling stats op footfall
    for lag in [1, 7, 14]:
        df[f"footfall_lag_{lag}"] = df["footfall"].shift(lag)

    df["footfall_roll_7"] = df["footfall"].rolling(window=7, min_periods=3).mean()
    df["footfall_roll_28"] = df["footfall"].rolling(window=28, min_periods=7).mean()

    # Weerfeatures toevoegen (historie + future)
    df, weather_future_df = _prepare_weather_for_training_and_forecast(
        df=df,
        weather_cfg=weather_cfg,
        use_weather=use_weather,
        horizon=horizon,
    )

    # Basisfeaturelijst
    feature_cols = [
        "dow",
        "month",
        "day_of_month",
        "is_weekend",
        "is_q4",
        "is_december_peak",
        "is_summer_holiday",
        "is_nl_holiday",
        "is_christmas_period",
        "is_black_friday_weekend",
        "footfall_lag_1",
        "footfall_lag_7",
        "footfall_lag_14",
        "footfall_roll_7",
        "footfall_roll_28",
    ]

    weather_feature_cols: list[str] = []
    if use_weather and {"temp", "precip", "windspeed"}.issubset(df.columns):
        weather_feature_cols = ["temp", "precip", "windspeed"]
        feature_cols += weather_feature_cols
    else:
        # Als weerdata niet beschikbaar is → niet gebruiken in features
        weather_future_df = None

    # Eerste rijen met NaN in lags/rollings droppen
    # Let op: we droppen NIET op weerkolommen (LightGBM kan NaN aan)
    df_model = df.dropna(
        subset=[
            "footfall",
            "footfall_lag_1",
            "footfall_lag_7",
            "footfall_lag_14",
            "footfall_roll_7",
            "footfall_roll_28",
        ]
    ).copy()

    if df_model.shape[0] < 30:
        # Te weinig data na feature engineering → fallback
        simple = build_simple_footfall_turnover_forecast(
            df_all_raw,
            horizon=horizon,
            min_history_days=min_history_days,
        )
        simple["used_simple_fallback"] = True
        return simple

    # Trainset
    X_train = df_model[feature_cols].values
    y_train = df_model["footfall"].values

    # LightGBM model
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Voor forecast: lijst met footfall-waarden (historie)
    foot_hist = df["footfall"].tolist()
    last_date = df["date"].max()
    global_spv = df["sales_per_visitor"].mean()

    # Helper om kalenderfeatures voor een target-dag te halen
    def _calendar_features_for_date(ts: pd.Timestamp) -> Dict[str, Any]:
        tmp = pd.DataFrame({"date": [ts]})
        tmp = add_retail_calendar_features(tmp, date_col="date")
        row = tmp.iloc[0]
        return {
            "dow": int(row["dow"]),
            "month": int(row["month"]),
            "day_of_month": int(row["day_of_month"]),
            "is_weekend": int(row["is_weekend"]),
            "is_q4": int(row["is_q4"]),
            "is_december_peak": int(row["is_december_peak"]),
            "is_summer_holiday": int(row["is_summer_holiday"]),
            "is_nl_holiday": int(row["is_nl_holiday"]),
            "is_christmas_period": int(row["is_christmas_period"]),
            "is_black_friday_weekend": int(row["is_black_friday_weekend"]),
        }

    # Future weather lookup helper
    def _weather_for_date(ts: pd.Timestamp) -> Dict[str, float]:
        if weather_future_df is None:
            return {"temp": 0.0, "precip": 0.0, "windspeed": 0.0}
        row = weather_future_df.loc[weather_future_df["date"] == ts]
        if row.empty:
            return {"temp": 0.0, "precip": 0.0, "windspeed": 0.0}
        return {
            "temp": float(row["temp"].iloc[0]),
            "precip": float(row["precip"].iloc[0]),
            "windspeed": float(row["windspeed"].iloc[0]),
        }

    # Iteratieve forecast voor horizon dagen
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    fc_rows: list[Dict[str, Any]] = []

    for target_date in future_dates:
        new_idx = len(foot_hist)

        # Kalenderfeatures
        cal = _calendar_features_for_date(target_date)

        # Lags uit foot_hist (historisch + eerder voorspelde waarden)
        def _lag_or_last(idx_offset: int) -> float:
            idx = new_idx - idx_offset
            if 0 <= idx < len(foot_hist):
                return float(foot_hist[idx])
            return float(foot_hist[0]) if foot_hist else 0.0

        lag_1 = _lag_or_last(1)
        lag_7 = _lag_or_last(7)
        lag_14 = _lag_or_last(14)

        # rolling means over de laatste 7 / 28 bekende punten (incl. voorspelde)
        if len(foot_hist) > 0:
            roll_7 = float(np.mean(foot_hist[-7:])) if len(foot_hist) >= 3 else float(
                np.mean(foot_hist)
            )
            roll_28 = (
                float(np.mean(foot_hist[-28:]))
                if len(foot_hist) >= 7
                else float(np.mean(foot_hist))
            )
        else:
            roll_7 = 0.0
            roll_28 = 0.0

        feat: Dict[str, Any] = {
            "dow": cal["dow"],
            "month": cal["month"],
            "day_of_month": cal["day_of_month"],
            "is_weekend": cal["is_weekend"],
            "is_q4": cal["is_q4"],
            "is_december_peak": cal["is_december_peak"],
            "is_summer_holiday": cal["is_summer_holiday"],
            "is_nl_holiday": cal["is_nl_holiday"],
            "is_christmas_period": cal["is_christmas_period"],
            "is_black_friday_weekend": cal["is_black_friday_weekend"],
            "footfall_lag_1": lag_1,
            "footfall_lag_7": lag_7,
            "footfall_lag_14": lag_14,
            "footfall_roll_7": roll_7,
            "footfall_roll_28": roll_28,
        }

        if weather_future_df is not None and weather_feature_cols:
            w = _weather_for_date(target_date)
            feat["temp"] = w["temp"]
            feat["precip"] = w["precip"]
            feat["windspeed"] = w["windspeed"]

        # Zorg dat de featurevector exact in dezelfde volgorde staat als feature_cols
        x_new = [feat.get(col, 0.0) for col in feature_cols]

        y_pred = float(model.predict(np.array([x_new]))[0])
        y_pred = max(y_pred, 0.0)  # geen negatieve footfall

        foot_hist.append(y_pred)

        turnover_pred = y_pred * (global_spv if not np.isnan(global_spv) else 0.0)

        fc_rows.append(
            {
                "date": target_date,
                "footfall_forecast": y_pred,
                "turnover_forecast": turnover_pred,
            }
        )

    fc = pd.DataFrame(fc_rows)

    # Vergelijk met afgelopen 14 dagen
    last_14_start = last_date - pd.Timedelta(days=13)
    recent = df[(df["date"] >= last_14_start) & (df["date"] <= last_date)].copy()

    recent_foot = recent["footfall"].sum()
    recent_turn = recent["turnover"].sum() if "turnover" in recent.columns else np.nan

    fut_foot = fc["footfall_forecast"].sum()
    fut_turn = fc["turnover_forecast"].sum()

    # Historiek voor grafiek: laatste 28 dagen
    hist_recent = df[df["date"] >= (last_date - pd.Timedelta(days=27))].copy()

    return {
        "enough_history": True,
        "hist_recent": hist_recent,
        "forecast": fc,
        "recent_footfall_sum": float(recent_foot),
        "recent_turnover_sum": float(recent_turn) if not pd.isna(recent_turn) else np.nan,
        "forecast_footfall_sum": float(fut_foot),
        "forecast_turnover_sum": float(fut_turn),
        "last_date": last_date,
        "model_type": "lightgbm_dow_lags_events_weather",
        "used_simple_fallback": False,
    }
