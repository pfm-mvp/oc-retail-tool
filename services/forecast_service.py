# services/forecast_service.py

import numpy as np
import pandas as pd

# Pro-model support (LightGBM). Als dit niet geïnstalleerd is,
# valt de pro-forecast automatisch terug op de simpele variant.
try:
    import lightgbm as lgb  # type: ignore
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


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
# PRO FORECAST (LightGBM + lags/rolling stats + events)
# -----------------------------------------------------------

def build_pro_footfall_turnover_forecast(
    df_all_raw: pd.DataFrame,
    horizon: int = 14,
    min_history_days: int = 60,
    weather_cfg: dict | None = None,      # placeholder voor toekomstige weather features
    use_weather: bool = False,            # momenteel niet actief gebruikt
    event_flags_hist: pd.DataFrame | None = None,
    event_flags_future: pd.DataFrame | None = None,
) -> dict:
    """
    Pro-forecast met:
    - LightGBM (als beschikbaar)
    - Features: dow, maand, dag, weekend, lags & rolling means
    - Optioneel: event-flags (bijv. feestdagen, vakanties, sale-periodes)

    Als LightGBM niet beschikbaar is of er te weinig historie is:
    -> automatische fallback naar build_simple_footfall_turnover_forecast(...),
       met 'used_simple_fallback' = True.

    Return: zelfde structuur als simple-forecast-result, plus:
        - "model_type": "lightgbm_dow_lags_events" of "simple_dow"
        - "used_simple_fallback": bool
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
    df["seq"] = np.arange(len(df))

    # Kalenderfeatures
    df["dow"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend_cal"] = df["dow"].isin([5, 6]).astype(int)

    # --- Event flags in historiek mergen (optioneel) ---
    event_feature_cols: list[str] = []
    if event_flags_hist is not None and not event_flags_hist.empty:
        ev = event_flags_hist.copy()
        ev["date"] = pd.to_datetime(ev["date"], errors="coerce").dt.normalize()
        ev = ev.dropna(subset=["date"]).drop_duplicates(subset=["date"])
        # neem alle kolommen behalve 'date'
        for c in ev.columns:
            if c == "date":
                continue
            # forceer naar 0/1
            ev[c] = ev[c].fillna(0).astype(int)
            event_feature_cols.append(c)
        df = df.merge(ev, on="date", how="left")
        for c in event_feature_cols:
            df[c] = df[c].fillna(0).astype(int)

    # Lags & rolling stats op footfall
    for lag in [1, 7, 14]:
        df[f"footfall_lag_{lag}"] = df["footfall"].shift(lag)

    df["footfall_roll_7"] = df["footfall"].rolling(window=7, min_periods=3).mean()
    df["footfall_roll_28"] = df["footfall"].rolling(window=28, min_periods=7).mean()

    # Eerste rijen met NaN in lags/rollings droppen
    base_feature_cols = [
        "dow",
        "month",
        "day_of_month",
        "is_weekend_cal",
        "footfall_lag_1",
        "footfall_lag_7",
        "footfall_lag_14",
        "footfall_roll_7",
        "footfall_roll_28",
    ]

    feature_cols = base_feature_cols + event_feature_cols

    df_model = df.dropna(subset=feature_cols + ["footfall"]).copy()
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

    # Voor events in de toekomst: map date → dict(feature:0/1)
    future_event_map: dict[pd.Timestamp, dict[str, int]] = {}
    if event_flags_future is not None and not event_flags_future.empty and event_feature_cols:
        evf = event_flags_future.copy()
        evf["date"] = pd.to_datetime(evf["date"], errors="coerce").dt.normalize()
        evf = evf.dropna(subset=["date"])
        for _, row in evf.iterrows():
            d = row["date"]
            future_event_map[d] = {}
            for c in event_feature_cols:
                future_event_map[d][c] = int(row.get(c, 0))

    # Iteratieve forecast voor horizon dagen
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    fc_rows = []

    for step, target_date in enumerate(future_dates, start=1):
        # index in "uitgebreide" tijdreeks
        new_idx = len(foot_hist)  # volgende positie

        # kalenderfeatures
        dow = target_date.weekday()
        month = target_date.month
        day_of_month = target_date.day
        is_weekend_cal = 1 if dow in [5, 6] else 0

        def _lag_or_last(idx_offset: int) -> float:
            idx = new_idx - idx_offset
            if 0 <= idx < len(foot_hist):
                return foot_hist[idx]
            else:
                return foot_hist[0] if len(foot_hist) > 0 else 0.0

        lag_1 = _lag_or_last(1)
        lag_7 = _lag_or_last(7)
        lag_14 = _lag_or_last(14)

        # rolling means over de laatste 7 / 28 bekende punten (incl. voorspelde)
        if len(foot_hist) > 0:
            roll_7 = np.mean(foot_hist[-7:]) if len(foot_hist) >= 3 else np.mean(foot_hist)
            roll_28 = np.mean(foot_hist[-28:]) if len(foot_hist) >= 7 else np.mean(foot_hist)
        else:
            roll_7 = 0.0
            roll_28 = 0.0

        # Event features voor deze datum
        event_vals = []
        if event_feature_cols:
            key = pd.to_datetime(target_date).normalize()
            flags = future_event_map.get(key, {})
            for c in event_feature_cols:
                event_vals.append(int(flags.get(c, 0)))

        x_new = [
            dow,
            month,
            day_of_month,
            is_weekend_cal,
            lag_1,
            lag_7,
            lag_14,
            roll_7,
            roll_28,
        ] + event_vals

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
        "model_type": "lightgbm_dow_lags_events",
        "used_simple_fallback": False,
    }
