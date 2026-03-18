# services/event_service.py

from __future__ import annotations
from typing import Iterable, Union

import numpy as np
import pandas as pd

DateInput = Union[pd.Series, pd.DatetimeIndex, Iterable]


def _to_date_series(dates: DateInput) -> pd.Series:
    """
    Converteer willekeurige input (list, Series, DatetimeIndex, ...) naar
    een nette pandas Series met genormaliseerde datums (zonder tijd).
    """
    # Alles eerst naar een list, dan naar Series
    if isinstance(dates, pd.Series):
        s = dates
    elif isinstance(dates, (pd.DatetimeIndex, pd.Index)):
        s = pd.Series(dates)
    else:
        s = pd.Series(list(dates))

    # Naar datetime, ongeldig -> NaT
    s = pd.to_datetime(s, errors="coerce")
    # NaT eruit
    s = s.dropna()
    if s.empty:
        return s

    # Alleen datum, tijd op 00:00:00
    s = s.dt.normalize()
    # Index 0..n-1
    s = s.reset_index(drop=True)
    return s


def _black_friday_for_year(year: int) -> pd.Timestamp:
    """
    Black Friday = laatste vrijdag van november.
    """
    last_nov = pd.Timestamp(year=year, month=11, day=30)
    # weekday(): maandag=0,... vrijdag=4
    offset = (last_nov.weekday() - 4) % 7
    return last_nov - pd.Timedelta(days=offset)


def _christmas_for_year(year: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=12, day=25)


def _boxing_day_for_year(year: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=12, day=26)


def build_event_flags_for_dates(
    dates: DateInput,
    country: str = "NL",
) -> pd.DataFrame:
    """
    Bouw een eenvoudige event-feature set voor een reeks datums.

    Output DataFrame met kolommen:
    - date
    - is_weekend
    - is_q4
    - is_black_friday
    - is_christmas
    - is_boxing_day
    - event_score
    """
    s = _to_date_series(dates)

    if s is None or len(s) == 0:
        return pd.DataFrame(
            columns=[
                "date",
                "is_weekend",
                "is_q4",
                "is_black_friday",
                "is_christmas",
                "is_boxing_day",
                "event_score",
            ]
        )

    df = pd.DataFrame({"date": s})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dow"] = df["date"].dt.weekday  # 0=ma, 6=zo

    # Weekend + Q4
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_q4"] = df["month"].isin([10, 11, 12]).astype(int)

    # Black Friday / Kerst / Boxing Day per jaar
    unique_years = df["year"].unique().tolist()
    bf_map = {y: _black_friday_for_year(y) for y in unique_years}

    df["is_black_friday"] = df.apply(
        lambda r: int(r["date"] == bf_map.get(r["year"])),
        axis=1,
    )

    df["is_christmas"] = (
        (df["month"] == 12) & (df["day"] == 25)
    ).astype(int)

    df["is_boxing_day"] = (
        (df["month"] == 12) & (df["day"] == 26)
    ).astype(int)

    # Ruwe event-intensiteitsscore
    df["event_score"] = (
        0.5 * df["is_weekend"]
        + 0.5 * df["is_q4"]
        + 2.0 * df["is_black_friday"]
        + 1.0 * df["is_christmas"]
        + 1.0 * df["is_boxing_day"]
    )

    return df[
        [
            "date",
            "is_weekend",
            "is_q4",
            "is_black_friday",
            "is_christmas",
            "is_boxing_day",
            "event_score",
        ]
    ]
