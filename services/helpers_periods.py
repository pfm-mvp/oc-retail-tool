# helpers_periods.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class PeriodDef:
    start: date
    end: date
    macro_year: int


def quarter_range(year: int, q: int) -> tuple[date, date]:
    if q == 1:
        return date(year, 1, 1), date(year, 3, 31)
    if q == 2:
        return date(year, 4, 1), date(year, 6, 30)
    if q == 3:
        return date(year, 7, 1), date(year, 9, 30)
    return date(year, 10, 1), date(year, 12, 31)


def period_catalog(today: date) -> dict[str, PeriodDef]:
    return {
        "Kalenderjaar 2024": PeriodDef(date(2024, 1, 1), date(2024, 12, 31), 2024),
        "Kalenderjaar 2025": PeriodDef(date(2025, 1, 1), date(2025, 12, 31), 2025),

        "Q1 2024": PeriodDef(*quarter_range(2024, 1), 2024),
        "Q2 2024": PeriodDef(*quarter_range(2024, 2), 2024),
        "Q3 2024": PeriodDef(*quarter_range(2024, 3), 2024),
        "Q4 2024": PeriodDef(*quarter_range(2024, 4), 2024),

        "Q1 2025": PeriodDef(*quarter_range(2025, 1), 2025),
        "Q2 2025": PeriodDef(*quarter_range(2025, 2), 2025),
        "Q3 2025": PeriodDef(*quarter_range(2025, 3), 2025),
        "Q4 2025": PeriodDef(*quarter_range(2025, 4), 2025),

        "Laatste 26 weken": PeriodDef(today - timedelta(weeks=26), today, today.year),
    }
