# services/pathzz_service.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

# We verwachten in /data een file zoals:
# pathzz_sample_weekly.csv met kolommen:
#   Week      -> "2024-01-07 To 2024-01-13"
#   Visits    -> 17578
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_CSV = DATA_DIR / "pathzz_sample_weekly.csv"


def _load_pathzz_weekly(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Leest de Pathzz weekly CSV en geeft een DataFrame met week_start & visits."""
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Kolomnamen robuust oppakken
    cols = {c.lower(): c for c in df.columns}
    week_col = cols.get("week") or cols.get("week comparison")
    visits_col = cols.get("visits") or cols.get("visit") or cols.get("volume")

    if not week_col or not visits_col:
        return pd.DataFrame()

    df = df[[week_col, visits_col]].rename(
        columns={week_col: "week_raw", visits_col: "visits"}
    )

    # "2024-01-07 To 2024-01-13" → startdatum pakken
    df["week_start"] = (
        df["week_raw"]
        .astype(str)
        .str.split("To")
        .str[0]
        .str.strip()
    )
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df = df.dropna(subset=["week_start"])

    return df


def fetch_monthly_street_traffic(
    lat: float,
    lon: float,
    start_date,
    end_date,
    radius_m: int = 100,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Simpele wrapper voor de Copilot:
    - Negeert lat/lon/radius (we gebruiken 1 vaste CSV voor de pilot)
    - Filtert de Pathzz-weken op de opgegeven periode
    - Aggregateert naar maandniveau → kolommen: ['month', 'street_footfall']
    """
    df = _load_pathzz_weekly(csv_path)
    if df.empty:
        return df

    start_date = pd.to_datetime(start_date).normalize()
    end_date = pd.to_datetime(end_date).normalize()

    mask = (df["week_start"] >= start_date) & (df["week_start"] <= end_date)
    df = df.loc[mask].copy()
    if df.empty:
        return df

    df["month"] = df["week_start"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby("month", as_index=False)["visits"]
        .sum()
        .rename(columns={"visits": "street_footfall"})
    )
    return monthly
