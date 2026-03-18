# services/pathzz_service.py

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_pathzz_sample(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Laadt de demo-Pathzz CSV en zet deze om naar een weekly dataframe.

    - Kolom 'Week' heeft formaat 'YYYY-MM-DD To YYYY-MM-DD'
    - Kolom 'Visits' gebruikt een punt als duizendtalscheiding (16.725 = 16725)
    """
    if csv_path is None:
        csv_path = DATA_DIR / "pathzz_sample_weekly.csv"

    df = pd.read_csv(csv_path, sep=";", dtype=str)

    # Eerste datum uit 'Week' gebruiken als week-anker
    df["week_start_raw"] = df["Week"].str.split(" To ").str[0]
    df["week_start_raw"] = pd.to_datetime(df["week_start_raw"], format="%Y-%m-%d", errors="coerce")

    # Visits: punt is duizendtalscheiding → strip punt en cast naar int
    visits_numeric = (
        df["Visits"]
        .str.replace(".", "", regex=False)   # "16.725" → "16725"
        .astype("int64")
    )

    result = pd.DataFrame(
        {
            "week_start": df["week_start_raw"],
            "street_footfall": visits_numeric,
        }
    ).dropna(subset=["week_start"])

    return result


def fetch_weekly_street_traffic(start_date, end_date) -> pd.DataFrame:
    """
    Geeft een subset van de Pathzz-weekdata terug tussen start_date en end_date.

    We filteren op basis van week_start (eerste dag van het week-interval),
    maar in de Copilot wordt straks op een 'week_label' gematcht zodat
    Pathzz-weken en store-weken altijd samenvallen.
    """
    all_weeks = _load_pathzz_sample()

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    mask = (all_weeks["week_start"] >= start) & (all_weeks["week_start"] <= end)
    return all_weeks.loc[mask].reset_index(drop=True)
