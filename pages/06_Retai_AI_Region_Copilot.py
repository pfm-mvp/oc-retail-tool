# pages/06_Retai_AI_Region_Copilot.py

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

from datetime import datetime, timedelta

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response

from services.cbs_service import (
    get_cci_series,
    get_retail_index,
)
from services.svi_service import build_store_vitality

st.set_page_config(
    page_title="PFM Region Performance Copilot",
    layout="wide"
)

# ----------------------
# PFM brand colors
# ----------------------
PFM_PURPLE = "#762181"
PFM_RED = "#F04438"
PFM_GREY = "#E5E7EB"
PFM_GREEN = "#22C55E"
PFM_ORANGE = "#F97316"
PFM_YELLOW = "#FACC15"

# ----------------------
# API URL / secrets setup
# ----------------------

raw_api_url = st.secrets["API_URL"].rstrip("/")

if raw_api_url.endswith("/get-report"):
    REPORT_URL = raw_api_url
    FASTAPI_BASE_URL = raw_api_url.rsplit("/get-report", 1)[0]
else:
    FASTAPI_BASE_URL = raw_api_url
    REPORT_URL = raw_api_url + "/get-report"


# -------------
# Format helpers
# -------------

def fmt_eur(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"€ {x:,.0f}".replace(",", ".")


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.1f}%".replace(".", ",")


def fmt_int(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.0f}".replace(",", ".")


# -------------
# Region & API helpers
# -------------

@st.cache_data(ttl=600)
def load_region_mapping(path: str = "data/regions.csv") -> pd.DataFrame:
    """
    Verwacht een CSV met minimaal:
    shop_id;region

    Optioneel:
    - sqm_override  (float)
    - store_label   (mooie naam per winkel)
    """
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return pd.DataFrame()

    if "shop_id" not in df.columns or "region" not in df.columns:
        return pd.DataFrame()

    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")
    df["region"] = df["region"].astype(str)

    if "sqm_override" in df.columns:
        df["sqm_override"] = pd.to_numeric(df["sqm_override"], errors="coerce")
    else:
        df["sqm_override"] = np.nan

    if "store_label" in df.columns:
        df["store_label"] = df["store_label"].astype(str)
    else:
        df["store_label"] = np.nan

    df = df.dropna(subset=["shop_id"])
    return df


@st.cache_data(ttl=600)
def get_locations_by_company(company_id: int) -> pd.DataFrame:
    """
    Wrapper rond /company/{company_id}/location van de vemcount-agent.
    Extra ruime timeout, fouten worden in main() afgehandeld.
    """
    url = f"{FASTAPI_BASE_URL.rstrip('/')}/company/{company_id}/location"

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "locations" in data:
        df = pd.DataFrame(data["locations"])
    else:
        df = pd.DataFrame(data)

    return df


@st.cache_data(ttl=600)
def get_report(
    shop_ids,
    data_outputs,
    period: str,
    step: str = "day",
    source: str = "shops",
):
    """
    Wrapper rond /get-report (POST) van de vemcount-agent, met querystring zonder [].
    """
    params: list[tuple[str, str]] = []

    for sid in shop_ids:
        params.append(("data", str(sid)))

    for dout in data_outputs:
        params.append(("data_output", dout))

    params.append(("period", period))
    params.append(("step", step))
    params.append(("source", source))

    resp = requests.post(REPORT_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=600)
def fetch_region_street_traffic(region: str, start_date, end_date) -> pd.DataFrame:
    """
    Leest demo-straattraffic per regio uit data/pathzz_sample_weekly.csv

    Verwachte CSV-structuur (opgeschoond):
    Region;Week;Visits

    Voorbeeld:
    Noord;2024-01-14 To 2024-01-20;43.931

    Visits zijn waardes als 43.931 (→ 43931 bezoekers).
    Return:
    - week_start (datetime)
    - street_footfall (float)
    """
    csv_path = "data/pathzz_sample_weekly.csv"
    try:
        # Alleen de eerste 3 kolommen lezen, ongeacht eventuele rommel erachter
        df = pd.read_csv(
            csv_path,
            sep=";",
            header=0,
            usecols=[0, 1, 2],
            dtype=str,
            engine="python",
        )
    except Exception:
        return pd.DataFrame()

    # Kolommen forceren naar vaste namen
    df.columns = ["region", "week", "street_footfall"]

    # Regio case-insensitive matchen
    df["region"] = df["region"].astype(str).str.strip()
    region_norm = str(region).strip().lower()
    df = df[df["region"].str.lower() == region_norm].copy()
    if df.empty:
        return pd.DataFrame()

    # Visits: "43.931" → "43931" → 43931.0
    df["street_footfall"] = (
        df["street_footfall"]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
    )
    df = df.dropna(subset=["street_footfall"])

    df["street_footfall"] = (
        df["street_footfall"]
        .str.replace(".", "", regex=False)   # punt = duizendscheiding
        .str.replace(",", ".", regex=False)  # safety
        .astype(float)
    )

    # "2024-01-14 To 2024-01-20" → 2024-01-14
    def _parse_week_start(s: str):
        if isinstance(s, str) and "To" in s:
            return pd.to_datetime(s.split("To")[0].strip(), errors="coerce")
        return pd.NaT

    df["week_start"] = df["week"].apply(_parse_week_start)
    df = df.dropna(subset=["week_start"])

    # Filter op aangevraagde periode
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df["week_start"] >= start) & (df["week_start"] <= end)]

    return df[["week_start", "street_footfall"]].reset_index(drop=True)


# -------------
# KPI helpers
# -------------

def index_from_first_nonzero(s: pd.Series) -> pd.Series:
    """
    Maak een indexreeks (100 = eerste niet-nul waarde).
    Waardes vóór die maand worden NaN, zodat de lijn daar pas start.

    s: 1D Series met getallen (bijv. maand-omzet).
    """
    s = pd.to_numeric(s, errors="coerce").astype(float)
    nonzero = s.replace(0, np.nan).dropna()
    if nonzero.empty:
        # nergens data → alles NaN
        return pd.Series(np.nan, index=s.index)

    base_idx = nonzero.index[0]
    base_val = nonzero.iloc[0]
    idx = s / base_val * 100.0
    # alles vóór de basismaand leeg laten
    idx.loc[s.index < base_idx] = np.nan
    return idx


def compute_daily_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "turnover" in df.columns and "footfall" in df.columns:
        df["sales_per_visitor"] = np.where(
            df["footfall"] > 0,
            df["turnover"] / df["footfall"],
            np.nan,
        )
    if "transactions" in df.columns and "footfall" in df.columns:
        df["conversion_rate"] = np.where(
            df["footfall"] > 0,
            df["transactions"] / df["footfall"] * 100,
            np.nan,
        )
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregatie naar weekniveau (week eindigt op zaterdag, start = zondag).
    Past bij Pathzz-weekranges '2025-10-26 To 2025-11-01' e.d.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["week_start"] = df["date"].dt.to_period("W-SAT").dt.start_time

    agg_dict: dict[str, str] = {}
    if "footfall" in df.columns:
        agg_dict["footfall"] = "sum"
    if "turnover" in df.columns:
        agg_dict["turnover"] = "sum"
    if "sales_per_visitor" in df.columns:
        agg_dict["sales_per_visitor"] = "mean"
    if "conversion_rate" in df.columns:
        agg_dict["conversion_rate"] = "mean"

    if not agg_dict:
        return df[["week_start"]].drop_duplicates().reset_index(drop=True)

    return df.groupby("week_start", as_index=False).agg(agg_dict)


@st.cache_data(ttl=600)
def debug_cbs_endpoint(dataset: str, top: int = 3) -> dict:
    """
    Simpele healthcheck voor een CBS OData dataset.
    Probeert https://opendata.cbs.nl/ODataApi/OData/{dataset}/TypedDataSet?$top={top}
    op te halen en geeft status + eerste records terug.
    """
    url = f"https://opendata.cbs.nl/ODataApi/OData/{dataset}/TypedDataSet?$top={top}"
    try:
        resp = requests.get(url, timeout=15)
        status = resp.status_code
        try:
            js = resp.json()
            sample = js.get("value", [])[:top]
        except Exception:
            # fallback: tekst tonen als JSON niet lukt
            sample = resp.text[:1000]
        return {
            "ok": resp.ok,
            "url": url,
            "status_code": status,
            "sample": sample,
        }
    except Exception as e:
        return {
            "ok": False,
            "url": url,
            "error": repr(e),
        }


# -------------
# MAIN UI (Region view)
# -------------

def main():
    st.title("PFM Region Performance Copilot – Regio-overzicht")

    radar_df = pd.DataFrame()

    # --- Retailer selectie via clients.json ---
    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)
    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    st.sidebar.header("Selecteer retailer & regio")

    client_label = st.sidebar.selectbox("Retailer", clients_df["label"].tolist())
    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    # --- Locations & regions inladen ---
    try:
        locations_df = get_locations_by_company(company_id)
    except requests.exceptions.ReadTimeout:
        st.error(
            "De verbinding met de FastAPI-server duurde te lang bij het ophalen van de winkels "
            "(timeout). Probeer het nog eens of kies voorlopig een andere retailer."
        )
        return
    except requests.exceptions.RequestException as e:
        st.error(f"Fout bij ophalen van winkels uit FastAPI: {e}")
        return

    if locations_df.empty:
        st.error("Geen winkels gevonden voor deze retailer.")
        return

    # Region mapping inladen
    region_map = load_region_mapping()
    if region_map.empty:
        st.error(
            "Geen geldige regions.csv gevonden (verwacht minimaal kolommen: shop_id;region)."
        )
        return

    # Koppel regions.csv aan locations op shop_id = id
    locations_df["id"] = pd.to_numeric(locations_df["id"], errors="coerce").astype("Int64")
    merged = locations_df.merge(
        region_map,
        left_on="id",
        right_on="shop_id",
        how="inner",
    )

    if merged.empty:
        st.warning("Er zijn geen winkels met een regio-mapping voor deze retailer.")
        return

    # Effectieve sqm per winkel bepalen: override > API-sqm > NaN
    if "sqm" in merged.columns:
        merged["sqm_effective"] = np.where(
            merged["sqm_override"].notna(),
            merged["sqm_override"],
            pd.to_numeric(merged["sqm"], errors="coerce"),
        )
    else:
        merged["sqm_effective"] = merged["sqm_override"]

    # Label per winkel (mooie naam)
    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        if "name" in merged.columns:
            merged["store_display"] = merged["name"]
        else:
            merged["store_display"] = merged["id"].astype(str)

    available_regions = sorted(merged["region"].unique().tolist())
    region_choice = st.sidebar.selectbox("Regio", available_regions)

    region_shops = merged[merged["region"] == region_choice].copy()
    shop_ids = region_shops["id"].dropna().astype(int).unique().tolist()

    if not shop_ids:
        st.warning(f"Geen winkels gevonden voor regio '{region_choice}'.")
        return

    # Alle winkels (alle regio's) voor de FastAPI-call
    all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()

    # --- Periode keuze ---
    period_choice = st.sidebar.selectbox(
        "Periode",
        [
            "Kalenderjaar 2024",
            "Laatste 26 weken",
        ],
        index=0,
    )

    today = datetime.today().date()

    if period_choice == "Kalenderjaar 2024":
        start_period = datetime(2024, 1, 1).date()
        end_period = datetime(2024, 12, 31).date()
    else:  # "Laatste 26 weken"
        end_period = today
        start_period = today - timedelta(weeks=26)

    run_btn = st.sidebar.button("Analyseer regio", type="primary")

    if not run_btn:
        st.info("Kies een retailer, regio en periode en klik op **Analyseer regio**.")
        return

    # --- Data ophalen uit FastAPI (ALLE winkels, alle regio's) ---
    with st.spinner("Regionale data ophalen uit Storescan / FastAPI..."):
        metric_map = {
            "count_in": "footfall",
            "turnover": "turnover",
        }

        resp_all = get_report(
            all_shop_ids,
            list(metric_map.keys()),
            period="this_year",
            step="day",
            source="shops",
        )
        df_all_raw = normalize_vemcount_response(
            resp_all,
            kpi_keys=metric_map.keys(),
        )
        df_all_raw = df_all_raw.rename(columns=metric_map)

    if df_all_raw.empty:
        st.warning("Geen data gevonden voor deze retailer.")
        return

    df_all_raw["date"] = pd.to_datetime(df_all_raw["date"], errors="coerce")
    df_all_raw = df_all_raw.dropna(subset=["date"])

    # Probeer een store-id kolom te vinden voor per-winkel-analyses
    store_key_col = None
    for cand in ["id", "shop_id", "location_id"]:
        if cand in df_all_raw.columns:
            store_key_col = cand
            break

    # Filter op periode (ALLE winkels)
    start_ts = pd.Timestamp(start_period)
    end_ts = pd.Timestamp(end_period)

    df_filtered = df_all_raw[
        (df_all_raw["date"] >= start_ts) & (df_all_raw["date"] <= end_ts)
    ].copy()

    if df_filtered.empty:
        st.warning("Geen data in de geselecteerde periode voor deze retailer.")
        return

    df_filtered = compute_daily_kpis(df_filtered)

    # Join met volledige mapping (alle regio's, alle winkels)
    if store_key_col is not None:
        join_cols = ["id", "store_display", "region", "sqm_effective"]
        join_cols_existing = [c for c in join_cols if c in merged.columns]
        if "id" in join_cols_existing:
            df_period = df_filtered.merge(
                merged[join_cols_existing],
                left_on=store_key_col,
                right_on="id",
                how="left",
            )
        else:
            df_period = df_filtered.copy()
    else:
        df_period = df_filtered.copy()

    # Slice voor de GESELECTEERDE regio
    if "region" not in df_period.columns:
        st.warning("Region-informatie ontbreekt in de data (check regions.csv).")
        return

    df_region = df_period[df_period["region"] == region_choice].copy()

    if df_region.empty:
        st.warning("Geen data in de geselecteerde periode voor deze regio.")
        return

    # --- Wekelijkse aggregatie voor de geselecteerde regio ---
    region_weekly = aggregate_weekly(df_region)

    # --- Pathzz street traffic per regio ---
    pathzz_weekly = fetch_region_street_traffic(
        region=region_choice,
        start_date=start_period,
        end_date=end_period,
    )

    capture_weekly = pd.DataFrame()
    avg_capture = None

    if not pathzz_weekly.empty and not region_weekly.empty:
        pathzz_weekly["week_start"] = pd.to_datetime(pathzz_weekly["week_start"])
        region_weekly["week_start"] = pd.to_datetime(region_weekly["week_start"])

        capture_weekly = pd.merge(
            region_weekly,
            pathzz_weekly[["week_start", "street_footfall"]],
            on="week_start",
            how="inner",
        )

        if not capture_weekly.empty:
            capture_weekly["capture_rate"] = np.where(
                capture_weekly["street_footfall"] > 0,
                capture_weekly["footfall"] / capture_weekly["street_footfall"] * 100,
                np.nan,
            )

            capture_weekly = capture_weekly.sort_values("week_start")
            avg_capture = capture_weekly["capture_rate"].mean()

    # -----------------------
    # KPI cards op regioniveau
    # -----------------------

    st.subheader(f"{selected_client['brand']} – Regio {region_choice}")

    foot_total = df_region["footfall"].sum() if "footfall" in df_region.columns else 0
    turn_total = df_region["turnover"].sum() if "turnover" in df_region.columns else 0
    spv_avg = df_region["sales_per_visitor"].mean() if "sales_per_visitor" in df_region.columns else np.nan

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Footfall (regio, periode)", fmt_int(foot_total))
    with col2:
        st.metric("Omzet (regio, periode)", fmt_eur(turn_total))
    with col3:
        if not pd.isna(spv_avg):
            val = f"€ {spv_avg:.2f}".replace(".", ",")
        else:
            val = "-"
        st.metric("Gem. besteding/visitor", val)
    with col4:
        if avg_capture is not None and not pd.isna(avg_capture):
            st.metric("Gem. capture rate (regio)", fmt_pct(avg_capture))
        else:
            st.metric("Gem. capture rate (regio)", "-")

    # -----------------------
    # Store-level sqm & index analyse (binnen regio)
    # -----------------------

    st.markdown("### Store performance in regio (per winkel)")

    store_table = pd.DataFrame()
    if store_key_col is not None and "turnover" in df_region.columns:
        # Per winkel aggregatie
        group_cols = [store_key_col]
        agg_dict = {
            "footfall": "sum",
            "turnover": "sum",
        }
        if "sales_per_visitor" in df_region.columns:
            agg_dict["sales_per_visitor"] = "mean"

        if "sqm_effective" in df_region.columns:
            agg_dict["sqm_effective"] = "max"

        store_agg = (
            df_region.groupby(group_cols, as_index=False)
            .agg(agg_dict)
        )

        # Turnover per sqm
        if "sqm_effective" in store_agg.columns:
            store_agg["turnover_per_sqm"] = np.where(
                store_agg["sqm_effective"] > 0,
                store_agg["turnover"] / store_agg["sqm_effective"],
                np.nan,
            )

            median_tps = store_agg["turnover_per_sqm"].median(skipna=True)
            store_agg["sqm_index"] = np.where(
                (store_agg["turnover_per_sqm"].notna()) & (median_tps > 0),
                store_agg["turnover_per_sqm"] / median_tps * 100,
                np.nan,
            )
        else:
            store_agg["turnover_per_sqm"] = np.nan
            store_agg["sqm_index"] = np.nan

        # Namen erbij hangen
        name_map = {}
        for _, r in region_shops.iterrows():
            name_map[int(r["id"])] = r.get("store_display", str(r["id"]))

        store_agg["store_name"] = store_agg[store_key_col].map(name_map)

        store_table = store_agg.copy()
        store_table = store_table.sort_values("sqm_index", ascending=True)

        # Mooie tabel – in expander zodat dashboard "clean" blijft
        with st.expander("📋 Details per winkel (tabel)"):
            tbl = store_table.copy()
            tbl["footfall"] = tbl["footfall"].map(fmt_int)
            tbl["turnover"] = tbl["turnover"].map(fmt_eur)
            tbl["sales_per_visitor"] = tbl["sales_per_visitor"].map(
                lambda x: f"€ {x:.2f}".replace(".", ",") if not pd.isna(x) else "-"
            )
            tbl["sqm_effective"] = tbl["sqm_effective"].map(fmt_int)
            tbl["turnover_per_sqm"] = tbl["turnover_per_sqm"].map(
                lambda x: fmt_eur(x) if not pd.isna(x) else "-"
            )
            tbl["sqm_index"] = tbl["sqm_index"].map(
                lambda x: fmt_pct(x - 100) if not pd.isna(x) else "-"
            )

            tbl = tbl.rename(
                columns={
                    "store_name": "Winkel",
                    "footfall": "Footfall",
                    "turnover": "Omzet",
                    "sales_per_visitor": "Gem. besteding/visitor",
                    "sqm_effective": "m² (effectief)",
                    "turnover_per_sqm": "Omzet per m²",
                    "sqm_index": "m²-index t.o.v. regio",
                }
            )

            st.dataframe(
                tbl[
                    [
                        "Winkel",
                        "Footfall",
                        "Omzet",
                        "Gem. besteding/visitor",
                        "m² (effectief)",
                        "Omzet per m²",
                        "m²-index t.o.v. regio",
                    ]
                ],
                use_container_width=True,
            )

            st.caption(
                "m²-index t.o.v. regio: 100 = gelijk aan regiomedian. "
                "Onder 100 → onderbenut potentieel per m², boven 100 → outperformer."
            )
    else:
        st.info(
            "Geen store-level ID of omzet beschikbaar in de dagdata – "
            "m²-indexanalyse wordt daarom overgeslagen."
        )

    # -----------------------
    # Grafiek: store vs street + capture-index (regio)
    # -----------------------

    st.markdown("### Regioweekbeeld – winkeltraffic vs straattraffic (Pathzz)")

    if not capture_weekly.empty:
        chart_df = capture_weekly[
            ["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]
        ].copy()

        iso_calendar = chart_df["week_start"].dt.isocalendar()
        chart_df["week_label"] = iso_calendar.week.apply(lambda w: f"W{int(w):02d}")

        week_order = (
            chart_df.sort_values("week_start")["week_label"]
            .unique()
            .tolist()
        )

        counts_long = chart_df.melt(
            id_vars=["week_label"],
            value_vars=["footfall", "street_footfall", "turnover"],
            var_name="metric",
            value_name="value",
        )

        bar_chart = (
            alt.Chart(counts_long)
            .mark_bar(width=20, opacity=0.8)
            .encode(
                x=alt.X("week_label:N", title="Week", sort=week_order),
                xOffset=alt.XOffset("metric:N"),
                y=alt.Y(
                    "value:Q",
                    axis=alt.Axis(
                        title="Footfall / streettraffic / omzet (regio)"
                    ),
                ),
                color=alt.Color(
                    "metric:N",
                    title="",
                    scale=alt.Scale(
                        domain=["footfall", "street_footfall", "turnover"],
                        range=[PFM_PURPLE, "#B48CD8", PFM_RED],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("week_label:N", title="Week"),
                    alt.Tooltip("metric:N", title="Type"),
                    alt.Tooltip("value:Q", title="Waarde", format=",.0f"),
                ],
            )
        )

        line_chart = (
            alt.Chart(chart_df)
            .mark_line(point=True, strokeWidth=2, color=PFM_RED)
            .encode(
                x=alt.X("week_label:N", title="Week", sort=week_order),
                y=alt.Y(
                    "capture_rate:Q",
                    axis=alt.Axis(title="Capture rate regio (%)"),
                    scale=alt.Scale(zero=True),
                ),
                tooltip=[
                    alt.Tooltip("week_label:N", title="Week"),
                    alt.Tooltip(
                        "capture_rate:Q",
                        title="Capture rate",
                        format=".1f",
                    ),
                ],
            )
        )

        combined = (
            alt.layer(bar_chart, line_chart)
            .resolve_scale(y="independent")
            .properties(height=350)
        )

        st.altair_chart(combined, use_container_width=True)

        st.markdown("### Weekly tabel – regio-footfall, straattraffic, omzet & capture rate")

        table_df = capture_weekly[
            ["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]
        ].copy()

        table_df["week_start"] = table_df["week_start"].dt.strftime("%Y-%m-%d")
        table_df["footfall"] = table_df["footfall"].map(fmt_int)
        table_df["street_footfall"] = table_df["street_footfall"].map(fmt_int)
        table_df["turnover"] = table_df["turnover"].map(fmt_eur)
        table_df["capture_rate"] = table_df["capture_rate"].map(
            lambda x: fmt_pct(x) if not pd.isna(x) else "-"
        )

        table_df = table_df.rename(
            columns={
                "week_start": "Week start",
                "footfall": "Store footfall (regio)",
                "street_footfall": "Street footfall (Pathzz)",
                "turnover": "Omzet (regio)",
                "capture_rate": "Capture rate",
            }
        )

        st.dataframe(table_df, use_container_width=True)

    else:
        st.info("Geen matchende Pathzz-weekdata gevonden voor deze regio/periode.")

    # ----------------------------------------------------
    # Store Vitality Index (SVI) – per winkel + Regio Vitality
    # ----------------------------------------------------
    radar_df = pd.DataFrame()

    if store_key_col is not None:
        # 1) SVI voor álle winkels in alle regio's
        svi_all = build_store_vitality(
            df_period=df_period,      # df_period = alle shops in periode
            region_shops=merged,      # mapping met alle winkels + sqm + labels
            store_key_col=store_key_col,
        )

        if not svi_all.empty:
            region_lookup = merged[["id", "region"]].drop_duplicates()
            svi_all = svi_all.merge(
                region_lookup,
                left_on=store_key_col,
                right_on="id",
                how="left",
            )

            # 2) Region Vitality Index per regio (gemiddelde SVI van winkels)
            region_scores = (
                svi_all.groupby("region", as_index=False)["svi_score"]
                .mean()
                .rename(columns={"svi_score": "region_svi"})
            )

            # Geselecteerde regio eruit halen
            row_cur = region_scores[region_scores["region"] == region_choice]
            if row_cur.empty:
                st.info("Geen SVI-berekening mogelijk voor deze regio.")
            else:
                region_svi = float(row_cur["region_svi"].iloc[0])
                region_svi = float(np.clip(region_svi, 0, 100))

                # Statuskleuren (0–100)
                if region_svi >= 75:
                    region_status = "High performance"
                    fill_color = PFM_GREEN
                elif region_svi >= 60:
                    region_status = "Good / stable"
                    fill_color = PFM_ORANGE
                elif region_svi >= 45:
                    region_status = "Attention required"
                    fill_color = PFM_YELLOW
                else:
                    region_status = "Under pressure"
                    fill_color = PFM_RED

                empty_color = PFM_GREY

                gauge_df = pd.DataFrame(
                    {
                        "segment": ["filled", "empty"],
                        "value": [region_svi, max(0.0, 100.0 - region_svi)],
                    }
                )

                st.markdown("### Regio Vitality Index")

                col_g1, col_g2 = st.columns([1, 1.6])

                # Gauge voor geselecteerde regio
                with col_g1:
                    gauge_arc = (
                        alt.Chart(gauge_df)
                        .mark_arc(innerRadius=60, outerRadius=80)
                        .encode(
                            theta="value:Q",
                            color=alt.Color(
                                "segment:N",
                                scale=alt.Scale(
                                    domain=["filled", "empty"],
                                    range=[fill_color, empty_color],
                                ),
                                legend=None,
                            ),
                        )
                        .properties(width=260, height=260)
                    )

                    gauge_text = (
                        alt.Chart(pd.DataFrame({"label": [f"{region_svi:.0f}"]}))
                        .mark_text(size=32, fontWeight="bold")
                        .encode(text="label:N")
                    )

                    st.altair_chart(gauge_arc + gauge_text, use_container_width=False)

                with col_g2:
                    st.markdown(
                        f"""
                        **Regio Vitality Index (geselecteerd):** {region_svi:.0f}  
                        **Status:** {region_status}  

                        Deze index is het gemiddelde van de Store Vitality Index (SVI)
                        van alle winkels in deze regio (0–100).  
                        Hoe dichter bij 100, hoe gezonder de regio presteert binnen de keten.
                        """
                    )

                    st.markdown("**Vergelijking met andere regio's**")

                    chart_regions = region_scores.copy()
                    chart_regions["is_selected"] = chart_regions["region"] == region_choice

                    region_chart = (
                        alt.Chart(chart_regions)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "region_svi:Q",
                                title="Region Vitality Index (0–100)",
                                scale=alt.Scale(domain=[0, 100]),
                            ),
                            y=alt.Y(
                                "region:N",
                                sort="-x",
                                title="Regio",
                            ),
                            color=alt.Color(
                                "is_selected:N",
                                scale=alt.Scale(
                                    domain=[True, False],
                                    range=[PFM_PURPLE, PFM_GREY],
                                ),
                                legend=None,
                            ),
                            tooltip=[
                                alt.Tooltip("region:N", title="Regio"),
                                alt.Tooltip(
                                    "region_svi:Q",
                                    title="RVI",
                                    format=".0f",
                                ),
                            ],
                        )
                        .properties(height=220)
                    )

                    st.altair_chart(region_chart, use_container_width=True)

            # 3) Store Vitality ranking voor alleen de geselecteerde regio
            svi_region = svi_all[svi_all["region"] == region_choice].copy()
            if svi_region.empty:
                st.info("Geen store-level SVI beschikbaar voor deze regio.")
            else:
                period_days = (end_ts - start_ts).days + 1
                year_factor = 365.0 / period_days if period_days > 0 else 1.0

                svi_region["svi_score"] = svi_region["svi_score"].round(0)
                svi_region["profit_potential_year"] = (
                    svi_region["profit_potential_period"] * year_factor
                )

                chart_rank = (
                    alt.Chart(
                        svi_region.sort_values("svi_score", ascending=False)
                    )
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "svi_score:Q",
                            title="Store Vitality Index (0–100)",
                            scale=alt.Scale(domain=[0, 100]),
                        ),
                        y=alt.Y(
                            "store_name:N",
                            sort="-x",
                            title="Winkel",
                        ),
                        color=alt.Color(
                            "svi_status:N",
                            title="Status",
                            scale=alt.Scale(
                                domain=[
                                    "High performance",
                                    "Good / stable",
                                    "Attention required",
                                    "Under pressure",
                                ],
                                range=[
                                    PFM_GREEN,
                                    PFM_ORANGE,
                                    PFM_YELLOW,
                                    PFM_RED,
                                ],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("store_name:N", title="Winkel"),
                            alt.Tooltip("svi_score:Q", title="SVI", format=".0f"),
                            alt.Tooltip(
                                "footfall_index_region:Q",
                                title="Footfall-index",
                                format=".0f",
                            ),
                            alt.Tooltip(
                                "capture_index_region:Q",
                                title="Capture-index",
                                format=".0f",
                            ),
                            alt.Tooltip(
                                "profit_potential_year:Q",
                                title="Jaarpotentieel (€)",
                                format=",.0f",
                            ),
                        ],
                    )
                    .properties(height=260)
                )

                st.markdown("### Store Vitality ranking – winkels in deze regio")
                st.altair_chart(chart_rank, use_container_width=True)

                table = svi_region.copy()
                table["Omzet"] = table["turnover"].map(fmt_eur)
                table["Footfall"] = table["footfall"].map(fmt_int)
                table["Gem. besteding/visitor"] = table["sales_per_visitor"].map(
                    lambda x: f"€ {x:.2f}".replace(".", ",") if not pd.isna(x) else "-"
                )
                table["Omzet per m²"] = table["turnover_per_sqm"].map(
                    lambda x: fmt_eur(x) if not pd.isna(x) else "-"
                )
                table["Footfall-index (regio = 100)"] = table[
                    "footfall_index_region"
                ].map(lambda x: fmt_pct(x - 100) if not pd.isna(x) else "-")
                table["Capture-index (regio = 100)"] = table[
                    "capture_index_region"
                ].map(lambda x: fmt_pct(x - 100) if not pd.isna(x) else "-")
                table["Jaarpotentieel"] = table["profit_potential_year"].map(fmt_eur)

                view_cols = table.rename(
                    columns={
                        "svi_icon": "",
                        "store_name": "Winkel",
                        "svi_score": "SVI-score",
                        "svi_status": "Status",
                        "reason_short": "Korte toelichting",
                    }
                )

                st.dataframe(
                    view_cols[
                        [
                            "",
                            "Winkel",
                            "SVI-score",
                            "Status",
                            "Korte toelichting",
                            "Omzet",
                            "Footfall",
                            "Gem. besteding/visitor",
                            "Omzet per m²",
                            "Footfall-index (regio = 100)",
                            "Capture-index (regio = 100)",
                            "Jaarpotentieel",
                        ]
                    ],
                    use_container_width=True,
                )

                st.caption(
                    "De SVI combineert omzet, footfall, besteding per bezoeker en omzet per m², "
                    "met daarbovenop een index t.o.v. de regio voor footfall en 'fair share' van traffic "
                    "gebaseerd op m². Jaarpotentieel = ruw geannualiseerd omzetverschil t.o.v. regiomedian "
                    "per m² binnen de gekozen periode."
                )

                radar_df = svi_region

    # -----------------------
    # Macro-context: CBS detailhandel & consumentenvertrouwen
    # -----------------------

    cbs_retail_month = pd.DataFrame()
    cbs_retail_error = None
    cci_df = pd.DataFrame()
    cci_error = None

    st.markdown("### Macro-context: CBS detailhandel & consumentenvertrouwen")
    st.caption(
        "Regio-footfall- en omzet worden genormaliseerd op 100 = eerste maand met data. "
        "CBS-detailhandelindex en consumentenvertrouwen worden ernaast gezet om de "
        "macro-ontwikkeling te vergelijken."
    )

    macro_chart_shown = False

    # --- 1) Regio: maandindex opbouwen (footfall & omzet) ---
    region_month = df_region.copy()
    region_month["month"] = region_month["date"].dt.to_period("M").dt.to_timestamp()

    region_month = (
        region_month
        .groupby("month", as_index=False)[["turnover", "footfall"]]
        .sum()
        .rename(columns={
            "turnover": "region_turnover",
            "footfall": "region_footfall",
        })
    )

    if not region_month.empty:
        region_month["region_turnover_index"] = index_from_first_nonzero(
            region_month["region_turnover"]
        )
        region_month["region_footfall_index"] = index_from_first_nonzero(
            region_month["region_footfall"]
        )
    else:
        region_month["region_turnover_index"] = np.nan
        region_month["region_footfall_index"] = np.nan

    # --- 2) CBS detailhandelindex ophalen & normaliseren (indien beschikbaar) ---
    try:
        retail_series = get_retail_index(
            months_back=24,
        )
    except Exception as e:
        retail_series = []
        cbs_retail_error = repr(e)

    if retail_series:
        cbs_retail_df = pd.DataFrame(retail_series)

        # period is bv. '2000MM01' → jaar = eerste 4, maand = laatste 2
        cbs_retail_df["date"] = pd.to_datetime(
            cbs_retail_df["period"].str[:4]
            + "-"
            + cbs_retail_df["period"].str[-2:]
            + "-15",
            errors="coerce",
        )
        cbs_retail_df = cbs_retail_df.dropna(subset=["date"])

        # maandgemiddelde en index 100 = eerste maand met data
        cbs_retail_month = (
            cbs_retail_df.groupby("date", as_index=False)["retail_value"].mean()
        )
        if not cbs_retail_month.empty and cbs_retail_month["retail_value"].notna().any():
            base_cbs = cbs_retail_month["retail_value"].dropna().iloc[0]
            if base_cbs != 0:
                cbs_retail_month["cbs_retail_index"] = (
                    cbs_retail_month["retail_value"] / base_cbs * 100.0
                )
            else:
                cbs_retail_month["cbs_retail_index"] = np.nan
        else:
            cbs_retail_month = pd.DataFrame()
    else:
        cbs_retail_month = pd.DataFrame()

    # --- 3) Hoofdgrafiek: Regio vs (optioneel) CBS detailhandel ---
    try:
        chart_lines = []

        # Regio-footfall-index
        if "region_footfall_index" in region_month.columns:
            reg_foot = region_month.rename(columns={"month": "date"})[
                ["date", "region_footfall_index"]
            ].copy()
            reg_foot["series"] = "Regio footfall-index"
            reg_foot = reg_foot.rename(columns={"region_footfall_index": "value"})
            chart_lines.append(reg_foot)

        # Regio-omzet-index
        if "region_turnover_index" in region_month.columns:
            reg_turn = region_month.rename(columns={"month": "date"})[
                ["date", "region_turnover_index"]
            ].copy()
            reg_turn["series"] = "Regio omzet-index"
            reg_turn = reg_turn.rename(columns={"region_turnover_index": "value"})
            chart_lines.append(reg_turn)

        # CBS-detailhandelindex (macro, indien data)
        if not cbs_retail_month.empty and "cbs_retail_index" in cbs_retail_month.columns:
            cbs_line = cbs_retail_month[["date", "cbs_retail_index"]].copy()
            cbs_line["series"] = "CBS detailhandelindex"
            cbs_line = cbs_line.rename(columns={"cbs_retail_index": "value"})
            chart_lines.append(cbs_line)

        if chart_lines:
            chart_all = pd.concat(chart_lines, ignore_index=True)

            macro_chart = (
                alt.Chart(chart_all)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Maand"),
                    y=alt.Y("value:Q", title="Index (100 = eerste maand met data)"),
                    color=alt.Color(
                        "series:N",
                        title="Reeks",
                        scale=alt.Scale(
                            domain=[
                                "Regio footfall-index",
                                "Regio omzet-index",
                                "CBS detailhandelindex",
                            ],
                            range=[PFM_PURPLE, PFM_RED, PFM_GREY],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("date:T", title="Maand"),
                        alt.Tooltip("series:N", title="Reeks"),
                        alt.Tooltip("value:Q", title="Index", format=".1f"),
                    ],
                )
                .properties(height=320)
            )

            st.altair_chart(macro_chart, use_container_width=True)
            st.caption(
                "Alle reeksen zijn herleid naar index 100 in de eerste maand met data. "
                "Zo vergelijk je de relatieve ontwikkeling van regio-footfall, omzet en, "
                "indien beschikbaar, de CBS-detailhandelindex."
            )
            macro_chart_shown = True
    except Exception:
        macro_chart_shown = macro_chart_shown or False

    # --- 4) Consumentenvertrouwen vs regionale performance ---
    st.markdown("### Consumentenvertrouwen vs regionale performance")

    try:
        cci_series = get_cci_series(months_back=24)
    except Exception as e:
        cci_series = []
        cci_error = repr(e)

    if cci_series:
        cci_df = pd.DataFrame(cci_series)
        cci_df["date"] = pd.to_datetime(
            cci_df["period"].str[:4]
            + "-"
            + cci_df["period"].str[-2:]
            + "-15",
            errors="coerce",
        )
        cci_df = cci_df.dropna(subset=["date"])

        if not cci_df.empty and cci_df["cci"].notna().any():
            base_cci = cci_df["cci"].dropna().iloc[0]
            if base_cci != 0:
                cci_df["cci_index"] = cci_df["cci"] / base_cci * 100.0
            else:
                cci_df["cci_index"] = np.nan
        else:
            cci_df = pd.DataFrame()
    else:
        cci_df = pd.DataFrame()

    if not cci_df.empty:
        # Lijnen bouwen: CCI + regio-footfall + regio-omzet
        lines_cc = []

        cci_line = cci_df[["date", "cci_index"]].copy()
        cci_line["series"] = "Consumentenvertrouwen-index"
        cci_line = cci_line.rename(columns={"cci_index": "value"})
        lines_cc.append(cci_line)

        if "region_footfall_index" in region_month.columns:
            reg_foot2 = region_month.rename(columns={"month": "date"})[
                ["date", "region_footfall_index"]
            ].copy()
            reg_foot2["series"] = "Regio footfall-index"
            reg_foot2 = reg_foot2.rename(columns={"region_footfall_index": "value"})
            lines_cc.append(reg_foot2)

        if "region_turnover_index" in region_month.columns:
            reg_turn2 = region_month.rename(columns={"month": "date"})[
                ["date", "region_turnover_index"]
            ].copy()
            reg_turn2["series"] = "Regio omzet-index"
            reg_turn2 = reg_turn2.rename(columns={"region_turnover_index": "value"})
            lines_cc.append(reg_turn2)

        chart_cc = (
            alt.Chart(pd.concat(lines_cc, ignore_index=True))
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Maand"),
                y=alt.Y("value:Q", title="Index (100 = eerste maand met data)"),
                color=alt.Color(
                    "series:N",
                    title="Reeks",
                    scale=alt.Scale(
                        domain=[
                            "Consumentenvertrouwen-index",
                            "Regio footfall-index",
                            "Regio omzet-index",
                        ],
                        range=[PFM_RED, PFM_PURPLE, PFM_GREY],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Maand"),
                    alt.Tooltip("series:N", title="Reeks"),
                    alt.Tooltip("value:Q", title="Index", format=".1f"),
                ],
            )
            .properties(height=260)
        )

        st.altair_chart(chart_cc, use_container_width=True)
        st.caption(
            "Consumentenvertrouwen (CCI) en regionale footfall/omzet zijn hier alle drie "
            "herleid naar 100 in de eerste maand met data. Zo zie je direct of de regio "
            "harder of minder hard groeit dan het consumentenvertrouwen."
        )
    else:
        st.info(
            "Geen bruikbare CCI-data beschikbaar vanuit de CBS-API (of geen data in de gekozen periode)."
        )

    # -----------------------
    # Debug-sectie
    # -----------------------
    with st.expander("🔧 Debug regio"):
        st.write("Geselecteerde retailer:", selected_client)
        st.write(
            "Region mapping (subset):",
            region_shops[["id", "store_display", "region", "sqm_effective"]].head(),
        )
        st.write("Shop IDs regio:", shop_ids)
        st.write("ALL shop IDs:", all_shop_ids)
        st.write("Periode:", start_period, "→", end_period)
        st.write("Store key column in df_all_raw:", store_key_col)
        st.write("df_all_raw (head):", df_all_raw.head())
        st.write("df_period (all regions, head):", df_period.head())
        st.write("df_region (selected region, head):", df_region.head())
        st.write("Region monthly:", region_month.head())
        st.write(
            "CBS retail (sample DataFrame):",
            cbs_retail_month.head() if not cbs_retail_month.empty else "empty",
        )
        st.write("CBS retail error:", cbs_retail_error)
        st.write(
            "CCI (sample DataFrame):",
            cci_df.head() if not cci_df.empty else "empty",
        )
        st.write("CCI error:", cci_error)
        st.write("Region weekly:", region_weekly.head())
        st.write("Pathzz weekly:", pathzz_weekly.head())
        st.write("Capture weekly:", capture_weekly.head())
        st.write(
            "Store table (raw):",
            store_table.head() if not store_table.empty else "n.v.t.",
        )
        st.write("Radar (head):", radar_df.head() if not radar_df.empty else "empty")

        st.markdown("#### 🔍 CBS endpoint healthcheck")

        cci_debug = debug_cbs_endpoint("83693NED", top=3)
        st.write("CCI 83693NED:", cci_debug)

        retail_debug = debug_cbs_endpoint("85828NED", top=3)
        st.write("Retail 85828NED:", retail_debug)


if __name__ == "__main__":
    main()
