# pages/05_Retail_AI_Store_Copilot.py  (jouw _Copilot.py)

import os
import time  # ✅ NEW: for retry backoff
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt  # eventueel later nog voor andere grafieken
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# BOVENAAN, naast andere imports
try:
    from openai import OpenAI
    _OPENAI_INSTALLED = True
except Exception:
    OpenAI = None
    _OPENAI_INSTALLED = False

from services.event_service import build_event_flags_for_dates

from datetime import datetime, timedelta

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response
from services.cbs_service import get_cbs_stats_for_postcode4
from services.forecast_service import (
    build_simple_footfall_turnover_forecast,
    build_pro_footfall_turnover_forecast,
)

# Probeer de service-import; als die er niet is, gebruik een lokale CSV-loader
try:
    from services.pathzz_service import fetch_monthly_street_traffic  # gebruikt sample weekly CSV
except Exception:
    def fetch_monthly_street_traffic(start_date, end_date):
        """
        Fallback: lees demo-straattraffic uit data/pathzz_sample_weekly.csv

        CSV-structuur:
        Week;Visits
        2025-10-05 To 2025-10-11;11.613  (→ 11613)
        ...

        Return:
        - week_start (datetime)
        - street_footfall (float)
        """
        csv_path = "data/pathzz_sample_weekly.csv"
        try:
            # Lees Visits expliciet als string, anders wordt 10.830 → 10.83 (float)
            df = pd.read_csv(csv_path, sep=";", dtype={"Visits": "string"})
        except Exception:
            return pd.DataFrame()

        # Kolommen hernoemen
        df = df.rename(columns={"Week": "week", "Visits": "street_footfall"})

        # Visits: "11.613" → "11613" → 11613.0
        df["street_footfall"] = (
            df["street_footfall"]
            .astype(str)
            .str.replace(".", "", regex=False)   # punt = duizendscheiding
            .str.replace(",", ".", regex=False)  # safety
            .astype(float)
        )

        # "2025-10-05 To 2025-10-11" → 2025-10-05
        def _parse_week_start(s):
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

st.set_page_config(
    page_title="PFM Retail Performance Copilot",
    layout="wide"
)


def _get_openai_client():
    if not _OPENAI_INSTALLED:
        return None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


_OPENAI_CLIENT = _get_openai_client()

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

VISUALCROSSING_KEY = st.secrets.get("visualcrossing_key", None)

# Maak de key ook beschikbaar als environment variable voor weather_service
if VISUALCROSSING_KEY:
    os.environ["VISUALCROSSING_API_KEY"] = VISUALCROSSING_KEY

# ----------------------
# PFM brand colors
# ----------------------
PFM_PURPLE = "#762181"
PFM_RED = "#F04438"
PFM_PEACH = "#FDBA8C"
PFM_PINK = "#F973A6"
PFM_BLUE = "#38BDF8"

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
# API helpers
# -------------

@st.cache_data(ttl=600)
def get_locations_by_company(company_id: int) -> pd.DataFrame:
    """
    ✅ Robust wrapper rond /company/{company_id}/location van de vemcount-agent.

    - Render/cold start friendly
    - Retry bij ReadTimeout
    - Langere timeouts
    """
    url = f"{FASTAPI_BASE_URL.rstrip('/')}/company/{company_id}/location"

    last_err = None
    # 2 pogingen: eerst redelijk, dan ruim (cold start / veel locaties)
    for attempt, timeout_s in enumerate([45, 120], start=1):
        try:
            resp = requests.get(url, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and "locations" in data:
                df = pd.DataFrame(data["locations"])
            else:
                df = pd.DataFrame(data)

            return df

        except requests.exceptions.ReadTimeout as e:
            last_err = e
            time.sleep(0.8 * attempt)  # kleine backoff
            continue

    # Als alles faalt → laat ReadTimeout omhoog bubbelen (main() vangt 'm af)
    raise requests.exceptions.ReadTimeout(
        f"Timeout bij ophalen locaties voor company {company_id} via {url}"
    ) from last_err


@st.cache_data(ttl=600)
def get_report(
    shop_ids,
    data_outputs,
    period: str,
    step: str = "day",
    source: str = "shops",
    company_id: int | None = None,  # voor toekomst, nu niet gebruikt
    form_date_from: str | None = None,
    form_date_to: str | None = None,
):
    """
    Wrapper rond /get-report (POST) van de vemcount-agent, met querystring zonder [].

    Als period="date" wordt gebruikt en form_date_from / form_date_to zijn gezet
    (YYYY-MM-DD), dan stuurt hij die als extra parameters mee.
    """
    params: list[tuple[str, str]] = []

    for sid in shop_ids:
        params.append(("data", str(sid)))

    for dout in data_outputs:
        params.append(("data_output", dout))

    params.append(("period", period))
    params.append(("step", step))
    params.append(("source", source))

    # Vrij datumbereik bij period="date"
    if period == "date" and form_date_from and form_date_to:
        params.append(("form_date_from", form_date_from))
        params.append(("form_date_to", form_date_to))

    resp = requests.post(REPORT_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# -------------
# Weather helper (Visual Crossing direct voor grafiek)
# -------------

@st.cache_data(ttl=3600)
def fetch_visualcrossing_history(location_str: str, start_date, end_date) -> pd.DataFrame:
    """
    Haalt historische daily weather data op via Visual Crossing.
    Wordt gebruikt voor de 'Weer vs footfall'-grafiek.
    Forecast gebruikt de centrale weather_service via forecast_service.
    """
    if not VISUALCROSSING_KEY:
        return pd.DataFrame()

    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location_str}/{start}/{end}"
    params = {
        "unitGroup": "metric",
        "key": VISUALCROSSING_KEY,
        "include": "days",
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "days" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["days"])
    df["date"] = pd.to_datetime(df["datetime"])
    return df[["date", "temp", "precip", "windspeed"]]


# -------------
# KPI helpers
# -------------

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
    Aggregatie naar weekniveau (zondag als week-start, passend bij Pathzz:
    '2025-11-02 To 2025-11-08' etc.).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    # Week met einde op zaterdag ⇒ start = zondag
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


def aggregate_monthly(df: pd.DataFrame, sqm: float | None) -> pd.DataFrame:
    """
    Aggregatie naar maandniveau.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    agg_dict: dict[str, str] = {}
    if "footfall" in df.columns:
        agg_dict["footfall"] = "sum"
    if "turnover" in df.columns:
        agg_dict["turnover"] = "sum"
    if "sales_per_visitor" in df.columns:
        agg_dict["sales_per_visitor"] = "mean"
    if "sales_per_sqm" in df.columns:
        agg_dict["sales_per_sqm"] = "mean"

    if not agg_dict:
        return df[["month"]].drop_duplicates().reset_index(drop=True)

    agg = df.groupby("month", as_index=False).agg(agg_dict)

    if sqm and sqm > 0:
        if "turnover" in agg.columns:
            agg["turnover_per_sqm"] = agg["turnover"] / sqm
        else:
            agg["turnover_per_sqm"] = np.nan

        if "footfall" in agg.columns:
            agg["footfall_per_sqm"] = agg["footfall"] / sqm
        else:
            agg["footfall_per_sqm"] = np.nan
    else:
        agg["turnover_per_sqm"] = np.nan
        agg["footfall_per_sqm"] = np.nan

    return agg


# -------------
# AI Store Coach helper
# -------------

def build_ai_store_coach_text(
    shop_name: str,
    brand: str,
    fc_res: dict,
    events_future_df: pd.DataFrame | None = None,
    weather_cfg: dict | None = None,
) -> str:
    """
    Genereert korte tekst met aanbevelingen.
    Probeert OpenAI te gebruiken; zo niet, simpele rule-based tekst.
    """
    recent_foot = fc_res.get("recent_footfall_sum", 0.0) or 0.0
    fut_foot = fc_res.get("forecast_footfall_sum", 0.0) or 0.0
    recent_turn = fc_res.get("recent_turnover_sum", np.nan)
    fut_turn = fc_res.get("forecast_turnover_sum", np.nan)

    model_type = fc_res.get("model_type", "unknown")
    used_fallback = fc_res.get("used_simple_fallback", False)

    # Event highlights (alleen toekomst)
    event_lines = []
    if events_future_df is not None and not events_future_df.empty:
        ef = events_future_df.copy()
        ef["date"] = pd.to_datetime(ef["date"]).dt.date
        for _, row in ef.iterrows():
            labels = []
            if row.get("is_black_friday", 0) == 1:
                labels.append("Black Friday-achtig moment")
            if row.get("is_december_trade", 0) == 1:
                labels.append("decemberpiek")
            if row.get("is_summer_sale", 0) == 1:
                labels.append("summer sale-periode")
            if row.get("is_school_holiday", 0) == 1:
                labels.append("schoolvakantie")
            if labels:
                event_lines.append(
                    f"- {row['date']}: " + ", ".join(labels)
                )

    base_summary = (
        f"Store: {shop_name} (brand: {brand}). "
        f"Laatste 14 dagen footfall: {recent_foot:.0f}, forecast volgende 14 dagen: {fut_foot:.0f}. "
    )
    if not pd.isna(recent_turn) and not pd.isna(fut_turn):
        base_summary += (
            f"Omzet laatste 14 dagen ~ {recent_turn:.0f}, verwacht ~ {fut_turn:.0f}. "
        )

    if used_fallback:
        base_summary += "Model heeft fallback naar simple DoW gebruikt. "
    else:
        base_summary += f"Modeltype: {model_type}. "

    if weather_cfg:
        base_summary += f"Weerlocatie: {weather_cfg.get('city','?')}, {weather_cfg.get('country','?')}. "

    if event_lines:
        base_summary += "Belangrijke komende dagen:\n" + "\n".join(event_lines)

    # OpenAI beschikbaar?
    if _OPENAI_CLIENT:
        prompt = f"""
Je bent een retail performance coach voor een filiaalmanager.

Gegevens:
{base_summary}

Geef in maximaal 5 bullets concrete acties voor de storemanager:
- focus op personeelsplanning rond drukke dagen,
- benutten van piek-momenten (events/feestdagen),
- ideeën voor conversie/SPV-verhoging,
- 1 bullet over hoe hij/zij dit aan het regioteam kan terugkoppelen.

Schrijf in het Nederlands, praktisch en to-the-point.
"""
        try:
            completion = _OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Je bent een ervaren retail operations coach."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
            return completion.choices[0].message.content.strip()
        except Exception:
            # Fallback naar simpele tekst
            return (
                "AI Store Coach kon niet worden aangeroepen. "
                "Basisinzicht: plan extra sterk op de drukste dagen, "
                "test één verbetering in conversie (bijv. begroetingsscript of kassaproces) "
                "en zorg dat je dit kort rapporteert aan je regiomanager."
            )

    # Rule-based fallback
    tips = [
        "• Bekijk de 2–3 drukste forecast-dagen en zorg daar voor extra bezetting aan de front-of-house.",
        "• Plan minimaal één test om de conversie te verhogen (bijv. actief begroeten of betere routing rond de bestverkopende categorie).",
        "• Gebruik de forecast-omzet als richtlijn voor je dagtargets en bespreek dit kort in de dagstart.",
        "• Kijk in de grafiek naar dagen waar footfall hoog is maar omzet of SPV achterblijft: dat zijn je snelste verbeterkansen.",
        "• Deel een korte samenvatting (1 slide of 3 bullets) met je regiomanager over wat je de komende 2 weken gaat testen.",
    ]
    if event_lines:
        tips.insert(
            1,
            "• Rond speciale dagen (bijvoorbeeld events/feestdagen in de forecast) kun je extra acties plannen: instore promo, demo of social-posts.",
        )
    return "\n".join(tips)

def build_store_ai_coach_text(
    store_name: str,
    recent_foot: float,
    recent_turn: float | float,
    fut_foot: float,
    fut_turn: float | float,
    spv_cur: float | float,
    conv_cur: float | float,
    fc: pd.DataFrame,
) -> str:
    """
    Bouwt een data-gedreven tekst voor de AI Store Coach
    op basis van historische vs forecast KPI's.
    """
    # Safeties
    recent_foot = float(recent_foot or 0)
    fut_foot = float(fut_foot or 0)
    recent_turn = float(recent_turn or 0)
    fut_turn = float(fut_turn or 0)

    has_recent_foot = recent_foot > 0
    has_recent_turn = recent_turn > 0
    has_spv_cur = pd.notna(spv_cur) and (spv_cur or 0) > 0
    has_conv_cur = pd.notna(conv_cur) and (conv_cur or 0) > 0

    # 1) Footfall-trend
    foot_msg = ""
    if has_recent_foot and fut_foot > 0:
        diff_foot = fut_foot - recent_foot
        diff_foot_pct = (diff_foot / recent_foot) * 100
        richting = "meer" if diff_foot > 0 else "minder"
        foot_msg = (
            f"- **Footfall-trend**: we verwachten ongeveer "
            f"{diff_foot_pct:+.1f}% {richting} bezoekers dan in de laatste 14 dagen "
            f"(~{fmt_int(abs(diff_foot))} bezoekers verschil).\n"
        )
    elif fut_foot > 0:
        foot_msg = (
            f"- **Footfall-trend**: forecast laat ongeveer {fmt_int(fut_foot)} bezoekers zien "
            f"voor de komende 14 dagen, maar er is te weinig historie om dit goed te vergelijken.\n"
        )
    else:
        foot_msg = (
            "- **Footfall-trend**: er is onvoldoende data om een betrouwbare bezoekersforecast te maken.\n"
        )

    # 2) Omzet-gap & scenario's
    omzet_msg = ""
    scenario_msg = ""

    if has_recent_turn and fut_turn > 0:
        diff_turn = fut_turn - recent_turn
        diff_turn_pct = (diff_turn / recent_turn) * 100
        richting = "meer" if diff_turn > 0 else "minder"
        omzet_msg = (
            f"- **Omzet-verwachting**: de forecast ligt ongeveer "
            f"{diff_turn_pct:+.1f}% {richting} dan de laatste 14 dagen "
            f"(~{fmt_eur(abs(diff_turn))} verschil).\n"
        )
    elif fut_turn > 0:
        omzet_msg = (
            f"- **Omzet-verwachting**: forecast omzet is ongeveer {fmt_eur(fut_turn)}, "
            "maar er is te weinig historie om dit te spiegelen aan een vorige periode.\n"
        )
    else:
        omzet_msg = (
            "- **Omzet-verwachting**: er is onvoldoende data om een betrouwbare omzetforecast te tonen.\n"
        )

    # 2a) Scenario: SPV +5%
    if fut_foot > 0 and fut_turn > 0:
        spv_forecast = fut_turn / fut_foot
        uplift_pct = 5.0
        extra_turn_spv = fut_foot * spv_forecast * uplift_pct / 100.0
        scenario_msg += (
            f"- **Scenario SPV +{uplift_pct:.0f}%**: als je in de komende 14 dagen de "
            f"besteding per bezoeker met ~{uplift_pct:.0f}% verhoogt, levert dat circa "
            f"{fmt_eur(extra_turn_spv)} extra omzet op bovenop de forecast.\n"
        )

    # 2b) Scenario: conversie +1 procentpunt (alleen als we conv & SPV hebben)
    if fut_foot > 0 and has_spv_cur and has_conv_cur:
        conv_baseline = float(conv_cur)
        spv_baseline = float(spv_cur)
        # ATV ≈ SPV / (conv% / 100)
        atv_est = spv_baseline * 100.0 / conv_baseline
        conv_new = conv_baseline + 1.0  # +1 pp
        extra_trans = fut_foot * (conv_new - conv_baseline) / 100.0
        extra_turn_conv = extra_trans * atv_est

        scenario_msg += (
            f"- **Scenario conversie +1 pp**: met een stijging van de conversie "
            f"van {conv_baseline:.1f}% naar {conv_new:.1f}% genereer je ongeveer "
            f"{fmt_int(extra_trans)} extra transacties en ~{fmt_eur(extra_turn_conv)} extra omzet "
            f"in de komende 14 dagen (bij gelijkblijvend bonbedrag).\n"
        )

    # 3) Piek- en rustigste dagen uit de forecast
    peak_msg = ""
    if isinstance(fc, pd.DataFrame) and not fc.empty and "footfall_forecast" in fc.columns:
        fc_local = fc.copy()
        fc_local["date"] = pd.to_datetime(fc_local["date"], errors="coerce")
        fc_local = fc_local.dropna(subset=["date"])

        if not fc_local.empty:
            top_days = (
                fc_local.sort_values("footfall_forecast", ascending=False)
                .head(3)
            )
            low_days = (
                fc_local.sort_values("footfall_forecast", ascending=True)
                .head(2)
            )

            def _fmt_day(row):
                d = row["date"]
                return f"{d.strftime('%a %d-%m')} (~{fmt_int(row['footfall_forecast'])} bezoekers)"

            top_str = ", ".join(_fmt_day(r) for _, r in top_days.iterrows())
            low_str = ", ".join(_fmt_day(r) for _, r in low_days.iterrows())

            peak_msg = (
                f"- **Piekmomenten benutten**: hoogste forecast ligt op {top_str}. "
                "Zorg hier voor maximale bemensing, actieve verkoop en duidelijke actiezones.\n"
                f"- **Stille momenten slim gebruiken**: rustigere dagen zijn {low_str}. "
                "Gebruik deze uren voor training, herindelen van het schap en voorbereiden van acties.\n"
            )

    # 4) Samenvattende actie
    action_msg = (
        "- **Focus voor deze periode**: combineer een scherpe personeelsplanning op drukke dagen "
        "met gerichte acties op SPV en conversie (bijvoorbeeld actieve begroeting, bundelaanbiedingen "
        "en duidelijke promoties bij de topcategorieën). Koppel na 2 weken kort terug wat het effect was "
        "op omzet vs. forecast.\n"
    )

    header = f"### AI Store Coach – komende 14 dagen\n\n"
    intro = (
        f"Voor **{store_name}** kijken we naar de combinatie van historische resultaten en de "
        "forecast voor de komende 14 dagen. Hieronder zie je waar je concreet op kunt sturen:\n\n"
    )

    return header + intro + foot_msg + omzet_msg + scenario_msg + peak_msg + action_msg

# -------------
# MAIN UI
# -------------

def main():
    st.title("PFM Retail Performance Copilot – Fase 1")

    # --- Retailer selectie via clients.json ---
    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)
    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    st.sidebar.header("Selecteer retailer & winkel")

    client_label = st.sidebar.selectbox("Retailer", clients_df["label"].tolist())
    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    # --- Winkels ophalen via FastAPI ---
    try:
        locations_df = get_locations_by_company(company_id)
    except requests.exceptions.ReadTimeout:
        st.error(
            "FastAPI locatie-endpoint reageert te traag (timeout). "
            "Dit gebeurt vaak bij een cold start of als er veel winkels zijn. "
            "Klik nog eens op **Analyseer** of probeer straks opnieuw."
        )
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Fout bij ophalen van winkels uit FastAPI: {e}")
        st.stop()

    if locations_df.empty:
        st.error("Geen winkels gevonden voor deze retailer.")
        return

    if "name" not in locations_df.columns:
        locations_df["name"] = locations_df["id"].astype(str)

    locations_df["label"] = locations_df.apply(
        lambda r: f"{r['name']} (ID: {r['id']})", axis=1
    )

    shop_label = st.sidebar.selectbox("Winkel", locations_df["label"].tolist())
    shop_row = locations_df[locations_df["label"] == shop_label].iloc[0].to_dict()
    shop_id = int(shop_row["id"])
    sqm = float(shop_row.get("sqm", 0) or 0)
    # postcode kan uit 'zip' of 'postcode' komen
    postcode = shop_row.get("zip") or shop_row.get("postcode", "")
    lat = float(shop_row.get("lat", 0) or 0)
    lon = float(shop_row.get("lon", 0) or 0)

    # --- Periode selectie (huidige vs vorige periode) ---
    period_choice = st.sidebar.selectbox(
        "Periode",
        [
            "Deze week",
            "Laatste week",
            "Deze maand",
            "Laatste maand",
            "Dit kwartaal",
            "Laatste kwartaal",
        ],
        index=2,  # default: Deze maand
    )

    today = datetime.today().date()

    # Vanaf welke datum mag de forecast-historie gebruikt worden?
    default_hist_start = today - timedelta(days=365)
    hist_start_input = st.sidebar.date_input(
        "Forecast-historie vanaf",
        value=default_hist_start,
        help=(
            "Deze datum bepaalt vanaf wanneer we historische data gebruiken om "
            "het forecast-model (Simple / Pro) te trainen."
        ),
    )

    def get_week_range(base_date):
        """Maandag–zondag week van base_date."""
        wd = base_date.weekday()  # 0=ma
        start = base_date - timedelta(days=wd)
        end = start + timedelta(days=6)
        return start, end

    def get_month_range(year, month):
        start = datetime(year, month, 1).date()
        if month == 12:
            next_start = datetime(year + 1, 1, 1).date()
        else:
            next_start = datetime(year, month + 1, 1).date()
        end = next_start - timedelta(days=1)
        return start, end

    def get_quarter_range(year, month):
        q = (month - 1) // 3 + 1
        start_month = 3 * (q - 1) + 1
        start = datetime(year, start_month, 1).date()
        if start_month == 10:
            next_start = datetime(year + 1, 1, 1).date()
        else:
            next_start = datetime(year, start_month + 3, 1).date()
        end = next_start - timedelta(days=1)
        return start, end

    # Bereken huidige + vorige periode
    if period_choice == "Deze week":
        start_cur, end_cur = get_week_range(today)
        start_prev, end_prev = start_cur - timedelta(days=7), start_cur - timedelta(days=1)

    elif period_choice == "Laatste week":
        this_week_start, _ = get_week_range(today)
        end_cur = this_week_start - timedelta(days=1)
        start_cur = end_cur - timedelta(days=6)
        start_prev = start_cur - timedelta(days=7)
        end_prev = start_cur - timedelta(days=1)

    elif period_choice == "Deze maand":
        start_cur, end_cur = get_month_range(today.year, today.month)
        if today.month == 1:
            prev_y, prev_m = today.year - 1, 12
        else:
            prev_y, prev_m = today.year, today.month - 1
        start_prev, end_prev = get_month_range(prev_y, prev_m)

    elif period_choice == "Laatste maand":
        if today.month == 1:
            cur_y, cur_m = today.year - 1, 12
        else:
            cur_y, cur_m = today.year, today.month - 1
        start_cur, end_cur = get_month_range(cur_y, cur_m)
        if cur_m == 1:
            prev_y, prev_m = cur_y - 1, 12
        else:
            prev_y, prev_m = cur_y, cur_m - 1
        start_prev, end_prev = get_month_range(prev_y, prev_m)

    elif period_choice == "Dit kwartaal":
        start_cur, end_cur = get_quarter_range(today.year, today.month)
        cur_q = (today.month - 1) // 3 + 1
        if cur_q == 1:
            prev_y, prev_q = today.year - 1, 4
        else:
            prev_y, prev_q = today.year, cur_q - 1
        prev_start_month = 3 * (prev_q - 1) + 1
        start_prev, end_prev = get_quarter_range(prev_y, prev_start_month)

    else:  # "Laatste kwartaal"
        cur_q = (today.month - 1) // 3 + 1
        if cur_q == 1:
            cur_y, cur_q_eff = today.year - 1, 4
        else:
            cur_y, cur_q_eff = today.year, cur_q - 1
        cur_start_month = 3 * (cur_q_eff - 1) + 1
        start_cur, end_cur = get_quarter_range(cur_y, cur_start_month)

        if cur_q_eff == 1:
            prev_y, prev_q = cur_y - 1, 4
        else:
            prev_y, prev_q = cur_y, cur_q_eff - 1
        prev_start_month = 3 * (prev_q - 1) + 1
        start_prev, end_prev = get_quarter_range(prev_y, prev_start_month)

    # --- Weather & CBS input ---
    default_city = shop_row.get("city") or "Amsterdam"
    weather_location = st.sidebar.text_input(
        "Weerlocatie",
        value=f"{default_city},NL",
    )
    postcode4 = st.sidebar.text_input(
        "CBS postcode (4-cijferig)",
        value=postcode[:4] if postcode else "",
    )

    forecast_mode = st.sidebar.radio(
        "Forecast mode",
        ["Simple (DoW)", "Pro (LightGBM beta)"],
        index=0,
    )

    run_btn = st.sidebar.button("Analyseer", type="primary")

    if not run_btn:
        st.info("Selecteer retailer & winkel, kies een periode en klik op **Analyseer**.")
        return

    # --- Data ophalen uit FastAPI ---
    with st.spinner("Data ophalen uit Storescan / FastAPI..."):
        metric_map = {
            "count_in": "footfall",
            "turnover": "turnover",
            "sales_per_sqm": "sales_per_sqm",
        }

        resp_all = get_report(
            [shop_id],
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
        st.warning("Geen data gevonden voor dit jaar voor deze winkel.")
        return

    df_all_raw["date"] = pd.to_datetime(df_all_raw["date"], errors="coerce")
    df_all_raw = df_all_raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ------------------------------
    # Extra history voor forecast (via period=date)
    # ------------------------------
    hist_end = datetime.today().date()
    # Gebruik de gekozen startdatum uit de sidebar, maar clamp naar vandaag
    hist_start = min(hist_start_input, hist_end)

    try:
        resp_hist = get_report(
            [shop_id],
            list(metric_map.keys()),
            period="date",
            step="day",
            source="shops",
            form_date_from=hist_start.strftime("%Y-%m-%d"),
            form_date_to=hist_end.strftime("%Y-%m-%d"),
        )
        df_hist_raw = normalize_vemcount_response(
            resp_hist,
            kpi_keys=metric_map.keys(),
        ).rename(columns=metric_map)

        df_hist_raw["date"] = pd.to_datetime(df_hist_raw["date"], errors="coerce")
        df_hist_raw = (
            df_hist_raw
            .dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        # safety fallback: als er om wat voor reden dan ook niks in zit
        if df_hist_raw.empty:
            df_hist_raw = df_all_raw.copy()

    except Exception:
        # Bij API-issue gewoon terugvallen op dit jaar
        df_hist_raw = df_all_raw.copy()

    # Slice naar huidige + vorige periode
    start_cur_ts = pd.Timestamp(start_cur)
    end_cur_ts = pd.Timestamp(end_cur)
    start_prev_ts = pd.Timestamp(start_prev)
    end_prev_ts = pd.Timestamp(end_prev)

    df_cur = df_all_raw[
        (df_all_raw["date"] >= start_cur_ts)
        & (df_all_raw["date"] <= end_cur_ts)
    ].copy()

    df_prev = df_all_raw[
        (df_all_raw["date"] >= start_prev_ts)
        & (df_all_raw["date"] <= end_prev_ts)
    ].copy()

    if df_cur.empty and df_prev.empty:
        st.warning("Geen data gevonden in de gekozen periodes.")
        return

    # KPI's berekenen op dag-niveau
    df_cur = compute_daily_kpis(df_cur)
    if not df_prev.empty:
        df_prev = compute_daily_kpis(df_prev)

    # --- Weerdata via Visual Crossing ---
    weather_df = pd.DataFrame()
    if weather_location and VISUALCROSSING_KEY:
        weather_df = fetch_visualcrossing_history(weather_location, start_cur, end_cur)

    # Config voor forecast-model (locatiestring)
    weather_cfg_base = None
    if VISUALCROSSING_KEY and weather_location:
        parts = weather_location.split(",")
        city = parts[0].strip() if parts else ""
        country = parts[1].strip() if len(parts) > 1 else ""
        weather_cfg_base = {
            "mode": "city_country",
            "city": city,
            "country": country,
            "location": weather_location.strip(),
        }

    # --- Pathzz street traffic (weekly, demo) ---
    pathzz_weekly = fetch_monthly_street_traffic(
        start_date=start_prev,
        end_date=end_cur,
    )

    capture_weekly = pd.DataFrame()
    avg_capture_cur = None
    avg_capture_prev = None

    if not pathzz_weekly.empty:
        # 1) Store weekly totals (één winkel)
        df_range = df_all_raw[
            (df_all_raw["date"] >= start_prev_ts)
            & (df_all_raw["date"] <= end_cur_ts)
        ].copy()
        df_range = compute_daily_kpis(df_range)
        store_weekly = aggregate_weekly(df_range)

        # 2) Street weekly – gemiddelde per week (i.p.v. som over alle regio's)
        pathzz_weekly["week_start"] = pd.to_datetime(pathzz_weekly["week_start"])
        street_weekly = (
            pathzz_weekly
            .groupby("week_start", as_index=False)["street_footfall"]
            .mean()
        )

        # 3) Merge store + street per week
        capture_weekly = pd.merge(
            store_weekly[["week_start", "footfall", "turnover"]],
            street_weekly,
            on="week_start",
            how="inner",
        )

        if not capture_weekly.empty:
            # 4) Capture rate per week (store vs street)
            capture_weekly["capture_rate"] = np.where(
                capture_weekly["street_footfall"] > 0,
                capture_weekly["footfall"] / capture_weekly["street_footfall"] * 100,
                np.nan,
            )

            # 5) Periode-tag voor delta in KPI-kaart
            capture_weekly = capture_weekly.sort_values("week_start")
            capture_weekly["period"] = np.where(
                (capture_weekly["week_start"] >= start_cur_ts)
                & (capture_weekly["week_start"] <= end_cur_ts),
                "huidige",
                "vorige",
            )

            # 6) Gemiddelde capture per periode
            avg_capture_cur = capture_weekly.loc[
                capture_weekly["period"] == "huidige", "capture_rate"
            ].mean()

            avg_capture_prev = capture_weekly.loc[
                capture_weekly["period"] == "vorige", "capture_rate"
            ].mean()

    # --- CBS context (data ophalen) ---
    cbs_stats = {}
    if postcode4:
        cbs_stats = get_cbs_stats_for_postcode4(postcode4)

    # --- KPI-cards met vergelijking vorige periode ---
    st.subheader(f"{selected_client['brand']} – {shop_row['name']}")

    foot_cur = df_cur["footfall"].sum() if "footfall" in df_cur.columns else 0
    foot_prev = df_prev["footfall"].sum() if ("footfall" in df_prev.columns and not df_prev.empty) else 0
    foot_delta = None
    if foot_prev > 0:
        foot_delta_val = (foot_cur - foot_prev) / foot_prev * 100
        foot_delta = f"{foot_delta_val:+.1f}%"

    turn_cur = df_cur["turnover"].sum() if "turnover" in df_cur.columns else 0
    turn_prev = df_prev["turnover"].sum() if ("turnover" in df_prev.columns and not df_prev.empty) else 0
    turn_delta = None
    if turn_prev > 0:
        turn_delta_val = (turn_cur - turn_prev) / turn_prev * 100
        turn_delta = f"{turn_delta_val:+.1f}%"

    spv_cur = df_cur["sales_per_visitor"].mean() if "sales_per_visitor" in df_cur.columns else np.nan
    spv_prev = df_prev["sales_per_visitor"].mean() if ("sales_per_visitor" in df_prev.columns and not df_prev.empty) else np.nan
    spv_delta = None
    if pd.notna(spv_cur) and pd.notna(spv_prev) and spv_prev > 0:
        spv_delta_val = (spv_cur - spv_prev) / spv_prev * 100
        spv_delta = f"{spv_delta_val:+.1f}%"

    conv_cur = df_cur["conversion_rate"].mean() if "conversion_rate" in df_cur.columns else np.nan
    conv_prev = df_prev["conversion_rate"].mean() if ("conversion_rate" in df_prev.columns and not df_prev.empty) else np.nan
    conv_delta = None
    if pd.notna(conv_cur) and pd.notna(conv_prev) and conv_prev > 0:
        conv_delta_val = (conv_cur - conv_prev) / conv_prev * 100
        conv_delta = f"{conv_delta_val:+.1f}%"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Footfall (periode)", fmt_int(foot_cur), delta=foot_delta)
    with col2:
        st.metric("Omzet (periode)", fmt_eur(turn_cur), delta=turn_delta)
    with col3:
        if "sales_per_visitor" in df_cur.columns:
            value = f"€ {spv_cur:.2f}".replace(".", ",") if pd.notna(spv_cur) else "-"
            st.metric("Gem. besteding/visitor", value, delta=spv_delta)
    with col4:
        if avg_capture_cur is not None and not pd.isna(avg_capture_cur) and avg_capture_prev not in (None, 0):
            delta_val = None
            if avg_capture_prev and avg_capture_prev > 0:
                delta_val = (avg_capture_cur - avg_capture_prev) / avg_capture_prev * 100
            st.metric(
                "Gem. capture rate",
                fmt_pct(avg_capture_cur),
                delta=f"{delta_val:+.1f}%" if delta_val is not None else None,
            )
        elif "conversion_rate" in df_cur.columns:
            st.metric(
                "Gem. conversie",
                fmt_pct(conv_cur) if pd.notna(conv_cur) else "-",
                delta=conv_delta,
            )

    # --- Weekly grafiek: straatdrukte vs winkeltraffic + omzet + capture rate ---
    st.markdown("### Straatdrukte vs winkeltraffic (weekly demo)")

    if not capture_weekly.empty:
        chart_df = capture_weekly[
            ["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]
        ].copy()

        # Weeklabel als nette weeknummers, bv. W01, W02, ...
        iso_cal = chart_df["week_start"].dt.isocalendar()
        chart_df["week_label"] = iso_cal.week.apply(lambda w: f"W{int(w):02d}")

        week_order = (
            chart_df.sort_values("week_start")["week_label"]
            .unique()
            .tolist()
        )

        fig_week = make_subplots(specs=[[{"secondary_y": True}]])

        # Footfall bar
        fig_week.add_bar(
            x=chart_df["week_label"],
            y=chart_df["footfall"],
            name="Footfall (store)",
            marker_color=PFM_PURPLE,
            offsetgroup=0,
        )

        # Street traffic bar
        fig_week.add_bar(
            x=chart_df["week_label"],
            y=chart_df["street_footfall"],
            name="Street traffic",
            marker_color=PFM_PEACH,
            opacity=0.7,
            offsetgroup=1,
        )

        # Turnover bar
        fig_week.add_bar(
            x=chart_df["week_label"],
            y=chart_df["turnover"],
            name="Omzet (€)",
            marker_color=PFM_PINK,
            opacity=0.7,
            offsetgroup=2,
        )

        # Capture rate line (store)
        fig_week.add_trace(
            go.Scatter(
                x=chart_df["week_label"],
                y=chart_df["capture_rate"],
                name="Capture rate (%)",
                mode="lines+markers",
                line=dict(color=PFM_RED, width=2),
            ),
            secondary_y=True,
        )

        fig_week.update_xaxes(
            title_text="Week",
            categoryorder="array",
            categoryarray=week_order,
        )
        fig_week.update_yaxes(
            title_text="Footfall / street traffic / omzet (€)",
            secondary_y=False,
        )
        fig_week.update_yaxes(
            title_text="Capture rate (%)",
            secondary_y=True,
        )

        fig_week.update_layout(
            barmode="group",
            height=350,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=40, r=40, t=40, b=40),
        )

        st.plotly_chart(fig_week, use_container_width=True)
    else:
        st.info("Geen Pathzz-weekdata beschikbaar voor deze periode.")

    # --- Dagelijkse grafiek ---
    st.markdown("### Dagelijkse footfall & omzet")
    if "footfall" in df_cur.columns and "turnover" in df_cur.columns:
        daily_df = df_cur[["date", "footfall", "turnover"]].copy()

        fig_daily = make_subplots(specs=[[{"secondary_y": True}]])

        fig_daily.add_bar(
            x=daily_df["date"],
            y=daily_df["footfall"],
            name="Footfall",
            marker_color=PFM_PURPLE,
        )

        fig_daily.add_trace(
            go.Scatter(
                x=daily_df["date"],
                y=daily_df["turnover"],
                name="Turnover (€)",
                mode="lines+markers",
                line=dict(color=PFM_PEACH, width=2),
            ),
            secondary_y=True,
        )

        fig_daily.update_xaxes(title_text="")
        fig_daily.update_yaxes(title_text="Footfall", secondary_y=False)
        fig_daily.update_yaxes(title_text="Omzet (€)", secondary_y=True)

        fig_daily.update_layout(
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=40, t=20, b=40),
        )

        st.plotly_chart(fig_daily, use_container_width=True)

    # --- Weer vs footfall (optioneel) ---
    if not weather_df.empty:
        st.markdown("### Weer vs footfall (indicatief)")

        weather_merge = pd.merge(
            df_cur[["date", "footfall"]],
            weather_df[["date", "temp", "precip"]],
            on="date",
            how="left",
        )

        fig_weather = make_subplots(specs=[[{"secondary_y": True}]])

        # Footfall bars
        fig_weather.add_bar(
            x=weather_merge["date"],
            y=weather_merge["footfall"],
            name="Footfall",
            marker_color=PFM_PURPLE,
        )

        # Temperature line
        fig_weather.add_trace(
            go.Scatter(
                x=weather_merge["date"],
                y=weather_merge["temp"],
                name="Temperatuur (°C)",
                mode="lines+markers",
                line=dict(color=PFM_RED, width=2),
            ),
            secondary_y=True,
        )

        # Precipitation line (mm)
        fig_weather.add_trace(
            go.Scatter(
                x=weather_merge["date"],
                y=weather_merge["precip"],
                name="Neerslag (mm)",
                mode="lines+markers",
                line=dict(color=PFM_BLUE, width=2, dash="dot"),
            ),
            secondary_y=True,
        )

        fig_weather.update_xaxes(title_text="")
        fig_weather.update_yaxes(title_text="Footfall", secondary_y=False)
        fig_weather.update_yaxes(
            title_text="Temperatuur (°C) / neerslag (mm)",
            secondary_y=True,
        )

        fig_weather.update_layout(
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=40, t=20, b=40),
        )

        st.plotly_chart(fig_weather, use_container_width=True)

    # --- Forecast: footfall & omzet (volgende 14 dagen) ---
    st.markdown("### Forecast: footfall & omzet (volgende 14 dagen)")

    # Uitleg voor demo / klant
    st.markdown(
        """
        <small>
        💡 <strong>Forecast uitleg</strong><br>
        - <strong>Simple (DoW)</strong> gebruikt gemiddelden per weekdag op basis van je historische data.<br>
        - <strong>Pro (LightGBM beta)</strong> bouwt hierop voort met seizoenseffecten (Q4, feestdagen),
          lags/rolling averages en optioneel weerdata.<br>
        - Als er (nog) te weinig bruikbare historie is of LightGBM niet beschikbaar is, valt Pro automatisch terug op Simple.
        </small>
        """,
        unsafe_allow_html=True,
    )

    # Weather config voor forecast_service (op basis van weerlocatie input)
    weather_cfg = None
    if VISUALCROSSING_KEY and weather_location:
        # Verwacht iets als "Amsterdam,NL" of "Rotterdam,Nederland"
        city_part = weather_location
        country = "Netherlands"

        parts = weather_location.split(",")
        if len(parts) >= 1:
            city_part = parts[0].strip()
        if len(parts) >= 2:
            country_code = parts[1].strip().upper()
            country_map = {
                "NL": "Netherlands",
                "BE": "Belgium",
                "DE": "Germany",
                "FR": "France",
                "UK": "United Kingdom",
                "GB": "United Kingdom",
            }
            country = country_map.get(country_code, country_code)

        weather_cfg = {
            "mode": "city_country",
            "city": city_part,
            "country": country,
            "api_key": VISUALCROSSING_KEY,
        }

    try:
        if forecast_mode == "Pro (LightGBM beta)":
            fc_res = build_pro_footfall_turnover_forecast(
                df_hist_raw,
                horizon=14,
                min_history_days=60,
                weather_cfg=weather_cfg,
                use_weather=bool(weather_cfg),
            )
        else:
            fc_res = build_simple_footfall_turnover_forecast(df_hist_raw)

        if not fc_res.get("enough_history", False):
            st.info("Te weinig historische data om een betrouwbare forecast te tonen.")
        else:
            hist_recent = fc_res["hist_recent"]
            fc = fc_res["forecast"]

            recent_foot = fc_res["recent_footfall_sum"]
            recent_turn = fc_res["recent_turnover_sum"]
            fut_foot = fc_res["forecast_footfall_sum"]
            fut_turn = fc_res["forecast_turnover_sum"]

            c_sm, c_model = st.columns([3, 1])
            with c_model:
                st.caption(f"Model: **{fc_res['model_type']}**")
                if fc_res.get("used_simple_fallback", False):
                    st.caption("Fallback → Simple DoW")

            c1, c2 = st.columns(2)
            with c1:
                delta_foot = None
                if recent_foot > 0:
                    delta_foot = f"{(fut_foot - recent_foot) / recent_foot * 100:+.1f}%"
                st.metric(
                    "Verwachte bezoekers (14 dagen)",
                    fmt_int(fut_foot),
                    delta=delta_foot,
                )

            with c2:
                delta_turn = None
                if not pd.isna(recent_turn) and recent_turn > 0:
                    delta_turn = f"{(fut_turn - recent_turn) / recent_turn * 100:+.1f}%"
                st.metric(
                    "Verwachte omzet (14 dagen)",
                    fmt_eur(fut_turn),
                    delta=delta_turn,
                )

            # --- Verwachte maandomzet (actueel + forecast rest van maand) ---
            month_start = datetime(today.year, today.month, 1).date()
            today_date = today

            # Gerealiseerde omzet deze maand (t/m vandaag)
            if "turnover" in df_all_raw.columns:
                actual_month_turn = df_all_raw[
                    (df_all_raw["date"].dt.date >= month_start)
                    & (df_all_raw["date"].dt.date <= today_date)
                ]["turnover"].sum()
            else:
                actual_month_turn = 0.0

            # Forecast voor resterende dagen van deze maand (binnen 14-daagse horizon)
            future_month_turn = 0.0
            if (
                isinstance(fc, pd.DataFrame)
                and "date" in fc.columns
                and "turnover_forecast" in fc.columns
            ):
                fc_month = fc.copy()
                fc_month["date"] = pd.to_datetime(fc_month["date"], errors="coerce")

                fc_month = fc_month.dropna(subset=["date"])
                if not fc_month.empty:
                    future_month_turn = fc_month[
                        (fc_month["date"].dt.date > today_date)
                        & (fc_month["date"].dt.month == today_date.month)
                    ]["turnover_forecast"].sum()

            expected_month_turn = actual_month_turn + future_month_turn

            col_month, _ = st.columns([2, 3])
            with col_month:
                st.markdown("#### Verwachte omzet – huidige maand")
                st.metric(
                    "Verwachte omzet deze maand",
                    fmt_eur(expected_month_turn),
                )
                if actual_month_turn > 0:
                    remaining = expected_month_turn - actual_month_turn
                    st.caption(
                        f"Gerealiseerd tot nu toe: {fmt_eur(actual_month_turn)} · "
                        f"Verwachte extra omzet rest van de maand: {fmt_eur(remaining)} "
                        f"(binnen 14-daagse forecast horizon)."
                    )

            # Grafiek: laatste 28 dagen historisch + forecast
            fig_fc = make_subplots(specs=[[{"secondary_y": True}]])

            fig_fc.add_bar(
                x=hist_recent["date"],
                y=hist_recent["footfall"],
                name="Footfall (historisch)",
                marker_color=PFM_PURPLE,
            )

            fig_fc.add_bar(
                x=fc["date"],
                y=fc["footfall_forecast"],
                name="Footfall forecast",
                marker_color=PFM_PEACH,
            )

            if "turnover" in hist_recent.columns:
                fig_fc.add_trace(
                    go.Scatter(
                        x=hist_recent["date"],
                        y=hist_recent["turnover"],
                        name="Omzet historisch",
                        mode="lines",
                        line=dict(color=PFM_PINK, width=2),
                    ),
                    secondary_y=True,
                )

            fig_fc.add_trace(
                go.Scatter(
                    x=fc["date"],
                    y=fc["turnover_forecast"],
                    name="Omzet forecast",
                    mode="lines+markers",
                    line=dict(color=PFM_RED, width=2, dash="dash"),
                ),
                secondary_y=True,
            )

            fig_fc.update_xaxes(title_text="")
            fig_fc.update_yaxes(title_text="Footfall", secondary_y=False)
            fig_fc.update_yaxes(title_text="Omzet (€)", secondary_y=True)

            fig_fc.update_layout(
                height=350,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
                margin=dict(l=40, r=40, t=20, b=40),
            )

            st.plotly_chart(fig_fc, use_container_width=True)

            # --- DATA-GEDREVEN AI STORE COACH ---
            coach_text = build_store_ai_coach_text(
                store_name=shop_row["name"],
                recent_foot=recent_foot,
                recent_turn=recent_turn,
                fut_foot=fut_foot,
                fut_turn=fut_turn,
                spv_cur=spv_cur,
                conv_cur=conv_cur,
                fc=fc,
            )
            st.markdown(coach_text)

    except Exception as e:
        st.info(
            "Forecast kon niet worden berekend (te weinig data, ontbrekende kolommen of weerdata-issue)."
        )
        st.exception(e)

    # --- Debug ---
    with st.expander("🔧 Debug"):
        st.write("Periode keuze:", period_choice)
        st.write("Huidige periode:", start_cur, "→", end_cur)
        st.write("Vorige periode:", start_prev, "→", end_prev)
        st.write("Shop row:", shop_row)
        st.write("Dagdata ALL (head):", df_all_raw.head())
        st.write("Dagdata (cur):", df_cur.head())
        st.write("Dagdata (prev):", df_prev.head())
        st.write("Pathzz weekly:", pathzz_weekly.head())
        st.write(
            "Capture weekly:",
            capture_weekly.head()
            if isinstance(capture_weekly, pd.DataFrame)
            else capture_weekly,
        )
        st.write("CBS stats:", cbs_stats)
        st.write("Weather df:", weather_df.head())
        st.write("Forecast mode:", forecast_mode)
        st.write("Weather cfg (base):", weather_cfg_base)

        try:
            st.write("Forecast model_type:", fc_res.get("model_type"))
            st.write("Forecast used_simple_fallback:", fc_res.get("used_simple_fallback"))
            st.write("Forecast head:", fc_res["forecast"].head())
        except Exception:
            st.write("Forecast object nog niet beschikbaar in deze run.")


if __name__ == "__main__":
    main()