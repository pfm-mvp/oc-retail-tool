    # pages/05_Retail_AI_Store_Copilot.py

import os
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import html  # ✅ for safe HTML escaping inside unsafe_allow_html blocks

try:
    from openai import OpenAI
    _OPENAI_INSTALLED = True
except Exception:
    OpenAI = None
    _OPENAI_INSTALLED = False

from datetime import datetime, timedelta, date

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response
from services.cbs_service import get_cbs_stats_for_postcode4
from services.forecast_service import (
    build_simple_footfall_turnover_forecast,
    build_pro_footfall_turnover_forecast,
)

# ✅ REUSE EXISTING SVI SERVICE (single source of truth)
from services.svi_service import build_store_vitality

# Probeer de service-import; als die er niet is, gebruik een lokale CSV-loader
try:
    from services.pathzz_service import fetch_monthly_street_traffic
except Exception:
    def fetch_monthly_street_traffic(start_date, end_date):
        csv_path = "data/pathzz_sample_weekly.csv"
        try:
            df = pd.read_csv(csv_path, sep=";", dtype={"Visits": "string"})
        except Exception:
            return pd.DataFrame()

        df = df.rename(columns={"Week": "week", "Visits": "street_footfall"})
        df["street_footfall"] = (
            df["street_footfall"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

        def _parse_week_start(s):
            if isinstance(s, str) and "To" in s:
                return pd.to_datetime(s.split("To")[0].strip(), errors="coerce")
            return pd.NaT

        df["week_start"] = df["week"].apply(_parse_week_start)
        df = df.dropna(subset=["week_start"])

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df = df[(df["week_start"] >= start) & (df["week_start"] <= end)]

        return df[["week_start", "street_footfall"]].reset_index(drop=True)

st.set_page_config(
    page_title="PFM Retail Performance Copilot",
    layout="wide"
)

# ----------------------
# OpenAI helper
# ----------------------
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
PFM_GRAY = "#6B7280"
PFM_LINE = "#E5E7EB"
PFM_DARK = "#111827"

# Light purple for forecast bars (requested)
PFM_PURPLE_LIGHT_RGBA = "rgba(118,33,129,0.28)"   # very light purple
PFM_PURPLE_LIGHT_LINE = "rgba(118,33,129,0.55)"   # slightly stronger for forecast line

PFM_GREEN = "#22C55E"
PFM_AMBER = "#F59E0B"

# ----------------------
# Global CSS: nicer KPI cards with colored deltas
# ----------------------
st.markdown(
    f"""
    <style>
      .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 14px;
        align-items: stretch;
      }}

      .kpi-card {{
        border: 1px solid {PFM_LINE};
        border-radius: 14px;
        background: white;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(17,24,39,0.04);
        min-height: 92px;
      }}

      .kpi-title {{
        font-size: 12px;
        font-weight: 700;
        color: {PFM_GRAY};
        margin-bottom: 6px;
      }}

      .kpi-value {{
        font-size: 26px;
        font-weight: 900;
        color: {PFM_DARK};
        line-height: 1.05;
      }}

      .kpi-sub {{
        font-size: 12px;
        color: {PFM_GRAY};
        margin-top: 8px;
      }}

      .delta-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-top: 8px;
        padding: 4px 8px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        width: fit-content;
      }}

      .delta-up {{
        background: rgba(34, 197, 94, 0.12);
        color: {PFM_GREEN};
        border: 1px solid rgba(34, 197, 94, 0.25);
      }}

      .delta-down {{
        background: rgba(240, 68, 56, 0.12);
        color: {PFM_RED};
        border: 1px solid rgba(240, 68, 56, 0.25);
      }}

      .delta-flat {{
        background: rgba(107, 114, 128, 0.10);
        color: {PFM_GRAY};
        border: 1px solid rgba(107, 114, 128, 0.18);
      }}

      .section-title {{
        margin-top: 14px;
        font-size: 22px;
        font-weight: 900;
        color: {PFM_DARK};
      }}

      .muted {{
        color: {PFM_GRAY};
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------
# Format helpers
# -------------
def fmt_eur(x: float) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"€ {float(x):,.0f}".replace(",", ".")

def fmt_pct(x: float) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):.1f}%".replace(".", ",")

def fmt_int(x: float) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):,.0f}".replace(",", ".")

def safe_num(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def pct_delta(cur: float, prev: float):
    cur = safe_num(cur)
    prev = safe_num(prev)
    if pd.isna(cur) or pd.isna(prev) or prev == 0:
        return None
    return (cur - prev) / prev * 100.0

def delta_pill_html(delta_pct: float | None, label_suffix: str = "") -> str:
    if delta_pct is None or pd.isna(delta_pct):
        return f'<div class="delta-pill delta-flat">— {label_suffix}</div>' if label_suffix else '<div class="delta-pill delta-flat">—</div>'
    arrow = "▲" if delta_pct > 0 else ("▼" if delta_pct < 0 else "•")
    cls = "delta-up" if delta_pct > 0 else ("delta-down" if delta_pct < 0 else "delta-flat")
    txt = f"{delta_pct:+.1f}%".replace(".", ",")
    if label_suffix:
        txt = f"{txt} {label_suffix}"
    return f'<div class="delta-pill {cls}">{arrow} {txt}</div>'

def kpi_card_html(title: str, value: str, sub: str = "", delta_pct: float | None = None, delta_label: str = "") -> str:
    delta_html = ""
    if delta_pct is not None or delta_label:
        delta_html = delta_pill_html(delta_pct, delta_label)
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
      <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
        {sub_html}
      </div>
    """

# ----------------------
# SVI UI helpers
# ----------------------
def svi_style(score: float | None):
    if score is None or pd.isna(score):
        return {"label": "No score", "color": PFM_LINE, "emoji": "⚪", "hint": "Not enough data"}
    s = float(score)
    if s >= 75:
        return {"label": "High performance", "color": PFM_GREEN, "emoji": "🟢", "hint": "Top tier in your region"}
    if s >= 60:
        return {"label": "Good / stable", "color": PFM_PURPLE, "emoji": "🟣", "hint": "Solid performance vs peers"}
    if s >= 45:
        return {"label": "Attention required", "color": PFM_AMBER, "emoji": "🟠", "hint": "Below peers on key drivers"}
    return {"label": "Under pressure", "color": PFM_RED, "emoji": "🔴", "hint": "Material gap vs regional peers"}

# ----------------------
# Region mapping
# ----------------------
@st.cache_data(ttl=600)
def load_region_mapping(path: str = "data/regions.csv") -> pd.DataFrame:
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

# -------------
# API helpers
# -------------
@st.cache_data(ttl=600)
def get_locations_by_company(company_id: int) -> pd.DataFrame:
    url = f"{FASTAPI_BASE_URL.rstrip('/')}/company/{company_id}/location"

    last_err = None
    for attempt, timeout_s in enumerate([45, 120], start=1):
        try:
            resp = requests.get(url, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "locations" in data:
                return pd.DataFrame(data["locations"])
            return pd.DataFrame(data)
        except requests.exceptions.ReadTimeout as e:
            last_err = e
            time.sleep(0.8 * attempt)

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
    form_date_from: str | None = None,
    form_date_to: str | None = None,
):
    params: list[tuple[str, str]] = []
    for sid in shop_ids:
        params.append(("data", str(sid)))
    for dout in data_outputs:
        params.append(("data_output", dout))
    params.append(("period", period))
    params.append(("step", step))
    params.append(("source", source))
    if period == "date" and form_date_from and form_date_to:
        params.append(("form_date_from", form_date_from))
        params.append(("form_date_to", form_date_to))
    resp = requests.post(REPORT_URL, params=params, timeout=90)
    resp.raise_for_status()
    return resp.json()

# -------------
# Weather helper
# -------------
@st.cache_data(ttl=3600)
def fetch_visualcrossing_history(location_str: str, start_date, end_date) -> pd.DataFrame:
    if not VISUALCROSSING_KEY:
        return pd.DataFrame()

    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location_str}/{start}/{end}"
    params = {"unitGroup": "metric", "key": VISUALCROSSING_KEY, "include": "days"}

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
    for c in ["footfall", "turnover", "transactions"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "turnover" in df.columns and "footfall" in df.columns:
        df["sales_per_visitor"] = np.where(df["footfall"] > 0, df["turnover"] / df["footfall"], np.nan)
    if "transactions" in df.columns and "footfall" in df.columns:
        df["conversion_rate"] = np.where(df["footfall"] > 0, df["transactions"] / df["footfall"] * 100, np.nan)
    return df

def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
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

# -------------
# Like-for-like slice helper (fixes your “half month vs full month” issue)
# -------------
def like_for_like_ranges(start_cur: date, end_cur: date, start_prev: date, end_prev: date, today: date):
    end_cur_eff = min(end_cur, today)  # only days that have happened
    cur_days = (end_cur_eff - start_cur).days + 1
    end_prev_eff = start_prev + timedelta(days=max(cur_days - 1, 0))
    end_prev_eff = min(end_prev_eff, end_prev)
    return start_cur, end_cur_eff, start_prev, end_prev_eff

# -------------
# Simple DOW forecast for arbitrary horizon (month remaining)
# -------------
def forecast_by_dow(df_hist: pd.DataFrame, future_dates: pd.DatetimeIndex):
    """
    Forecast footfall + turnover for arbitrary future dates using weekday means from history.
    """
    if df_hist is None or df_hist.empty or future_dates is None or len(future_dates) == 0:
        return pd.DataFrame()

    h = df_hist.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna(subset=["date"])
    if h.empty:
        return pd.DataFrame()

    # last 90 days = stable-ish for “booth demo” (less spiky)
    cutoff = h["date"].max() - pd.Timedelta(days=90)
    h = h[h["date"] >= cutoff].copy()

    for col in ["footfall", "turnover"]:
        if col not in h.columns:
            h[col] = np.nan
        h[col] = pd.to_numeric(h[col], errors="coerce")

    h["dow"] = h["date"].dt.weekday
    means = h.groupby("dow", as_index=False)[["footfall", "turnover"]].mean()

    fc = pd.DataFrame({"date": future_dates})
    fc["dow"] = fc["date"].dt.weekday
    fc = fc.merge(means, on="dow", how="left")

    fc = fc.rename(columns={"footfall": "footfall_forecast", "turnover": "turnover_forecast"})
    return fc[["date", "footfall_forecast", "turnover_forecast"]]

# -------------
# AI Store Coach helper (unchanged)
# -------------
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
    recent_foot = float(recent_foot or 0)
    fut_foot = float(fut_foot or 0)
    recent_turn = float(recent_turn or 0)
    fut_turn = float(fut_turn or 0)

    has_recent_foot = recent_foot > 0
    has_recent_turn = recent_turn > 0
    has_spv_cur = pd.notna(spv_cur) and (spv_cur or 0) > 0
    has_conv_cur = pd.notna(conv_cur) and (conv_cur or 0) > 0

    foot_msg = ""
    if has_recent_foot and fut_foot > 0:
        diff_foot = fut_foot - recent_foot
        diff_foot_pct = (diff_foot / recent_foot) * 100
        direction = "more" if diff_foot > 0 else "less"
        foot_msg = (
            f"- **Footfall outlook**: ~{diff_foot_pct:+.1f}% {direction} visitors vs last 14 days "
            f"(~{fmt_int(abs(diff_foot))} difference).\n"
        )
    elif fut_foot > 0:
        foot_msg = (
            f"- **Footfall outlook**: ~{fmt_int(fut_foot)} visitors expected in the next 14 days, "
            "but we lack enough history for a solid comparison.\n"
        )
    else:
        foot_msg = "- **Footfall outlook**: insufficient data for a reliable forecast.\n"

    omzet_msg = ""
    if has_recent_turn and fut_turn > 0:
        diff_turn = fut_turn - recent_turn
        diff_turn_pct = (diff_turn / recent_turn) * 100
        direction = "more" if diff_turn > 0 else "less"
        omzet_msg = (
            f"- **Turnover outlook**: ~{diff_turn_pct:+.1f}% {direction} vs last 14 days "
            f"(~{fmt_eur(abs(diff_turn))} difference).\n"
        )
    elif fut_turn > 0:
        omzet_msg = (
            f"- **Turnover outlook**: forecast turnover ~{fmt_eur(fut_turn)}, "
            "but we lack enough history for a solid comparison.\n"
        )
    else:
        omzet_msg = "- **Turnover outlook**: insufficient data for a reliable turnover forecast.\n"

    scenario_msg = ""
    if fut_foot > 0 and fut_turn > 0:
        spv_forecast = fut_turn / fut_foot
        uplift_pct = 5.0
        extra_turn_spv = fut_foot * spv_forecast * uplift_pct / 100.0
        scenario_msg += (
            f"- **Scenario SPV +{uplift_pct:.0f}%**: ~{fmt_eur(extra_turn_spv)} extra turnover "
            f"on top of the forecast (next 14 days).\n"
        )

    if fut_foot > 0 and has_spv_cur and has_conv_cur:
        conv_baseline = float(conv_cur)
        spv_baseline = float(spv_cur)
        atv_est = spv_baseline * 100.0 / conv_baseline
        conv_new = conv_baseline + 1.0
        extra_trans = fut_foot * (conv_new - conv_baseline) / 100.0
        extra_turn_conv = extra_trans * atv_est
        scenario_msg += (
            f"- **Scenario +1pp conversion**: ~{fmt_int(extra_trans)} extra transactions and "
            f"~{fmt_eur(extra_turn_conv)} extra turnover (next 14 days).\n"
        )

    peak_msg = ""
    if isinstance(fc, pd.DataFrame) and not fc.empty and "footfall_forecast" in fc.columns:
        fc_local = fc.copy()
        fc_local["date"] = pd.to_datetime(fc_local["date"], errors="coerce")
        fc_local = fc_local.dropna(subset=["date"])
        if not fc_local.empty:
            top_days = fc_local.sort_values("footfall_forecast", ascending=False).head(3)
            low_days = fc_local.sort_values("footfall_forecast", ascending=True).head(2)

            def _fmt_day(row):
                d = row["date"]
                return f"{d.strftime('%a %d-%m')} (~{fmt_int(row['footfall_forecast'])})"

            top_str = ", ".join(_fmt_day(r) for _, r in top_days.iterrows())
            low_str = ", ".join(_fmt_day(r) for _, r in low_days.iterrows())

            peak_msg = (
                f"- **Peak days to win**: {top_str}. Staff up + push active selling.\n"
                f"- **Quieter days to optimize**: {low_str}. Use for training + setup.\n"
            )

    action_msg = (
        "- **Focus for this period**: nail staffing on peak days + run one conversion/SPV experiment "
        "(greeting script, bundles, sharper promo placement). Share results back to the region team.\n"
    )

    header = "### AI Store Coach — next 14 days\n\n"
    intro = (
        f"For **{store_name}**, we combine recent performance with the next 14-day forecast. "
        "Here’s what to act on:\n\n"
    )

    return header + intro + foot_msg + omzet_msg + scenario_msg + peak_msg + action_msg

# -------------
# MAIN UI
# -------------
def main():
    st.title("PFM Retail Performance Copilot – Store Manager")

    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)
    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    st.sidebar.header("Select retailer & store")

    client_label = st.sidebar.selectbox("Retailer", clients_df["label"].tolist())
    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    # --- Locations ---
    try:
        locations_df = get_locations_by_company(company_id)
    except requests.exceptions.ReadTimeout:
        st.error(
            "FastAPI location endpoint timed out (cold start / many locations). "
            "Click **Analyse** again or try later."
        )
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching locations: {e}")
        st.stop()

    if locations_df.empty:
        st.error("No stores found for this retailer.")
        return

    if "name" not in locations_df.columns:
        locations_df["name"] = locations_df["id"].astype(str)

    locations_df["label"] = locations_df.apply(lambda r: f"{r['name']} (ID: {r['id']})", axis=1)

    shop_label = st.sidebar.selectbox("Store", locations_df["label"].tolist())
    shop_row = locations_df[locations_df["label"] == shop_label].iloc[0].to_dict()
    shop_id = int(shop_row["id"])
    postcode = shop_row.get("zip") or shop_row.get("postcode", "")

    # --- Period selector ---
    period_choice = st.sidebar.selectbox(
        "Period",
        ["This week", "Last week", "This month", "Last month", "This quarter", "Last quarter"],
        index=2,
    )

    today = datetime.today().date()

    default_hist_start = today - timedelta(days=365)
    hist_start_input = st.sidebar.date_input(
        "Forecast history from",
        value=default_hist_start,
        help="This date controls how much history we use to train the forecast model.",
    )

    def get_week_range(base_date):
        wd = base_date.weekday()
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

    if period_choice == "This week":
        start_cur, end_cur = get_week_range(today)
        start_prev, end_prev = start_cur - timedelta(days=7), start_cur - timedelta(days=1)

    elif period_choice == "Last week":
        this_week_start, _ = get_week_range(today)
        end_cur = this_week_start - timedelta(days=1)
        start_cur = end_cur - timedelta(days=6)
        start_prev = start_cur - timedelta(days=7)
        end_prev = start_cur - timedelta(days=1)

    elif period_choice == "This month":
        start_cur, end_cur = get_month_range(today.year, today.month)
        if today.month == 1:
            prev_y, prev_m = today.year - 1, 12
        else:
            prev_y, prev_m = today.year, today.month - 1
        start_prev, end_prev = get_month_range(prev_y, prev_m)

    elif period_choice == "Last month":
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

    elif period_choice == "This quarter":
        start_cur, end_cur = get_quarter_range(today.year, today.month)
        cur_q = (today.month - 1) // 3 + 1
        if cur_q == 1:
            prev_y, prev_q = today.year - 1, 4
        else:
            prev_y, prev_q = today.year, cur_q - 1
        prev_start_month = 3 * (prev_q - 1) + 1
        start_prev, end_prev = get_quarter_range(prev_y, prev_start_month)

    else:  # Last quarter
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

    default_city = shop_row.get("city") or "Amsterdam"
    weather_location = st.sidebar.text_input("Weather location", value=f"{default_city},NL")
    postcode4 = st.sidebar.text_input("CBS postcode (4 digits)", value=postcode[:4] if postcode else "")

    forecast_mode = st.sidebar.radio(
        "Forecast mode",
        ["Simple (DoW)", "Pro (LightGBM beta)"],
        index=0,
    )
    
        # ---------------------------
    # ✅ Persistent run-state + targets (NO duplicate widgets, NO blank page on rerun)
    # ---------------------------
    if "has_run" not in st.session_state:
        st.session_state.has_run = False
    
    if "targets" not in st.session_state:
        st.session_state.targets = {
            "turnover_target_month": 20000.0,
            "conversion_target_pct": 20.0,
            "spv_target": 3.0,
        }
    
    # --- Monthly targets (demo) — render ONCE in sidebar
    st.sidebar.markdown("### 🎯 Monthly targets (demo)")
    
    turnover_target_month = st.sidebar.number_input(
        "Turnover target (€)",
        min_value=0.0,
        value=float(st.session_state.targets["turnover_target_month"]),
        step=1000.0,
        key="turnover_target_month_input",
    )
    
    conversion_target_pct = st.sidebar.slider(
        "Conversion target (%)",
        min_value=5.0,
        max_value=60.0,
        value=float(st.session_state.targets["conversion_target_pct"]),
        step=0.5,
        key="conversion_target_pct_input",
    )
    
    spv_target = st.sidebar.number_input(
        "SPV target (€ per visitor)",
        min_value=0.0,
        value=float(st.session_state.targets["spv_target"]),
        step=0.10,
        key="spv_target_input",
    )
    
    # --- Buttons
    c_btn = st.sidebar.columns([1, 1])
    run_btn = c_btn[0].button("Analyse", type="primary", key="analyse_btn")
    reset_btn = c_btn[1].button("Reset", key="reset_btn")
    
    if reset_btn:
        st.session_state.has_run = False
    
    if run_btn:
        # ✅ Freeze targets at the moment of Analyse
        st.session_state.targets = {
            "turnover_target_month": float(turnover_target_month),
            "conversion_target_pct": float(conversion_target_pct),
            "spv_target": float(spv_target),
        }
        st.session_state.has_run = True
    
    # ✅ If user hasn't analysed yet, show message but DO NOT st.stop() later in the script
    if not st.session_state.has_run:
        st.info("Select retailer & store, set targets, pick a period and click **Analyse**.")
        return
    
    # ✅ Use frozen targets from here onward
    turnover_target_month = float(st.session_state.targets["turnover_target_month"])
    conversion_target_pct = float(st.session_state.targets["conversion_target_pct"])
    spv_target = float(st.session_state.targets["spv_target"])

    # ✅ Force refresh regions.csv cache when running analysis (prevents stale region_map)
    try:
        load_region_mapping.clear()
    except Exception:
        pass

    # ---------------------------
    # Meta build (for SVI)
    # ---------------------------
    region_map = load_region_mapping()
    locations_df = locations_df.copy()
    locations_df["id"] = pd.to_numeric(locations_df["id"], errors="coerce").astype("Int64")

    merged_meta = pd.DataFrame()
    if not region_map.empty:
        merged_meta = locations_df.merge(region_map, left_on="id", right_on="shop_id", how="left")
    else:
        merged_meta = locations_df.copy()
        merged_meta["region"] = np.nan
        merged_meta["sqm_override"] = np.nan
        merged_meta["store_label"] = np.nan

    if "sqm" in merged_meta.columns:
        merged_meta["sqm_effective"] = np.where(
            merged_meta.get("sqm_override", pd.Series([np.nan] * len(merged_meta))).notna(),
            merged_meta.get("sqm_override"),
            pd.to_numeric(merged_meta["sqm"], errors="coerce"),
        )
    else:
        merged_meta["sqm_effective"] = pd.to_numeric(merged_meta.get("sqm_override", np.nan), errors="coerce")

    if "store_label" in merged_meta.columns and merged_meta["store_label"].notna().any():
        merged_meta["store_display"] = np.where(
            merged_meta["store_label"].notna(),
            merged_meta["store_label"],
            merged_meta.get("name", merged_meta["id"].astype(str)),
        )
    else:
        merged_meta["store_display"] = merged_meta.get("name", merged_meta["id"].astype(str))
        
    # ---------------------------
    # Fetch store data (this year)  ✅ incl. transactions for conversion
    # ---------------------------
    with st.spinner("Fetching data via FastAPI..."):
        metric_keys_primary = ["count_in", "turnover", "transactions"]  # <- essentieel voor conversie
        try:
            resp_all = get_report(
                [shop_id],
                metric_keys_primary,
                period="this_year",
                step="day",
                source="shops",
            )
            df_all_raw = normalize_vemcount_response(resp_all, kpi_keys=metric_keys_primary).rename(
                columns={"count_in": "footfall", "turnover": "turnover", "transactions": "transactions"}
            )
        except Exception:
            # fallback als 'transactions' niet supported is voor deze klant
            metric_keys_fallback = ["count_in", "turnover"]
            resp_all = get_report(
                [shop_id],
                metric_keys_fallback,
                period="this_year",
                step="day",
                source="shops",
            )
            df_all_raw = normalize_vemcount_response(resp_all, kpi_keys=metric_keys_fallback).rename(
                columns={"count_in": "footfall", "turnover": "turnover"}
            )

    if df_all_raw.empty:
        st.warning("No data found for this year for this store.")
        return

    df_all_raw["date"] = pd.to_datetime(df_all_raw["date"], errors="coerce")
    df_all_raw = df_all_raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df_all_raw = compute_daily_kpis(df_all_raw)

    # detect store key col
    store_key_col_all = None
    for cand in ["id", "shop_id", "location_id"]:
        if cand in df_all_raw.columns:
            store_key_col_all = cand
            break
    if store_key_col_all is None:
        st.error("No store id column found in report response (id/shop_id/location_id).")
        return

    df_all_raw[store_key_col_all] = pd.to_numeric(df_all_raw[store_key_col_all], errors="coerce").astype("Int64")

    # Forecast history (date range)
    hist_end = datetime.today().date()
    hist_start = min(hist_start_input, hist_end)

    try:
        resp_hist = get_report(
            [shop_id],
            metric_keys_primary,
            period="date",
            step="day",
            source="shops",
            form_date_from=hist_start.strftime("%Y-%m-%d"),
            form_date_to=hist_end.strftime("%Y-%m-%d"),
        )
        df_hist_raw = normalize_vemcount_response(resp_hist, kpi_keys=metric_keys_primary).rename(
            columns={"count_in": "footfall", "turnover": "turnover", "transactions": "transactions"}
        )
    except Exception:
        # fallback als transactions niet supported is
        metric_keys_fallback = ["count_in", "turnover"]
        resp_hist = get_report(
            [shop_id],
            metric_keys_fallback,
            period="date",
            step="day",
            source="shops",
            form_date_from=hist_start.strftime("%Y-%m-%d"),
            form_date_to=hist_end.strftime("%Y-%m-%d"),
        )
        df_hist_raw = normalize_vemcount_response(resp_hist, kpi_keys=metric_keys_fallback).rename(
            columns={"count_in": "footfall", "turnover": "turnover"}
        )
        df_hist_raw["date"] = pd.to_datetime(df_hist_raw["date"], errors="coerce")
        df_hist_raw = df_hist_raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        df_hist_raw = compute_daily_kpis(df_hist_raw)
        if df_hist_raw.empty:
            df_hist_raw = df_all_raw.copy()
    except Exception:
        df_hist_raw = df_all_raw.copy()

    # ---------------------------
    # ✅ Like-for-like ranges for KPI row 1
    # ---------------------------
    start_cur_eff, end_cur_eff, start_prev_eff, end_prev_eff = like_for_like_ranges(
        start_cur=start_cur, end_cur=end_cur, start_prev=start_prev, end_prev=end_prev, today=today
    )

    start_cur_ts = pd.Timestamp(start_cur_eff)
    end_cur_ts = pd.Timestamp(end_cur_eff)
    start_prev_ts = pd.Timestamp(start_prev_eff)
    end_prev_ts = pd.Timestamp(end_prev_eff)

    df_cur = df_all_raw[(df_all_raw["date"] >= start_cur_ts) & (df_all_raw["date"] <= end_cur_ts)].copy()
    df_prev = df_all_raw[(df_all_raw["date"] >= start_prev_ts) & (df_all_raw["date"] <= end_prev_ts)].copy()

    # Weather
    weather_df = pd.DataFrame()
    if weather_location and VISUALCROSSING_KEY:
        weather_df = fetch_visualcrossing_history(weather_location, start_cur_eff, end_cur_eff)

    # Pathzz demo (weekly)
    pathzz_weekly = fetch_monthly_street_traffic(start_date=start_prev_eff, end_date=end_cur_eff)

    capture_weekly = pd.DataFrame()
    avg_capture_cur = None
    avg_capture_prev = None

    if not pathzz_weekly.empty:
        df_range = df_all_raw[(df_all_raw["date"] >= start_prev_ts) & (df_all_raw["date"] <= end_cur_ts)].copy()
        store_weekly = aggregate_weekly(df_range)

        pathzz_weekly["week_start"] = pd.to_datetime(pathzz_weekly["week_start"])
        street_weekly = pathzz_weekly.groupby("week_start", as_index=False)["street_footfall"].mean()

        capture_weekly = pd.merge(
            store_weekly[["week_start", "footfall", "turnover"]],
            street_weekly,
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
            capture_weekly["period"] = np.where(
                (capture_weekly["week_start"] >= start_cur_ts) & (capture_weekly["week_start"] <= end_cur_ts),
                "current",
                "previous",
            )

            avg_capture_cur = capture_weekly.loc[capture_weekly["period"] == "current", "capture_rate"].mean()
            avg_capture_prev = capture_weekly.loc[capture_weekly["period"] == "previous", "capture_rate"].mean()

    # CBS context
    cbs_stats = {}
    if postcode4:
        cbs_stats = get_cbs_stats_for_postcode4(postcode4)

    # ---------------------------
    # KPI row 1 values (like-for-like)
    # ---------------------------
    st.subheader(f"{selected_client['brand']} – {shop_row['name']}")

    foot_cur = df_cur["footfall"].sum() if "footfall" in df_cur.columns else 0
    foot_prev = df_prev["footfall"].sum() if ("footfall" in df_prev.columns and not df_prev.empty) else np.nan

    turn_cur = df_cur["turnover"].sum() if "turnover" in df_cur.columns else 0
    turn_prev = df_prev["turnover"].sum() if ("turnover" in df_prev.columns and not df_prev.empty) else np.nan

    spv_cur = df_cur["sales_per_visitor"].mean() if "sales_per_visitor" in df_cur.columns else np.nan
    spv_prev = df_prev["sales_per_visitor"].mean() if ("sales_per_visitor" in df_prev.columns and not df_prev.empty) else np.nan

    conv_cur = df_cur["conversion_rate"].mean() if "conversion_rate" in df_cur.columns else np.nan
    conv_prev = df_prev["conversion_rate"].mean() if ("conversion_rate" in df_prev.columns and not df_prev.empty) else np.nan

    foot_delta_pct = pct_delta(foot_cur, foot_prev)
    turn_delta_pct = pct_delta(turn_cur, turn_prev)
    spv_delta_pct = pct_delta(spv_cur, spv_prev)
    conv_delta_pct = pct_delta(conv_cur, conv_prev)

    # ---------------------------
    # ✅ SVI block (unchanged logic, just nicer rendering)
    # ---------------------------
    store_svi_score = None
    store_svi_rank = None
    store_svi_peer_n = None
    store_region = None
    store_svi_status = None
    store_svi_reason = None
    store_svi_error = None

    try:
        meta = merged_meta.copy()
        meta["id"] = pd.to_numeric(meta["id"], errors="coerce").astype("Int64")
        meta["sqm_effective"] = pd.to_numeric(meta.get("sqm_effective", np.nan), errors="coerce")

        row_meta = meta.loc[meta["id"] == pd.Series([shop_id], dtype="Int64").iloc[0]]
        if not row_meta.empty:
            r = row_meta.get("region", pd.Series([None])).iloc[0]
            store_region = str(r) if (pd.notna(r) and str(r).strip() != "") else None
    except Exception as e:
        meta = pd.DataFrame()
        store_svi_error = f"Meta build failed: {e}"

    if store_region and isinstance(meta, pd.DataFrame) and not meta.empty:
        try:
            peer_ids = (
                meta.loc[meta["region"].astype(str) == str(store_region), "id"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            if shop_id not in peer_ids:
                peer_ids.append(shop_id)

            resp_peers = get_report(
                peer_ids,
                ["count_in", "turnover"],
                period="date",
                step="day",
                source="shops",
                form_date_from=start_cur_eff.strftime("%Y-%m-%d"),
                form_date_to=end_cur_eff.strftime("%Y-%m-%d"),
            )

            df_peers = normalize_vemcount_response(
                resp_peers, kpi_keys=["count_in", "turnover"]
            ).rename(columns={"count_in": "footfall", "turnover": "turnover"})

            if df_peers is not None and not df_peers.empty:
                df_peers["date"] = pd.to_datetime(df_peers["date"], errors="coerce")
                df_peers = df_peers.dropna(subset=["date"])

                store_key_col = None
                for cand in ["id", "shop_id", "location_id"]:
                    if cand in df_peers.columns:
                        store_key_col = cand
                        break

                if store_key_col is None:
                    store_svi_error = "No store id column found in peers response (id/shop_id/location_id)."
                else:
                    df_peers[store_key_col] = pd.to_numeric(df_peers[store_key_col], errors="coerce").astype("Int64")
                    df_peers = df_peers.merge(
                        meta[["id", "store_display", "sqm_effective"]],
                        left_on=store_key_col,
                        right_on="id",
                        how="left",
                    )

                    df_peers = compute_daily_kpis(df_peers)

                    svi_df = build_store_vitality(
                        df_period=df_peers,
                        region_shops=meta,
                        store_key_col=store_key_col,
                    )

                    if svi_df is not None and not svi_df.empty:
                        svi_df["svi_score"] = pd.to_numeric(svi_df["svi_score"], errors="coerce")
                        svi_df = svi_df.dropna(subset=["svi_score"]).sort_values("svi_score", ascending=False).reset_index(drop=True)
                        svi_df["rank"] = np.arange(1, len(svi_df) + 1)

                        target_id = pd.Series([shop_id], dtype="Int64").iloc[0]
                        row = svi_df.loc[pd.to_numeric(svi_df[store_key_col], errors="coerce").astype("Int64") == target_id]

                        if not row.empty:
                            store_svi_score = float(np.clip(row["svi_score"].iloc[0], 0, 100))
                            store_svi_rank = int(row["rank"].iloc[0])
                            store_svi_peer_n = int(len(svi_df))
                            store_svi_status = row.get("svi_status", pd.Series([None])).iloc[0]
                            store_svi_reason = row.get("reason_short", pd.Series([None])).iloc[0]
                    else:
                        store_svi_error = "SVI DF empty (no vitality computed)."
            else:
                store_svi_error = "Peers DF empty (no peer data returned)."

        except Exception as e:
            store_svi_error = f"SVI build failed: {e}"
    #  ✅ SVI UI object first (must exist before escaping label)
    svi_ui = svi_style(store_svi_score)

    # ✅ sanitize dynamic values used inside unsafe HTML
    _region_safe = html.escape(str(store_region)) if store_region else ""
    _status_safe = html.escape(str(store_svi_status)) if store_svi_status else ""
    _label_safe = html.escape(str(svi_ui.get("label", "")))  # safe even if missing
    svi_badge = f"""
      <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px;">
        <div>
          <div class="kpi-title">Store Vitality (SVI)</div>
          <div style="display:flex; align-items:baseline; gap:8px;">
            <div style="font-size:28px; font-weight:900; color:{PFM_DARK}; line-height:1.05;">{f"{store_svi_score:.0f}" if store_svi_score is not None else "—"}</div>
            <div style="color:{PFM_GRAY}; font-weight:800;">/ 100</div>
          </div>
          <div class="kpi-sub">
            Rank: {f"{store_svi_rank} / {store_svi_peer_n}" if (store_svi_rank and store_svi_peer_n) else "—"}
            · benchmark vs regional peers
            {f" · Region: {_region_safe}" if store_region else ""}
            <div class="kpi-sub">
              Rank: {f"{store_svi_rank} / {store_svi_peer_n}" if (store_svi_rank and store_svi_peer_n) else "—"}
              · benchmark vs regional peers
              {f" · Region: {_region_safe}" if store_region else ""}
            </div>
            {_status_safe if store_svi_status else _label_safe}
          </div>
        </div>

        <div style="
          background: rgba(118,33,129,0.08);
          border: 1px solid rgba(118,33,129,0.18);
          color: {svi_ui['color']};
          padding: 6px 10px;
          border-radius: 999px;
          font-size: 12px;
          font-weight: 900;
          white-space: nowrap;">
          {svi_ui['emoji']} {html.escape(str(store_svi_status)) if store_svi_status else html.escape(str(svi_ui['label']))}
        </div>
      </div>
    """

    # ---------------------------
    # ✅ KPI Row 1 (now custom cards, delta always colored)
    # ---------------------------
    row1 = st.columns([1, 1, 1, 1, 1.2])

    with row1[0]:
        st.markdown(kpi_card_html(
            "Footfall (period)",
            fmt_int(foot_cur),
            sub="vs vorige periode (like-for-like)",
            delta_pct=foot_delta_pct
        ), unsafe_allow_html=True)

    with row1[1]:
        st.markdown(kpi_card_html(
            "Turnover (period)",
            fmt_eur(turn_cur),
            sub="vs vorige periode (like-for-like)",
            delta_pct=turn_delta_pct
        ), unsafe_allow_html=True)

    with row1[2]:
        spv_val = f"€ {spv_cur:.2f}".replace(".", ",") if pd.notna(spv_cur) else "-"
        st.markdown(kpi_card_html(
            "Avg spend / visitor",
            spv_val,
            sub="vs vorige periode (like-for-like)",
            delta_pct=spv_delta_pct
        ), unsafe_allow_html=True)

    with row1[3]:
        conv_val = fmt_pct(conv_cur) if pd.notna(conv_cur) else "-"
        st.markdown(kpi_card_html(
            "Avg conversion",
            conv_val,
            sub="vs vorige periode (like-for-like)",
            delta_pct=conv_delta_pct
        ), unsafe_allow_html=True)

    with row1[4]:
        st.markdown(f"""
          <div class="kpi-card" style="border-left: 6px solid {svi_ui['color']};">
            {svi_badge}
          </div>
        """, unsafe_allow_html=True)
        if store_svi_reason and store_svi_reason not in ("None", "nan"):
            reason_txt = str(store_svi_reason)
            # strip the most common html leakage
            reason_txt = reason_txt.replace("</div>", "").replace("<div>", "").strip()
            st.caption(reason_txt)
   
    # ---------------------------
    # ✅ Month outlook (forecast + actual) with YoY deltas + ring
    # ---------------------------
    st.markdown('<div class="section-title">Month outlook (forecast + actual)</div>', unsafe_allow_html=True)
    
    # current calendar month
    month_start, month_end = get_month_range(today.year, today.month)
    mtd_end = today
    
    # actual MTD from df_all_raw
    mtd_mask = (df_all_raw["date"] >= pd.Timestamp(month_start)) & (df_all_raw["date"] <= pd.Timestamp(mtd_end))
    df_mtd = df_all_raw.loc[mtd_mask].copy()
    
    turnover_mtd = float(df_mtd["turnover"].sum()) if not df_mtd.empty and "turnover" in df_mtd.columns else np.nan
    footfall_mtd = float(df_mtd["footfall"].sum()) if not df_mtd.empty and "footfall" in df_mtd.columns else np.nan
    
    # forecast remaining days of month via DOW means
    future_dates = pd.date_range(pd.Timestamp(mtd_end) + pd.Timedelta(days=1), pd.Timestamp(month_end), freq="D")
    fc_month = forecast_by_dow(df_hist_raw, future_dates)
    
    turnover_rem_fc = float(fc_month["turnover_forecast"].sum()) if not fc_month.empty else np.nan
    footfall_rem_fc = float(fc_month["footfall_forecast"].sum()) if not fc_month.empty else np.nan
    
    turnover_total_fc = (turnover_mtd + turnover_rem_fc) if (pd.notna(turnover_mtd) and pd.notna(turnover_rem_fc)) else np.nan
    
    # last year same month (MTD and full month) via API
    ly_turnover_mtd = np.nan
    ly_turnover_month = np.nan
    
    try:
        month_start_ly = (pd.Timestamp(month_start) - pd.DateOffset(years=1)).date()
        month_end_ly = (pd.Timestamp(month_end) - pd.DateOffset(years=1)).date()
    
        # same “day-of-month progress” for MTD comparison
        mtd_len = (mtd_end - month_start).days
        mtd_end_ly = month_start_ly + timedelta(days=max(mtd_len, 0))
    
        resp_ly = get_report(
            [shop_id],
            ["count_in", "turnover"],
            period="date",
            step="day",
            source="shops",
            form_date_from=month_start_ly.strftime("%Y-%m-%d"),
            form_date_to=month_end_ly.strftime("%Y-%m-%d"),
        )
        df_ly = normalize_vemcount_response(resp_ly, kpi_keys=["count_in", "turnover"]).rename(
            columns={"count_in": "footfall", "turnover": "turnover"}
        )
        if df_ly is not None and not df_ly.empty:
            df_ly["date"] = pd.to_datetime(df_ly["date"], errors="coerce")
            df_ly = df_ly.dropna(subset=["date"])
            df_ly["turnover"] = pd.to_numeric(df_ly["turnover"], errors="coerce")
    
            ly_turnover_month = float(df_ly["turnover"].sum())
    
            mtd_ly_mask = (df_ly["date"] >= pd.Timestamp(month_start_ly)) & (df_ly["date"] <= pd.Timestamp(mtd_end_ly))
            ly_turnover_mtd = float(df_ly.loc[mtd_ly_mask, "turnover"].sum())
    except Exception:
        pass
    
    # deltas for month cards
    mtd_yoy_delta = pct_delta(turnover_mtd, ly_turnover_mtd)
    total_yoy_delta = pct_delta(turnover_total_fc, ly_turnover_month)
    
    # ✅ 5 columns now (4 KPIs + ring)
    cols_m = st.columns([1, 1, 1, 1, 1.2])
    
    with cols_m[0]:
        st.markdown(kpi_card_html(
            "Turnover MTD (actual)",
            fmt_eur(turnover_mtd),
            sub="vs same month last year (MTD)",
            delta_pct=mtd_yoy_delta
        ), unsafe_allow_html=True)
    
    with cols_m[1]:
        st.markdown(kpi_card_html(
            "Turnover remaining (forecast)",
            fmt_eur(turnover_rem_fc),
            sub=f"Forecast covers {len(future_dates)} remaining days." if len(future_dates) > 0 else "No remaining days.",
            delta_pct=None
        ), unsafe_allow_html=True)
    
    with cols_m[2]:
        st.markdown(kpi_card_html(
            "Turnover total month (forecast)",
            fmt_eur(turnover_total_fc),
            sub="vs same month last year (full month)",
            delta_pct=total_yoy_delta
        ), unsafe_allow_html=True)
    
    with cols_m[3]:
        st.markdown(kpi_card_html(
            "Footfall remaining (forecast)",
            fmt_int(footfall_rem_fc),
            sub="Expected in-store traffic for the remainder of the month",
            delta_pct=None
        ), unsafe_allow_html=True)
    
    with cols_m[4]:
        # ✅ 3-ring "Activity" (Turnover / Conversion / SPV) — using frozen session targets
        def _pct(val, target):
            if target in (None, 0) or pd.isna(val):
                return 0.0
            return float(np.clip((float(val) / float(target)) * 100.0, 0, 120))
    
        # Use month total forecast for turnover progress
        turnover_prog = _pct(turnover_total_fc, turnover_target_month)
        conv_prog = _pct(conv_cur, conversion_target_pct)
        spv_prog = _pct(spv_cur, spv_target)
    
        # Rings (outer → inner) with consistent colors
        fig_ring = go.Figure()
    
        # Outer ring: Turnover (RED)
        fig_ring.add_trace(go.Pie(
            values=[turnover_prog, max(0, 100 - turnover_prog)],
            hole=0.58,
            marker=dict(colors=[PFM_RED, PFM_LINE]),
            textinfo="none",
            sort=False,
            direction="clockwise",
            showlegend=False,
        ))
    
        # Middle ring: Conversion (PURPLE)
        fig_ring.add_trace(go.Pie(
            values=[conv_prog, max(0, 100 - conv_prog)],
            hole=0.73,
            marker=dict(colors=[PFM_PURPLE, PFM_LINE]),
            textinfo="none",
            sort=False,
            direction="clockwise",
            showlegend=False,
        ))
    
        # Inner ring: SPV (PEACH)
        fig_ring.add_trace(go.Pie(
            values=[spv_prog, max(0, 100 - spv_prog)],
            hole=0.86,
            marker=dict(colors=[PFM_PEACH, PFM_LINE]),
            textinfo="none",
            sort=False,
            direction="clockwise",
            showlegend=False,
        ))
    
        fig_ring.update_layout(
            height=190,
            margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(
                text="Targets",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=PFM_DARK, family="sans-serif")
            )],
        )
    
        st.plotly_chart(fig_ring, use_container_width=True)
    
        # ✅ Clear legend that matches ring colors
        st.markdown(f"""
        <div style="display:flex; gap:10px; flex-wrap:wrap; font-size:12px; color:{PFM_GRAY}; font-weight:700; margin-top:4px;">
          <span><span style="display:inline-block;width:10px;height:10px;border-radius:99px;background:{PFM_RED};margin-right:6px;"></span>Turnover: {turnover_prog:.0f}%</span>
          <span><span style="display:inline-block;width:10px;height:10px;border-radius:99px;background:{PFM_PURPLE};margin-right:6px;"></span>Conv: {conv_prog:.0f}%</span>
          <span><span style="display:inline-block;width:10px;height:10px;border-radius:99px;background:{PFM_PEACH};margin-right:6px;"></span>SPV: {spv_prog:.0f}%</span>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------
    # ✅ Daily footfall & turnover (actual + forecast) — forecast bars light purple
    # ---------------------------
    st.markdown('<div class="section-title">Daily footfall & turnover (actual + forecast)</div>', unsafe_allow_html=True)

    # Build daily series for current month: actual up to today, forecast after today
    df_month_actual = df_all_raw[(df_all_raw["date"] >= pd.Timestamp(month_start)) & (df_all_raw["date"] <= pd.Timestamp(mtd_end))].copy()
    df_month_actual = df_month_actual[["date", "footfall", "turnover"]].copy() if not df_month_actual.empty else pd.DataFrame(columns=["date", "footfall", "turnover"])

    df_month_fc = pd.DataFrame()
    if not fc_month.empty:
        df_month_fc = fc_month.copy()
        df_month_fc = df_month_fc.rename(columns={"footfall_forecast": "footfall", "turnover_forecast": "turnover"})
        df_month_fc = df_month_fc[["date", "footfall", "turnover"]]

    fig_daily = make_subplots(specs=[[{"secondary_y": True}]])

    # Actual bars + line
    if not df_month_actual.empty:
        fig_daily.add_bar(
            x=df_month_actual["date"],
            y=df_month_actual["footfall"],
            name="Footfall (actual)",
            marker_color=PFM_PURPLE,
            opacity=1.0,
        )
        fig_daily.add_trace(
            go.Scatter(
                x=df_month_actual["date"],
                y=df_month_actual["turnover"],
                name="Turnover (actual)",
                mode="lines+markers",
                line=dict(color=PFM_PEACH, width=2),
            ),
            secondary_y=True,
        )

    # Forecast bars + line (light purple bars)
    if not df_month_fc.empty:
        fig_daily.add_bar(
            x=df_month_fc["date"],
            y=df_month_fc["footfall"],
            name="Footfall (forecast)",
            marker_color=PFM_PURPLE_LIGHT_RGBA,
            opacity=1.0,
        )
        fig_daily.add_trace(
            go.Scatter(
                x=df_month_fc["date"],
                y=df_month_fc["turnover"],
                name="Turnover (forecast)",
                mode="lines",
                line=dict(color=PFM_PURPLE_LIGHT_LINE, width=2, dash="dash"),
            ),
            secondary_y=True,
        )

    fig_daily.update_yaxes(title_text="Footfall", secondary_y=False)
    fig_daily.update_yaxes(title_text="Turnover (€)", secondary_y=True)
    fig_daily.update_layout(
        barmode="overlay",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=20, b=40),
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    # ---------------------------
    # Street traffic vs store traffic (weekly demo) — unchanged
    # ---------------------------
    st.markdown('<div class="section-title">Street traffic vs store traffic (weekly demo)</div>', unsafe_allow_html=True)

    if not capture_weekly.empty:
        chart_df = capture_weekly[["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]].copy()
        iso_cal = chart_df["week_start"].dt.isocalendar()
        chart_df["week_label"] = iso_cal.week.apply(lambda w: f"W{int(w):02d}")
        week_order = chart_df.sort_values("week_start")["week_label"].unique().tolist()

        fig_week = make_subplots(specs=[[{"secondary_y": True}]])
        fig_week.add_bar(x=chart_df["week_label"], y=chart_df["footfall"], name="Footfall (store)", marker_color=PFM_PURPLE, offsetgroup=0)
        fig_week.add_bar(x=chart_df["week_label"], y=chart_df["street_footfall"], name="Street traffic", marker_color=PFM_PEACH, opacity=0.7, offsetgroup=1)
        fig_week.add_bar(x=chart_df["week_label"], y=chart_df["turnover"], name="Turnover (€)", marker_color=PFM_PINK, opacity=0.7, offsetgroup=2)
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

        fig_week.update_xaxes(title_text="Week", categoryorder="array", categoryarray=week_order)
        fig_week.update_yaxes(title_text="Footfall / street / turnover", secondary_y=False)
        fig_week.update_yaxes(title_text="Capture rate (%)", secondary_y=True)
        fig_week.update_layout(
            barmode="group",
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_week, use_container_width=True)
    else:
        st.info("No Pathzz weekly demo data available for this period.")

    # ---------------------------
    # Weather vs footfall — unchanged
    # ---------------------------
    if not weather_df.empty:
        st.markdown('<div class="section-title">Weather vs footfall (indicative)</div>', unsafe_allow_html=True)

        weather_merge = pd.merge(
            df_cur[["date", "footfall"]],
            weather_df[["date", "temp", "precip"]],
            on="date",
            how="left",
        )

        fig_weather = make_subplots(specs=[[{"secondary_y": True}]])
        fig_weather.add_bar(x=weather_merge["date"], y=weather_merge["footfall"], name="Footfall", marker_color=PFM_PURPLE)

        fig_weather.add_trace(
            go.Scatter(x=weather_merge["date"], y=weather_merge["temp"], name="Temp (°C)", mode="lines+markers",
                       line=dict(color=PFM_RED, width=2)),
            secondary_y=True,
        )

        fig_weather.add_trace(
            go.Scatter(x=weather_merge["date"], y=weather_merge["precip"], name="Precip (mm)", mode="lines+markers",
                       line=dict(color=PFM_BLUE, width=2, dash="dot")),
            secondary_y=True,
        )

        fig_weather.update_yaxes(title_text="Footfall", secondary_y=False)
        fig_weather.update_yaxes(title_text="Temp / precip", secondary_y=True)
        fig_weather.update_layout(
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig_weather, use_container_width=True)

    # ---------------------------
    # Forecast (next 14 days) — keep as-is, but now you already have month forecast chart above
    # ---------------------------
    st.markdown('<div class="section-title">Forecast: footfall & turnover (next 14 days)</div>', unsafe_allow_html=True)

    weather_cfg = None
    if VISUALCROSSING_KEY and weather_location:
        parts = weather_location.split(",")
        city_part = parts[0].strip() if parts else weather_location
        country = "Netherlands"
        if len(parts) >= 2:
            cc = parts[1].strip().upper()
            country_map = {"NL": "Netherlands", "BE": "Belgium", "DE": "Germany", "FR": "France", "UK": "United Kingdom", "GB": "United Kingdom"}
            country = country_map.get(cc, cc)
        weather_cfg = {"mode": "city_country", "city": city_part, "country": country, "api_key": VISUALCROSSING_KEY}

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
            st.info("Not enough historical data for a reliable forecast.")
        else:
            hist_recent = fc_res["hist_recent"]
            fc = fc_res["forecast"]

            recent_foot = fc_res["recent_footfall_sum"]
            recent_turn = fc_res["recent_turnover_sum"]
            fut_foot = fc_res["forecast_footfall_sum"]
            fut_turn = fc_res["forecast_turnover_sum"]

            c1, c2 = st.columns(2)
            with c1:
                delta_foot = f"{(fut_foot - recent_foot) / recent_foot * 100:+.1f}%".replace(".", ",") if recent_foot > 0 else None
                st.markdown(kpi_card_html("Expected visitors (14d)", fmt_int(fut_foot), sub="vs last 14 days", delta_pct=(pct_delta(fut_foot, recent_foot))), unsafe_allow_html=True)

            with c2:
                st.markdown(kpi_card_html("Expected turnover (14d)", fmt_eur(fut_turn), sub="vs last 14 days", delta_pct=(pct_delta(fut_turn, recent_turn))), unsafe_allow_html=True)

            fig_fc = make_subplots(specs=[[{"secondary_y": True}]])
            fig_fc.add_bar(x=hist_recent["date"], y=hist_recent["footfall"], name="Footfall (hist)", marker_color=PFM_PURPLE)
            fig_fc.add_bar(x=fc["date"], y=fc["footfall_forecast"], name="Footfall (fc)", marker_color=PFM_PURPLE_LIGHT_RGBA)

            if "turnover" in hist_recent.columns:
                fig_fc.add_trace(
                    go.Scatter(x=hist_recent["date"], y=hist_recent["turnover"], name="Turnover (hist)", mode="lines",
                               line=dict(color=PFM_PINK, width=2)),
                    secondary_y=True,
                )

            fig_fc.add_trace(
                go.Scatter(x=fc["date"], y=fc["turnover_forecast"], name="Turnover (fc)", mode="lines+markers",
                           line=dict(color=PFM_RED, width=2, dash="dash")),
                secondary_y=True,
            )

            fig_fc.update_yaxes(title_text="Footfall", secondary_y=False)
            fig_fc.update_yaxes(title_text="Turnover (€)", secondary_y=True)
            fig_fc.update_layout(
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=40, r=40, t=20, b=40),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

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
        st.info("Forecast could not be computed (missing data / model / weather issue).")
        st.exception(e)

    # ---------------------------
    # Debug
    # ---------------------------
    with st.expander("🔧 Debug"):
        st.write("Period choice:", period_choice)
        st.write("Like-for-like current:", start_cur_eff, "→", end_cur_eff)
        st.write("Like-for-like previous:", start_prev_eff, "→", end_prev_eff)

        st.write("Month start/end:", month_start, "→", month_end, "MTD end:", mtd_end)
        st.write("Month forecast days remaining:", len(future_dates))

        st.write("Store region:", store_region)
        st.write("SVI score:", store_svi_score, "Rank:", store_svi_rank, "Peers:", store_svi_peer_n)
        st.write("SVI status:", store_svi_status)
        st.write("SVI reason:", store_svi_reason)
        st.write("SVI error:", store_svi_error)

        st.write("MTD turnover:", turnover_mtd, "LY MTD turnover:", ly_turnover_mtd, "Δ%:", mtd_yoy_delta)
        st.write("Month total forecast:", turnover_total_fc, "LY month turnover:", ly_turnover_month, "Δ%:", total_yoy_delta)

        st.write("df_cur head:", df_cur.head())
        st.write("df_prev head:", df_prev.head())
        st.write("fc_month head:", fc_month.head() if isinstance(fc_month, pd.DataFrame) else None)

if __name__ == "__main__":
    main()
