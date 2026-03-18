# pages/07A_Region_Reasoner.py
# ------------------------------------------------------------
# PFM Region Reasoner — Agentic Workload (NO CHARTS)
# - Same selectors + fetch pattern as Region Copilot v2
# - Apples-to-apples benchmarking by store_type (from data/regions.csv)
# - SVI:
#     * per store vs same store_type (company-wide)
#     * region vs company
#     * region vs other regions (leaderboard)
# - Fast region scan table + concise agentic narrative (OpenAI, optional)
# - No staffing assumptions; no generic product-category advice
# ------------------------------------------------------------

import os
import re
import json
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st

from datetime import datetime

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response
from helpers_periods import period_catalog
from helpers_vemcount_api import VemcountApiConfig, fetch_report, build_report_params

from services.svi_service import (
    SVI_DRIVERS,
    BASE_SVI_WEIGHTS,
    get_svi_weights_for_store_type,
    get_svi_weights_for_region_mix,
    compute_driver_values_from_period,
    compute_svi_explainable,
)

from stylesheet import inject_css

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="PFM Region Reasoner (Agentic)", layout="wide")

# ----------------------
# Brand colors
# ----------------------
PFM_PURPLE = "#762181"
PFM_RED = "#F04438"
PFM_DARK = "#111827"
PFM_GRAY = "#6B7280"
PFM_LIGHT = "#F3F4F6"
PFM_LINE = "#E5E7EB"
PFM_GREEN = "#22C55E"
PFM_AMBER = "#F59E0B"

inject_css(
    PFM_PURPLE=PFM_PURPLE,
    PFM_RED=PFM_RED,
    PFM_DARK=PFM_DARK,
    PFM_GRAY=PFM_GRAY,
    PFM_LIGHT=PFM_LIGHT,
    PFM_LINE=PFM_LINE,
)

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

# ----------------------
# Format helpers (EU)
# ----------------------
def fmt_eur(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"€ {x:,.0f}".replace(",", ".")

def fmt_eur_2(x: float) -> str:
    if pd.isna(x):
        return "-"
    s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€ {s}"

def fmt_int(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.0f}".replace(",", ".")

def fmt_pct(x: float, decimals: int = 0) -> str:
    if pd.isna(x):
        return "-"
    if decimals == 0:
        return f"{x:.0f}%".replace(".", ",")
    return f"{x:.1f}%".replace(".", ",")

def safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0.0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan

# ----------------------
# UI helpers
# ----------------------
def kpi_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def status_from_score(score: float):
    if score >= 75:
        return "High performance", PFM_GREEN
    if score >= 60:
        return "Good / stable", PFM_PURPLE
    if score >= 45:
        return "Attention required", PFM_AMBER
    return "Under pressure", PFM_RED

def _idx_style(v):
    """PFM-brand heatmap styling for index values (% vs benchmark)."""
    try:
        if pd.isna(v):
            return ""
        v = float(v)
        if v >= 115:
            return f"background-color:{PFM_PURPLE}1A; color:{PFM_DARK}; font-weight:900;"
        if v >= 105:
            return f"background-color:{PFM_PURPLE}0F; color:{PFM_DARK}; font-weight:800;"
        if v >= 95:
            return "background-color:#FFFFFF; color:#111827; font-weight:700;"
        if v >= 85:
            return f"background-color:{PFM_RED}12; color:{PFM_DARK}; font-weight:800;"
        return f"background-color:{PFM_RED}22; color:{PFM_DARK}; font-weight:900;"
    except Exception:
        return ""

# ----------------------
# Region mapping
# ----------------------
@st.cache_data(ttl=600)
def load_region_mapping(path: str = "data/regions.csv") -> pd.DataFrame:
    """
    Expected (sep=';'):
      shop_id;region;sqm_override;store_type
    """
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Required columns
    if "shop_id" not in df.columns or "region" not in df.columns:
        return pd.DataFrame()

    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")
    df["region"] = df["region"].astype(str).str.strip()

    if "sqm_override" in df.columns:
        df["sqm_override"] = pd.to_numeric(df["sqm_override"], errors="coerce")
    else:
        df["sqm_override"] = np.nan

    if "store_type" in df.columns:
        df["store_type"] = (
            df["store_type"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        )
    else:
        df["store_type"] = "Unknown"

    df = df.dropna(subset=["shop_id"])
    return df

@st.cache_data(ttl=600)
def get_locations_by_company(company_id: int) -> pd.DataFrame:
    url = f"{FASTAPI_BASE_URL.rstrip('/')}/company/{company_id}/location"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "locations" in data:
        return pd.DataFrame(data["locations"])
    return pd.DataFrame(data)

# ----------------------
# Optional Pathzz capture (safe, no-hard-fail)
# ----------------------
@st.cache_data(ttl=600)
def load_pathzz_weekly_store(csv_path: str, _mtime: float) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep=";", dtype=str, engine="python")
    except Exception:
        return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    # Normalize columns if needed
    df = df.rename(columns={"Region": "region", "Week": "week", "Visits": "visits"}).copy()

    for c in ["region", "week", "visits"]:
        if c not in df.columns:
            return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    if "shop_id" not in df.columns and df.shape[1] >= 4:
        df["shop_id"] = df.iloc[:, -1]

    if "shop_id" not in df.columns:
        return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    df["region"] = df["region"].astype(str).str.strip()
    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")

    df["visits"] = (
        df["visits"]
        .astype(str).str.strip()
        .replace("", np.nan)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["visits"] = pd.to_numeric(df["visits"], errors="coerce")

    def _parse_week_start(s: str):
        if isinstance(s, str) and "To" in s:
            return pd.to_datetime(s.split("To")[0].strip(), errors="coerce")
        return pd.NaT

    df["week_start"] = df["week"].apply(_parse_week_start)
    df = df.dropna(subset=["week_start", "shop_id", "visits"])
    return df[["region", "week", "week_start", "visits", "shop_id"]].reset_index(drop=True)

def filter_pathzz_for_period(df_pathzz: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    if df_pathzz is None or df_pathzz.empty:
        return pd.DataFrame(columns=["region", "week_start", "visits", "shop_id"])
    tmp = df_pathzz.copy()
    tmp["week_start"] = pd.to_datetime(tmp["week_start"], errors="coerce")
    tmp = tmp.dropna(subset=["week_start"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    out = tmp[(tmp["week_start"] >= start) & (tmp["week_start"] <= end)].copy()
    return out if not out.empty else pd.DataFrame(columns=["region", "week_start", "visits", "shop_id"])

# ----------------------
# Data helpers
# ----------------------
def _coerce_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def collapse_to_daily_store(df: pd.DataFrame, store_key_col: str) -> pd.DataFrame:
    """
    Ensures daily + store rows.
    Sums: footfall/turnover/transactions
    Means: other ratios (and recompute robustly)
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out.get("date", None), errors="coerce")
    out = out.dropna(subset=["date"])

    numeric_cols = [
        "footfall", "turnover", "transactions",
        "sales_per_visitor", "conversion_rate",
        "avg_basket_size", "sales_per_sqm", "sales_per_transaction",
    ]
    out = _coerce_numeric(out, [c for c in numeric_cols if c in out.columns])

    group_cols = [store_key_col, "date"]
    agg = {}
    for c in numeric_cols:
        if c not in out.columns:
            continue
        if c in ("footfall", "turnover", "transactions"):
            agg[c] = "sum"
        else:
            agg[c] = "mean"

    out = out.groupby(group_cols, as_index=False).agg(agg)

    # recompute derived metrics robustly
    if "turnover" in out.columns and "footfall" in out.columns:
        out["sales_per_visitor"] = np.where(out["footfall"] > 0, out["turnover"] / out["footfall"], np.nan)
    if "transactions" in out.columns and "footfall" in out.columns:
        out["conversion_rate"] = np.where(out["footfall"] > 0, out["transactions"] / out["footfall"] * 100.0, np.nan)
    if "turnover" in out.columns and "transactions" in out.columns:
        out["sales_per_transaction"] = np.where(out["transactions"] > 0, out["turnover"] / out["transactions"], np.nan)

    return out

def mark_closed_days_as_nan(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in ["footfall", "turnover", "transactions"]:
        if c not in out.columns:
            return out

    base = (
        pd.to_numeric(out["footfall"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(out["turnover"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(out["transactions"], errors="coerce").fillna(0).eq(0)
    )
    cols_to_nan = [
        "footfall", "turnover", "transactions",
        "conversion_rate", "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"
    ]
    for c in cols_to_nan:
        if c in out.columns:
            out.loc[base, c] = np.nan
    return out

def _ensure_store_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "store_type" not in out.columns:
        out["store_type"] = "Unknown"
    out["store_type"] = (
        out["store_type"]
        .fillna("Unknown")
        .astype(str).str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    )
    return out

def idx_vs(a, b):
    return (a / b * 100.0) if (pd.notna(a) and pd.notna(b) and float(b) != 0.0) else np.nan

def agg_period(df_: pd.DataFrame) -> dict:
    foot = float(pd.to_numeric(df_.get("footfall", 0), errors="coerce").dropna().sum())
    turn = float(pd.to_numeric(df_.get("turnover", 0), errors="coerce").dropna().sum())
    trans = float(pd.to_numeric(df_.get("transactions", 0), errors="coerce").dropna().sum())
    sqm = pd.to_numeric(df_.get("sqm_effective", np.nan), errors="coerce")
    sqm_sum = float(sqm.dropna().drop_duplicates().sum()) if sqm.notna().any() else np.nan
    return {"footfall": foot, "turnover": turn, "transactions": trans, "sqm_sum": sqm_sum}

def compute_company_store_type_benchmarks(df_daily_store: pd.DataFrame, store_dim: pd.DataFrame) -> pd.DataFrame:
    """
    Benchmarks per store_type (company-wide), using sums for totals and derived KPIs weighted.
    Requires: turnover, footfall, transactions, sqm_effective, store_type in joined dataset.
    """
    if df_daily_store is None or df_daily_store.empty:
        return pd.DataFrame()

    d = df_daily_store.copy()
    d = _ensure_store_type(d)

    for c in ["turnover", "footfall", "transactions", "sqm_effective"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Build store-level sqm per store_type (unique stores)
    sd = store_dim.copy()
    sd = _ensure_store_type(sd)
    sd["id"] = pd.to_numeric(sd.get("id", np.nan), errors="coerce").astype("Int64")
    sd["sqm_effective"] = pd.to_numeric(sd.get("sqm_effective", np.nan), errors="coerce")
    sqm_by_type = (
        sd.dropna(subset=["id"])
          .drop_duplicates(subset=["id"])
          .groupby("store_type", as_index=False)
          .agg(sqm_sum=("sqm_effective", "sum"), n_stores=("id", "nunique"))
    )

    g = d.groupby("store_type", as_index=False).agg(
        turnover=("turnover", "sum"),
        footfall=("footfall", "sum"),
        transactions=("transactions", "sum"),
    )

    g["conversion_rate"] = np.where(g["footfall"] > 0, g["transactions"] / g["footfall"] * 100.0, np.nan)
    g["sales_per_visitor"] = np.where(g["footfall"] > 0, g["turnover"] / g["footfall"], np.nan)
    g["sales_per_transaction"] = np.where(g["transactions"] > 0, g["turnover"] / g["transactions"], np.nan)

    g = g.merge(sqm_by_type, on="store_type", how="left")
    g["sales_per_sqm"] = np.where(g["sqm_sum"] > 0, g["turnover"] / g["sqm_sum"], np.nan)

    return g

def compute_region_store_type_benchmarks(df_daily_store: pd.DataFrame, store_dim: pd.DataFrame, region_choice: str) -> pd.DataFrame:
    if df_daily_store is None or df_daily_store.empty:
        return pd.DataFrame()
    d = df_daily_store[df_daily_store.get("region", "") == region_choice].copy()
    if d.empty:
        return pd.DataFrame()
    sd = store_dim[store_dim.get("region", "") == region_choice].copy()
    return compute_company_store_type_benchmarks(d, sd)

def estimate_upside(store_row: pd.Series, bench_vals: dict) -> tuple[float, str]:
    """
    Data-driven upside estimate (no product-category assumptions).
    Bench is expected to be same store_type (company), conservative.
    Drivers (choose best):
      - Low SPV (lift to benchmark, holding footfall)
      - Low Sales/m² (lift to benchmark, holding sqm)
      - Low Conversion (lift to benchmark, holding footfall, using store ATV or bench ATV)
    """
    foot = pd.to_numeric(store_row.get("footfall", np.nan), errors="coerce")
    turn = pd.to_numeric(store_row.get("turnover", np.nan), errors="coerce")
    trans = pd.to_numeric(store_row.get("transactions", np.nan), errors="coerce")
    sqm = pd.to_numeric(store_row.get("sqm_effective", np.nan), errors="coerce")

    spv_s = safe_div(turn, foot)
    spm2_s = safe_div(turn, sqm)
    cr_s = safe_div(trans, foot) * 100.0 if pd.notna(trans) and pd.notna(foot) else np.nan
    atv_s = safe_div(turn, trans)

    spv_b = bench_vals.get("sales_per_visitor", np.nan)
    spm2_b = bench_vals.get("sales_per_sqm", np.nan)
    cr_b = bench_vals.get("conversion_rate", np.nan)
    atv_b = bench_vals.get("sales_per_transaction", np.nan)

    atv_use = atv_s if pd.notna(atv_s) else atv_b
    candidates = []

    if pd.notna(foot) and foot > 0 and pd.notna(spv_s) and pd.notna(spv_b) and spv_s < spv_b:
        candidates.append(("Low SPV vs type", float(foot) * float(spv_b - spv_s)))

    if pd.notna(sqm) and sqm > 0 and pd.notna(spm2_s) and pd.notna(spm2_b) and spm2_s < spm2_b:
        candidates.append(("Low Sales/m² vs type", float(sqm) * float(spm2_b - spm2_s)))

    if pd.notna(foot) and foot > 0 and pd.notna(cr_s) and pd.notna(cr_b) and cr_s < cr_b and pd.notna(atv_use):
        extra_trans = float(foot) * (float(cr_b - cr_s) / 100.0)
        candidates.append(("Low Conversion vs type", max(0.0, extra_trans) * float(atv_use)))

    if not candidates:
        return np.nan, ""

    best = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
    upside = float(best[1]) if best[1] > 0 else np.nan
    return upside, best[0]

# ----------------------
# OpenAI (optional) — safe wrapper
# ----------------------
def try_llm_reasoning(prompt: str) -> str:
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if not api_key:
            return ""

        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        resp = client.responses.create(
            model=st.secrets.get("OPENAI_MODEL", "gpt-4.1"),
            input=prompt,
            temperature=0.3,
            max_output_tokens=500,
        )

        if not resp.output or not resp.output[0].content:
            return ""

        return resp.output[0].content[0].text.strip()

    except Exception as e:
        st.warning(f"AI narrative failed: {e}")
        return ""

# ----------------------
# MAIN
# ----------------------
def main():
    # ---- session defaults ----
    if "rr_last_key" not in st.session_state:
        st.session_state.rr_last_key = None
    if "rr_payload" not in st.session_state:
        st.session_state.rr_payload = None
    if "rr_ran" not in st.session_state:
        st.session_state.rr_ran = False

    # ----------------------
    # Load clients
    # ----------------------
    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)

    if clients_df.empty:
        st.error("No clients found in clients.json")
        return

    required_cols = {"brand", "name", "company_id"}
    if not required_cols.issubset(set(clients_df.columns)):
        st.error(f"clients.json missing columns. Required: {sorted(required_cols)}")
        return

    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    # ----------------------
    # Periods
    # ----------------------
    periods = period_catalog(today=datetime.now().date())
    if not isinstance(periods, dict) or len(periods) == 0:
        st.error("period_catalog() returned no periods.")
        return
    period_labels = list(periods.keys())

    # ======================
    # ROW 1 — Title + Client + Run button
    # ======================
    r1_left, r1_right = st.columns([3.6, 2.0], vertical_alignment="top")

    with r1_left:
        st.markdown(
            """
            <div class="pfm-header pfm-header--fixed">
              <div>
                <div class="pfm-title">PFM Region Reasoner <span class="pill">agentic</span></div>
                <div class="pfm-sub">Fast region scan + apples-to-apples store_type benchmarks + SVI (no charts)</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r1_right:
        st.markdown('<div class="pfm-header-controls">', unsafe_allow_html=True)

        c_sel, c_btn = st.columns([3.2, 1.2], vertical_alignment="center")
        with c_sel:
            client_label = st.selectbox(
                "Client",
                clients_df["label"].tolist(),
                label_visibility="collapsed",
                key="rr_client",
            )
        with c_btn:
            run_btn = st.button("Run analysis", type="primary", key="rr_run")

        st.markdown("</div>", unsafe_allow_html=True)

    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    # ----------------------
    # Load locations + region mapping
    # ----------------------
    try:
        locations_df = get_locations_by_company(company_id)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching stores from FastAPI: {e}")
        return

    if locations_df.empty:
        st.error("No stores found for this retailer.")
        return

    region_map = load_region_mapping()
    if region_map.empty:
        st.error("No valid data/regions.csv found (min required: shop_id;region).")
        return

    locations_df["id"] = pd.to_numeric(locations_df["id"], errors="coerce").astype("Int64")
    merged = locations_df.merge(region_map, left_on="id", right_on="shop_id", how="inner")

    if merged.empty:
        st.warning("No stores matched your regions.csv mapping for this retailer.")
        return
    # ---- NORMALISE REGION STRINGS (CRITICAL) ----
    merged["region"] = (
        merged["region"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    
    # Store display name
    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        merged["store_display"] = merged["name"] if "name" in merged.columns else merged["id"].astype(str)

    available_regions = sorted(merged["region"].dropna().unique().tolist())

    # ======================
    # ROW 2 — Selection + Region + Options
    # ======================
    c_sel, c_reg, c_opt = st.columns([1.2, 1.0, 2.0], vertical_alignment="bottom")

    with c_sel:
        st.markdown('<div class="panel"><div class="panel-title">Selection</div>', unsafe_allow_html=True)
        period_choice = st.selectbox(
            "Period",
            period_labels,
            index=period_labels.index("Q3 2024") if "Q3 2024" in period_labels else 0,
            label_visibility="collapsed",
            key="rr_period",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    start_period = periods[period_choice].start
    end_period = periods[period_choice].end

    with c_reg:
        st.markdown('<div class="panel"><div class="panel-title">Region</div>', unsafe_allow_html=True)
        region_choice = st.selectbox(
            "Region",
            available_regions,
            label_visibility="collapsed",
            key="rr_region",
        )
        region_choice = str(region_choice).strip().lower()
        st.markdown("</div>", unsafe_allow_html=True)

    with c_opt:
        st.markdown('<div class="panel"><div class="panel-title">Options</div>', unsafe_allow_html=True)
        o1, o2, o3 = st.columns(3)
        with o1:
            include_capture = st.toggle("Use capture if available", value=False, key="rr_capture")
        with o2:
            show_llm = st.toggle("Generate AI narrative", value=True, key="rr_llm")
        with o3:
            top_n = st.selectbox("Top N stores", [5, 8, 10, 15], index=1, key="rr_topn")
        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # Run/fetch logic (same pattern)
    # ----------------------
    lever_floor = 80
    lever_cap = 200 - lever_floor

    run_key = (company_id, region_choice, str(start_period), str(end_period), int(include_capture), int(top_n))
    selection_changed = st.session_state.rr_last_key != run_key
    should_fetch = bool(run_btn) or bool(selection_changed) or (not bool(st.session_state.rr_ran))

    if run_btn:
        st.toast("Running analysis…", icon="🧠")

    if (not should_fetch) and (st.session_state.rr_payload is None):
        st.info("Select retailer / region / period and click **Run analysis**.")
        return

    # ----------------------
    # FETCH
    # ----------------------
    if should_fetch:
        all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()
        if not all_shop_ids:
            st.error("No shop IDs available after mapping.")
            return

        metric_map = {
            "count_in": "footfall",
            "turnover": "turnover",
            "transactions": "transactions",
            "conversion_rate": "conversion_rate",
            "sales_per_visitor": "sales_per_visitor",
            "sales_per_sqm": "sales_per_sqm",
            "sales_per_transaction": "sales_per_transaction",
        }

        cfg = VemcountApiConfig(report_url=REPORT_URL)

        params_preview = build_report_params(
            shop_ids=all_shop_ids,
            data_outputs=list(metric_map.keys()),
            period="date",
            step="day",
            source="shops",
            date_from=start_period,
            date_to=end_period,
        )

        with st.status("Fetching data…", expanded=False) as status:
            status.write("Calling FastAPI /get-report…")
            try:
                resp = fetch_report(
                    cfg=cfg,
                    shop_ids=all_shop_ids,
                    data_outputs=list(metric_map.keys()),
                    period="date",
                    step="day",
                    source="shops",
                    date_from=start_period,
                    date_to=end_period,
                    timeout=120,
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ HTTPError from /get-report: {e}")
                try:
                    st.code(e.response.text)
                except Exception:
                    pass
                with st.expander("🔧 Debug request (params)"):
                    st.write("REPORT_URL:", REPORT_URL)
                    st.write("Params:", params_preview)
                return
            except requests.exceptions.RequestException as e:
                st.error(f"❌ RequestException from /get-report: {e}")
                with st.expander("🔧 Debug request (params)"):
                    st.write("REPORT_URL:", REPORT_URL)
                    st.write("Params:", params_preview)
                return

            status.update(label="Fetch complete ✅", state="complete")

        df_norm = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)
        if df_norm is None or df_norm.empty:
            st.warning("No data returned for the current selection.")
            return

        # Find store key column
        store_key_col = None
        for cand in ["shop_id", "id", "location_id"]:
            if cand in df_norm.columns:
                store_key_col = cand
                break
        if store_key_col is None:
            st.error("No store-id column found in response (shop_id/id/location_id).")
            return

        # Enrich sqm_effective
        merged2 = merged.copy()

        # Best effort: pick sqm from locations if present
        sqm_col = None
        for cand in ["sqm", "sq_meter", "sq_meters", "square_meters"]:
            if cand in merged2.columns:
                sqm_col = cand
                break

        base_sqm = pd.to_numeric(merged2[sqm_col], errors="coerce") if sqm_col is not None else np.nan

        merged2["sqm_effective"] = np.where(
            merged2["sqm_override"].notna(),
            pd.to_numeric(merged2["sqm_override"], errors="coerce"),
            base_sqm
        )

        # Daily store table
        df_daily_store = collapse_to_daily_store(df_norm, store_key_col=store_key_col)
        if df_daily_store is None or df_daily_store.empty:
            st.warning("No data after cleaning (daily/store collapse).")
            return

        # Join dimensions
        join_cols = ["id", "store_display", "region", "store_type", "sqm_effective"]
        for c in join_cols:
            if c not in merged2.columns:
                # ensure columns exist
                merged2[c] = np.nan
        df_daily_store = df_daily_store.merge(
            merged2[join_cols].drop_duplicates(),
            left_on=store_key_col,
            right_on="id",
            how="left",
        )

        # Compute sales_per_sqm robustly if missing
        if "sales_per_sqm" not in df_daily_store.columns:
            df_daily_store["sales_per_sqm"] = np.nan
        sqm_eff = pd.to_numeric(df_daily_store.get("sqm_effective", np.nan), errors="coerce")
        turn = pd.to_numeric(df_daily_store.get("turnover", np.nan), errors="coerce")
        calc_spm2 = np.where((pd.notna(sqm_eff) & (sqm_eff > 0)), (turn / sqm_eff), np.nan)
        df_daily_store["sales_per_sqm"] = pd.to_numeric(df_daily_store["sales_per_sqm"], errors="coerce")
        df_daily_store["sales_per_sqm"] = df_daily_store["sales_per_sqm"].combine_first(pd.Series(calc_spm2, index=df_daily_store.index))

        df_daily_store = mark_closed_days_as_nan(df_daily_store)

        # Region subset
        df_region_daily = df_daily_store[df_daily_store["region"] == region_choice].copy()

        # Optional capture (Pathzz)
        capture_store = {}
        avg_capture_region = np.nan

        if include_capture:
            pz_path = "data/pathzz_sample_weekly.csv"
            pz_mtime = os.path.getmtime(pz_path) if os.path.exists(pz_path) else 0.0
            pathzz_all = load_pathzz_weekly_store(pz_path, pz_mtime)
            pathzz_period = filter_pathzz_for_period(pathzz_all, start_period, end_period)

            if pathzz_period is not None and not pathzz_period.empty:
                # Match region naming (string compare)
                pzr = pathzz_period[
                    pathzz_period["region"].astype(str).str.strip().str.lower()
                    == str(region_choice).strip().lower()
                ].copy()

                # Build store-week footfall
                dd = df_region_daily.copy()
                dd["date"] = pd.to_datetime(dd["date"], errors="coerce")
                dd = dd.dropna(subset=["date"])
                dd["week_start"] = dd["date"].dt.to_period("W-SAT").dt.start_time

                for c in ["footfall", "turnover", "transactions"]:
                    dd[c] = pd.to_numeric(dd.get(c, np.nan), errors="coerce").fillna(0.0)

                store_week = dd.groupby(["id", "week_start"], as_index=False).agg(
                    footfall=("footfall", "sum"),
                    turnover=("turnover", "sum"),
                    transactions=("transactions", "sum"),
                )

                pzr = pzr.rename(columns={"shop_id": "id"}).copy()
                pzr["id"] = pd.to_numeric(pzr["id"], errors="coerce").astype("Int64")
                pzr["week_start"] = pd.to_datetime(pzr["week_start"], errors="coerce")
                pzr = pzr.dropna(subset=["id", "week_start", "visits"])

                pathzz_store_week = pzr.groupby(["id", "week_start"], as_index=False).agg(visits=("visits", "sum"))

                merged_sw = store_week.merge(pathzz_store_week, on=["id", "week_start"], how="inner")
                if not merged_sw.empty:
                    merged_sw["capture_rate"] = np.where(
                        merged_sw["visits"] > 0,
                        merged_sw["footfall"] / merged_sw["visits"] * 100.0,
                        np.nan,
                    )

                    # region avg capture
                    tot_vis = float(pd.to_numeric(merged_sw["visits"], errors="coerce").dropna().sum())
                    tot_ff = float(pd.to_numeric(merged_sw["footfall"], errors="coerce").dropna().sum())
                    avg_capture_region = (tot_ff / tot_vis * 100.0) if tot_vis > 0 else np.nan

                    # store capture across period
                    store_agg = merged_sw.groupby("id", as_index=False).agg(
                        footfall=("footfall", "sum"),
                        visits=("visits", "sum"),
                    )
                    store_agg["capture_rate"] = np.where(
                        store_agg["visits"] > 0,
                        store_agg["footfall"] / store_agg["visits"] * 100.0,
                        np.nan
                    )
                    for _, r in store_agg.iterrows():
                        try:
                            capture_store[int(r["id"])] = float(r["capture_rate"]) if pd.notna(r["capture_rate"]) else np.nan
                        except Exception:
                            pass

        # Store_dim
        store_dim = merged2[["id", "region", "store_type", "sqm_effective", "store_display"]].drop_duplicates().copy()

        # Store_type benchmarks
        company_type_bench = compute_company_store_type_benchmarks(df_daily_store, store_dim)
        region_type_bench = compute_region_store_type_benchmarks(df_daily_store, store_dim, region_choice)

        # Region SVI vs company
        reg_tot = agg_period(df_region_daily)
        comp_tot = agg_period(df_daily_store)

        reg_vals = compute_driver_values_from_period(
            footfall=reg_tot["footfall"],
            turnover=reg_tot["turnover"],
            transactions=reg_tot["transactions"],
            sqm_sum=reg_tot["sqm_sum"],
            capture_pct=avg_capture_region if include_capture else np.nan,
        )
        comp_vals = compute_driver_values_from_period(
            footfall=comp_tot["footfall"],
            turnover=comp_tot["turnover"],
            transactions=comp_tot["transactions"],
            sqm_sum=comp_tot["sqm_sum"],
            capture_pct=np.nan,  # company capture often unavailable; keep neutral
        )

        # Use region mix weights (store_type mix), but keep capture weight = 0 if no capture
        region_types = store_dim.loc[store_dim["region"] == region_choice, "store_type"]
        region_weights = get_svi_weights_for_region_mix(region_types)
        if not include_capture:
            region_weights = dict(region_weights)
            region_weights["capture_rate"] = 0.0

        region_svi, region_avg_ratio, region_bd = compute_svi_explainable(
            vals_a=reg_vals,
            vals_b=comp_vals,
            floor=float(lever_floor),
            cap=float(lever_cap),
            weights=region_weights,
        )

        # Region leaderboard (SVI vs company, capture excluded for fair compare)
        weights_no_cap = dict(BASE_SVI_WEIGHTS)
        weights_no_cap["capture_rate"] = 0.0

        rows = []
        d_all = df_daily_store.copy()
        d_all = d_all.dropna(subset=["region"])
        for rgn, g in d_all.groupby("region"):
            if pd.isna(rgn) or str(rgn).strip() == "":
                continue
            rg_tot = agg_period(g)
            rg_vals = compute_driver_values_from_period(
                footfall=rg_tot["footfall"],
                turnover=rg_tot["turnover"],
                transactions=rg_tot["transactions"],
                sqm_sum=rg_tot["sqm_sum"],
                capture_pct=np.nan,
            )
            svi_r, avg_r, _ = compute_svi_explainable(
                vals_a=rg_vals,
                vals_b=comp_vals,
                floor=float(lever_floor),
                cap=float(lever_cap),
                weights=weights_no_cap,
            )
            rows.append({
                "region": str(rgn),
                "svi": svi_r,
                "avg_ratio": avg_r,
                "turnover": rg_tot["turnover"],
                "footfall": rg_tot["footfall"],
            })

        df_region_rank = pd.DataFrame(rows)
        df_region_rank["svi"] = pd.to_numeric(df_region_rank["svi"], errors="coerce")
        df_region_rank = df_region_rank.dropna(subset=["svi"]).sort_values("svi", ascending=False).reset_index(drop=True)

        # Store-level summary for selected period (region only)
        reg_store = df_region_daily.copy()
        reg_store["id"] = pd.to_numeric(reg_store["id"], errors="coerce").astype("Int64")

        agg = reg_store.groupby(["id", "store_display", "store_type", "sqm_effective"], as_index=False).agg(
            turnover=("turnover", "sum"),
            footfall=("footfall", "sum"),
            transactions=("transactions", "sum"),
        )
        agg["conversion_rate"] = np.where(agg["footfall"] > 0, agg["transactions"] / agg["footfall"] * 100.0, np.nan)
        agg["sales_per_visitor"] = np.where(agg["footfall"] > 0, agg["turnover"] / agg["footfall"], np.nan)
        agg["sales_per_transaction"] = np.where(agg["transactions"] > 0, agg["turnover"] / agg["transactions"], np.nan)
        # 1. force numeric
        agg["sales_per_sqm"] = pd.to_numeric(
            agg.get("sales_per_sqm", np.nan),
            errors="coerce"
        )

        # 2. fallback calculation only if missing
        sqm = pd.to_numeric(agg.get("sqm_effective", np.nan), errors="coerce")
        turn = pd.to_numeric(agg.get("turnover", np.nan), errors="coerce")

        calc_spm2 = np.where(
            (pd.notna(sqm) & (sqm > 0)),
            turn / sqm,
            np.nan
        )

        agg["sales_per_sqm"] = agg["sales_per_sqm"].combine_first(
            pd.Series(calc_spm2, index=agg.index)
        )

        # attach capture if available
        if include_capture:
            agg["capture_rate"] = agg["id"].apply(lambda x: capture_store.get(int(x), np.nan) if pd.notna(x) else np.nan)
        else:
            agg["capture_rate"] = np.nan

        # per store: compute idx vs same store_type (company) and SVI vs same store_type (company)
        company_type_bench = _ensure_store_type(company_type_bench)
        region_type_bench = _ensure_store_type(region_type_bench)
        agg = _ensure_store_type(agg)

        # helper: type bench row
        def _bench_for_type(stype: str) -> dict:
            r = company_type_bench[company_type_bench["store_type"] == stype]
            if r is None or r.empty:
                return {}
            return r.iloc[0].to_dict()

        def _bench_for_type_region(stype: str) -> dict:
            r = region_type_bench[region_type_bench["store_type"] == stype]
            if r is None or r.empty:
                return {}
            return r.iloc[0].to_dict()

        svi_list = []
        idx_spv_type = []
        idx_cr_type = []
        idx_atv_type = []
        idx_spm2_type = []
        idx_traffic_type = []
        idx_cap_type = []

        idx_spv_regiontype = []
        idx_cr_regiontype = []
        idx_atv_regiontype = []
        idx_spm2_regiontype = []
        idx_traffic_regiontype = []
        idx_cap_regiontype = []

        upside_list = []
        upside_driver_list = []

        for _, r in agg.iterrows():
            stype = str(r.get("store_type", "Unknown")).strip() if pd.notna(r.get("store_type", None)) else "Unknown"
            stype = stype if stype else "Unknown"

            b = _bench_for_type(stype)
            br = _bench_for_type_region(stype)

            # idx vs company same type
            idx_spv_type.append(idx_vs(r.get("sales_per_visitor", np.nan), b.get("sales_per_visitor", np.nan)))
            idx_cr_type.append(idx_vs(r.get("conversion_rate", np.nan), b.get("conversion_rate", np.nan)))
            idx_atv_type.append(idx_vs(r.get("sales_per_transaction", np.nan), b.get("sales_per_transaction", np.nan)))
            idx_spm2_type.append(idx_vs(r.get("sales_per_sqm", np.nan), b.get("sales_per_sqm", np.nan)))
            idx_traffic_type.append(idx_vs(r.get("footfall", np.nan), b.get("footfall", np.nan)))
            idx_cap_type.append(idx_vs(r.get("capture_rate", np.nan), b.get("capture_rate", np.nan)) if include_capture else np.nan)

            # idx vs region same type (if available)
            idx_spv_regiontype.append(idx_vs(r.get("sales_per_visitor", np.nan), br.get("sales_per_visitor", np.nan)))
            idx_cr_regiontype.append(idx_vs(r.get("conversion_rate", np.nan), br.get("conversion_rate", np.nan)))
            idx_atv_regiontype.append(idx_vs(r.get("sales_per_transaction", np.nan), br.get("sales_per_transaction", np.nan)))
            idx_spm2_regiontype.append(idx_vs(r.get("sales_per_sqm", np.nan), br.get("sales_per_sqm", np.nan)))
            idx_traffic_regiontype.append(idx_vs(r.get("footfall", np.nan), br.get("footfall", np.nan)))
            idx_cap_regiontype.append(idx_vs(r.get("capture_rate", np.nan), br.get("capture_rate", np.nan)) if include_capture else np.nan)

            # store SVI vs type (company)
            store_vals = compute_driver_values_from_period(
                footfall=r.get("footfall", np.nan),
                turnover=r.get("turnover", np.nan),
                transactions=r.get("transactions", np.nan),
                sqm_sum=r.get("sqm_effective", np.nan),
                capture_pct=r.get("capture_rate", np.nan) if include_capture else np.nan,
            )

            # Bench values as driver-like dict
            bench_vals = compute_driver_values_from_period(
                footfall=b.get("footfall", np.nan),
                turnover=b.get("turnover", np.nan),
                transactions=b.get("transactions", np.nan),
                sqm_sum=b.get("sqm_sum", np.nan),
                capture_pct=b.get("capture_rate", np.nan) if include_capture else np.nan,
            )

            w_store = get_svi_weights_for_store_type(stype)
            if not include_capture:
                w_store = dict(w_store)
                w_store["capture_rate"] = 0.0

            store_svi, _, _ = compute_svi_explainable(
                vals_a=store_vals,
                vals_b=bench_vals,
                floor=float(lever_floor),
                cap=float(lever_cap),
                weights=w_store,
            )
            svi_list.append(store_svi)

            # Upside vs type bench (data-driven only)
            up, up_driver = estimate_upside(r, b)
            upside_list.append(up)
            upside_driver_list.append(up_driver)

        agg["SVI vs type"] = pd.to_numeric(pd.Series(svi_list), errors="coerce")
        agg["SPV idx vs type"] = pd.to_numeric(pd.Series(idx_spv_type), errors="coerce")
        agg["CR idx vs type"] = pd.to_numeric(pd.Series(idx_cr_type), errors="coerce")
        agg["ATV idx vs type"] = pd.to_numeric(pd.Series(idx_atv_type), errors="coerce")
        agg["Sales/m² idx vs type"] = pd.to_numeric(pd.Series(idx_spm2_type), errors="coerce")
        agg["Traffic idx vs type"] = pd.to_numeric(pd.Series(idx_traffic_type), errors="coerce")
        agg["Capture idx vs type"] = pd.to_numeric(pd.Series(idx_cap_type), errors="coerce")

        agg["SPV idx vs region type"] = pd.to_numeric(pd.Series(idx_spv_regiontype), errors="coerce")
        agg["CR idx vs region type"] = pd.to_numeric(pd.Series(idx_cr_regiontype), errors="coerce")
        agg["ATV idx vs region type"] = pd.to_numeric(pd.Series(idx_atv_regiontype), errors="coerce")
        agg["Sales/m² idx vs region type"] = pd.to_numeric(pd.Series(idx_spm2_regiontype), errors="coerce")
        agg["Traffic idx vs region type"] = pd.to_numeric(pd.Series(idx_traffic_regiontype), errors="coerce")
        agg["Capture idx vs region type"] = pd.to_numeric(pd.Series(idx_cap_regiontype), errors="coerce")

        agg["Upside (period)"] = pd.to_numeric(pd.Series(upside_list), errors="coerce")
        agg["Main driver"] = pd.Series(upside_driver_list).astype(str)

        # Store flags (fast scan)
        def _flag(row):
            svi = row.get("SVI vs type", np.nan)
            cr = row.get("CR idx vs type", np.nan)
            spv = row.get("SPV idx vs type", np.nan)
            trf = row.get("Traffic idx vs type", np.nan)
            up = row.get("Upside (period)", np.nan)

            if pd.notna(up) and up > 0 and pd.notna(cr) and cr < 90 and pd.notna(trf) and trf >= 100:
                return "🔥 Biggest upside"
            if pd.notna(svi) and svi < 55:
                return "⚠️ Under pressure"
            if pd.notna(svi) and svi < 70:
                return "⚠️ Needs attention"
            if pd.notna(svi) and svi >= 80 and pd.notna(cr) and cr >= 105 and pd.notna(spv) and spv >= 105:
                return "✅ Strong"
            return "—"

        agg["Flag"] = agg.apply(_flag, axis=1)

        # Cache payload
        st.session_state.rr_last_key = run_key
        st.session_state.rr_payload = {
            "df_daily_store": df_daily_store,
            "df_region_daily": df_region_daily,
            "store_dim": store_dim,
            "merged": merged2,
            "start_period": start_period,
            "end_period": end_period,
            "selected_client": selected_client,
            "region_choice": region_choice,
            "region_svi": region_svi,
            "region_avg_ratio": region_avg_ratio,
            "region_weights": region_weights,
            "reg_vals": reg_vals,
            "comp_vals": comp_vals,
            "df_region_rank": df_region_rank,
            "company_type_bench": company_type_bench,
            "region_type_bench": region_type_bench,
            "agg_store": agg,
            "include_capture": include_capture,
            "avg_capture_region": avg_capture_region,
            "params_preview": params_preview,
        }
        st.session_state.rr_ran = True

    # ----------------------
    # READ CACHE
    # ----------------------
    payload = st.session_state.rr_payload
    if payload is None:
        st.info("Select retailer / region / period and click **Run analysis**.")
        return

    df_region_daily = payload["df_region_daily"]
    selected_client = payload["selected_client"]
    region_choice = payload["region_choice"]
    start_period = payload["start_period"]
    end_period = payload["end_period"]

    region_svi = payload["region_svi"]
    region_avg_ratio = payload["region_avg_ratio"]
    df_region_rank = payload["df_region_rank"]
    company_type_bench = payload["company_type_bench"]
    region_type_bench = payload["region_type_bench"]
    agg = payload["agg_store"]
    include_capture = payload["include_capture"]
    avg_capture_region = payload["avg_capture_region"]

    # ----------------------
    # Header + Region summary KPIs
    # ----------------------
    st.markdown(f"## {selected_client['brand']} — Region **{region_choice}** · {start_period} → {end_period}")

    foot_total = float(pd.to_numeric(df_region_daily.get("footfall", np.nan), errors="coerce").dropna().sum())
    turn_total = float(pd.to_numeric(df_region_daily.get("turnover", np.nan), errors="coerce").dropna().sum())
    trans_total = float(pd.to_numeric(df_region_daily.get("transactions", np.nan), errors="coerce").dropna().sum())

    conv = (trans_total / foot_total * 100.0) if foot_total > 0 else np.nan
    spv = (turn_total / foot_total) if foot_total > 0 else np.nan
    atv = (turn_total / trans_total) if trans_total > 0 else np.nan

    st_cols = st.columns(6)
    with st_cols[0]:
        kpi_card("Footfall", fmt_int(foot_total), "Region · period total")
    with st_cols[1]:
        kpi_card("Revenue", fmt_eur(turn_total), "Region · period total")
    with st_cols[2]:
        kpi_card("Conversion", fmt_pct(conv, 1), "Transactions / Visitors")
    with st_cols[3]:
        kpi_card("SPV", fmt_eur_2(spv), "Revenue / Visitor")
    with st_cols[4]:
        kpi_card("ATV", fmt_eur_2(atv), "Revenue / Transaction")
    with st_cols[5]:
        cap_help = "Derived from Pathzz store-week match" if include_capture else "Capture disabled"
        kpi_card("Capture", fmt_pct(avg_capture_region, 1), cap_help)

    # Region SVI card + rank
    st.markdown('<div class="panel"><div class="panel-title">Region SVI — context</div>', unsafe_allow_html=True)

    big_score = 0 if pd.isna(region_svi) else float(region_svi)
    status_txt, status_color = status_from_score(big_score)

    # rank
    rank_txt = "-"
    if df_region_rank is not None and not df_region_rank.empty and "region" in df_region_rank.columns:
        rlist = df_region_rank["region"].tolist()
        if region_choice in rlist:
            rank = int(rlist.index(region_choice)) + 1
            rank_txt = f"#{rank} / {len(rlist)}"

    c1, c2, c3 = st.columns([1.2, 1.2, 2.6], vertical_alignment="top")
    with c1:
        st.markdown(
            f"""
            <div style="display:flex; align-items:baseline; gap:0.55rem;">
              <div style="font-size:3.2rem;font-weight:950;line-height:1;color:{status_color};letter-spacing:-0.02em;">
                {big_score:.0f}
              </div>
              <div class="pill">/ 100</div>
            </div>
            <div class="muted" style="margin-top:0.35rem;">
              Status: <span style="font-weight:900;color:{status_color}">{status_txt}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="muted" style="margin-top:0.6rem; line-height:1.5;">
              Rank vs other regions: <b>{rank_txt}</b><br/>
              Avg driver ratio vs company: <b>{"" if pd.isna(region_avg_ratio) else f"{region_avg_ratio:.0f}%"} </b><br/>
              Benchmarking: <span class="pill">store_type mix</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="muted" style="margin-top:0.25rem; line-height:1.5;">
              This page intentionally avoids charts: it’s built for <b>speed</b>.<br/>
              Everything below is apples-to-apples: stores are compared to <b>same store_type</b> (company-wide), so a mall store isn't judged by an out-of-town retail park.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # Fast Region Scan (agentic workload)
    # ----------------------
    st.markdown("## Fast region scan — where to focus first")

    # Prepare scan table
    scan = agg.copy()
    scan["SVI vs type"] = pd.to_numeric(scan["SVI vs type"], errors="coerce")
    scan["Upside (period)"] = pd.to_numeric(scan["Upside (period)"], errors="coerce")

    # Sorting: biggest upside first, then lowest SVI
    scan_sorted = scan.sort_values(
        by=["Upside (period)", "SVI vs type"],
        ascending=[False, True]
    ).copy()

    # Show Top N
    scan_show = scan_sorted.head(int(top_n)).copy()

    # Display
    disp = scan_show[[
        "Flag",
        "store_display",
        "store_type",
        "SVI vs type",
        "Traffic idx vs type",
        "CR idx vs type",
        "SPV idx vs type",
        "ATV idx vs type",
        "Sales/m² idx vs type",
        "Upside (period)",
        "Main driver",
    ]].rename(columns={
        "store_display": "Store",
        "store_type": "Store type",
        "SVI vs type": "SVI (vs type)",
        "Traffic idx vs type": "Traffic idx",
        "CR idx vs type": "CR idx",
        "SPV idx vs type": "SPV idx",
        "ATV idx vs type": "ATV idx",
        "Sales/m² idx vs type": "Sales/m² idx",
    })

    sty = disp.style
    heat_cols = ["SVI (vs type)", "Traffic idx", "CR idx", "SPV idx", "ATV idx", "Sales/m² idx"]
    for c in heat_cols:
        if c in disp.columns:
            sty = sty.applymap(_idx_style, subset=[c])

    sty = sty.format({
        "SVI (vs type)": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}",
        "Traffic idx": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%",
        "CR idx": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%",
        "SPV idx": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%",
        "ATV idx": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%",
        "Sales/m² idx": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%",
        "Upside (period)": lambda x: "-" if pd.isna(x) else fmt_eur(float(x)),
    })

    st.dataframe(sty, use_container_width=True, hide_index=True)

    # ----------------------
    # “Agentic” narrative (optional)
    # ----------------------
    st.markdown("## Agentic workload — executive summary")

    # derive quick facts for prompt
    # Top upside stores
    top_up = scan_sorted.dropna(subset=["Upside (period)"]).head(3).copy()
    top_risk = scan_sorted.sort_values("SVI vs type", ascending=True).head(3).copy()

    def _row_brief(df_: pd.DataFrame) -> list[dict]:
        out = []
        for _, r in df_.iterrows():
            out.append({
                "store": str(r.get("store_display", "")),
                "store_type": str(r.get("store_type", "")),
                "svi_vs_type": None if pd.isna(r.get("SVI vs type", np.nan)) else float(r.get("SVI vs type")),
                "traffic_idx": None if pd.isna(r.get("Traffic idx vs type", np.nan)) else float(r.get("Traffic idx vs type")),
                "cr_idx": None if pd.isna(r.get("CR idx vs type", np.nan)) else float(r.get("CR idx vs type")),
                "spv_idx": None if pd.isna(r.get("SPV idx vs type", np.nan)) else float(r.get("SPV idx vs type")),
                "atv_idx": None if pd.isna(r.get("ATV idx vs type", np.nan)) else float(r.get("ATV idx vs type")),
                "salesm2_idx": None if pd.isna(r.get("Sales/m² idx vs type", np.nan)) else float(r.get("Sales/m² idx vs type")),
                "upside_period": None if pd.isna(r.get("Upside (period)", np.nan)) else float(r.get("Upside (period)")),
                "driver": str(r.get("Main driver", "")),
                "flag": str(r.get("Flag", "")),
            })
        return out

    # Base text (always shown, even if LLM off)
    st.markdown(
        f"""
- **Region SVI:** **{big_score:.0f}/100** ({status_txt}; rank **{rank_txt}**)
- **Selection:** {start_period} → {end_period} · Store benchmarking = **same store_type (company-wide)**
- **Focus logic:** prioritize stores with **high traffic idx** + **low conversion idx** (quick wins) and stores with **very low SVI** (risk).
        """.strip()
    )

    if show_llm:
        prompt = {
            "client": selected_client.get("brand", ""),
            "region": region_choice,
            "period": f"{start_period} → {end_period}",
            "region_svi": None if pd.isna(region_svi) else float(region_svi),
            "region_rank": rank_txt,
            "totals": {
                "footfall": foot_total,
                "revenue": turn_total,
                "conversion_pct": None if pd.isna(conv) else float(conv),
                "spv": None if pd.isna(spv) else float(spv),
                "atv": None if pd.isna(atv) else float(atv),
                "capture_pct": None if pd.isna(avg_capture_region) else float(avg_capture_region),
            },
            "top_opportunities": _row_brief(top_up),
            "top_risks": _row_brief(top_risk),
            "constraints": [
                "Do not assume staffing data exists; do not recommend staffing actions unless explicitly stated as hypothesis.",
                "Do not give generic product-category upsell advice; keep actions tied to observed KPI gaps (traffic, conversion, SPV/ATV, sales per sqm, capture).",
                "Be concise, but concrete. Provide 3-5 action bullets with store references."
            ]
        }

        llm_text = try_llm_reasoning(
            "Create a short, data-driven region manager briefing.\n"
            "Return:\n"
            "1) 2–3 sentence summary\n"
            "2) A 'Do next' list (3–5 bullets) referencing store names\n"
            "3) A 'Questions to ask' list (3 bullets)\n\n"
            f"DATA:\n{json.dumps(prompt, indent=2)}"
        )

        if llm_text:
            st.markdown(llm_text)
        else:
            st.info("AI narrative unavailable (no OpenAI key/model, or request failed). The scan table above is complete and correct.")

    # ----------------------
    # Store-type benchmarks tables (region + company)
    # ----------------------
    st.markdown("## Store_type benchmarks (apples-to-apples)")

    cc1, cc2 = st.columns(2, vertical_alignment="top")

    with cc1:
        st.markdown('<div class="panel"><div class="panel-title">Company-wide benchmarks — per store_type</div>', unsafe_allow_html=True)
        if company_type_bench is None or company_type_bench.empty:
            st.info("No company store_type benchmarks available.")
        else:
            show = company_type_bench[[
                "store_type", "n_stores",
                "footfall", "turnover", "transactions",
                "conversion_rate", "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"
            ]].copy()

            show = show.rename(columns={
                "store_type": "Store type",
                "n_stores": "Stores",
                "sales_per_visitor": "SPV",
                "sales_per_transaction": "ATV",
                "sales_per_sqm": "Sales/m²",
                "turnover": "Revenue",
                "footfall": "Footfall",
                "transactions": "Transactions",
                "conversion_rate": "CR",
            })

            st.dataframe(
                show.style.format({
                    "Stores": lambda x: "-" if pd.isna(x) else f"{int(x)}",
                    "Footfall": lambda x: "-" if pd.isna(x) else fmt_int(float(x)),
                    "Revenue": lambda x: "-" if pd.isna(x) else fmt_eur(float(x)),
                    "Transactions": lambda x: "-" if pd.isna(x) else fmt_int(float(x)),
                    "CR": lambda x: "-" if pd.isna(x) else fmt_pct(float(x), 1),
                    "SPV": lambda x: "-" if pd.isna(x) else fmt_eur_2(float(x)),
                    "ATV": lambda x: "-" if pd.isna(x) else fmt_eur_2(float(x)),
                    "Sales/m²": lambda x: "-" if pd.isna(x) else fmt_eur_2(float(x)),
                }),
                use_container_width=True,
                hide_index=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with cc2:
        st.markdown('<div class="panel"><div class="panel-title">Region benchmarks — per store_type</div>', unsafe_allow_html=True)
        if region_type_bench is None or region_type_bench.empty:
            st.info("No region store_type benchmarks available (check region store mix).")
        else:
            show = region_type_bench[[
                "store_type", "n_stores",
                "footfall", "turnover", "transactions",
                "conversion_rate", "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"
            ]].copy()

            show = show.rename(columns={
                "store_type": "Store type",
                "n_stores": "Stores",
                "sales_per_visitor": "SPV",
                "sales_per_transaction": "ATV",
                "sales_per_sqm": "Sales/m²",
                "turnover": "Revenue",
                "footfall": "Footfall",
                "transactions": "Transactions",
                "conversion_rate": "CR",
            })

            st.dataframe(
                show.style.format({
                    "Stores": lambda x: "-" if pd.isna(x) else f"{int(x)}",
                    "Footfall": lambda x: "-" if pd.isna(x) else fmt_int(float(x)),
                    "Revenue": lambda x: "-" if pd.isna(x) else fmt_eur(float(x)),
                    "Transactions": lambda x: "-" if pd.isna(x) else fmt_int(float(x)),
                    "CR": lambda x: "-" if pd.isna(x) else fmt_pct(float(x), 1),
                    "SPV": lambda x: "-" if pd.isna(x) else fmt_eur_2(float(x)),
                    "ATV": lambda x: "-" if pd.isna(x) else fmt_eur_2(float(x)),
                    "Sales/m²": lambda x: "-" if pd.isna(x) else fmt_eur_2(float(x)),
                }),
                use_container_width=True,
                hide_index=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # Optional store drilldown (no charts)
    # ----------------------
    st.markdown("## Store drilldown (no charts)")

    region_stores = agg.copy()
    region_stores = region_stores.dropna(subset=["id"]).copy()
    if region_stores.empty:
        st.info("No stores available for drilldown.")
    else:
        region_stores["id_int"] = region_stores["id"].astype(int)
        region_stores["dd_label"] = region_stores["store_display"].astype(str) + " · " + region_stores["id_int"].astype(str)

        if "rr_store_choice" not in st.session_state:
            st.session_state.rr_store_choice = int(region_stores["id_int"].iloc[0])

        store_choice_label = st.selectbox(
            "Store",
            region_stores["dd_label"].tolist(),
            index=int(np.where(region_stores["id_int"].values == st.session_state.rr_store_choice)[0][0]) if (st.session_state.rr_store_choice in region_stores["id_int"].values) else 0,
            key="rr_store_select",
        )

        chosen_id = int(store_choice_label.split("·")[-1].strip())
        st.session_state.rr_store_choice = chosen_id

        row = region_stores[region_stores["id_int"] == chosen_id].iloc[0].to_dict()
        store_name = row.get("store_display", str(chosen_id))
        stype = row.get("store_type", "Unknown")

        st.markdown(
            f"### **{store_name}** · storeID {chosen_id} <span class='pill'>{stype}</span>",
            unsafe_allow_html=True
        )

        # KPIs
        s_foot = row.get("footfall", np.nan)
        s_turn = row.get("turnover", np.nan)
        s_trans = row.get("transactions", np.nan)
        s_cr = row.get("conversion_rate", np.nan)
        s_spv = row.get("sales_per_visitor", np.nan)
        s_atv = row.get("sales_per_transaction", np.nan)
        s_spm2 = row.get("sales_per_sqm", np.nan)
        s_svi = row.get("SVI vs type", np.nan)
        s_up = row.get("Upside (period)", np.nan)
        s_driver = row.get("Main driver", "")

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            kpi_card("Footfall", fmt_int(s_foot), "Store · period total")
        with k2:
            kpi_card("Revenue", fmt_eur(s_turn), "Store · period total")
        with k3:
            kpi_card("Conversion", fmt_pct(s_cr, 1), "Store")
        with k4:
            kpi_card("SPV", fmt_eur_2(s_spv), "Store")
        with k5:
            kpi_card("SVI (vs type)", "-" if pd.isna(s_svi) else f"{float(s_svi):.0f} / 100", "Store vs same store_type")

        st.markdown('<div class="panel"><div class="panel-title">Apples-to-apples indices</div>', unsafe_allow_html=True)

        idx_tbl = pd.DataFrame([{
            "Traffic idx vs type": row.get("Traffic idx vs type", np.nan),
            "CR idx vs type": row.get("CR idx vs type", np.nan),
            "SPV idx vs type": row.get("SPV idx vs type", np.nan),
            "ATV idx vs type": row.get("ATV idx vs type", np.nan),
            "Sales/m² idx vs type": row.get("Sales/m² idx vs type", np.nan),
            "Capture idx vs type": row.get("Capture idx vs type", np.nan) if include_capture else np.nan,
        }])

        sty2 = idx_tbl.style.applymap(_idx_style, subset=idx_tbl.columns.tolist()).format(
            {c: (lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%") for c in idx_tbl.columns}
        )
        st.dataframe(sty2, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Upside
        st.markdown(
            f"""
            <div class="callout">
              <div class="callout-title">Estimated upside (period): {fmt_eur(s_up) if pd.notna(s_up) else "-"}</div>
              <div class="callout-sub">Main driver: <b>{s_driver if s_driver else "-"}</b> — data-driven estimate vs same store_type benchmark.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Note: No staffing data is used. Any operational suggestions should be validated in-store.")

    # ----------------------
    # Debug
    # ----------------------
    with st.expander("🔧 Debug (Region Reasoner)"):
        st.write("REPORT_URL:", REPORT_URL)
        st.write("FASTAPI_BASE_URL:", FASTAPI_BASE_URL)
        st.write("Company:", company_id)
        st.write("Region:", region_choice)
        st.write("Period:", start_period, "→", end_period)
        st.write("Include capture:", include_capture)
        st.write("Params preview:", payload.get("params_preview", {}))
        st.write("regions.csv columns:", load_region_mapping().columns.tolist())
        st.write("agg_store cols:", payload["agg_store"].columns.tolist() if "agg_store" in payload else None)

# Streamlit multipage: call once
main()