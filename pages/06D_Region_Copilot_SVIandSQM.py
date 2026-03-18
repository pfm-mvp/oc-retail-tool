# pages/06C_Regio_Copilot_SVI.py
# ------------------------------------------------------------
# PFM Region Copilot v2 — FINAL FIXED VERSION
# - Fixes NameError: clients_df / period_labels
# - Fixes Header Row 1 alignment: right card stacked (Client above Run)
# - Keeps FULL app logic (fetch → cache → render)
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import requests
import os
import re
import streamlit as st
import altair as alt
import streamlit.components.v1 as components

from services.svi_service import (
    SVI_DRIVERS,
    BASE_SVI_WEIGHTS,
    SIZE_CAL_KEYS,
    sqm_calibration_factor,
    get_svi_weights_for_store_type,
    get_svi_weights_for_region_mix,
    compute_driver_values_from_period,
    compute_svi_explainable,
    ratio_to_score_0_100,
)


from datetime import datetime

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response
from helpers_periods import period_catalog
from helpers_vemcount_api import VemcountApiConfig, fetch_report, build_report_params

from services.cbs_service import (
    get_cci_series,
    get_retail_index,
)

from stylesheet import inject_css

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="PFM Region Manager Tool v2",
    layout="wide"
)

# ----------------------
# PFM brand-ish colors
# ----------------------
PFM_PURPLE = "#762181"
PFM_RED = "#F04438"
PFM_DARK = "#111827"
PFM_GRAY = "#6B7280"
PFM_LIGHT = "#F3F4F6"
PFM_LINE = "#E5E7EB"
PFM_GREEN = "#22C55E"
PFM_AMBER = "#F59E0B"

OTHER_REGION_PURPLE = "#C4B5FD"
BLACK = "#111111"

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
# Format helpers
# ----------------------
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

def fmt_eur_2(x: float) -> str:
    if pd.isna(x):
        return "-"
    s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€ {s}"

def safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0.0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan

def norm_key(x: str) -> str:
    return str(x).strip().lower() if x is not None else ""

# ----------------------
# Region & API helpers
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

    if "store_type" in df.columns:
        df["store_type"] = df["store_type"].astype(str)
    else:
        df["store_type"] = np.nan

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
# Pathzz store-level capture (FIXED)
# ----------------------
@st.cache_data(ttl=600)
def load_pathzz_weekly_store(csv_path: str, _mtime: float) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep=";", dtype=str, engine="python")
    except Exception:
        return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    df = df.rename(columns={"Region": "region", "Week": "week", "Visits": "visits"}).copy()

    for c in ["region", "week", "visits"]:
        if c not in df.columns:
            return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    if "shop_id" not in df.columns and df.shape[1] >= 4:
        df["shop_id"] = df.iloc[:, -1]

    if "shop_id" not in df.columns:
        return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    df["region"] = df["region"].astype(str).str.strip()
    df["visits"] = df["visits"].astype(str).str.strip().replace("", np.nan)
    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["visits", "shop_id"])
    if df.empty:
        return pd.DataFrame(columns=["region", "week", "week_start", "visits", "shop_id"])

    df["visits"] = (
        df["visits"]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    def _parse_week_start(s: str):
        if isinstance(s, str) and "To" in s:
            return pd.to_datetime(s.split("To")[0].strip(), errors="coerce")
        return pd.NaT

    df["week_start"] = df["week"].apply(_parse_week_start)
    df = df.dropna(subset=["week_start"])

    return df[["region", "week", "week_start", "visits", "shop_id"]].reset_index(drop=True)

def filter_pathzz_for_period(df_pathzz: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    expected_cols = ["region", "week_start", "visits", "shop_id"]
    if df_pathzz is None or df_pathzz.empty:
        return pd.DataFrame(columns=expected_cols)

    for c in expected_cols:
        if c not in df_pathzz.columns:
            return pd.DataFrame(columns=expected_cols)

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    tmp = df_pathzz.copy()
    tmp["week_start"] = pd.to_datetime(tmp["week_start"], errors="coerce")
    tmp = tmp.dropna(subset=["week_start"])

    out = tmp[(tmp["week_start"] >= start) & (tmp["week_start"] <= end)].copy()
    return out if not out.empty else pd.DataFrame(columns=expected_cols)

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

def gauge_chart(score_0_100: float, fill_color: str):
    score_0_100 = float(np.clip(score_0_100, 0, 100))
    gauge_df = pd.DataFrame(
        {"segment": ["filled", "empty"], "value": [score_0_100, max(0.0, 100.0 - score_0_100)]}
    )
    arc = (
        alt.Chart(gauge_df)
        .mark_arc(innerRadius=50, outerRadius=64)
        .encode(
            theta="value:Q",
            color=alt.Color(
                "segment:N",
                scale=alt.Scale(domain=["filled", "empty"], range=[fill_color, PFM_LINE]),
                legend=None,
            ),
        )
        .properties(width=180, height=180)
    )
    text = (
        alt.Chart(pd.DataFrame({"label": [f"{score_0_100:.0f}"]}))
        .mark_text(size=28, fontWeight="bold", color=PFM_DARK)
        .encode(text="label:N")
        .properties(width=180, height=180)
    )
    return arc + text

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
    if df is None or df.empty:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
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

def enrich_merged_with_sqm_from_df_norm(merged: pd.DataFrame, df_norm: pd.DataFrame, store_key_col: str) -> pd.DataFrame:
    if merged is None or merged.empty or df_norm is None or df_norm.empty:
        return merged

    sqm_col_norm = None
    for cand in ["sq_meter", "sqm", "sq_meters", "square_meters"]:
        if cand in df_norm.columns:
            sqm_col_norm = cand
            break
    if sqm_col_norm is None:
        return merged

    tmp = df_norm[[store_key_col, sqm_col_norm]].copy()
    tmp[store_key_col] = pd.to_numeric(tmp[store_key_col], errors="coerce")
    tmp[sqm_col_norm] = pd.to_numeric(tmp[sqm_col_norm], errors="coerce")
    tmp = tmp.dropna(subset=[store_key_col, sqm_col_norm])
    if tmp.empty:
        return merged

    sqm_by_shop = (
        tmp.groupby(store_key_col, as_index=False)[sqm_col_norm]
        .first()
        .rename(columns={sqm_col_norm: "sqm_api"})
    )

    out = merged.copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out = out.merge(sqm_by_shop, left_on="id", right_on=store_key_col, how="left")
    if store_key_col in out.columns:
        out = out.drop(columns=[store_key_col])
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
    cols_to_nan = ["footfall", "turnover", "transactions", "conversion_rate", "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"]
    for c in cols_to_nan:
        if c in out.columns:
            out.loc[base, c] = np.nan
    return out

def _ensure_store_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "store_type" not in out.columns:
        out["store_type"] = "Unknown"
    out["store_type"] = out["store_type"].fillna("Unknown").astype(str).str.strip()
    out.loc[out["store_type"].isin(["", "nan", "None"]), "store_type"] = "Unknown"
    return out

def agg_store_type_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted aggregation per store_type.
    Assumes daily rows with: turnover, footfall, transactions, sqm_effective (store-level field).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = _ensure_store_type(df)

    # numeric coercion
    for c in ["turnover", "footfall", "transactions", "sqm_effective", "avg_basket_size"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # base sums
    g = d.groupby("store_type", as_index=False).agg(
        turnover=("turnover", "sum"),
        footfall=("footfall", "sum"),
        transactions=("transactions", "sum"),
        # sqm_effective is store-level; in df_daily_store it's repeated per day.
        # We'll compute sqm_sum via unique stores outside if needed; here we approximate if id exists.
    )

    # derived metrics (weighted)
    g["conversion_rate"] = np.where(g["footfall"] > 0, g["transactions"] / g["footfall"] * 100.0, np.nan)
    g["sales_per_visitor"] = np.where(g["footfall"] > 0, g["turnover"] / g["footfall"], np.nan)
    g["sales_per_transaction"] = np.where(g["transactions"] > 0, g["turnover"] / g["transactions"], np.nan)

    return g

def add_sqm_sum_per_store_type(df_daily_store: pd.DataFrame, store_dim: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sqm_sum per store_type using unique stores from merged/store_dim.
    store_dim must include: id, store_type, sqm_effective
    """
    if store_dim is None or store_dim.empty:
        return pd.DataFrame()

    sd = store_dim.copy()
    sd = _ensure_store_type(sd)
    sd["sqm_effective"] = pd.to_numeric(sd.get("sqm_effective", np.nan), errors="coerce")

    sqm_by_type = (
        sd.dropna(subset=["id"])
          .drop_duplicates(subset=["id"])
          .groupby("store_type", as_index=False)
          .agg(sqm_sum=("sqm_effective", "sum"), n_stores=("id", "nunique"))
    )
    return sqm_by_type

def compute_store_type_benchmarks(
    df_daily_store: pd.DataFrame,
    region_choice: str,
    store_dim: pd.DataFrame,
    capture_store_week: pd.DataFrame | None = None
):
    """
    Returns:
      region_type_kpis (per store_type),
      company_type_kpis (per store_type),
      region_vs_company_index (per store_type with idx columns),
      region_mix (store_type counts / sqm share)
    """
    if df_daily_store is None or df_daily_store.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    d = _ensure_store_type(df_daily_store)

    region_df = d[d["region"] == region_choice].copy()
    comp_df = d.copy()

    reg_k = agg_store_type_kpis(region_df)
    com_k = agg_store_type_kpis(comp_df)

    # sqm_sum + n_stores from store_dim
    sqm_type = add_sqm_sum_per_store_type(df_daily_store, store_dim)
    if not sqm_type.empty:
        reg_store_dim = _ensure_store_type(store_dim[store_dim["region"] == region_choice].copy())
        sqm_type_reg = add_sqm_sum_per_store_type(df_daily_store, reg_store_dim)
    else:
        sqm_type_reg = pd.DataFrame()

    if not sqm_type.empty:
        com_k = com_k.merge(sqm_type[["store_type","sqm_sum","n_stores"]], on="store_type", how="left")
        com_k["sales_per_sqm"] = np.where(com_k["sqm_sum"] > 0, com_k["turnover"] / com_k["sqm_sum"], np.nan)

    if not sqm_type_reg.empty:
        reg_k = reg_k.merge(sqm_type_reg[["store_type","sqm_sum","n_stores"]], on="store_type", how="left")
        reg_k["sales_per_sqm"] = np.where(reg_k["sqm_sum"] > 0, reg_k["turnover"] / reg_k["sqm_sum"], np.nan)

    # Capture per store_type (optional)
    def _capture_by_type(csw: pd.DataFrame, store_dim: pd.DataFrame) -> pd.DataFrame:
        if csw is None or csw.empty:
            return pd.DataFrame(columns=["store_type","capture_rate"])
        tmp = csw.copy()
        tmp["id"] = pd.to_numeric(tmp.get("id", np.nan), errors="coerce").astype("Int64")
        sd = store_dim[["id","store_type"]].copy()
        sd = _ensure_store_type(sd)
        sd["id"] = pd.to_numeric(sd["id"], errors="coerce").astype("Int64")

        tmp = tmp.merge(sd.drop_duplicates("id"), on="id", how="left")
        tmp = _ensure_store_type(tmp)

        tmp["footfall"] = pd.to_numeric(tmp.get("footfall", np.nan), errors="coerce")
        tmp["visits"] = pd.to_numeric(tmp.get("visits", np.nan), errors="coerce")

        g = tmp.groupby("store_type", as_index=False).agg(
            footfall=("footfall","sum"),
            visits=("visits","sum")
        )
        g["capture_rate"] = np.where(g["visits"] > 0, g["footfall"]/g["visits"]*100.0, np.nan)
        return g[["store_type","capture_rate"]]

    if capture_store_week is not None:
        cap_reg = _capture_by_type(
            capture_store_week.merge(store_dim[["id", "region"]], on="id", how="left"),
            store_dim[store_dim["region"] == region_choice].copy()
        ) if "region" in store_dim.columns else pd.DataFrame()
    
        cap_com = _capture_by_type(capture_store_week, store_dim)

        if not cap_reg.empty:
            reg_k = reg_k.merge(cap_reg, on="store_type", how="left")
        if not cap_com.empty:
            com_k = com_k.merge(cap_com, on="store_type", how="left")

    # Build indices: region/store_type vs company/store_type
    out = reg_k.merge(
        com_k[["store_type","conversion_rate","sales_per_visitor","sales_per_transaction","sales_per_sqm","capture_rate"]]
        .rename(columns={
            "conversion_rate":"conv_b",
            "sales_per_visitor":"spv_b",
            "sales_per_transaction":"atv_b",
            "sales_per_sqm":"spm2_b",
            "capture_rate":"cap_b"
        }),
        on="store_type",
        how="left"
    )

    def _idx(a, b):
        return np.where((pd.notna(a) & pd.notna(b) & (b != 0)), (a/b*100.0), np.nan)

    out["SPV idx"]   = _idx(out.get("sales_per_visitor", np.nan), out.get("spv_b", np.nan))
    out["CR idx"]    = _idx(out.get("conversion_rate", np.nan), out.get("conv_b", np.nan))
    out["ATV idx"]   = _idx(out.get("sales_per_transaction", np.nan), out.get("atv_b", np.nan))
    out["Sales/m² idx"] = _idx(out.get("sales_per_sqm", np.nan), out.get("spm2_b", np.nan))
    out["Capture idx"]  = _idx(out.get("capture_rate", np.nan), out.get("cap_b", np.nan))

    # Region mix (counts)
    mix = store_dim[store_dim["region"] == region_choice].copy()
    mix = _ensure_store_type(mix)
    mix["id"] = pd.to_numeric(mix.get("id", np.nan), errors="coerce").astype("Int64")
    mix["sqm_effective"] = pd.to_numeric(mix.get("sqm_effective", np.nan), errors="coerce")

    mix_g = mix.dropna(subset=["id"]).drop_duplicates("id").groupby("store_type", as_index=False).agg(
        Stores=("id","nunique"),
        sqm=("sqm_effective","sum"),
    )
    mix_g["Store share"] = np.where(mix_g["Stores"].sum() > 0, mix_g["Stores"]/mix_g["Stores"].sum()*100.0, np.nan)

    return reg_k, com_k, out, mix_g

# ----------------------
# Lever scan score mapping
# ----------------------
def compute_svi_by_region_companywide(
    df_daily_store: pd.DataFrame,
    lever_floor: float,
    lever_cap: float,
) -> pd.DataFrame:
    if df_daily_store is None or df_daily_store.empty:
        return pd.DataFrame(columns=["region", "svi", "avg_ratio", "footfall", "turnover"])

    d = df_daily_store.copy()
    if "region" not in d.columns:
        return pd.DataFrame(columns=["region", "svi", "avg_ratio", "footfall", "turnover"])

    for c in ["footfall", "turnover", "transactions", "sqm_effective"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    comp_tot = {
        "footfall": float(d["footfall"].dropna().sum()) if "footfall" in d.columns else np.nan,
        "turnover": float(d["turnover"].dropna().sum()) if "turnover" in d.columns else np.nan,
        "transactions": float(d["transactions"].dropna().sum()) if "transactions" in d.columns else np.nan,
        "sqm_sum": float(d["sqm_effective"].dropna().drop_duplicates().sum()) if "sqm_effective" in d.columns else np.nan,
    }
    comp_vals = compute_driver_values_from_period(
        footfall=comp_tot["footfall"],
        turnover=comp_tot["turnover"],
        transactions=comp_tot["transactions"],
        sqm_sum=comp_tot["sqm_sum"],
        capture_pct=np.nan,
    )

    weights_no_capture = dict(BASE_SVI_WEIGHTS)
    weights_no_capture["capture_rate"] = 0.0

    rows = []
    for region, g in d.groupby("region"):
        if pd.isna(region) or str(region).strip() == "":
            continue

        reg_tot = {
            "footfall": float(g["footfall"].dropna().sum()) if "footfall" in g.columns else np.nan,
            "turnover": float(g["turnover"].dropna().sum()) if "turnover" in g.columns else np.nan,
            "transactions": float(g["transactions"].dropna().sum()) if "transactions" in g.columns else np.nan,
            "sqm_sum": float(g["sqm_effective"].dropna().drop_duplicates().sum()) if "sqm_effective" in g.columns else np.nan,
        }
        reg_vals = compute_driver_values_from_period(
            footfall=reg_tot["footfall"],
            turnover=reg_tot["turnover"],
            transactions=reg_tot["transactions"],
            sqm_sum=reg_tot["sqm_sum"],
            capture_pct=np.nan,
        )

        svi, avg_ratio, _ = compute_svi_explainable(
            vals_a=reg_vals,
            vals_b=comp_vals,
            floor=float(lever_floor),
            cap=float(lever_cap),
            weights=weights_no_capture,
        )

        rows.append({
            "region": str(region),
            "svi": svi,
            "avg_ratio": avg_ratio,
            "footfall": reg_tot["footfall"],
            "turnover": reg_tot["turnover"],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["region", "svi", "avg_ratio", "footfall", "turnover"])

    out["svi"] = pd.to_numeric(out["svi"], errors="coerce")
    out = out.dropna(subset=["svi"]).sort_values("svi", ascending=False).reset_index(drop=True)
    return out

# ----------------------
# Macro charts (FIXED)
# ----------------------
def plot_macro_panel(df_region_daily: pd.DataFrame, macro_start, macro_end):
    st.markdown(
        '<div class="panel"><div class="panel-title">Macro context — Consumer Confidence & Retail Index</div>',
        unsafe_allow_html=True
    )

    ms = pd.to_datetime(macro_start)
    me = pd.to_datetime(macro_end)

    if df_region_daily is None or df_region_daily.empty:
        st.info("No region data available for macro context.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    dd = df_region_daily.copy()
    dd["date"] = pd.to_datetime(dd["date"], errors="coerce")
    dd = dd.dropna(subset=["date"])
    dd = dd[(dd["date"] >= ms) & (dd["date"] <= me)].copy()

    if dd.empty:
        st.info("No region index series for this macro window (no footfall/turnover data).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    dd["footfall"] = pd.to_numeric(dd.get("footfall", np.nan), errors="coerce")
    dd["turnover"] = pd.to_numeric(dd.get("turnover", np.nan), errors="coerce")

    region_m = (
        dd.set_index("date")[["footfall", "turnover"]]
        .resample("MS")
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"date": "month"})
    )

    def _idx(series: pd.Series, k_base: int = 3) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").copy()
        valid = s[(s.notna()) & (s > 0)]
        if valid.empty:
            return pd.Series([np.nan] * len(s), index=s.index)
        base = float(valid.iloc[:k_base].mean())
        if pd.isna(base) or base <= 0:
            return pd.Series([np.nan] * len(s), index=s.index)
        return (s / base) * 100.0

    region_m["Region footfall-index"] = _idx(region_m["footfall"])
    region_m["Region omzet-index"] = _idx(region_m["turnover"])

    region_m = region_m.dropna(subset=["month"]).copy()
    if region_m.empty:
        st.info("No region index series for this macro window (after monthly aggregation).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    months_needed = int(((me - ms).days / 30.5) + 2)
    months_back = max(36, months_needed + 6)

    try:
        cci_raw = get_cci_series(months_back=months_back)
        ridx_raw = get_retail_index(months_back=months_back)
    except Exception as e:
        st.info(f"Macro data not available right now: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    def _parse_period_to_monthstart(s: str):
        if s is None or str(s).strip() == "":
            return pd.NaT
        s = str(s).strip().replace("MM", "M")

        m = re.match(r"^(\d{4})M(\d{2})$", s)
        if m:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)

        m = re.match(r"^(\d{4})(\d{2})$", s)
        if m:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)

        m = re.match(r"^(\d{4})[-/](\d{2})$", s)
        if m:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)

        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return pd.NaT
        return pd.Timestamp(dt.year, dt.month, 1)

    cci_df = pd.DataFrame(cci_raw) if isinstance(cci_raw, list) else pd.DataFrame()
    if not cci_df.empty and {"period", "cci"}.issubset(set(cci_df.columns)):
        cci_df["month"] = cci_df["period"].apply(_parse_period_to_monthstart)
        cci_df["value"] = pd.to_numeric(cci_df["cci"], errors="coerce")
        cci_df = cci_df.dropna(subset=["month", "value"])[["month", "value"]].sort_values("month")
    else:
        cci_df = pd.DataFrame(columns=["month", "value"])

    ridx_df = pd.DataFrame(ridx_raw) if isinstance(ridx_raw, list) else pd.DataFrame()
    if not ridx_df.empty and {"period", "retail_value"}.issubset(set(ridx_df.columns)):
        ridx_df["month"] = ridx_df["period"].apply(_parse_period_to_monthstart)
        ridx_df["value"] = pd.to_numeric(ridx_df["retail_value"], errors="coerce")
        ridx_df = ridx_df.dropna(subset=["month", "value"])[["month", "value"]].sort_values("month")
    else:
        ridx_df = pd.DataFrame(columns=["month", "value"])

    cci_df = cci_df[(cci_df["month"] >= ms) & (cci_df["month"] <= me)].copy()
    ridx_df = ridx_df[(ridx_df["month"] >= ms) & (ridx_df["month"] <= me)].copy()

    x_enc = alt.X(
        "month:T",
        title=None,
        scale=alt.Scale(domain=[ms, me]),
        axis=alt.Axis(format="%b %Y", labelAngle=30, labelOverlap=False, labelPadding=14)
    )

    reg_long = region_m.melt(
        id_vars=["month"],
        value_vars=["Region footfall-index", "Region omzet-index"],
        var_name="series",
        value_name="idx",
    )

    region_color_scale = alt.Scale(
        domain=["Region footfall-index", "Region omzet-index"],
        range=[PFM_PURPLE, PFM_RED],
    )

    region_lines = (
        alt.Chart(reg_long)
        .mark_line(point=True)
        .encode(
            x=x_enc,
            y=alt.Y("idx:Q", title="Regio-index (100 = start)"),
            color=alt.Color("series:N", scale=region_color_scale, legend=alt.Legend(title="", orient="right")),
            tooltip=[
                alt.Tooltip("month:T", title="Maand"),
                alt.Tooltip("series:N", title="Reeks"),
                alt.Tooltip("idx:Q", title="Index", format=".1f"),
            ],
        )
    )

    def macro_line(df_macro: pd.DataFrame, label: str, dash: bool):
        if df_macro is None or df_macro.empty:
            return None

        dfm = df_macro.copy()
        dfm["series"] = label

        return (
            alt.Chart(dfm)
            .mark_line(
                point=True,
                strokeWidth=2,
                strokeDash=[6, 4] if dash else [1, 0],
            )
            .encode(
                x=x_enc,
                y=alt.Y("value:Q", title=label, axis=alt.Axis(orient="right")),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(domain=[label], range=[BLACK]),
                    legend=alt.Legend(title="", orient="right"),
                ),
                tooltip=[
                    alt.Tooltip("month:T", title="Maand"),
                    alt.Tooltip("value:Q", title=label, format=".1f"),
                ],
            )
        )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**CBS detailhandelindex vs Regio**")
        mline = macro_line(ridx_df, "CBS retail index", dash=True)
        chart = alt.layer(region_lines, mline) if mline is not None else region_lines
        st.altair_chart(
            chart.resolve_scale(y="independent", color="independent").properties(height=280).configure_view(strokeWidth=0),
            use_container_width=True,
        )

    with c2:
        st.markdown("**Consumentenvertrouwen (CCI) vs Regio**")
        mline = macro_line(cci_df, "CCI", dash=True)
        chart = alt.layer(region_lines, mline) if mline is not None else region_lines
        st.altair_chart(
            chart.resolve_scale(y="independent", color="independent").properties(height=280).configure_view(strokeWidth=0),
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# MAIN APP
# ----------------------
def main():
    # ---- session defaults ----
    if "rcp_last_key" not in st.session_state:
        st.session_state.rcp_last_key = None
    if "rcp_payload" not in st.session_state:
        st.session_state.rcp_payload = None
    if "rcp_ran" not in st.session_state:
        st.session_state.rcp_ran = False

    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

    # ----------------------
    # Load clients (MUST be before Row 1)
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
    # Periods (MUST be before Row 2)
    # ----------------------
    periods = period_catalog(today=datetime.now().date())
    if not isinstance(periods, dict) or len(periods) == 0:
        st.error("period_catalog() returned no periods.")
        return
    period_labels = list(periods.keys())

    # ======================
    # ROW 1 — Title + Client + Run button (STACKED RIGHT)
    # ======================
    r1_left, r1_right = st.columns([3.6, 2.0], vertical_alignment="top")

    with r1_left:
        st.markdown(
            f"""
            <div class="pfm-header pfm-header--fixed">
              <div>
                <div class="pfm-title">PFM Region Performance Copilot <span class="pill">v2</span></div>
                <div class="pfm-sub">Region-level: explainable SVI + heatmap scanning + value upside + drilldown + macro context</div>
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
                key="rcp_client",
            )
    
        with c_btn:
            run_btn = st.button("Run analysis", type="primary", key="rcp_run")
    
        st.markdown("</div>", unsafe_allow_html=True)

    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    # Load locations + regions
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
        st.warning("No stores matched your region mapping for this retailer.")
        return

    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        merged["store_display"] = merged["name"] if "name" in merged.columns else merged["id"].astype(str)

    available_regions = sorted(merged["region"].dropna().unique().tolist())

    # ======================
    # ROW 2 — Selection (period) + Region + Options + SVI
    # ======================
    c_sel, c_reg, c_opt, c_svi = st.columns([1.1, 1.0, 1.4, 0.9], vertical_alignment="bottom")

    with c_sel:
        st.markdown('<div class="panel"><div class="panel-title">Selection</div>', unsafe_allow_html=True)
        period_choice = st.selectbox(
            "Period",
            period_labels,
            index=period_labels.index("Q3 2024") if "Q3 2024" in period_labels else 0,
            label_visibility="collapsed",
            key="rcp_period",
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
            key="rcp_region",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c_opt:
        st.markdown('<div class="panel"><div class="panel-title">Options</div>', unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        with t1:
            show_macro = st.toggle("Show macro context", value=True, key="rcp_macro")
        with t2:
            show_quadrant = st.toggle("Show quadrant", value=True, key="rcp_quadrant")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_svi:
        st.markdown('<div class="panel"><div class="panel-title">SVI sensitivity</div>', unsafe_allow_html=True)
        lever_floor = st.selectbox(
            "SVI floor",
            [70, 75, 80, 85],
            index=2,
            label_visibility="collapsed",
            key="rcp_floor"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Everything below is unchanged from your last provided script
    # ----------------------------------------------------------
    lever_cap = 200 - lever_floor
    run_key = (company_id, region_choice, str(start_period), str(end_period), int(lever_floor), int(lever_cap))

    selection_changed = st.session_state.rcp_last_key != run_key
    should_fetch = bool(run_btn) or bool(selection_changed) or (not bool(st.session_state.rcp_ran))

    if run_btn:
        st.toast("Running analysis…", icon="🚀")

    if (not should_fetch) and (st.session_state.rcp_payload is None):
        st.info("Select retailer / region / period and click **Run analysis**.")
        return

    # ----------------------
    # FETCH (only when needed)
    # ----------------------
    if should_fetch:
        region_shops = merged[merged["region"] == region_choice].copy()
        region_shop_ids = region_shops["id"].dropna().astype(int).unique().tolist()
        if not region_shop_ids:
            st.warning(f"No stores found for region '{region_choice}'.")
            return

        all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()

        metric_map = {
            "count_in": "footfall",
            "turnover": "turnover",
            "transactions": "transactions",
            "conversion_rate": "conversion_rate",
            "sales_per_visitor": "sales_per_visitor",
            "avg_basket_size": "avg_basket_size",
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

        with st.spinner("Fetching data via FastAPI..."):
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

        df_norm = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)
        if df_norm is None or df_norm.empty:
            st.warning("No data returned for the current selection.")
            return

        store_key_col = None
        for cand in ["shop_id", "id", "location_id"]:
            if cand in df_norm.columns:
                store_key_col = cand
                break
        if store_key_col is None:
            st.error("No store-id column found in response (shop_id/id/location_id).")
            return

        merged2 = enrich_merged_with_sqm_from_df_norm(merged, df_norm, store_key_col=store_key_col)

        # sqm_effective: override > locations sqm > sqm_api
        sqm_col = None
        for cand in ["sqm", "sq_meter", "sq_meters", "square_meters"]:
            if cand in merged2.columns:
                sqm_col = cand
                break

        base_sqm = pd.to_numeric(merged2[sqm_col], errors="coerce") if sqm_col is not None else np.nan
        sqm_api = pd.to_numeric(merged2["sqm_api"], errors="coerce") if "sqm_api" in merged2.columns else np.nan

        merged2["sqm_effective"] = np.where(
            merged2["sqm_override"].notna(),
            pd.to_numeric(merged2["sqm_override"], errors="coerce"),
            np.where(pd.notna(base_sqm), base_sqm, sqm_api)
        )

        df_daily_store = collapse_to_daily_store(df_norm, store_key_col=store_key_col)
        if df_daily_store is None or df_daily_store.empty:
            st.warning("No data after cleaning (daily/store collapse).")
            return

        join_cols = ["id", "store_display", "region", "sqm_effective"]
        if "store_type" in merged2.columns:
            join_cols.append("store_type")

        df_daily_store = df_daily_store.merge(
            merged2[join_cols].drop_duplicates(),
            left_on=store_key_col,
            right_on="id",
            how="left",
        )

        # Compute sales_per_sqm robustly (if API delivers empty)
        if "sales_per_sqm" not in df_daily_store.columns:
            df_daily_store["sales_per_sqm"] = np.nan

        sqm_eff = pd.to_numeric(df_daily_store.get("sqm_effective", np.nan), errors="coerce")
        turn = pd.to_numeric(df_daily_store.get("turnover", np.nan), errors="coerce")
        calc_spm2 = np.where((pd.notna(sqm_eff) & (sqm_eff > 0)), (turn / sqm_eff), np.nan)

        df_daily_store["sales_per_sqm"] = pd.to_numeric(df_daily_store["sales_per_sqm"], errors="coerce")
        df_daily_store["sales_per_sqm"] = df_daily_store["sales_per_sqm"].combine_first(pd.Series(calc_spm2, index=df_daily_store.index))

        df_daily_store = mark_closed_days_as_nan(df_daily_store)
        df_region_daily = df_daily_store[df_daily_store["region"] == region_choice].copy()

        st.session_state.rcp_last_key = run_key
        st.session_state.rcp_payload = {
            "df_daily_store": df_daily_store,
            "df_region_daily": df_region_daily,
            "merged": merged2,
            "store_key_col": store_key_col,
            "start_period": start_period,
            "end_period": end_period,
            "selected_client": selected_client,
            "region_choice": region_choice,
            "all_shop_ids": merged2["id"].dropna().astype(int).unique().tolist(),
            "report_url": REPORT_URL,
        }
        st.session_state.rcp_ran = True

    # ----------------------
    # READ CACHE (ALWAYS)
    # ----------------------
    payload = st.session_state.rcp_payload
    if payload is None:
        st.info("Select retailer / region / period and click **Run analysis**.")
        return

    # Everything below is your existing logic (unchanged)
    # ---------------------------------------------------
    df_daily_store = payload["df_daily_store"]
    df_region_daily = payload["df_region_daily"]
    merged = payload["merged"]
    store_key_col = payload["store_key_col"]
    start_period = payload["start_period"]
    end_period = payload["end_period"]
    selected_client = payload["selected_client"]
    region_choice = payload["region_choice"]
    all_shop_ids = payload.get("all_shop_ids", [])
    cfg = VemcountApiConfig(report_url=payload.get("report_url", REPORT_URL))

    if df_region_daily is None or df_region_daily.empty:
        st.warning("No data for selected region in this period.")
        return

    # ----------------------
    # KPI header
    # ----------------------
    st.markdown(f"## {selected_client['brand']} — Region **{region_choice}** · {start_period} → {end_period}")

    foot_total = float(pd.to_numeric(df_region_daily["footfall"], errors="coerce").dropna().sum()) if "footfall" in df_region_daily.columns else 0.0
    turn_total = float(pd.to_numeric(df_region_daily["turnover"], errors="coerce").dropna().sum()) if "turnover" in df_region_daily.columns else 0.0
    trans_total = float(pd.to_numeric(df_region_daily["transactions"], errors="coerce").dropna().sum()) if "transactions" in df_region_daily.columns else 0.0

    conv = (trans_total / foot_total * 100.0) if foot_total > 0 else np.nan
    spv = (turn_total / foot_total) if foot_total > 0 else np.nan
    atv = (turn_total / trans_total) if trans_total > 0 else np.nan

    # ----------------------
    # Pathzz store-level capture
    # ----------------------
    pz_path = "data/pathzz_sample_weekly.csv"
    pz_mtime = os.path.getmtime(pz_path) if os.path.exists(pz_path) else 0.0
    pathzz_all = load_pathzz_weekly_store(pz_path, pz_mtime)
    pathzz_period = filter_pathzz_for_period(pathzz_all, start_period, end_period)

    pathzz_region = pd.DataFrame(columns=["region", "shop_id", "store_type", "week_start", "visits"])
    if pathzz_period is not None and not pathzz_period.empty and "region" in pathzz_period.columns:
        pathzz_region = pathzz_period[
            pathzz_period["region"].astype(str).str.strip().str.lower()
            == str(region_choice).strip().lower()
        ].copy()

    dd_region = df_region_daily.copy()
    dd_region["date"] = pd.to_datetime(dd_region["date"], errors="coerce")
    dd_region = dd_region.dropna(subset=["date"])
    dd_region["week_start"] = dd_region["date"].dt.to_period("W-SAT").dt.start_time

    for c in ["footfall", "turnover", "transactions"]:
        dd_region[c] = pd.to_numeric(dd_region.get(c, np.nan), errors="coerce").fillna(0.0)

    store_week = (
        dd_region.groupby(["id", "week_start"], as_index=False)
        .agg(footfall=("footfall", "sum"), turnover=("turnover", "sum"), transactions=("transactions", "sum"))
    )

    if pathzz_region is None or pathzz_region.empty:
        pathzz_store_week = pd.DataFrame(columns=["id", "week_start", "visits"])
    else:
        pathzz_store_week = (
            pathzz_region.groupby(["shop_id", "week_start"], as_index=False)
            .agg(visits=("visits", "sum"))
        ).rename(columns={"shop_id": "id"})

    capture_store_week = store_week.merge(pathzz_store_week, on=["id", "week_start"], how="inner")
    if not capture_store_week.empty:
        capture_store_week["capture_rate"] = np.where(
            capture_store_week["visits"] > 0,
            capture_store_week["footfall"] / capture_store_week["visits"] * 100.0,
            np.nan,
        )

    region_weekly = (
        capture_store_week.groupby("week_start", as_index=False)
        .agg(footfall=("footfall", "sum"), visits=("visits", "sum"), turnover=("turnover", "sum"))
        if not capture_store_week.empty else pd.DataFrame()
    )
    if not region_weekly.empty:
        region_weekly["capture_rate"] = np.where(
            region_weekly["visits"] > 0,
            region_weekly["footfall"] / region_weekly["visits"] * 100.0,
            np.nan
        )

    avg_capture = np.nan
    if not capture_store_week.empty:
        total_visits = float(pd.to_numeric(capture_store_week["visits"], errors="coerce").dropna().sum())
        total_ff = float(pd.to_numeric(capture_store_week["footfall"], errors="coerce").dropna().sum())
        avg_capture = (total_ff / total_visits * 100.0) if total_visits > 0 else np.nan

    store_capture = {}
    if not capture_store_week.empty:
        tmp = capture_store_week.copy()
        tmp["footfall"] = pd.to_numeric(tmp["footfall"], errors="coerce")
        tmp["visits"] = pd.to_numeric(tmp["visits"], errors="coerce")
        store_agg = tmp.groupby("id", as_index=False).agg(footfall=("footfall", "sum"), visits=("visits", "sum"))
        store_agg["capture_rate"] = np.where(
            store_agg["visits"] > 0,
            store_agg["footfall"] / store_agg["visits"] * 100.0,
            np.nan
        )
        for _, r in store_agg.iterrows():
            store_capture[int(r["id"])] = float(r["capture_rate"]) if pd.notna(r["capture_rate"]) else np.nan

    def agg_period(df_: pd.DataFrame) -> dict:
        foot = float(pd.to_numeric(df_.get("footfall", 0), errors="coerce").dropna().sum())
        turn = float(pd.to_numeric(df_.get("turnover", 0), errors="coerce").dropna().sum())
        trans = float(pd.to_numeric(df_.get("transactions", 0), errors="coerce").dropna().sum())
        sqm = pd.to_numeric(df_.get("sqm_effective", np.nan), errors="coerce")
        sqm_sum = float(sqm.dropna().drop_duplicates().sum()) if sqm.notna().any() else np.nan
        return {"footfall": foot, "turnover": turn, "transactions": trans, "sqm_sum": sqm_sum}

    reg_tot = agg_period(df_region_daily)
    comp_tot = agg_period(df_daily_store)

    reg_vals = compute_driver_values_from_period(
        footfall=reg_tot["footfall"],
        turnover=reg_tot["turnover"],
        transactions=reg_tot["transactions"],
        sqm_sum=reg_tot["sqm_sum"],
        capture_pct=avg_capture,
    )
    comp_vals = compute_driver_values_from_period(
        footfall=comp_tot["footfall"],
        turnover=comp_tot["turnover"],
        transactions=comp_tot["transactions"],
        sqm_sum=comp_tot["sqm_sum"],
        capture_pct=np.nan,
    )
    if pd.isna(comp_vals.get("capture_rate", np.nan)):
        comp_vals["capture_rate"] = reg_vals.get("capture_rate", np.nan)

    # --- Region store-type mix weighting (instead of dominant type) ---
    region_types = (
        merged.loc[merged["region"] == region_choice, "store_type"]
        if ("store_type" in merged.columns and "region" in merged.columns)
        else pd.Series([], dtype=str)
    )
    
    region_weights = get_svi_weights_for_region_mix(region_types)
    
    # Optional: still compute dominant type for display only (NOT for weighting)
    dominant_store_type = ""
    if region_types is not None and not region_types.dropna().empty:
        _rt = region_types.dropna().astype(str).str.strip()
        _rt = _rt[_rt.str.lower() != "nan"]
        if not _rt.empty:
            dominant_store_type = str(_rt.value_counts().index[0]).strip()

    region_svi, region_avg_ratio, region_bd = compute_svi_explainable(
        vals_a=reg_vals,
        vals_b=comp_vals,
        floor=float(lever_floor),
        cap=float(lever_cap),
        weights=region_weights,
    )
    status_txt, status_color = status_from_score(region_svi if pd.notna(region_svi) else 0)

    k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 1])
    with k1:
        kpi_card("Footfall", fmt_int(foot_total), "Region · selected period")
    with k2:
        kpi_card("Revenue", fmt_eur(turn_total), "Region · selected period")
    with k3:
        kpi_card("Conversion", fmt_pct(conv), "Transactions / Visitors")
    with k4:
        kpi_card("ATV", fmt_eur_2(atv), "Revenue / Transaction")
    with k5:
        cap_help = "Weighted capture from Pathzz visits (store-week matched)"
        if pd.isna(avg_capture):
            cap_help = "No matched Pathzz store-week rows for this region/period (check mapping + weeks)."
        kpi_card("Capture", fmt_pct(avg_capture), cap_help)

    st.markdown(
        "<div class='muted'>Benchmark: Company = all shops in company (same period). Capture is derived from Pathzz weekly store visits matched to Vemcount weekly footfall (by shop_id + week_start).</div>",
        unsafe_allow_html=True,
    )

    store_dim = merged[["id","region","store_type","sqm_effective","store_display"]].drop_duplicates().copy() if "store_type" in merged.columns else merged[["id","region","sqm_effective","store_display"]].drop_duplicates().copy()
    
    reg_type_k, com_type_k, type_idx, mix_g = compute_store_type_benchmarks(
        df_daily_store=df_daily_store,
        region_choice=region_choice,
        store_dim=store_dim,
        capture_store_week=capture_store_week if "capture_store_week" in locals() else None
    )

    # --- Company-wide region leaderboard ---
    df_region_rank = compute_svi_by_region_companywide(df_daily_store, lever_floor, lever_cap)
    
    # ======================
    # SVI ROW — 3 blocks on 1 row
    # (1) Region vs regions bar (left)
    # (2) SVI card (middle)
    # (3) Store types table (right)
    # ======================
    col_bar, col_mid, col_types = st.columns([2.2, 1.6, 2.2], vertical_alignment="top")
    
    # -------- (1) LEFT: Region SVI bar --------
    with col_bar:
        st.markdown(
            '<div class="panel"><div class="panel-title">Region SVI — vs other regions (company-wide)</div>',
            unsafe_allow_html=True
        )
    
        if df_region_rank is None or df_region_rank.empty:
            st.info("No region leaderboard available.")
        else:
            TOP_N = 8
            df_plot = df_region_rank.head(TOP_N).copy()
    
            if region_choice not in df_plot["region"].tolist() and region_choice in df_region_rank["region"].tolist():
                df_sel = df_region_rank[df_region_rank["region"] == region_choice].copy()
                df_plot = pd.concat([df_plot, df_sel], ignore_index=True)
    
            df_plot = df_plot.sort_values("svi", ascending=True).copy()
    
            bar = (
                alt.Chart(df_plot)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    y=alt.Y("region:N", sort=df_plot["region"].tolist(), title=None),
                    x=alt.X("svi:Q", title="SVI (0–100)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.condition(
                        alt.datum.region == region_choice,
                        alt.value(PFM_PURPLE),
                        alt.value(PFM_LINE)
                    ),
                    tooltip=[
                        alt.Tooltip("region:N", title="Region"),
                        alt.Tooltip("svi:Q", title="SVI", format=".0f"),
                        alt.Tooltip("avg_ratio:Q", title="Avg ratio vs company", format=".0f"),
                        alt.Tooltip("turnover:Q", title="Revenue", format=",.0f"),
                        alt.Tooltip("footfall:Q", title="Footfall", format=",.0f"),
                    ],
                )
                .properties(height=250)
            )
    
            st.altair_chart(bar, use_container_width=True)
            st.markdown(
                "<div class='hint'>Highlighted = selected region. (Capture excluded for fair comparison across regions.)</div>",
                unsafe_allow_html=True
            )
    
        st.markdown("</div>", unsafe_allow_html=True)
    
    # -------- (2) MIDDLE: SVI card --------
    with col_mid:
        big_score = 0 if pd.isna(region_svi) else float(region_svi)
    
        st.markdown(
            f"""
            <div class="panel">
              <div class="panel-title">Store Vitality Index (SVI) — region vs company-wide</div>
    
              <div style="height:0.55rem"></div>
    
              <div style="display:flex; align-items:baseline; gap:0.55rem;">
                <div style="font-size:3.9rem;font-weight:950;line-height:1;color:{status_color};letter-spacing:-0.02em;">
                  {big_score:.0f}
                </div>
                <div class="pill">/ 100</div>
              </div>
    
              <div style="height:0.6rem"></div>
    
              <div class="muted" style="line-height:1.35;">
                Status: <span style="font-weight:900;color:{status_color}">{status_txt}</span><br/>
                Weighted driver ratio vs company ≈ <b>{"" if pd.isna(region_avg_ratio) else f"{region_avg_ratio:.0f}%"} </b>
                <span class="hint">(ratios clipped {lever_floor}–{lever_cap}% → 0–100)</span>
              </div>
    
              <div class="hint" style="margin-top:0.7rem">
                Weighting: region store-type mix <span class="pill">see table</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # -------- (3) RIGHT: Store types table --------
    with col_types:
        st.markdown(
            '<div class="panel"><div class="panel-title">Store types in this region — vs company (same store type)</div>',
            unsafe_allow_html=True
        )
    
        if mix_g.empty:
            st.info("No store_type mix available (check regions.csv store_type column).")
        else:
            show = mix_g.merge(
                type_idx[["store_type", "SPV idx", "CR idx", "Sales/m² idx", "ATV idx", "Capture idx"]],
                on="store_type",
                how="left"
            ).sort_values("Stores", ascending=False)
    
            disp = show.rename(columns={"store_type": "Store type"}).copy()
    
            idx_cols = ["SPV idx", "CR idx", "Sales/m² idx", "ATV idx", "Capture idx"]
    
            if "Store share" in disp.columns:
                disp["Store share"] = pd.to_numeric(disp["Store share"], errors="coerce")
    
            for c in idx_cols:
                if c in disp.columns:
                    disp[c] = pd.to_numeric(disp[c], errors="coerce")
    
            def _idx_style(v):
                try:
                    if pd.isna(v):
                        return ""
                    v = float(v)
    
                    if v >= 115:
                        return f"background-color:{PFM_PURPLE}1A; color:{PFM_DARK}; font-weight:900;"
                    if v >= 105:
                        return f"background-color:{PFM_PURPLE}0F; color:{PFM_DARK}; font-weight:800;"
                    if v >= 95:
                        return f"background-color:#FFFFFF; color:{PFM_DARK}; font-weight:700;"
                    if v >= 85:
                        return f"background-color:{PFM_RED}12; color:{PFM_DARK}; font-weight:800;"
                    return f"background-color:{PFM_RED}22; color:{PFM_DARK}; font-weight:900;"
                except Exception:
                    return ""
    
            def _share_style(v):
                try:
                    if pd.isna(v):
                        return ""
                    return f"background-color:{PFM_LIGHT}; font-weight:800; color:{PFM_DARK};"
                except Exception:
                    return ""
    
            show_cols = ["Store type", "Stores", "Store share"] + [c for c in idx_cols if c in disp.columns]
            df_show = disp[show_cols].copy()
    
            styler = df_show.style
    
            heat_cols = [c for c in idx_cols if c in df_show.columns]
            if heat_cols:
                styler = styler.applymap(_idx_style, subset=heat_cols)
    
            if "Store share" in df_show.columns:
                styler = styler.applymap(_share_style, subset=["Store share"])
    
            fmt = {}
            if "Store share" in df_show.columns:
                fmt["Store share"] = lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%"
            for c in heat_cols:
                fmt[c] = lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%"
    
            styler = styler.format(fmt)
    
            st.dataframe(styler, use_container_width=True, hide_index=True)
    
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Macro charts ---
    if show_macro:
        macro_start = (pd.to_datetime(start_period) - pd.Timedelta(days=365)).date()
        macro_end = pd.to_datetime(end_period).date()

        with st.spinner("Fetching macro-window data (region indices)..."):
            try:
                resp_macro = fetch_report(
                    cfg=cfg,
                    shop_ids=all_shop_ids,
                    data_outputs=["count_in", "turnover"],
                    period="date",
                    step="day",
                    source="shops",
                    date_from=macro_start,
                    date_to=macro_end,
                    timeout=120,
                )
            except Exception as e:
                st.info(f"Macro window fetch failed: {e}")
                resp_macro = None

        if resp_macro is not None:
            df_macro = normalize_vemcount_response(
                resp_macro, kpi_keys=["count_in", "turnover"]
            ).rename(columns={"count_in": "footfall", "turnover": "turnover"})

            if df_macro is not None and not df_macro.empty:
                store_key_col_macro = store_key_col
                if store_key_col_macro not in df_macro.columns:
                    for cand in ["shop_id", "id", "location_id"]:
                        if cand in df_macro.columns:
                            store_key_col_macro = cand
                            break

                df_macro = collapse_to_daily_store(df_macro, store_key_col=store_key_col_macro)

                df_macro = df_macro.merge(
                    merged[["id", "region"]].drop_duplicates(),
                    left_on=store_key_col_macro,
                    right_on="id",
                    how="left",
                )

                df_region_macro = df_macro[df_macro["region"] == region_choice].copy()
                plot_macro_panel(df_region_macro, macro_start, macro_end)
            else:
                st.info("Macro window data returned empty (no region indices available).")
        else:
            st.info("Macro window data not available right now.")


    # ============================================================
    # Everything below was part of Region Copilot v2 and is restored
    # ============================================================

    # Region benchmark alias (stores vs region baseline)
    reg_bench = reg_vals

    # ======================
    # Weekly trend — Footfall vs Pathzz visits + Capture
    # ======================
    st.markdown('<div class="panel"><div class="panel-title">Weekly trend — Footfall vs Street Traffic (Pathzz Visits) + Capture</div>', unsafe_allow_html=True)

    if region_weekly is None or region_weekly.empty:
        st.info("No matching Pathzz store-week data for this region/period. (Check: region naming, shop_id mapping, and week alignment.)")
    else:
        chart_df = region_weekly[["week_start", "footfall", "visits", "turnover", "capture_rate"]].copy()
        chart_df = chart_df.sort_values("week_start")

        iso = chart_df["week_start"].dt.isocalendar()
        chart_df["week_label"] = iso.week.apply(lambda w: f"W{int(w):02d}")
        week_order = chart_df["week_label"].tolist()

        long = chart_df.melt(
            id_vars=["week_label"],
            value_vars=["footfall", "visits"],
            var_name="metric",
            value_name="value",
        )
        label_map = {"footfall": "Footfall (stores)", "visits": "Street traffic (Pathzz)"}
        long["metric_label"] = long["metric"].map(label_map).fillna(long["metric"])

        bars = (
            alt.Chart(long)
            .mark_bar(opacity=0.85, cornerRadiusEnd=4)
            .encode(
                x=alt.X("week_label:N", sort=week_order, title=None),
                xOffset=alt.XOffset("metric_label:N"),
                y=alt.Y("value:Q", title=""),
                color=alt.Color(
                    "metric_label:N",
                    scale=alt.Scale(domain=["Footfall (stores)", "Street traffic (Pathzz)"], range=[PFM_PURPLE, PFM_LINE]),
                    legend=alt.Legend(title="", orient="right"),
                ),
                tooltip=[
                    alt.Tooltip("week_label:N", title="Week"),
                    alt.Tooltip("metric_label:N", title="Metric"),
                    alt.Tooltip("value:Q", title="Value", format=",.0f"),
                ],
            )
        )

        chart_df2 = chart_df.copy()
        chart_df2["series"] = "Capture %"

        line = (
            alt.Chart(chart_df2)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("week_label:N", sort=week_order, title=None),
                y=alt.Y("capture_rate:Q", title="Capture %"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(domain=["Capture %"], range=[PFM_DARK]),
                    legend=alt.Legend(title=""),
                ),
                tooltip=[
                    alt.Tooltip("week_label:N", title="Week"),
                    alt.Tooltip("capture_rate:Q", title="Capture", format=".1f"),
                ],
            )
        )

        st.altair_chart(
            alt.layer(bars, line).resolve_scale(y="independent", color="independent").properties(height=260),
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # Heatmap — stores vs region benchmark (scanning-machine)
    # ======================
    st.markdown("## Heatmap — stores vs benchmark (scanning-machine)")

    reg_store_daily = df_daily_store[df_daily_store["region"] == region_choice].copy()
    agg = reg_store_daily.groupby(["id", "store_display"], as_index=False).agg(
        turnover=("turnover", "sum"),
        footfall=("footfall", "sum"),
        transactions=("transactions", "sum"),
    )

    agg["conversion_rate"] = np.where(agg["footfall"] > 0, agg["transactions"] / agg["footfall"] * 100.0, np.nan)
    agg["sales_per_visitor"] = np.where(agg["footfall"] > 0, agg["turnover"] / agg["footfall"], np.nan)
    agg["sales_per_transaction"] = np.where(agg["transactions"] > 0, agg["turnover"] / agg["transactions"], np.nan)

    sqm_map_cols = ["id", "sqm_effective"]
    if "store_type" in merged.columns:
        sqm_map_cols.append("store_type")

    sqm_map = merged.loc[merged["region"] == region_choice, sqm_map_cols].drop_duplicates()
    sqm_map["sqm_effective"] = pd.to_numeric(sqm_map["sqm_effective"], errors="coerce")
    agg = agg.merge(sqm_map, on="id", how="left")

    agg["sales_per_sqm"] = np.where(
        (agg["sqm_effective"] > 0) & pd.notna(agg["sqm_effective"]),
        agg["turnover"] / agg["sqm_effective"],
        np.nan,
    )

    # store capture from Pathzz (weighted across period)
    agg["capture_rate"] = agg["id"].apply(lambda x: store_capture.get(int(x), np.nan) if pd.notna(x) else np.nan)

    def store_driver_vals(row):
        return compute_driver_values_from_period(
            footfall=row["footfall"],
            turnover=row["turnover"],
            transactions=row["transactions"],
            sqm_sum=row["sqm_effective"],
            capture_pct=row["capture_rate"],
        )

    svi_list = []
    ratios_map = {k: [] for k, _ in SVI_DRIVERS}

    for _, r in agg.iterrows():
        vals = store_driver_vals(r)
        stype = r.get("store_type", "")
        w = get_svi_weights_for_store_type(stype)

        svi, _, bd_store = compute_svi_explainable(vals, reg_bench, float(lever_floor), float(lever_cap), weights=w)
        svi_list.append(svi)

        bd_store = bd_store.copy()
        for dk, _ in SVI_DRIVERS:
            rr = bd_store.loc[bd_store["driver_key"] == dk, "ratio_pct"]
            ratios_map[dk].append(float(rr.iloc[0]) if (not rr.empty and pd.notna(rr.iloc[0])) else np.nan)

    agg["SVI"] = svi_list
    for dk, _ in SVI_DRIVERS:
        agg[f"{dk}_idx"] = ratios_map[dk]

    # ======================
    # Value Upside (scenario) — actionable drivers only
    # ======================
    days_in_period = max(1, (pd.to_datetime(end_period) - pd.to_datetime(start_period)).days + 1)

    def calc_upside_for_store(row):
        '''
        Returns (upside_period_eur, driver_label)

        Conservative, actionable:
          - Low SPV: lift SPV to benchmark, footfall constant
          - Low Sales/m²: lift Sales/m² to benchmark, sqm constant
          - Low Conversion: lift conversion to benchmark, footfall constant, turnover via ATV
        Capture is visible but excluded as 'main driver' by design.
        '''
        foot = row["footfall"]
        turn = row["turnover"]
        sqm = row["sqm_effective"]
        trans = row["transactions"]

        spv_s = safe_div(turn, foot)
        spsqm_s = safe_div(turn, sqm)
        cr_s = safe_div(trans, foot) * 100.0 if (pd.notna(trans) and pd.notna(foot) and float(foot) != 0.0) else np.nan
        atv_s = safe_div(turn, trans)

        spv_b = reg_bench.get("sales_per_visitor", np.nan)
        spsqm_b = reg_bench.get("sales_per_sqm", np.nan)
        cr_b = reg_bench.get("conversion_rate", np.nan)
        atv_b = reg_bench.get("sales_per_transaction", np.nan)

        atv_use = atv_s if pd.notna(atv_s) else atv_b
        candidates = []

        if pd.notna(foot) and foot > 0 and pd.notna(spv_s) and pd.notna(spv_b) and spv_s < spv_b:
            candidates.append(("Low SPV", float(foot) * float(spv_b - spv_s)))

        if pd.notna(sqm) and sqm > 0 and pd.notna(spsqm_s) and pd.notna(spsqm_b) and spsqm_s < spsqm_b:
            candidates.append(("Low Sales / m²", float(sqm) * float(spsqm_b - spsqm_s)))

        if pd.notna(foot) and foot > 0 and pd.notna(cr_s) and pd.notna(cr_b) and cr_s < cr_b and pd.notna(atv_use):
            extra_trans = float(foot) * (float(cr_b - cr_s) / 100.0)
            candidates.append(("Low Conversion", max(0.0, extra_trans) * float(atv_use)))

        if not candidates:
            return np.nan, ""

        best = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
        upside = float(best[1]) if best[1] > 0 else np.nan
        return upside, best[0]

    heat = agg.copy()
    ups = heat.apply(calc_upside_for_store, axis=1, result_type="expand")
    heat["upside_period_eur"] = pd.to_numeric(ups.iloc[:, 0], errors="coerce")
    heat["upside_driver"] = ups.iloc[:, 1].astype(str)
    heat["upside_annual_eur"] = heat["upside_period_eur"] * (365.0 / float(days_in_period))

    heat_show = heat[[
        "store_display",
        "SVI",
        "turnover",
        "footfall",
        "sales_per_visitor_idx",
        "sales_per_sqm_idx",
        "capture_rate_idx",
        "conversion_rate_idx",
        "sales_per_transaction_idx",
        "upside_period_eur",
        "upside_annual_eur",
        "upside_driver",
    ]].copy()

    heat_show = heat_show.rename(columns={
        "store_display": "Store",
        "turnover": "Revenue",
        "footfall": "Footfall",
        "sales_per_visitor_idx": "SPV idx",
        "sales_per_sqm_idx": "Sales/m² idx",
        "capture_rate_idx": "Capture idx",
        "conversion_rate_idx": "CR idx",
        "sales_per_transaction_idx": "ATV idx",
        "upside_period_eur": "Upside (period)",
        "upside_annual_eur": "Upside (annualized)",
        "upside_driver": "Main driver",
    })

    cA, cB = st.columns([2, 1])
    with cA:
        st.caption("Sort tip: click **SVI** (low → high) or **Upside (annualized)** (high → low) to focus fast.")
        st.caption("Note: Capture is shown in the heatmap, but intentionally not used as the main upside driver.")
    with cB:
        show_heat_styling = st.toggle("Show heatmap colors", value=True, key="rcp_heat_colors")

    def style_heatmap_ratio(val):
        try:
            if pd.isna(val):
                return ""
            v = float(val)
            if v >= 110:
                return "background-color:#F5F3FF; color:#4C1D95; font-weight:900;"
            if v >= 95:
                return "background-color:#FFF7ED; color:#9A3412; font-weight:900;"
            return "background-color:#FFF1F2; color:#9F1239; font-weight:900;"
        except Exception:
            return ""

    if not show_heat_styling:
        disp = heat_show.copy()
        disp["SVI"] = pd.to_numeric(disp["SVI"], errors="coerce").apply(lambda x: "-" if pd.isna(x) else f"{x:.0f}")
        disp["Revenue"] = pd.to_numeric(disp["Revenue"], errors="coerce").apply(fmt_eur)
        disp["Footfall"] = pd.to_numeric(disp["Footfall"], errors="coerce").apply(fmt_int)
        for c in ["SPV idx", "Sales/m² idx", "Capture idx", "CR idx", "ATV idx"]:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").apply(lambda x: "-" if pd.isna(x) else f"{x:.0f}%")
        disp["Upside (period)"] = pd.to_numeric(disp["Upside (period)"], errors="coerce").apply(lambda x: "-" if pd.isna(x) else fmt_eur(x))
        disp["Upside (annualized)"] = pd.to_numeric(disp["Upside (annualized)"], errors="coerce").apply(lambda x: "-" if pd.isna(x) else fmt_eur(x))
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        styled = heat_show.copy()
        styler = styled.style
        for col in ["SPV idx", "Sales/m² idx", "Capture idx", "CR idx", "ATV idx"]:
            if col in styled.columns:
                styler = styler.applymap(style_heatmap_ratio, subset=[col])

        def _svi_row_style(v):
            try:
                if pd.isna(v):
                    return ""
                v = float(v)
                if v < 45:
                    return "background-color:#fff1f2; font-weight:900;"
                if v < 60:
                    return "background-color:#fffbeb; font-weight:900;"
                return ""
            except Exception:
                return ""

        styler = styler.applymap(_svi_row_style, subset=["SVI"])

        styler = styler.format({
            "SVI": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}",
            "Revenue": lambda x: "-" if pd.isna(x) else fmt_eur(float(x)),
            "Footfall": lambda x: "-" if pd.isna(x) else fmt_int(float(x)),
            "SPV idx": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%",
            "Sales/m² idx": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%",
            "Capture idx": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%",
            "CR idx": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%",
            "ATV idx": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}%",
            "Upside (period)": lambda x: "-" if pd.isna(x) else fmt_eur(float(x)),
            "Upside (annualized)": lambda x: "-" if pd.isna(x) else fmt_eur(float(x)),
        })

        st.dataframe(styler, use_container_width=True, hide_index=True)

    st.markdown("## Value Upside (scenario) — biggest opportunities")

    opp = heat.copy()
    opp["up_period"] = pd.to_numeric(opp["upside_period_eur"], errors="coerce")
    opp["up_annual"] = pd.to_numeric(opp["upside_annual_eur"], errors="coerce")
    opp = opp.dropna(subset=["up_period"]).sort_values("up_period", ascending=False).head(5)

    total_period = float(pd.to_numeric(opp["up_period"], errors="coerce").dropna().sum()) if not opp.empty else np.nan
    total_annual = float(pd.to_numeric(opp["up_annual"], errors="coerce").dropna().sum()) if not opp.empty else np.nan

    st.markdown(
        f"""
        <div class="callout">
          <div class="callout-title">Top 5 upside (period): {fmt_eur(total_period) if pd.notna(total_period) else "-"}</div>
          <div class="callout-sub">
            Annualized upside: <b>{fmt_eur(total_annual) if pd.notna(total_annual) else "-"}</b> / year
            <span class="hint">(simple extrapolation; seasonality & feasibility decide realism)</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    if opp.empty:
        st.info("No clear upside computed yet (check drivers: SPV / Sales/m² / Conversion).")
    else:
        show_opp = pd.DataFrame({
            "Store": opp["store_display"].values,
            "Main driver": opp["upside_driver"].values,
            "Upside (period)": opp["up_period"].apply(fmt_eur).values,
            "Upside (annualized)": opp["up_annual"].apply(fmt_eur).values,
        })
        st.dataframe(show_opp, use_container_width=True, hide_index=True)

    # ======================
    # Quadrant — Conversion vs SPV (stores in region) — WITH store_type
    # ======================
    if show_quadrant:
        st.markdown("## Quadrant — Conversion vs SPV (stores in region)")
    
        q = df_daily_store[df_daily_store["region"] == region_choice].copy()
        if q.empty:
            st.info("No store data for quadrant.")
        else:
            q_agg = q.groupby(["id", "store_display"], as_index=False).agg(
                footfall=("footfall", "sum"),
                turnover=("turnover", "sum"),
                transactions=("transactions", "sum"),
            )
    
            q_agg["conversion_rate"] = np.where(
                q_agg["footfall"] > 0,
                q_agg["transactions"] / q_agg["footfall"] * 100.0,
                np.nan
            )
            q_agg["sales_per_visitor"] = np.where(
                q_agg["footfall"] > 0,
                q_agg["turnover"] / q_agg["footfall"],
                np.nan
            )
    
            # ---- attach store_type from merged (region mapping) ----
            # merged should contain: id, store_type (from regions.csv)
            if "store_type" in merged.columns:
                stype_map = merged[["id", "store_type"]].drop_duplicates().copy()
                stype_map["id"] = pd.to_numeric(stype_map["id"], errors="coerce")
                stype_map["store_type"] = (
                    stype_map["store_type"]
                    .fillna("Unknown")
                    .astype(str)
                    .str.strip()
                    .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
                )
    
                q_agg["id"] = pd.to_numeric(q_agg["id"], errors="coerce")
                q_agg = q_agg.merge(stype_map, on="id", how="left")
            else:
                q_agg["store_type"] = "Unknown"
    
            q_agg["store_type"] = (
                q_agg["store_type"]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
            )
    
            q_agg = q_agg.dropna(subset=["conversion_rate", "sales_per_visitor"])
            if q_agg.empty:
                st.info("Not enough data to plot quadrant (missing conversion/SPV).")
            else:
                x_med = float(q_agg["conversion_rate"].median())
                y_med = float(q_agg["sales_per_visitor"].median())
    
                base = alt.Chart(q_agg).mark_circle(size=140).encode(
                    x=alt.X("conversion_rate:Q", title="Conversion (%)"),
                    y=alt.Y("sales_per_visitor:Q", title="SPV (€/visitor)"),
                    color=alt.Color("store_type:N", legend=alt.Legend(title="Store type")),
                    tooltip=[
                        alt.Tooltip("store_display:N", title="Store"),
                        alt.Tooltip("store_type:N", title="Store type"),
                        alt.Tooltip("conversion_rate:Q", title="Conversion", format=".1f"),
                        alt.Tooltip("sales_per_visitor:Q", title="SPV", format=",.2f"),
                        alt.Tooltip("turnover:Q", title="Revenue", format=",.0f"),
                        alt.Tooltip("footfall:Q", title="Footfall", format=",.0f"),
                    ],
                )
    
                vline = (
                    alt.Chart(pd.DataFrame({"x": [x_med]}))
                    .mark_rule(strokeDash=[6, 4])
                    .encode(x="x:Q")
                )
                hline = (
                    alt.Chart(pd.DataFrame({"y": [y_med]}))
                    .mark_rule(strokeDash=[6, 4])
                    .encode(y="y:Q")
                )
    
                st.altair_chart(
                    alt.layer(base, vline, hline)
                      .properties(height=360)
                      .configure_view(strokeWidth=0),
                    use_container_width=True,
                )

    # ----------------------
    # Store drilldown
    # ----------------------
    st.markdown("## Store drilldown")

    region_stores = merged[merged["region"] == region_choice].copy()
    region_stores = region_stores.dropna(subset=["id"]).copy()
    region_stores["id_int"] = region_stores["id"].astype(int)
    region_stores["dd_label"] = region_stores["store_display"].fillna(region_stores["id"].astype(str)) + " · " + region_stores["id"].astype(str)

    if "rcp_store_choice" not in st.session_state:
        st.session_state.rcp_store_choice = int(region_stores["id_int"].iloc[0])

    store_choice_label = st.selectbox(
        "Store",
        region_stores["dd_label"].tolist(),
        index=int(np.where(region_stores["id_int"].values == st.session_state.rcp_store_choice)[0][0]) if (st.session_state.rcp_store_choice in region_stores["id_int"].values) else 0,
        key="rcp_store_select",
    )
    chosen_id = int(store_choice_label.split("·")[-1].strip())
    st.session_state.rcp_store_choice = chosen_id

    df_store = df_daily_store[pd.to_numeric(df_daily_store["id"], errors="coerce").astype("Int64") == chosen_id].copy()
    store_name = region_stores.loc[region_stores["id_int"] == chosen_id, "store_display"].iloc[0] if (region_stores["id_int"] == chosen_id).any() else str(chosen_id)

    store_type_store = ""
    if "store_type" in region_stores.columns:
        try:
            store_type_store = str(region_stores.loc[region_stores["id_int"] == chosen_id, "store_type"].iloc[0])
        except Exception:
            store_type_store = ""

    st.markdown(
        f"### **{store_name}** · storeID {chosen_id} <span class='pill'>{store_type_store if store_type_store else 'unknown'}</span>",
        unsafe_allow_html=True,
    )

    foot_s = float(pd.to_numeric(df_store["footfall"], errors="coerce").dropna().sum()) if "footfall" in df_store.columns else 0.0
    turn_s = float(pd.to_numeric(df_store["turnover"], errors="coerce").dropna().sum()) if "turnover" in df_store.columns else 0.0
    trans_s = float(pd.to_numeric(df_store["transactions"], errors="coerce").dropna().sum()) if "transactions" in df_store.columns else 0.0

    conv_s = (trans_s / foot_s * 100.0) if foot_s > 0 else np.nan
    sqm_eff_store = pd.to_numeric(region_stores.loc[region_stores["id_int"] == chosen_id, "sqm_effective"], errors="coerce")
    sqm_eff_store = float(sqm_eff_store.iloc[0]) if (sqm_eff_store is not None and not sqm_eff_store.empty and pd.notna(sqm_eff_store.iloc[0])) else np.nan
    spm2_s = (turn_s / sqm_eff_store) if (pd.notna(sqm_eff_store) and sqm_eff_store > 0) else np.nan
    cap_store = store_capture.get(int(chosen_id), np.nan)

    store_vals = compute_driver_values_from_period(
        footfall=foot_s,
        turnover=turn_s,
        transactions=trans_s,
        sqm_sum=sqm_eff_store,
        capture_pct=cap_store,
    )

    stype = store_type_store if store_type_store else "Unknown"
    
    # Benchmarks for same store_type
    reg_type_row = reg_type_k[reg_type_k["store_type"] == stype].iloc[0].to_dict() if (not reg_type_k.empty and (reg_type_k["store_type"] == stype).any()) else {}
    com_type_row = com_type_k[com_type_k["store_type"] == stype].iloc[0].to_dict() if (not com_type_k.empty and (com_type_k["store_type"] == stype).any()) else {}
    
    # Values (store)
    store_spv = store_vals.get("sales_per_visitor", np.nan)
    store_cr  = store_vals.get("conversion_rate", np.nan)
    store_atv = store_vals.get("sales_per_transaction", np.nan)
    store_spm2 = safe_div(turn_s, sqm_eff_store) if pd.notna(sqm_eff_store) and sqm_eff_store > 0 else np.nan
    store_cap = cap_store
    
    def idx_vs(a, b):
        return (a/b*100.0) if (pd.notna(a) and pd.notna(b) and float(b) != 0.0) else np.nan
    
    # Region same-type
    reg_spv = reg_type_row.get("sales_per_visitor", np.nan)
    reg_cr  = reg_type_row.get("conversion_rate", np.nan)
    reg_atv = reg_type_row.get("sales_per_transaction", np.nan)
    reg_spm2 = reg_type_row.get("sales_per_sqm", np.nan)
    reg_cap = reg_type_row.get("capture_rate", np.nan)
    
    # Company same-type
    com_spv = com_type_row.get("sales_per_visitor", np.nan)
    com_cr  = com_type_row.get("conversion_rate", np.nan)
    com_atv = com_type_row.get("sales_per_transaction", np.nan)
    com_spm2 = com_type_row.get("sales_per_sqm", np.nan)
    com_cap = com_type_row.get("capture_rate", np.nan)
    
    # ----------------------
    # Store vs same store_type (PFM heatmap tables)
    # ----------------------
    st.markdown('<div class="panel"><div class="panel-title">This store vs same store type</div>', unsafe_allow_html=True)
    
    def _idx_style(v):
        """PFM-brand heatmap styling for index values (percent vs benchmark)."""
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
    
    def _as_pct(x):
        return "-" if pd.isna(x) else f"{float(x):.0f}%"
    
    # Build index rows (numeric) once, reuse for both tables
    row_vs_region = {
        "SPV idx": idx_vs(store_spv, reg_spv),
        "CR idx": idx_vs(store_cr, reg_cr),
        "Sales/m² idx": idx_vs(store_spm2, reg_spm2),
        "ATV idx": idx_vs(store_atv, reg_atv),
        "Capture idx": np.nan if pd.isna(store_cap) or pd.isna(reg_cap) else idx_vs(store_cap, reg_cap),
    }
    row_vs_company = {
        "SPV idx": idx_vs(store_spv, com_spv),
        "CR idx": idx_vs(store_cr, com_cr),
        "Sales/m² idx": idx_vs(store_spm2, com_spm2),
        "ATV idx": idx_vs(store_atv, com_atv),
        "Capture idx": np.nan if pd.isna(store_cap) or pd.isna(reg_cap) else idx_vs(store_cap, reg_cap),
    }
    
    df_vs_region = pd.DataFrame([row_vs_region])
    df_vs_company = pd.DataFrame([row_vs_company])
    
    c1, c2 = st.columns(2, vertical_alignment="top")
    
    with c1:
        st.markdown("**vs Region (same store type)**")
        sty = (
            df_vs_region.style
            .applymap(_idx_style, subset=df_vs_region.columns.tolist())
            .format({c: _as_pct for c in df_vs_region.columns})
        )
        st.dataframe(sty, use_container_width=True, hide_index=True)
    
    with c2:
        st.markdown("**vs Company (same store type)**")
        sty = (
            df_vs_company.style
            .applymap(_idx_style, subset=df_vs_company.columns.tolist())
            .format({c: _as_pct for c in df_vs_company.columns})
        )
        st.dataframe(sty, use_container_width=True, hide_index=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    store_weights = get_svi_weights_for_store_type(store_type_store)

    # --- SQM calibration: compare apples to apples (size-adjusted benchmark) ---
    try:
        if isinstance(region_stores, pd.DataFrame) and ('sqm_effective' in region_stores.columns):
            if ('store_type' in region_stores.columns) and pd.notna(store_type_store):
                bench_sqm_series = region_stores.loc[region_stores['store_type'] == store_type_store, 'sqm_effective']
            else:
                bench_sqm_series = region_stores['sqm_effective']
        else:
            bench_sqm_series = None
    except Exception:
        bench_sqm_series = None

    store_svi, store_avg_ratio, store_bd = compute_svi_explainable(
        vals_a=store_vals,
        vals_b=reg_bench,
        floor=float(lever_floor),
        cap=float(lever_cap),
        weights=store_weights,
        store_sqm=sqm_eff_store,
        benchmark_sqm_series=bench_sqm_series,
        sqm_calibrate=True,
    )
    store_status, store_status_color = status_from_score(store_svi if pd.notna(store_svi) else 0)

    sk1, sk2, sk3, sk4, sk5 = st.columns([1, 1, 1, 1, 1])
    with sk1:
        kpi_card("Footfall", fmt_int(foot_s), "Store · selected period")
    with sk2:
        kpi_card("Revenue", fmt_eur(turn_s), "Store · selected period")
    with sk3:
        kpi_card("Conversion", fmt_pct(conv_s), "Store · selected period")
    with sk4:
        kpi_card("Sales / m²", fmt_eur(spm2_s), "Store · selected period")
    with sk5:
        kpi_card("Store SVI", "-" if pd.isna(store_svi) else f"{store_svi:.0f} / 100", "vs region benchmark")

    st.markdown(
        f"<div class='muted'>Status: <span style='font-weight:900;color:{store_status_color}'>{store_status}</span> · "
        f"Weighted ratio vs region ≈ <b>{'' if pd.isna(store_avg_ratio) else f'{store_avg_ratio:.0f}%'}</b> · "
        f"Capture weight: <b>{store_weights.get('capture_rate', 0):.2f}</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="panel"><div class="panel-title">Store SVI breakdown (vs region)</div>', unsafe_allow_html=True)
    bd2 = store_bd.copy()
    bd2["ratio_pct"] = pd.to_numeric(bd2["ratio_pct"], errors="coerce")
    bd2 = bd2.dropna(subset=["ratio_pct"])
    if bd2.empty:
        st.info("No breakdown available for this store (missing drivers).")
    else:
        bd2_show = bd2.copy()
        bd2_show["Ratio vs region"] = bd2_show["ratio_pct"].apply(lambda x: "-" if pd.isna(x) else f"{x:.0f}%")
        bd2_show["Weight"] = bd2_show["weight"].apply(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
        bd2_show = bd2_show[["driver", "Ratio vs region", "Weight"]].rename(columns={"driver": "Driver"})
        st.dataframe(bd2_show, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # Debug
    # ----------------------
    with st.expander("🔧 Debug (v2)"):
        st.write("REPORT_URL:", REPORT_URL)
        st.write("Company:", company_id)
        st.write("Region:", region_choice)
        st.write("Period:", start_period, "→", end_period)
        st.write("SVI floor/cap:", lever_floor, lever_cap)
        st.write("BASE_SVI_WEIGHTS:", BASE_SVI_WEIGHTS)
        st.write("Region dominant store_type:", dominant_store_type)
        st.write("Region weights:", region_weights)
        st.write("Reg bench:", reg_vals)
        st.write("Company vals:", comp_vals)
        st.write("Pathzz mtime:", pz_mtime)
        st.write("Pathzz file exists:", os.path.exists("data/pathzz_sample_weekly.csv"))
        st.write("df_daily_store cols:", df_daily_store.columns.tolist())

# Streamlit multipage: call once.
main()