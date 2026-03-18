# pages/06B_Region_Copilot_Screen.py

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

from datetime import datetime, timedelta, date

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response
from helpers_periods import period_catalog
from helpers_vemcount_api import VemcountApiConfig, fetch_report, build_report_params

from services.cbs_service import (
    get_cci_series,
    get_retail_index,
)
from services.svi_service import build_store_vitality

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="PFM Region Copilot (One Screen)",
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
# Minimal CSS
# ----------------------
st.markdown(
    f"""
    <style>
      .block-container {{
        padding-top: 2.25rem;
        padding-bottom: 2rem;
      }}
      .pfm-header {{
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding: 0.75rem 1rem;
        border: 1px solid {PFM_LINE};
        border-radius: 14px;
        background: white;
        margin-bottom: 0.75rem;
      }}
      .pfm-title {{
        font-size: 1.25rem;
        font-weight: 800;
        color: {PFM_DARK};
      }}
      .pfm-sub {{
        color: {PFM_GRAY};
        font-size: 0.9rem;
        margin-top: 0.15rem;
      }}
      .kpi-card {{
        border: 1px solid {PFM_LINE};
        border-radius: 14px;
        background: white;
        padding: 0.85rem 1rem;
      }}
      .kpi-label {{
        color: {PFM_GRAY};
        font-size: 0.85rem;
        font-weight: 600;
      }}
      .kpi-value {{
        color: {PFM_DARK};
        font-size: 1.45rem;
        font-weight: 900;
        margin-top: 0.2rem;
      }}
      .kpi-help {{
        color: {PFM_GRAY};
        font-size: 0.8rem;
        margin-top: 0.25rem;
      }}
      .panel {{
        border: 1px solid {PFM_LINE};
        border-radius: 14px;
        background: white;
        padding: 0.75rem 1rem;
      }}
      .panel-title {{
        font-weight: 800;
        color: {PFM_DARK};
        margin-bottom: 0.5rem;
      }}
      div.stButton > button {{
        background: {PFM_RED} !important;
        color: white !important;
        border: 0px !important;
        border-radius: 12px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 800 !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

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

    # store_type bestaat, maar is (nog) leeg bij jou -> we laten hem gewoon doorlopen
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

@st.cache_data(ttl=600)
def fetch_region_street_traffic(region: str, start_date, end_date) -> pd.DataFrame:
    csv_path = "data/pathzz_sample_weekly.csv"
    try:
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

    df.columns = ["region", "week", "street_footfall"]
    df["region"] = df["region"].astype(str).str.strip()
    region_norm = str(region).strip().lower()
    df = df[df["region"].str.lower() == region_norm].copy()
    if df.empty:
        return pd.DataFrame()

    df["street_footfall"] = df["street_footfall"].astype(str).str.strip().replace("", np.nan)
    df = df.dropna(subset=["street_footfall"])

    df["street_footfall"] = (
        df["street_footfall"]
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

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df["week_start"] >= start) & (df["week_start"] <= end)]

    return df[["week_start", "street_footfall"]].reset_index(drop=True)

# ----------------------
# Robust helpers
# ----------------------
def ensure_region_column(df: pd.DataFrame, merged_map: pd.DataFrame, store_key_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "region" in df.columns:
        return df

    for cand in ("region_x", "region_y"):
        if cand in df.columns:
            out = df.copy()
            out["region"] = out[cand]
            return out

    if merged_map is None or merged_map.empty:
        return df
    if "id" not in merged_map.columns or "region" not in merged_map.columns:
        return df

    region_lookup = merged_map[["id", "region"]].drop_duplicates().copy()
    out = df.copy()

    if store_key_col in out.columns:
        out[store_key_col] = pd.to_numeric(out[store_key_col], errors="coerce").astype("Int64")
        region_lookup["id"] = pd.to_numeric(region_lookup["id"], errors="coerce").astype("Int64")
        out = out.merge(region_lookup, left_on=store_key_col, right_on="id", how="left")
        return out

    if "id" in out.columns:
        out["id"] = pd.to_numeric(out["id"], errors="coerce").astype("Int64")
        region_lookup["id"] = pd.to_numeric(region_lookup["id"], errors="coerce").astype("Int64")
        out = out.merge(region_lookup, on="id", how="left")
        return out

    return df

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
        .mark_arc(innerRadius=54, outerRadius=70)
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
# KPI computations + de-duplication
# ----------------------
def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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
        out["avg_basket_size"] = np.where(out["transactions"] > 0, out["turnover"] / out["transactions"], np.nan)
        out["sales_per_transaction"] = out["avg_basket_size"]

    return out

def aggregate_weekly_region(df_region_daily: pd.DataFrame) -> pd.DataFrame:
    if df_region_daily is None or df_region_daily.empty:
        return pd.DataFrame()

    df = df_region_daily.copy()
    df["week_start"] = df["date"].dt.to_period("W-SAT").dt.start_time

    agg = {}
    if "footfall" in df.columns:
        agg["footfall"] = "sum"
    if "turnover" in df.columns:
        agg["turnover"] = "sum"
    if "transactions" in df.columns:
        agg["transactions"] = "sum"

    out = df.groupby("week_start", as_index=False).agg(agg)

    if "turnover" in out.columns and "footfall" in out.columns:
        out["sales_per_visitor"] = np.where(out["footfall"] > 0, out["turnover"] / out["footfall"], np.nan)
    if "transactions" in out.columns and "footfall" in out.columns:
        out["conversion_rate"] = np.where(out["footfall"] > 0, out["transactions"] / out["footfall"] * 100.0, np.nan)

    return out

# ----------------------
# sqm enrichment from report response
# ----------------------
def enrich_merged_with_sqm_from_df_norm(merged: pd.DataFrame, df_norm: pd.DataFrame, store_key_col: str) -> pd.DataFrame:
    """
    df_norm kan sqm bevatten (bv 'sq_meter'). Locations endpoint heeft het bij jou soms niet.
    We pakken per shop de eerste niet-null sqm en mergen die terug op merged.
    """
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

# ----------------------
# Opportunity logic (explainable)
# ----------------------
def build_opportunities(
    df_store_period: pd.DataFrame,
    svi_region: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    period_days = (end_ts - start_ts).days + 1
    year_factor = 365.0 / period_days if period_days > 0 else 1.0

    if df_store_period is None or df_store_period.empty:
        return pd.DataFrame()

    needed = ["id", "store_display", "footfall", "turnover", "sales_per_visitor", "region"]
    for c in needed:
        if c not in df_store_period.columns:
            return pd.DataFrame()

    store_agg = (
        df_store_period
        .groupby(["id", "store_display", "region"], as_index=False)
        .agg(
            footfall=("footfall", "sum"),
            turnover=("turnover", "sum"),
            spv=("sales_per_visitor", "mean"),
        )
    )

    store_agg["spv"] = pd.to_numeric(store_agg["spv"], errors="coerce")
    store_agg["footfall"] = pd.to_numeric(store_agg["footfall"], errors="coerce")
    store_agg["turnover"] = pd.to_numeric(store_agg["turnover"], errors="coerce")

    if svi_region is not None and not svi_region.empty:
        use_cols = ["id", "svi_score", "svi_status", "reason_short", "profit_potential_period"]
        use_cols = [c for c in use_cols if c in svi_region.columns]
        svi_min = svi_region[use_cols].drop_duplicates("id")
        store_agg = store_agg.merge(svi_min, on="id", how="left")

    store_agg["profit_potential_period"] = pd.to_numeric(
        store_agg.get("profit_potential_period", np.nan),
        errors="coerce"
    )
    store_agg["profit_potential_year"] = store_agg["profit_potential_period"] * year_factor

    all_zeroish = store_agg["profit_potential_year"].fillna(0).abs().sum() < 1e-6

    if all_zeroish:
        spv_vals = store_agg["spv"].dropna()
        if spv_vals.empty:
            return pd.DataFrame()

        benchmark_spv = float(spv_vals.quantile(0.75))
        store_agg["benchmark_spv"] = benchmark_spv
        store_agg["uplift_spv"] = np.maximum(0.0, store_agg["benchmark_spv"] - store_agg["spv"])
        store_agg["profit_potential_period"] = store_agg["uplift_spv"] * store_agg["footfall"]
        store_agg["profit_potential_year"] = store_agg["profit_potential_period"] * year_factor

        store_agg["opportunity_driver"] = np.where(
            store_agg["uplift_spv"] > 0,
            "SPV uplift to top quartile",
            "—"
        )
        if "reason_short" not in store_agg.columns:
            store_agg["reason_short"] = np.nan
    else:
        store_agg["opportunity_driver"] = np.where(
            store_agg["profit_potential_year"].fillna(0) > 0,
            "SVI profit potential",
            "—"
        )

    store_agg = store_agg.replace([np.inf, -np.inf], np.nan)
    return store_agg

# ----------------------
# MAIN
# ----------------------
def main():
    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    header_left, header_right = st.columns([2.2, 1.8])

    with header_left:
        st.markdown(
            f"""
            <div class="pfm-header">
              <div>
                <div class="pfm-title">PFM Region Performance Copilot</div>
                <div class="pfm-sub">One-screen layout – snel lezen, weinig scrollen (macro optioneel onderaan)</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)
    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1,
    )

    today = datetime.today().date()
    periods = period_catalog(today)
    period_labels = list(periods.keys())

    with header_right:
        c1, c2 = st.columns(2)
        with c1:
            client_label = st.selectbox("Retailer", clients_df["label"].tolist(), label_visibility="collapsed")
        with c2:
            period_choice = st.selectbox(
                "Periode",
                period_labels,
                index=period_labels.index("Q3 2024") if "Q3 2024" in period_labels else 0,
                label_visibility="collapsed",
            )

    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    # Load locations + regions
    try:
        locations_df = get_locations_by_company(company_id)
    except requests.exceptions.RequestException as e:
        st.error(f"Fout bij ophalen van winkels uit FastAPI: {e}")
        return

    if locations_df.empty:
        st.error("Geen winkels gevonden voor deze retailer.")
        return

    region_map = load_region_mapping()
    if region_map.empty:
        st.error("Geen geldige data/regions.csv gevonden (minimaal: shop_id;region).")
        return

    locations_df["id"] = pd.to_numeric(locations_df["id"], errors="coerce").astype("Int64")
    merged = locations_df.merge(region_map, left_on="id", right_on="shop_id", how="inner")
    if merged.empty:
        st.warning("Er zijn geen winkels met een regio-mapping voor deze retailer.")
        return

    # store display name
    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        merged["store_display"] = merged["name"] if "name" in merged.columns else merged["id"].astype(str)

    # Controls
    available_regions = sorted(merged["region"].dropna().unique().tolist())
    top_controls = st.columns([1.3, 1.3, 1.2, 1.4])
    with top_controls[0]:
        region_choice = st.selectbox("Regio", available_regions)
    with top_controls[1]:
        compare_all_regions = st.toggle("Vergelijk met andere regio’s", value=True)
    with top_controls[2]:
        show_macro = st.toggle("Toon macro (CBS/CCI)", value=False)
    with top_controls[3]:
        run_btn = st.button("Analyseer", type="primary")

    if not run_btn:
        st.info("Selecteer retailer/regio/periode en klik op **Analyseer**.")
        return

    # ✅ use PeriodDef from helpers_periods
    start_period = periods[period_choice].start
    end_period = periods[period_choice].end
    macro_year = periods[period_choice].macro_year

    start_ts = pd.Timestamp(start_period)
    end_ts = pd.Timestamp(end_period)

    region_shops = merged[merged["region"] == region_choice].copy()
    region_shop_ids = region_shops["id"].dropna().astype(int).unique().tolist()
    if not region_shop_ids:
        st.warning(f"Geen winkels gevonden voor regio '{region_choice}'.")
        return

    all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()
    fetch_ids = all_shop_ids if compare_all_regions else region_shop_ids

    # Keep EXACTLY as you had (stable)
    metric_map = {"count_in": "footfall", "turnover": "turnover"}

    # Use helpers_vemcount_api
    cfg = VemcountApiConfig(report_url=REPORT_URL)

    params_preview = build_report_params(
        shop_ids=fetch_ids,
        data_outputs=list(metric_map.keys()),
        period="date",
        step="day",
        source="shops",
        date_from=start_period,
        date_to=end_period,
    )

    with st.spinner("Data ophalen via FastAPI..."):
        try:
            resp = fetch_report(
                cfg=cfg,
                shop_ids=fetch_ids,
                data_outputs=list(metric_map.keys()),
                period="date",
                step="day",
                source="shops",
                date_from=start_period,
                date_to=end_period,
                timeout=120,
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTPError bij /get-report: {e}")
            try:
                st.code(e.response.text)
            except Exception:
                pass
            with st.expander("🔧 Debug request (params)"):
                st.write("REPORT_URL:", REPORT_URL)
                st.write("Params:", params_preview)
                st.write("Period choice:", period_choice)
                st.write("start/end:", start_period, end_period)
            return
        except requests.exceptions.RequestException as e:
            st.error(f"❌ RequestException bij /get-report: {e}")
            with st.expander("🔧 Debug request (params)"):
                st.write("REPORT_URL:", REPORT_URL)
                st.write("Params:", params_preview)
                st.write("Period choice:", period_choice)
                st.write("start/end:", start_period, end_period)
            return

    # normalize + rename
    df_norm = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)

    if df_norm is None or df_norm.empty:
        st.warning("Geen data ontvangen voor de gekozen selectie.")
        with st.expander("🔧 Debug response keys"):
            if isinstance(resp, dict):
                st.write("Top-level keys:", list(resp.keys()))
                st.write("Response snippet:", {k: resp[k] for k in list(resp.keys())[:5]})
            else:
                st.write("Response type:", type(resp))
            st.write("Params:", params_preview)
        return

    store_key_col = None
    for cand in ["shop_id", "id", "location_id"]:
        if cand in df_norm.columns:
            store_key_col = cand
            break
    if store_key_col is None:
        st.error("Geen store-id kolom gevonden in de response (shop_id/id/location_id).")
        with st.expander("🔧 Debug df_norm columns"):
            st.write(df_norm.columns.tolist())
        return

    # ✅ sqm fix: verrijk merged met sqm uit df_norm en herbouw sqm_effective
    merged = enrich_merged_with_sqm_from_df_norm(merged, df_norm, store_key_col=store_key_col)

    # sqm_effective: override > locations sqm > sqm_api
    sqm_col = None
    for cand in ["sqm", "sq_meter", "sq_meters", "square_meters"]:
        if cand in merged.columns:
            sqm_col = cand
            break

    base_sqm = pd.to_numeric(merged[sqm_col], errors="coerce") if sqm_col is not None else np.nan
    sqm_api = pd.to_numeric(merged["sqm_api"], errors="coerce") if "sqm_api" in merged.columns else np.nan

    merged["sqm_effective"] = np.where(
        merged["sqm_override"].notna(),
        pd.to_numeric(merged["sqm_override"], errors="coerce"),
        np.where(pd.notna(base_sqm), base_sqm, sqm_api)
    )

    df_daily_store = collapse_to_daily_store(df_norm, store_key_col=store_key_col)
    if df_daily_store.empty:
        st.warning("Geen data na opschonen (daily/store collapse).")
        with st.expander("🔧 Debug df_norm head"):
            st.write(df_norm.head())
        return

    join_cols = ["id", "store_display", "region", "sqm_effective"]
    if "store_type" in merged.columns:
        join_cols.append("store_type")

    df_daily_store = df_daily_store.merge(
        merged[join_cols].drop_duplicates(),
        left_on=store_key_col,
        right_on="id",
        how="left",
    )

    df_region_daily = df_daily_store[df_daily_store["region"] == region_choice].copy()
    if df_region_daily.empty:
        st.warning("Geen data voor geselecteerde regio binnen de periode.")
        with st.expander("🔧 Debug join coverage"):
            st.write("Region choice:", region_choice)
            st.write("Unique regions in df_daily_store:", sorted(df_daily_store["region"].dropna().unique().tolist()))
            st.write(df_daily_store[[store_key_col, "id", "region", "store_display"]].head(20))
        return

    # KPIs
    foot_total = float(df_region_daily["footfall"].sum()) if "footfall" in df_region_daily.columns else 0.0
    turn_total = float(df_region_daily["turnover"].sum()) if "turnover" in df_region_daily.columns else 0.0

    spv_avg = np.nan
    if "turnover" in df_region_daily.columns and "footfall" in df_region_daily.columns:
        ff = df_region_daily["footfall"].sum()
        spv_avg = (df_region_daily["turnover"].sum() / ff) if ff > 0 else np.nan

    region_weekly = aggregate_weekly_region(df_region_daily)
    pathzz_weekly = fetch_region_street_traffic(region=region_choice, start_date=start_period, end_date=end_period)

    capture_weekly = pd.DataFrame()
    avg_capture = np.nan

    if not region_weekly.empty:
        region_weekly = region_weekly.copy()
        region_weekly["week_start"] = pd.to_datetime(region_weekly["week_start"], errors="coerce")
        region_weekly = region_weekly.dropna(subset=["week_start"])

        # ✅ force 1 row per week (region total)
        region_weekly = (
            region_weekly.groupby("week_start", as_index=False)
            .agg(
                footfall=("footfall", "sum"),
                turnover=("turnover", "sum") if "turnover" in region_weekly.columns else ("footfall", "sum"),
                transactions=("transactions", "sum") if "transactions" in region_weekly.columns else ("footfall", "sum"),
            )
        )

    if not pathzz_weekly.empty:
        pathzz_weekly = pathzz_weekly.copy()
        pathzz_weekly["week_start"] = pd.to_datetime(pathzz_weekly["week_start"], errors="coerce")
        pathzz_weekly = pathzz_weekly.dropna(subset=["week_start"])

        # ✅ CRUCIAAL: voorkom duplicates op week_start
        pathzz_weekly = (
            pathzz_weekly.groupby("week_start", as_index=False)
               .agg(street_footfall=("street_footfall", "mean"))
        )

    if not region_weekly.empty and not pathzz_weekly.empty:
        capture_weekly = pd.merge(region_weekly, pathzz_weekly, on="week_start", how="inner")

    if not capture_weekly.empty:
        capture_weekly["capture_rate"] = np.where(
            capture_weekly["street_footfall"] > 0,
            capture_weekly["footfall"] / capture_weekly["street_footfall"] * 100.0,
            np.nan,
        )
        avg_capture = float(capture_weekly["capture_rate"].mean())

    # SVI
    svi_all = build_store_vitality(
        df_period=df_daily_store,
        region_shops=merged,
        store_key_col=store_key_col,
    )
    svi_all = ensure_region_column(svi_all, merged, store_key_col) if isinstance(svi_all, pd.DataFrame) else pd.DataFrame()

    region_scores = pd.DataFrame()
    region_svi = np.nan
    region_status, region_color = ("-", PFM_LINE)

    if not svi_all.empty and "region" in svi_all.columns and "svi_score" in svi_all.columns:
        region_scores = (
            svi_all.groupby("region", as_index=False)["svi_score"]
            .mean()
            .rename(columns={"svi_score": "region_svi"})
            .dropna(subset=["region"])
        )
        cur = region_scores[region_scores["region"] == region_choice]
        if not cur.empty:
            region_svi = float(np.clip(cur["region_svi"].iloc[0], 0, 100))
            region_status, region_color = status_from_score(region_svi)

    svi_region = pd.DataFrame()
    if not svi_all.empty and "region" in svi_all.columns:
        svi_region = svi_all[svi_all["region"] == region_choice].copy()

    # Opportunities
    opp_base = build_opportunities(
        df_store_period=df_region_daily,
        svi_region=svi_region,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    # ----------------------
    # UI
    # ----------------------
    st.markdown(f"## {selected_client['brand']} — Regio **{region_choice}** · {start_period} → {end_period}")

    # KPI row
    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
    with k1:
        kpi_card("Footfall", fmt_int(foot_total), "Regio · periode")
    with k2:
        kpi_card("Omzet", fmt_eur(turn_total), "Regio · periode")
    with k3:
        kpi_card("SPV", (fmt_eur_2(spv_avg) if not pd.isna(spv_avg) else "-"), "Omzet / bezoeker (gewogen)")
    with k4:
        kpi_card("Capture", (fmt_pct(avg_capture) if not pd.isna(avg_capture) else "-"), "Regio totaal (Pathzz)")

    # ----------------------
    # Weekly + region compare + vitality gauge
    # ----------------------
    r2_a, r2_b, r2_c = st.columns([1.7, 1.05, 0.75])

    with r2_a:
        st.markdown('<div class="panel"><div class="panel-title">Weekly trend — Store vs Street + Capture</div>', unsafe_allow_html=True)

        if capture_weekly.empty:
            st.info("Geen matchende Pathzz-weekdata gevonden voor deze regio/periode.")
        else:
            chart_df = capture_weekly[["week_start", "footfall", "street_footfall", "turnover", "capture_rate"]].copy()
            chart_df = chart_df.sort_values("week_start")

            iso = chart_df["week_start"].dt.isocalendar()
            chart_df["week_label"] = iso.week.apply(lambda w: f"W{int(w):02d}")
            week_order = chart_df["week_label"].tolist()

            long = chart_df.melt(
                id_vars=["week_label"],
                value_vars=["footfall", "street_footfall", "turnover"],
                var_name="metric",
                value_name="value",
            )

            bar = (
                alt.Chart(long)
                .mark_bar(opacity=0.85, cornerRadiusEnd=4)
                .encode(
                    x=alt.X("week_label:N", sort=week_order, title=None),
                    xOffset=alt.XOffset("metric:N"),
                    y=alt.Y("value:Q", title=""),
                    color=alt.Color(
                        "metric:N",
                        scale=alt.Scale(
                            domain=["footfall", "street_footfall", "turnover"],
                            range=[PFM_PURPLE, PFM_LINE, PFM_RED],
                        ),
                        legend=alt.Legend(title=""),
                    ),
                    tooltip=[
                        alt.Tooltip("week_label:N", title="Week"),
                        alt.Tooltip("metric:N", title="Type"),
                        alt.Tooltip("value:Q", title="Waarde", format=",.0f"),
                    ],
                )
            )

            line = (
                alt.Chart(chart_df)
                .mark_line(point=True, strokeWidth=2, color=PFM_DARK)
                .encode(
                    x=alt.X("week_label:N", sort=week_order, title=None),
                    y=alt.Y("capture_rate:Q", title="Capture %"),
                    tooltip=[
                        alt.Tooltip("week_label:N", title="Week"),
                        alt.Tooltip("capture_rate:Q", title="Capture", format=".1f"),
                    ],
                )
            )

            st.altair_chart(
                alt.layer(bar, line).resolve_scale(y="independent").properties(height=260),
                use_container_width=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with r2_b:
        st.markdown('<div class="panel"><div class="panel-title">Regio vergelijking — RVI (SVI gemiddeld)</div>', unsafe_allow_html=True)

        if not compare_all_regions:
            st.info("Zet ‘Vergelijk met andere regio’s’ aan om alle regio’s te tonen.")
        elif region_scores.empty or region_scores["region"].nunique() <= 1:
            st.info("Nog onvoldoende regio’s of data om te vergelijken.")
        else:
            chart_regions = region_scores.copy()
            chart_regions["is_selected"] = chart_regions["region"] == region_choice

            region_chart = (
                alt.Chart(chart_regions.sort_values("region_svi", ascending=False))
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("region_svi:Q", title="RVI (0–100)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("region:N", sort="-x", title=None),
                    color=alt.Color(
                        "is_selected:N",
                        scale=alt.Scale(domain=[True, False], range=[PFM_PURPLE, PFM_LINE]),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("region:N", title="Regio"),
                        alt.Tooltip("region_svi:Q", title="RVI", format=".0f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(region_chart, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with r2_c:
        st.markdown('<div class="panel"><div class="panel-title">Regio Vitality</div>', unsafe_allow_html=True)
        if not pd.isna(region_svi):
            st.altair_chart(gauge_chart(region_svi, region_color), use_container_width=True)
            st.markdown(f"**{region_svi:.0f}** · {region_status}")
        else:
            st.info("Nog geen regio-score.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # Ranking + Opportunities
    # ----------------------
    r3_left, r3_right = st.columns([1.45, 1.15])

    with r3_left:
        st.markdown('<div class="panel"><div class="panel-title">Store Vitality ranking — geselecteerde regio</div>', unsafe_allow_html=True)

        if svi_region.empty:
            st.info("Geen stores in deze regio met SVI (of regio-koppeling ontbreekt).")
        else:
            tmp = svi_region.copy()
            if "svi_score" in tmp.columns:
                tmp["svi_score"] = pd.to_numeric(tmp["svi_score"], errors="coerce")
                tmp = tmp.dropna(subset=["svi_score"])

            if tmp.empty or "svi_score" not in tmp.columns:
                st.info("Geen valide SVI-scores gevonden.")
            else:
                y_col = "store_name" if "store_name" in tmp.columns else ("store_display" if "store_display" in tmp.columns else "id")

                chart_rank = (
                    alt.Chart(tmp.sort_values("svi_score", ascending=False).head(12))
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("svi_score:Q", title="SVI (0–100)", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y(f"{y_col}:N", sort="-x", title=None),
                        color=alt.Color(
                            "svi_status:N",
                            scale=alt.Scale(
                                domain=["High performance", "Good / stable", "Attention required", "Under pressure"],
                                range=[PFM_GREEN, PFM_PURPLE, PFM_AMBER, PFM_RED],
                            ),
                            legend=alt.Legend(title=""),
                        ),
                        tooltip=[
                            alt.Tooltip(f"{y_col}:N", title="Winkel"),
                            alt.Tooltip("svi_score:Q", title="SVI", format=".0f"),
                            alt.Tooltip("svi_status:N", title="Status"),
                            alt.Tooltip("reason_short:N", title="Waarom"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_rank, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with r3_right:
        st.markdown('<div class="panel"><div class="panel-title">Biggest opportunities</div>', unsafe_allow_html=True)

        if opp_base is None or opp_base.empty:
            st.info("Geen opportunity data beschikbaar.")
        else:
            opp = opp_base.copy()
            opp["profit_potential_year"] = pd.to_numeric(opp["profit_potential_year"], errors="coerce")
            if "profit_potential_period" in opp.columns:
                opp["profit_potential_period"] = pd.to_numeric(opp["profit_potential_period"], errors="coerce")

            opp = opp.dropna(subset=["profit_potential_year"])
            opp = opp[opp["profit_potential_year"] > 0].copy()

            if opp.empty:
                st.info("Geen positieve opportunity gevonden (mogelijk te weinig data / benchmark effect).")
            else:
                topn = opp.sort_values("profit_potential_year", ascending=False).head(6).copy()

                opp_chart = (
                    alt.Chart(topn)
                    .mark_bar(cornerRadiusEnd=4, color=PFM_RED)
                    .encode(
                        x=alt.X("profit_potential_year:Q", title="Indicatieve potentie (€ / jaar)", axis=alt.Axis(format=",.0f")),
                        y=alt.Y("store_display:N", sort="-x", title=None),
                        tooltip=[
                            alt.Tooltip("store_display:N", title="Winkel"),
                            alt.Tooltip("profit_potential_year:Q", title="Indicatieve potentie € / jaar", format=",.0f"),
                            alt.Tooltip("opportunity_driver:N", title="Driver"),
                            alt.Tooltip("reason_short:N", title="Toelichting"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(opp_chart, use_container_width=True)

                # ✅ NEW: period vs year + clear wording
                total_top5_period = float(topn["profit_potential_period"].head(5).sum()) if "profit_potential_period" in topn.columns else np.nan
                total_top5_year = float(topn["profit_potential_year"].head(5).sum())

                if not pd.isna(total_top5_period):
                    st.markdown(f"**Top 5 potentie (in gekozen periode):** {fmt_eur(total_top5_period)}")
                st.markdown(f"**Indicatief geannualiseerd:** {fmt_eur(total_top5_year)} / jaar")
                st.caption(
                    "Let op: dit is geen ‘extra omzet gegarandeerd’, maar een indicatie van upside als onderliggende drivers structureel verbeteren (SVI/benchmark)."
                )

                st.caption("Hoe dit wordt berekend:")
                if (topn.get("opportunity_driver") == "SVI profit potential").any():
                    st.caption("• Primair: `profit_potential_period` uit Store Vitality (SVI) → geannualiseerd voor vergelijkbaarheid.")
                else:
                    st.caption("• Fallback: uplift naar benchmark **SPV (top quartile)** × **footfall** → geannualiseerd.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # MACRO (CBS/CCI) optioneel
    # ----------------------
    if show_macro:
        st.markdown("## Macro-context (optioneel)")

        # 1 jaar terug vanaf START van de gekozen periode t/m einddatum
        macro_start = start_period - timedelta(days=365)
        macro_end = end_period

        st.caption(
            f"Macro toont: {macro_start} → {macro_end} (1 jaar terug vanaf start van je periode t/m einddatum)."
        )

        # --- 1) EXTRA FETCH: haal regio-shopdata op voor macro window ---
        macro_metric_map = {"count_in": "footfall", "turnover": "turnover"}

        macro_params_preview = build_report_params(
            shop_ids=region_shop_ids,
            data_outputs=list(macro_metric_map.keys()),
            period="date",
            step="day",
            source="shops",
            date_from=macro_start,
            date_to=macro_end,
        )

        resp_macro = None
        with st.spinner("Macro data ophalen (regio footfall/omzet) ..."):
            try:
                resp_macro = fetch_report(
                    cfg=cfg,
                    shop_ids=region_shop_ids,
                    data_outputs=list(macro_metric_map.keys()),
                    period="date",
                    step="day",
                    source="shops",
                    date_from=macro_start,
                    date_to=macro_end,
                    timeout=120,
                )
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Macro fetch faalde: {e}")
                with st.expander("🔧 Debug macro request (params)"):
                    st.write("REPORT_URL:", REPORT_URL)
                    st.write("Params:", macro_params_preview)

        # --- Region month aggregation (macro window) ---
        region_month = pd.DataFrame()

        if resp_macro:
            df_norm_macro = normalize_vemcount_response(
                resp_macro, kpi_keys=macro_metric_map.keys()
            ).rename(columns=macro_metric_map)

            if df_norm_macro is not None and not df_norm_macro.empty:
                macro_store_key = None
                for cand in ["shop_id", "id", "location_id"]:
                    if cand in df_norm_macro.columns:
                        macro_store_key = cand
                        break

                if macro_store_key:
                    df_macro_daily_store = collapse_to_daily_store(df_norm_macro, store_key_col=macro_store_key)

                    if df_macro_daily_store is not None and not df_macro_daily_store.empty:
                        df_macro_daily_store["date"] = pd.to_datetime(df_macro_daily_store["date"], errors="coerce")
                        df_macro_daily_store = df_macro_daily_store.dropna(subset=["date"])

                        # filter macro window (extra safety)
                        df_macro_daily_store = df_macro_daily_store[
                            (df_macro_daily_store["date"] >= pd.Timestamp(macro_start)) &
                            (df_macro_daily_store["date"] <= pd.Timestamp(macro_end))
                        ].copy()

                        region_month = df_macro_daily_store.copy()
                        region_month["month"] = region_month["date"].dt.to_period("M").dt.to_timestamp()
                        region_month = (
                            region_month.groupby("month", as_index=False)[["turnover", "footfall"]]
                            .sum()
                            .rename(columns={"turnover": "region_turnover", "footfall": "region_footfall"})
                        )

        # index helper
        def index_from_first_nonzero(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce").astype(float)
            nonzero = s.replace(0, np.nan).dropna()
            if nonzero.empty:
                return pd.Series(np.nan, index=s.index)
            base_val = float(nonzero.iloc[0])
            return (s / base_val) * 100.0

        if not region_month.empty:
            region_month["region_turnover_index"] = index_from_first_nonzero(region_month["region_turnover"])
            region_month["region_footfall_index"] = index_from_first_nonzero(region_month["region_footfall"])

        # --- 2) CBS / CCI ophalen + filter op macro window ---
        months_back = int(((macro_end.year - macro_start.year) * 12 + (macro_end.month - macro_start.month)) + 6)
        months_back = max(60, months_back)   # altijd minimaal 60 maanden terug
        months_back = min(240, months_back)  # safety cap: 20 jaar

        # -------- CBS Retail (robust) --------
        cbs_retail_month = pd.DataFrame()
        retail_series = []
        try:
            retail_series = get_retail_index(months_back=months_back)
        except Exception:
            retail_series = []

        if not retail_series:
            try:
                retail_series = get_retail_index(months_back=60)
            except Exception:
                retail_series = []

        if retail_series:
            raw = pd.DataFrame(retail_series)

            value_col = None
            for cand in ["retail_value", "value", "index", "retail_index"]:
                if cand in raw.columns:
                    value_col = cand
                    break

            period_col = None
            for cand in ["period", "periode", "month", "maand"]:
                if cand in raw.columns:
                    period_col = cand
                    break

            if value_col and period_col:
                raw["date"] = pd.to_datetime(
                    raw[period_col].astype(str).str[:4] + "-" + raw[period_col].astype(str).str[-2:] + "-01",
                    errors="coerce",
                )
                raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
                raw = raw.dropna(subset=["date", value_col])

                raw = raw[
                    (raw["date"] >= pd.Timestamp(macro_start.replace(day=1))) &
                    (raw["date"] <= pd.Timestamp(macro_end.replace(day=1)))
                ].copy()

                if not raw.empty:
                    cbs_retail_month = raw.groupby("date", as_index=False)[value_col].mean()
                    cbs_retail_month = cbs_retail_month.rename(columns={value_col: "retail_value"})

                    base = cbs_retail_month["retail_value"].replace(0, np.nan).dropna()
                    if not base.empty:
                        base = float(base.iloc[0])
                        cbs_retail_month["cbs_retail_index"] = (cbs_retail_month["retail_value"] / base) * 100.0

        # -------- CCI (robust) --------
        cci_df = pd.DataFrame()
        cci_series = []
        try:
            cci_series = get_cci_series(months_back=months_back)
        except Exception:
            cci_series = []

        if not cci_series:
            try:
                cci_series = get_cci_series(months_back=60)
            except Exception:
                cci_series = []

        if cci_series:
            raw = pd.DataFrame(cci_series)

            value_col = None
            for cand in ["cci", "value", "index", "cci_value"]:
                if cand in raw.columns:
                    value_col = cand
                    break

            period_col = None
            for cand in ["period", "periode", "month", "maand"]:
                if cand in raw.columns:
                    period_col = cand
                    break

            if value_col and period_col:
                raw["date"] = pd.to_datetime(
                    raw[period_col].astype(str).str[:4] + "-" + raw[period_col].astype(str).str[-2:] + "-01",
                    errors="coerce",
                )
                raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
                raw = raw.dropna(subset=["date", value_col])

                raw = raw[
                    (raw["date"] >= pd.Timestamp(macro_start.replace(day=1))) &
                    (raw["date"] <= pd.Timestamp(macro_end.replace(day=1)))
                ].copy()

                if not raw.empty:
                    cci_df = raw.groupby("date", as_index=False)[value_col].mean().rename(columns={value_col: "cci"})
                    base = cci_df["cci"].replace(0, np.nan).dropna()
                    if not base.empty:
                        base = float(base.iloc[0])
                        cci_df["cci_index"] = (cci_df["cci"] / base) * 100.0

        # ----------------------
        # ✅ Dual-axis macro chart helper (region on left, macro on right)
        # ✅ Update: macro line has legend item (CBS/CCI) via series+color encoding
        # ----------------------
        def _macro_chart_dual(
            region_month_df: pd.DataFrame,
            macro_df: pd.DataFrame,
            macro_series_name: str,
            title: str,
            macro_label: str,
        ):
            st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>', unsafe_allow_html=True)

            if region_month_df is None or region_month_df.empty:
                st.info("Geen regio maanddata beschikbaar.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            # region long
            reg = region_month_df.rename(columns={"month": "date"}).copy()
            region_long = []
            if "region_footfall_index" in reg.columns:
                a = reg[["date", "region_footfall_index"]].copy()
                a["series"] = "Regio footfall-index"
                a = a.rename(columns={"region_footfall_index": "value"})
                region_long.append(a)
            if "region_turnover_index" in reg.columns:
                b = reg[["date", "region_turnover_index"]].copy()
                b["series"] = "Regio omzet-index"
                b = b.rename(columns={"region_turnover_index": "value"})
                region_long.append(b)

            region_long_df = pd.concat(region_long, ignore_index=True) if region_long else pd.DataFrame()
            region_long_df["value"] = pd.to_numeric(region_long_df["value"], errors="coerce")

            # macro df must be: date, series, value
            m = pd.DataFrame()
            if macro_df is not None and not macro_df.empty:
                m = macro_df.copy()
                if "series" not in m.columns:
                    m["series"] = macro_series_name
                m["value"] = pd.to_numeric(m["value"], errors="coerce")

            # scales for independent y-axes
            reg_min = float(region_long_df["value"].min()) if not region_long_df.empty else 0.0
            reg_max = float(region_long_df["value"].max()) if not region_long_df.empty else 100.0
            reg_pad = (reg_max - reg_min) * 0.10 if reg_max > reg_min else 10.0

            mac_min = float(m["value"].min()) if not m.empty else 0.0
            mac_max = float(m["value"].max()) if not m.empty else 100.0
            mac_pad = (mac_max - mac_min) * 0.10 if mac_max > mac_min else 5.0

            base = alt.Chart(region_long_df.dropna(subset=["date", "value"]))

            region_chart = (
                base.mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X(
                        "date:T",
                        title="Maand",
                        axis=alt.Axis(format="%b %Y", labelAngle=0),
                    ),
                    y=alt.Y(
                        "value:Q",
                        title="Regio index (100 = start)",
                        scale=alt.Scale(domain=[reg_min - reg_pad, reg_max + reg_pad], zero=False),
                    ),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(
                            domain=["Regio footfall-index", "Regio omzet-index"],
                            range=[PFM_PURPLE, "#7FB7FF"],
                        ),
                        legend=alt.Legend(title="Regio"),
                    ),
                    tooltip=[
                        alt.Tooltip("date:T", title="Maand", format="%b %Y"),
                        alt.Tooltip("series:N", title="Reeks"),
                        alt.Tooltip("value:Q", title="Index", format=".1f"),
                    ],
                )
            )

            if m.empty:
                st.altair_chart(region_chart.properties(height=260), use_container_width=True)
                st.caption("Geen macro data beschikbaar om te plotten.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            macro_chart = (
                alt.Chart(m.dropna(subset=["date", "value"]))
                .mark_line(point=True, strokeDash=[6, 4], strokeWidth=2)
                .encode(
                    x=alt.X("date:T", title="Maand"),
                    y=alt.Y(
                        "value:Q",
                        title=macro_label,
                        scale=alt.Scale(domain=[mac_min - mac_pad, mac_max + mac_pad], zero=False),
                        axis=alt.Axis(orient="right"),
                    ),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(domain=[macro_series_name], range=["#000000"]),  # <- zwart
                        legend=alt.Legend(title="Macro"),
                    ),
                    tooltip=[
                        alt.Tooltip("date:T", title="Maand", format="%b %Y"),
                        alt.Tooltip("series:N", title="Reeks"),
                        alt.Tooltip("value:Q", title=macro_label, format=".1f"),
                    ],
                )
            )

            chart = alt.layer(region_chart, macro_chart).resolve_scale(y="independent").properties(height=260)
            st.altair_chart(chart, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Build macro dfs: (date, series, value)
        macro_df_cbs = pd.DataFrame()
        if not cbs_retail_month.empty and "cbs_retail_index" in cbs_retail_month.columns:
            macro_df_cbs = cbs_retail_month[["date", "cbs_retail_index"]].copy()
            macro_df_cbs["value"] = pd.to_numeric(macro_df_cbs["cbs_retail_index"], errors="coerce")
            macro_df_cbs["series"] = "CBS detailhandelindex"
            macro_df_cbs = macro_df_cbs[["date", "series", "value"]].dropna()

        macro_df_cci = pd.DataFrame()
        if not cci_df.empty and "cci_index" in cci_df.columns:
            macro_df_cci = cci_df[["date", "cci_index"]].copy()
            macro_df_cci["value"] = pd.to_numeric(macro_df_cci["cci_index"], errors="coerce")
            macro_df_cci["series"] = "Consumentenvertrouwen (CCI)"
            macro_df_cci = macro_df_cci[["date", "series", "value"]].dropna()

        macro_col1, macro_col2 = st.columns(2)
        with macro_col1:
            _macro_chart_dual(
                region_month_df=region_month,
                macro_df=macro_df_cbs,
                macro_series_name="CBS detailhandelindex",
                title="CBS detailhandelindex vs Regio",
                macro_label="CBS index (100 = start)",
            )
        with macro_col2:
            _macro_chart_dual(
                region_month_df=region_month,
                macro_df=macro_df_cci,
                macro_series_name="Consumentenvertrouwen (CCI)",
                title="Consumentenvertrouwen (CCI) vs Regio",
                macro_label="CCI index (100 = start)",
            )

        # --- Macro debug
        with st.expander("🔧 Debug macro (CBS/CCI)"):
            st.write("Macro window:", macro_start, "→", macro_end)
            st.write("months_back used:", months_back)

            st.write("Region_month rows:", len(region_month))
            if not region_month.empty:
                st.write("Region_month min/max:", region_month["month"].min(), region_month["month"].max())
                st.dataframe(region_month.head())

            st.write("CBS rows:", len(cbs_retail_month))
            if not cbs_retail_month.empty:
                st.write("CBS min/max:", cbs_retail_month["date"].min(), cbs_retail_month["date"].max())
                st.dataframe(cbs_retail_month.head())

            st.write("CCI rows:", len(cci_df))
            if not cci_df.empty:
                st.write("CCI min/max:", cci_df["date"].min(), cci_df["date"].max())
                st.dataframe(cci_df.head())

    # ----------------------
    # Debug
    # ----------------------
    with st.expander("🔧 Debug"):
        st.write("Retailer:", selected_client)
        st.write("Regio:", region_choice)
        st.write("Periode:", start_period, "→", end_period, f"({period_choice})")
        st.write("Macro year:", macro_year)
        st.write("Compare all regions:", compare_all_regions)
        st.write("Store key col:", store_key_col)
        st.write("All shops:", len(all_shop_ids), "Region shops:", len(region_shop_ids))

        st.write("REPORT_URL:", REPORT_URL)
        st.write("Params used:", params_preview)

        st.write("Locations columns:", locations_df.columns.tolist())
        st.write("Merged columns:", merged.columns.tolist())
        st.write("Detected base sqm column:", sqm_col)
        st.write("sqm_api non-null:", int(pd.to_numeric(merged.get("sqm_api", pd.Series(dtype=float)), errors="coerce").notna().sum()) if "sqm_api" in merged.columns else 0)
        st.write("sqm_override non-null:", int(merged["sqm_override"].notna().sum()) if "sqm_override" in merged.columns else 0)
        st.write("sqm_effective non-null:", int(pd.to_numeric(merged["sqm_effective"], errors="coerce").notna().sum()) if "sqm_effective" in merged.columns else 0)

        st.write("df_norm head:", df_norm.head())
        st.write("df_daily_store head:", df_daily_store.head())
        st.write("df_region_daily head:", df_region_daily.head())


if __name__ == "__main__":
    main()
