# pages/07_Store_Manager_Copilot.py
# ------------------------------------------------------------
# PFM Store Manager Copilot (v1) — built ON TOP of Region Copilot patterns
# - Minimal choices: Client + Store + Week (2024 only)
# - Uses SAME API_URL setup, SAME client loading (clients.json), SAME Vemcount helpers
# - Fetches FULL KPI set (same metric_map as Region tool)
# - Shows: Store SVI + driver bars + actions + leaderboards + MTD progress donut (demo target)
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

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


from datetime import date, timedelta

from helpers_clients import load_clients
from helpers_normalize import normalize_vemcount_response
from helpers_vemcount_api import VemcountApiConfig, fetch_report, build_report_params

from stylesheet import inject_css, pfm_altair

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="PFM Store Manager Copilot", layout="wide")

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

inject_css(
    PFM_PURPLE=PFM_PURPLE,
    PFM_RED=PFM_RED,
    PFM_DARK=PFM_DARK,
    PFM_GRAY=PFM_GRAY,
    PFM_LIGHT=PFM_LIGHT,
    PFM_LINE=PFM_LINE,
)

# ----------------------
# API URL / secrets setup (SAME AS REGION COPILOT)
# ----------------------
raw_api_url = st.secrets["API_URL"].rstrip("/")

if raw_api_url.endswith("/get-report"):
    REPORT_URL = raw_api_url
    FASTAPI_BASE_URL = raw_api_url.rsplit("/get-report", 1)[0]
else:
    FASTAPI_BASE_URL = raw_api_url
    REPORT_URL = raw_api_url + "/get-report"

# ----------------------
# Format helpers (match Region feel)
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

def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.1f}%".replace(".", ",")

def fmt_int(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.0f}".replace(",", ".")

def safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0.0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan

def norm_key(x: str) -> str:
    return str(x).strip().lower() if x is not None else ""

def fmt_delta_html(x):
    if pd.isna(x):
        return f"<span style='color:{PFM_GRAY};font-weight:800;'>vs region type: -</span>"
    color = PFM_PURPLE if x >= 0 else PFM_RED
    sign = "+" if x >= 0 else ""
    return f"<span style='color:{color};font-weight:900;'>vs region type: {sign}{x:.0f}%</span>"

def pct_change(a, b):
    # (a/b - 1) * 100
    if pd.isna(a) or pd.isna(b) or float(b) == 0:
        return np.nan
    return (float(a) / float(b) - 1.0) * 100.0

# ----------------------
# UI helpers (same KPI card pattern)
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

# ----------------------
# Regions & locations (same as Region Copilot)
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
# Data shaping (ported from Region Copilot essentials)
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

# ----------------------
# SVI logic (aligned with Region Copilot)
# ----------------------
def idx_vs(a, b):
    return (a / b * 100.0) if (pd.notna(a) and pd.notna(b) and float(b) != 0.0) else np.nan

# ----------------------
# Week helpers (2024 only)
# ----------------------
def iso_week_range(year: int, week: int):
    d = date.fromisocalendar(year, week, 1)
    return d, d + timedelta(days=6)

def make_week_options_2024():
    opts = []
    for w in range(1, 53):
        s, e = iso_week_range(2024, w)
        label = f"2024-W{w:02d} ({s.strftime('%b %d')}–{e.strftime('%b %d')})"
        opts.append((label, s, e, w))
    return opts

# ----------------------
# Actions (improved)
# ----------------------
def make_actions(store_bd: pd.DataFrame, exclude_capture: bool = True, top_n: int = 3):
    """
    store_bd = breakdown table from compute_svi_explainable (vs chosen benchmark)
    Picks top actions based on weighted gap (100 - ratio_pct) * weight.
    - Only considers ratios below 100% (real underperformance).
    - Optionally excludes capture_rate as primary action (default True).
    Returns: list[dict]
    """
    if store_bd is None or store_bd.empty:
        return []

    df = store_bd.copy()

    # Robust numeric
    df["ratio_pct"] = pd.to_numeric(df.get("ratio_pct", np.nan), errors="coerce")
    df["weight"] = pd.to_numeric(df.get("weight", np.nan), errors="coerce").fillna(0.0)

    # Keep only usable rows
    df = df.dropna(subset=["ratio_pct"]).copy()
    if df.empty:
        return []

    # Only real gaps
    df = df[df["ratio_pct"] < 100.0].copy()
    if df.empty:
        return []

    # Optional: don't push capture as main action
    if exclude_capture and "driver_key" in df.columns:
        df = df[df["driver_key"].astype(str) != "capture_rate"].copy()
        if df.empty:
            return []

    # Priority score = gap * weight (gap in %points vs benchmark)
    df["gap_pp"] = (100.0 - df["ratio_pct"]).clip(lower=0.0)
    df["priority"] = df["gap_pp"] * df["weight"]

    df = df.sort_values(["priority", "gap_pp"], ascending=False)

    # Severity helper
    def _severity(gap_pp: float) -> str:
        if pd.isna(gap_pp):
            return "unknown"
        g = float(gap_pp)
        if g >= 20:
            return "large"
        if g >= 10:
            return "medium"
        return "small"

    # Slightly more concrete action copy
    action_map = {
        "conversion_rate": {
            "main": "Lift conversion: staff focus on greeting → need discovery → closing routines; reduce wait/queue friction.",
            "quick": "Quick win: test 1 ‘closing script’ + assign a greeter during peak 2 hours.",
        },
        "sales_per_transaction": {
            "main": "Increase ATV: bundles, add-ons, premium options, guided selling (make the upsell feel like help).",
            "quick": "Quick win: introduce 3 default bundles at the counter and coach staff for 1 week.",
        },
        "sales_per_visitor": {
            "main": "Lift SPV: improve conversion *and* ATV — this is the most direct lever on total revenue.",
            "quick": "Quick win: pick 1 hero product + 1 add-on and train ‘pairing’ suggestions.",
        },
        "sales_per_sqm": {
            "main": "Improve Sales/m²: optimize hot zones, layout, visibility, and promotions per m² (space must earn its rent).",
            "quick": "Quick win: relocate top sellers to entrance/hot zone and measure 2-week effect.",
        },
        "capture_rate": {
            "main": "Increase capture: window/entrance attraction, signage, local activation, and staffing on peak street-traffic moments.",
            "quick": "Quick win: refresh window messaging + add 1 outside CTA during peak hour.",
        },
    }

    out = []
    for _, r in df.head(int(top_n)).iterrows():
        key = str(r.get("driver_key", "")).strip()
        label = str(r.get("driver", key)).strip()

        gap_pp = float(r["gap_pp"]) if pd.notna(r["gap_pp"]) else np.nan
        ratio = float(r["ratio_pct"]) if pd.notna(r["ratio_pct"]) else np.nan
        w = float(r["weight"]) if pd.notna(r["weight"]) else 0.0

        sev = _severity(gap_pp)

        # choose main vs quick copy depending on severity
        copy = action_map.get(key, {"main": "Improve this driver to lift SVI.", "quick": "Run a small test to improve this driver."})
        action_txt = copy["main"] if sev in ("medium", "large") else copy.get("quick", copy["main"])

        out.append({
            "driver": label,
            "driver_key": key,
            "ratio_pct": ratio,
            "gap_pp": gap_pp,
            "severity": sev,
            "weight": w,
            "action": action_txt,
        })

    return out

# ----------------------
# Donut progress (Altair)
# ----------------------
def donut_progress(progress_pct: float):
    p = 0.0 if pd.isna(progress_pct) else float(progress_pct)
    p = max(0.0, min(120.0, p))

    df = pd.DataFrame({"label": ["Progress", "Remaining"], "value": [p, max(0.0, 100.0 - p)]})

    c = (
        alt.Chart(df)
        .mark_arc(innerRadius=55, outerRadius=75)
        .encode(
            theta="value:Q",
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=["Progress", "Remaining"], range=[PFM_PURPLE, PFM_LINE]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("label:N"), alt.Tooltip("value:Q", format=".0f")],
        )
        .properties(height=190)
    )

    txt = alt.Chart(pd.DataFrame({"t": [f"{p:.0f}%"]})).mark_text(
        fontSize=26, fontWeight=900, color=PFM_DARK
    ).encode(text="t:N")

    st.altair_chart((c + txt).configure_view(strokeWidth=0), use_container_width=True)

# ============================================================
# MAIN
# ============================================================
def main():
    # ---- session defaults ----
    if "sm_payload" not in st.session_state:
        st.session_state.sm_payload = None
    if "sm_last_key" not in st.session_state:
        st.session_state.sm_last_key = None
    if "sm_ran" not in st.session_state:
        st.session_state.sm_ran = False
    if "sm_store_id" not in st.session_state:
        st.session_state.sm_store_id = None
    if "sm_week_label" not in st.session_state:
        st.session_state.sm_week_label = make_week_options_2024()[26][0]  # mid-year default

    # ----------------------
    # Load clients (SAME AS REGION COPILOT)
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

    # ======================
    # ROW 1 — Title + Client + Run button (stacked right like Region tool)
    # ======================
    r1_left, r1_right = st.columns([3.6, 2.0], vertical_alignment="top")

    with r1_left:
        st.markdown(
            """
            <div class="pfm-header pfm-header--fixed">
              <div>
                <div class="pfm-title">PFM Store Manager Copilot <span class="pill">v1</span></div>
                <div class="pfm-sub">Store-level: SVI drivers → focus actions → weekly leaderboard → MTD progress</div>
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
                key="sm_client",
            )

        with c_btn:
            run_btn = st.button("Run analysis", type="primary", key="sm_run")

        st.markdown("</div>", unsafe_allow_html=True)

    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    # ----------------------
    # Load locations + region mapping (for store choices + defaults)
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
        st.warning("No stores matched your region mapping for this retailer.")
        return

    if "store_label" in merged.columns and merged["store_label"].notna().any():
        merged["store_display"] = merged["store_label"]
    else:
        merged["store_display"] = merged["name"] if "name" in merged.columns else merged["id"].astype(str)

    merged["store_type"] = (
        merged.get("store_type", "Unknown")
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    )

    store_dim = merged[["id", "store_display", "region", "store_type", "sqm_override"]].drop_duplicates().copy()

    store_dim["id_int"] = pd.to_numeric(store_dim["id"], errors="coerce").astype("Int64")
    store_dim = store_dim.dropna(subset=["id_int"]).copy()
    store_dim["id_int"] = store_dim["id_int"].astype(int)

    store_dim["dd_label"] = store_dim["store_display"].fillna(store_dim["id_int"].astype(str)) + " · " + store_dim["id_int"].astype(str)

    if store_dim.empty:
        st.error("No stores available after mapping (check regions.csv shop_id vs /company/{id}/location ids).")
        return

    # default store selection
    if st.session_state.sm_store_id is None or int(st.session_state.sm_store_id) not in store_dim["id_int"].tolist():
        st.session_state.sm_store_id = int(store_dim["id_int"].iloc[0])

    # week options (2024)
    weeks = make_week_options_2024()
    week_labels = [w[0] for w in weeks]
    if st.session_state.sm_week_label not in week_labels:
        st.session_state.sm_week_label = week_labels[26]

    # ======================
    # ROW 2 — Store + Week(2024) + Demo monthly target
    # ======================
    c_store, c_week, c_target = st.columns([2.2, 2.2, 1.6], vertical_alignment="bottom")

    with c_store:
        st.markdown('<div class="panel"><div class="panel-title">Store</div>', unsafe_allow_html=True)
        store_choice_label = st.selectbox(
            "Store",
            store_dim["dd_label"].tolist(),
            index=int(np.where(store_dim["id_int"].values == int(st.session_state.sm_store_id))[0][0]) if int(st.session_state.sm_store_id) in store_dim["id_int"].values else 0,
            key="sm_store_select",
            label_visibility="collapsed",
        )
        chosen_store_id = int(store_choice_label.split("·")[-1].strip())
        st.session_state.sm_store_id = chosen_store_id
        st.markdown("</div>", unsafe_allow_html=True)

    with c_week:
        st.markdown('<div class="panel"><div class="panel-title">Week (2024)</div>', unsafe_allow_html=True)
        week_label = st.selectbox(
            "Week",
            week_labels,
            index=week_labels.index(st.session_state.sm_week_label) if st.session_state.sm_week_label in week_labels else 0,
            key="sm_week_select",
            label_visibility="collapsed",
        )
        st.session_state.sm_week_label = week_label
        week_meta = [w for w in weeks if w[0] == week_label][0]
        wk_start, wk_end, wk_num = week_meta[1], week_meta[2], week_meta[3]
        st.markdown("</div>", unsafe_allow_html=True)

    with c_target:
        st.markdown('<div class="panel"><div class="panel-title">Demo target</div>', unsafe_allow_html=True)
        month_target = st.number_input(
            "Monthly revenue target (€)",
            min_value=0,
            value=250000,
            step=5000,
            key="sm_month_target",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # SVI sensitivity: keep simple but deterministic
    lever_floor = 80
    lever_cap = 200 - lever_floor

    run_key = (company_id, int(chosen_store_id), str(wk_start), str(wk_end), int(month_target))
    selection_changed = st.session_state.sm_last_key != run_key
    should_fetch = bool(run_btn) or bool(selection_changed) or (not bool(st.session_state.sm_ran))

    if run_btn:
        st.toast("Running store analysis…", icon="🚀")

    if (not should_fetch) and (st.session_state.sm_payload is None):
        st.info("Select store + week and click **Run analysis**.")
        return

    # ----------------------
    # FETCH (only when needed) — SAME KPI SET AS REGION COPILOT
    # ----------------------
    if should_fetch:
        all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()
        if not all_shop_ids:
            st.error("No shop IDs found after merging locations with regions.csv.")
            return

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
            date_from=wk_start,
            date_to=wk_end,
        )

        with st.spinner("Fetching week data via FastAPI..."):
            try:
                resp = fetch_report(
                    cfg=cfg,
                    shop_ids=all_shop_ids,
                    data_outputs=list(metric_map.keys()),
                    period="date",
                    step="day",
                    source="shops",
                    date_from=wk_start,
                    date_to=wk_end,
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
            st.warning("No data returned for the selected week.")
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
        sqm_api = pd.to_numeric(merged2.get("sqm_api", np.nan), errors="coerce")

        merged2["sqm_effective"] = np.where(
            merged2["sqm_override"].notna(),
            pd.to_numeric(merged2["sqm_override"], errors="coerce"),
            np.where(pd.notna(base_sqm), base_sqm, sqm_api)
        )

        df_daily_store = collapse_to_daily_store(df_norm, store_key_col=store_key_col)
        if df_daily_store is None or df_daily_store.empty:
            st.warning("No data after cleaning (daily/store collapse).")
            return

        join_cols = ["id", "store_display", "region", "sqm_effective", "store_type"]
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

        st.session_state.sm_last_key = run_key
        st.session_state.sm_payload = {
            "df_daily_store": df_daily_store,
            "merged": merged2,
            "store_key_col": store_key_col,
            "wk_start": wk_start,
            "wk_end": wk_end,
            "wk_num": wk_num,
            "selected_client": selected_client,
            "company_id": company_id,
            "report_url": REPORT_URL,
        }
        st.session_state.sm_ran = True

    # ----------------------
    # READ CACHE (ALWAYS)
    # ----------------------
    payload = st.session_state.sm_payload
    if payload is None:
        st.info("Select store + week and click **Run analysis**.")
        return

    df_daily_store = payload["df_daily_store"]
    merged2 = payload["merged"]
    wk_start = payload["wk_start"]
    wk_end = payload["wk_end"]
    wk_num = payload["wk_num"]

    # chosen store row
    chosen_id = int(st.session_state.sm_store_id)
    sd = merged2.copy()
    sd["id_int"] = pd.to_numeric(sd["id"], errors="coerce").astype("Int64")
    sd = sd.dropna(subset=["id_int"]).copy()
    sd["id_int"] = sd["id_int"].astype(int)

    if chosen_id not in sd["id_int"].tolist():
        st.error("Selected store not found in merged store dimension.")
        return

    store_row = sd[sd["id_int"] == chosen_id].iloc[0]
    store_name = str(store_row.get("store_display", chosen_id))
    store_type = str(store_row.get("store_type", "Unknown") or "Unknown")
    store_region = str(store_row.get("region", "") or "")
    sqm_eff_store = pd.to_numeric(store_row.get("sqm_effective", np.nan), errors="coerce")
    sqm_eff_store = float(sqm_eff_store) if pd.notna(sqm_eff_store) else np.nan

    st.markdown(f"## {selected_client['brand']} — **{store_name}** · storeID {chosen_id} <span class='pill'>{store_type}</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='muted'>Week {wk_num} · {wk_start} → {wk_end} · Default region: <b>{store_region}</b></div>", unsafe_allow_html=True)

    # ----------------------
    # Aggregate week per store
    # ----------------------
    d = df_daily_store.copy()
    d["id_int"] = pd.to_numeric(d["id"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["id_int"]).copy()
    d["id_int"] = d["id_int"].astype(int)

    agg = d.groupby(["id_int", "store_display", "region", "store_type"], as_index=False).agg(
        turnover=("turnover", "sum"),
        footfall=("footfall", "sum"),
        transactions=("transactions", "sum"),
        sqm_effective=("sqm_effective", "first"),
    )

    agg["conversion_rate"] = np.where(agg["footfall"] > 0, agg["transactions"] / agg["footfall"] * 100.0, np.nan)
    agg["sales_per_visitor"] = np.where(agg["footfall"] > 0, agg["turnover"] / agg["footfall"], np.nan)
    agg["sales_per_transaction"] = np.where(agg["transactions"] > 0, agg["turnover"] / agg["transactions"], np.nan)
    agg["sales_per_sqm"] = np.where(
        (pd.to_numeric(agg["sqm_effective"], errors="coerce") > 0) & pd.notna(pd.to_numeric(agg["sqm_effective"], errors="coerce")),
        agg["turnover"] / pd.to_numeric(agg["sqm_effective"], errors="coerce"),
        np.nan
    )

    s = agg[agg["id_int"] == chosen_id].copy()
    if s.empty:
        st.warning("No aggregated data for selected store in this week.")
        return

    foot_s = float(pd.to_numeric(s["footfall"], errors="coerce").fillna(0).sum())
    turn_s = float(pd.to_numeric(s["turnover"], errors="coerce").fillna(0).sum())
    trans_s = float(pd.to_numeric(s["transactions"], errors="coerce").fillna(0).sum())
    conv_s = (trans_s / foot_s * 100.0) if foot_s > 0 else np.nan
    spv_s = (turn_s / foot_s) if foot_s > 0 else np.nan
    atv_s = (turn_s / trans_s) if trans_s > 0 else np.nan
    spm2_s = (turn_s / sqm_eff_store) if (pd.notna(sqm_eff_store) and sqm_eff_store > 0) else np.nan

    # ----------------------
    # Benchmarks: same store_type (region + company)
    # + KPI benchmark dict for deltas in cards
    # ----------------------
    same_type_company = agg[agg["store_type"] == store_type].copy()
    same_type_region = agg[(agg["store_type"] == store_type) & (agg["region"] == store_region)].copy()

    # Fallbacks: voorkom lege benchmarks (komt vaak voor bij kleine retailers / nieuwe types)
    if same_type_region.empty:
        same_type_region = agg[agg["region"] == store_region].copy()
    if same_type_company.empty:
        same_type_company = agg.copy()

    def bench_from_df(df_in: pd.DataFrame) -> dict:
        if df_in is None or df_in.empty:
            return {
                "sales_per_visitor": np.nan,
                "sales_per_sqm": np.nan,
                "capture_rate": np.nan,
                "conversion_rate": np.nan,
                "sales_per_transaction": np.nan
            }

        ff = float(pd.to_numeric(df_in["footfall"], errors="coerce").dropna().sum())
        to = float(pd.to_numeric(df_in["turnover"], errors="coerce").dropna().sum())
        tr = float(pd.to_numeric(df_in["transactions"], errors="coerce").dropna().sum())

        sqm = pd.to_numeric(df_in["sqm_effective"], errors="coerce")
        # sqm per store is "first" in agg, dus duplicates per store vermijden via drop_duplicates op id_int
        # (agg is al per store, dus dit is safe; maar blijft netjes)
        sqm_sum = float(sqm.dropna().sum()) if sqm.notna().any() else np.nan

        return compute_driver_values_from_period(
            footfall=ff,
            turnover=to,
            transactions=tr,
            sqm_sum=sqm_sum,
            capture_pct=np.nan,
        )

    # driver benchmarks (voor SVI explainable)
    reg_bench_vals = bench_from_df(same_type_region)
    com_bench_vals = bench_from_df(same_type_company)

    def kpi_bench_from_df(df_in: pd.DataFrame) -> dict:
        """
        KPI benchmark used for KPI-card deltas.
        IMPORTANT: This must represent the 'typical store' in the cohort,
        so we use PER-STORE averages (not totals).
        df_in here is already at store-grain (agg), one row per store.
        """
        if df_in is None or df_in.empty:
            return {
                "footfall": np.nan,
                "turnover": np.nan,
                "transactions": np.nan,
                "conversion_rate": np.nan,
                "sales_per_visitor": np.nan,
                "sales_per_transaction": np.nan,
                "sales_per_sqm": np.nan,
            }
    
        tmp = df_in.copy()
    
        # ensure numeric
        for c in ["footfall", "turnover", "transactions", "conversion_rate",
                  "sales_per_visitor", "sales_per_transaction", "sales_per_sqm", "sqm_effective"]:
            if c in tmp.columns:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    
        # if derived metrics are missing (shouldn't be), recompute per-store
        if "conversion_rate" not in tmp.columns:
            tmp["conversion_rate"] = np.where(tmp["footfall"] > 0, tmp["transactions"] / tmp["footfall"] * 100.0, np.nan)
    
        if "sales_per_visitor" not in tmp.columns:
            tmp["sales_per_visitor"] = np.where(tmp["footfall"] > 0, tmp["turnover"] / tmp["footfall"], np.nan)
    
        if "sales_per_transaction" not in tmp.columns:
            tmp["sales_per_transaction"] = np.where(tmp["transactions"] > 0, tmp["turnover"] / tmp["transactions"], np.nan)
    
        if "sales_per_sqm" not in tmp.columns:
            tmp["sales_per_sqm"] = np.where(
                (tmp["sqm_effective"] > 0) & tmp["sqm_effective"].notna(),
                tmp["turnover"] / tmp["sqm_effective"],
                np.nan
            )
    
        # PER-STORE averages = "benchmark store"
        return {
            "footfall": float(tmp["footfall"].mean(skipna=True)) if "footfall" in tmp.columns else np.nan,
            "turnover": float(tmp["turnover"].mean(skipna=True)) if "turnover" in tmp.columns else np.nan,
            "transactions": float(tmp["transactions"].mean(skipna=True)) if "transactions" in tmp.columns else np.nan,
            "conversion_rate": float(tmp["conversion_rate"].mean(skipna=True)) if "conversion_rate" in tmp.columns else np.nan,
            "sales_per_visitor": float(tmp["sales_per_visitor"].mean(skipna=True)) if "sales_per_visitor" in tmp.columns else np.nan,
            "sales_per_transaction": float(tmp["sales_per_transaction"].mean(skipna=True)) if "sales_per_transaction" in tmp.columns else np.nan,
            "sales_per_sqm": float(tmp["sales_per_sqm"].mean(skipna=True)) if "sales_per_sqm" in tmp.columns else np.nan,
        }

    reg_bench = kpi_bench_from_df(same_type_region)   # <-- dit miste je
    com_bench = kpi_bench_from_df(same_type_company)  # (optioneel later)

    # Store vals (voor SVI)
    store_vals = compute_driver_values_from_period(
        footfall=foot_s,
        turnover=turn_s,
        transactions=trans_s,
        sqm_sum=sqm_eff_store,
        capture_pct=np.nan,
    )

    store_weights = get_svi_weights_for_store_type(store_type)

    # SVI vs region same-type (default) + company comparison
    store_svi_reg, store_avg_ratio_reg, store_bd_reg = compute_svi_explainable(
        vals_a=store_vals,
        vals_b=reg_bench_vals,
        floor=float(lever_floor),
        cap=float(lever_cap),
        weights=store_weights,
        store_sqm=sqm_eff_store,
        benchmark_sqm_series=same_type_region.get('sqm_effective'),
        sqm_calibrate=True,
    )
    # Backwards compat: some older UI blocks still refer to store_bd
    store_bd = store_bd_reg

    store_svi_com, store_avg_ratio_com, _ = compute_svi_explainable(
        vals_a=store_vals,
        vals_b=com_bench_vals,
        floor=float(lever_floor),
        cap=float(lever_cap),
        weights=store_weights,
        store_sqm=sqm_eff_store,
        benchmark_sqm_series=same_type_company.get('sqm_effective'),
        sqm_calibrate=True,
    )

    status_txt, status_color = status_from_score(store_svi_reg if pd.notna(store_svi_reg) else 0)

    # ----------------------
    # KPI header row
    # ----------------------
    # store KPI values (nogmaals: ok om hier opnieuw te zetten, maar mag ook weg)
    conv_s = safe_div(trans_s, foot_s) * 100.0 if foot_s > 0 else np.nan
    spv_s  = safe_div(turn_s, foot_s) if foot_s > 0 else np.nan
    atv_s  = safe_div(turn_s, trans_s) if trans_s > 0 else np.nan

    # region same-type benchmark KPI values (nu bestaat reg_bench echt)
    ff_b   = reg_bench.get("footfall", np.nan)
    rev_b  = reg_bench.get("turnover", np.nan)
    cr_b   = reg_bench.get("conversion_rate", np.nan)
    spv_b  = reg_bench.get("sales_per_visitor", np.nan)
    atv_b  = reg_bench.get("sales_per_transaction", np.nan)

    # deltas (%)
    d_ff  = pct_change(foot_s, ff_b)
    d_rev = pct_change(turn_s, rev_b)
    d_cr  = pct_change(conv_s, cr_b)
    d_spv = pct_change(spv_s, spv_b)
    d_atv = pct_change(atv_s, atv_b)
    
    k1, k2, k3, k4, k5, k6 = st.columns([1, 1, 1, 1, 1, 1])

    with k1:
        kpi_card("Footfall", fmt_int(foot_s), f"Week {wk_num} · 2024 · {fmt_delta_html(d_ff)}")
    with k2:
        kpi_card("Revenue", fmt_eur(turn_s), f"Week {wk_num} · 2024 · {fmt_delta_html(d_rev)}")
    with k3:
        kpi_card("Conversion", fmt_pct(conv_s), f"Transactions / Visitors · {fmt_delta_html(d_cr)}")
    with k4:
        kpi_card("SPV", fmt_eur(spv_s), f"Revenue / Visitor · {fmt_delta_html(d_spv)}")
    with k5:
        kpi_card("ATV", fmt_eur(atv_s), f"Revenue / Transaction · {fmt_delta_html(d_atv)}")
    with k6:
        kpi_card("Store SVI", "-" if pd.isna(store_svi_reg) else f"{store_svi_reg:.0f} / 100", "vs region same-type")

    st.markdown(
        f"<div class='muted'>Status: <span style='font-weight:900;color:{status_color}'>{status_txt}</span> · "
        f"Avg ratio vs region same-type ≈ <b>{'' if pd.isna(store_avg_ratio_reg) else f'{store_avg_ratio_reg:.0f}%'}</b> · "
        f"vs company same-type ≈ <b>{'' if pd.isna(store_avg_ratio_com) else f'{store_avg_ratio_com:.0f}%'}</b></div>",
        unsafe_allow_html=True,
    )

    # ======================
    # Drivers + Actions + MTD progress
    # ======================
    col_a, col_b, col_c = st.columns([2.3, 2.1, 1.6], vertical_alignment="top")

    with col_a:
        st.markdown(
            '<div class="panel"><div class="panel-title">SVI drivers — index vs region (same store type)</div>',
            unsafe_allow_html=True
        )
    
        bd = store_bd_reg.copy() if (store_bd_reg is not None and not store_bd_reg.empty) else pd.DataFrame()
    
        if not bd.empty:
            bd["ratio_pct"] = pd.to_numeric(bd.get("ratio_pct", np.nan), errors="coerce")
            bd["weight"] = pd.to_numeric(bd.get("weight", np.nan), errors="coerce")
            bd = bd.dropna(subset=["ratio_pct"]).copy()
    
        label_map = {
            "sales_per_visitor": "SPV",
            "sales_per_sqm": "Sales / m²",
            "capture_rate": "Capture",
            "conversion_rate": "Conversion",
            "sales_per_transaction": "ATV",
        }
    
        if (not bd.empty) and ("driver_key" not in bd.columns) and ("driver" in bd.columns):
            rev_map = {v: k for k, v in label_map.items()}
            bd["driver_key"] = bd["driver"].map(rev_map).fillna(bd["driver"])
    
        if not bd.empty:
            bd["driver_label"] = bd["driver_key"].map(label_map).fillna(bd["driver"])
            bd["ratio_clip"] = bd["ratio_pct"].clip(lower=60, upper=140)
    
        if bd.empty:
            st.info("No driver breakdown available.")
        else:
            # Zorg dat labels nooit leeg zijn
            bd["driver_label"] = bd["driver_label"].fillna("")
            
            order = ["SPV", "Sales / m²", "Capture", "Conversion", "ATV"]
            
            y_axis = alt.Axis(
                title=None,
                labels=True,
                labelLimit=200,
                labelPadding=10,
                labelFontSize=12,
                labelColor=PFM_GRAY,
                ticks=False,
                domain=False,
            )
            
            x_axis = alt.Axis(
                title="Index (100 = benchmark)",
                labelColor=PFM_GRAY,
                titleColor=PFM_GRAY,
                tickColor=PFM_LINE,
                gridColor=PFM_LINE,
            )
            
            bars = (
                alt.Chart(bd)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    y=alt.Y("driver_label:N", sort=order, axis=y_axis),
                    x=alt.X("ratio_clip:Q", axis=x_axis, scale=alt.Scale(domain=[60, 140])),
                    color=alt.condition(
                        alt.datum.ratio_pct >= 100,
                        alt.value(PFM_PURPLE),
                        alt.value(PFM_LINE),
                    ),
                    tooltip=[
                        alt.Tooltip("driver_label:N", title="Driver"),
                        alt.Tooltip("ratio_pct:Q", title="Index", format=".0f"),
                        alt.Tooltip("weight:Q", title="Weight", format=".2f"),
                    ],
                )
                .properties(height=210)
            )
            
            text = (
                alt.Chart(bd)
                .mark_text(align="left", dx=6, fontWeight=800, color=PFM_DARK)
                .encode(
                    y=alt.Y("driver_label:N", sort=order),
                    x=alt.X(
                        "ratio_clip:Q",
                        axis=x_axis,
                        scale=alt.Scale(domain=[60, 140], clamp=True)
                    ),
                    text=alt.Text("ratio_pct:Q", format=".0f"),
                )
            )
            

            chart = pfm_altair((bars + text), height=210)
            st.altair_chart(chart, use_container_width=True)

    with col_b:
        st.markdown('<div class="panel"><div class="panel-title">Focus actions (this week)</div>', unsafe_allow_html=True)

        actions = make_actions(store_bd_reg)
        if not actions:
            st.info("Not enough data to generate actions.")
        else:
            for i, a in enumerate(actions, start=1):
                idx = a["ratio_pct"]
                badge = f"{idx:.0f}%" if pd.notna(idx) else "-"
                color = PFM_RED if pd.notna(idx) and idx < 100 else PFM_PURPLE
                st.markdown(
                    f"""
                    <div class="callout" style="margin-bottom:0.6rem;">
                      <div class="callout-title">#{i} — {a["driver"]} <span class="pill" style="margin-left:0.4rem;border-color:{color};">{badge}</span></div>
                      <div class="callout-sub">{a["action"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="panel"><div class="panel-title">Month-to-date progress (demo)</div>', unsafe_allow_html=True)

        # MTD = from month start to end of selected week (wk_end)
        month_start = date(wk_start.year, wk_start.month, 1)
        cfg = VemcountApiConfig(report_url=REPORT_URL)

        with st.spinner("Fetching MTD revenue..."):
            try:
                resp_mtd = fetch_report(
                    cfg=cfg,
                    shop_ids=[chosen_id],
                    data_outputs=["turnover"],
                    period="date",
                    step="day",
                    source="shops",
                    date_from=month_start,
                    date_to=wk_end,
                    timeout=120,
                )
                df_mtd = normalize_vemcount_response(resp_mtd, kpi_keys=["turnover"]).rename(columns={"turnover": "turnover"})
            except Exception:
                df_mtd = pd.DataFrame()

        mtd = 0.0
        if df_mtd is not None and not df_mtd.empty and "turnover" in df_mtd.columns:
            mtd = float(pd.to_numeric(df_mtd["turnover"], errors="coerce").fillna(0).sum())

        progress_pct = (mtd / float(month_target) * 100.0) if float(month_target) > 0 else 0.0
        donut_progress(progress_pct)

        st.markdown(
            f"<div class='muted'>MTD revenue: <b>{fmt_eur(mtd)}</b><br/>Target: <b>{fmt_eur(month_target)}</b></div>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # Leaderboards
    # ======================
    st.markdown("## Weekly leaderboards")

    # helper: compute SVI for each store vs a chosen benchmark vals
    def svi_for_row(row, bench_vals, bench_sqm_series=None):
        vals = compute_driver_values_from_period(
            footfall=row["footfall"],
            turnover=row["turnover"],
            transactions=row["transactions"],
            sqm_sum=pd.to_numeric(row.get("sqm_effective", np.nan), errors="coerce"),
            capture_pct=np.nan,
        )
        w = get_svi_weights_for_store_type(row.get("store_type", "Unknown"))
        svi, avg_ratio, _ = compute_svi_explainable(
            vals_a=vals,
            vals_b=bench_vals,
            floor=float(lever_floor),
            cap=float(lever_cap),
            weights=w,
            store_sqm=float(pd.to_numeric(row.get('sqm_effective', float('nan')), errors='coerce')) if row is not None else None,
            benchmark_sqm_series=bench_sqm_series,
            sqm_calibrate=True,
        )
        return svi, avg_ratio

    # compute per-store SVI vs region same-type benchmark (for chosen store_type)
    tmp = agg.copy()
    tmp["svi_vs_region_type"] = np.nan
    tmp["svi_vs_company_type"] = np.nan

    for idx, r in tmp.iterrows():
        # pick correct benchmarks by r's store_type + region
        stype_r = str(r.get("store_type", "Unknown") or "Unknown")
        region_r = str(r.get("region", "") or "")

        df_reg = agg[(agg["store_type"] == stype_r) & (agg["region"] == region_r)]
        df_com = agg[(agg["store_type"] == stype_r)]

        b_reg = bench_from_df(df_reg)
        b_com = bench_from_df(df_com)

        sreg, _ = svi_for_row(r, b_reg, bench_sqm_series=df_reg['sqm_effective'] if 'sqm_effective' in df_reg.columns else None)
        scom, _ = svi_for_row(r, b_com, bench_sqm_series=df_com['sqm_effective'] if 'sqm_effective' in df_com.columns else None)

        tmp.loc[idx, "svi_vs_region_type"] = sreg
        tmp.loc[idx, "svi_vs_company_type"] = scom

    # A) Region leaderboard (same region as selected store)
    reg_lb = tmp[tmp["region"] == store_region].copy()
    reg_lb = reg_lb.sort_values("svi_vs_region_type", ascending=False)
    reg_lb["Rank"] = np.arange(1, len(reg_lb) + 1)

    show_reg = reg_lb[[
        "Rank", "store_display", "store_type",
        "svi_vs_region_type", "turnover", "conversion_rate",
        "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"
    ]].copy()

    show_reg = show_reg.rename(columns={
        "store_display": "Store",
        "store_type": "Store type",
        "svi_vs_region_type": "SVI",
        "turnover": "Revenue",
        "conversion_rate": "CR",
        "sales_per_visitor": "SPV",
        "sales_per_transaction": "ATV",
        "sales_per_sqm": "Sales/m²",
    })

    # B) Company leaderboard (same store_type as selected store, company-wide)
    com_lb = tmp[tmp["store_type"] == store_type].copy()
    com_lb = com_lb.sort_values("svi_vs_company_type", ascending=False)
    com_lb["Rank"] = np.arange(1, len(com_lb) + 1)

    show_com = com_lb[[
        "Rank", "store_display", "region",
        "svi_vs_company_type", "turnover", "conversion_rate",
        "sales_per_visitor", "sales_per_transaction", "sales_per_sqm"
    ]].copy()

    show_com = show_com.rename(columns={
        "store_display": "Store",
        "region": "Region",
        "svi_vs_company_type": "SVI",
        "turnover": "Revenue",
        "conversion_rate": "CR",
        "sales_per_visitor": "SPV",
        "sales_per_transaction": "ATV",
        "sales_per_sqm": "Sales/m²",
    })

    def _highlight_selected(row, selected_name):
        try:
            if str(row.get("Store", "")).strip() == str(selected_name).strip():
                return ["background-color:#F3F4F6; font-weight:900;"] * len(row)
        except Exception:
            pass
        return [""] * len(row)

    cL, cR = st.columns(2, vertical_alignment="top")

    with cL:
        st.markdown(f"### Region leaderboard — **{store_region}** (SVI vs same-type)")
        st.dataframe(
            show_reg.style
                .apply(lambda r: _highlight_selected(r, store_name), axis=1)
                .format({
                    "SVI": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}",
                    "Revenue": lambda x: fmt_eur(float(x)) if pd.notna(x) else "-",
                    "CR": lambda x: fmt_pct(float(x)) if pd.notna(x) else "-",
                    "SPV": lambda x: fmt_eur_2(float(x)) if pd.notna(x) else "-",
                    "ATV": lambda x: fmt_eur_2(float(x)) if pd.notna(x) else "-",
                    "Sales/m²": lambda x: fmt_eur_2(float(x)) if pd.notna(x) else "-",
                }),
            use_container_width=True,
            hide_index=True,
        )

    with cR:
        st.markdown(f"### Company leaderboard — **{store_type}** (SVI vs same-type)")
        st.dataframe(
            show_com.style
                .apply(lambda r: _highlight_selected(r, store_name), axis=1)
                .format({
                    "SVI": lambda x: "-" if pd.isna(x) else f"{float(x):.0f}",
                    "Revenue": lambda x: fmt_eur(float(x)) if pd.notna(x) else "-",
                    "CR": lambda x: fmt_pct(float(x)) if pd.notna(x) else "-",
                    "SPV": lambda x: fmt_eur_2(float(x)) if pd.notna(x) else "-",
                    "ATV": lambda x: fmt_eur_2(float(x)) if pd.notna(x) else "-",
                    "Sales/m²": lambda x: fmt_eur_2(float(x)) if pd.notna(x) else "-",
                }),
            use_container_width=True,
            hide_index=True,
        )

    # ----------------------
    # Debug
    # ----------------------
    with st.expander("🔧 Debug"):
        st.write("REPORT_URL:", REPORT_URL)
        st.write("Company:", company_id)
        st.write("Store:", chosen_id, store_name)
        st.write("Store type:", store_type)
        st.write("Region:", store_region)
        st.write("Week:", wk_start, "→", wk_end)
        st.write("Lever floor/cap:", lever_floor, lever_cap)
        st.write("df_daily_store cols:", df_daily_store.columns.tolist())

# Run once
main()
