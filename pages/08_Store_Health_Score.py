# pages/08_Store_Health_Score.py
# ------------------------------------------------------------
# PFM Store Health Score — één score (0-100) die de gezondheid
# van je winkel samenvat. AI Health Coach + actiegericht.
# ------------------------------------------------------------
import os, sys
from pathlib import Path
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_INSTALLED = True
except ImportError:
    _OPENAI_INSTALLED = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers_clients import load_clients
from helpers_periods import period_catalog
from helpers_normalize import normalize_vemcount_response
from stylesheet import inject_css as inject_full_css, get_css, pfm_altair
from services.health_score_service import (
    compute_store_health,
    compute_health_batch,
    classify_score,
    get_weights,
    SCORE_COLORS,
    DEFAULT_WEIGHTS,
    FORMAT_WEIGHTS,
)

# ── API URL — same pattern as other pages (05C, 06, 07) ────────────────────
raw_api_url = st.secrets["API_URL"].rstrip("/")
if raw_api_url.endswith("/get-report"):
    REPORT_URL = raw_api_url
    FASTAPI_BASE_URL = raw_api_url.rsplit("/get-report", 1)[0]
else:
    REPORT_URL = raw_api_url + "/get-report"
    FASTAPI_BASE_URL = raw_api_url

# ── Ollama / GLM-5.1 client for AI Health Coach ──────────────────────────────
# Uses OpenAI-compatible endpoint at https://ollama.com/v1 with OLLAMA_API_KEY secret.
# Falls back to localhost:11434/v1 for local dev (no API key needed).
OLLAMA_CLOUD_URL = "https://ollama.com/v1"
OLLAMA_LOCAL_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "glm-5.1:cloud"


def _get_ollama_client():
    """Create an OpenAI-compatible client for Ollama Cloud.
    Streamlit Cloud: uses OLLAMA_API_KEY secret + https://ollama.com/v1.
    Local dev: falls back to localhost:11434/v1 (no API key needed).
    """
    if not _OPENAI_INSTALLED:
        return None
    try:
        api_key = st.secrets.get("OLLAMA_API_KEY", "")
        if api_key:
            # Ollama Cloud (production)
            return _OpenAI(base_url=OLLAMA_CLOUD_URL, api_key=api_key)
        else:
            # Local Ollama (dev)
            return _OpenAI(base_url=OLLAMA_LOCAL_URL, api_key="ollama")
    except Exception:
        return None


_OLLAMA_CLIENT = _get_ollama_client()


def _strip_thinking(text: str) -> str:
    """Remove GLM-5.1 thinking/reasoning blocks from response content.
    Handles: <think>...</think>, <thinking>...</thinking>, and plain reasoning paragraphs.
    """
    import re
    # Remove <think>...</think> and <thinking>...</thinking> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    # If content starts with "Analyze" / "Let me" / numbered analysis steps, skip to first Dutch line
    # (the model sometimes outputs English reasoning before the Dutch answer)
    lines = text.strip().split('\n')
    clean_lines = []
    in_reasoning = True
    for line in lines:
        stripped = line.strip()
        # First line that looks like a Dutch retail action = start of real answer
        if in_reasoning and stripped:
            # Detect Dutch retail answer patterns: numbered actions, bold titles, emoji headers
            if re.match(r'(\d+\.\s|\*\*|🎯|📊|💡|🔥|✅|➡️|Action \d|Formuleer|Title:)', stripped):
                in_reasoning = False
            elif re.match(r'^[A-Z][a-z].*:.*', stripped) and len(stripped) < 120:
                # Title-like lines (Dutch): "Maximale footfall..." etc
                in_reasoning = False
            elif not re.match(r'(Analyze|Problem|Wait|Formul|Action|Goal:|Explanation:|Title:|Data|Rule|Store|Role|Task)', stripped):
                # Not an English reasoning line — probably Dutch answer
                in_reasoning = False
        if not in_reasoning:
            clean_lines.append(line)
    if clean_lines:
        return '\n'.join(clean_lines).strip()
    # Fallback: return original if we can't parse it
    return text.strip()


def ask_health_coach(system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> str | None:
    """Call GLM-5.1 via Ollama. Returns the assistant content string or None."""
    if _OLLAMA_CLIENT is None:
        return None
    try:
        resp = _OLLAMA_CLIENT.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        content = resp.choices[0].message.content
        # GLM-5.1 is a thinking model — strip reasoning blocks from content
        if content and content.strip():
            content = _strip_thinking(content)
        # If content is empty after stripping, try reasoning field
        if not content or not content.strip():
            reasoning = getattr(resp.choices[0].message, "reasoning", None) or ""
            if reasoning.strip():
                content = _strip_thinking(reasoning)
        return content.strip() if content and content.strip() else None
    except Exception as e:
        return None
# ── Region mapping ─────────────────────────────────────────────────────────
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

def api_get_report_local(params, timeout=60):
    """POST to /get-report with flat repeated params."""
    expanded = []
    for k, v in params:
        key = str(k)
        if key.endswith("[]"):
            key = key[:-2]
        if isinstance(v, (list, tuple)):
            for vi in v:
                expanded.append((key, str(vi)))
        else:
            expanded.append((key, str(v)))
    try:
        r = requests.post(REPORT_URL, params=expanded, timeout=timeout)
        if r.status_code >= 400:
            return {"_error": True, "status": r.status_code, "_url": r.request.url, "_method": "POST", "exception": f"HTTP {r.status_code}"}
        return r.json()
    except Exception as e:
        return {"_error": True, "status": 500, "_url": REPORT_URL, "_method": "POST", "exception": str(e)}

def friendly_error(js):
    if isinstance(js, dict) and js.get("_error"):
        st.error(f"API call failed — status: {js.get('status')} | url: {js.get('_url')} | {js.get('exception')}")
        return True
    return False

# ── Brand colours ────────────────────────────────────────────────────────────
PFM_PURPLE = "#762181"
PFM_RED = "#F04438"
PFM_DARK = "#111827"
PFM_GRAY = "#6B7280"
PFM_LIGHT = "#F3F4F6"
PFM_LINE = "#E5E7EB"
PFM_GREEN = "#22C55E"
PFM_AMBER = "#F59E0B"

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Store Health Score — PFM",
    page_icon="🩺",
    layout="wide",
)
inject_full_css(
    PFM_PURPLE=PFM_PURPLE,
    PFM_RED=PFM_RED,
    PFM_DARK=PFM_DARK,
    PFM_GRAY=PFM_GRAY,
    PFM_LIGHT=PFM_LIGHT,
    PFM_LINE=PFM_LINE,
)

TZ = ZoneInfo("Europe/Amsterdam")

# ── Format helpers ────────────────────────────────────────────────────────────
def fmt_eur(x, d=0):
    try:
        return f"€ {x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "–"

def fmt_pct(x, d=1):
    try:
        return f"{x:,.{d}f}%".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "–"

# ── KPI keys ────────────────────────────────────────────────────────────────
KPI_KEYS = [
    "count_in", "count_out", "turnover",
    "conversion_rate", "sales_per_visitor",
    "sales_per_sqm", "inside",
]

# ── Score bands ─────────────────────────────────────────────────────────────
band_labels = {
    "excellent": "🟢 Uitstekend",
    "good": "🟢 Goed",
    "attention": "🟠 Aandacht nodig",
    "critical": "🔴 Kritiek",
    "unknown": "⚪ Onbekend",
}

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 Store Health Score")
    st.caption("Één score. Vijf pijlers. Direct actiegericht.")

    # ── 1. Client selectie ───────────────────────────────────────────────────
    clients = load_clients()
    client_options = {c["company_id"]: f"{c['brand']} ({c['name']})" for c in clients}
    selected_client_id = st.selectbox(
        "Retailer",
        options=list(client_options.keys()),
        format_func=lambda x: client_options[x],
        index=0,
    )

    # ── 2. Fetch shops for this client ──────────────────────────────────────
    try:
        locations_df = get_locations_by_company(selected_client_id)
    except Exception as e:
        st.error(f"Kon winkels niet ophalen: {e}")
        locations_df = pd.DataFrame()

    selected_shop_id = None
    selected_shop_name = None
    all_shop_ids = []
    sqm_map = {}  # shop_id → sqm

    if not locations_df.empty:
        locations_df["id"] = pd.to_numeric(locations_df["id"], errors="coerce").astype("Int64")

        # Merge with region mapping for store names / types / sqm
        region_map = load_region_mapping()
        if not region_map.empty:
            merged = locations_df.merge(region_map, left_on="id", right_on="shop_id", how="inner")
        else:
            merged = locations_df.copy()
            merged["region"] = "Onbekend"
            merged["shop_id"] = merged["id"]
            merged["sqm_override"] = np.nan
            merged["store_type"] = "Unknown"

        if "store_label" in merged.columns and merged["store_label"].notna().any():
            merged["store_display"] = merged["store_label"]
        elif "name" in merged.columns:
            merged["store_display"] = merged["name"]
        else:
            merged["store_display"] = merged["id"].astype(str)

        merged["store_type"] = (
            merged.get("store_type", "Unknown")
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        )

        merged["id_int"] = pd.to_numeric(merged["id"], errors="coerce").astype("Int64")
        merged = merged.dropna(subset=["id_int"])
        merged["id_int"] = merged["id_int"].astype(int)

        store_dim = merged[["id_int", "store_display", "region", "store_type"]].drop_duplicates()

        # Build sqm_map: prefer sqm_override from regions.csv, else try sq_meter from API if available
        for _, row in merged.drop_duplicates(subset=["id_int"]).iterrows():
            sid = int(row["id_int"])
            # sqm_override from regions.csv
            sqm_override = row.get("sqm_override")
            if pd.notna(sqm_override) and float(sqm_override) > 0:
                sqm_map[sid] = float(sqm_override)

        if not store_dim.empty:
            store_dim["dd_label"] = (
                store_dim["store_display"].fillna(store_dim["id_int"].astype(str))
                + " · " + store_dim["region"]
                + " (" + store_dim["id_int"].astype(str) + ")"
            )

            selected_shop_label = st.selectbox(
                "Winkel",
                options=store_dim["dd_label"].tolist(),
                index=0,
            )
            selected_row = store_dim[store_dim["dd_label"] == selected_shop_label].iloc[0]
            selected_shop_id = int(selected_row["id_int"])
            selected_shop_name = selected_row["store_display"]
            selected_region = selected_row["region"]

            # All shop IDs for this client (for benchmark computation)
            all_shop_ids = store_dim["id_int"].tolist()

            with st.expander("ℹ️ Wat betekent Winkelformaat?"):
                st.markdown("""
**⚖️ Standaard** — Evenwichtige weging, geschikt voor de meeste winkels.

**📈 Groeiformaat** — Voor winkels in groeimarkten of expansiefase.
- Traffic Vitality weegt zwaarder (35%): instroom is cruciaal
- Space Efficiency weegt lichter (10%): ruimte wordt nog ingericht

**📉 Krimpformaat** — Voor winkels die krimpende markten bedienen.
- Space Efficiency weegt zwaarder (35%): elke m² moet renderen
- Traffic Vitality weegt lichter (15%): minder focus op groei, meer op efficiëntie
""")
        else:
            st.warning("Geen winkels gevonden voor deze retailer.")
    else:
        st.warning("Geen locaties gevonden voor deze retailer.")

    st.markdown("---")

    # ── 3. Periode ──────────────────────────────────────────────────────────
    today = date.today()
    periods = period_catalog(today)
    period_labels = list(periods.keys())
    selected_period_label = st.selectbox("Periode", period_labels, index=len(period_labels) - 1)
    selected_period = periods[selected_period_label]

    # ── 4. Formaat ──────────────────────────────────────────────────────────
    store_format = st.radio(
        "Winkel formaat",
        options=["default", "growth", "shrink"],
        format_func=lambda x: {"default": "⚖️ Standaard", "growth": "📈 Groeiformaat", "shrink": "📉 Krimpformaat"}[x],
        index=0,
        horizontal=True,
    )

    st.markdown("---")
    st.markdown("**Pijler gewichten**")
    weights = get_weights(store_format)
    for key, w in weights.items():
        labels = {
            "traffic_vitality": "🚶 Traffic",
            "conversion_power": "💰 Conversie",
            "space_efficiency": "📐 Ruimte",
            "customer_value": "🎯 Klantwaarde",
            "data_trust": "📡 Data Trust",
        }
        st.caption(f"{labels[key]}: {w:.0%}")

# ── Guard: need shop ─────────────────────────────────────────────────────────
if selected_shop_id is None:
    st.info("Selecteer een retailer en winkel in de sidebar.")
    st.stop()

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="pfm-header pfm-header--fixed">
  <div>
    <div class="pfm-title">🩺 Store Health Score</div>
    <div class="pfm-sub">Hoe gezond is je winkel? — Één score, vijf pijlers, direct actie</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Data ophalen ─────────────────────────────────────────────────────────────
# Fetch data for ALL shops of this retailer (for benchmark)
# AND compute health for selected shop using cross-store benchmarks
with st.spinner(f"Data ophalen voor {selected_shop_name}..."):
    # Build params for all shops (needed for benchmark)
    params = []
    for sid in all_shop_ids:
        params.append(("data", sid))
    for k in KPI_KEYS:
        params.append(("data_output", k))
    params += [
        ("source", "shops"),
        ("period", "date"),
        ("step", "day"),
        ("form_date_from", str(selected_period.start)),
        ("form_date_to", str(selected_period.end)),
    ]

    js = api_get_report_local(params)
    if friendly_error(js):
        st.stop()

    # Name map for all shops
    shop_name_map = {int(sid): name for sid, name in zip(
        store_dim["id_int"].tolist(),
        store_dim["store_display"].tolist()
    )}

    # ── Extract sq_meter from API response metadata (per shop) ────────────
    # The API returns shop metadata including sq_meter in data[date_key][shop_id]["data"]
    # This is NOT included by normalize_vemcount_response, so we extract it separately.
    api_sqm_map = {}  # shop_id → sqm from API
    if isinstance(js, dict) and "data" in js:
        for _bucket_key, shops_dict in js["data"].items():
            if not isinstance(shops_dict, dict):
                continue
            for shop_id_str, shop_node in shops_dict.items():
                if not isinstance(shop_node, dict):
                    continue
                # Extract sq_meter from shop metadata
                meta = shop_node.get("data", {})
                if isinstance(meta, dict) and "sq_meter" in meta:
                    try:
                        sid = int(shop_id_str)
                        sq_val = float(meta["sq_meter"])
                        if sq_val > 0 and sid not in api_sqm_map:
                            api_sqm_map[sid] = sq_val
                    except (ValueError, TypeError):
                        pass
                # Also check inside dates dict
                dates = shop_node.get("dates", {})
                if isinstance(dates, dict):
                    for _ts, day_node in dates.items():
                        if isinstance(day_node, dict):
                            day_meta = day_node.get("data", {})
                            if isinstance(day_meta, dict) and "sq_meter" in day_meta:
                                try:
                                    sid = int(shop_id_str)
                                    sq_val = float(day_meta["sq_meter"])
                                    if sq_val > 0 and sid not in api_sqm_map:
                                        api_sqm_map[sid] = sq_val
                                except (ValueError, TypeError):
                                    pass

    # Merge sqm: override > api > None
    final_sqm_map = {}
    for sid in all_shop_ids:
        if sid in sqm_map and pd.notna(sqm_map[sid]) and float(sqm_map[sid]) > 0:
            final_sqm_map[sid] = float(sqm_map[sid])
        elif sid in api_sqm_map:
            final_sqm_map[sid] = api_sqm_map[sid]

    df = normalize_vemcount_response(js, shop_name_map, kpi_keys=KPI_KEYS)

    if df is None or df.empty:
        st.warning("Geen data ontvangen voor deze periode/parameters.")
        with st.expander("🔧 Debug"):
            st.write("Params:", params)
            st.write("API response keys:", list(js.keys()) if isinstance(js, dict) else type(js))
        st.stop()

# ── Ensure numeric columns ───────────────────────────────────────────────────
for col in KPI_KEYS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "date" not in df.columns:
    df["date"] = pd.to_datetime(df.get("timestamp"), errors="coerce").dt.date

df = df[pd.notna(df.get("shop_id"))]

if df.empty:
    st.warning("Geen geldige data na filtering.")
    st.stop()

# ── Aggregate per store ────────────────────────────────────────────────────
agg_cols = {}
for col in KPI_KEYS:
    if col in df.columns:
        if col in ("conversion_rate", "sales_per_visitor", "sales_per_sqm"):
            agg_cols[col] = "mean"
        else:
            agg_cols[col] = "sum"

store_agg = df.groupby(["shop_id", "shop_name"], as_index=False).agg(agg_cols)
store_agg = store_agg.rename(columns={"count_in": "footfall"})

# ── Add sqm column ────────────────────────────────────────────────────────
# Priority: regions.csv sqm_override > API sq_meter
store_agg["sqm_effective"] = store_agg["shop_id"].map(final_sqm_map)

# ── Compute health scores for ALL stores (for benchmarks) ────────────────
all_results = compute_health_batch(store_agg, store_key_col="shop_id", store_format=store_format)

if not all_results:
    st.warning("Kon geen health scores berekenen — onvoldoende data.")
    st.stop()

# ── Compute cross-store benchmarks ──────────────────────────────────────────
# Use medians across all stores of this retailer as benchmarks
footfall_median = store_agg["footfall"].median(skipna=True) if "footfall" in store_agg.columns else np.nan
conv_median = store_agg["conversion_rate"].median(skipna=True) if "conversion_rate" in store_agg.columns else np.nan
spv_median = store_agg["sales_per_visitor"].median(skipna=True) if "sales_per_visitor" in store_agg.columns else np.nan
turnover_median = store_agg["turnover"].median(skipna=True) if "turnover" in store_agg.columns else np.nan

# Compute turnover_per_sqm median for benchmark
if "sqm_effective" in store_agg.columns:
    tpsm_series = store_agg["turnover"] / store_agg["sqm_effective"].replace(0, np.nan)
    tpsm_median = tpsm_series.median(skipna=True)
else:
    tpsm_median = np.nan

# ATV: sales_per_visitor / (conversion_rate / 100)
if "conversion_rate" in store_agg.columns and "sales_per_visitor" in store_agg.columns:
    atv_series = store_agg["sales_per_visitor"] / (store_agg["conversion_rate"] / 100).replace(0, np.nan)
    atv_median = atv_series.median(skipna=True)
else:
    atv_median = np.nan

# ── Find selected store's data ──────────────────────────────────────────────
selected_row = store_agg[store_agg["shop_id"] == selected_shop_id]

if selected_row.empty:
    st.warning(f"Geen data gevonden voor {selected_shop_name}.")
    st.stop()

row = selected_row.iloc[0]

# ── Recompute health score WITH cross-store benchmarks ─────────────────────
result = compute_store_health(
    store_id=selected_shop_id,
    store_name=selected_shop_name,
    footfall=row.get("footfall", np.nan),
    turnover=row.get("turnover", np.nan),
    conversion_rate=row.get("conversion_rate", np.nan),
    spv=row.get("sales_per_visitor", np.nan),
    sqm=row.get("sqm_effective", np.nan),
    footfall_region_median=footfall_median,
    footfall_ly=row.get("footfall_ly", np.nan),
    conversion_benchmark=conv_median,
    spv_benchmark=spv_median,
    tpsm_benchmark=tpsm_median,
    atv_benchmark=atv_median,
    capture_index=row.get("capture_index", np.nan),
    sensor_uptime_pct=row.get("sensor_uptime", np.nan),
    data_completeness_pct=row.get("data_completeness", np.nan),
    store_format=store_format,
)

# ── Also compute scores for all stores (for overview) with benchmarks ───────
benchmarked_results = []
for _, srow in store_agg.iterrows():
    sid = int(srow["shop_id"])
    sname = str(srow.get("shop_name", sid))
    r = compute_store_health(
        store_id=sid,
        store_name=sname,
        footfall=srow.get("footfall", np.nan),
        turnover=srow.get("turnover", np.nan),
        conversion_rate=srow.get("conversion_rate", np.nan),
        spv=srow.get("sales_per_visitor", np.nan),
        sqm=srow.get("sqm_effective", np.nan),
        footfall_region_median=footfall_median,
        footfall_ly=srow.get("footfall_ly", np.nan),
        conversion_benchmark=conv_median,
        spv_benchmark=spv_median,
        tpsm_benchmark=tpsm_median,
        atv_benchmark=atv_median,
        capture_index=srow.get("capture_index", np.nan),
        sensor_uptime_pct=srow.get("sensor_uptime", np.nan),
        data_completeness_pct=srow.get("data_completeness", np.nan),
        store_format=store_format,
    )
    benchmarked_results.append(r)

# ── Score gauge ────────────────────────────────────────────────────────────
col_gauge, col_detail = st.columns([1, 2])

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result.health_score if pd.notna(result.health_score) else 0,
        domain={"x": [0, 1], "y": [0, 1]},
        title={
            "text": f"<b>{result.store_name}</b><br><span style='font-size:0.8em;color:gray'>Store Health Score</span>",
            "font": {"size": 16},
        },
        number={"font": {"size": 48, "color": result.health_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": PFM_GRAY},
            "bar": {"color": result.health_color, "thickness": 0.3},
            "bgcolor": PFM_LIGHT,
            "steps": [
                {"range": [0, 45], "color": "#FEE2E2"},
                {"range": [45, 60], "color": "#FEF3C7"},
                {"range": [60, 75], "color": "#DCFCE7"},
                {"range": [75, 100], "color": "#BBF7D0"},
            ],
            "threshold": {"line": {"color": PFM_PURPLE, "width": 4}, "thickness": 0.8, "value": 75},
        },
    ))
    fig_gauge.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        font={"color": PFM_DARK, "family": "Inter, system-ui"},
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown(
        f"<div style='text-align:center; margin-top:-10px;'>"
        f"<span style='font-size:1.3rem; font-weight:800; color:{result.health_color};'>"
        f"{band_labels.get(result.health_band, '–')}</span></div>",
        unsafe_allow_html=True,
    )

with col_detail:
    # ── Pillar bars ────────────────────────────────────────────────────────
    st.markdown("#### Pijler Scores")

    fig_pillars = go.Figure()
    for p in result.pillars:
        score_val = p.score if pd.notna(p.score) else 0
        color = "#22C55E" if score_val >= 75 else "#84CC16" if score_val >= 60 else "#F59E0B" if score_val >= 45 else "#EF4444"

        fig_pillars.add_trace(go.Bar(
            y=[p.label],
            x=[score_val],
            orientation="h",
            marker_color=color,
            text=[f"  {score_val:.0f}" if pd.notna(p.score) else "  –"],
            textposition="inside",
            textfont={"color": "white", "size": 14, "family": "Inter"},
            hovertext=[f"{p.reason} (gewicht: {p.weight:.0%})"],
            hoverinfo="text",
            showlegend=False,
        ))

    fig_pillars.update_layout(
        height=220,
        margin=dict(l=140, r=20, t=10, b=10),
        xaxis={"range": [0, 100], "showgrid": True, "gridcolor": PFM_LINE, "tickfont": {"color": PFM_GRAY}},
        yaxis={"showgrid": False, "tickfont": {"color": PFM_DARK, "size": 13}},
        paper_bgcolor="white",
        plot_bgcolor="white",
        bargap=0.35,
    )
    st.plotly_chart(fig_pillars, use_container_width=True)

    # ── Action hint ────────────────────────────────────────────────────────
    if result.action_hint:
        st.markdown(f"""
        <div class="callout">
          <div class="callout-title">💡 Actie</div>
          <div class="callout-sub">{result.action_hint}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Pillar detail table ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Pijler details")

detail_rows = []
for p in result.pillars:
    score_display = f"{p.score:.0f}" if pd.notna(p.score) else "–"
    raw_display = f"{p.value_raw:.1f}" if pd.notna(p.value_raw) else "–"
    detail_rows.append({
        "Pijler": p.label,
        "Score": score_display,
        "Gewicht": f"{p.weight:.0%}",
        "Band": band_labels.get(classify_score(p.score if pd.notna(p.score) else np.nan)[0], "–"),
        "Ruwe waarde": raw_display,
        "Toelichting": p.reason or "–",
    })

detail_df = pd.DataFrame(detail_rows)
st.dataframe(detail_df, use_container_width=True, hide_index=True)

# ── All stores overview ─────────────────────────────────────────────────────
if len(benchmarked_results) > 1:
    st.markdown("---")
    st.markdown("### 🏪 Alle winkels — vergelijking")

    overview_rows = []
    for r in benchmarked_results:
        pillar_scores = {p.key: (p.score if pd.notna(p.score) else None) for p in r.pillars}
        overview_rows.append({
            "Winkel": r.store_name,
            "Health": r.health_score if pd.notna(r.health_score) else None,
            "Traffic": pillar_scores.get("traffic_vitality"),
            "Conversie": pillar_scores.get("conversion_power"),
            "Ruimte": pillar_scores.get("space_efficiency"),
            "Klantwaarde": pillar_scores.get("customer_value"),
            "Data Trust": pillar_scores.get("data_trust"),
        })

    overview_df = pd.DataFrame(overview_rows)
    overview_df = overview_df.sort_values("Health", ascending=False, na_position="last").reset_index(drop=True)

    # Format scores
    format_dict = {"Health": "{:.0f}", "Traffic": "{:.0f}", "Conversie": "{:.0f}",
                   "Ruimte": "{:.0f}", "Klantwaarde": "{:.0f}", "Data Trust": "{:.0f}"}
    display_df = overview_df.copy()
    for col, fmt in format_dict.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "–")

    # Highlight selected store
    display_df[""] = ["▶" if r.store_id == selected_shop_id else "" for r in sorted(benchmarked_results, key=lambda r: r.health_score if pd.notna(r.health_score) else -1, reverse=True)]

    st.dataframe(display_df[["", "Winkel", "Health", "Traffic", "Conversie", "Ruimte", "Klantwaarde", "Data Trust"]], use_container_width=True, hide_index=True)

# ── Trend chart ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Health Score Trend")

if len(df) > 0 and "date" in df.columns:
    # Filter for selected shop only
    shop_df = df[df["shop_id"] == selected_shop_id].copy()

    if not shop_df.empty:
        daily_scores = []
        for date_val, day_df in shop_df.groupby("date"):
            day_agg = day_df.groupby(["shop_id", "shop_name"], as_index=False).agg(
                {col: "mean" if col in ("conversion_rate", "sales_per_visitor", "sales_per_sqm") else "sum"
                 for col in KPI_KEYS if col in day_df.columns}
            )
            day_agg = day_agg.rename(columns={"count_in": "footfall"})
            # Add sqm
            day_agg["sqm_effective"] = day_agg["shop_id"].map(final_sqm_map)
            # Compute with benchmarks
            day_result = compute_store_health(
                store_id=selected_shop_id,
                store_name=selected_shop_name,
                footfall=day_agg.iloc[0].get("footfall", np.nan) if len(day_agg) > 0 else np.nan,
                turnover=day_agg.iloc[0].get("turnover", np.nan) if len(day_agg) > 0 else np.nan,
                conversion_rate=day_agg.iloc[0].get("conversion_rate", np.nan) if len(day_agg) > 0 else np.nan,
                spv=day_agg.iloc[0].get("sales_per_visitor", np.nan) if len(day_agg) > 0 else np.nan,
                sqm=day_agg.iloc[0].get("sqm_effective", np.nan) if len(day_agg) > 0 else np.nan,
                footfall_region_median=footfall_median,
                conversion_benchmark=conv_median,
                spv_benchmark=spv_median,
                tpsm_benchmark=tpsm_median,
                atv_benchmark=atv_median,
                store_format=store_format,
            )
            if pd.notna(day_result.health_score):
                daily_scores.append({"date": date_val, "health": day_result.health_score})

        if daily_scores:
            trend_df = pd.DataFrame(daily_scores).sort_values("date")

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_df["date"],
                y=trend_df["health"],
                mode="lines+markers",
                name="Health Score",
                line={"color": PFM_PURPLE, "width": 3},
                marker={"size": 6, "color": PFM_PURPLE},
                fill="tozeroy",
                fillcolor="rgba(118, 33, 129, 0.08)",
            ))

            fig_trend.add_hline(y=75, line_dash="dot", line_color="#22C55E",
                               annotation_text="Uitstekend", annotation_position="top right")
            fig_trend.add_hline(y=60, line_dash="dot", line_color="#84CC16",
                               annotation_text="Goed", annotation_position="top right")
            fig_trend.add_hline(y=45, line_dash="dot", line_color="#F59E0B",
                               annotation_text="Aandacht", annotation_position="top right")

            fig_trend.update_layout(
                height=320,
                margin=dict(l=50, r=20, t=30, b=40),
                yaxis={"range": [0, 100], "title": "Health Score", "gridcolor": PFM_LINE, "tickfont": {"color": PFM_GRAY}},
                xaxis={"title": "", "gridcolor": PFM_LINE, "tickfont": {"color": PFM_GRAY}},
                paper_bgcolor="white",
                plot_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Niet genoeg data voor een trend.")
    else:
        st.info("Geen data beschikbaar voor geselecteerde winkel.")
else:
    st.info("Geen datumdata beschikbaar voor trend.")

# ── AI Health Coach ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🧠 AI Health Coach")

if result.health_score is not None and not pd.isna(result.health_score):
    valid_pillars = [p for p in result.pillars if pd.notna(p.score)]
    pillar_desc = []
    for p in valid_pillars:
        pillar_desc.append(f"- {p.label}: {p.score:.0f} — {p.reason or 'zie boven'}")
    pillar_text = "\n".join(pillar_desc)
    weakest = min(valid_pillars, key=lambda p: p.score) if valid_pillars else None
    strongest = max(valid_pillars, key=lambda p: p.score) if valid_pillars else None

    # ── 1. Priority Actions ─────────────────────────────────────────────────
    with st.spinner("🧠 Prioriteiten berekenen..."):
        action_prompt = f"""Je bent een retail performance coach voor PFM Intelligence. Analyseer deze Store Health Score en geef 3 prioritaire acties.

Winkel: {result.store_name}
Health Score: {result.health_score:.0f} ({band_labels.get(result.health_band, '–')})
Formaat: {store_format}
Pijler-scores:
{pillar_text}

Regels:
- Geef exact 3 acties, gerangschikt naar geschatte impact (hoogste eerst)
- Elke actie: korte titel + 1 zin uitleg + concreet getal/doel
- Schrijf in het Nederlands
- Gebruik retail-termen (footfall, conversie, ATV, SPV, capture rate, omzet per m²)
- Wees direct en praktisch — geen intro, geen samenvatting"""

        action_result = ask_health_coach(
            "Je bent een retail analytics expert. Antwoord ALLEEN met de 3 acties in het Nederlands. Geen denken, geen uitleg van je methode, geen intro. Start direct met actie 1.",
            action_prompt,
            max_tokens=1500,
        )

    if action_result:
        st.markdown("#### 🎯 Prioriteiten")
        st.markdown(action_result)
    else:
        # Fallback: rule-based
        if weakest:
            st.markdown("#### 🎯 Prioriteit")
            st.markdown(f"Focus op **{weakest.label}** ({weakest.score:.0f}) — {weakest.reason}")
        st.caption("*AI coach niet beschikbaar — toon regelgebaseerde hint.*")

    # ── 2. What-If Scenarios ────────────────────────────────────────────────
    with st.spinner("🧠 What-if scenario's berekenen..."):
        # Calculate potential score improvements
        conv_bump = 2  # pp
        traffic_bump_pct = 10  # percent more footfall

        # Simulate: what if conversion goes up by 2pp?
        new_conv = float(row.get("conversion_rate", np.nan))
        conv_bump_note = ""
        if pd.notna(new_conv):
            new_conv_adj = new_conv + conv_bump
            result_bump_conv = compute_store_health(
                store_id=selected_shop_id,
                store_name=selected_shop_name,
                footfall=row.get("footfall", np.nan),
                turnover=row.get("turnover", np.nan),
                conversion_rate=new_conv_adj,
                spv=row.get("sales_per_visitor", np.nan),
                sqm=row.get("sqm_effective", np.nan),
                footfall_region_median=footfall_median,
                conversion_benchmark=conv_median,
                spv_benchmark=spv_median,
                tpsm_benchmark=tpsm_median,
                atv_benchmark=atv_median,
                store_format=store_format,
            )
            conv_bump_note = f"Conversie +{conv_bump}pp → Health Score {result_bump_conv.health_score:.0f}" if pd.notna(result_bump_conv.health_score) else ""
        else:
            result_bump_conv = None
            conv_bump_note = "(geen conversiedata beschikbaar)"

        # Simulate: what if footfall goes up by 10%?
        new_footfall = float(row.get("footfall", np.nan))
        traffic_bump_note = ""
        if pd.notna(new_footfall) and new_footfall > 0:
            bumped_footfall = new_footfall * (1 + traffic_bump_pct / 100)
            result_bump_traffic = compute_store_health(
                store_id=selected_shop_id,
                store_name=selected_shop_name,
                footfall=bumped_footfall,
                turnover=row.get("turnover", np.nan),
                conversion_rate=row.get("conversion_rate", np.nan),
                spv=row.get("sales_per_visitor", np.nan),
                sqm=row.get("sqm_effective", np.nan),
                footfall_region_median=footfall_median,
                conversion_benchmark=conv_median,
                spv_benchmark=spv_median,
                tpsm_benchmark=tpsm_median,
                atv_benchmark=atv_median,
                store_format=store_format,
            )
            traffic_bump_note = f"Footfall +{traffic_bump_pct}% → Health Score {result_bump_traffic.health_score:.0f}" if pd.notna(result_bump_traffic.health_score) else ""
        else:
            result_bump_traffic = None
            traffic_bump_note = "(geen footfalldata beschikbaar)"

        whatif_data = []
        if conv_bump_note and result_bump_conv:
            whatif_data.append({"Scenario": f"Conversie +{conv_bump}pp", "Nieuwe Score": f"{result_bump_conv.health_score:.0f}", "Verschil": f"{result_bump_conv.health_score - result.health_score:+.0f}"})
        if traffic_bump_note and result_bump_traffic:
            whatif_data.append({"Scenario": f"Footfall +{traffic_bump_pct}%", "Nieuwe Score": f"{result_bump_traffic.health_score:.0f}", "Verschil": f"{result_bump_traffic.health_score - result.health_score:+.0f}"})

        if whatif_data:
            st.markdown("#### 🔄 What-If Scenario's")
            whatif_df = pd.DataFrame(whatif_data)
            st.dataframe(whatif_df, use_container_width=True, hide_index=True)
            st.caption(f"Huidige Health Score: **{result.health_score:.0f}**")
        else:
            st.info("Niet genoeg data voor what-if scenario's.")

    # ── 3. Benchmark Context ────────────────────────────────────────────────
    if len(benchmarked_results) > 1:
        with st.spinner("🧠 Benchmark-analyse..."):
            bench_rows = []
            for r in sorted(benchmarked_results, key=lambda r: r.health_score if pd.notna(r.health_score) else -1, reverse=True):
                if pd.notna(r.health_score):
                    bench_rows.append(f"  {r.store_name}: {r.health_score:.0f}")
            bench_text = "\n".join(bench_rows[:15])

            all_scores = [r.health_score for r in benchmarked_results if pd.notna(r.health_score)]
            median_score = np.median(all_scores) if all_scores else result.health_score
            position = "boven" if result.health_score >= median_score else "onder"

            bench_prompt = f"""Analyseer de positie van {result.store_name} binnen deze retailer.

{result.store_name} score: {result.health_score:.0f} ({band_labels.get(result.health_band, '–')})
Retailer mediaan: {median_score:.0f}
{result.store_name} is {position} het mediaan.

Alle winkel-scores:
{bench_text}

Regels:
- Identificeer patronen: welke winkels presteren goed/slecht en waarom?
- Vergelijk {result.store_name} met de top-performer
- Geef 1 concreet inzicht over wat deze winkel kan leren van de betere performers
- Schrijf in het Nederlands, max 100 woorden
- Geen intro, direct ter zake"""

            bench_result = ask_health_coach(
                "Je bent een retail benchmark analyst. Antwoord ALLEEN met je analyse in het Nederlands. Geen denken, geen methode-uitleg, direct ter zake.",
                bench_prompt,
                max_tokens=1000,
            )

        st.markdown("#### 📊 Benchmark Context")
        if bench_result:
            st.markdown(bench_result)
        else:
            st.markdown(f"{result.store_name} scoort **{result.health_score:.0f}** — {position} het mediaan van **{median_score:.0f}**.")
            st.caption("*AI benchmark analyse niet beschikbaar.*")
    else:
        st.info("Benchmark context beschikbaar vanaf 2 winkels.")

else:
    st.info("Nog onvoldoende data voor de AI Health Coach. Selecteer een periode met data.")

# ── Methodology ─────────────────────────────────────────────────────────────
with st.expander("📖 Methodologie — Hoe wordt de Health Score berekend?"):
    st.markdown("""
    ### Store Health Score Methodologie

    De Store Health Score combineert **5 pijlers** in één score (0-100):

    | Pijler | Gewicht (standaard) | Wat meet het? |
    |--------|---------------------|---------------|
    | 🚶 Traffic Vitality | 25% | Footfall index vs regio + YoY trend |
    | 💰 Conversion Power | 25% | Conversie ratio + Sales Per Visitor |
    | 📐 Space Efficiency | 20% | Omzet per m² + Capture index |
    | 🎯 Customer Value | 15% | Average Transaction Value + SPV diepte |
    | 📡 Data Trust | 15% | Sensor uptime + data compleetheid |

    **Benchmark:** Pijler-scores worden berekend t.o.v. het mediaan van alle winkels binnen dezelfde retailer. Een score van 100 betekent 'op mediaan niveau'.

    **Format-specifieke gewichten:**
    - 📈 Groeiformaat: Traffic 35%, Conversie 25%, Ruimte 10%, Klantwaarde 20%, Data 10%
    - 📉 Krimpformaat: Traffic 15%, Conversie 20%, Ruimte 35%, Klantwaarde 15%, Data 15%

    **Score bands:**
    - 🟢 **75+ Uitstekend** — Sterke positie, focus op behoud en premium beleving
    - 🟢 **60-74 Goed** — Goede basis, gerichte optimalisatie voor extra groei
    - 🟠 **45-59 Aandacht** — Meerdere KPI's onder regio; plan nodig op traffic én SPV
    - 🔴 **0-44 Kritiek** — Structurele achterstand; herijk formule, team en marketingmix

    **Actiegerichte aanbevelingen** worden automatisch gegenereerd op basis van de zwakste pijler(s).
    """)

# ── Debug ───────────────────────────────────────────────────────────────────
with st.expander("🔧 Debug — ruwe data"):
    st.write("Shop ID:", selected_shop_id)
    st.write("Shop name:", selected_shop_name)
    st.write("Periode:", selected_period.start, "→", selected_period.end)
    st.write("Formaat:", store_format)
    st.write("Aantal winkels in benchmark:", len(all_shop_ids))
    st.write("Benchmarks — footfall median:", footfall_median)
    st.write("Benchmarks — conversion median:", conv_median)
    st.write("Benchmarks — SPV median:", spv_median)
    st.write("Benchmarks — TPSM median:", tpsm_median)
    st.write("Benchmarks — ATV median:", atv_median)
    st.write("SQM map (regions.csv):", sqm_map)
    st.write("SQM map (API):", api_sqm_map)
    st.write("SQM map (final, merged):", final_sqm_map)
    st.write("Rows ontvangen:", len(df))
    st.dataframe(store_agg, use_container_width=True)