# pages/07B_Region_Reasoner_AI.py
# ------------------------------------------------------------
# PFM Region Reasoner — Agentic Workload Edition (deterministic-first)
#
# Adds:
# - Agentic workload cards (store-type focus + why + impact + actions)
# - Region SVI rank vs other regions
# - Store-type performance vs other regions (same type; excluding current region)
# - Strength cards (what’s working)
# - Data quality / confidence card
# - LLM rewrite toggle via OpenRouter (ONE CALL, free model)
#
# Notes:
# - get-report response includes sq_meter inside payload; we request it explicitly.
# - sqm_effective = sqm_override (if set) else sq_meter (from report).
# - sales_per_sqm: use API value; only backfill when NaN.
# ------------------------------------------------------------

from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import requests
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------
# Ensure repo root is on sys.path (for outcome_explainer.py in root)
# pages/.. -> repo root
# ------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# --- helpers / services (existing in repo) ---
from helpers_clients import load_clients
from helpers_periods import period_catalog
from helpers_vemcount_api import (
    VemcountApiConfig,
    fetch_report,
    build_report_params,
)
from helpers_normalize import normalize_vemcount_response

from services.svi_service import (
    compute_driver_values_from_period,
    compute_svi_explainable,
    BASE_SVI_WEIGHTS,
)

# Optional: advisor may not exist in all repos → safe import
try:
    from services.advisor import make_actions
except Exception:
    make_actions = None

PFM_PURPLE = "#762181"

# ------------------------------------------------------------
# Optional: OutcomeExplainer import + diagnostics (keep, but not required)
# ------------------------------------------------------------
_OUTCOME_IMPORT_ERROR = None
OutcomeExplainer = None
try:
    from outcome_explainer import OutcomeExplainer  # file in repo root
except Exception as e:
    OutcomeExplainer = None
    _OUTCOME_IMPORT_ERROR = repr(e)

# ------------------------------------------------------------
# OpenRouter constants + helpers
# ------------------------------------------------------------
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_FREE_MODEL = "google/gemma-3-1b-it:free"

FREE_MODEL_CHOICES = [
    # Keep a short, reliable list. You can expand later.
    "google/gemma-3-1b-it:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
]

def _has_secret(key: str) -> bool:
    try:
        return key in st.secrets and bool(str(st.secrets.get(key, "")).strip())
    except Exception:
        return False

def openrouter_chat_completion(
    messages: list[dict],
    model: str,
    api_key: str,
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> str:
    """
    OpenRouter OpenAI-compatible chat.completions.
    Raises RuntimeError with response body on failure (so 400 becomes self-explanatory).
    """
    model = (model or "").strip()
    if not model:
        model = DEFAULT_FREE_MODEL

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Recommended by OpenRouter (helps attribution; usually not required)
        "HTTP-Referer": str(st.secrets.get("OPENROUTER_REFERER", st.secrets.get("APP_URL", "http://localhost:8501"))),
        "X-Title": str(st.secrets.get("OPENROUTER_APP_TITLE", "PFM Region Reasoner")),
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    r = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=120)
    if not r.ok:
        raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {r.text}")

    data = r.json()
    return str(data["choices"][0]["message"]["content"]).strip()

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="PFM Region Reasoner — Agentic", layout="wide")

# ------------------------------------------------------------
# Formatting helpers (EU)
# ------------------------------------------------------------
def fmt_eur(x):
    if pd.isna(x):
        return "-"
    return f"€ {float(x):,.0f}".replace(",", ".")

def fmt_pct(x, d=0):
    if pd.isna(x):
        return "-"
    return f"{float(x):.{d}f}%".replace(".", ",")

def safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0.0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan

def idx_vs(a, b):
    return (a / b * 100.0) if (pd.notna(a) and pd.notna(b) and float(b) != 0.0) else np.nan

# ------------------------------------------------------------
# Region mapping
# ------------------------------------------------------------
@st.cache_data(ttl=600)
def load_region_mapping(path="data/regions.csv") -> pd.DataFrame:
    """
    Expected (semicolon separated):
    shop_id;region;sqm_override;store_type
    """
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()
    if "shop_id" not in df.columns or "region" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")
    df["region"] = df["region"].astype(str).str.strip().str.lower()

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

    return df.dropna(subset=["shop_id"]).copy()

# ------------------------------------------------------------
# FastAPI locations endpoint (often only id+name)
# ------------------------------------------------------------
@st.cache_data(ttl=600)
def get_locations_by_company(fastapi_base_url: str, company_id: int) -> pd.DataFrame:
    url = f"{fastapi_base_url.rstrip('/')}/company/{company_id}/location"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "locations" in data:
        return pd.DataFrame(data["locations"])
    return pd.DataFrame(data)

# ------------------------------------------------------------
# Styling helpers
# ------------------------------------------------------------
def highlight_region_row(df: pd.DataFrame, region_choice: str):
    rc = str(region_choice).strip().lower()

    def _style_row(row):
        is_target = str(row.get("region", "")).strip().lower() == rc
        if is_target:
            return [f"background-color: {PFM_PURPLE}; color: white; font-weight: 700;"] * len(row)
        return [""] * len(row)

    return df.style.apply(_style_row, axis=1)

# ------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------
def compute_company_type_bench(df_all: pd.DataFrame) -> dict:
    out = {}
    if df_all is None or df_all.empty:
        return out

    tmp = df_all.copy()
    if "store_type" not in tmp.columns:
        tmp["store_type"] = "Unknown"

    tmp["store_type"] = (
        tmp["store_type"]
        .fillna("Unknown")
        .astype(str).str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    )

    for stype, g in tmp.groupby("store_type"):
        foot = pd.to_numeric(g.get("footfall"), errors="coerce").dropna().sum()
        turn = pd.to_numeric(g.get("turnover"), errors="coerce").dropna().sum()
        trans = pd.to_numeric(g.get("transactions"), errors="coerce").dropna().sum()

        if "id" in g.columns:
            sqm = g[["id", "sqm_effective"]].drop_duplicates(subset=["id"])
            sqm_sum = pd.to_numeric(sqm["sqm_effective"], errors="coerce").dropna().sum()
        else:
            sqm_sum = pd.to_numeric(g.get("sqm_effective"), errors="coerce").dropna().sum()

        out[stype] = {
            "footfall": float(foot),
            "turnover": float(turn),
            "transactions": float(trans),
            "sales_per_visitor": safe_div(turn, foot),
            "conversion_rate": safe_div(trans, foot) * 100.0 if pd.notna(foot) else np.nan,
            "sales_per_transaction": safe_div(turn, trans),
            "sales_per_sqm": safe_div(turn, sqm_sum),
            "sqm_sum": float(sqm_sum) if pd.notna(sqm_sum) else np.nan,
        }
    return out

# ------------------------------------------------------------
# Upside estimator (deterministic)
# ------------------------------------------------------------
def estimate_upside(store_row: pd.Series, bench_vals: dict) -> tuple[float, str]:
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

# ------------------------------------------------------------
# Region SVI table (rank across regions)
# ------------------------------------------------------------
def compute_region_svi_table(df_all: pd.DataFrame, region_list: list[str]) -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return pd.DataFrame()

    comp_vals = compute_driver_values_from_period(
        footfall=float(pd.to_numeric(df_all["footfall"], errors="coerce").dropna().sum()),
        turnover=float(pd.to_numeric(df_all["turnover"], errors="coerce").dropna().sum()),
        transactions=float(pd.to_numeric(df_all["transactions"], errors="coerce").dropna().sum()),
        sqm_sum=float(pd.to_numeric(df_all["sqm_effective"], errors="coerce").dropna().sum()),
        capture_pct=np.nan,
    )

    rows = []
    for reg in region_list:
        dfr = df_all[df_all["region"] == reg].copy()
        if dfr.empty:
            continue

        reg_vals = compute_driver_values_from_period(
            footfall=float(pd.to_numeric(dfr["footfall"], errors="coerce").dropna().sum()),
            turnover=float(pd.to_numeric(dfr["turnover"], errors="coerce").dropna().sum()),
            transactions=float(pd.to_numeric(dfr["transactions"], errors="coerce").dropna().sum()),
            sqm_sum=float(pd.to_numeric(dfr["sqm_effective"], errors="coerce").dropna().sum()),
            capture_pct=np.nan,
        )

        svi, avg_ratio, _ = compute_svi_explainable(
            vals_a=reg_vals,
            vals_b=comp_vals,
            weights=BASE_SVI_WEIGHTS,
            floor=80,
            cap=120,
        )

        rows.append({
            "region": reg,
            "svi": float(svi) if pd.notna(svi) else np.nan,
            "avg_ratio_vs_company": float(avg_ratio) if pd.notna(avg_ratio) else np.nan,
            "turnover": float(pd.to_numeric(dfr["turnover"], errors="coerce").dropna().sum()),
            "footfall": float(pd.to_numeric(dfr["footfall"], errors="coerce").dropna().sum()),
            "transactions": float(pd.to_numeric(dfr["transactions"], errors="coerce").dropna().sum()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("svi", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out

# ------------------------------------------------------------
# Store-type comparison vs other regions (same type)
# ------------------------------------------------------------
def compute_storetype_vs_other_regions(df_all: pd.DataFrame, region_choice: str) -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return pd.DataFrame()

    df_all = df_all.copy()
    df_all["store_type"] = (
        df_all.get("store_type", "Unknown")
        .fillna("Unknown")
        .astype(str).str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    )

    def kpis_from(df_: pd.DataFrame) -> dict:
        foot = pd.to_numeric(df_["footfall"], errors="coerce").dropna().sum()
        turn = pd.to_numeric(df_["turnover"], errors="coerce").dropna().sum()
        trans = pd.to_numeric(df_["transactions"], errors="coerce").dropna().sum()

        if "id" in df_.columns:
            sqm = df_[["id", "sqm_effective"]].drop_duplicates(subset=["id"])
            sqm_sum = pd.to_numeric(sqm["sqm_effective"], errors="coerce").dropna().sum()
        else:
            sqm_sum = pd.to_numeric(df_.get("sqm_effective"), errors="coerce").dropna().sum()

        return {
            "footfall": float(foot),
            "turnover": float(turn),
            "transactions": float(trans),
            "spv": safe_div(turn, foot),
            "cr": safe_div(trans, foot) * 100.0 if pd.notna(foot) else np.nan,
            "spm2": safe_div(turn, sqm_sum),
        }

    rows = []
    for stype in sorted(df_all["store_type"].dropna().unique().tolist()):
        dfr = df_all[(df_all["region"] == region_choice) & (df_all["store_type"] == stype)]
        dfo = df_all[(df_all["region"] != region_choice) & (df_all["store_type"] == stype)]

        if dfr.empty or dfo.empty:
            continue

        a = kpis_from(dfr)
        b = kpis_from(dfo)

        rows.append({
            "store_type": stype,
            "SPV idx vs other regions": idx_vs(a["spv"], b["spv"]),
            "CR idx vs other regions": idx_vs(a["cr"], b["cr"]),
            "Sales/m² idx vs other regions": idx_vs(a["spm2"], b["spm2"]),
            "Traffic idx vs other regions": idx_vs(a["footfall"], b["footfall"]),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Sales/m² idx vs other regions", ascending=True).reset_index(drop=True)

# ------------------------------------------------------------
# Agentic "why" + actions (rule-based)
# ------------------------------------------------------------
def infer_root_cause(row: dict) -> tuple[str, str]:
    spm2 = row.get("Sales/m² idx vs other regions", np.nan)
    spv = row.get("SPV idx vs other regions", np.nan)
    cr = row.get("CR idx vs other regions", np.nan)
    traf = row.get("Traffic idx vs other regions", np.nan)

    vals = {
        "Space productivity (Sales/m²)": spm2,
        "Basket value & mix (SPV)": spv,
        "Conversion (CR)": cr,
        "Traffic": traf,
    }

    clean = [(k, v) for k, v in vals.items() if pd.notna(v)]
    if not clean:
        return "Unknown", "Not enough data to isolate a driver."

    driver, idxv = sorted(clean, key=lambda x: x[1])[0]

    if driver == "Traffic":
        return driver, f"Traffic lags vs other regions (idx {fmt_pct(idxv,0)}). Fixing traffic is likely the fastest lever."
    if driver == "Conversion (CR)":
        return driver, f"Conversion lags vs other regions (idx {fmt_pct(idxv,0)}). Focus on in-store execution."
    if driver == "Basket value & mix (SPV)":
        return driver, f"Sales per visitor lags (idx {fmt_pct(idxv,0)}). Often ATV/mix & merchandising."
    if driver == "Space productivity (Sales/m²)":
        return driver, f"Sales per m² lags (idx {fmt_pct(idxv,0)}). Suggests layout/assortment productivity."
    return "Unknown", "No clear driver."

def actions_for_driver(driver: str) -> list[str]:
    if "Traffic" in driver:
        return [
            "Run local traffic play: window/visual hook + local activation for 2 weeks",
            "Check opening hours + staffing coverage on peaks (avoid ‘closed-in-practice’)",
            "Validate street/catchment shifts (if available: capture rate / external context)"
        ]
    if "Conversion" in driver:
        return [
            "Audit conversion blockers: queueing, fitting rooms, service coverage",
            "Set micro-goals: greet rate, assisted selling routines, staff allocation by hour",
            "Check POS completeness: transactions missing = fake low CR"
        ]
    if "SPV" in driver or "visitor" in driver.lower():
        return [
            "Push attach-rate / bundling: top 5 bundles + checkout prompts",
            "Merch refresh: hero products in hot zones + price architecture check",
            "Check promo dilution: discounting can lift traffic but kill SPV"
        ]
    if "m²" in driver or "Space" in driver:
        return [
            "Rebalance space: reduce low-yield categories, expand winners",
            "Hotspot planogram: entrance + mid-store conversion zones",
            "Check sqm accuracy (wrong sqm = wrong sales/m²)"
        ]
    return [
        "Validate data quality first",
        "Then pick one lever: Traffic / Conversion / SPV / Space"
    ]

# ------------------------------------------------------------
# Data quality / confidence
# ------------------------------------------------------------
def compute_data_quality(df_region: pd.DataFrame) -> dict:
    out = {}
    if df_region is None or df_region.empty:
        return out

    def miss_pct(col: str) -> float:
        if col not in df_region.columns:
            return 100.0
        s = pd.to_numeric(df_region[col], errors="coerce")
        return float(s.isna().mean() * 100.0)

    out["sqm_missing_pct"] = miss_pct("sqm_effective")
    out["turnover_missing_pct"] = miss_pct("turnover")
    out["transactions_missing_pct"] = miss_pct("transactions")
    out["footfall_missing_pct"] = miss_pct("footfall")

    avg_miss = np.mean([
        out["sqm_missing_pct"],
        out["turnover_missing_pct"],
        out["transactions_missing_pct"],
        out["footfall_missing_pct"],
    ])
    conf = max(0.0, min(100.0, 100.0 - avg_miss))
    out["confidence"] = conf
    return out

# ------------------------------------------------------------
# Cards builder + renderer (Step 3)
# ------------------------------------------------------------
def build_card(title: str, body_md: str, bullets: list[str] | None = None) -> dict:
    return {"title": title, "body": body_md, "bullets": bullets or []}

def render_cards(cards: list[dict], llm_note: str | None = None):
    if llm_note:
        st.caption(llm_note)
    for c in cards:
        with st.container(border=True):
            st.markdown(f"### {c.get('title','')}")
            st.markdown(c.get("body",""))
            for b in (c.get("bullets") or []):
                st.markdown(f"- {b}")

def rewrite_cards_with_openrouter(cards: list[dict], meta: dict, model: str) -> tuple[list[dict], str | None]:
    """
    ONE CALL rewrite. Returns (cards, error_str)
    """
    api_key = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
    if not api_key:
        return cards, "OPENROUTER_API_KEY missing in secrets."

    model = (model or "").strip() or DEFAULT_FREE_MODEL

    system = (
        "You are a retail performance co-pilot for a Region Manager.\n"
        "Rewrite the cards to be punchy, actionable, and quantified.\n"
        "Hard rules:\n"
        "- Keep numbers EXACTLY as provided (€, %, ranks, counts). Do not change digits.\n"
        "- Do not invent new metrics or claims.\n"
        "- Keep markdown.\n"
        "- Return ONLY valid JSON: a list of objects with keys: title, body, bullets.\n"
        "- bullets must be a JSON array of strings.\n"
    )

    user_payload = {
        "context": {
            "client": meta.get("client"),
            "region": meta.get("region"),
            "period": meta.get("period_label"),
        },
        "cards": cards,
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    try:
        txt = openrouter_chat_completion(
            messages=messages,
            model=model,
            api_key=api_key,
            max_tokens=1400,
            temperature=0.2,
        )
        rewritten = json.loads(txt)
        if isinstance(rewritten, list) and all(isinstance(x, dict) for x in rewritten):
            # Light validation
            out = []
            for x in rewritten:
                out.append({
                    "title": str(x.get("title","")).strip() or "Card",
                    "body": str(x.get("body","")).strip(),
                    "bullets": x.get("bullets") if isinstance(x.get("bullets"), list) else [],
                })
            return out, None
        return cards, "LLM returned non-list JSON. Falling back to deterministic cards."
    except Exception as e:
        return cards, str(e)

# ------------------------------------------------------------
# Workload → build cards (deterministic)
# ------------------------------------------------------------
def build_agentic_workload_cards(
    meta: dict,
    region_rank: dict,
    scores: dict,
    stype_vs_other: pd.DataFrame,
    store_summary: pd.DataFrame,
    company_type_bench: dict,
    annual_factor: float,
    data_quality: dict,
) -> list[dict]:
    cards: list[dict] = []

    svi = scores.get("region_svi", np.nan)
    rank = region_rank.get("rank", "-")
    total = region_rank.get("total", "-")
    gap = region_rank.get("gap_to_best", np.nan)

    # Region summary card
    worst_types = []
    if stype_vs_other is not None and not stype_vs_other.empty:
        worst_types = (
            stype_vs_other.sort_values("Sales/m² idx vs other regions", ascending=True)
            .head(2)["store_type"].tolist()
        )

    headline = (
        f"**{meta.get('client','Client')} · {meta.get('region','region')} · {meta.get('period_label','period')}**\n\n"
        f"- Region SVI: **{int(round(float(svi)))} / 100**\n"
        f"- Rank vs regions: **{rank}/{total}**\n"
        f"- Gap to #1 (SVI ratio): **{fmt_pct(gap, 0)}**\n"
        f"- Confidence: **{int(round(data_quality.get('confidence', 0)))} / 100**\n\n"
        f"Most likely friction sits in: **{', '.join(worst_types) if worst_types else 'store-type differences'}**."
    )
    cards.append(build_card("Region summary (30 seconds)", headline))

    # Focus store-types
    if stype_vs_other is None or stype_vs_other.empty:
        cards.append(build_card(
            "Focus areas (by store type)",
            "No store-type vs other-regions comparison available (need same store_type across ≥2 regions).",
        ))
    else:
        focus = stype_vs_other.sort_values("Sales/m² idx vs other regions", ascending=True).head(3)

        # Precompute store upside within region
        opps_store = []
        for _, r in store_summary.iterrows():
            stype = str(r.get("store_type", "Unknown")).strip() or "Unknown"
            bench = company_type_bench.get(stype, {})
            up, driver = estimate_upside(r, bench)
            if pd.notna(up) and up > 0:
                opps_store.append({
                    "store_type": stype,
                    "store_display": r.get("store_display", str(r.get("id", ""))),
                    "impact_period": float(up),
                    "impact_annual": float(up) * annual_factor,
                    "driver": driver,
                })
        df_opps = pd.DataFrame(opps_store)

        for _, row in focus.iterrows():
            stype = row["store_type"]
            driver, expl = infer_root_cause(row.to_dict())
            actions = actions_for_driver(driver)

            up_period = np.nan
            up_annual = np.nan
            examples = []
            if df_opps is not None and not df_opps.empty:
                dft = df_opps[df_opps["store_type"] == stype].copy()
                if not dft.empty:
                    up_period = float(dft["impact_period"].sum())
                    up_annual = float(dft["impact_annual"].sum())
                    top = dft.sort_values("impact_period", ascending=False).head(2)
                    examples = [
                        f"{r['store_display']} — {fmt_eur(r['impact_period'])} (driver: {r['driver']})"
                        for _, r in top.iterrows()
                    ]

            body = (
                f"**Why this store type matters now:** {expl}\n\n"
                f"**Indices vs other regions (same type):**\n"
                f"- Sales/m² idx: **{fmt_pct(row.get('Sales/m² idx vs other regions', np.nan), 0)}**\n"
                f"- SPV idx: **{fmt_pct(row.get('SPV idx vs other regions', np.nan), 0)}**\n"
                f"- CR idx: **{fmt_pct(row.get('CR idx vs other regions', np.nan), 0)}**\n"
                f"- Traffic idx: **{fmt_pct(row.get('Traffic idx vs other regions', np.nan), 0)}**\n\n"
                f"**Estimated upside:** **{fmt_eur(up_period)}** (period) · **{fmt_eur(up_annual)} / year**"
            )

            bullets = []
            bullets.append("Recommended actions (next 14 days):")
            bullets += actions
            if examples:
                bullets.append("Start here (top stores inside this store type):")
                bullets += examples

            cards.append(build_card(f"Workcard — Store type: {stype}", body, bullets=bullets))

    # Strength card
    if store_summary is not None and not store_summary.empty:
        ss = store_summary.copy()
        ss["strength_score"] = (
            pd.to_numeric(ss.get("Sales/m² idx vs type"), errors="coerce").fillna(100)
            + pd.to_numeric(ss.get("SPV idx vs type"), errors="coerce").fillna(100)
            + pd.to_numeric(ss.get("CR idx vs type"), errors="coerce").fillna(100)
        )
        best = ss.sort_values("strength_score", ascending=False).head(3)

        bullets = []
        for _, r in best.iterrows():
            bullets.append(
                f"{r.get('store_display','')} ({r.get('store_type','')}) · "
                f"Sales/m² idx {fmt_pct(r.get('Sales/m² idx vs type', np.nan),0)} · "
                f"SPV idx {fmt_pct(r.get('SPV idx vs type', np.nan),0)} · "
                f"CR idx {fmt_pct(r.get('CR idx vs type', np.nan),0)}"
            )

        cards.append(build_card(
            "Strength card — replicate playbook",
            "These stores outperform their **own store type benchmark**. Protect them, and copy their playbook.",
            bullets=bullets
        ))

    # Confidence card
    dq = data_quality or {}
    dq_body = (
        f"- sqm missing: **{fmt_pct(dq.get('sqm_missing_pct', np.nan),0)}**\n"
        f"- turnover missing: **{fmt_pct(dq.get('turnover_missing_pct', np.nan),0)}**\n"
        f"- transactions missing: **{fmt_pct(dq.get('transactions_missing_pct', np.nan),0)}**\n"
        f"- footfall missing: **{fmt_pct(dq.get('footfall_missing_pct', np.nan),0)}**\n\n"
        f"Confidence score: **{int(round(dq.get('confidence', 0)))} / 100**"
    )
    cards.append(build_card("Confidence card", dq_body))

    return cards

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    # ---- Sidebar settings (SINGLE source of truth) ----
    with st.sidebar:
        st.markdown("## Settings")

        use_llm = st.toggle("Rewrite cards (LLM via OpenRouter)", value=False)
        model_default = str(st.secrets.get("OPENROUTER_MODEL", "")).strip() or DEFAULT_FREE_MODEL

        # Ensure model is never blank
        choices = FREE_MODEL_CHOICES.copy()
        if model_default not in choices:
            choices.insert(0, model_default)

        model = st.selectbox(
            "OpenRouter model",
            options=choices,
            index=choices.index(model_default) if model_default in choices else 0,
            disabled=not use_llm,
        )

        if use_llm and not _has_secret("OPENROUTER_API_KEY"):
            st.warning("OPENROUTER_API_KEY missing in secrets → deterministic cards only.")

        st.caption("Tip: rewrite gebruikt 1 call (alle cards in één keer).")

    # ---- API URL setup ----
    raw_api_url = st.secrets["API_URL"].rstrip("/")
    if raw_api_url.endswith("/get-report"):
        report_url = raw_api_url
        fastapi_base = raw_api_url.rsplit("/get-report", 1)[0]
    else:
        fastapi_base = raw_api_url
        report_url = raw_api_url + "/get-report"

    st.title("PFM Region Reasoner — Agentic Workload Edition")

    # ---- Clients ----
    clients = load_clients("clients.json")
    clients_df = pd.DataFrame(clients)
    if clients_df.empty:
        st.error("No clients found in clients.json")
        st.stop()

    clients_df["label"] = clients_df.apply(
        lambda r: f"{r['brand']} – {r['name']} (company_id {r['company_id']})",
        axis=1
    )
    client_label = st.selectbox("Client", clients_df["label"].tolist())
    selected_client = clients_df[clients_df["label"] == client_label].iloc[0].to_dict()
    company_id = int(selected_client["company_id"])

    # ---- Periods ----
    periods = period_catalog(today=datetime.now().date())
    period_label = st.selectbox("Period", list(periods.keys()))
    p = periods[period_label]
    start_date, end_date = p.start, p.end

    # ---- Region mapping ----
    region_map = load_region_mapping()
    if region_map.empty:
        st.error("No valid data/regions.csv found (need shop_id;region;store_type).")
        st.stop()

    region_choice = st.selectbox("Region", sorted(region_map["region"].unique()))
    region_choice = str(region_choice).strip().lower()

    run = st.button("Run analysis", type="primary")
    if not run:
        st.stop()

    # ---- Locations (id, name) ----
    locations_df = get_locations_by_company(fastapi_base, company_id)
    if locations_df is None or locations_df.empty:
        st.error("No locations returned from /company/{company}/location")
        st.stop()

    if "id" not in locations_df.columns:
        st.error(f"Locations payload has no 'id'. Columns: {locations_df.columns.tolist()}")
        st.stop()

    locations_df = locations_df.copy()
    locations_df["id"] = pd.to_numeric(locations_df["id"], errors="coerce").astype("Int64")

    # join regions mapping
    merged = locations_df.merge(region_map, left_on="id", right_on="shop_id", how="inner")
    if merged.empty:
        st.warning("No stores matched regions.csv mapping.")
        st.stop()

    merged["region"] = merged["region"].astype(str).str.strip().str.lower()
    merged["store_display"] = merged["name"] if "name" in merged.columns else merged["id"].astype(str)

    if "store_type" not in merged.columns:
        merged["store_type"] = "Unknown"
    merged["store_type"] = (
        merged["store_type"]
        .fillna("Unknown")
        .astype(str).str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    )

    # ---- Fetch report (include sq_meter!) ----
    all_shop_ids = merged["id"].dropna().astype(int).unique().tolist()
    if not all_shop_ids:
        st.error("No shop IDs after mapping.")
        st.stop()

    cfg = VemcountApiConfig(report_url=report_url)

    metric_map = {
        "count_in": "footfall",
        "turnover": "turnover",
        "transactions": "transactions",
        "sales_per_sqm": "sales_per_sqm",
        "sq_meter": "sq_meter",
    }

    params_preview = build_report_params(
        shop_ids=all_shop_ids,
        data_outputs=list(metric_map.keys()),
        period="date",
        step="day",
        source="shops",
        date_from=start_date,
        date_to=end_date,
    )

    try:
        resp = fetch_report(
            cfg=cfg,
            shop_ids=all_shop_ids,
            data_outputs=list(metric_map.keys()),
            period="date",
            step="day",
            source="shops",
            date_from=start_date,
            date_to=end_date,
            timeout=120,
        )
    except Exception as e:
        st.error(f"Report fetch failed: {e}")
        st.write("Params preview:", params_preview)
        st.stop()

    df_norm = normalize_vemcount_response(resp, kpi_keys=metric_map.keys()).rename(columns=metric_map)
    if df_norm is None or df_norm.empty:
        st.warning("No data returned for this selection.")
        st.stop()

    store_key = None
    for cand in ["shop_id", "id", "location_id"]:
        if cand in df_norm.columns:
            store_key = cand
            break
    if store_key is None:
        st.error("No store id column in report response.")
        st.stop()

    df_norm[store_key] = pd.to_numeric(df_norm[store_key], errors="coerce").astype("Int64")

    df = df_norm.merge(
        merged[["id", "store_display", "region", "store_type", "sqm_override"]].drop_duplicates(),
        left_on=store_key,
        right_on="id",
        how="left"
    )

    # numeric
    for c in ["footfall", "turnover", "transactions", "sales_per_sqm", "sq_meter", "sqm_override"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # sqm_effective = override else sq_meter
    if "sq_meter" not in df.columns:
        df["sq_meter"] = np.nan
    df["sqm_effective"] = pd.to_numeric(df["sqm_override"], errors="coerce").combine_first(
        pd.to_numeric(df["sq_meter"], errors="coerce")
    )
    df["sqm_effective"] = pd.Series(df["sqm_effective"], index=df.index, dtype="float64")
    df.loc[df["sqm_effective"] <= 0, "sqm_effective"] = np.nan

    # sales_per_sqm backfill only if NaN
    if "sales_per_sqm" not in df.columns:
        df["sales_per_sqm"] = np.nan
    df["sales_per_sqm"] = pd.to_numeric(df["sales_per_sqm"], errors="coerce")
    computed_spm2 = pd.Series(
        np.where((df["sqm_effective"] > 0) & df["turnover"].notna(), df["turnover"] / df["sqm_effective"], np.nan),
        index=df.index
    )
    df["sales_per_sqm"] = df["sales_per_sqm"].combine_first(computed_spm2)

    # subset region
    df_region = df[df["region"] == region_choice].copy()
    if df_region.empty:
        st.warning("No rows for selected region after join.")
        st.stop()

    # store summary totals
    store_summary = (
        df_region
        .groupby(["id", "store_display", "store_type", "sqm_effective"], as_index=False)
        .agg(
            turnover=("turnover", "sum"),
            footfall=("footfall", "sum"),
            transactions=("transactions", "sum"),
        )
    )

    store_summary["conversion_rate"] = np.where(
        store_summary["footfall"] > 0,
        store_summary["transactions"] / store_summary["footfall"] * 100.0,
        np.nan
    )
    store_summary["sales_per_visitor"] = np.where(
        store_summary["footfall"] > 0,
        store_summary["turnover"] / store_summary["footfall"],
        np.nan
    )
    store_summary["sales_per_transaction"] = np.where(
        store_summary["transactions"] > 0,
        store_summary["turnover"] / store_summary["transactions"],
        np.nan
    )
    store_summary["sales_per_sqm"] = np.where(
        store_summary["sqm_effective"] > 0,
        store_summary["turnover"] / store_summary["sqm_effective"],
        np.nan
    )

    # company benchmarks
    df_all = df.copy()
    company_type_bench = compute_company_type_bench(df_all)

    # indices vs type
    store_summary["Traffic idx vs type"] = store_summary.apply(
        lambda r: idx_vs(r["footfall"], company_type_bench.get(r["store_type"], {}).get("footfall", np.nan)),
        axis=1
    )
    store_summary["CR idx vs type"] = store_summary.apply(
        lambda r: idx_vs(r["conversion_rate"], company_type_bench.get(r["store_type"], {}).get("conversion_rate", np.nan)),
        axis=1
    )
    store_summary["SPV idx vs type"] = store_summary.apply(
        lambda r: idx_vs(r["sales_per_visitor"], company_type_bench.get(r["store_type"], {}).get("sales_per_visitor", np.nan)),
        axis=1
    )
    store_summary["Sales/m² idx vs type"] = store_summary.apply(
        lambda r: idx_vs(r["sales_per_sqm"], company_type_bench.get(r["store_type"], {}).get("sales_per_sqm", np.nan)),
        axis=1
    )

    # Region SVI vs company
    reg_vals = compute_driver_values_from_period(
        footfall=float(pd.to_numeric(df_region["footfall"], errors="coerce").dropna().sum()),
        turnover=float(pd.to_numeric(df_region["turnover"], errors="coerce").dropna().sum()),
        transactions=float(pd.to_numeric(df_region["transactions"], errors="coerce").dropna().sum()),
        sqm_sum=float(pd.to_numeric(df_region["sqm_effective"], errors="coerce").dropna().sum()),
        capture_pct=np.nan,
    )
    comp_vals = compute_driver_values_from_period(
        footfall=float(pd.to_numeric(df_all["footfall"], errors="coerce").dropna().sum()),
        turnover=float(pd.to_numeric(df_all["turnover"], errors="coerce").dropna().sum()),
        transactions=float(pd.to_numeric(df_all["transactions"], errors="coerce").dropna().sum()),
        sqm_sum=float(pd.to_numeric(df_all["sqm_effective"], errors="coerce").dropna().sum()),
        capture_pct=np.nan,
    )
    region_svi, avg_ratio, _ = compute_svi_explainable(
        vals_a=reg_vals,
        vals_b=comp_vals,
        weights=BASE_SVI_WEIGHTS,
        floor=80,
        cap=120,
    )

    # annual factor
    days = max(1, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)
    annual_factor = 365.0 / float(days)

    # Rank across regions
    region_list = sorted(df_all["region"].dropna().unique().tolist())
    svi_table = compute_region_svi_table(df_all, region_list)

    region_rank = {}
    if svi_table is not None and not svi_table.empty:
        row = svi_table[svi_table["region"] == region_choice]
        if not row.empty:
            rank = int(row.iloc[0]["rank"])
            total = int(svi_table["rank"].max())
            best = float(svi_table.iloc[0]["svi"]) if pd.notna(svi_table.iloc[0]["svi"]) else np.nan
            cur = float(row.iloc[0]["svi"]) if pd.notna(row.iloc[0]["svi"]) else np.nan
            region_rank = {
                "rank": rank,
                "total": total,
                "best_svi": best,
                "gap_to_best": (cur / best * 100.0) if (pd.notna(cur) and pd.notna(best) and best != 0) else np.nan,
            }

    stype_vs_other = compute_storetype_vs_other_regions(df_all, region_choice)

    meta = {
        "client": selected_client.get("brand", ""),
        "region": region_choice,
        "period_label": f"{period_label} ({start_date} → {end_date})"
    }
    scores = {
        "region_svi": float(region_svi) if pd.notna(region_svi) else np.nan,
        "avg_ratio_vs_company": float(avg_ratio) if pd.notna(avg_ratio) else np.nan
    }

    data_quality = compute_data_quality(df_region)

    # Optional expanders: comparisons
    with st.expander("📊 Region comparison (SVI vs other regions)", expanded=False):
        if svi_table is None or svi_table.empty:
            st.info("No SVI table available.")
        else:
            show = svi_table.copy()
            show["SVI"] = show["svi"].round(0)
            show["Turnover"] = show["turnover"].map(fmt_eur)
            show = show[["rank", "region", "SVI", "Turnover", "footfall", "transactions"]]
            styled = highlight_region_row(show, region_choice)
            st.dataframe(styled, use_container_width=True)

    with st.expander("🧩 Store types vs other regions (same type)", expanded=False):
        if stype_vs_other is None or stype_vs_other.empty:
            st.info("No store-type comparison available.")
        else:
            st.dataframe(stype_vs_other, use_container_width=True)

    # ---- Build cards (deterministic) ----
    cards = build_agentic_workload_cards(
        meta=meta,
        region_rank=region_rank,
        scores=scores,
        stype_vs_other=stype_vs_other,
        store_summary=store_summary,
        company_type_bench=company_type_bench,
        annual_factor=annual_factor,
        data_quality=data_quality,
    )

    # ---- Optional: ONE CALL rewrite ----
    llm_note = None
    if use_llm:
        if not _has_secret("OPENROUTER_API_KEY"):
            llm_note = "LLM rewrite skipped: OPENROUTER_API_KEY missing."
        else:
            cards_rewritten, err = rewrite_cards_with_openrouter(cards, meta=meta, model=model)
            if err:
                llm_note = f"LLM rewrite skipped: {err}"
            else:
                llm_note = f"LLM rewrite applied (OpenRouter: {model})"
                cards = cards_rewritten

    st.markdown("## Agentic workload")
    render_cards(cards, llm_note=llm_note)

    # Debug / Diagnostics
    with st.expander("🔧 Debug / Diagnostics"):
        st.write("REPORT_URL:", report_url)
        st.write("FASTAPI_BASE:", fastapi_base)
        st.write("Company:", company_id)
        st.write("Region:", region_choice)
        st.write("Period:", start_date, "→", end_date)
        st.write("Params preview:", params_preview)

        st.write("OutcomeExplainer file exists:", bool((_REPO_ROOT / "outcome_explainer.py").exists()))
        st.write("OutcomeExplainer import error:", _OUTCOME_IMPORT_ERROR)

        st.write("use_llm:", use_llm)
        st.write("OPENROUTER_API_KEY present:", _has_secret("OPENROUTER_API_KEY"))
        st.write("Selected OpenRouter model:", (model or "").strip() or DEFAULT_FREE_MODEL)
        st.write("DEFAULT_FREE_MODEL:", DEFAULT_FREE_MODEL)

        st.write("df_norm cols:", df_norm.columns.tolist())
        st.write("sqm_effective filled:", int(pd.to_numeric(df["sqm_effective"], errors="coerce").notna().sum()), "/", len(df))
        st.write("Example sqm:", df[["id", "store_display", "sqm_override", "sq_meter", "sqm_effective"]].head(10))

main()