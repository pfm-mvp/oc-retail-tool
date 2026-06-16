# pages/08_Store_Health_Score.py
# ------------------------------------------------------------
# PFM Store Health Score — één score (0-100) die de gezondheid
# van je winkel samenvat. Sales call opener + actiegericht.
# ------------------------------------------------------------
import os, sys
from pathlib import Path
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers_shop import ID_TO_NAME, SHOP_NAME_MAP_NORM
from helpers_clients import load_clients
from helpers_periods import period_catalog
from utils_pfmx import api_get_report, friendly_error, inject_css
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

def fmt_int(x):
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return "–"

def fmt_score(x):
    try:
        return f"{x:.0f}"
    except Exception:
        return "–"

# ── KPI keys ────────────────────────────────────────────────────────────────
KPI_KEYS = [
    "count_in", "count_out", "turnover",
    "conversion_rate", "sales_per_visitor",
    "sales_per_sqm", "inside",
]

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 Store Health Score")
    st.caption("Één score. Vijf pijlers. Direct actiegericht.")

    # Client selectie
    clients = load_clients()
    client_options = {c["company_id"]: f"{c['brand']} ({c['name']})" for c in clients}
    selected_client = st.selectbox(
        "Client",
        options=list(client_options.keys()),
        format_func=lambda x: client_options[x],
        index=0,
    )

    # Periode
    today = date.today()
    periods = period_catalog(today)
    period_labels = list(periods.keys())
    selected_period_label = st.selectbox("Periode", period_labels, index=len(period_labels) - 1)
    selected_period = periods[selected_period_label]

    # Formaat
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
with st.spinner("Data ophalen..."):
    params = []
    for sid in ID_TO_NAME.keys():
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

    js = api_get_report(params)
    if friendly_error(js):
        st.stop()

    df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=KPI_KEYS)

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

# ── Aggregate per store ────────────────────────────────────────────────────
agg_cols = {}
for col in KPI_KEYS:
    if col in df.columns:
        if col in ("conversion_rate", "sales_per_visitor", "sales_per_sqm"):
            # Weighted average by footfall
            agg_cols[col] = "mean"
        else:
            agg_cols[col] = "sum"

store_agg = df.groupby(["shop_id", "shop_name"], as_index=False).agg(agg_cols)

# Rename for clarity
rename_map = {"count_in": "footfall"}
store_agg = store_agg.rename(columns=rename_map)

# ── Compute health scores ──────────────────────────────────────────────────
results = compute_health_batch(store_agg, store_key_col="shop_id", store_format=store_format)

if not results:
    st.warning("Kon geen health scores berekenen — onvoldoende data.")
    st.stop()

# ── Main display ────────────────────────────────────────────────────────────
# Sort by health score descending
results_sorted = sorted(results, key=lambda r: r.health_score if pd.notna(r.health_score) else -1, reverse=True)

# ── Hero: Top store or selected store ───────────────────────────────────────
best = results_sorted[0]
worst = results_sorted[-1]

# ── Score gauge ────────────────────────────────────────────────────────────
col_gauge, col_detail = st.columns([1, 2])

with col_gauge:
    # Main score gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=best.health_score if pd.notna(best.health_score) else 0,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"<b>{best.store_name}</b><br><span style='font-size:0.8em;color:gray'>Store Health Score</span>", "font": {"size": 16}},
        number={"font": {"size": 48, "color": best.health_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": PFM_GRAY},
            "bar": {"color": best.health_color, "thickness": 0.3},
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

    # Band label
    band_labels = {
        "excellent": "🟢 Uitstekend",
        "good": "🟢 Goed",
        "attention": "🟠 Aandacht nodig",
        "critical": "🔴 Kritiek",
        "unknown": "⚪ Onbekend",
    }
    st.markdown(f"<div style='text-align:center; margin-top:-10px;'>"
                f"<span style='font-size:1.3rem; font-weight:800; color:{best.health_color};'>"
                f"{band_labels.get(best.health_band, '–')}</span></div>",
                unsafe_allow_html=True)

with col_detail:
    # ── Pillar bars ────────────────────────────────────────────────────────
    st.markdown("#### Pijler Scores")

    pillar_data = []
    for p in best.pillars:
        score_display = f"{p.score:.0f}" if pd.notna(p.score) else "–"
        pillar_data.append({
            "Pijler": p.label,
            "Score": p.score if pd.notna(p.score) else 0,
            "Score Display": score_display,
            "Gewicht": f"{p.weight:.0%}",
            "Toelichting": p.reason or "–",
        })

    fig_pillars = go.Figure()
    for i, p in enumerate(best.pillars):
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
    if best.action_hint:
        st.markdown(f"""
        <div class="callout">
          <div class="callout-title">💡 Actie</div>
          <div class="callout-sub">{best.action_hint}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Store selector ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Alle winkels — Health Score overzicht")

# Build summary dataframe
summary_rows = []
for r in results:
    pillar_scores = {p.key: (p.score if pd.notna(p.score) else None) for p in r.pillars}
    summary_rows.append({
        "Winkel": r.store_name,
        "Health": r.health_score if pd.notna(r.health_score) else None,
        "Band": band_labels.get(r.health_band, "–"),
        "Traffic": pillar_scores.get("traffic_vitality"),
        "Conversie": pillar_scores.get("conversion_power"),
        "Ruimte": pillar_scores.get("space_efficiency"),
        "Klantwaarde": pillar_scores.get("customer_value"),
        "Data Trust": pillar_scores.get("data_trust"),
        "Actie": r.action_hint,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("Health", ascending=False).reset_index(drop=True)

# Display with color coding
def color_health(val):
    if pd.isna(val):
        return "background-color: #F3F4F6; color: #9CA3AF"
    if val >= 75:
        return "background-color: #DCFCE7; color: #166534; font-weight:700"
    if val >= 60:
        return "background-color: #FEF9C3; color: #854D0E; font-weight:700"
    if val >= 45:
        return "background-color: #FEF3C7; color: #92400E; font-weight:600"
    return "background-color: #FEE2E2; color: #991B1B; font-weight:700"

def color_pillar(val):
    if pd.isna(val) or val is None:
        return "color: #9CA3AF"
    if val >= 75:
        return "color: #166534; font-weight:600"
    if val >= 60:
        return "color: #854D0E; font-weight:600"
    if val >= 45:
        return "color: #92400E"
    return "color: #991B1B; font-weight:600"

styled = summary_df.style.format({
    "Health": "{:.0f}",
    "Traffic": "{:.0f}",
    "Conversie": "{:.0f}",
    "Ruimte": "{:.0f}",
    "Klantwaarde": "{:.0f}",
    "Data Trust": "{:.0f}",
}).applymap(color_health, subset=["Health"]).applymap(color_pillar, subset=["Traffic", "Conversie", "Ruimte", "Klantwaarde", "Data Trust"])

st.dataframe(styled, use_container_width=True, height=300)

# ── Trend chart ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Health Score Trend (30 dagen)")

# Compute daily health scores for trend
if len(df) > 0 and "date" in df.columns:
    daily_scores = []
    for date_val, day_df in df.groupby("date"):
        day_agg = day_df.groupby(["shop_id", "shop_name"], as_index=False).agg(
            {col: "mean" if col in ("conversion_rate", "sales_per_visitor", "sales_per_sqm") else "sum"
             for col in KPI_KEYS if col in day_df.columns}
        )
        day_agg = day_agg.rename(columns={"count_in": "footfall"})
        day_results = compute_health_batch(day_agg, store_key_col="shop_id", store_format=store_format)
        if day_results:
            # Average health across stores for this day
            scores = [r.health_score for r in day_results if pd.notna(r.health_score)]
            if scores:
                daily_scores.append({"date": date_val, "health_avg": np.mean(scores)})

    if daily_scores:
        trend_df = pd.DataFrame(daily_scores).sort_values("date")

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend_df["date"],
            y=trend_df["health_avg"],
            mode="lines+markers",
            name="Health Score",
            line={"color": PFM_PURPLE, "width": 3},
            marker={"size": 6, "color": PFM_PURPLE},
            fill="tozeroy",
            fillcolor="rgba(118, 33, 129, 0.08)",
        ))

        # Reference lines
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
    st.info("Geen datumdata beschikbaar voor trend.")

# ── Sales Call Opener ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📞 Sales Call Opener")

if best.health_score is not None and not pd.isna(best.health_score):
    # Identify weakest pillar
    valid_pillars = [p for p in best.pillars if pd.notna(p.score)]
    if valid_pillars:
        weakest = min(valid_pillars, key=lambda p: p.score)
        strongest = max(valid_pillars, key=lambda p: p.score)

        opener_lines = [
            f"**\"Hoe gezond is je winkel in {best.store_name}?\"**",
            "",
            f"De Store Health Score meet de vitale signalen van een winkel op een schaal van 0-100.",
            f"Score voor **{best.store_name}**: **{best.health_score:.0f}** — {band_labels.get(best.health_band, '–')}",
            "",
            f"💪 Sterkste pijler: **{strongest.label}** ({strongest.score:.0f})",
            f"⚠️ Zwakste pijler: **{weakest.label}** ({weakest.score:.0f})",
            "",
            f"💡 {best.action_hint}",
            "",
            "Deze score is berekend op basis van live footfall-, conversie- en omzetdata.",
            "Wil je zien hoe we dit kunnen verbeteren? Ik loop er in 2 minuten doorheen.",
        ]
    else:
        opener_lines = [
            f"**\"Hoe gezond is je winkel?\"**",
            "",
            f"Score voor **{best.store_name}**: **{best.health_score:.0f}** — {band_labels.get(best.health_band, '–')}",
            "",
            f"💡 {best.action_hint}",
        ]
else:
    opener_lines = [
        "**\"Hoe gezond is je winkel?\"**",
        "",
        "Nog onvoldoende data om een Health Score te berekenen. ",
        "Meer historische data of een langere periode-selectie kan helpen.",
    ]

st.markdown("\n".join(opener_lines))

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
    st.write("Stores:", len(results))
    st.write("Periode:", selected_period.start, "→", selected_period.end)
    st.write("Formaat:", store_format)
    st.dataframe(store_agg, use_container_width=True)