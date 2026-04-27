# pages/08_Store_Health_Score.py
# Store Health Score + Dead Hour Optimizer — MVP Phase 1
import os, sys
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ── Imports from project root ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers_shop import ID_TO_NAME
from utils_pfmx import inject_css
from health_score import compute_health_score, compute_health_trend, SEGMENT_WEIGHTS
from dead_hours import detect_dead_hours, recommend_actions
from health_explainer import explain_health_score, DIM_NAMES_NL

# ── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Store Health Score", layout="wide")
inject_css()
TZ = ZoneInfo("Europe/Amsterdam")

PFM_PURPLE = "#762181"
PFM_RED = "#D32F2F"
PFM_DARK = "#1F1F1F"
PFM_GRAY = "#6B7280"
PFM_LIGHT = "#F7F7F7"
PFM_LINE = "#E5E7EB"

# ── Load sample data ────────────────────────────────────────────────────────
@st.cache_data
def load_sample_data():
    """Load PFM sample data for demo."""
    try:
        # Load weekly footfall data
        df_pathzz = pd.read_csv(ROOT / "data" / "pathzz_sample_weekly.csv", sep=";")
        # Parse Week column: "2023-10-01 To 2023-10-07" → start date
        df_pathzz["week_start"] = df_pathzz["Week"].str.extract(r"(\d{4}-\d{2}-\d{2})").iloc[:, 0]
        df_pathzz["week_start"] = pd.to_datetime(df_pathzz["week_start"], errors="coerce")
        df_pathzz = df_pathzz.rename(columns={"Visits": "visits"})

        # Load regions for sqm
        df_regions = pd.read_csv(ROOT / "data" / "regions.csv", sep=";")

        return df_pathzz, df_regions
    except Exception as e:
        st.error(f"Failed to load sample data: {e}")
        return pd.DataFrame(), pd.DataFrame()

df_pathzz, df_regions = load_sample_data()

# ── Helper functions ────────────────────────────────────────────────────────
def get_segment_for_store(store_id: int) -> str:
    """Get segment for store (mock based on store_id)."""
    # In production, this would come from metadata
    segments = ["fashion", "grocery", "home_living", "general"]
    return segments[store_id % len(segments)]


def generate_mock_hourly_data(store_id: int, days: int = 30) -> pd.DataFrame:
    """Generate mock hourly footfall data for dead hour analysis."""
    # Create hourly data from weekly totals with realistic patterns
    records = []
    base_date = datetime.now() - timedelta(days=days)

    for d in range(days):
        date = base_date + timedelta(days=d)
        day_of_week = date.weekday()

        # Weekly pattern: higher on weekends
        weekend_factor = 1.3 if day_of_week >= 5 else 1.0

        for hour in range(8, 21):  # Store hours 8am-9pm
            # Daily pattern: peak at lunch + evening
            hour_factor = 1.0
            if 11 <= hour <= 14:
                hour_factor = 1.5
            elif 17 <= hour <= 19:
                hour_factor = 1.6
            elif hour > 19:
                hour_factor = 0.7

            # Random variation
            noise = np.random.normal(1.0, 0.15)

            # Base visitors per hour (mock)
            base_visitors = 50
            visitors = int(base_visitors * weekend_factor * hour_factor * noise)

            # Inject some dead hours (Tue/Thu 14-16, low footfall)
            if day_of_week in [1, 3] and 14 <= hour <= 16:
                visitors = int(visitors * 0.4)  # 40% of normal = dead hour

            records.append({
                "date": date.date(),
                "hour": hour,
                "day_of_week": day_of_week,
                "footfall": max(0, visitors),
            })

    return pd.DataFrame(records)


def get_health_color(score: int) -> str:
    """Get color for health score indicator."""
    if score >= 80:
        return "#22C55E"  # Green
    elif score >= 60:
        return "#EAB308"  # Yellow
    elif score >= 40:
        return "#F97316"  # Orange
    else:
        return "#EF4444"  # Red


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏥 Store Health Score + Dead Hour Optimizer")
st.markdown("Composite score (0-100) met dode-uur-detectie en actie-aanbevelingen")

# Store selector
store_id = st.selectbox(
    "Selecteer winkel",
    list(ID_TO_NAME.keys()),
    format_func=lambda x: f"{x} — {ID_TO_NAME.get(x, 'Unknown')}",
    key="store_selector",
)

segment = st.selectbox(
    "Segment (voor gewichten)",
    list(SEGMENT_WEIGHTS.keys()),
    index=0,
    key="segment_selector",
)

# Get store data
df_store = df_pathzz[df_pathzz["shop_id"] == store_id].copy() if not df_pathzz.empty else pd.DataFrame()

# Calculate mock health score
footfall_series = df_store["visits"] if not df_store.empty else pd.Series(dtype=float)
conversion_rate = 0.22 if segment == "fashion" else 0.25  # Mock

health_result = compute_health_score(
    store_id=store_id,
    segment=segment,
    footfall_series=footfall_series,
    conversion_rate=conversion_rate,
)

# Generate mock hourly data for dead hour analysis
hourly_data = generate_mock_hourly_data(store_id, days=30)
dead_hours_result = detect_dead_hours(store_id, hourly_data, threshold_pct=0.60)
recommendations = recommend_actions(store_id, health_result["health_score"], dead_hours_result["dead_hours"])

# Get explanation
explanation = explain_health_score(health_result, dead_hours_result)

# ── Main dashboard layout ─────────────────────────────────────────────────────

# Health Score Gauge
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    score = health_result["health_score"]
    color = get_health_color(score)

    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0;">
        <div style="font-size: 1.2rem; color: {PFM_GRAY}; margin-bottom: 0.5rem;">
            Health Score
        </div>
        <div style="font-size: 5rem; font-weight: 900; color: {color}; line-height: 1.2;">
            {score}
        </div>
        <div style="font-size: 0.9rem; color: {PFM_GRAY}; margin-top: 0.5rem;">
            {len(health_result['flagged_dimensions'])} dimensie(s) onder drempel
        </div>
    </div>
    """, unsafe_allow_html=True)

# Explanation
st.markdown("### 📝 Samenvatting")
st.info(explanation["summary_nl"])

if explanation["dead_hour_impact"]:
    st.warning(explanation["dead_hour_impact"])

st.divider()

# Dimension bars
st.markdown("### 📊 Dimensie Scores")

for dim_name, dim_data in health_result["dimensions"].items():
    score = dim_data["score"]
    weight = dim_data["weight"]
    is_flag = dim_data["flag"]
    nl_name = DIM_NAMES_NL.get(dim_name, dim_name)

    bar_color = get_health_color(score)
    flag_icon = "⚠️" if is_flag else ""

    col_label, col_bar, col_value = st.columns([3, 6, 2])

    with col_label:
        st.markdown(f"**{nl_name}** {flag_icon}")
        st.caption(f"Gewicht: {weight*100:.0f}%")

    with col_bar:
        st.markdown(f"""
        <div style="
            width: 100%;
            height: 24px;
            background: {PFM_LINE};
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        ">
            <div style="
                width: {score}%;
                height: 100%;
                background: {bar_color};
                border-radius: 4px;
            "></div>
        </div>
        """, unsafe_allow_html=True)

    with col_value:
        st.markdown(f"**{score}**/{int(weight*100)}")

st.divider()

# Dead Hour Heatmap
st.markdown("### 🔥 Dode Uren Heatmap")

if not hourly_data.empty:
    # Prepare heatmap data: average footfall by day-of-week and hour
    heatmap_data = (
        hourly_data.groupby(["day_of_week", "hour"])["footfall"]
        .mean()
        .reset_index()
    )
    heatmap_data.columns = ["day_of_week", "hour", "avg_footfall"]

    # Calculate expected per hour-of-week
    expected_by_hour = (
        hourly_data.groupby(["day_of_week", "hour"])["footfall"]
        .mean()
        .reset_index()
    )
    expected_by_hour.columns = ["day_of_week", "hour", "expected"]

    # Merge and calculate ratio
    heatmap_data = heatmap_data.merge(expected_by_hour, on=["day_of_week", "hour"], suffixes=("", "_exp"))
    heatmap_data["ratio"] = heatmap_data["avg_footfall"] / heatmap_data["expected"].replace(0, np.nan)

    day_names = ["Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo"]
    heatmap_data["day_name"] = heatmap_data["day_of_week"].apply(lambda x: day_names[x] if 0 <= x < 7 else "")

    chart = (
        alt.Chart(heatmap_data)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="Uur", scale=alt.Scale(domain=list(range(8, 21)))),
            y=alt.Y("day_name:O", title="Dag", sort=day_names),
            color=alt.Color(
                "ratio:Q",
                scale=alt.Scale(
                    domain=[0, 0.6, 1, 1.5],
                    range=["#EF4444", "#F97316", "#EAB308", "#22C55E"],  # Red → Orange → Yellow → Green
                ),
                title="Ratio t.o.v. gemiddeld",
            ),
            tooltip=[
                alt.Tooltip("day_name", title="Dag"),
                alt.Tooltip("hour", title="Uur"),
                alt.Tooltip("avg_footfall:Q", title="Gem. bezoekers", format=".0f"),
                alt.Tooltip("ratio:Q", title="Ratio", format=".2f"),
            ],
        )
        .properties(width=600, height=200)
    )

    st.altair_chart(chart, use_container_width=True)

    # Legend
    st.caption("🟢 ≥100% 🟡 ~100% 🟠 60-100% 🔴 <60% (dode uren)")

# Dead Hour Blocks
st.markdown("### 🎯 Gedetecteerde Dode Uren")

if dead_hours_result.get("dead_hours"):
    day_names = ["Maandag", "Dinsdag", "Woensdag", "Donderdag", "Vrijdag", "Zaterdag", "Zondag"]

    for block in dead_hours_result["dead_hours"]:
        day_name = day_names[block["day_of_week"]] if 0 <= block["day_of_week"] < 7 else ""
        type_nl = "structureel" if block["type"] == "structural" else "incidenteel"
        color = PFM_RED if block["type"] == "structural" else PFM_GRAY

        st.markdown(f"""
        <div style="
            border-left: 4px solid {color};
            padding: 0.75rem 1rem;
            background: {PFM_LIGHT};
            margin-bottom: 0.5rem;
        ">
            <strong>{day_name} {block['start_hour']:02d}:00 – {block['end_hour'] + 1:02d}:00</strong>
            <span style="float: right; color: {PFM_GRAY};">{type_nl}</span><br>
            <span style="color: {PFM_GRAY}; font-size: 0.85rem;">
                {block['avg_footfall']:.0f} vs {block['expected_footfall']:.0f} bezoekers | 
                Ged. omzet: €{block['missed_revenue_est']:,.0f}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"**Totaal week:** {dead_hours_result['total_dead_hours_week']:.0f} dode uren, **€{dead_hours_result['total_missed_revenue_week']:,.0f}** gederfde omzet")
else:
    st.success("Geen dode uren gedetecteerd — alle uren boven de 60% drempel.")

st.divider()

# Action Recommendations
st.markdown("### 💡 Top 3 Aanbevelingen")

if recommendations:
    for i, action in enumerate(recommendations, 1):
        effort_badge = {"low": "🟢 Laag", "medium": "🟡 Gemiddeld", "high": "🔴 Hoog"}.get(action["effort"], action["effort"])
        cost = f"€{action['cost_est']:,.0f}" if action["cost_est"] > 0 else "Geen kosten"

        st.markdown(f"""
        <div style="
            border: 1px solid {PFM_LINE};
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            background: white;
        ">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-weight: 700;">{i}. {action['type'].capitalize()} actie</span>
                <span style="font-size: 0.85rem; color: {PFM_GRAY};">
                    +{action['expected_health_score_impact']} punten | {effort_badge} | {cost}
                </span>
            </div>
            <div style="margin-top: 0.5rem;">{action['description']}</div>
            <div style="font-size: 0.85rem; color: {PFM_GRAY}; margin-top: 0.5rem;">
                Verwachte lift: +{action['footfall_lift_pct']}% bezoekers, +{action['conversion_lift_pct']}% conversie
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Geen specifieke acties nodig — de winkel is gezond.")

st.divider()

# Footer
st.caption(f"Store Health Score MVP — PFM Retail Tools | Segment: {segment} | Store ID: {store_id}")
