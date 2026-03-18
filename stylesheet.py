# stylesheet.py
import streamlit as st


def get_css(
    *,
    PFM_PURPLE: str,
    PFM_RED: str,
    PFM_DARK: str,
    PFM_GRAY: str,
    PFM_LIGHT: str,
    PFM_LINE: str,
) -> str:
    """
    Returns the full CSS string (no <style> tags). Keep this file as the single source of truth.
    """

    return f"""
/* ---------------- Layout ---------------- */
.block-container {{
  padding-top: 2.25rem;
  padding-bottom: 2rem;
}}

/* ---------------- Header (title card) ---------------- */
.pfm-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  border: 1px solid {PFM_LINE};
  border-radius: 14px;
  background: white;
  margin-bottom: 0.75rem;
}}

.pfm-header--fixed {{
  height: 92px;
  display: flex;
  align-items: center;
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

/* ---------------- Panels ---------------- */
.panel {{
  border: 1px solid {PFM_LINE};
  border-radius: 14px;
  background: white;
  padding: 0.55rem 0.75rem;
}}

.panel-title {{
  font-weight: 800;
  color: {PFM_DARK};
  margin-bottom: 0.25rem;
}}

/* ---------------- KPI Cards ---------------- */
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

/* ---------------- Pill / text ---------------- */
.pill {{
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 800;
  border: 1px solid {PFM_LINE};
  background: {PFM_LIGHT};
  color: {PFM_DARK};
}}

.muted {{
  color: {PFM_GRAY};
  font-size: 0.86rem;
}}

.hint {{
  color: {PFM_GRAY};
  font-size: 0.82rem;
}}

/* ---------------- Callout ---------------- */
.callout {{
  border: 1px solid {PFM_LINE};
  border-radius: 14px;
  background: #fff7ed;
  padding: 0.75rem 1rem;
}}
.callout-title {{
  font-weight: 900;
  color: {PFM_DARK};
  margin-bottom: 0.15rem;
}}
.callout-sub {{
  color: {PFM_GRAY};
  font-size: 0.86rem;
}}

/* ---------------- Buttons (global) ---------------- */
div.stButton > button {{
  background: {PFM_RED} !important;
  color: white !important;
  border: 0 !important;
  border-radius: 12px !important;
  padding: 0.65rem 1rem !important;
  font-weight: 800 !important;
}}

/* ---------------- Optional: compact select labels ---------------- */
/* (Laat dit staan: het voorkomt 'lege labelruimte' boven sommige selects) */
div[data-testid="stSelectbox"] label {{
  display: none !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}}

div[data-testid="stSelectbox"] > div {{
  min-height: 44px !important;
}}
div[data-testid="stSelectbox"] div[role="combobox"] {{
  min-height: 44px !important;
  display: flex !important;
  align-items: center !important;
}}
"""

def inject_css(**colors) -> None:
    """
    Injects CSS into the Streamlit app. Call once near the top of each page.
    """
    css = get_css(**colors)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

import altair as alt

def pfm_altair(chart: alt.Chart, *, height: int | None = None) -> alt.Chart:
    """
    Standard PFM Altair styling.
    Apply this on the FINAL chart object (after layering / concatenation).
    """
    if height is not None:
        chart = chart.properties(height=height)

    return (
        chart
        .configure_view(strokeWidth=0)
        # Axis defaults (both)
        .configure_axis(
            domain=False,
            ticks=False,
            grid=True,
        )
        # Y-axis: fix left whitespace & label alignment
        .configure_axisY(
            labelFlush=True,
            labelPadding=8,
        )
        # X-axis: keep it readable
        .configure_axisX(
            labelPadding=6,
            titlePadding=10,
        )
        # Optional: reduce chart padding inside Vega canvas
        .configure(padding=0)
    )
