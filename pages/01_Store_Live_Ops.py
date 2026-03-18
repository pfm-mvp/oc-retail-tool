import os, sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

# Ensure root importable (shop_mapping.py in root)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers_shop import ID_TO_NAME, REGIONS, get_ids_by_region
from utils_pfmx import api_get_report, friendly_error, inject_css
from helpers_normalize import normalize_vemcount_response, to_wide

st.set_page_config(layout="wide")
inject_css()
TZ = ZoneInfo("Europe/Amsterdam")
NAME_TO_ID = {v:k for k,v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k:v for k,v in SHOP_NAME_MAP.items()}

def fmt_int(n): 
    try: return f"{int(round(float(n))):,}".replace(",", ".")
    except: return "0"

def fmt_eur(n):
    try: return f"€{float(n):,.0f}".replace(",", ".")
    except: return "€0"

def fmt_pct(x, digits=1):
    try: return f"{float(x):.{digits}f}%"
    except: return "0.0%"

st.title("👣 Store Live Ops — vandaag vs. gisteren")

names = list(NAME_TO_ID.keys())
sname = st.selectbox("Winkel", names, index=0)
shop_id = NAME_TO_ID[sname]

open_t, close_t = st.slider("Venster vandaag & gisteren", value=(time(9,0), time(18,0)), format="HH:mm")
now = datetime.now(TZ).time()
now_cut = max(open_t, min(now, close_t))
h_from = open_t.strftime("%H:%M"); h_to = now_cut.strftime("%H:%M")

# KPIs we want for cards
kpis = ["count_in","conversion_rate","turnover","sales_per_visitor"]

base = [("data", shop_id)]
for k in kpis: base.append(("data_output", k))
base += [("source","shops"), ("show_hours_from", h_from), ("show_hours_to", h_to)]

with st.spinner("Data ophalen..."):
    js_today = api_get_report([("period","day")] + base)
    js_yest  = api_get_report([("period","yesterday")] + base)

if friendly_error(js_today, "day") or friendly_error(js_yest, "yesterday"):
    st.stop()

df_t = normalize_vemcount_response(js_today, SHOP_NAME_MAP)
df_y = normalize_vemcount_response(js_yest,  SHOP_NAME_MAP)

def _safe_sum(df, col):
    return float(df[col].sum()) if (df is not None and not df.empty and col in df.columns) else 0.0

vis_t = _safe_sum(df_t, "count_in"); vis_y = _safe_sum(df_y, "count_in")
turn_t = _safe_sum(df_t, "turnover"); turn_y = _safe_sum(df_y, "turnover")
# conversie als gemiddelde (niet sommen)
conv_t = float(df_t["conversion_rate"].mean()) if ("conversion_rate" in df_t.columns and not df_t.empty) else 0.0
conv_y = float(df_y["conversion_rate"].mean()) if ("conversion_rate" in df_y.columns and not df_y.empty) else 0.0

# Cards
c1,c2,c3,c4 = st.columns(4)
c1.metric("👥 Bezoekers (tot nu)", fmt_int(vis_t), delta=f"{fmt_int(vis_t - vis_y)} vs gister")
c2.metric("🛒 Conversie", fmt_pct(conv_t,1), delta=f"{fmt_pct(conv_t - conv_y,1)} vs gister")
c3.metric("💶 Omzet (tot nu)", fmt_eur(turn_t), delta=f"{fmt_eur(turn_t - turn_y)} vs gister")
prod_t = (turn_t / vis_t) if vis_t>0 else 0.0
prod_y = (turn_y / vis_y) if vis_y>0 else 0.0
c4.metric("💸 Sales/visitor", f"{prod_t:.2f}", delta=f"{(prod_t - prod_y):+.2f} vs gister")

st.divider()

# AI alerts
alerts = []
if vis_t > vis_y * 1.1 and conv_t < conv_y * 0.98:
    alerts.append("Traffic is **hoog** (+10% vs gister) maar conversie zakt (~−2%). → **Zet extra medewerker op vloer of paskamers.**")
if prod_t < prod_y * 0.95 and vis_t >= vis_y:
    alerts.append("Omzet per bezoeker loopt achter bij gelijke/hogere traffic. → **Push cross‑sell bij kassa, activeer kleine bundelkorting.**")
if now_cut < close_t and vis_t < (vis_y * (now_cut.hour - open_t.hour + 1)/(close_t.hour - open_t.hour + 1)) * 0.9:
    alerts.append("Je ligt **achter op het dagtempo** vs gisteren. → **Activeer 2‑uur micro‑promo** op trage categorie.")

st.subheader("🤖 AI Coach")
if alerts:
    for a in alerts: st.markdown(f"- {a}")
else:
    st.markdown("Alles **on track** t.o.v. gisteren binnen dit venster.")

with st.expander("🔧 Debug"):
    st.code([('period','day')]+base)
    st.code([('period','yesterday')]+base)
    st.dataframe(df_t)
