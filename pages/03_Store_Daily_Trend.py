
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

from helpers_shop import ID_TO_NAME, NAME_TO_ID, REGIONS, get_ids_by_region, get_name_by_id, get_region_by_id
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

st.title("📈 Store Daily Trend")

sname = st.selectbox("Winkel", list(NAME_TO_ID.keys()), index=0)
shop_id = NAME_TO_ID[sname]
kpis = ["count_in","conversion_rate","turnover","sales_per_visitor","sales_per_sqm"]
period = st.selectbox("Periode", ["this_month","last_month","last_week","this_quarter","this_year"], index=0)

params = [("data", shop_id)]
for k in kpis: params.append(("data_output", k))
params += [("source","shops"), ("period", period)]

with st.spinner("Data ophalen..."):
    js = api_get_report(params)

if friendly_error(js, period): st.stop()
df = normalize_vemcount_response(js, SHOP_NAME_MAP)

# basic cards
tot_turn = df["turnover"].sum() if "turnover" in df else 0
avg_conv = df["conversion_rate"].mean() if "conversion_rate" in df else 0
avg_spv  = df["sales_per_visitor"].mean() if "sales_per_visitor" in df else 0
avg_spsqm= df["sales_per_sqm"].mean() if "sales_per_sqm" in df else 0

c1,c2,c3,c4 = st.columns(4)
c1.metric("💶 Omzet (periode)", fmt_eur(tot_turn))
c2.metric("🛒 Gem. conversie", fmt_pct(avg_conv,1))
c3.metric("💸 Gem. SPV", f"{avg_spv:.2f}")
c4.metric("🏁 Gem. sales/m²", f"{avg_spsqm:.2f}")

st.subheader("Dagelijks verloop")
st.dataframe(df, use_container_width=True)

# simple trend advice
advice = []
if "count_in" in df and "turnover" in df and not df.empty:
    # compute daily SPV realized
    x = df.copy()
    x["spv_real"] = x["turnover"] / x["count_in"].replace({0:np.nan})
    last = x.tail(3)
    if last["spv_real"].mean() < x["spv_real"].mean()*0.9:
        advice.append("**Dalende omzet per bezoeker in laatste dagen** → herzie pricing/promoties of leg focus op cross‑sell.")
    if last["count_in"].mean() > x["count_in"].mean()*1.2 and ("conversion_rate" in x and last["conversion_rate"].mean() < x["conversion_rate"].mean()*0.95):
        advice.append("**Meer traffic maar lagere conversie** → stuur extra personeel naar servicepunten in druktes.")

st.subheader("🤖 AI Trend Coach")
if advice:
    for a in advice: st.markdown(f"- {a}")
else:
    st.markdown("Geen opvallende negatieve trends in de laatste week.")

with st.expander("🔧 Debug"):
    st.code(params)
