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

st.title("🧭 Region Performance — radar")

sel_names = st.multiselect("Winkels", list(NAME_TO_ID.keys()), default=list(NAME_TO_ID.keys())[:3])
shop_ids = [NAME_TO_ID[n] for n in sel_names] or [next(iter(ID_TO_NAME.keys()))]

kpis = ["count_in","conversion_rate","turnover","sales_per_visitor","sales_per_sqm"]
period = st.selectbox("Periode", ["last_week","this_month","last_month","this_quarter","last_quarter","this_year","last_year"], index=0)

params = []
for sid in shop_ids: params.append(("data", sid))
for k in kpis: params.append(("data_output", k))
params += [("source","shops"), ("period", period)]

with st.spinner("Data ophalen..."):
    js = api_get_report(params)

if friendly_error(js, period):
    st.stop()

df = normalize_vemcount_response(js, SHOP_NAME_MAP)
wide = to_wide(df)

# Aggregate metrics
avg_conv = wide["conversion_rate"].mean() if "conversion_rate" in wide else 0
avg_spv  = wide["sales_per_visitor"].mean() if "sales_per_visitor" in wide else 0
avg_spsqm= wide["sales_per_sqm"].mean() if "sales_per_sqm" in wide else 0
tot_turn = wide["turnover"].sum() if "turnover" in wide else 0

# Identify best/worst by sales_per_sqm
best = None; worst = None
if "sales_per_sqm" in wide:
    wsort = wide.sort_values("sales_per_sqm", ascending=False)
    best = wsort.head(1)
    worst= wsort.tail(1)

c1,c2,c3,c4 = st.columns(4)
c1.metric("💶 Totale omzet", fmt_eur(tot_turn))
c2.metric("🛒 Gem. conversie", fmt_pct(avg_conv,1))
c3.metric("💸 Gem. SPV", f"{avg_spv:.2f}")
c4.metric("🏁 Gem. sales/m²", f"{avg_spsqm:.2f}")

st.subheader("🏆 Tops & Flops (sales/m²)")
colA,colB = st.columns(2)
with colA:
    if best is not None and not best.empty:
        r = best.iloc[0]
        st.success(f"Top: **{r['shop_name']}** — {r['sales_per_sqm']:.2f} €/m²")
with colB:
    if worst is not None and not worst.empty:
        r = worst.iloc[0]
        st.error(f"Bottom: **{r['shop_name']}** — {r['sales_per_sqm']:.2f} €/m²")

# AI tips per store: traffic high + conv low; conv high + SPV low
tips = []
if not wide.empty:
    # Build helper columns
    w = wide.copy()
    # normalize
    if "count_in" in w: 
        w["count_idx"] = (w["count_in"] - w["count_in"].mean())/w["count_in"].std(ddof=0).replace({0:1})
    else:
        w["count_idx"] = 0
    if "conversion_rate" in w:
        w["conv_idx"] = (w["conversion_rate"] - w["conversion_rate"].mean())/w["conversion_rate"].std(ddof=0).replace({0:1})
    else:
        w["conv_idx"] = 0
    if "sales_per_visitor" in w:
        w["spv_idx"] = (w["sales_per_visitor"] - w["sales_per_visitor"].mean())/w["sales_per_visitor"].std(ddof=0).replace({0:1})
    else:
        w["spv_idx"] = 0

    hi_tr_lo_conv = w[(w["count_idx"]>0.5) & (w["conv_idx"]<-0.3)]
    if not hi_tr_lo_conv.empty:
        names = ", ".join(hi_tr_lo_conv["shop_name"].head(5).tolist())
        tips.append(f"**Hoog verkeer, lage conversie** → {names}. **Actie:** floor training / demo's, manager op vloer in piekuren.")

    hi_conv_lo_spv = w[(w["conv_idx"]>0.5) & (w["spv_idx"]<-0.3)]
    if not hi_conv_lo_spv.empty:
        names = ", ".join(hi_conv_lo_spv["shop_name"].head(5).tolist())
        tips.append(f"**Hoge conversie, lage SPV** → {names}. **Actie:** bundels/upsell, kassadisplays, prijsarchitectuur.")

st.subheader("🤖 AI Region Coach")
if tips:
    for t in tips: st.markdown(f"- {t}")
else:
    st.markdown("Geen duidelijke outliers — regio presteert consistent.")

with st.expander("🔧 Debug"):
    st.code(params)
    st.dataframe(wide)
