# pages/04_Executive_ROI_Scenarios.py
import os, sys
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

# ── Imports from project root ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers_shop import ID_TO_NAME                     # {id -> name}
from utils_pfmx import api_get_report, friendly_error, inject_css
from helpers_normalize import normalize_vemcount_response

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
inject_css()
TZ = ZoneInfo("Europe/Amsterdam")

def fmt_int(n):
    try:
        return f"{int(round(float(n))):,}".replace(",", ".")
    except Exception:
        return "0"

def fmt_eur(n, d=0):
    try:
        return f"€{float(n):,.{d}f}".replace(",", ".")
    except Exception:
        return "€0"

def fmt_pct(x, digits=1):
    try:
        return f"{float(x):.{digits}f}%"
    except Exception:
        return "0.0%"

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("💼 Executive ROI Scenarios")

period = st.selectbox(
    "Periode",
    ["last_month", "this_quarter", "last_quarter", "this_year", "last_year"],
    index=0,
    key="roi_period",
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    # Conversie in **procentpunten** (pp). Voorbeeld: +5.0 betekent +5pp.
    conv_add_pp = st.slider("Conversie uplift (+pp)", 0.0, 20.0, 5.0, 0.5, key="roi_conv_pp")
with c2:
    atv_uplift_pct = st.slider("ATV-uplift (%)", 0, 50, 10, 1, key="roi_atv_uplift") / 100.0
with c3:
    gross_margin = st.slider("Brutomarge (%)", 20, 80, 55, 1, key="roi_margin") / 100.0
with c4:
    capex = st.number_input("CAPEX per store (€)", min_value=0, value=1500, step=100, key="roi_capex")

payback_target = st.slider("Payback-target (mnd)", 6, 24, 12, 1, key="roi_payback_target")

# ── Data ophalen ───────────────────────────────────────────────────────────────
KPI_KEYS = ["count_in", "conversion_rate", "turnover", "sales_per_visitor"]
params = []
for sid in ID_TO_NAME.keys():
    params.append(("data", sid))
for k in KPI_KEYS:
    params.append(("data_output", k))
params += [("source", "shops"), ("period", period), ("step", "day")]

js = api_get_report(params)
if friendly_error(js, period):
    st.stop()

# normalize expects id->name map
df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=KPI_KEYS)

if df is None or df.empty:
    st.warning("Geen data ontvangen voor deze periode/parameters.")
    with st.expander("🔧 Debug"):
        st.write("Params:", params)
        st.write("Normalize → empty DataFrame")
    st.stop()

# Ensure expected columns and numeric dtypes
for col in ["count_in", "conversion_rate", "turnover", "sales_per_visitor"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = np.nan

# Derive 'date' if missing (some envs only have timestamp)
if "date" not in df.columns:
    df["date"] = pd.to_datetime(df.get("timestamp"), errors="coerce").dt.date

# Safety: keep only valid shop rows
df = df[pd.notna(df.get("shop_id"))]

with st.expander("🔧 Debug — fetch result"):
    st.write("Rows:", len(df), "Shops:", df["shop_id"].nunique())
    st.dataframe(df.head(12))

# ── Baseline aggregatie per winkel ─────────────────────────────────────────────
# Weighted averages (naar bezoekers) voor conversie en SPV
def _weighted_avg(x, w):
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    denom = w.sum()
    return (x * w).sum() / denom if denom > 0 else np.nan

g = (
    df.groupby(["shop_id", "shop_name"], as_index=False)
      .agg(count_in=("count_in", "sum"),
           turnover=("turnover", "sum"))
)

w_conv = (
    df.groupby(["shop_id"])
      .apply(lambda x: _weighted_avg(x["conversion_rate"], x["count_in"]))
      .reset_index()
      .rename(columns={0: "conversion_rate"})
)

w_spv = (
    df.groupby(["shop_id"])
      .apply(lambda x: _weighted_avg(x["sales_per_visitor"], x["count_in"]))
      .reset_index()
      .rename(columns={0: "sales_per_visitor"})
)

base = g.merge(w_conv, on="shop_id", how="left").merge(w_spv, on="shop_id", how="left")

# ATV = SPV / (conv%/100)
conv_frac = pd.to_numeric(base["conversion_rate"], errors="coerce") / 100.0
base["ATV"] = np.where(conv_frac > 0, base["sales_per_visitor"] / conv_frac, 0.0)

# ── Scenario (Optie 1: uplift op conversie in **pp** én uplift op **ATV**) ─────
# Nieuwe conversie in % met klamp 0..100
new_conv_pct = np.clip(base["conversion_rate"] + conv_add_pp, 0.0, 100.0)
new_conv_frac = new_conv_pct / 100.0

# Nieuwe ATV met uplift
new_atv = base["ATV"] * (1.0 + atv_uplift_pct)

# Nieuwe SPV = (nieuwe conversie) × (nieuwe ATV)
new_spv = new_conv_frac * new_atv

# Extra omzet en brutowinst
extra_rev = base["count_in"] * (new_spv - base["sales_per_visitor"])
extra_gp  = extra_rev * gross_margin

# ── Payback / ROI ─────────────────────────────────────────────────────────────
stores = int(base["shop_id"].nunique())
total_capex = capex * stores

days = df["date"].nunique() if "date" in df.columns else 30
months = max(1.0, days / 30.44)

monthly_gp = (extra_gp.sum() / months) if months > 0 else extra_gp.sum()
payback_months = (total_capex / monthly_gp) if monthly_gp > 0 else np.inf
roi_pct = ((extra_gp.sum() - total_capex) / total_capex * 100.0) if total_capex > 0 else 0.0

# ── KPI cards ─────────────────────────────────────────────────────────────────
kc1, kc2, kc3, kc4 = st.columns(4)
kc1.metric("📈 Extra omzet (scenario)", fmt_eur(extra_rev.sum()))
kc2.metric("💵 Extra brutowinst", fmt_eur(extra_gp.sum()))
kc3.metric("⏳ Payback", "∞ mnd" if np.isinf(payback_months) else f"{payback_months:.1f} mnd",
           delta=f"Target {payback_target} mnd")
kc4.metric("📊 ROI", fmt_pct(roi_pct, 1))

# ── Tabel (top 10 extra brutowinst) ───────────────────────────────────────────
st.subheader("Stores (top 10 extra brutowinst)")

view = base.copy()
view["extra_revenue"]      = extra_rev
view["extra_gross_profit"] = extra_gp
view = view.sort_values("extra_gross_profit", ascending=False).head(10)

# Pretty columns
out = view[["shop_name", "count_in", "sales_per_visitor", "conversion_rate", "ATV", "extra_gross_profit"]].copy()
out.rename(columns={
    "shop_name": "winkel",
    "count_in": "bezoekers",
    "sales_per_visitor": "SPV",
    "conversion_rate": "conversie",
    "extra_gross_profit": "extra_brutowinst",
}, inplace=True)

out["bezoekers"]       = out["bezoekers"].map(fmt_int)
out["SPV"]             = out["SPV"].map(lambda x: fmt_eur(x, 2))
out["conversie"]       = out["conversie"].map(lambda x: fmt_pct(x, 2))
out["ATV"]             = out["ATV"].map(lambda x: fmt_eur(x, 2))
out["extra_brutowinst"]= out["extra_brutowinst"].map(fmt_eur)

st.dataframe(out, use_container_width=True)

# ── Board tips ────────────────────────────────────────────────────────────────
st.subheader("🤖 Board Tips")
st.markdown(
    f"- **+{conv_add_pp:.1f} pp conversie** en **+{atv_uplift_pct*100:.0f}% ATV** leveren "
    f"**{fmt_eur(extra_gp.sum())}** extra brutowinst op in ~{months:.1f} mnd; "
    f"payback ≈ **{('∞' if np.isinf(payback_months) else f'{payback_months:.1f}')} mnd**."
)
st.markdown("- Richt CAPEX op winkels met **hoog verkeer maar lage SPV/conversie** (zie Region Performance).")

# ── Extra debug ───────────────────────────────────────────────────────────────
with st.expander("🔧 Debug — berekening"):
    st.write("Stores:", stores, "Days in period:", days, "Months (approx):", months)
    st.write("Total CAPEX:", total_capex, "Monthly GP:", monthly_gp)
    st.write("Sum extra_rev:", extra_rev.sum(), "Sum extra_gp:", extra_gp.sum())
    st.dataframe(base.head(10))
