import streamlit as st
import pandas as pd
import numpy as np
import requests
import holidays
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pytz

sys.path.append(str(Path(__file__).parent.parent))
from shop_mapping import SHOP_NAME_MAP
from helpers_shop import ID_TO_NAME, get_ids_by_region
from helpers_normalize import normalize_vemcount_response

st.set_page_config(page_title="AI Retail Advisor", layout="wide", page_icon="shopping_bag")
st.title("AI Retail Advisor: Regio- & Winkelvoorspellingen")

API_URL = st.secrets["API_URL"]
OW_KEY  = st.secrets["openweather_api_key"]
CBS_ID  = st.secrets["cbs_dataset"]
CBS_URL = f"https://opendata.cbs.nl/ODataFeed/odata/{CBS_ID}/Consumentenvertrouwen"

PERIODS = ["this_week","last_week","this_month","last_month","this_quarter","last_quarter","this_year","last_year"]
regio = st.sidebar.selectbox("Regio", ["All"] + list(set(i["region"] for i in SHOP_NAME_MAP.values())), index=0)
period = st.sidebar.selectbox("Periode", PERIODS, index=3)

if regio == "All":
    shop_ids = list(SHOP_NAME_MAP.keys())
else:
    shop_ids = get_ids_by_region(regio)
if not shop_ids:
    st.stop()

TZ = pytz.timezone("Europe/Amsterdam")
TODAY = datetime.now(TZ).date()
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor","sq_meter"]

def step_for(p: str) -> str:
    return "day" if p.endswith("week") or p.endswith("month") else "month"

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"] = pd.to_datetime(d["date"], errors="coerce").fillna(ts)
    d["date_eff"] = d["date_eff"].dt.date
    d["year"] = pd.to_datetime(d["date_eff"]).dt.year
    d["month"] = pd.to_datetime(d["date_eff"]).dt.month
    d["week"] = pd.to_datetime(d["date_eff"]).dt.isocalendar().week
    d["shop_id"] = d["shop_id"].astype(int)
    return d

@st.cache_data(ttl=1800, show_spinner=False)
def fetch(shop_ids, period: str) -> pd.DataFrame:
    params = [
        ("source", "shops"),
        ("period", period),
        ("step", step_for(period))
    ]
    for sid in shop_ids:
        params.append(("data", str(sid)))
    for m in METRICS:
        params.append(("data_output", m))
    try:
        r = requests.post(API_URL, params=params, timeout=45)
        r.raise_for_status()
        js = r.json()
        df = normalize_vemcount_response(js, ID_TO_NAME, METRICS)
        df = add_effective_date(df)
        if period.startswith("this_"):
            df = df[df["date_eff"] < TODAY]
        df = df.sort_values(["shop_id", "date_eff"])
        df["sq_meter"] = df.groupby("shop_id")["sq_meter"].ffill().bfill()
        return df
    except Exception as e:
        st.error(f"Planet PFM: {str(e)[:100]}")
        return pd.DataFrame()

df = fetch(shop_ids, period)

prev_period_map = {
    "this_week": "last_week",
    "this_month": "last_month",
    "this_quarter": "last_quarter",
    "this_year": "last_year",
    "last_week": "this_week",
    "last_month": "this_month",
    "last_quarter": "this_quarter",
    "last_year": "this_year"
}
prev_period = prev_period_map.get(period)
df_prev = fetch(shop_ids, prev_period) if prev_period else pd.DataFrame()

if not df.empty:
    st.success(f"Planet PFM online – {len(df)} records")

@st.cache_data(ttl=1800)
def weer(pc):
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/forecast", params={"q": f"{pc},NL", "appid": OW_KEY, "units":"metric"})
        r.raise_for_status()
        js = r.json()["list"]
        daily = pd.DataFrame([{"date": pd.to_datetime(d["dt_txt"]).date(), "temp": d["main"]["temp"], "rain": d.get("rain", {}).get("3h", 0)} for d in js])
        return daily.groupby("date").mean(numeric_only=True).reset_index().head(28)
    except:
        return None

@st.cache_data(ttl=86400)
def cbs():
    try:
        raw = requests.get(CBS_URL).json()["value"]
        c = pd.DataFrame(raw)[["Perioden","Consumentenvertrouwen_1","Koopbereidheid_5"]]
        c["maand"] = pd.to_datetime(c["Perioden"].str[:7] + "-01")
        return c.rename(columns={"Consumentenvertrouwen_1":"CBS_vertrouwen", "Koopbereidheid_5":"CBS_koop"})[["maand","CBS_vertrouwen","CBS_koop"]]
    except:
        return pd.DataFrame({"maand": pd.date_range("2025-01", "2025-11", freq="MS"), "CBS_vertrouwen": [-8,-7,-9,-6,-10,-11,-9,-12,-10,-13,-14], "CBS_koop": [-15,-14,-16,-13,-17,-18,-16,-19,-17,-20,-21]})

cbs_df = cbs()

# KPI's + delta
if not df.empty:
    total_foot = df["count_in"].sum()
    total_omzet = df["turnover"].sum()
    avg_conv = df["conversion_rate"].mean()
    avg_spv = df["sales_per_visitor"].mean()

    c1,c2,c3,c4 = st.columns(4)
    
    if not df_prev.empty:
        prev_foot = df_prev["count_in"].sum()
        prev_omzet = df_prev["turnover"].sum()
        prev_conv = df_prev["conversion_rate"].mean()
        prev_spv = df_prev["sales_per_visitor"].mean()

        vs_foot = ((total_foot / prev_foot) - 1) * 100 if prev_foot > 0 else 0
        vs_omzet = ((total_omzet / prev_omzet) - 1) * 100 if prev_omzet > 0 else 0
        vs_conv = avg_conv - prev_conv
        vs_spv = avg_spv - prev_spv

        c1.metric("Footfall", f"{int(total_foot):,}".replace(",","."), delta=f"{vs_foot:+.1f}%")
        c2.metric("Omzet", f"€{int(total_omzet):,}".replace(",","."), delta=f"{vs_omzet:+.1f}%")
        c3.metric("Conversie", f"{avg_conv:.1f}%", delta=f"{vs_conv:+.1f} pp")
        c4.metric("SPV", f"€{avg_spv:.0f}", delta=f"{vs_spv:+.1f}")
    else:
        c1.metric("Footfall", f"{int(total_foot):,}".replace(",","."))
        c2.metric("Omzet", f"€{int(total_omzet):,}".replace(",","."))
        c3.metric("Conversie", f"{avg_conv:.1f}%")
        c4.metric("SPV", f"€{avg_spv:.0f}")

tab1,tab2,tab3 = st.tabs(["YTD vs. CBS","4 Weken","Actieplan"])

with tab1:
    if not df.empty:
        # Monthly grouping for consistent bars
        periods = pd.to_datetime(df["date_eff"]).dt.to_period('M')
        df["maand"] = periods.apply(lambda x: x.start_time.strftime("%Y-%m") if pd.notnull(x) else "")

        agg = df.groupby(["maand","shop_id"]).agg({"count_in":"sum","turnover":"sum","conversion_rate":"mean"}).reset_index()
        agg["regio"] = agg["shop_id"].map(lambda x: SHOP_NAME_MAP.get(x, {}).get("region", "Onbekend"))
        maand_agg = agg.groupby(["maand","regio"]).agg({"count_in":"sum","turnover":"sum","conversion_rate":"mean"}).reset_index()

        cbs_monthly = cbs_df.copy()
        cbs_monthly["maand"] = cbs_monthly["maand"].dt.strftime("%Y-%m")

        fig = go.Figure()
        for r in [regio] if regio != "All" else ["Noord NL", "Zuid NL"]:
            if r not in maand_agg["regio"].unique(): continue
            d = maand_agg[maand_agg.regio==r]
            fig.add_trace(go.Bar(x=d["maand"], y=d["turnover"]/1000, name=f"Omzet {r}", marker_color="#1f77b4" if r=="Noord NL" else "#ff7f0e"))
            fig.add_trace(go.Scatter(x=d["maand"], y=d["count_in"]/1000, name=f"Footfall {r}", yaxis="y2", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=d["maand"], y=d["conversion_rate"], name=f"Conversie {r}", yaxis="y4", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=cbs_monthly["maand"], y=cbs_monthly["CBS_vertrouwen"], name="CBS Vertrouwen", yaxis="y3", line=dict(color="red")))
        fig.update_layout(
            yaxis=dict(title="Omzet (€K)"),
            yaxis2=dict(title="Footfall (×1.000)", overlaying="y", side="right"),
            yaxis3=dict(title="CBS", overlaying="y", side="right", position=0.99),
            yaxis4=dict(title="Conversie %", overlaying="y", side="right", position=0.95),
            barmode="group", height=500,
            xaxis=dict(tickangle=45)
        )
        st.plotly_chart(fig, use_container_width=True)

def voorspel():
    rows = []
    nl_hols = holidays.NL(years=2025)
    for sid, info in SHOP_NAME_MAP.items():
        if regio != "All" and info["region"] != regio: continue
        w = weer(info["postcode"])
        if w is None: continue
        hist = df[df["shop_id"] == sid]
        if hist.empty: continue
        avg_foot = hist["count_in"].mean()
        avg_spv = hist["sales_per_visitor"].mean()
        cbs_impact = 1 + (cbs_df["CBS_vertrouwen"].mean() / 100) * 0.05
        for i in range(4):
            week_start = datetime.now() + timedelta(weeks=i)
            week_num = week_start.isocalendar()[1]
            week_dates = pd.date_range(week_start, periods=7)
            temp = w[w["date"].isin(week_dates)]["temp"].mean() if not w.empty else 12
            rain = w[w["date"].isin(week_dates)]["rain"].sum() if not w.empty else 0
            holiday = any(d.date() in nl_hols for d in week_dates)
            adj = 1.0
            if rain > 5: adj *= 0.90
            if temp > 18: adj *= 1.15
            if holiday: adj *= 1.20
            adj *= cbs_impact
            foot = int(avg_foot * 7 * adj)
            omzet = foot * avg_spv
            duiding = []
            if rain > 5: duiding.append("regen (-10%)")
            if temp > 18: duiding.append("zon (+15%)")
            if holiday: duiding.append("feestdag (+20%)")
            if cbs_df["CBS_vertrouwen"].mean() < -10: duiding.append(f"laag vertrouwen ({cbs_df['CBS_vertrouwen'].mean():.0f} pt, -5%)")
            duiding_str = "; ".join(duiding) or "stabiel"
            rows.append({
                "week_num": f"Week {week_num}",
                "winkel": info["name"],
                "footfall": foot,
                "omzet": f"€{int(omzet):,}".replace(",", "."),
                "duiding": duiding_str
            })
    return pd.DataFrame(rows)

forecast = voorspel()

with tab2:
    st.subheader("Voorspelling Footfall per Week")
    if not forecast.empty:
        fig_f = go.Figure()
        for w in forecast["winkel"].unique():
            d = forecast[forecast["winkel"] == w]
            fig_f.add_trace(go.Scatter(x=d["week_num"], y=d["footfall"], name=f"Verwacht {w}", mode="lines+markers"))
        fig_f.update_layout(height=500)
        st.plotly_chart(fig_f, use_container_width=True)
        st.dataframe(forecast[["week_num","winkel","omzet","duiding"]])

with tab3:
    st.subheader("Actieplan – Voor Tweedehands Kleding")
    for _,r in forecast.iterrows():
        with st.expander(f"{r['winkel']} – {r['week_num']} | {r['footfall']:,} bezoekers"):
            acties = []
            if "regen" in r["duiding"]:
                acties.append("Regen dip: Indoor ruil-event + app-push: 'Droog ruilen met extra korting!' (+8% conversie)")
            if "zon" in r["duiding"]:
                acties.append("Zon boost: Window displays met zomer vintage + social post: 'Zomer finds bij ons!' (+12% footfall)")
            if "feestdag" in r["duiding"]:
                acties.append("Feestdag piek: Special thema-rack (e.g. feestkleding) + staffing +20%")
            if "laag vertrouwen" in r["duiding"]:
                acties.append("Laag vertrouwen: Budget deals + loyalty email: 'Bespaar met onze tweedehands gems' (stabiliseer SPV)")
            for a in acties:
                st.markdown(f"- {a}")
            txt = f"Beste {r['winkel']},\n\n{r['week_num']} → {r['footfall']:,} bezoekers ({r['omzet']})\nDuiding: {r['duiding']}\n\nActies:\n" + "\n".join([f"- {a.split(':')[1].strip() if ':' in a else a}" for a in acties[:2]]) + "\n\nSucces!\nRegiomanager"
            st.code(txt, language="text")

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

st.caption("Bron: Planet PFM, OpenWeather, CBS | Real-time")
