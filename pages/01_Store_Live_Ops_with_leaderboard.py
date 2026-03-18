
import os, sys
from datetime import datetime
import pytz
import requests
import pandas as pd
import streamlit as st

# ---------- Imports / mapping ----------
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from shop_mapping import SHOP_NAME_MAP
from helpers_normalize import normalize_vemcount_response

st.set_page_config(page_title="Store Live Ops — Gisteren vs Eergisteren + Leaderboard", page_icon="🛍️", layout="wide")
st.title("🛍️ Store Live Ops — Gisteren vs Eergisteren + Leaderboard")

API_URL = st.secrets["API_URL"]

# ---------- Colors ----------
PFM_RED = "#F04438"
PFM_GREEN = "#22C55E"
PFM_PURPLE = "#6C4EE3"
PFM_GRAY = "#6B7280"
PFM_GRAY_BG = "rgba(107,114,128,.10)"

# ---------- Small CSS for cards ----------
st.markdown(f"""
<style>
.kpi-card {{ border: 1px solid #EEE; border-radius: 14px; padding: 18px 18px 14px 18px; }}
.kpi-title {{ color:#0C111D; font-weight:600; font-size:16px; margin-bottom:8px; }}
.kpi-value {{ font-size:40px; font-weight:800; line-height:1.1; margin-bottom:6px; }}
.kpi-delta {{ font-size:14px; font-weight:700; padding:4px 10px; border-radius:999px; display:inline-block; }}
.kpi-delta.up {{ color:{PFM_GREEN}; background: rgba(34,197,94,.10); }}
.kpi-delta.down {{ color:{PFM_RED}; background: rgba(240,68,56,.10); }}
.kpi-delta.flat {{ color:{PFM_GRAY}; background: {PFM_GRAY_BG}; }}
.lb-card {{ border: 1px dashed #DDD; border-radius: 12px; padding: 12px 14px; margin-bottom: 8px; background: #FAFAFC; }}
.lb-title {{ font-size:14px; color:#0C111D; font-weight:600; }}
.lb-val {{ font-size:18px; font-weight:800; }}
</style>
""", unsafe_allow_html=True)

NAME_TO_ID = {v:k for k,v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k:v for k,v in SHOP_NAME_MAP.items()}

store_name = st.selectbox("Kies winkel", list(NAME_TO_ID.keys()), index=0)
store_id = NAME_TO_ID[store_name]

METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor"]
TZ = pytz.timezone("Europe/Amsterdam")
TODAY = datetime.now(TZ).date()

def post_report(params):
    r = requests.post(API_URL, params=params, timeout=45)
    r.raise_for_status()
    return r

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        d["date"] = pd.Series([None]*len(d))
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    date_series = pd.to_datetime(d["date"], errors="coerce")
    d["date_eff"] = date_series.fillna(ts).dt.date
    return d

def fetch_df(shop_ids, period, step, metrics):
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in metrics]
    params += [("source","shops"), ("period", period), ("step", step)]
    resp = post_report(params)
    js = resp.json()
    df = normalize_vemcount_response(js, SHOP_NAME_MAP, kpi_keys=metrics)
    dfe = add_effective_date(df)
    return dfe, params, resp.status_code

# ---------- Cards (gisteren vs eergisteren) ----------
df_cards, p_cards, status_cards = fetch_df([store_id], "this_week", "day", METRICS)
df_cards = df_cards[df_cards["date_eff"] < TODAY]
dates = sorted(df_cards["date_eff"].unique())
if len(dates) < 2:
    df_cards, p_cards, status_cards = fetch_df([store_id], "last_week", "day", METRICS)
    df_cards = df_cards[df_cards["date_eff"] < TODAY]
    dates = sorted(df_cards["date_eff"].unique())

if len(dates) < 2:
    st.error("Niet genoeg dagdata om kaarten te tonen.")
    st.stop()

ydate, bdate = dates[-1], dates[-2]
gy = df_cards[(df_cards["date_eff"]==ydate) & (df_cards["shop_id"]==store_id)][METRICS].sum(numeric_only=True)
gb = df_cards[(df_cards["date_eff"]==bdate) & (df_cards["shop_id"]==store_id)][METRICS].sum(numeric_only=True)

def delta_badge(value: float, kind: str):
    if value > 0: cls, arrow = "up","↑"
    elif value < 0: cls, arrow = "down","↓"
    else: cls, arrow = "flat","→"
    if kind=="int": disp = f"{abs(int(value)):,}".replace(",", ".")
    elif kind=="pct2": disp = f"{abs(value):.2f}%"
    elif kind=="eur0": disp = f"€{abs(value):,.0f}".replace(",", ".")
    elif kind=="eur2": disp = f"€{abs(value):,.2f}".replace(",", ".")
    else: disp = f"{abs(value)}"
    return f'<span class="kpi-delta {cls}">{arrow} <b>{disp}</b> t.o.v. dag ervoor</span>'

c1,c2,c3,c4 = st.columns(4)

with c1:
    val = int(gy.get("count_in", 0))
    diff = gy.get("count_in", 0) - gb.get("count_in", 0)
    st.markdown(f"""
<div class="kpi-card"><div class="kpi-title">Bezoekers <small>(gisteren)</small></div>
<div class="kpi-value">{val:,}</div>{delta_badge(float(diff),"int")}</div>
""".replace(",", "."), unsafe_allow_html=True)

with c2:
    conv_y = float(gy.get("conversion_rate", 0.0))
    conv_b = float(gb.get("conversion_rate", 0.0))
    st.markdown(f"""
<div class="kpi-card"><div class="kpi-title">Conversie <small>(gisteren)</small></div>
<div class="kpi-value">{conv_y:.2f}%</div>{delta_badge((conv_y-conv_b),"pct2")}</div>
""", unsafe_allow_html=True)

with c3:
    turn_y = float(gy.get("turnover", 0.0)); turn_b = float(gb.get("turnover", 0.0))
    st.markdown(f"""
<div class="kpi-card"><div class="kpi-title">Omzet <small>(gisteren)</small></div>
<div class="kpi-value">€{turn_y:,.0f}</div>{delta_badge((turn_y-turn_b),"eur0")}</div>
""".replace(",", "."), unsafe_allow_html=True)

with c4:
    spv_y = float(gy.get("sales_per_visitor", 0.0)); spv_b = float(gb.get("sales_per_visitor", 0.0))
    if spv_y==0 and val>0: spv_y = turn_y/val
    val_b = int(gb.get("count_in", 0))
    if spv_b==0 and val_b>0: spv_b = turn_b/val_b
    st.markdown(f"""
<div class="kpi-card"><div class="kpi-title">Sales per visitor <small>(gisteren)</small></div>
<div class="kpi-value">€{spv_y:,.2f}</div>{delta_badge((spv_y-spv_b),"eur2")}</div>
""".replace(",", "."), unsafe_allow_html=True)

st.markdown("---")

# ---------- Leaderboard (WTD t/m gisteren) ----------
all_ids = list(SHOP_NAME_MAP.keys())

def fetch_wtd(period):
    df, p, s = fetch_df(all_ids, period, "day", METRICS)
    df = df[df["date_eff"] < TODAY]
    return df, p, s

df_this, p_this, s_this = fetch_wtd("this_week")
df_last, p_last, s_last = fetch_wtd("last_week")

st.subheader("🏁 Leaderboard — huidige week (t/m gisteren)")
rank_choice = st.radio("Ranking op basis van", ["Conversie", "SPV"], horizontal=True, index=0)

def wtd_agg(d: pd.DataFrame) -> pd.DataFrame:
    if d is None or d.empty: return pd.DataFrame()
    g = d.groupby("shop_id", as_index=False).agg({"count_in":"sum","turnover":"sum"})
    g["sales_per_visitor"] = g.apply(lambda r: (r["turnover"]/r["count_in"]) if r["count_in"] else 0.0, axis=1)
    conv = d.groupby("shop_id").apply(
        lambda x: (x["conversion_rate"]*x["count_in"]).sum()/x["count_in"].sum() if x["count_in"].sum() else float(x["conversion_rate"].mean())
    ).reset_index()
    conv.columns = ["shop_id","conversion_rate"]
    g = g.merge(conv, on="shop_id", how="left")
    g["shop_name"] = g["shop_id"].map(SHOP_NAME_MAP)
    return g

agg_this = wtd_agg(df_this)
agg_last = wtd_agg(df_last)

if agg_this.empty:
    st.info("Geen WTD data beschikbaar.")
    st.stop()

metric_map = {"Conversie":"conversion_rate", "SPV":"sales_per_visitor"}
rank_metric = metric_map[rank_choice]

agg_this["rank_now"] = agg_this[rank_metric].rank(method="min", ascending=False).astype(int)
tmp = agg_last[["shop_id", rank_metric]].copy()
tmp["rank_last"] = tmp[rank_metric].rank(method="min", ascending=False)
agg = agg_this.merge(tmp[["shop_id","rank_last"]], on="shop_id", how="left")

def safe_int(x):
    try:
        return int(x) if pd.notna(x) else None
    except Exception:
        return None

agg["rank_last_int"] = agg["rank_last"].apply(safe_int)
agg["rank_change"] = agg.apply(lambda r: (r["rank_last_int"] - r["rank_now"]) if r["rank_last_int"] is not None else 0, axis=1)

me = agg[agg["shop_id"]==store_id].iloc[0]
ch = int(me["rank_change"]) if pd.notna(me["rank_change"]) else 0
arrow = "🔺" if ch>0 else ("🔻" if ch<0 else "→")
col = PFM_PURPLE if ch>0 else (PFM_RED if ch<0 else PFM_GRAY)
st.markdown(f"""
<div class="lb-card">
  <div class="lb-title">Jouw positie op <b>{'Conversie' if rank_metric=='conversion_rate' else 'SPV'}</b></div>
  <div class="lb-val" style="color:{col}">#{int(me["rank_now"])} {arrow} {abs(ch)} t.o.v. vorige week</div>
</div>
""", unsafe_allow_html=True)

agg = agg.sort_values("rank_now").reset_index(drop=True)
def pos_text(r):
    last = r["rank_last_int"]; now = int(r["rank_now"])
    last_txt = "–" if last is None else str(last)
    arrow = "⬆" if r["rank_change"]>0 else ("⬇" if r["rank_change"]<0 else "→")
    return f"{now} {arrow} {last_txt}"
agg["positie (nu vs lw)"] = agg.apply(pos_text, axis=1)

show_cols = ["positie (nu vs lw)","shop_name","count_in","conversion_rate","sales_per_visitor","turnover"]
fmt = agg[show_cols].copy()
fmt["count_in"] = fmt["count_in"].map(lambda x: f"{int(x):,}".replace(",", "."))
fmt["conversion_rate"] = fmt["conversion_rate"].map(lambda x: f"{float(x):.2f}%")
fmt["sales_per_visitor"] = fmt["sales_per_visitor"].map(lambda x: f"€{float(x):,.2f}".replace(",", "."))
fmt["turnover"] = fmt["turnover"].map(lambda x: f"€{float(x):,.0f}".replace(",", "."))

try:
    idx_me = fmt.index[agg["shop_name"] == store_name][0]
except Exception:
    idx_me = None

def highlight_rows(row):
    if idx_me is not None and row.name == idx_me:
        ch = int(agg.loc[row.name, "rank_change"]) if pd.notna(agg.loc[row.name, "rank_change"]) else 0
        if ch > 0: bg = PFM_PURPLE
        elif ch < 0: bg = PFM_RED
        else: bg = PFM_GRAY
        return [f"background-color: {bg}; color: white" for _ in row]
    return [""]*len(row)

styler = fmt.style.apply(highlight_rows, axis=1)

def color_pos_col(col):
    styles=[]
    for i,_ in enumerate(col):
        ch = int(agg.loc[fmt.index[i], "rank_change"]) if pd.notna(agg.loc[fmt.index[i], "rank_change"]) else 0
        if ch > 0: styles.append(f"color: {PFM_PURPLE}; font-weight: 700")
        elif ch < 0: styles.append(f"color: {PFM_RED}; font-weight: 700")
        else: styles.append(f"color: {PFM_GRAY}; font-weight: 700")
    return styles

styler = styler.apply(color_pos_col, subset=["positie (nu vs lw)"])
st.dataframe(styler, use_container_width=True)

with st.expander("🔧 Debug — API calls"):
    st.write("Cards call params:", p_cards)
    st.write("Cards HTTP status:", status_cards)
    st.write("Leaderboard this_week params:", p_this)
    st.write("Leaderboard last_week params:", p_last)
