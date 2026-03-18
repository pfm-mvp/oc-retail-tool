# pages/01_Store_Live_Ops_with_Leaderboard_AI_Region.py
import os, sys, math, json
from datetime import datetime
import pytz
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ───────────────────────── Page config (exact 1x) ─────────────────────────
st.set_page_config(page_title="Store Live Ops — Gisteren vs Eergisteren + Leaderboard",
                   page_icon="🛍️", layout="wide")
st.title("🛍️ Store Live Ops — Gisteren vs Eergisteren + Leaderboard")

# ───────────────────────── Imports / mapping ──────────────────────────────
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from helpers_shop import ID_TO_NAME, NAME_TO_ID            # {id->name}, {name->id}
from helpers_normalize import normalize_vemcount_response

API_URL = st.secrets["API_URL"]

# ───────────────────────── Kleuren & CSS ──────────────────────────────────
PFM_RED = "#F04438"; PFM_GREEN = "#22C55E"; PFM_PURPLE = "#6C4EE3"
PFM_GRAY = "#6B7280"; PFM_GRAY_BG = "rgba(107,114,128,.10)"

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

# ───────────────────────── Store picker ───────────────────────────────────
if not NAME_TO_ID:
    st.error("Geen winkels geladen (NAME_TO_ID is leeg). Controleer helpers_shop.py of shop_mapping.py.")
    st.stop()

store_options = sorted(ID_TO_NAME.values())
store_name    = st.selectbox("Kies winkel", store_options, index=0, key="store_pick")
store_id      = NAME_TO_ID.get(store_name)
if store_id is None:
    st.error("Kon de geselecteerde winkel niet mappen naar een ID.")
    st.stop()

# ───────────────────────── Tijd & KPI’s ───────────────────────────────────
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor"]
TZ = pytz.timezone("Europe/Amsterdam")
TODAY = datetime.now(TZ).date()

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    """Zorgt dat date_eff altijd datetime64 is; vult vanuit 'date' of 'timestamp'."""
    d = df.copy()
    if "date" not in d.columns:
        d["date"] = pd.NaT
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce", utc=False)
    d["date_eff"] = pd.to_datetime(d["date"], errors="coerce", utc=False)
    d["date_eff"] = d["date_eff"].fillna(ts)
    # Normalize tz-naive
    d["date_eff"] = pd.to_datetime(d["date_eff"]).dt.tz_localize(None)
    return d

def normalize_json(js, id_to_name, metrics):
    df = normalize_vemcount_response(js, id_to_name, kpi_keys=metrics)
    df = add_effective_date(df)
    # Backfill shop_name in case normalizer didn’t add it
    if "shop_name" not in df.columns and "shop_id" in df.columns:
        df["shop_name"] = df["shop_id"].map(ID_TO_NAME)
    return df

def _date_bounds(series: pd.Series):
    if series is None or series.empty:
        return None, None
    s = pd.to_datetime(series, errors="coerce")
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    mn_s = None if pd.isna(mn) else str(pd.to_datetime(mn).date())
    mx_s = None if pd.isna(mx) else str(pd.to_datetime(mx).date())
    return mn_s, mx_s

def post_report(params):
    r = requests.post(API_URL, params=params, timeout=45)
    r.raise_for_status()
    return r

def fetch_df(shop_ids, period, step, metrics, label=""):
    """1 call met meerdere data=…; geeft df + debug terug."""
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in metrics]
    params += [("source","shops"), ("period", period), ("step", step)]
    resp = post_report(params)
    js = resp.json()
    df = normalize_json(js, ID_TO_NAME, metrics)

    # zorg dat date_eff datetime64 is
    if "date_eff" in df.columns:
        df["date_eff"] = pd.to_datetime(df["date_eff"], errors="coerce").dt.tz_localize(None)

    mn_s, mx_s = _date_bounds(df["date_eff"]) if "date_eff" in df.columns else (None, None)
    dbg = {
        "label": label,
        "status": resp.status_code,
        "params_sample": params[:10],
        "n_rows": int(len(df)),
        "n_unique_shops": int(df["shop_id"].nunique()) if "shop_id" in df.columns else 0,
        "date_eff_min": mn_s,
        "date_eff_max": mx_s,
    }
    return df, params, resp.status_code, dbg

def fetch_df_fallback_each(shop_ids, period, step, metrics, label="fallback_each"):
    """Fallback: call per winkel en concat, voor API’s die geen multi-id ondersteunen."""
    parts = []
    calls = []
    for sid in shop_ids:
        dfi, pi, si, dbgi = fetch_df([sid], period, step, metrics, label=f"{label}:{sid}")
        parts.append(dfi)
        calls.append({"sid": sid, **dbgi})
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    mn_s, mx_s = _date_bounds(df["date_eff"]) if "date_eff" in df.columns else (None, None)
    dbg = {"label": label, "mode": "per_shop_concat", "n_rows": int(len(df)),
           "n_unique_shops": int(df["shop_id"].nunique()) if "shop_id" in df.columns else 0,
           "date_eff_min": mn_s, "date_eff_max": mx_s, "calls": calls[:5]}
    return df, None, None, dbg

# ───────────────── Cards (gisteren vs eergisteren) ────────────────────────
df_cards, p_cards, status_cards, dbg_cards = fetch_df([store_id], "this_week", "day", METRICS, label="cards:this_week")
pre_rows_cards = len(df_cards)
if "date_eff" in df_cards.columns:
    df_cards = df_cards[df_cards["date_eff"].dt.date < TODAY]
post_rows_cards = len(df_cards)

dates = sorted(df_cards["date_eff"].dt.date.unique()) if "date_eff" in df_cards.columns and not df_cards.empty else []
if len(dates) < 2:
    df_cards, p_cards, status_cards, dbg_cards2 = fetch_df([store_id], "last_week", "day", METRICS, label="cards:last_week")
    pre_rows_cards2 = len(df_cards)
    if "date_eff" in df_cards.columns:
        df_cards = df_cards[df_cards["date_eff"].dt.date < TODAY]
    post_rows_cards2 = len(df_cards)
    dates = sorted(df_cards["date_eff"].dt.date.unique()) if "date_eff" in df_cards.columns and not df_cards.empty else []
    dbg_cards = {"first": dbg_cards, "second": dbg_cards2,
                 "pre_rows_first": pre_rows_cards, "post_rows_first": post_rows_cards,
                 "pre_rows_second": pre_rows_cards2, "post_rows_second": post_rows_cards2}

if len(dates) < 2:
    st.error("Niet genoeg dagdata om kaarten te tonen.")
    st.stop()

ydate, bdate = dates[-1], dates[-2]
gy = df_cards[(df_cards["date_eff"].dt.date==ydate) & (df_cards["shop_id"]==store_id)][METRICS].sum(numeric_only=True)
gb = df_cards[(df_cards["date_eff"].dt.date==bdate) & (df_cards["shop_id"]==store_id)][METRICS].sum(numeric_only=True)

def delta_badge(value: float, kind: str):
    if value > 0: cls, arrow = "up","↑"
    elif value < 0: cls, arrow = "down","↓"
    else: cls, arrow = "flat","→"
    if kind=="int":  disp = f"{abs(int(value)):,}".replace(",", ".")
    elif kind=="pct2": disp = f"{abs(value):.2f}%"
    elif kind=="eur0": disp = f"€{abs(value):,.0f}".replace(",", ".")
    elif kind=="eur2": disp = f"€{abs(value):,.2f}".replace(",", ".")
    else: disp = f"{abs(value)}"
    return f'<span class="kpi-delta {cls}">{arrow} <b>{disp}</b> t.o.v. dag ervoor</span>'

c1,c2,c3,c4 = st.columns(4)
with c1:
    val = int(gy.get("count_in", 0)); diff = gy.get("count_in", 0) - gb.get("count_in", 0)
    st.markdown(f"""
<div class="kpi-card"><div class="kpi-title">Bezoekers <small>(gisteren)</small></div>
<div class="kpi-value">{val:,}</div>{delta_badge(float(diff),"int")}</div>
""".replace(",", "."), unsafe_allow_html=True)
with c2:
    conv_y = float(gy.get("conversion_rate", 0.0)); conv_b = float(gb.get("conversion_rate", 0.0))
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
<div class="kpi-card"><div class="kpi-title">SPV <small>(gisteren)</small></div>
<div class="kpi-value">€{spv_y:,.2f}</div>{delta_badge((spv_y-spv_b),"eur2")}</div>
""".replace(",", "."), unsafe_allow_html=True)

st.markdown("---")

# ───────────────── Leaderboard (WTD t/m gisteren) ─────────────────────────
all_ids = list(ID_TO_NAME.keys())

def fetch_wtd_multi(period, label="wtd_multi"):
    df, p, s, dbg = fetch_df(all_ids, period, "day", METRICS, label=label)
    rows_before = len(df)
    if "date_eff" in df.columns:
        df = df[df["date_eff"].dt.date < TODAY]
    rows_after = len(df)
    dbg["rows_before_filter"] = rows_before
    dbg["rows_after_filter"]  = rows_after
    return df, p, s, dbg

df_this, p_this, s_this, dbg_this = fetch_wtd_multi("this_week", "wtd:this_week")
df_last, p_last, s_last, dbg_last = fetch_wtd_multi("last_week", "wtd:last_week")

# Fallback: als multi-call leeg is, call per winkel en concat
if df_this.empty:
    df_this, _, _, dbg_this_fb = fetch_df_fallback_each(all_ids, "this_week", "day", METRICS, label="wtd_fb:this_week")
    dbg_this = {"multi": dbg_this, "fallback": dbg_this_fb}
if df_last.empty:
    df_last, _, _, dbg_last_fb = fetch_df_fallback_each(all_ids, "last_week", "day", METRICS, label="wtd_fb:last_week")
    dbg_last = {"multi": dbg_last, "fallback": dbg_last_fb}

st.subheader("🏁 Leaderboard — huidige week (t/m gisteren)")
rank_choice = st.radio("Ranking op basis van", ["Conversie", "SPV"], horizontal=True, index=0, key="rank_choice")

def wtd_agg(d: pd.DataFrame) -> pd.DataFrame:
    if d is None or d.empty: return pd.DataFrame()
    g = d.groupby("shop_id", as_index=False).agg({"count_in":"sum","turnover":"sum"})
    # SPV robuust (omzet / bezoekers)
    g["sales_per_visitor"] = g.apply(lambda r: (r["turnover"]/r["count_in"]) if r["count_in"] else 0.0, axis=1)
    # conversie gewogen op traffic (val terug op mean als geen traffic)
    conv = d.groupby("shop_id").apply(
        lambda x: (x["conversion_rate"]*x["count_in"]).sum()/x["count_in"].sum()
                  if x["count_in"].sum() else float(x["conversion_rate"].mean())
    ).reset_index()
    conv.columns = ["shop_id","conversion_rate"]
    g = g.merge(conv, on="shop_id", how="left")
    g["shop_name"] = g["shop_id"].map(ID_TO_NAME)
    return g

# ── Omzetpositie vorige week (compacte card, 1x) ───────────────────────────────
_last = df_last.copy()
if not _last.empty:
    grp = (
        _last.groupby("shop_id", as_index=False)
             .agg(turnover=("turnover", "sum"))
    )
    grp["shop_name"] = grp["shop_id"].map(ID_TO_NAME)
    grp = grp.sort_values("turnover", ascending=False).reset_index(drop=True)

    top_turn = float(grp.iloc[0]["turnover"]) if len(grp) else 0.0
    top_name = grp.iloc[0]["shop_name"] if len(grp) else "—"

    me = grp[grp["shop_id"] == store_id]
    my_turn = float(me["turnover"].iloc[0]) if not me.empty else 0.0
    my_rank = (int(me.index[0]) + 1) if not me.empty else None
    n_shops = int(grp["shop_id"].nunique())

    pct_vs_top = (my_turn / top_turn * 100.0) if top_turn > 0 else float("nan")

    def eur0(x): return f"€{x:,.0f}".replace(",", ".")
    def pct0(x):
        try: return f"{x:.0f}%"
        except: return "—"

    st.markdown(
        f"""
<div style="border:1px solid #EEE; border-radius:14px; padding:16px 18px; background:#FBFBFD; margin:6px 0 18px 0;">
  <div style="font-weight:700; font-size:16px; margin-bottom:8px;">
    📊 Jouw omzetpositie <span style="color:#6B7280">(vorige week)</span>
  </div>

  <div style="display:flex; align-items:baseline; gap:10px; margin-bottom:10px;">
    <div style="font-size:48px; font-weight:900; line-height:1;">#{my_rank if my_rank else '—'}</div>
    <div style="font-size:20px; color:#6B7280;">van {n_shops}</div>
  </div>

  <div style="display:flex; gap:10px; flex-wrap:wrap;">
    <span style="display:inline-block; padding:8px 12px; border-radius:999px; background:rgba(34,197,94,.10); color:#22C55E; font-weight:700; font-size:14px;">
      {pct0(pct_vs_top)} van #1 {top_name}
    </span>
    <span style="display:inline-block; padding:8px 12px; border-radius:999px; background:#F3F4F6; color:#0C111D; font-weight:700; font-size:14px;">
      {eur0(my_turn)} (jouw weekomzet)
    </span>
  </div>
</div>
        """,
        unsafe_allow_html=True
    )

# -- leaderboard
agg_this = wtd_agg(df_this)
agg_last = wtd_agg(df_last)

if agg_this.empty:
    with st.expander("🔧 WTD Debug (openen bij lege tabel)"):
        st.write("this_week (multi/fallback):", dbg_this)
        st.write("last_week  (multi/fallback):", dbg_last)
        st.write("df_this head:", df_this.head() if not df_this.empty else "(leeg)")
        st.write("df_last head:", df_last.head() if not df_last.empty else "(leeg)")
    st.info("Geen WTD data beschikbaar.")
    st.stop()

# ── NIEUW: peers-medianen (exclusief eigen winkel) voor AI‑coach ─────────
try:
    _peers = agg_this.copy()
    if not _peers.empty:
        _peers = _peers[_peers["shop_id"] != store_id]
    if _peers.empty:
        peer_conv_med = float(agg_this["conversion_rate"].median()) if not agg_this.empty else None
        peer_spv_med  = float(agg_this["sales_per_visitor"].median()) if not agg_this.empty else None
    else:
        peer_conv_med = float(_peers["conversion_rate"].median()) if "conversion_rate" in _peers.columns else None
        peer_spv_med  = float(_peers["sales_per_visitor"].median()) if "sales_per_visitor" in _peers.columns else None
except Exception:
    peer_conv_med = None
    peer_spv_med  = None

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
def color_pos_col(col_series):
    styles=[]
    for i,_ in enumerate(col_series):
        ch_i = int(agg.loc[fmt.index[i], "rank_change"]) if pd.notna(agg.loc[fmt.index[i], "rank_change"]) else 0
        if ch_i > 0: styles.append(f"color: {PFM_PURPLE}; font-weight: 700")
        elif ch_i < 0: styles.append(f"color: {PFM_RED}; font-weight: 700")
        else: styles.append(f"color: {PFM_GRAY}; font-weight: 700")
    return styles
styler = styler.apply(color_pos_col, subset=["positie (nu vs lw)"])
st.dataframe(styler, use_container_width=True)

# === AI Card styling (drop‑in) ==========================================
st.markdown("""
<style>
.ai-card {
  border: 1px solid #E9EAF0;
  border-radius: 16px;
  padding: 18px 18px 14px 18px;
  background: linear-gradient(180deg, #FFFFFF 0%, #FCFCFE 100%);
  box-shadow: 0 1px 0 #F1F2F6, 0 8px 24px rgba(12,17,29,0.06);
  margin-top: 8px;
}
.ai-title {
  display:flex; align-items:center; gap:10px;
  font-weight:800; font-size:18px; color:#0C111D; margin-bottom:4px;
}
.ai-title .dot {
  width:10px;height:10px;border-radius:50%;
  background: radial-gradient(circle at 30% 30%, #9E77ED 0, #6C4EE3 60%, #9E77ED 100%);
  box-shadow: 0 0 12px rgba(108,78,227,.6);
}
.ai-caption { color:#6B7280; font-size:13px; margin-bottom:10px; }
.ai-body { font-size:15px; line-height:1.55; }
.ai-body ul { margin:0 0 0 16px; padding:0; }
.ai-body li { margin: 0 0 6px 0; }
</style>
""", unsafe_allow_html=True)

def render_ai_card(markdown_text: str, subtitle: str = "Live acties op basis van jouw cijfers en de peer‑mediaan."):
    st.markdown(
        '<div class="ai-card">'
        '<div class="ai-title"><span class="dot"></span>🤖 AI‑Insights</div>'
        f'<div class="ai-caption">{subtitle}</div>'
        '<div class="ai-body">', unsafe_allow_html=True
    )
    st.markdown(markdown_text)
    st.markdown('</div></div>', unsafe_allow_html=True)

# === 🤖 AI Insights (verbeterde versie) ======================================
try:

    # 2) Compacte context voor het model
    def _safe_num(x, d=2):
        try:
            return round(float(x), d)
        except Exception:
            return None

    _vis_y  = _safe_num(gy.get("count_in", 0), 0)
    _turn_y = _safe_num(gy.get("turnover", 0), 2)
    _conv_y = _safe_num(gy.get("conversion_rate", 0), 2)
    _spv_y  = _safe_num(gy.get("sales_per_visitor", 0), 2)
    if (_spv_y in (None, 0)) and _vis_y not in (None, 0):
        _spv_y = _safe_num(_turn_y / _vis_y, 2)

    _vis_b  = _safe_num(gb.get("count_in", 0), 0)
    _turn_b = _safe_num(gb.get("turnover", 0), 2)
    _conv_b = _safe_num(gb.get("conversion_rate", 0), 2)
    _spv_b  = _safe_num(gb.get("sales_per_visitor", 0), 2)
    if (_spv_b in (None, 0)) and _vis_b not in (None, 0):
        _spv_b = _safe_num(_turn_b / _vis_b, 2)

    _rank_now = None
    _rank_ch   = 0
    try:
        _me_row = agg[agg["shop_id"] == store_id].iloc[0]
        _rank_now = int(_me_row["rank_now"])
        if not pd.isna(_me_row.get("rank_change", np.nan)):
            _rank_ch = int(_me_row["rank_change"])
    except Exception:
        pass

    ai_context = {
        "store_name": store_name,
        "yesterday": {
            "visitors": _vis_y,
            "turnover": _turn_y,
            "conversion_pct": _conv_y,
            "spv_eur": _spv_y
        },
        "day_before": {
            "visitors": _vis_b,
            "turnover": _turn_b,
            "conversion_pct": _conv_b,
            "spv_eur": _spv_b
        },
        "leaderboard": {
            "rank_now": _rank_now,
            "rank_change_vs_last_week": _rank_ch
        },
        "peers_median": {
            "conversion_pct": peer_conv_med if 'peer_conv_med' in locals() else None,
            "spv_eur": peer_spv_med if 'peer_spv_med' in locals() else None
        }
    }

    # 3) Prompt (NL, kort & actiegericht)
    sys_msg = (
        "Je bent een retail floor coach. Geef maximaal 5 korte en concrete punten: "
        "2 bevindingen (opvallende trends of afwijkingen van peers_median) en maximaal 3 acties voor vandaag. "
        "Gebruik Nederlands, wees meetbaar (noem bedragen als €X en percentages als 12,3%). "
        "Geef altijd 1 meettip (wat per uur te checken). "
        "Als peers_median ontbreekt, negeer die vergelijking. "
        "Focus bij KPI’s lager dan peers_median op verbeteracties."
    )

    usr_msg = (
        "Context (JSON):\n"
        f"{ai_context}\n\n"
        "Schrijf puntsgewijs. Gebruik emoji spaarzaam (max 1). "
        "Vermijd algemene adviezen; maak het winkelvloer-concreet "
        "(begroeting, paskamers, kassascripts, bundels, voorraad bij de hand, etc.)."
    )

    # 4) Call OpenAI (alleen als key aanwezig)

    try:
        from openai import OpenAI
        if "OPENAI_API_KEY" not in st.secrets or not st.secrets["OPENAI_API_KEY"]:
            st.info("Geen `OPENAI_API_KEY` in secrets. Voeg die toe voor live AI‑inzichten.")
        else:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.25,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg},
                ],
            )
            insight = resp.choices[0].message.content.strip()
            render_ai_card(insight)
    except Exception as e:
        st.warning(f"AI‑insights konden niet geladen worden: {e}")

    # Optionele debug
    with st.expander("🔧 AI‑debug"):
        st.json(ai_context)

except Exception as _e:
    st.warning(f"AI‑inhoud kon niet worden opgebouwd: {_e}")

# ───────────────────────── Debug (onderaan) ───────────────────────────────
with st.expander("🔧 Debug — API calls"):
    st.write("Cards call — http:", dbg_cards if isinstance(dbg_cards, dict) else {})
    st.write("WTD this_week —", dbg_this)
    st.write("WTD last_week —", dbg_last)
