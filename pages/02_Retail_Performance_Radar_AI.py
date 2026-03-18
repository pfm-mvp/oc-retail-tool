# pages/02_Retail_Performance_Radar_AI.py
import os, sys
from datetime import datetime
import pytz
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- Imports / mapping ----------
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from helpers_shop import ID_TO_NAME, NAME_TO_ID, REGIONS, get_ids_by_region
from helpers_normalize import normalize_vemcount_response

# ---------- Page ----------
st.set_page_config(page_title="Region Performance Radar", page_icon="🧭", layout="wide")
st.title("🧭 Region Performance Radar")

API_URL = st.secrets["API_URL"]

# ---------- PFM-styling ----------
PFM_RED    = "#F04438"
PFM_GREEN  = "#22C55E"
PFM_PURPLE = "#6C4EE3"
PFM_GRAY   = "#6B7280"

# PFM brand palette (donker → licht)
PFM_PALETTE = ["#21114E", "#5B167E", "#922B80", "#CC3F71", "#F56B5C", "#FEAC76"]

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """'#RRGGBB' -> 'rgba(r,g,b,a)' (Plotly vereist echte rgba, geen hex+alpha)."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

st.markdown(f"""
<style>
.kpi {{ border:1px solid #eee; border-radius:14px; padding:16px; }}
.kpi .t {{ font-weight:600; color:#0C111D; }}
.kpi .v {{ font-size:38px; font-weight:800; }}
.badge {{ font-size:13px; font-weight:700; padding:4px 10px; border-radius:999px; display:inline-block; }}
.badge.up {{ color:{PFM_GREEN}; background: rgba(34,197,94,.10); }}
.badge.down {{ color:{PFM_RED}; background: rgba(240,68,56,.10); }}
.badge.flat {{ color:{PFM_GRAY}; background: rgba(107,114,128,.10); }}
.box {{ border:1px dashed #ddd; border-radius:12px; padding:14px; background:#FAFAFC; }}
.box h4 {{ margin:0 0 8px 0; }}
</style>
""", unsafe_allow_html=True)

# ---------- Inputs ----------
PERIODS = ["this_week","last_week","this_month","last_month","this_quarter","last_quarter","this_year","last_year"]
colP, colR = st.columns([1,1])
with colP:
    period = st.selectbox("Periode", PERIODS, index=1, key="radar_period")
with colR:
    regio = st.selectbox("Regio", ["All"] + REGIONS, index=0, key="radar_region")

# Shop-ids vanuit regio
if regio == "All":
    ALL_IDS = list(ID_TO_NAME.keys())
else:
    shop_ids_from_region = get_ids_by_region(regio)
    ALL_IDS = shop_ids_from_region if shop_ids_from_region else list(ID_TO_NAME.keys())

if not ALL_IDS:
    st.warning("Geen winkels gevonden (mapping leeg).")
    st.stop()

# ---------- Helpers ----------
TZ = pytz.timezone("Europe/Amsterdam")
TODAY = datetime.now(TZ).date()
METRICS = ["count_in","conversion_rate","turnover","sales_per_visitor","sq_meter"]

def post_report(params):
    r = requests.post(API_URL, params=params, timeout=45)
    r.raise_for_status()
    return r

def add_effective_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        d["date"] = pd.NaT
    ts = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"] = pd.to_datetime(d["date"], errors="coerce").fillna(ts)
    d["date_eff"] = d["date_eff"].dt.date
    return d

def fetch_df(shop_ids, period, step, metrics):
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in metrics]
    params += [("source","shops"), ("period", period), ("step", step)]
    r = post_report(params)
    js = r.json()
    # mapping id->name
    df = normalize_vemcount_response(js, ID_TO_NAME, kpi_keys=metrics)
    dfe = add_effective_date(df)
    return dfe, params, r.status_code

def weighted_avg(series, weights):
    try:
        w = weights.fillna(0.0)
        s = series.fillna(0.0)
        d = w.sum()
        return (s*w).sum()/d if d else np.nan
    except Exception:
        return np.nan

def fmt_eur0(x): return f"€{x:,.0f}".replace(",", ".")
def fmt_eur2(x): return f"€{x:,.2f}".replace(",", ".")
def fmt_pct2(x): return f"{x:.2f}%"

# ---------- Data ophalen ----------
df_cur, p_cur, s_cur = fetch_df(ALL_IDS, period, "day", METRICS)

# Alleen voor this_* is er een zinnige vorige periode (last_*)
prev_map = {
    "this_week": "last_week",
    "this_month": "last_month",
    "this_quarter": "last_quarter",
    "this_year": "last_year",
}
has_true_previous = period in prev_map
period_prev = prev_map.get(period, None)

if has_true_previous:
    df_prev, p_prev, s_prev = fetch_df(ALL_IDS, period_prev, "day", METRICS)
else:
    df_prev, p_prev, s_prev = (pd.DataFrame(), [], None)

# Voor 'this_*' alleen t/m gisteren nemen
if period.startswith("this_"):
    df_cur = df_cur[df_cur["date_eff"] < TODAY]

# ---------- Aggregatie per winkel ----------
def agg_store(d: pd.DataFrame) -> pd.DataFrame:
    if d is None or d.empty: return pd.DataFrame()
    g = d.groupby("shop_id", as_index=False).agg({"count_in":"sum","turnover":"sum"})
    w = d.groupby("shop_id").apply(lambda x: pd.Series({
        "conversion_rate": weighted_avg(x["conversion_rate"], x["count_in"]),
        "sales_per_visitor": weighted_avg(x["sales_per_visitor"], x["count_in"]),
    })).reset_index()
    g = g.merge(w, on="shop_id", how="left")
    sqm = (d.sort_values("date_eff").groupby("shop_id")["sq_meter"]
           .apply(lambda s: float(s.dropna().iloc[-1]) if s.dropna().size else np.nan)
           ).reset_index()
    g = g.merge(sqm, on="shop_id", how="left")
    g["sales_per_sqm"] = g.apply(
        lambda r: (r["turnover"]/r["sq_meter"]) if (pd.notna(r["sq_meter"]) and r["sq_meter"]>0) else np.nan,
        axis=1
    )
    g["shop_name"] = g["shop_id"].map(ID_TO_NAME)
    return g

cur = agg_store(df_cur)
prev = agg_store(df_prev)

if cur.empty:
    st.warning("Geen data voor deze periode/regio.")
    st.stop()

# ---------- Region KPI's ----------
total_turn = cur["turnover"].sum()
total_vis  = cur["count_in"].sum()
total_sqm  = cur["sq_meter"].fillna(0).sum()

avg_conv   = weighted_avg(cur["conversion_rate"],   cur["count_in"])
avg_spv    = weighted_avg(cur["sales_per_visitor"], cur["count_in"])
avg_spsqm  = (total_turn / total_sqm) if total_sqm > 0 else np.nan

if has_true_previous and not prev.empty:
    prev_total_turn = prev["turnover"].sum()
    prev_total_sqm  = prev["sq_meter"].fillna(0).sum()
    prev_avg_conv   = weighted_avg(prev["conversion_rate"],   prev["count_in"])
    prev_avg_spv    = weighted_avg(prev["sales_per_visitor"], prev["count_in"])
    prev_avg_spsqm  = (prev_total_turn / prev_total_sqm) if prev_total_sqm > 0 else np.nan
else:
    prev_total_turn = prev_avg_conv = prev_avg_spv = prev_avg_spsqm = np.nan

def delta(this, last):
    if pd.isna(this) or pd.isna(last):
        return (np.nan, "flat", False)
    diff = float(this) - float(last)
    cls = "up" if diff>0 else ("down" if diff<0 else "flat")
    return (diff, cls, True)

def badge(label_value, cls, is_real_delta, money=False, pp=False):
    if not is_real_delta:
        return f'<span class="badge flat">n.v.t.</span>'
    if money:
        val = f"{'+' if label_value>0 else ''}{'€'}{abs(label_value):,.0f}".replace(",",".")
    elif pp:
        val = f"{'+' if label_value>0 else ''}{abs(label_value):.2f}pp"
    else:
        val = f"{'+' if label_value>0 else ''}{abs(label_value):.2f}"
    return f'<span class="badge {cls}">{val} vs vorige periode</span>'

if has_true_previous:
    d_turn,  cls_turn,  ok1 = delta(total_turn, prev_total_turn)
    d_conv,  cls_conv,  ok2 = delta(avg_conv,   prev_avg_conv)
    d_spv,   cls_spv,   ok3 = delta(avg_spv,    prev_avg_spv)
    d_spsqm, cls_spsqm, ok4 = delta(avg_spsqm,  prev_avg_spsqm)
else:
    d_turn = d_conv = d_spv = d_spsqm = np.nan
    cls_turn = cls_conv = cls_spv = cls_spsqm = "flat"
    ok1 = ok2 = ok3 = ok4 = False

# ---------- KPI Cards ----------
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="kpi">
    <div class="t">💶 Totale omzet</div>
    <div class="v">{fmt_eur0(total_turn)}</div>
    {badge(d_turn, cls_turn, ok1, money=True)}
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi">
    <div class="t">🛒 Gem. conversie</div>
    <div class="v">{fmt_pct2(avg_conv)}</div>
    {badge(d_conv, cls_conv, ok2, pp=True)}
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi">
    <div class="t">💸 Gem. SPV</div>
    <div class="v">{fmt_eur2(avg_spv)}</div>
    {badge(d_spv, cls_spv, ok3, money=True)}
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi">
    <div class="t">🏁 Gem. sales/m²</div>
    <div class="v">{fmt_eur2(avg_spsqm)}</div>
    {badge(d_spsqm, cls_spsqm, ok4, money=True)}
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ---------- Radarvergelijking ----------
st.subheader("📈 Radarvergelijking (Conversie / SPV / Sales per m²)")

# 1) Basis met ruwe metriek-kolommen
metric_cols = ["conversion_rate", "sales_per_visitor", "sales_per_sqm"]
norm_base = cur[["shop_id", "shop_name"] + metric_cols].copy()

# 2) Default selectie = winkels van actuele regio (max 6). Kun je nog steeds wijzigen.
region_names = [ID_TO_NAME[i] for i in ALL_IDS if i in ID_TO_NAME]
default_names = region_names[:6] if region_names else list(ID_TO_NAME.values())[:6]

sel_names = st.multiselect(
    "Vergelijk winkels (max 6)",
    options=region_names if region_names else list(ID_TO_NAME.values()),
    default=default_names,
    max_selections=6,
    key=f"radar_multiselect_{regio}_{period}",   # key afhankelijk van filters → nette refresh
)

# 3) Normaliseer **binnen de selectie** (niet over alle winkels) zodat de vormen leesbaar zijn
sel_rows = norm_base[norm_base["shop_name"].isin(sel_names)].copy()

def _minmax_norm(df: pd.DataFrame, col: str) -> pd.Series:
    v = pd.to_numeric(df[col], errors="coerce")
    vmin, vmax = v.min(skipna=True), v.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series([0.0] * len(df), index=df.index)
    return (v - vmin) / (vmax - vmin)

if not sel_rows.empty:
    for m in metric_cols:
        sel_rows[m + "_norm"] = _minmax_norm(sel_rows, m).fillna(0.0)
else:
    st.info("Kies één of meer winkels om de radar te tonen.")
    sel_rows = pd.DataFrame(columns=["shop_name"] + [m + "_norm" for m in metric_cols])

# 4) Plotly radar met PFM-kleuren + transparant vlak
PFM_RADAR_PALETTE = ["#21114E", "#5B167E", "#922B80", "#CC3F71", "#F56B5C", "#FEAC76"]

def _hex_to_rgba(hex_color: str, alpha: float = 0.22) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

categories = ["Conversie", "SPV", "Sales/m²"]
fig = go.Figure()

for idx, (_, row) in enumerate(sel_rows.iterrows()):
    color = PFM_RADAR_PALETTE[idx % len(PFM_RADAR_PALETTE)]
    rgba  = _hex_to_rgba(color, 0.22)
    values = [
        float(row.get("conversion_rate_norm", 0.0)),
        float(row.get("sales_per_visitor_norm", 0.0)),
        float(row.get("sales_per_sqm_norm", 0.0)),
    ]
    # Sluit de vorm door het 1e punt opnieuw toe te voegen
    fig.add_trace(go.Scatterpolar(
        r=values + values[:1],
        theta=categories + categories[:1],
        name=row["shop_name"],
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=4),
        fill="toself",
        fillcolor=rgba,
        hovertemplate="<b>%{text}</b><br>%{theta}: %{r:.2f}<extra></extra>",
        text=[row["shop_name"]] * (len(values) + 1),
        opacity=0.9,
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(0,0,0,0.15)"),
        angularaxis=dict(gridcolor="rgba(0,0,0,0.08)")
    ),
    legend=dict(orientation="v"),
    height=520,
    margin=dict(l=20, r=20, t=20, b=20),
    showlegend=True,
)
st.plotly_chart(fig, use_container_width=True)

st.caption("""
ℹ️ **Hoe lees je deze radar:**  
Conversie, SPV en Sales/m² zijn genormaliseerd naar 0–1 binnen deze regio/periode.  
- Een **hoekig vlak** betekent dat de winkel duidelijke sterktes/zwaktes heeft.  
- Een **rond vlak of stip** betekent dat de winkel overal ongeveer gelijk scoort.  
""")

# ---------- Tops & Flops ----------
st.subheader("🏆 Tops & Flops")
rank_spm = cur[["shop_name","sales_per_sqm","turnover","count_in","conversion_rate","sales_per_visitor"]].copy()
rank_spm = rank_spm.sort_values("sales_per_sqm", ascending=False)

cA, cB = st.columns(2)
with cA:
    st.caption("Top 5 (sales/m²)")
    top5 = rank_spm.head(5).copy()
    top5["sales_per_sqm"]     = top5["sales_per_sqm"].map(fmt_eur2)
    top5["turnover"]          = top5["turnover"].map(fmt_eur0)
    top5["count_in"]          = top5["count_in"].map(lambda x: f"{int(x):,}".replace(",", "."))
    top5["conversion_rate"]   = top5["conversion_rate"].map(lambda x: f"{float(x):.2f}%")
    top5["sales_per_visitor"] = top5["sales_per_visitor"].map(fmt_eur2)
    st.dataframe(top5, use_container_width=True)

with cB:
    st.caption("Bottom 5 (sales/m²)")
    flop5 = rank_spm.tail(5).copy().sort_values("sales_per_sqm", ascending=True)
    flop5["sales_per_sqm"]     = flop5["sales_per_sqm"].map(fmt_eur2)
    flop5["turnover"]          = flop5["turnover"].map(fmt_eur0)
    flop5["count_in"]          = flop5["count_in"].map(lambda x: f"{int(x):,}".replace(",", "."))
    flop5["conversion_rate"]   = flop5["conversion_rate"].map(lambda x: f"{float(x):.2f}%")
    flop5["sales_per_visitor"] = flop5["sales_per_visitor"].map(fmt_eur2)
    st.dataframe(flop5, use_container_width=True)

# ---------- Leaderboard t.o.v. regio-gemiddelde ----------
st.subheader("🏁 Leaderboard — sales/m² t.o.v. regio-gemiddelde")

comp = cur[["shop_name", "sales_per_sqm", "turnover", "sq_meter"]].copy()
comp["region_avg_spsqm"] = avg_spsqm
comp["delta_eur_sqm"] = comp["sales_per_sqm"] - comp["region_avg_spsqm"]
comp["delta_pct"] = np.where(
    comp["region_avg_spsqm"] > 0,
    (comp["delta_eur_sqm"] / comp["region_avg_spsqm"]) * 100.0,
    np.nan
)

sort_best_first = st.toggle("Beste afwijking eerst", value=True, key="radar_toggle_best")
comp = comp.sort_values("delta_eur_sqm", ascending=not sort_best_first)

show = comp[["shop_name","sales_per_sqm","region_avg_spsqm","delta_eur_sqm","delta_pct"]].rename(columns={
    "shop_name": "winkel",
    "sales_per_sqm": "sales/m²",
    "region_avg_spsqm": "gem. sales/m² (regio)",
    "delta_eur_sqm": "Δ vs gem. (€/m²)",
    "delta_pct": "Δ vs gem. (%)",
})

def color_delta(series):
    styles = []
    for v in series:
        if pd.isna(v) or v == 0:
            styles.append("color: #6B7280;")
        elif v > 0:
            styles.append("color: #22C55E;")
        else:
            styles.append("color: #F04438;")
    return styles

styler = (
    show.style
        .format({
            "sales/m²": "€{:.2f}",
            "gem. sales/m² (regio)": "€{:.2f}",
            "Δ vs gem. (€/m²)": "€{:+.2f}",
            "Δ vs gem. (%)": "{:+.1f}%"
        })
        .apply(color_delta, subset=["Δ vs gem. (€/m²)"])
        .apply(color_delta, subset=["Δ vs gem. (%)"])
)

st.dataframe(styler, use_container_width=True)

# === 🤖 AI‑Insights — Regio Manager (pagina 02) ==========================
# Vereist: st.secrets["OPENAI_API_KEY"]
from importlib import import_module

# 1) Styling hergebruiken van pagina 01 (of plakken als je 'm daar niet hebt)
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

def render_ai_card(markdown_text: str, subtitle: str = "Regio‑analyse en concrete acties voor de filialen."):
    st.markdown(
        '<div class="ai-card">'
        '<div class="ai-title"><span class="dot"></span>🤖 AI‑Insights (Regio)</div>'
        f'<div class="ai-caption">{subtitle}</div>'
        '<div class="ai-body">', unsafe_allow_html=True
    )
    st.markdown(markdown_text)
    st.markdown('</div></div>', unsafe_allow_html=True)

# 2) Veilig helpers
def _sf(x, d=2):
    try:
        v = float(x)
        if np.isnan(v): return None
        return round(v, d)
    except Exception:
        return None

def _eur(x, d=0):
    try: return ("€{:,.%df}" % d).format(float(x)).replace(",", ".")
    except: return "€–"

def _pct(x, d=2):
    try: return f"{float(x):.{d}f}%"
    except: return "–"

# 3) Bouw compacte JSON‑context voor de manager
try:
    region_name = regio if "regio" in locals() else (regio_choice if "regio_choice" in locals() else "All")
    # Huidige periode (cur) is al per winkel geaggregeerd in jouw script (cur)
    # Zorg voor robuuste berekeningen:
    _cur = cur.copy()
    _cur["sales_per_sqm"] = _cur.get("sales_per_sqm", np.nan)

    # Regionale kpi’s (huidig)
    reg_now = {
        "total_turnover": _sf(_cur["turnover"].sum(), 0),
        "avg_conversion_pct": _sf((_cur["conversion_rate"] * _cur["count_in"]).sum() / _cur["count_in"].sum() if _cur["count_in"].sum() else np.nan, 2),
        "avg_spv_eur": _sf((_cur["sales_per_visitor"] * _cur["count_in"]).sum() / _cur["count_in"].sum() if _cur["count_in"].sum() else np.nan, 2),
        "avg_sales_per_sqm": _sf((_cur["turnover"].sum() / _cur["sq_meter"].fillna(0).sum()) if _cur["sq_meter"].fillna(0).sum() > 0 else np.nan, 2),
        "stores": int(_cur["shop_id"].nunique())
    }

    # Vorige periode (prev) als beschikbaar
    if 'prev' in locals() and prev is not None and not prev.empty:
        _prev = prev.copy()
        _prev["sales_per_sqm"] = _prev.get("sales_per_sqm", np.nan)
        reg_prev = {
            "total_turnover": _sf(_prev["turnover"].sum(), 0),
            "avg_conversion_pct": _sf((_prev["conversion_rate"] * _prev["count_in"]).sum() / _prev["count_in"].sum() if _prev["count_in"].sum() else np.nan, 2),
            "avg_spv_eur": _sf((_prev["sales_per_visitor"] * _prev["count_in"]).sum() / _prev["count_in"].sum() if _prev["count_in"].sum() else np.nan, 2),
            "avg_sales_per_sqm": _sf((_prev["turnover"].sum() / _prev["sq_meter"].fillna(0).sum()) if _prev["sq_meter"].fillna(0).sum() > 0 else np.nan, 2),
        }
    else:
        reg_prev = None

    # Tops & flops (sales/m²); val terug op SPV indien sqm ontbreekt
    rank_key = "sales_per_sqm" if _cur["sales_per_sqm"].notna().any() else "sales_per_visitor"
    tops = _cur.sort_values(rank_key, ascending=False).head(3)[["shop_name", rank_key]].to_dict(orient="records")
    flops = _cur.sort_values(rank_key, ascending=True).head(3)[["shop_name", rank_key]].to_dict(orient="records")

    # Uur‑norm (optioneel): best_hour per winkel o.b.v. omzet of bezoeken
    # (alleen als je per‑uur dataframe in deze pagina beschikbaar hebt; anders overslaan)
    best_hours = None
    if "df_cur" in locals() and not df_cur.empty and "hour" in df_cur.columns:
        metric_for_hour = "turnover" if (df_cur["turnover"].fillna(0).sum() > 0) else "count_in"
        bh = (df_cur.groupby(["shop_id","shop_name","hour"], as_index=False)
                     .agg(val=(metric_for_hour,"sum")))
        best_hours = (bh.sort_values(["shop_id","val"], ascending=[True,False])
                        .groupby(["shop_id","shop_name"]).head(1)[["shop_name","hour","val"]]
                        .to_dict(orient="records"))

    ai_region_context = {
        "region": region_name,
        "period": period,
        "kpis_now": reg_now,
        "kpis_prev": reg_prev,
        "ranking_basis": rank_key,
        "tops": tops,
        "flops": flops,
        "best_hours_sample": best_hours[:8] if best_hours else None  # korte lijst
    }
except Exception as e:
    ai_region_context = {"error": f"context_build_failed: {e}"}

# 4) Prompt (NL, regiomanager, actiegericht)
sys_msg_rm = (
    "Je bent regiomanager retail. Analyseer kort en formuleer max 5 concrete acties: "
    "2 bevindingen (vergelijk prestaties van winkels in de regio, afwijkingen t.o.v. peers_median). "
    "Gebruik bezoekersaantallen om de bevindingen te onderbouwen."
    "en maximaal 3 acties die je vandaag met store managers kunt afspreken. "
    "Gebruik Nederlands, wees meetbaar (noem bedragen als €X en percentages als 12,3%). "
    "Geef altijd 1 meettip (welke KPI dagelijks per winkel te volgen). "
    "Als peers_median ontbreekt, negeer die vergelijking. "
    "Leg nadruk op het delen van best practices en het aanpakken van zwakkere winkels."
)

usr_msg_rm = (
    "Context (JSON):\n"
    f"{ai_region_context}\n\n"
    "Schrijf puntsgewijs, maximaal 5 bullets, inclusief 1 meetritueel (wat per dag of uur te checken). "
    "Vermijd vage adviezen: geef concrete voorbeelden (bijv. extra FTE bij paskamers 12‑15u, bundelactie X, kassascript Y)."
    "(vergelijk vestigingen, benoem welke winkel uitvalt of juist sterk presteert). "
    "Maak het regio-concreet. "
)

# 5) OpenAI call
_openai_ready = False
try:
    openai_mod = import_module("openai")
    from openai import OpenAI
    _openai_ready = True
except Exception:
    pass

st.markdown("### 🤖 AI‑Coach (Regio)")
if not _openai_ready:
    render_ai_card("_OpenAI‑client niet gevonden. Voeg `openai>=1.40.0` toe aan requirements._")
elif "OPENAI_API_KEY" not in st.secrets or not st.secrets["OPENAI_API_KEY"]:
    render_ai_card("_Geen `OPENAI_API_KEY` in secrets. Voeg die toe voor live AI‑inzichten._")
else:
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.25,
            messages=[
                {"role": "system", "content": sys_msg_rm},
                {"role": "user", "content": usr_msg_rm},
            ],
        )
        insight_rm = resp.choices[0].message.content.strip()
        render_ai_card(insight_rm, subtitle="Regio‑analyse en concrete acties voor je filialen.")
    except Exception as e:
        render_ai_card(f"_AI‑insights konden niet geladen worden: {e}_")

# Optionele debug
with st.expander("🔧 AI‑debug (regio)"):
    st.json(ai_region_context)

# ---------- Debug ----------
with st.expander("🔧 Debug — API calls en samples"):
    st.write("Cur params:", p_cur, "status", s_cur)
    st.write("Prev params:", p_prev, "status", s_prev if has_true_previous else None)
    st.write("Cur head:", df_cur.head())
