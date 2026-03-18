# pages/03_AI_Advisor_Weather_CBS.py
import os, sys, json
from typing import Dict, List
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────── Setup ───────────────────────────
st.set_page_config(page_title="AI Advisor — Weer + CBS", page_icon="🧭", layout="wide")
st.title("🧭 AI Advisor — Weer + CBS")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Project helpers / API
try:
    from shop_mapping import SHOP_NAME_MAP
except Exception:
    SHOP_NAME_MAP = None
from helpers_normalize import normalize_vemcount_response
from helpers_shop import ID_TO_NAME, get_ids_by_region, REGIONS
from utils_pfmx import api_get_report, friendly_error

# Services
from services.weather_service import get_daily_forecast
from services.cbs_service import (
    get_consumer_confidence,
    get_cci_series,
    get_retail_index,
    list_retail_branches,
)
from services.advisor import build_advice

# Secrets
def _get_secret(key: str, env_fallback: str = "") -> str:
    val = st.secrets.get(key) or os.getenv(env_fallback or key.upper()) or ""
    return (val or "").strip()

OPENWEATHER_KEY = _get_secret("openweather_api_key", "OPENWEATHER_API_KEY")
CBS_DATASET     = _get_secret("cbs_dataset", "CBS_DATASET")
API_URL         = _get_secret("API_URL").rstrip("/")

missing = []
if not OPENWEATHER_KEY: missing.append("openweather_api_key")
if not CBS_DATASET:     missing.append("cbs_dataset")
if not API_URL:         missing.append("API_URL")
if missing:
    st.error("Missing secrets: " + ", ".join(missing) + "\n\nCheck Streamlit → Settings → Secrets.")
    st.stop()

# ─────────────────────────── UI ───────────────────────────
c_top = st.columns([1,1,1,2])
with c_top[0]:
    region = st.selectbox("Regio", options=["ALL"] + list(REGIONS), index=0)
with c_top[1]:
    period_hist = st.selectbox("Historische periode", ["last_month", "this_year", "last_year"], index=0)
with c_top[2]:
    days_ahead = st.slider("Dagen vooruit (weerfactor)", 7, 30, 14)
with c_top[3]:
    st.caption("Korte termijn = weer-correctie; langere termijn = CCI-trend + laatste SPV/conv.")

st.subheader("Macro-context (CBS)")
c1, c2, c3 = st.columns([1,1,2])
with c1:
    months_back = st.slider("Maanden terug (CCI)", 6, 36, 18)
with c2:
    use_retail = st.checkbox("Toon detailhandel-index (85828NED)", value=False)
with c3:
    default_branch = "DH_NONFOOD"
    try:
        dim_name, branch_items = list_retail_branches("85828NED")
    except Exception:
        dim_name, branch_items = None, []
    if branch_items:
        title_to_key = {b["title"]: str(b["key"]) for b in branch_items}
        titles = list(title_to_key.keys())
        def_idx = next((i for i,t in enumerate(titles) if "nonfood" in t.lower()), 0)
        branch_title = st.selectbox("Branche (CBS)", titles, index=def_idx)
        branch_key = title_to_key[branch_title]
    else:
        branch_title = st.selectbox("Branche (CBS)", ["DH_NONFOOD","DH_TOTAAL","DH_FOOD"], index=0)
        branch_key = branch_title

# CCI series vóór de knop (voor tiles)
try:
    cci_series = get_cci_series(months_back=months_back, dataset=CBS_DATASET)
except Exception as e:
    cci_series = []
    st.info(f"CCI niet beschikbaar: {e}")

# Detailhandel (optioneel)
try:
    retail_series = []
    if use_retail:
        retail_series = (
            get_retail_index(branch_code_or_title=branch_key, months_back=months_back)
            or get_retail_index(branch_code_or_title=branch_title, months_back=months_back)
            or get_retail_index(branch_code_or_title="DH_TOTAAL", months_back=months_back)
        ) or []
except Exception as e:
    retail_series = []
    if use_retail:
        st.info(f"Detailhandelreeks (85828NED) niet beschikbaar: {e}")

# ─────────────────────────── Helpers ───────────────────────────
def parse_shop_meta(val):
    if isinstance(val, dict):
        return {"name": val.get("name"), "postcode": val.get("postcode"), "region": val.get("region")}
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("{") and ("name" in s or "postcode" in s or "region" in s):
            try:
                j = json.loads(s.replace("''", '"').replace("'", '"'))
                return {"name": j.get("name"), "postcode": j.get("postcode"), "region": j.get("region")}
            except Exception:
                pass
        return {"name": s, "postcode": None, "region": None}
    return {"name": None, "postcode": None, "region": None}

def add_effective_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    dt_date = pd.to_datetime(d.get("date"), errors="coerce")
    dt_ts   = pd.to_datetime(d.get("timestamp"), errors="coerce")
    d["date_eff"]  = dt_date.fillna(dt_ts)
    d["weekday"]   = d["date_eff"].dt.weekday
    d["ym"]        = d["date_eff"].dt.to_period("M").astype(str)
    d["date_only"] = d["date_eff"].dt.date
    iso = d["date_eff"].dt.isocalendar()
    d["iso_week"] = iso.week.astype("Int64")

    meta = d["shop_name"].apply(parse_shop_meta)
    d["shop_clean"] = meta.apply(lambda x: x.get("name"))
    d["postcode"]   = meta.apply(lambda x: x.get("postcode"))
    d["region"]     = meta.apply(lambda x: x.get("region"))

    d["date_debug"] = d["date_eff"].dt.date
    return d

def fetch_hist_kpis_df(shop_ids, period: str) -> pd.DataFrame:
    metrics = ["count_in", "conversion_rate", "turnover", "sales_per_visitor"]
    params = [("data", int(sid)) for sid in shop_ids]
    params += [("data_output", k) for k in metrics]
    params += [("source","shops"), ("period", period), ("step","day")]
    js = api_get_report(params)
    if friendly_error(js, period):
        return pd.DataFrame()
    name_map = SHOP_NAME_MAP or ID_TO_NAME
    return normalize_vemcount_response(js, name_map, kpi_keys=metrics)

def build_weekday_baselines(df: pd.DataFrame) -> dict:
    if df is None or df.empty: return {}
    d = add_effective_date_cols(df)
    out: Dict[int, Dict[str, dict]] = {}
    for wd, g in d.groupby("weekday"):
        stores = {}
        for sid, gs in g.groupby("shop_id"):
            if "sales_per_visitor" in gs.columns:
                spv_series = gs["sales_per_visitor"]
            else:
                denom = gs["count_in"].replace(0, pd.NA)
                spv_series = (gs["turnover"] / denom).fillna(0)
            stores[str(int(sid))] = {
                "visitors":     float(gs["count_in"].mean()),
                "conversion":   float(gs["conversion_rate"].mean()),  # in %
                "spv":          float(spv_series.mean()),
                "spv_median":   float(spv_series.median()),
                "visitors_p30": float(gs["count_in"].quantile(0.30)),
            }
        out[wd] = stores
    return out

def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_date_cols(df)
    g = d.groupby("ym", as_index=False).agg(visitors=("count_in","sum"), turnover=("turnover","sum"))
    d["weighted_conv"] = d["conversion_rate"] * d["count_in"]
    conv = d.groupby("ym", as_index=False).agg(conv_num=("weighted_conv","sum"), conv_den=("count_in","sum"))
    conv["conversion"] = (conv["conv_num"]/conv["conv_den"]).fillna(0)
    out = g.merge(conv[["ym","conversion"]], on="ym", how="left")
    out["spv"] = (out["turnover"]/out["visitors"]).replace([float("inf")], 0).fillna(0)
    return out

def mom_yoy(dfm: pd.DataFrame):
    if dfm is None or dfm.empty: return {}
    m = dfm.copy()
    m["ym_dt"] = pd.to_datetime(m["ym"].astype(str) + "-01", errors="coerce")
    m = m.dropna(subset=["ym_dt"]).sort_values("ym_dt").reset_index(drop=True)
    if m.empty: return {}
    last = m.iloc[-1]
    prev = m.iloc[-2] if len(m)>1 else None
    yoy_dt  = last["ym_dt"] - pd.DateOffset(years=1)
    yoy_row = m.loc[m["ym_dt"] == yoy_dt]
    yoy_row = yoy_row.iloc[0] if not yoy_row.empty else None

    def pct(a,b):
        if b in [0,None] or pd.isna(b): return None
        try: return (float(a)/float(b) - 1) * 100
        except Exception: return None

    return {
        "this_label": last["ym_dt"].strftime("%Y-%m"),
        "turnover":   float(last.get("turnover", 0)),
        "visitors":   float(last.get("visitors", 0)),
        "conversion": float(last.get("conversion", 0)),
        "spv":        float(last.get("spv", 0)),
        "mom": {
            "turnover":   pct(last.get("turnover", 0),   prev.get("turnover", 0))   if prev is not None else None,
            "visitors":   pct(last.get("visitors", 0),   prev.get("visitors", 0))   if prev is not None else None,
            "conversion": pct(last.get("conversion", 0), prev.get("conversion", 0)) if prev is not None else None,
            "spv":        pct(last.get("spv", 0),        prev.get("spv", 0))        if prev is not None else None,
        },
        "yoy": {
            "turnover":   pct(last.get("turnover", 0),   yoy_row.get("turnover", 0))   if yoy_row is not None else None,
            "visitors":   pct(last.get("visitors", 0),   yoy_row.get("visitors", 0))   if yoy_row is not None else None,
            "conversion": pct(last.get("conversion", 0), yoy_row.get("conversion", 0)) if yoy_row is not None else None,
            "spv":        pct(last.get("spv", 0),        yoy_row.get("spv", 0))        if yoy_row is not None else None,
        },
    }

def estimate_weather_uplift(baseline_day: dict, forecast_days: List[dict]) -> dict:
    daily = []
    for f in forecast_days:
        wd = pd.to_datetime(f["date"]).weekday()
        base = baseline_day.get(wd, {"visitors": 0.0, "spv": 0.0})
        base_vis = float(base["visitors"]); spv = float(base["spv"])
        pop  = float(f.get("pop", 0.0))
        temp = float(f.get("temp", 15.0))
        adj_vis = base_vis * (1 - 0.20*pop) * (1 + 0.01*(temp-15.0))
        row = {
            "date": f["date"], "weekday": wd, "temp": temp, "pop": pop,
            "base_visitors": base_vis, "adj_visitors": adj_vis, "spv": spv,
            "base_turnover": base_vis * spv, "adj_turnover": adj_vis * spv
        }
        row["delta_turnover"] = row["adj_turnover"] - row["base_turnover"]
        daily.append(row)
    base_total = sum(x["base_turnover"] for x in daily)
    adj_total  = sum(x["adj_turnover"]  for x in daily)
    delta_total = adj_total - base_total
    delta_pct = (delta_total/base_total*100) if base_total else None
    return {"daily": daily, "base_total": base_total, "adj_total": adj_total,
            "delta_total": delta_total, "delta_pct": delta_pct}

def _eur0(x):  # € 0 decimals
    try: return "€{:,.0f}".format(float(x)).replace(",", ".")
    except Exception: return "€0"

def _eur2(x):  # € 2 decimals
    try: return "€{:,.2f}".format(float(x)).replace(",", ".")
    except Exception: return "€0.00"

def _pct1(x):
    return "—" if (x is None or pd.isna(x)) else f"{float(x):+.1f}%"

# ─────────────────────────── Shop selectie ───────────────────────────
st.caption("Selecteer regio en druk op de knop om aanbevelingen te genereren.")
if region == "ALL":
    shop_ids = sorted([int(k) for k in (SHOP_NAME_MAP or ID_TO_NAME).keys()])
else:
    shop_ids = get_ids_by_region(region) or sorted([int(k) for k in (SHOP_NAME_MAP or ID_TO_NAME).keys()])

st.write(f"**{len(shop_ids)}** winkels geselecteerd in regio: **{region}**")
st.text(f"ShopIDs → {shop_ids[:25]}{' …' if len(shop_ids)>25 else ''}")

# ─────────────────────────── Action ───────────────────────────
if st.button("Genereer aanbevelingen"):
    # 1) Historisch ophalen
    df_raw = fetch_hist_kpis_df(shop_ids, period_hist)
    with st.expander("🛠️ Debug — eerste rijen (hist KPI’s)"):
        if df_raw.empty:
            st.write("Leeg.")
        else:
            dbg = add_effective_date_cols(df_raw)
            dshow = dbg[["date_debug","shop_id","shop_clean","postcode","region","count_in","conversion_rate","turnover","sales_per_visitor"]].rename(
                columns={"date_debug":"date","shop_clean":"name"})
            st.dataframe(dshow.head(15), use_container_width=True)

    if df_raw.empty:
        st.warning("Geen historische KPI-data. Probeer ‘this_year’ of ‘last_year’.")
        st.stop()

    # 2) Baselines + maand-aggregaten
    df  = add_effective_date_cols(df_raw)
    baseline = build_weekday_baselines(df)
    dfm = monthly_agg(df)
    trend = mom_yoy(dfm)

    # 3) CCI (laatste waarde + periode)
    try:
        cci_info = get_consumer_confidence(CBS_DATASET)
        last_cci = float(cci_info.get("value", 0))
        cci_period = str(cci_info.get("period", "n/a"))
    except Exception:
        # fallback: neem laatste uit cci_series
        if cci_series:
            last_cci = float(cci_series[-1].get("cci", 0))
            cci_period = str(cci_series[-1].get("period", "n/a"))
        else:
            last_cci, cci_period = 0.0, "n/a"

    # 4) Weer-forecast ophalen + normaliseren (ISO-datumstrings)
    try:
        raw_fc = get_daily_forecast(lat=52.37, lon=4.90, api_key=OPENWEATHER_KEY, days=days_ahead)
    except Exception:
        raw_fc = [{"date": pd.Timestamp.today().date() + pd.Timedelta(days=i), "temp": 15.0, "pop": 0.0} for i in range(days_ahead)]

    def _normalize_forecast_dates(fx):
        norm = []
        for f in fx or []:
            d_iso = pd.to_datetime(f.get("date"), errors="coerce")
            if pd.isna(d_iso): continue
            norm.append({
                "date": d_iso.date().isoformat(),
                "temp": float(f.get("temp", 0.0) or 0.0),
                "pop":  float(f.get("pop", 0.0) or 0.0),
            })
        return norm

    forecast = _normalize_forecast_dates(raw_fc)

    # 5) AI-advies (per dag) — vereist last_cci & forecast
    advice = build_advice("Your Company", baseline, forecast, float(last_cci))

    # 6) Regionale baseline (gemiddelden over winkels per weekday) + weer-uplift
    baseline_day = {}
    for wd, storemap in baseline.items():
        if not storemap: continue
        visitors = pd.Series([v["visitors"] for v in storemap.values()]).mean()
        spv      = pd.Series([v["spv"]      for v in storemap.values()]).mean()
        baseline_day[wd] = {"visitors": float(visitors), "spv": float(spv)}
    if not baseline_day:
        st.warning("Geen baseline beschikbaar (te weinig dagen). Kies ruimere periode.")
        st.stop()

    wx = estimate_weather_uplift(baseline_day, forecast)

    # ── Silver-platter regio
    st.subheader("🔎 Silver-platter samenvatting (regio)")
    if trend:
        colA, colB, colC, colD = st.columns(4)
        colA.metric(f"Omzet {trend.get('this_label','')}", _eur0(trend.get('turnover', 0)), _pct1(trend['mom'].get('turnover')))
        colB.metric("Bezoekers", f"{trend.get('visitors', 0):,.0f}".replace(",", "."), _pct1(trend['mom'].get('visitors')))
        colC.metric("Conversie", f"{trend.get('conversion', 0):.2f}%", _pct1(trend['mom'].get('conversion')))
        colD.metric("SPV", _eur2(trend.get('spv', 0)), _pct1(trend['mom'].get('spv')))

    # ── Verwachting per week (2 eerstvolgende weken)
    wdf = pd.DataFrame(wx["daily"])
    if not wdf.empty:
        wdf["date"] = pd.to_datetime(wdf["date"])
        wdf["isoweek"] = wdf["date"].dt.isocalendar().week.astype(int)
        wsum = (
            wdf.groupby("isoweek", as_index=False)
               .agg(omzet=("adj_turnover","sum"), bezoekers=("adj_visitors","sum"))
               .sort_values("isoweek")
        )
        st.subheader("🗺️ Verwachting per week & maand (regio)")
        bullets = " • ".join([f"Week {int(r.isoweek)}: verwacht {_eur0(r.omzet)} omzet en {int(r.bezoekers):,} bezoekers".replace(",", ".")
                               for _, r in wsum.iloc[:2].iterrows()])
        st.write("• " + bullets)

        richting = "neutraal"
        if wx["delta_pct"] is not None:
            richting = "opwaarts" if wx["delta_pct"] > 1 else ("neerwaarts" if wx["delta_pct"] < -1 else "neutraal")
        st.markdown(
            f'<div style="background:#feecec;padding:10px;border-radius:8px;">'
            f'<b>Conclusie:</b> {richting}. Voor de komende {days_ahead} dagen verwachten we in totaal '
            f'<b>{_eur0(wx["adj_total"])}</b> omzet. '
            f'Weercomponent: regen & temperatuur t.o.v. normaal; CCI-effect: lichte SPV-correctie.</div>',
            unsafe_allow_html=True
        )

    # ── CCI tiles & grafiek (maandelijks, huidig jaar)
    c1, c2 = st.columns([1,5])
    with c1:
        cci_mm = None
        if len(cci_series) >= 2:
            try:
                cci_mm = float(cci_series[-1]["cci"]) - float(cci_series[-2]["cci"])
            except Exception:
                cci_mm = None
        st.metric("Consumentenvertrouwen (CBS)", f"{last_cci:.1f}", delta=(f"{cci_mm:+.1f} m/m" if cci_mm is not None else None), help=f"Periode: {cci_period}")
        if abs(last_cci) > 100:
            st.caption("⚠️ CCI lijkt buiten de normale bandbreedte (verwacht ~-60..+40). Controleer dataset.")
    with c2:
        if cci_series:
            try:
                cci_df = pd.DataFrame(cci_series)
                # normaliseer period "YYYYMM" of "YYYYMMxx" / "YYYYMMnn"
                def _p2dt(s):
                    s = str(s)
                    # vang '2025MM11' of '2025-11' of '202511'
                    s = s.replace("MM", "").replace("-", "")
                    s = s[:6]
                    return pd.to_datetime(s + "01", format="%Y%m%d", errors="coerce")
                cci_df["ym_dt"] = cci_df["period"].apply(_p2dt) if "period" in cci_df.columns else pd.NaT
                this_year = pd.Timestamp.today().year
                cci_y = cci_df.dropna(subset=["ym_dt"])
                cci_y = cci_y[cci_y["ym_dt"].dt.year == this_year]
                if not cci_y.empty:
                    cci_y = cci_y.sort_values("ym_dt")
                    cci_y = cci_y.set_index("ym_dt")[["cci"]].rename(columns={"cci":"CCI"})
                    with st.expander("📈 CCI dit jaar (maandelijks)"):
                        st.line_chart(cci_y)
            except Exception:
                pass

    # ── Optioneel: dag-adviesregels samengevat (geen dag-explosie)
    with st.expander("📅 Komende dagen — acties & weer"):
        for d in advice["days"]:
            st.write(f"• {d['date']}: {d['weather']['temp']:.1f}°C, regen {int(d['weather']['pop']*100)}%")
        st.caption("Dag-adviezen zijn indicatief; de week/m maand-blokken hierboven zijn leidend.")

    # ── CCI vs. Omzet — genormaliseerde trend
    st.subheader("📊 CCI vs. Omzet — genormaliseerde trend")
    def _merge_cci_vs_turnover(dfm_in: pd.DataFrame, cci_series_in: List[dict]) -> pd.DataFrame:
        if dfm_in is None or dfm_in.empty or not cci_series_in:
            return pd.DataFrame()
        a = dfm_in.copy()
        a["ym_dt"] = pd.to_datetime(a["ym"].astype(str) + "-01", errors="coerce")
        a = a.dropna(subset=["ym_dt"])
        a["ym_key"] = a["ym_dt"].dt.strftime("%YMM")  # '2025MM10'

        b = pd.DataFrame(cci_series_in).copy()
        if "period" in b.columns:
            b["ym_key"] = b["period"].astype(str)      # '2025MM10'
        elif "Periods" in b.columns:
            b["ym_key"] = b["Periods"].astype(str)
        else:
            return pd.DataFrame()
        try:
            b["CCI_val"] = b["cci"].astype(float)
        except Exception:
            return pd.DataFrame()

        merged = a.merge(b[["ym_key","CCI_val"]], on="ym_key", how="inner").sort_values("ym_dt")
        if merged.empty: return merged
        base_cci, base_turn = merged["CCI_val"].iloc[0], merged["turnover"].iloc[0]
        if base_cci == 0 or base_turn == 0: return pd.DataFrame()
        merged["CCI_idx"]   = (merged["CCI_val"]/base_cci)*100
        merged["Omzet_idx"] = (merged["turnover"]/base_turn)*100
        return merged

    merged = _merge_cci_vs_turnover(dfm, cci_series)
    if merged.empty:
        st.info("Te weinig overlap tussen CCI-maanden en omzetmaanden om een trend te tonen.")
    else:
        st.line_chart(merged.set_index("ym_dt")[["CCI_idx","Omzet_idx"]])

    # ── Detailhandel (optioneel)
    if use_retail:
        if retail_series:
            last_r = retail_series[-1]
            st.metric(f"Detailhandel ({last_r['branch']}) — {last_r['series']}", f"{last_r['retail_value']:.1f}")
            with st.expander("🛍️ Detailhandel reeks (CBS 85828NED)"):
                st.line_chart({"Retail": [x["retail_value"] for x in retail_series]})
        else:
            st.info("Geen detailhandelreeks gevonden voor deze branche/periode (85828NED).")
