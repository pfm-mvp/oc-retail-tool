
import os, sys
from pathlib import Path
import streamlit as st
import pandas as pd
from datetime import datetime, time
from zoneinfo import ZoneInfo

# Ensure root importable (shop_mapping.py in root)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shop_mapping import SHOP_NAME_MAP
from utils_pfmx import api_get_report, friendly_error, inject_css
from helpers_normalize import normalize_vemcount_response

st.set_page_config(layout="wide")
inject_css()
TZ = ZoneInfo("Europe/Amsterdam")
NAME_TO_ID = {v:k for k,v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k:v for k,v in SHOP_NAME_MAP.items()}

st.title("👻 Ghost / API Smoke Test — plain keys, repeated")

sid = next(iter(SHOP_NAME_MAP.keys()))
tests = {
    "A_single_shop_single_kpi": [("data", sid), ("data_output","count_in"), ("source","shops"), ("period","day")],
    "B_single_shop_multi_kpi":  [("data", sid), ("data_output","count_in"), ("data_output","conversion_rate"),
                                 ("data_output","turnover"), ("data_output","sales_per_visitor"),
                                 ("source","shops"), ("period","last_week")],
    "C_multi_shop_multi_kpi":   [("data", sid), ("data", list(SHOP_NAME_MAP.keys())[1]), ("data", list(SHOP_NAME_MAP.keys())[2]),
                                 ("data_output","count_in"), ("data_output","turnover"),
                                 ("source","shops"), ("period","yesterday")],
}

for name, p in tests.items():
    st.subheader(name)
    js = api_get_report(p)
    if friendly_error(js):
        continue
    df = normalize_vemcount_response(js, SHOP_NAME_MAP)
    st.json({"rows": 0 if df is None else len(df)})
    if df is not None and not df.empty:
        st.dataframe(df.head(10))
