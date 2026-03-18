
import streamlit as st
from urllib.parse import urlencode
from shop_mapping import SHOP_NAME_MAP
from utils_pfmx import inject_css, build_params_reports_plain, api_get_report, api_get_live_inside, friendly_error

st.set_page_config(page_title="API Smoke Test", page_icon="🧪", layout="wide")
inject_css()
st.title("🧪 API Smoke Test")

ids = list(SHOP_NAME_MAP.keys())
if not ids:
    st.warning("Geen shop_ids in SHOP_NAME_MAP."); st.stop()

st.subheader("1) get-report (multi data & multi data_output)")
test_ids = ids[: min(3, len(ids))]
outputs = ["count_in","conversion_rate","turnover","sales_per_visitor"]
params = build_params_reports_plain("shops","yesterday",test_ids,outputs,period_step="day")
st.code("/get-report?" + urlencode(params, doseq=True))

js = api_get_report(params)
if not friendly_error(js, "smoke:get-report"):
    with st.expander("JSON get-report"):
        st.json(js)

st.subheader("2) live-inside (source=locations & data=<shop_id>)")
st.code("/report/live-inside?source=locations&" + urlencode([('data', i) for i in test_ids], doseq=True))
ljs = api_get_live_inside(test_ids, source="locations")
if not friendly_error(ljs, "smoke:live-inside"):
    with st.expander("JSON live-inside"):
        st.json(ljs)

st.success("Smoke test uitgevoerd.")
