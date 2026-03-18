# Home.py — robust import of utils_pfmx
import streamlit as st

try:
    from utils_pfmx import inject_css
except Exception as e:
    st.error("Kon `utils_pfmx` niet importeren. Details:")
    st.exception(e)   # toont de echte stacktrace i.p.v. redacted
    st.stop()

st.set_page_config(page_title="PFM Retail Tools", page_icon="🧰", layout="wide")
inject_css()

st.title("PFM Retail Tools")
st.markdown("Kies een app via de linker navigatie.")
