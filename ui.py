
import streamlit as st

def brand_colors():
    primary = "#762181"
    success = st.secrets.get("SUCCESS_COLOR", "#16A34A")
    danger = st.secrets.get("DANGER_COLOR", "#E63946")
    tint = "#F9F7FB"
    return {"primary": primary, "success": success, "danger": danger, "tint": tint}

def kpi_card(label:str, value_html:str, subtitle:str, tone:str="neutral"):
    colors = brand_colors()
    border = "#eeeeee"
    if tone == "good":
        border = colors["success"]
    elif tone == "bad":
        border = colors["danger"]
    elif tone == "primary":
        border = colors["primary"]
    st.markdown(f"""
    <div style="border:2px solid {border};border-radius:16px;padding:16px;margin-bottom:8px;">
      <div style="font-size:28px;font-weight:700;line-height:1">{value_html}</div>
      <div style="color:#666;font-size:12px">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)
