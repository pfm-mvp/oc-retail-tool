"""
TOOL-002: Revenue Opportunity Quantifier
========================================
Vertaalt prospect-kansen naar concrete omzet-opportunity voor PFM.

Pricing model:
  Setup fee:    # winkels × €1.000
  Recurring:    # winkels × €240/jaar

Bron: Chris brain dump 2026-06-06
"""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils_pfmx import inject_css

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
inject_css()

BRAND = "#762181"
SETUP_PER_STORE = 1_000
RECURRING_PER_STORE = 240

# ── Helper ─────────────────────────────────────────────────────────────────────
def fmt_eur(n, d=0):
    try:
        return f"€{float(n):,.{d}f}".replace(",", ".")
    except Exception:
        return "€0"

def fmt_int(n):
    try:
        return f"{int(round(float(n))):,}".replace(",", ".")
    except Exception:
        return "0"

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("💰 Revenue Opportunity Quantifier")
st.caption("Vertaal prospect-kansen naar concrete PFM omzet-opportunity")

# ── Pricing model sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pricing Model")
    setup = st.number_input("Setup fee per winkel (€)", min_value=0, value=SETUP_PER_STORE, step=100)
    recurring = st.number_input("Recurring per winkel/jaar (€)", min_value=0, value=RECURRING_PER_STORE, step=50)
    st.divider()
    st.markdown(f"""
    **Huidig model:**
    - Setup: **{fmt_eur(setup)}** per winkel
    - Recurring: **{fmt_eur(recurring)}/jaar** per winkel
    
    Totaal per winkel jaar 1: **{fmt_eur(setup + recurring)}**
    """)

# ── Prospect input ─────────────────────────────────────────────────────────────
st.subheader("📋 Prospects toevoegen")

input_mode = st.radio("Invoer", ["Handmatig", "Bulk (CSV)"], horizontal=True, label_visibility="collapsed")

if "prospects_df" not in st.session_state:
    st.session_state.prospects_df = pd.DataFrame(
        columns=["retailer", "country", "stores_eu", "status", "source"]
    )

if input_mode == "Handmatig":
    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    with c1:
        retailer = st.text_input("Retailer", placeholder="Bijv. H&M, Zara, C&A...")
    with c2:
        country = st.text_input("Land (primair)", value="NL", placeholder="NL, BE, DE...")
    with c3:
        stores = st.number_input("# EU winkels", min_value=0, value=0, step=1)
    with c4:
        status = st.selectbox("Status", ["Prospect", "Lead", "Active", "Lost"])
    
    source = st.text_input("Bron", placeholder="Artikel, brain dump, referral...")
    
    if st.button("➕ Toevoegen", type="primary") and retailer and stores > 0:
        new_row = pd.DataFrame([{
            "retailer": retailer.strip(),
            "country": country.strip().upper(),
            "stores_eu": int(stores),
            "status": status,
            "source": source.strip()
        }])
        st.session_state.prospects_df = pd.concat(
            [st.session_state.prospects_df, new_row], ignore_index=True
        )
        st.rerun()

else:
    st.markdown("""
    Upload CSV met kolommen: `retailer`, `country`, `stores_eu`, `status`, `source`
    
    `status` waarden: Prospect, Lead, Active, Lost
    """)
    uploaded = st.file_uploader("CSV upload", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            required = {"retailer", "country", "stores_eu"}
            if required.issubset(df_upload.columns):
                for col in ["status", "source"]:
                    if col not in df_upload.columns:
                        df_upload[col] = "Prospect" if col == "status" else ""
                df_upload["stores_eu"] = pd.to_numeric(df_upload["stores_eu"], errors="coerce").fillna(0).astype(int)
                df_upload["country"] = df_upload["country"].astype(str).str.strip().str.upper()
                df_upload["retailer"] = df_upload["retailer"].astype(str).str.strip()
                st.session_state.prospects_df = pd.concat(
                    [st.session_state.prospects_df, df_upload], ignore_index=True
                )
                st.success(f"{len(df_upload)} prospects geladen")
                st.rerun()
            else:
                st.error(f"Ontbrekende kolommen: {required - set(df_upload.columns)}")
        except Exception as e:
            st.error(f"CSV fout: {e}")

# ── Prospect table ─────────────────────────────────────────────────────────────
df = st.session_state.prospects_df.copy()

if df.empty:
    st.info("Nog geen prospects. Voeg retailers toe bovenaan.")
    st.stop()

# Compute opportunity
df["setup_fee"] = df["stores_eu"] * setup
df["recurring_annual"] = df["stores_eu"] * recurring
df["total_y1"] = df["setup_fee"] + df["recurring_annual"]

# ── KPI cards ──────────────────────────────────────────────────────────────────
total_stores = df["stores_eu"].sum()
total_setup = df["setup_fee"].sum()
total_recurring = df["recurring_annual"].sum()
total_y1 = df["total_y1"].sum()
active_pipeline = df[df["status"].isin(["Prospect", "Lead"])]["total_y1"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("🏪 Totaal EU winkels", fmt_int(total_stores))
k2.metric("💶 Setup fee (totaal)", fmt_eur(total_setup))
k3.metric("🔄 Recurring/jaar", fmt_eur(total_recurring))
k4.metric("📊 Pipeline jaar 1", fmt_eur(active_pipeline), help="Prospects + Leads only")

# ── Per status breakdown ───────────────────────────────────────────────────────
st.subheader("📈 Opportunity per status")
status_summary = (
    df.groupby("status", as_index=False)
      .agg(
          retailers=("retailer", "count"),
          stores=("stores_eu", "sum"),
          setup=("setup_fee", "sum"),
          recurring=("recurring_annual", "sum"),
          total_y1=("total_y1", "sum"),
      )
      .sort_values("total_y1", ascending=False)
)

display = status_summary.copy()
display["stores"] = display["stores"].map(fmt_int)
display["setup"] = display["setup"].map(fmt_eur)
display["recurring"] = display["recurring"].map(fmt_eur)
display["total_y1"] = display["total_y1"].map(fmt_eur)
display.rename(columns={
    "status": "Status",
    "retailers": "Retailers",
    "stores": "EU Winkels",
    "setup": "Setup Fee",
    "recurring": "Recurring/jaar",
    "total_y1": "Jaar 1 Totaal",
}, inplace=True)
st.dataframe(display, use_container_width=True, hide_index=True)

# ── Per country breakdown ──────────────────────────────────────────────────────
st.subheader("🌍 Opportunity per land")
country_summary = (
    df.groupby("country", as_index=False)
      .agg(
          retailers=("retailer", "count"),
          stores=("stores_eu", "sum"),
          setup=("setup_fee", "sum"),
          recurring=("recurring_annual", "sum"),
          total_y1=("total_y1", "sum"),
      )
      .sort_values("total_y1", ascending=False)
)

display_c = country_summary.copy()
display_c["stores"] = display_c["stores"].map(fmt_int)
display_c["setup"] = display_c["setup"].map(fmt_eur)
display_c["recurring"] = display_c["recurring"].map(fmt_eur)
display_c["total_y1"] = display_c["total_y1"].map(fmt_eur)
display_c.rename(columns={
    "country": "Land",
    "retailers": "Retailers",
    "stores": "EU Winkels",
    "setup": "Setup Fee",
    "recurring": "Recurring/jaar",
    "total_y1": "Jaar 1 Totaal",
}, inplace=True)
st.dataframe(display_c, use_container_width=True, hide_index=True)

# ── Full prospect table ────────────────────────────────────────────────────────
st.subheader("📋 Alle prospects")

# Allow delete
edit_df = df.copy()
edit_df["setup_fee"] = edit_df["setup_fee"].map(fmt_eur)
edit_df["recurring_annual"] = edit_df["recurring_annual"].map(fmt_eur)
edit_df["total_y1"] = edit_df["total_y1"].map(fmt_eur)
edit_df.rename(columns={
    "retailer": "Retailer",
    "country": "Land",
    "stores_eu": "EU Winkels",
    "status": "Status",
    "source": "Bron",
    "setup_fee": "Setup Fee",
    "recurring_annual": "Recurring/jaar",
    "total_y1": "Jaar 1 Totaal",
}, inplace=True)
st.dataframe(edit_df, use_container_width=True, hide_index=True)

# ── Actions ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    if st.button("🗑️ Alle prospects wissen", type="secondary"):
        st.session_state.prospects_df = pd.DataFrame(
            columns=["retailer", "country", "stores_eu", "status", "source"]
        )
        st.rerun()
with c2:
    csv = df[["retailer", "country", "stores_eu", "status", "source", "setup_fee", "recurring_annual", "total_y1"]].to_csv(index=False)
    st.download_button("📥 Export CSV", csv, "revenue_opportunities.csv", "text/csv")

# ── 3-year projection ──────────────────────────────────────────────────────────
st.subheader("📊 3-jaar projectie")
years = [1, 2, 3]
proj_data = []
cumulative = 0
for y in years:
    if y == 1:
        yearly = total_setup + total_recurring
    else:
        yearly = total_recurring
    cumulative += yearly
    proj_data.append({
        "Jaar": y,
        "Setup": total_setup if y == 1 else 0,
        "Recurring": total_recurring,
        "Jaaromzet": yearly,
        "Cumulatief": cumulative,
    })

proj_df = pd.DataFrame(proj_data)
proj_display = proj_df.copy()
for col in ["Setup", "Recurring", "Jaaromzet", "Cumulatief"]:
    proj_display[col] = proj_display[col].map(fmt_eur)
st.dataframe(proj_display, use_container_width=True, hide_index=True)

st.caption("TOOL-002 · Revenue Opportunity Quantifier · PFM Intelligence")