# services/radar_service.py

import numpy as np
import pandas as pd


def _safe_median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if s.dropna().empty:
        return np.nan
    return float(s.median(skipna=True))


def build_region_store_radar(
    df_period: pd.DataFrame,
    region_shops: pd.DataFrame,
    store_key_col: str,
    capture_weekly: pd.DataFrame | None = None,
    cbs_retail_month: pd.DataFrame | None = None,
    cci_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Bouwt een samengestelde radar-index per winkel in de regio.

    Input:
    - df_period: dagdata in geselecteerde periode (met kolommen:
        date, footfall, turnover, sales_per_visitor (optioneel), conversion_rate (optioneel))
    - region_shops: mapping met per winkel o.a. id, store_display, sqm_effective
    - store_key_col: kolomnaam in df_period die de winkel-id bevat
    - capture_weekly: regioweekdata met capture_rate (optioneel, wordt alleen gebruikt als context)
    - cbs_retail_month: CBS retail index (optioneel)
    - cci_df: consumentenvertrouwen-index (optioneel)

    Output:
    DataFrame met o.a.:
    - radar_icon (emoji)
    - store_name
    - radar_score (0–200, ~100 = regio-median)
    - headline (tekst)
    - short_reason (korte toelichting)
    - turnover, footfall, sales_per_visitor, turnover_per_sqm
    - potential_period (extra omzet in periode als winkel op regiomedian per m² zit)
    - potential_annual (geannualiseerde potentie)
    """

    if df_period is None or df_period.empty:
        return pd.DataFrame()

    df = df_period.copy()

    # Zorg dat date datetime is
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame()

    # Periode lengte voor jaarprojectie
    period_days = (df["date"].max() - df["date"].min()).days + 1
    if period_days <= 0:
        period_days = 1
    annual_factor = 365.0 / period_days

    # --- Aggregatie per winkel over de gekozen periode ---
    agg_dict: dict[str, str] = {}
    if "footfall" in df.columns:
        agg_dict["footfall"] = "sum"
    if "turnover" in df.columns:
        agg_dict["turnover"] = "sum"
    if "sales_per_visitor" in df.columns:
        agg_dict["sales_per_visitor"] = "mean"
    if "conversion_rate" in df.columns:
        agg_dict["conversion_rate"] = "mean"

    if not agg_dict:
        return pd.DataFrame()

    store_agg = (
        df.groupby(store_key_col, as_index=False)
        .agg(agg_dict)
    )

    # Meta-data uit region_shops ernaast
    meta_cols = ["id", "store_display", "sqm_effective"]
    meta_existing = [c for c in meta_cols if c in region_shops.columns]

    if not meta_existing:
        return pd.DataFrame()

    meta = region_shops[meta_existing].copy()
    meta = meta.rename(columns={"id": "id_meta"})

    store_agg = store_agg.merge(
        meta,
        left_on=store_key_col,
        right_on="id_meta",
        how="left",
    )

    # Mooie naam + sqm
    store_agg["store_name"] = store_agg.get("store_display", store_agg[store_key_col].astype(str))
    store_agg["sqm_effective"] = pd.to_numeric(
        store_agg.get("sqm_effective", np.nan), errors="coerce"
    )

    # Omzet per m²
    if "turnover" in store_agg.columns:
        store_agg["turnover_per_sqm"] = np.where(
            (store_agg["sqm_effective"] > 0) & (~store_agg["sqm_effective"].isna()),
            store_agg["turnover"] / store_agg["sqm_effective"],
            np.nan,
        )
    else:
        store_agg["turnover_per_sqm"] = np.nan

    # --- Regionale referenties (medians) ---
    med_turnover = _safe_median(store_agg["turnover"]) if "turnover" in store_agg.columns else np.nan
    med_footfall = _safe_median(store_agg["footfall"]) if "footfall" in store_agg.columns else np.nan
    med_spv = _safe_median(store_agg["sales_per_visitor"]) if "sales_per_visitor" in store_agg.columns else np.nan
    med_tps = _safe_median(store_agg["turnover_per_sqm"])

    # Indexen t.o.v. regio-median
    def _rel_index(x, med):
        x = pd.to_numeric(x, errors="coerce").astype(float)
        if np.isnan(med) or med == 0:
            return np.nan
        return x / med * 100.0

    store_agg["turnover_index"] = _rel_index(store_agg["turnover"], med_turnover)
    store_agg["footfall_index"] = _rel_index(store_agg["footfall"], med_footfall)
    store_agg["spv_index"] = _rel_index(store_agg["sales_per_visitor"], med_spv)
    store_agg["sqm_index"] = _rel_index(store_agg["turnover_per_sqm"], med_tps)

    # --- Radar-score: samengestelde index ---
    # We nemen 4 dimensies:
    # - omzet
    # - footfall
    # - besteding per bezoeker
    # - omzet per m²
    # En middelen die indexen (met wat clipping)
    idx_cols = ["turnover_index", "footfall_index", "spv_index", "sqm_index"]

    for c in idx_cols:
        store_agg[c] = store_agg[c].clip(lower=20, upper=200)  # extreem outliers afvangen

    store_agg["radar_score"] = store_agg[idx_cols].mean(axis=1, skipna=True)

    # --- Potentie-berekening (euro) ---
    # Kernprincipe:
    # - Als omzet per m² < regiomedian → er is structureel potentieel op m².
    # - Potentieel (periode) = (median_tps - store_tps) * sqm_effective
    # - Potentieel (jaar)    = potentieel_periode * annual_factor
    store_agg["potential_period"] = 0.0

    mask_pot = (
        store_agg["turnover_per_sqm"].notna()
        & store_agg["sqm_effective"].notna()
        & (store_agg["sqm_effective"] > 0)
        & (~np.isnan(med_tps))
        & (store_agg["turnover_per_sqm"] < med_tps)
    )

    store_agg.loc[mask_pot, "potential_period"] = (
        (med_tps - store_agg.loc[mask_pot, "turnover_per_sqm"])
        * store_agg.loc[mask_pot, "sqm_effective"]
    )

    store_agg["potential_annual"] = store_agg["potential_period"] * annual_factor

    # --- Status & toelichting ---
    # Kleurcode op radar_score en potentieel
    icons = []
    headlines = []
    reasons = []

    # Regionale gemiddelde capture als context (geen harde input in formule, alleen in tekst)
    avg_capture = None
    if capture_weekly is not None and not capture_weekly.empty and "capture_rate" in capture_weekly.columns:
        avg_capture = float(capture_weekly["capture_rate"].mean(skipna=True))

    for _, row in store_agg.iterrows():
        score = row.get("radar_score", np.nan)
        t_idx = row.get("turnover_index", np.nan)
        spv_idx = row.get("spv_index", np.nan)
        sqm_idx = row.get("sqm_index", np.nan)
        pot_annual = row.get("potential_annual", 0.0)

        # icon
        if np.isnan(score):
            icon = "⚪"
        elif score < 90:
            icon = "🔴"
        elif score < 110:
            icon = "🟠"
        else:
            icon = "🟢"

        # headline
        if np.isnan(score):
            headline = "Onvoldoende data"
        elif score < 90:
            headline = "Presteert onder verwachting"
        elif score < 110:
            headline = "Heeft aandacht nodig"
        else:
            headline = "Gaat goed"

        # Korte toelichting
        parts = []

        # Omzet / m² / SPV
        if not np.isnan(t_idx):
            if t_idx < 90:
                parts.append(f"Omzet ligt ~{round(100 - t_idx):d}% onder regiomedian")
            elif t_idx > 110:
                parts.append(f"Omzet ligt ~{round(t_idx - 100):d}% boven regiomedian")

        if not np.isnan(sqm_idx):
            if sqm_idx < 90:
                parts.append(f"Omzet per m² ~{round(100 - sqm_idx):d}% lager dan regiomedian")
            elif sqm_idx > 110:
                parts.append(f"Omzet per m² ~{round(sqm_idx - 100):d}% hoger dan regiomedian")

        if not np.isnan(spv_idx):
            if spv_idx < 90:
                parts.append(f"Besteding per bezoeker ~{round(100 - spv_idx):d}% lager dan regiomedian")
            elif spv_idx > 110:
                parts.append(f"Besteding per bezoeker ~{round(spv_idx - 100):d}% hoger dan regiomedian")

        # Potentie
        if pot_annual > 0:
            # we geven alleen grove orde (duizenden afronden)
            pot_k = int(round(pot_annual / 1000.0) * 1000)
            if pot_k > 0:
                parts.append(f"Ruimte voor ~€{pot_k:,.0f} extra omzet per jaar op basis van m²-benchmark".replace(",", "."))

        # Capture-context
        if avg_capture is not None and not np.isnan(avg_capture):
            parts.append(f"Regio capture rate rond ~{avg_capture:.1f}% (straat → winkel)")

        if not parts:
            short_reason = "Presteert ongeveer op regiogemiddelde."
        else:
            # maak het compact, maximaal 2–3 stukjes
            short_reason = "; ".join(parts[:3])

        icons.append(icon)
        headlines.append(headline)
        reasons.append(short_reason)

    store_agg["radar_icon"] = icons
    store_agg["headline"] = headlines
    store_agg["short_reason"] = reasons

    # Kolommen netjes sorteren en op radar_score sorteren (slechtste eerst)
    cols_order = [
        "radar_icon",
        "store_name",
        "radar_score",
        "headline",
        "short_reason",
        "turnover",
        "footfall",
        "sales_per_visitor",
        "turnover_per_sqm",
        "potential_period",
        "potential_annual",
        "turnover_index",
        "footfall_index",
        "spv_index",
        "sqm_index",
    ]
    cols_existing = [c for c in cols_order if c in store_agg.columns]

    result = store_agg[cols_existing].copy()
    result = result.sort_values("radar_score", ascending=True)

    return result.reset_index(drop=True)
