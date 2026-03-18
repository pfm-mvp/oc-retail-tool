# services/svi_service.py

import numpy as np
import pandas as pd


def _normalize(series: pd.Series) -> pd.Series:
    """
    Normaliseer naar 0–100.
    Als er geen variatie is (of alleen NaN), dan alles op 50 = neutraal.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return pd.Series([50] * len(s), index=s.index)
    min_v, max_v = s.min(), s.max()
    if max_v - min_v == 0:
        return pd.Series([50] * len(s), index=s.index)
    return ((s - min_v) / (max_v - min_v) * 100).clip(0, 100)


def build_store_vitality(
    df_period: pd.DataFrame,
    region_shops: pd.DataFrame,
    store_key_col: str,
) -> pd.DataFrame:
    """
    Bouwt per winkel een Store Vitality Index (SVI, 0–100) + footfall/capture index
    en omzetpotentieel.

    Verwacht:
    - df_period: dagdata met kolommen:
        - store_key_col (bijv. 'id')
        - footfall
        - turnover
        - sales_per_visitor (optioneel, mag NaN zijn)
        - sqm_effective (optioneel, mag NaN zijn)
    - region_shops: mapping met per winkel:
        - id
        - store_display
        - sqm_effective (optioneel)

    Returnt één rij per winkel met o.a.:
    - store_id
    - store_name
    - footfall, turnover, sales_per_visitor, sqm_effective
    - turnover_per_sqm
    - footfall_index_region (100 = regiomedian)
    - capture_index_region (100 = fair share vs m²-aandeel)
    - svi_score (0–100)
    - svi_status (tekst)
    - svi_icon (emoji)
    - reason_short (korte toelichting)
    - profit_potential_period (extra omzetpotentieel voor de geanalyseerde periode)
    """

    if df_period is None or df_period.empty:
        return pd.DataFrame()

    # ---------------------------
    # 1. Basis aggregatie per winkel
    # ---------------------------
    group_cols = [store_key_col]
    agg = (
        df_period.groupby(group_cols)
        .agg(
            footfall=("footfall", "sum"),
            turnover=("turnover", "sum"),
            sales_per_visitor=("sales_per_visitor", "mean"),
            sqm_effective=("sqm_effective", "max"),
        )
        .reset_index()
    )

    # Omzet per m²
    agg["turnover_per_sqm"] = np.where(
        (agg["sqm_effective"] > 0) & (~agg["sqm_effective"].isna()),
        agg["turnover"] / agg["sqm_effective"],
        np.nan,
    )

    # Store-namen erbij
    name_map = {}
    for _, row in region_shops.iterrows():
        try:
            sid = int(row["id"])
        except Exception:
            continue
        name_map[sid] = row.get("store_display", str(sid))

    agg["store_name"] = agg[store_key_col].map(name_map).fillna(
        agg[store_key_col].astype(str)
    )

    # ---------------------------
    # 2. Footfall index vs regio
    # ---------------------------
    median_footfall = agg["footfall"].median(skipna=True)
    if not np.isnan(median_footfall) and median_footfall > 0:
        agg["footfall_index_region"] = (
            agg["footfall"] / median_footfall * 100.0
        )
    else:
        agg["footfall_index_region"] = np.nan

    # ---------------------------
    # 3. Capture index vs regio (fair share op basis van m²)
    # ---------------------------
    total_footfall = agg["footfall"].sum(skipna=True)
    total_sqm = agg["sqm_effective"].sum(skipna=True)

    if total_footfall > 0 and total_sqm > 0:
        # expected_share = m²-aandeel
        exp_share = np.where(
            (agg["sqm_effective"] > 0) & (~agg["sqm_effective"].isna()),
            agg["sqm_effective"] / total_sqm,
            1.0 / max(len(agg), 1),
        )
        # actual_share = footfall-aandeel
        act_share = np.where(
            agg["footfall"] > 0,
            agg["footfall"] / total_footfall,
            0.0,
        )
        capture_idx = np.where(
            exp_share > 0,
            act_share / exp_share * 100.0,
            np.nan,
        )
        agg["capture_index_region"] = capture_idx
    else:
        agg["capture_index_region"] = np.nan

    # ---------------------------
    # 4. Pijlers (0–100)
    # ---------------------------

    # 4.1 Commercial output (omzet + omzet/m² + SPV)
    commercial_metric = (
        agg["turnover"].fillna(0) * 0.5
        + agg["turnover_per_sqm"].fillna(0) * 0.3
        + agg["sales_per_visitor"].fillna(0) * 0.2
    )
    agg["p_commercial"] = _normalize(commercial_metric)

    # 4.2 Market power (footfall + capture index)
    market_metric = (
        agg["footfall_index_region"].fillna(100) * 0.6
        + agg["capture_index_region"].fillna(100) * 0.4
    )
    agg["p_market"] = _normalize(market_metric)

    # 4.3 Customer value (SPV)
    agg["p_customer"] = _normalize(agg["sales_per_visitor"])

    # 4.4 Space efficiency (omzet/m²)
    agg["p_space"] = _normalize(agg["turnover_per_sqm"])

    # ---------------------------
    # 5. Eindscore (SVI 0–100)
    # ---------------------------
    agg["svi_score"] = (
        agg["p_commercial"] * 0.45
        + agg["p_market"] * 0.30
        + agg["p_customer"] * 0.15
        + agg["p_space"] * 0.10
    )

    # ---------------------------
    # 6. Status & icon + korte reason (0–100 schaal)
    # ---------------------------
    def classify(score: float):
        if pd.isna(score):
            return "Onbekend", "⚪"
        if score >= 75:
            return "High performance", "🟢"
        elif score >= 60:
            return "Good / stable", "🟠"
        elif score >= 45:
            return "Attention required", "🟠"
        else:
            return "Under pressure", "🔴"

    statuses = agg["svi_score"].apply(classify)
    agg["svi_status"] = statuses.map(lambda x: x[0])
    agg["svi_icon"] = statuses.map(lambda x: x[1])

    reasons = []
    for _, r in agg.iterrows():
        s = r["svi_score"]
        pc = r["p_commercial"]
        pm = r["p_market"]
        pv = r["p_customer"]
        ps = r["p_space"]

        if pd.isna(s):
            reasons.append("Te weinig data om een goede beoordeling te maken.")
            continue

        if s >= 75:
            if pc >= pm:
                reasons.append(
                    "Sterke omzet en m²-productiviteit; focus op vasthouden en premium beleving."
                )
            else:
                reasons.append(
                    "Sterke positie in traffic en marktaandeel; benut dit met hogere SPV."
                )
        elif s >= 60:
            if pv < 50:
                reasons.append(
                    "Boven regiogemiddelde, maar besteding per bezoeker blijft achter – focus op ATV/upsell."
                )
            else:
                reasons.append(
                    "Goede basis; optimaliseer traffic of m²-benutting voor extra groei."
                )
        elif s >= 45:
            if pm < 50:
                reasons.append(
                    "Onder regio op traffic/capture – meer instroom en zichtbaarheid nodig."
                )
            elif pc < 50:
                reasons.append(
                    "Omzet en m²-productiviteit onder regio – kijk naar assortiment en pricing."
                )
            else:
                reasons.append(
                    "Meerdere KPI’s net onder regio; gericht plan op traffic én SPV gewenst."
                )
        else:
            if pm < 50 and pc < 50:
                reasons.append(
                    "Structureel onder regio op traffic én omzet – herijk formule, team en marketingmix."
                )
            else:
                reasons.append(
                    "Significante achterstand; plan nodig op traffic, conversie en m²-benutting."
                )

    agg["reason_short"] = reasons

    # ---------------------------
    # 7. Omzetpotentieel (periode)
    # ---------------------------
    median_tps = agg["turnover_per_sqm"].median(skipna=True)
    potentials = []
    for _, r in agg.iterrows():
        if pd.isna(r["turnover_per_sqm"]) or pd.isna(r["sqm_effective"]):
            potentials.append(0.0)
            continue
        if np.isnan(median_tps) or median_tps <= 0:
            potentials.append(0.0)
            continue
        ideal_turnover = median_tps * r["sqm_effective"]
        delta = ideal_turnover - r["turnover"]
        potentials.append(max(0.0, float(delta)))

    agg["profit_potential_period"] = potentials

    return agg


# ============================================================
# Centralized SVI utilities (single source of truth)
# Used by Region Copilot + Store Copilot pages
# ============================================================

import numpy as np
import pandas as pd

SVI_DRIVERS = [
    ("sales_per_visitor", "SPV (€/visitor)"),
    ("sales_per_sqm", "Sales / m² (€)"),
    ("capture_rate", "Capture (location-driven) (%)"),
    ("conversion_rate", "Conversion (%)"),
    ("sales_per_transaction", "ATV (€)"),
]

BASE_SVI_WEIGHTS = {
    "sales_per_visitor": 1.0,
    "sales_per_sqm": 1.0,
    "conversion_rate": 1.0,
    "sales_per_transaction": 0.8,
    "capture_rate": 0.4,
}

# Metrics that can be systematically impacted by store size (sqm).
# We calibrate the benchmark value for these metrics when sqm_calibrate=True.
SIZE_CAL_KEYS = {"sales_per_visitor", "conversion_rate", "sales_per_transaction", "capture_rate"}

def safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0.0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan

def norm_key(x: str) -> str:
    return str(x).strip().lower() if x is not None else ""

def sqm_calibration_factor(store_sqm: float, benchmark_sqm_series: pd.Series) -> float:
    """Calibrate benchmark based on relative store size (sqm).
    Returns ~1.0 for median-sized stores in the benchmark group.
    """
    try:
        if pd.isna(store_sqm) or benchmark_sqm_series is None or benchmark_sqm_series.dropna().empty:
            return 1.0
        median_sqm = float(pd.to_numeric(benchmark_sqm_series, errors="coerce").dropna().median())
        if median_sqm <= 0:
            return 1.0
        return float(store_sqm) / median_sqm
    except Exception:
        return 1.0

def ratio_to_score_0_100(ratio_pct: float, floor: float, cap: float) -> float:
    """Map a ratio (in %) to a 0–100 score, clipped to [floor, cap]."""
    if pd.isna(ratio_pct):
        return np.nan
    r = float(np.clip(float(ratio_pct), float(floor), float(cap)))
    return (r - float(floor)) / (float(cap) - float(floor)) * 100.0

def get_svi_weights_for_store_type(store_type: str) -> dict:
    """Return driver weights; capture weight changes by store_type."""
    w = dict(BASE_SVI_WEIGHTS)
    s = norm_key(store_type)

    if ("high" in s and "street" in s) or ("city" in s) or ("downtown" in s) or ("centre" in s) or ("center" in s and "city" in s):
        w["capture_rate"] = 0.7
    elif ("retail" in s and "park" in s) or ("park" in s):
        w["capture_rate"] = 0.2
    elif ("shopping" in s and "center" in s) or ("shopping" in s and "centre" in s) or ("mall" in s) or ("center" in s) or ("centre" in s):
        w["capture_rate"] = 0.4
    else:
        w["capture_rate"] = BASE_SVI_WEIGHTS["capture_rate"]

    return w

def get_svi_weights_for_region_mix(store_types_series: pd.Series) -> dict:
    """Region-level weights as a weighted mix of store-type weights (store-count share)."""
    if store_types_series is None or store_types_series.dropna().empty:
        return dict(BASE_SVI_WEIGHTS)

    s = store_types_series.dropna().astype(str).str.strip()
    s = s[s.str.lower() != "nan"]
    if s.empty:
        return dict(BASE_SVI_WEIGHTS)

    shares = s.value_counts(normalize=True).to_dict()

    mix = {k: 0.0 for k in BASE_SVI_WEIGHTS.keys()}
    for stype, w_share in shares.items():
        w = get_svi_weights_for_store_type(stype)
        for k in mix.keys():
            mix[k] += float(w.get(k, 0.0)) * float(w_share)

    return mix

def compute_driver_values_from_period(footfall, turnover, transactions, sqm_sum, capture_pct):
    """Compute the SVI driver values for a period (aggregated totals)."""
    spv = safe_div(turnover, footfall)
    spsqm = safe_div(turnover, sqm_sum)
    cr = safe_div(transactions, footfall) * 100.0 if (pd.notna(transactions) and pd.notna(footfall) and float(footfall) != 0.0) else np.nan
    atv = safe_div(turnover, transactions)
    cap = capture_pct
    return {
        "sales_per_visitor": spv,
        "sales_per_sqm": spsqm,
        "capture_rate": cap,
        "conversion_rate": cr,
        "sales_per_transaction": atv,
    }

def compute_svi_explainable(
    vals_a: dict,
    vals_b: dict,
    floor: float,
    cap: float,
    weights: dict | None = None,
    *,
    store_sqm: float | None = None,
    benchmark_sqm_series: pd.Series | None = None,
    sqm_calibrate: bool = False,
):
    """Explainable SVI:
    - Computes per-driver ratios (A/B in %)
    - Clips to [floor, cap] and maps to 0–100
    - Produces weighted avg ratio and final 0–100 SVI score
    - Optional: sqm-based benchmark calibration for size-sensitive drivers
    """
    if weights is None:
        weights = {k: float(BASE_SVI_WEIGHTS.get(k, 1.0)) for k, _ in SVI_DRIVERS}

    rows = []
    for key, label in SVI_DRIVERS:
        va = vals_a.get(key, np.nan)
        vb = vals_b.get(key, np.nan)

        # Optional sqm-calibration: adjust benchmark to store size
        vb_adj = vb
        if sqm_calibrate and (key in SIZE_CAL_KEYS) and (store_sqm is not None) and (benchmark_sqm_series is not None):
            f = sqm_calibration_factor(store_sqm, benchmark_sqm_series)
            if pd.notna(vb_adj):
                vb_adj = float(vb_adj) * float(f)

        ratio = np.nan
        if pd.notna(va) and pd.notna(vb_adj) and float(vb_adj) != 0.0:
            ratio = (float(va) / float(vb_adj)) * 100.0

        score = ratio_to_score_0_100(ratio, floor=float(floor), cap=float(cap))
        w = float(weights.get(key, 1.0))

        include = pd.notna(ratio) and pd.notna(score)
        rows.append({
            "driver_key": key,
            "driver": label,
            "value": va,
            "benchmark": vb_adj,      # store-size calibrated benchmark (if enabled)
            "benchmark_raw": vb,      # original benchmark
            "ratio_pct": ratio,
            "score": score,
            "weight": w,
            "include": include,
        })

    bd = pd.DataFrame(rows)
    usable = bd[bd["include"]].copy()
    if usable.empty:
        return np.nan, np.nan, bd.drop(columns=["include"])

    usable["w"] = usable["weight"].astype(float)
    wsum = float(usable["w"].sum()) if float(usable["w"].sum()) > 0 else float(len(usable))
    avg_ratio = float((usable["ratio_pct"] * usable["w"]).sum() / wsum)
    svi = ratio_to_score_0_100(avg_ratio, floor=float(floor), cap=float(cap))
    return float(svi), float(avg_ratio), bd.drop(columns=["include"])
