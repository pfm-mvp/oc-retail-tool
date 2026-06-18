# services/health_score_service.py
"""
Store Health Score engine — single 0-100 score per store.

Five pillars:
  1. Traffic Vitality  (25%) — footfall trend, traffic vs region
  2. Conversion Power  (25%) — conversion rate, SPV
  3. Space Efficiency  (20%) — turnover/m², capture index
  4. Customer Value    (15%) — ATV, revenue depth
  5. Data Trust        (15%) — sensor uptime, data completeness

Format-specific thresholds:
  - Shrink formats (e.g. Galeria) → efficiency weight up
  - Growth formats (e.g. dm)     → traffic & experience weight up
  - Default                       → balanced weights
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Brand colours for score bands ────────────────────────────────────────────
SCORE_COLORS = {
    "excellent": "#22C55E",   # green
    "good":      "#84CC16",   # lime
    "attention": "#F59E0B",   # amber
    "critical":  "#EF4444",   # red
    "unknown":   "#9CA3AF",   # gray
}

SCORE_ICONS = {
    "excellent": "🟢",
    "good":      "🟢",
    "attention": "🟠",
    "critical":  "🔴",
    "unknown":   "⚪",
}


@dataclass
class PillarScore:
    key: str
    label: str
    weight: float
    score: float          # 0-100
    value_raw: float       # underlying metric value
    benchmark_raw: float   # benchmark value
    reason: str = ""


@dataclass
class HealthScoreResult:
    store_id: int
    store_name: str
    health_score: float          # 0-100
    health_band: str             # excellent / good / attention / critical / unknown
    health_color: str            # hex
    health_icon: str              # emoji
    pillars: List[PillarScore] = field(default_factory=list)
    trend_30d: Optional[pd.Series] = None   # daily health scores
    action_hint: str = ""


# ── Weights ──────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "traffic_vitality":  0.25,
    "conversion_power":  0.25,
    "space_efficiency":  0.20,
    "customer_value":    0.15,
    "data_trust":        0.15,
}

# Format-specific weight overrides
FORMAT_WEIGHTS = {
    "shrink": {   # declining formats — efficiency matters more
        "traffic_vitality":  0.15,
        "conversion_power":  0.20,
        "space_efficiency":  0.35,
        "customer_value":    0.15,
        "data_trust":        0.15,
    },
    "growth": {   # expanding formats — traffic & experience matter more
        "traffic_vitality":  0.35,
        "conversion_power":  0.25,
        "space_efficiency":  0.10,
        "customer_value":    0.20,
        "data_trust":        0.10,
    },
}


def get_weights(store_format: str = "default") -> Dict[str, float]:
    """Return pillar weights based on store format type."""
    return FORMAT_WEIGHTS.get(store_format, DEFAULT_WEIGHTS)


# ── Score classification ────────────────────────────────────────────────────

def classify_score(score: float) -> Tuple[str, str, str]:
    """Return (band, color_hex, icon) for a 0-100 score."""
    if pd.isna(score):
        return "unknown", SCORE_COLORS["unknown"], SCORE_ICONS["unknown"]
    if score >= 75:
        return "excellent", SCORE_COLORS["excellent"], SCORE_ICONS["excellent"]
    if score >= 60:
        return "good", SCORE_COLORS["good"], SCORE_ICONS["good"]
    if score >= 45:
        return "attention", SCORE_COLORS["attention"], SCORE_ICONS["attention"]
    return "critical", SCORE_COLORS["critical"], SCORE_ICONS["critical"]


# ── Normalization ───────────────────────────────────────────────────────────

def _normalize_0_100(series: pd.Series) -> pd.Series:
    """Min-max normalize to 0-100; flat series → 50."""
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty or s.max() == s.min():
        return pd.Series([50.0] * len(s), index=s.index)
    return ((s - s.min()) / (s.max() - s.min()) * 100).clip(0, 100)


def _ratio_to_score(ratio_pct: float, floor: float = 30, cap: float = 200) -> float:
    """Map a percentage ratio to 0-100 score, clipped to [floor, cap]."""
    if pd.isna(ratio_pct):
        return np.nan
    r = float(np.clip(ratio_pct, floor, cap))
    return (r - floor) / (cap - floor) * 100.0


# ── Pillar computation ──────────────────────────────────────────────────────

def compute_traffic_vitality(
    footfall: float,
    footfall_region_median: float,
    footfall_ly: float = np.nan,
) -> Tuple[float, float, str]:
    """
    Traffic Vitality pillar score.
    Combines: footfall vs region median + YoY trend.
    Returns (score 0-100, raw_value, reason).
    """
    if pd.isna(footfall) or footfall == 0:
        return np.nan, np.nan, "No footfall data available."

    # Index vs region (100 = median)
    idx = (footfall / footfall_region_median * 100) if (pd.notna(footfall_region_median) and footfall_region_median > 0) else np.nan

    # YoY growth ratio
    yoy_ratio = (footfall / footfall_ly * 100) if (pd.notna(footfall_ly) and footfall_ly > 0) else np.nan

    # Weighted composite
    if pd.notna(idx) and pd.notna(yoy_ratio):
        composite = idx * 0.6 + yoy_ratio * 0.4
    elif pd.notna(idx):
        composite = idx
    elif pd.notna(yoy_ratio):
        composite = yoy_ratio
    else:
        return np.nan, np.nan, "Te weinig data voor traffic vitaliteit."

    score = _ratio_to_score(composite, floor=30, cap=200)

    # Reason
    parts = []
    if pd.notna(idx):
        parts.append(f"traffic index {idx:.0f} vs regio (100=gemiddeld)")
    if pd.notna(yoy_ratio):
        trend = "stijgend" if yoy_ratio > 105 else "dalend" if yoy_ratio < 95 else "stabiel"
        parts.append(f"YoY trend {trend} ({yoy_ratio:.0f}%)")
    reason = "; ".join(parts) if parts else ""

    return score, idx, reason


def compute_conversion_power(
    conversion_rate: float,
    spv: float,
    conversion_benchmark: float = np.nan,
    spv_benchmark: float = np.nan,
) -> Tuple[float, float, str]:
    """Conversion Power: conversion rate + sales per visitor."""
    parts = []
    scores = []
    weights = []

    # Conversion rate score
    if pd.notna(conversion_rate) and pd.notna(conversion_benchmark) and conversion_benchmark > 0:
        cr_ratio = (conversion_rate / conversion_benchmark) * 100
        cr_score = _ratio_to_score(cr_ratio, floor=30, cap=200)
        scores.append(cr_score)
        weights.append(0.6)
        parts.append(f"conversion {conversion_rate:.1f}% vs {conversion_benchmark:.1f}% benchmark")
    elif pd.notna(conversion_rate):
        # Absolute scoring: <10 poor, 15-25 good, >30 excellent
        cr_abs = np.clip(conversion_rate, 0, 40)
        cr_score = (cr_abs / 40) * 100
        scores.append(cr_score)
        weights.append(0.6)
        parts.append(f"conversion {conversion_rate:.1f}%")

    # SPV score
    if pd.notna(spv) and pd.notna(spv_benchmark) and spv_benchmark > 0:
        spv_ratio = (spv / spv_benchmark) * 100
        spv_score = _ratio_to_score(spv_ratio, floor=20, cap=250)
        scores.append(spv_score)
        weights.append(0.4)
        parts.append(f"SPV €{spv:.2f} vs €{spv_benchmark:.2f}")
    elif pd.notna(spv):
        spv_score = np.clip(spv / 50 * 100, 0, 100)  # rough: €50 SPV = 100
        scores.append(spv_score)
        weights.append(0.4)
        parts.append(f"SPV €{spv:.2f}")

    if not scores:
        return np.nan, np.nan, "No conversion data available."

    total_w = sum(weights)
    score = sum(s * w for s, w in zip(scores, weights)) / total_w
    return score, (conversion_rate if pd.notna(conversion_rate) else spv), "; ".join(parts)


def compute_space_efficiency(
    turnover_per_sqm: float,
    capture_index: float = np.nan,
    tpsm_benchmark: float = np.nan,
) -> Tuple[float, float, str]:
    """Space Efficiency: turnover/m² + capture index."""
    scores = []
    weights = []
    parts = []

    if pd.notna(turnover_per_sqm) and pd.notna(tpsm_benchmark) and tpsm_benchmark > 0:
        tpsm_ratio = (turnover_per_sqm / tpsm_benchmark) * 100
        tpsm_score = _ratio_to_score(tpsm_ratio, floor=20, cap=250)
        scores.append(tpsm_score)
        weights.append(0.6)
        parts.append(f"€{turnover_per_sqm:.0f}/m² vs €{tpsm_benchmark:.0f} benchmark")
    elif pd.notna(turnover_per_sqm):
        scores.append(np.clip(turnover_per_sqm / 500 * 100, 0, 100))
        weights.append(0.6)
        parts.append(f"€{turnover_per_sqm:.0f}/m²")

    if pd.notna(capture_index):
        cap_score = _ratio_to_score(capture_index, floor=30, cap=200)
        scores.append(cap_score)
        weights.append(0.4)
        parts.append(f"capture index {capture_index:.0f}")

    if not scores:
        return np.nan, np.nan, "No space efficiency data."

    total_w = sum(weights)
    score = sum(s * w for s, w in zip(scores, weights)) / total_w
    return score, turnover_per_sqm, "; ".join(parts)


def compute_customer_value(
    atv: float,
    spv: float,
    atv_benchmark: float = np.nan,
) -> Tuple[float, float, str]:
    """Customer Value: ATV depth + SPV depth."""
    scores = []
    weights = []
    parts = []

    if pd.notna(atv) and pd.notna(atv_benchmark) and atv_benchmark > 0:
        atv_ratio = (atv / atv_benchmark) * 100
        atv_score = _ratio_to_score(atv_ratio, floor=30, cap=200)
        scores.append(atv_score)
        weights.append(0.6)
        parts.append(f"ATV €{atv:.2f} vs €{atv_benchmark:.2f}")
    elif pd.notna(atv):
        scores.append(np.clip(atv / 80 * 100, 0, 100))
        weights.append(0.6)
        parts.append(f"ATV €{atv:.2f}")

    if pd.notna(spv):
        spv_score = np.clip(spv / 50 * 100, 0, 100)
        scores.append(spv_score)
        weights.append(0.4)
        parts.append(f"SPV €{spv:.2f}")

    if not scores:
        return np.nan, np.nan, "No customer value data."

    total_w = sum(weights)
    score = sum(s * w for s, w in zip(scores, weights)) / total_w
    return score, atv, "; ".join(parts)


def compute_data_trust(
    sensor_uptime_pct: float = np.nan,
    data_completeness_pct: float = np.nan,
) -> Tuple[float, float, str]:
    """Data Trust: sensor uptime + data completeness."""
    scores = []
    weights = []
    parts = []

    if pd.notna(sensor_uptime_pct):
        # 95%+ uptime = excellent, <80% = poor
        uptime_score = np.clip((sensor_uptime_pct - 60) / 40 * 100, 0, 100)
        scores.append(uptime_score)
        weights.append(0.5)
        parts.append(f"sensor uptime {sensor_uptime_pct:.0f}%")

    if pd.notna(data_completeness_pct):
        comp_score = np.clip((data_completeness_pct - 50) / 50 * 100, 0, 100)
        scores.append(comp_score)
        weights.append(0.5)
        parts.append(f"data compleet {data_completeness_pct:.0f}%")

    if not scores:
        # If no sensor data, use a neutral default — don't penalize missing data
        return 70.0, np.nan, "Sensor data unavailable; neutral score applied."

    total_w = sum(weights)
    score = sum(s * w for s, w in zip(scores, weights)) / total_w
    return score, sensor_uptime_pct, "; ".join(parts)


# ── Action hints ─────────────────────────────────────────────────────────────

def generate_action_hint(result: HealthScoreResult) -> str:
    """Generate 1-2 sentence action hint based on weakest pillar(s)."""
    if not result.pillars:
        return ""

    sorted_pillars = sorted(result.pillars, key=lambda p: p.score if pd.notna(p.score) else 999)

    # Find weakest pillar(s)
    weakest = sorted_pillars[0]
    if pd.isna(weakest.score):
        return "Voldoende data ontbreekt voor actieve aanbeveling."

    hints = {
        "traffic_vitality": "Meer instroom en zichtbaarheid nodig — overweeg lokale marketing of locatie-optimalisatie.",
        "conversion_power": "Visitors not buying enough — focus on ATV, upselling and store experience.",
        "space_efficiency": "Ruimte wordt niet optimaal benut — heroverweeg plattegrond, assortiment en prijsstrategie.",
        "customer_value": "Spend per visitor lags behind — improve product presentation and customer experience.",
        "data_trust": "Data betrouwbaarheid is een risico — controleer sensoren en data-kwaliteit.",
    }

    hint = hints.get(weakest.key, "")

    # If health score is critical, add urgency
    if result.health_score < 45:
        hint = "⚠️ Structurele ingreep nodig. " + hint
    elif result.health_score < 60:
        hint = "📌 Gerichte actie aanbevolen. " + hint

    return hint


# ── Main compute function ────────────────────────────────────────────────────

def compute_store_health(
    store_id: int,
    store_name: str,
    footfall: float,
    turnover: float,
    conversion_rate: float = np.nan,
    spv: float = np.nan,
    sqm: float = np.nan,
    footfall_region_median: float = np.nan,
    footfall_ly: float = np.nan,
    conversion_benchmark: float = np.nan,
    spv_benchmark: float = np.nan,
    tpsm_benchmark: float = np.nan,
    atv_benchmark: float = np.nan,
    capture_index: float = np.nan,
    sensor_uptime_pct: float = np.nan,
    data_completeness_pct: float = np.nan,
    store_format: str = "default",
) -> HealthScoreResult:
    """
    Compute the full Store Health Score.

    Returns a HealthScoreResult with the composite score, per-pillar scores,
    and an action hint.
    """
    weights = get_weights(store_format)

    # Derived metrics
    turnover_per_sqm = (turnover / sqm) if (pd.notna(sqm) and sqm > 0) else np.nan
    atv = (spv / (conversion_rate / 100)) if (pd.notna(spv) and pd.notna(conversion_rate) and conversion_rate > 0) else np.nan

    # ── Pillar 1: Traffic Vitality ─────────────────────────────────────────
    tv_score, tv_raw, tv_reason = compute_traffic_vitality(
        footfall, footfall_region_median, footfall_ly
    )
    pillar_tv = PillarScore(
        key="traffic_vitality",
        label="🚶 Traffic Vitality",
        weight=weights["traffic_vitality"],
        score=tv_score if pd.notna(tv_score) else np.nan,
        value_raw=tv_raw if pd.notna(tv_raw) else np.nan,
        benchmark_raw=footfall_region_median,
        reason=tv_reason,
    )

    # ── Pillar 2: Conversion Power ────────────────────────────────────────
    cp_score, cp_raw, cp_reason = compute_conversion_power(
        conversion_rate, spv, conversion_benchmark, spv_benchmark
    )
    pillar_cp = PillarScore(
        key="conversion_power",
        label="💰 Conversion Power",
        weight=weights["conversion_power"],
        score=cp_score if pd.notna(cp_score) else np.nan,
        value_raw=cp_raw if pd.notna(cp_raw) else np.nan,
        benchmark_raw=conversion_benchmark,
        reason=cp_reason,
    )

    # ── Pillar 3: Space Efficiency ────────────────────────────────────────
    se_score, se_raw, se_reason = compute_space_efficiency(
        turnover_per_sqm, capture_index, tpsm_benchmark
    )
    pillar_se = PillarScore(
        key="space_efficiency",
        label="📐 Space Efficiency",
        weight=weights["space_efficiency"],
        score=se_score if pd.notna(se_score) else np.nan,
        value_raw=se_raw if pd.notna(se_raw) else np.nan,
        benchmark_raw=tpsm_benchmark,
        reason=se_reason,
    )

    # ── Pillar 4: Customer Value ──────────────────────────────────────────
    cv_score, cv_raw, cv_reason = compute_customer_value(
        atv, spv, atv_benchmark
    )
    pillar_cv = PillarScore(
        key="customer_value",
        label="🎯 Customer Value",
        weight=weights["customer_value"],
        score=cv_score if pd.notna(cv_score) else np.nan,
        value_raw=cv_raw if pd.notna(cv_raw) else np.nan,
        benchmark_raw=atv_benchmark,
        reason=cv_reason,
    )

    # ── Pillar 5: Data Trust ──────────────────────────────────────────────
    dt_score, dt_raw, dt_reason = compute_data_trust(
        sensor_uptime_pct, data_completeness_pct
    )
    pillar_dt = PillarScore(
        key="data_trust",
        label="📡 Data Trust",
        weight=weights["data_trust"],
        score=dt_score if pd.notna(dt_score) else np.nan,
        value_raw=dt_raw if pd.notna(dt_raw) else np.nan,
        benchmark_raw=np.nan,
        reason=dt_reason,
    )

    pillars = [pillar_tv, pillar_cp, pillar_se, pillar_cv, pillar_dt]

    # ── Composite Health Score ────────────────────────────────────────────
    valid = [(p.score, p.weight) for p in pillars if pd.notna(p.score)]
    if valid:
        total_w = sum(w for _, w in valid)
        health_score = sum(s * w for s, w in valid) / total_w if total_w > 0 else np.nan
    else:
        health_score = np.nan

    band, color, icon = classify_score(health_score)

    result = HealthScoreResult(
        store_id=store_id,
        store_name=store_name,
        health_score=health_score,
        health_band=band,
        health_color=color,
        health_icon=icon,
        pillars=pillars,
    )

    result.action_hint = generate_action_hint(result)

    return result


# ── Batch compute ────────────────────────────────────────────────────────────

def compute_health_batch(
    df: pd.DataFrame,
    store_key_col: str = "shop_id",
    store_format: str = "default",
) -> List[HealthScoreResult]:
    """
    Compute Health Score for all stores in a DataFrame.
    Expects columns: shop_id, shop_name, footfall, turnover,
    conversion_rate, sales_per_visitor, sqm_effective, etc.
    """
    if df is None or df.empty:
        return []

    results = []
    footfall_median = df["footfall"].median(skipna=True) if "footfall" in df.columns else np.nan
    conv_median = df["conversion_rate"].median(skipna=True) if "conversion_rate" in df.columns else np.nan
    spv_median = df["sales_per_visitor"].median(skipna=True) if "sales_per_visitor" in df.columns else np.nan
    tpsm_median = (df["turnover"] / df["sqm_effective"]).median(skipna=True) if ("sqm_effective" in df.columns and (df["sqm_effective"] > 0).any()) else np.nan
    atv_median = df.get("atv", pd.Series(dtype=float)).median(skipna=True) if "atv" in df.columns else np.nan

    for _, row in df.iterrows():
        sid = int(row.get(store_key_col, 0))
        sname = str(row.get("shop_name", sid))

        result = compute_store_health(
            store_id=sid,
            store_name=sname,
            footfall=row.get("footfall", np.nan),
            turnover=row.get("turnover", np.nan),
            conversion_rate=row.get("conversion_rate", np.nan),
            spv=row.get("sales_per_visitor", np.nan),
            sqm=row.get("sqm_effective", np.nan),
            footfall_region_median=footfall_median,
            footfall_ly=row.get("footfall_ly", np.nan),
            conversion_benchmark=conv_median,
            spv_benchmark=spv_median,
            tpsm_benchmark=tpsm_median,
            atv_benchmark=atv_median,
            capture_index=row.get("capture_index", np.nan),
            sensor_uptime_pct=row.get("sensor_uptime", np.nan),
            data_completeness_pct=row.get("data_completeness", np.nan),
            store_format=store_format,
        )
        results.append(result)

    return results