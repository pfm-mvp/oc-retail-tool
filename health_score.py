# health_score.py — Store Health Score computation
# Composite 0-100 score from 5 dimensions with segment-specific weights
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# ── Segment weight defaults ────────────────────────────────────────────────
SEGMENT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "fashion": {
        "traffic_trend": 0.20,
        "conversion_efficiency": 0.25,
        "revenue_productivity": 0.20,
        "staffing_alignment": 0.15,
        "catchment_potential": 0.20,
    },
    "grocery": {
        "traffic_trend": 0.25,
        "conversion_efficiency": 0.15,
        "revenue_productivity": 0.25,
        "staffing_alignment": 0.20,
        "catchment_potential": 0.15,
    },
    "home_living": {
        "traffic_trend": 0.15,
        "conversion_efficiency": 0.30,
        "revenue_productivity": 0.20,
        "staffing_alignment": 0.10,
        "catchment_potential": 0.25,
    },
    "general": {
        "traffic_trend": 0.20,
        "conversion_efficiency": 0.25,
        "revenue_productivity": 0.20,
        "staffing_alignment": 0.15,
        "catchment_potential": 0.20,
    },
}

# Benchmarks per segment (conversion rate, revenue/m²)
SEGMENT_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "fashion": {"conversion_rate": 0.20, "revenue_per_sqm": 3500},
    "grocery": {"conversion_rate": 0.70, "revenue_per_sqm": 5000},
    "home_living": {"conversion_rate": 0.15, "revenue_per_sqm": 2800},
    "general": {"conversion_rate": 0.25, "revenue_per_sqm": 3200},
}

# Optimal FTE per visitor per segment
SEGMENT_STAFF_OPTIMA: Dict[str, float] = {
    "fashion": 1 / 50,     # 1 FTE per 50 visitors
    "grocery": 1 / 80,
    "home_living": 1 / 30,
    "general": 1 / 60,
}

FLAG_THRESHOLD = 40  # flag dimensions below this score


def _cap_score(val: float) -> int:
    """Cap score to 0-100 integer."""
    return int(max(0, min(100, round(val))))


def _score_traffic_trend(
    footfall_series: pd.Series,
    footfall_yoy: Optional[pd.Series] = None,
) -> int:
    """
    Score traffic trend dimension (0-100).
    If YoY data available: compare current vs same period last year.
    Else: 4-week rolling trend.
    """
    if footfall_series is None or len(footfall_series) == 0:
        return 50  # neutral fallback

    current = float(footfall_series.iloc[-1]) if len(footfall_series) > 0 else 0

    if footfall_yoy is not None and len(footfall_yoy) > 0:
        expected = float(footfall_yoy.iloc[-1])
    elif len(footfall_series) >= 4:
        expected = float(footfall_series.iloc[-4:].mean())
    else:
        expected = float(footfall_series.mean())

    if expected == 0:
        return 50

    score = 50 + 50 * (current - expected) / expected
    return _cap_score(score)


def _score_conversion_efficiency(
    conversion_rate: Optional[float],
    segment: str,
) -> int:
    """Score conversion efficiency (0-100) vs segment benchmark."""
    if conversion_rate is None:
        return 50
    benchmark = SEGMENT_BENCHMARKS.get(segment, SEGMENT_BENCHMARKS["general"])["conversion_rate"]
    if benchmark == 0:
        return 50
    return _cap_score(100 * conversion_rate / benchmark)


def _score_revenue_productivity(
    revenue_per_sqm: Optional[float],
    segment: str,
) -> int:
    """Score revenue productivity (0-100) vs segment benchmark."""
    if revenue_per_sqm is None:
        return 50
    benchmark = SEGMENT_BENCHMARKS.get(segment, SEGMENT_BENCHMARKS["general"])["revenue_per_sqm"]
    if benchmark == 0:
        return 50
    return _cap_score(100 * revenue_per_sqm / benchmark)


def _score_staffing_alignment(
    visitors: Optional[float],
    fte: Optional[float],
    segment: str,
) -> int:
    """Score staffing alignment (0-100) based on FTE/visitor ratio vs optimum."""
    if visitors is None or fte is None or visitors == 0:
        return 50
    optimum = SEGMENT_STAFF_OPTIMA.get(segment, SEGMENT_STAFF_OPTIMA["general"])
    actual_ratio = fte / visitors
    if optimum == 0:
        return 50
    # Score 100 when ratio = optimum, degrades linearly
    ratio = actual_ratio / optimum
    if ratio <= 1.0:
        score = 100 * ratio  # understaffed
    else:
        score = 100 / ratio  # overstaffed
    return _cap_score(score)


def _score_catchment_potential(
    actual_capture: Optional[float],
    expected_capture: Optional[float],
) -> int:
    """Score catchment potential (0-100) — actual vs expected capture rate."""
    if actual_capture is None or expected_capture is None or expected_capture == 0:
        return 50
    return _cap_score(100 * actual_capture / expected_capture)


def compute_health_score(
    store_id: int,
    segment: str,
    footfall_series: Optional[pd.Series] = None,
    footfall_yoy: Optional[pd.Series] = None,
    conversion_rate: Optional[float] = None,
    revenue_per_sqm: Optional[float] = None,
    visitors: Optional[float] = None,
    fte: Optional[float] = None,
    actual_capture: Optional[float] = None,
    expected_capture: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute Store Health Score (0-100) with 5 dimensions.
    Returns structured dict with dimension scores, flags, and composite.
    """
    weights = SEGMENT_WEIGHTS.get(segment, SEGMENT_WEIGHTS["general"])

    dims = {
        "traffic_trend": _score_traffic_trend(footfall_series, footfall_yoy),
        "conversion_efficiency": _score_conversion_efficiency(conversion_rate, segment),
        "revenue_productivity": _score_revenue_productivity(revenue_per_sqm, segment),
        "staffing_alignment": _score_staffing_alignment(visitors, fte, segment),
        "catchment_potential": _score_catchment_potential(actual_capture, expected_capture),
    }

    # Compute composite with available weights
    total_weight = 0.0
    weighted_sum = 0.0
    flagged = []

    dim_output = {}
    for dim_name, score in dims.items():
        w = weights.get(dim_name, 0.20)
        weighted_sum += score * w
        total_weight += w
        is_flag = score < FLAG_THRESHOLD
        dim_output[dim_name] = {
            "score": score,
            "weight": round(w, 2),
            "flag": is_flag,
        }
        if is_flag:
            flagged.append(dim_name)

    composite = int(round(weighted_sum / total_weight)) if total_weight > 0 else 0
    composite = max(0, min(100, composite))

    return {
        "store_id": store_id,
        "health_score": composite,
        "dimensions": dim_output,
        "flagged_dimensions": flagged,
        "segment": segment,
    }


def compute_health_trend(
    store_id: int,
    segment: str,
    daily_df: pd.DataFrame,
    days: int = 30,
) -> List[int]:
    """
    Compute health score for each of the last N days.
    Uses daily footfall data to produce a trend.
    """
    if daily_df is None or daily_df.empty:
        return []

    store_df = daily_df[daily_df["shop_id"] == store_id].tail(days)
    if store_df.empty:
        return []

    trend = []
    footfall_vals = store_df.get("visits", store_df.get("count_in", pd.Series(dtype=float)))
    if footfall_vals.empty:
        return []

    for i in range(len(footfall_vals)):
        # Use a rolling window up to current point
        window = footfall_vals.iloc[:i + 1]
        result = compute_health_score(
            store_id=store_id,
            segment=segment,
            footfall_series=window,
        )
        trend.append(result["health_score"])

    return trend
